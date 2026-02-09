#!/usr/bin/env python3
"""
phragmen_seq.py (v6)

Sequential Phragmén (credit/time) with:

- Base voter ballots (weighted approval; identical approvals collapsed into groups)
- Mega ballots (quota-floor reserve, soft racing; multi-mode inputs)
- Party ballots (quota-floor reserve, soft racing; order ONLY tie-break for dt ties)
- Projection fraction p = franchise participation coverage (BASE ballots only, no double-count)
- Iterative A/B passes with signature mapping:
    * Signature = Part-B prefix winners up to projection > 5/9 (strict)
    * Stop at the FIRST repeat signature (covers convergence + cycles)
- Full chamber completion (after repeat-signature stop):
    * Continue selecting until BOTH:
        (a) rounds >= input seats
        (b) projection > 2/3 (strict)
- Prefix interventions:
    * prefix_allow_only consumed FIRST (higher priority than iterative pools)
    * ban set always excluded
- CLI prompting:
    * If not converged after each block of --max_iters, prompt for more iterations (default +19)
    * Use --no_prompt for batch mode

JSON Input (top-level keys):
- seats: int
- candidates: [str] (optional; auto-expanded from ballots + party lists)
- ballots: [{"approvals":[...], "weight":number}]  (base ballots)
- candidate_meta: {cand: {"groups":[...], "tags":[...]}} (optional)
- mega_ballots: [ ... ] (optional)
- party_ballots: [ ... ] (optional)
- prefix_intervention: {"allow_only":[...], "ban":[...]} (optional)

Mega/Party ballot definitions support:
- Mode A: {"abs_weight":..., "population":..., "approvals":[...], "auto_include":...}
- Mode B: {"quota_floor":..., "population":optional, "abs_weight":optional, "approvals":optional, "auto_include":optional}

auto_include:
  {"groups_any":[...], "tags_any":[...]}

Notes:
- Quota floors are SOFT: quota groups only become ACTIVE when their set is currently below requirement.
- Projection p ignores mega+party ballots by design.

Outputs in --outdir:
- pass01_rounds.csv, pass01_quota.csv, pass01_projection.csv
- iterXXA_*.csv and iterXXB_*.csv per iteration
- converged_full_*.csv if repeat signature found and completion run executed
- optional quota_meta_csv: normalized mega/party parameters

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
import argparse
import csv
import json
import math
import os

EPS = 1e-12


# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class Group:
    gid: str
    kind: str  # "base" | "mega" | "party"
    approvals: Tuple[str, ...]
    weight: float
    quota_floor: Optional[float] = None
    population: Optional[float] = None
    abs_weight: Optional[float] = None
    share: Optional[float] = None


# ---------------------------
# Utility
# ---------------------------

def canon_set(xs: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in xs)))


def open_csv(path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


def write_meta_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------
# Candidate meta / auto include
# ---------------------------

def parse_candidate_meta(raw: dict) -> Dict[str, Dict[str, Set[str]]]:
    """
    Normalize candidate_meta to sets for fast membership checks.
    Expected:
      { "Cand": {"groups":[...], "tags":[...]}, ... }
    """
    meta: Dict[str, Dict[str, Set[str]]] = {}
    for cand, m in (raw or {}).items():
        c = str(cand)
        groups = set(str(x) for x in (m.get("groups") or []))
        tags = set(str(x) for x in (m.get("tags") or []))
        meta[c] = {"groups": groups, "tags": tags}
    return meta


def auto_include_candidates(
    candidate_set: Set[str],
    candidate_meta: Dict[str, Dict[str, Set[str]]],
    auto_include: Optional[dict],
) -> Set[str]:
    if not auto_include:
        return set()

    want_groups = set(str(x) for x in (auto_include.get("groups_any") or []))
    want_tags = set(str(x) for x in (auto_include.get("tags_any") or []))

    out: Set[str] = set()
    for c in candidate_set:
        m = candidate_meta.get(c)
        if not m:
            continue
        if want_groups and (m["groups"] & want_groups):
            out.add(c)
            continue
        if want_tags and (m["tags"] & want_tags):
            out.add(c)
            continue
    return out


# ---------------------------
# Base ballots
# ---------------------------

def canonicalize_base_ballots(raw_ballots: List[dict]) -> List[Group]:
    """Collapse identical approval sets by summing weights."""
    agg: Dict[Tuple[str, ...], float] = {}
    for b in raw_ballots or []:
        approvals = canon_set(b.get("approvals", []))
        if not approvals:
            continue
        w = float(b.get("weight", 1.0))
        if w <= 0:
            continue
        agg[approvals] = agg.get(approvals, 0.0) + w

    groups: List[Group] = []
    for i, (appr, wsum) in enumerate(agg.items()):
        groups.append(Group(gid=f"B:{i}", kind="base", approvals=appr, weight=wsum))
    return groups


def candidates_from_base_groups(base_groups: List[Group]) -> Set[str]:
    s: Set[str] = set()
    for g in base_groups:
        s.update(g.approvals)
    return s


def total_normal_ballots_weight(base_groups: List[Group]) -> float:
    return float(sum(g.weight for g in base_groups))


# ---------------------------
# Quota-floor math
# ---------------------------

def projection_total_for_quota_from_base(base_groups: List[Group]) -> int:
    """proj_total_for_quota = ceil((2/3)*total_voter_ballots) used ONLY for r_eff in quota."""
    tot = total_normal_ballots_weight(base_groups)
    return int(math.ceil((2.0 / 3.0) * tot - 1e-15))


def derive_quota_and_rel_weight_from_abs(
    abs_weight: float,
    population: float,
    total_voter_ballots: float
) -> Tuple[float, float, float]:
    """
    share = abs_weight / population
    rel_weight = share * total_voter_ballots
    quota_floor = min((2/3)*share, 1/3)
    """
    if population <= 0:
        raise ValueError("population must be > 0")
    share = abs_weight / population
    rel_weight = share * total_voter_ballots
    quota_floor = min((2.0 / 3.0) * share, 1.0 / 3.0)
    return share, rel_weight, quota_floor


def invert_share_from_quota_floor(qf: float) -> Tuple[float, bool]:
    """
    quota_floor = min((2/3)*share, 1/3)
    If qf < 1/3 => share = (3/2)*qf
    If qf == 1/3 => share is not uniquely determined; choose minimal share that reaches saturation: share=1/2
    Returns (share, saturation_assumed_min_share_flag)
    """
    if qf < (1.0 / 3.0) - 1e-15:
        return 1.5 * qf, False
    return 0.5, True


# ---------------------------
# Party ballots
# ---------------------------

def parse_party_ballots(
    raw_parties: List[dict],
    total_voter_ballots: float,
) -> Tuple[List[Group], Dict[str, List[str]], List[dict], Set[str]]:
    """
    Party ballots behave like mega ballots:
    - Mode A: abs_weight + population
    - Mode B: quota_floor (population/abs_weight optional)
    - approvals set = party list candidates (dedup)
    - order kept for tie-break only
    """
    groups: List[Group] = []
    party_lists: Dict[str, List[str]] = {}
    meta_rows: List[dict] = []
    party_cands: Set[str] = set()

    for i, p in enumerate(raw_parties or []):
        gid = str(p.get("id", f"P:{i}"))

        quota_floor = p.get("quota_floor", None)
        abs_weight = p.get("abs_weight", p.get("weight", None))
        population = p.get("population", None)

        share = None
        rel_weight = None
        derived_mode = None
        saturation_assumed = False
        pop_val = None
        abs_val = None

        if quota_floor is not None:
            qf = float(quota_floor)
            if qf <= 0:
                continue
            share, saturation_assumed = invert_share_from_quota_floor(qf)
            rel_weight = share * total_voter_ballots
            derived_mode = "quota_floor"
            if population is not None:
                pop_val = float(population)
                abs_val = float(abs_weight) if abs_weight is not None else (share * pop_val)
            else:
                pop_val = None
                abs_val = float(abs_weight) if abs_weight is not None else None
            quota_floor = qf
        else:
            if abs_weight is None or population is None:
                continue
            abs_val = float(abs_weight)
            pop_val = float(population)
            if abs_val <= 0 or pop_val <= 0:
                continue
            share, rel_weight, qf = derive_quota_and_rel_weight_from_abs(abs_val, pop_val, total_voter_ballots)
            quota_floor = qf
            derived_mode = "abs_weight_population"
            saturation_assumed = (abs(qf - (1.0/3.0)) <= 1e-12)

        ordered_raw = p.get("candidates", p.get("approvals", []))
        ordered = [str(x) for x in (ordered_raw or [])]

        seen: Set[str] = set()
        ordered_dedup: List[str] = []
        for c in ordered:
            if c not in seen:
                seen.add(c)
                ordered_dedup.append(c)

        if not ordered_dedup:
            meta_rows.append({
                "id": gid, "kind": "party",
                "mode": derived_mode,
                "population": pop_val if pop_val is not None else "",
                "abs_weight": abs_val if abs_val is not None else "",
                "share": share,
                "rel_weight": rel_weight,
                "quota_floor": quota_floor,
                "party_list_len": 0,
                "approvals_simplified": 0,
                "saturation_assumed_min_share": saturation_assumed,
                "dropped_from_algorithm": True,
            })
            continue

        party_lists[gid] = ordered_dedup
        party_cands.update(ordered_dedup)

        approvals = tuple(sorted(set(ordered_dedup)))
        groups.append(Group(
            gid=gid, kind="party",
            approvals=approvals,
            weight=float(rel_weight),
            quota_floor=float(quota_floor),
            population=float(pop_val) if pop_val is not None else None,
            abs_weight=float(abs_val) if abs_val is not None else None,
            share=float(share) if share is not None else None
        ))

        meta_rows.append({
            "id": gid, "kind": "party",
            "mode": derived_mode,
            "population": pop_val if pop_val is not None else "",
            "abs_weight": abs_val if abs_val is not None else "",
            "share": share,
            "rel_weight": rel_weight,
            "quota_floor": quota_floor,
            "party_list_len": len(ordered_dedup),
            "approvals_simplified": len(approvals),
            "saturation_assumed_min_share": saturation_assumed,
            "dropped_from_algorithm": False,
        })

    return groups, party_lists, meta_rows, party_cands


# ---------------------------
# Mega ballots (multi-mode + auto include)
# ---------------------------

def parse_mega_ballots(
    raw_megas: List[dict],
    total_voter_ballots: float,
    candidate_set: Set[str],
    candidate_meta: Dict[str, Dict[str, Set[str]]],
) -> Tuple[List[Group], List[dict]]:
    groups: List[Group] = []
    meta_rows: List[dict] = []

    for i, m in enumerate(raw_megas or []):
        gid = str(m.get("id", f"M:{i}"))

        quota_floor = m.get("quota_floor", None)
        abs_weight = m.get("abs_weight", m.get("weight", None))
        population = m.get("population", None)

        share = None
        rel_weight = None
        derived_mode = None
        saturation_assumed = False
        pop_val = None
        abs_val = None

        if quota_floor is not None:
            qf = float(quota_floor)
            if qf <= 0:
                continue
            share, saturation_assumed = invert_share_from_quota_floor(qf)
            rel_weight = share * total_voter_ballots
            derived_mode = "quota_floor"
            if population is not None:
                pop_val = float(population)
                abs_val = float(abs_weight) if abs_weight is not None else (share * pop_val)
            else:
                pop_val = None
                abs_val = float(abs_weight) if abs_weight is not None else None
            quota_floor = qf
        else:
            if abs_weight is None or population is None:
                continue
            abs_val = float(abs_weight)
            pop_val = float(population)
            if abs_val <= 0 or pop_val <= 0:
                continue
            share, rel_weight, qf = derive_quota_and_rel_weight_from_abs(abs_val, pop_val, total_voter_ballots)
            quota_floor = qf
            derived_mode = "abs_weight_population"
            saturation_assumed = (abs(qf - (1.0/3.0)) <= 1e-12)

        manual = set(str(x) for x in (m.get("approvals") or []))
        auto = auto_include_candidates(candidate_set, candidate_meta, m.get("auto_include"))
        approvals = sorted((manual | auto) & candidate_set)

        if len(approvals) == 0:
            meta_rows.append({
                "id": gid, "kind": "mega",
                "mode": derived_mode,
                "population": pop_val if pop_val is not None else "",
                "abs_weight": abs_val if abs_val is not None else "",
                "share": share,
                "rel_weight": rel_weight,
                "quota_floor": quota_floor,
                "approvals_original": len(manual),
                "approvals_auto": len(auto),
                "approvals_simplified": 0,
                "saturation_assumed_min_share": saturation_assumed,
                "dropped_from_algorithm": True,
            })
            continue

        groups.append(Group(
            gid=gid, kind="mega",
            approvals=tuple(approvals),
            weight=float(rel_weight),
            quota_floor=float(quota_floor),
            population=float(pop_val) if pop_val is not None else None,
            abs_weight=float(abs_val) if abs_val is not None else None,
            share=float(share) if share is not None else None
        ))

        meta_rows.append({
            "id": gid, "kind": "mega",
            "mode": derived_mode,
            "population": pop_val if pop_val is not None else "",
            "abs_weight": abs_val if abs_val is not None else "",
            "share": share,
            "rel_weight": rel_weight,
            "quota_floor": quota_floor,
            "approvals_original": len(manual),
            "approvals_auto": len(auto),
            "approvals_simplified": len(approvals),
            "saturation_assumed_min_share": saturation_assumed,
            "dropped_from_algorithm": False,
        })

    return groups, meta_rows


# ---------------------------
# Prefix intervention (allow-only + ban)
# ---------------------------

def parse_prefix_intervention(data: dict) -> Tuple[List[str], Set[str]]:
    pi = data.get("prefix_intervention") or {}
    allow = [str(x) for x in (pi.get("allow_only") or [])]
    ban = set(str(x) for x in (pi.get("ban") or []))
    return allow, ban


# ---------------------------
# Phragmén mechanics
# ---------------------------

def build_supporters(groups: List[Group], candidates: List[str]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {c: [] for c in candidates}
    cand_set = set(candidates)
    for gi, g in enumerate(groups):
        for c in g.approvals:
            if c in cand_set:
                idx[c].append(gi)
    return idx


def quota_required(quota_floor: float, r_eff: int) -> int:
    return int(math.ceil(quota_floor * r_eff - 1e-15))


def compute_quota_active_info(
    quota_groups: List[Group],
    winners_so_far: List[str],
    r_eff: int
) -> Dict[str, Tuple[bool, int, int]]:
    """
    Returns {gid: (active, winners_in_set_before, required)}
    active=True means group is currently below its quota requirement and participates in the race.
    """
    info: Dict[str, Tuple[bool, int, int]] = {}
    for g in quota_groups:
        qf = float(g.quota_floor or 0.0)
        req = quota_required(qf, r_eff)
        s = set(g.approvals)
        in_set = sum(1 for w in winners_so_far if w in s)
        info[g.gid] = (in_set < req, in_set, req)
    return info


def apply_time_step(balances: List[float], weights: List[float], dt: float) -> None:
    if dt <= 0:
        return
    for i in range(len(balances)):
        balances[i] += dt * weights[i]


def candidate_dt(
    cand: str,
    balances: List[float],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
) -> Optional[Tuple[float, float, float]]:
    """
    Returns (dt, have, rate) where:
      have = current sum of balances among active supporters
      rate = earning rate among active supporters
      dt = time needed to reach cost=1
    """
    have = 0.0
    rate = 0.0
    for gi in supporters.get(cand, []):
        if not active_mask[gi]:
            continue
        have += balances[gi]
        rate += weights[gi]
    if rate <= EPS:
        return None
    if have >= 1.0 - EPS:
        return (0.0, have, rate)
    return ((1.0 - have) / rate, have, rate)


def spend_winner(cand: str, balances: List[float], supporters: Dict[str, List[int]], active_mask: List[bool]) -> None:
    for gi in supporters.get(cand, []):
        if active_mask[gi]:
            balances[gi] = 0.0


def party_rank_maps(party_lists: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    return {pid: {c: i for i, c in enumerate(lst)} for pid, lst in party_lists.items()}


def tie_break_by_party_order(tied_candidates: List[str], party_lists: Dict[str, List[str]]) -> str:
    rank_maps = party_rank_maps(party_lists)
    best = None  # (rank_index, candidate)
    for c in tied_candidates:
        best_rank = None
        for rm in rank_maps.values():
            if c in rm:
                r = rm[c]
                if best_rank is None or r < best_rank:
                    best_rank = r
        if best_rank is not None:
            key = (best_rank, c)
            if best is None or key < best:
                best = key
    return best[1] if best is not None else min(tied_candidates)


def choose_candidate_for_round(
    remaining: List[str],
    balances: List[float],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    party_lists: Dict[str, List[str]],
    allow_only_pool: Optional[Set[str]] = None,
) -> Tuple[str, float, float, float, bool]:
    """
    If allow_only_pool is non-empty, restrict eligibility to remaining ∩ pool.
    If that intersection is empty, fallback to remaining.
    Returns (chosen, dt, have, rate, allow_used_flag).
    """
    allow_used = False
    if allow_only_pool is not None and len(allow_only_pool) > 0:
        filtered = [c for c in remaining if c in allow_only_pool]
        if filtered:
            allow_used = True
        else:
            filtered = list(remaining)
    else:
        filtered = list(remaining)

    scored: List[Tuple[float, str, float, float]] = []
    for c in filtered:
        res = candidate_dt(c, balances, weights, supporters, active_mask)
        if res is None:
            continue
        dt, have, rate = res
        scored.append((dt, c, have, rate))

    if not scored:
        raise ValueError("No candidates affordable (no active supporters?).")

    scored.sort(key=lambda x: (x[0], x[1]))
    best_dt = scored[0][0]
    tied = [c for (dt, c, _, _) in scored if abs(dt - best_dt) <= 1e-12]

    if len(tied) > 1:
        chosen = tie_break_by_party_order(tied, party_lists)
        for dt, c, have, rate in scored:
            if c == chosen and abs(dt - best_dt) <= 1e-12:
                return chosen, dt, have, rate, allow_used

    dt, chosen, have, rate = scored[0]
    return chosen, dt, have, rate, allow_used


# ---------------------------
# Projection accounting
# ---------------------------

def compute_projection_delta_for_chosen(
    chosen: str,
    groups: List[Group],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    base_used: List[bool],
) -> float:
    """
    Newly used voter ballots (weight) among ACTIVE supporters of chosen.
    A base group counts only once globally.
    """
    newly_used = 0.0
    for gi in supporters.get(chosen, []):
        if not active_mask[gi]:
            continue
        if groups[gi].kind != "base":
            continue
        if base_used[gi]:
            continue
        newly_used += groups[gi].weight
        base_used[gi] = True
    return newly_used


# ---------------------------
# CSV Schemas
# ---------------------------

ROUND_FIELDS = [
    "label",
    "round",
    "chosen",
    "time",
    "dt",
    "have",
    "rate",

    "total_voter_ballots",
    "delta_voter_ballots_used",
    "delta_projection",
    "total_projection",
    "used_voter_ballots_cum",

    "projection_interval_prev",
    "projection_interval_curr",

    "r_eff",
    "projection_total_for_quota",
    "projection_delta_for_quota",

    "active_quota_groups",

    "prefix_allow_pool_size_before",
    "iter_allow_pool_size_before",
    "allow_pool_source",
    "allow_only_used",
]

QUOTA_FIELDS = [
    "label",
    "round",
    "chosen",
    "quota_gid",
    "quota_kind",
    "quota_floor",
    "required_by_r_eff",
    "winners_in_set_before",
    "active_for_race",
    "reserve_balance_after",
    "rel_weight",
    "abs_weight",
    "population",
    "share",
    "approvals_size",
]

PROJ_FIELDS = [
    "label",
    "round",
    "chosen",
    "delta_voter_ballots_used",
    "delta_projection",
    "total_projection",
    "used_voter_ballots_cum",
]


# ---------------------------
# Run engine
# ---------------------------

def run_sequential(
    label: str,
    candidates: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    party_lists: Dict[str, List[str]],
    stop_when_proj_gt: Optional[float],            # stop when p_total > threshold (strict)
    prefix_allow_only_init: Optional[List[str]],   # higher priority pool
    ban_set: Optional[Set[str]],                   # always banned
    iter_allow_only_init: Optional[List[str]],     # lower priority pool
    rounds_csv_path: Optional[str],
    quota_csv_path: Optional[str],
    proj_csv_path: Optional[str],
) -> Dict[str, Any]:
    groups = base_groups + mega_groups + party_groups
    weights = [g.weight for g in groups]
    balances = [0.0 for _ in groups]

    supporters = build_supporters(groups, candidates)
    gid_to_index = {g.gid: i for i, g in enumerate(groups)}
    quota_groups = [g for g in groups if g.kind in ("mega", "party")]

    total_voter_ballots = total_normal_ballots_weight(base_groups)
    proj_total_for_quota = projection_total_for_quota_from_base(base_groups)

    p_total = 0.0
    used_voter_ballots_cum = 0.0
    base_used = [False] * len(groups)

    prefix_pool: Optional[Set[str]] = set(prefix_allow_only_init) if prefix_allow_only_init else None
    iter_pool: Optional[Set[str]] = set(iter_allow_only_init) if iter_allow_only_init else None
    ban = ban_set or set()

    rounds_f = rounds_w = None
    quota_f = quota_w = None
    proj_f = proj_w = None
    if rounds_csv_path:
        rounds_f, rounds_w = open_csv(rounds_csv_path, ROUND_FIELDS)
    if quota_csv_path:
        quota_f, quota_w = open_csv(quota_csv_path, QUOTA_FIELDS)
    if proj_csv_path:
        proj_f, proj_w = open_csv(proj_csv_path, PROJ_FIELDS)

    winners: List[str] = []
    winners_set: Set[str] = set()
    time_now = 0.0

    try:
        for r in range(1, seats + 1):
            if stop_when_proj_gt is not None and p_total > stop_when_proj_gt + 1e-15:
                break

            remaining = [c for c in candidates if c not in winners_set and c not in ban]
            if not remaining:
                break

            r_eff = max(r, proj_total_for_quota)
            projection_delta_for_quota = r_eff - r

            quota_info = compute_quota_active_info(quota_groups, winners, r_eff)

            # ACTIVE groups: base groups always active; quota groups only active if below requirement
            active_mask = [True] * len(groups)
            for g in quota_groups:
                active_mask[gid_to_index[g.gid]] = bool(quota_info[g.gid][0])

            prefix_size_before = len(prefix_pool) if prefix_pool is not None else 0
            iter_size_before = len(iter_pool) if iter_pool is not None else 0

            pool_to_use = None
            pool_source = "none"
            if prefix_pool is not None and len(prefix_pool) > 0:
                pool_to_use = prefix_pool
                pool_source = "prefix"
            elif iter_pool is not None and len(iter_pool) > 0:
                pool_to_use = iter_pool
                pool_source = "iter"

            chosen, dt, have, rate, allow_used = choose_candidate_for_round(
                remaining,
                balances, weights, supporters, active_mask,
                party_lists,
                allow_only_pool=pool_to_use
            )

            apply_time_step(balances, weights, dt)
            time_now += dt

            newly_used_ballots = compute_projection_delta_for_chosen(
                chosen, groups, supporters, active_mask, base_used
            )
            delta_p = (newly_used_ballots / total_voter_ballots) if total_voter_ballots > 0 else 0.0

            p_prev = p_total
            p_total += delta_p
            used_voter_ballots_cum += newly_used_ballots

            spend_winner(chosen, balances, supporters, active_mask)
            winners.append(chosen)
            winners_set.add(chosen)

            # consume pools
            if pool_source == "prefix" and prefix_pool is not None and chosen in prefix_pool:
                prefix_pool.remove(chosen)
                if len(prefix_pool) == 0:
                    prefix_pool = None
            elif pool_source == "iter" and iter_pool is not None and chosen in iter_pool:
                iter_pool.remove(chosen)
                if len(iter_pool) == 0:
                    iter_pool = None

            active_quota_groups = sum(1 for g in quota_groups if quota_info[g.gid][0])

            if rounds_w:
                rounds_w.writerow({
                    "label": label,
                    "round": r,
                    "chosen": chosen,
                    "time": f"{time_now:.12f}",
                    "dt": f"{dt:.12f}",
                    "have": f"{have:.12f}",
                    "rate": f"{rate:.12f}",

                    "total_voter_ballots": f"{total_voter_ballots:.6f}",
                    "delta_voter_ballots_used": f"{newly_used_ballots:.6f}",
                    "delta_projection": f"{delta_p:.12f}",
                    "total_projection": f"{p_total:.12f}",
                    "used_voter_ballots_cum": f"{used_voter_ballots_cum:.6f}",

                    "projection_interval_prev": f"{p_prev:.12f}",
                    "projection_interval_curr": f"{p_total:.12f}",

                    "r_eff": r_eff,
                    "projection_total_for_quota": proj_total_for_quota,
                    "projection_delta_for_quota": projection_delta_for_quota,

                    "active_quota_groups": active_quota_groups,

                    "prefix_allow_pool_size_before": prefix_size_before,
                    "iter_allow_pool_size_before": iter_size_before,
                    "allow_pool_source": pool_source,
                    "allow_only_used": 1 if allow_used else 0,
                })
                rounds_f.flush()

            if proj_w:
                proj_w.writerow({
                    "label": label,
                    "round": r,
                    "chosen": chosen,
                    "delta_voter_ballots_used": f"{newly_used_ballots:.6f}",
                    "delta_projection": f"{delta_p:.12f}",
                    "total_projection": f"{p_total:.12f}",
                    "used_voter_ballots_cum": f"{used_voter_ballots_cum:.6f}",
                })
                proj_f.flush()

            if quota_w:
                for g in quota_groups:
                    active, in_set, req = quota_info[g.gid]
                    gi = gid_to_index[g.gid]
                    quota_w.writerow({
                        "label": label,
                        "round": r,
                        "chosen": chosen,
                        "quota_gid": g.gid,
                        "quota_kind": g.kind,
                        "quota_floor": float(g.quota_floor or 0.0),
                        "required_by_r_eff": int(req),
                        "winners_in_set_before": int(in_set),
                        "active_for_race": bool(active),
                        "reserve_balance_after": float(balances[gi]),
                        "rel_weight": float(g.weight),
                        "abs_weight": float(g.abs_weight or 0.0),
                        "population": float(g.population or 0.0),
                        "share": float(g.share or 0.0),
                        "approvals_size": len(g.approvals),
                    })
                quota_f.flush()

    finally:
        if rounds_f:
            rounds_f.close()
        if quota_f:
            quota_f.close()
        if proj_f:
            proj_f.close()

    return {
        "label": label,
        "winners": winners,
        "final_time": time_now,
        "final_projection": p_total,
        "stopped_at_round": len(winners),
        "total_voter_ballots": total_voter_ballots,
        "projection_total_for_quota": proj_total_for_quota,
    }


# ---------------------------
# Read back intervals from rounds CSV
# ---------------------------

def recompute_intervals_from_rounds_csv(rounds_csv_path: str) -> Tuple[List[str], List[Tuple[float, float]]]:
    winners: List[str] = []
    intervals: List[Tuple[float, float]] = []
    with open(rounds_csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            winners.append(row["chosen"])
            p0 = float(row["projection_interval_prev"])
            p1 = float(row["projection_interval_curr"])
            intervals.append((p0, p1))
    return winners, intervals


def select_pool_by_coverage(winners: List[str], intervals: List[Tuple[float, float]], a: float, b: float) -> List[str]:
    """
    winners[i] covers (p_prev, p_curr]
    Include if overlaps [a,b]:
      include if p_prev < b and p_curr >= a
    """
    out: List[str] = []
    for w, (p0, p1) in zip(winners, intervals):
        if p0 < b - 1e-15 and p1 >= a - 1e-15:
            out.append(w)
    return out


def prefix_until_projection_strict_gt(winners: List[str], intervals: List[Tuple[float, float]], target: float) -> List[str]:
    """
    Prefix until first round where p_curr > target (strict).
    """
    out: List[str] = []
    for w, (p0, p1) in zip(winners, intervals):
        out.append(w)
        if p1 > target + 1e-15:
            break
    return out


# ---------------------------
# Full chamber completion (C):
# full chamber size = max(seats, first round where projection > 2/3)
# ---------------------------

def run_full_chamber_completion(
    outdir: str,
    candidates_list: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    party_lists: Dict[str, List[str]],
    prefix_allow: List[str],
    ban: Set[str],
) -> Dict[str, Any]:
    """
    Run from scratch with cap = len(candidates_list), then truncate to minimal R s.t.:
      R >= seats AND projection > 2/3 (strict).
    """
    cap = max(seats, len(candidates_list))
    label = "converged_full"
    rounds_csv = os.path.join(outdir, f"{label}_rounds.csv")
    quota_csv = os.path.join(outdir, f"{label}_quota.csv")
    proj_csv = os.path.join(outdir, f"{label}_projection.csv")

    _res = run_sequential(
        label=label,
        candidates=candidates_list,
        seats=cap,
        base_groups=base_groups,
        mega_groups=mega_groups,
        party_groups=party_groups,
        party_lists=party_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=rounds_csv,
        quota_csv_path=quota_csv,
        proj_csv_path=proj_csv,
    )

    winners, intervals = recompute_intervals_from_rounds_csv(rounds_csv)

    R2 = None
    for i, (p0, p1) in enumerate(intervals, start=1):
        if i >= seats and p1 > (2/3) + 1e-15:
            R2 = i
            break

    if R2 is None:
        # If never reaches >2/3, fall back to at least seats (or whatever ran)
        R2 = max(seats, len(winners))

    return {
        "label": label,
        "cap_run": cap,
        "full_chamber_rounds": R2,
        "full_chamber_winners": winners[:R2],
        "full_chamber_projection": intervals[R2-1][1] if R2-1 < len(intervals) else _res["final_projection"],
        "rounds_csv": rounds_csv,
        "quota_csv": quota_csv,
        "projection_csv": proj_csv,
    }


# ---------------------------
# Prompting
# ---------------------------

def prompt_more_iters(default: int = 19) -> Optional[int]:
    """
    Returns:
      - an integer > 0 to continue with that many additional iterations
      - None if user chose to quit
    """
    while True:
        s = input(
            f"Not converged. Continue with more iterations? "
            f"Enter a number (default {default}), or 'q' to quit: "
        ).strip()
        if s == "":
            return default
        if s.lower() in ("q", "quit", "exit"):
            return None
        try:
            n = int(s)
            if n > 0:
                return n
        except ValueError:
            pass
        print("Please enter a positive integer or 'q'.")


# ---------------------------
# Main iterative driver
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Sequential Phragmén with mega+party quota floors + projection signature repeats (v6).")
    ap.add_argument("input_json", help="Election JSON file.")
    ap.add_argument("--outdir", default="out", help="Output directory for CSVs.")
    ap.add_argument("--quota_meta_csv", default=None, help="Write normalized mega/party meta CSV (path).")
    ap.add_argument("--max_iters", type=int, default=19, help="Max iterations INCLUDING the first pass.")
    ap.add_argument("--no_prompt", action="store_true",
                    help="Do not prompt for more iterations; exit when --max_iters is reached without repeat signature.")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    seats = int(data["seats"])

    # Base groups first
    base_groups = canonicalize_base_ballots(data.get("ballots", []))
    total_voter_ballots = total_normal_ballots_weight(base_groups)

    # Candidate meta
    candidate_meta = parse_candidate_meta(data.get("candidate_meta") or {})

    # Candidate universe: declared + canonical base groups
    candidates: Set[str] = set(str(c) for c in data.get("candidates", []))
    candidates.update(candidates_from_base_groups(base_groups))

    # Prefix intervention
    prefix_allow, ban = parse_prefix_intervention(data)

    # Party ballots (may add candidates)
    party_groups, party_lists, party_meta, party_cands = parse_party_ballots(
        data.get("party_ballots", []),
        total_voter_ballots=total_voter_ballots
    )
    candidates.update(party_cands)

    # Mega ballots simplified against full candidate set (includes party candidates)
    mega_groups, mega_meta = parse_mega_ballots(
        data.get("mega_ballots", []),
        total_voter_ballots=total_voter_ballots,
        candidate_set=candidates,
        candidate_meta=candidate_meta
    )

    candidates_list = sorted(candidates)

    # Optional quota meta CSV
    if args.quota_meta_csv:
        write_meta_csv(args.quota_meta_csv, mega_meta + party_meta)

    os.makedirs(args.outdir, exist_ok=True)

    # PASS 1 (normal; prefix intervention applies)
    pass1_label = "pass01"
    pass1_rounds = os.path.join(args.outdir, f"{pass1_label}_rounds.csv")
    pass1_quota = os.path.join(args.outdir, f"{pass1_label}_quota.csv")
    pass1_proj = os.path.join(args.outdir, f"{pass1_label}_projection.csv")

    _res1 = run_sequential(
        label=pass1_label,
        candidates=candidates_list,
        seats=seats,
        base_groups=base_groups,
        mega_groups=mega_groups,
        party_groups=party_groups,
        party_lists=party_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=pass1_rounds,
        quota_csv_path=pass1_quota,
        proj_csv_path=pass1_proj,
    )

    pass1_winners, pass1_intervals = recompute_intervals_from_rounds_csv(pass1_rounds)

    # Signature baseline: prefix up to projection > 5/9
    pass1_prefix = prefix_until_projection_strict_gt(pass1_winners, pass1_intervals, 5/9)
    seen: Dict[Tuple[str, ...], int] = {tuple(pass1_prefix): 1}
    seen_label: Dict[Tuple[str, ...], str] = {tuple(pass1_prefix): "pass01"}

    prev_B_full_winners = pass1_winners
    prev_B_full_intervals = pass1_intervals

    solved = False
    solved_iter: Optional[int] = None
    exit_reason: Optional[str] = None
    twin_iter: Optional[int] = None
    twin_label: Optional[str] = None
    cycle_len: Optional[int] = None

    it = 2
    max_iters_current = args.max_iters

    while True:
        while it <= max_iters_current:
            labelA = f"iter{it:02d}A"
            labelB = f"iter{it:02d}B"

            # Pool A from previous B covering [1/9, 5/9]
            pool_A = select_pool_by_coverage(prev_B_full_winners, prev_B_full_intervals, 1/9, 5/9)

            A_rounds = os.path.join(args.outdir, f"{labelA}_rounds.csv")
            A_quota = os.path.join(args.outdir, f"{labelA}_quota.csv")
            A_proj = os.path.join(args.outdir, f"{labelA}_projection.csv")

            _ = run_sequential(
                label=labelA,
                candidates=candidates_list,
                seats=seats,
                base_groups=base_groups,
                mega_groups=mega_groups,
                party_groups=party_groups,
                party_lists=party_lists,
                stop_when_proj_gt=5/9,              # strict stop
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_A,         # lower priority than prefix pool
                rounds_csv_path=A_rounds,
                quota_csv_path=A_quota,
                proj_csv_path=A_proj,
            )

            A_winners, A_intervals = recompute_intervals_from_rounds_csv(A_rounds)

            # Pool B = entire Part A winners
            pool_B = list(A_winners)

            B_rounds = os.path.join(args.outdir, f"{labelB}_rounds.csv")
            B_quota = os.path.join(args.outdir, f"{labelB}_quota.csv")
            B_proj = os.path.join(args.outdir, f"{labelB}_projection.csv")

            _ = run_sequential(
                label=labelB,
                candidates=candidates_list,
                seats=seats,
                base_groups=base_groups,
                mega_groups=mega_groups,
                party_groups=party_groups,
                party_lists=party_lists,
                stop_when_proj_gt=5/9,              # strict stop
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_B,
                rounds_csv_path=B_rounds,
                quota_csv_path=B_quota,
                proj_csv_path=B_proj,
            )

            B_winners, B_intervals = recompute_intervals_from_rounds_csv(B_rounds)
            B_prefix = prefix_until_projection_strict_gt(B_winners, B_intervals, 5/9)
            sig = tuple(B_prefix)

            # Unified convergence/cycle detection: stop at first repeat signature
            if sig in seen:
                solved = True
                solved_iter = it
                exit_reason = "repeat_signature"
                twin_iter = seen[sig]
                twin_label = seen_label.get(sig, f"iter{twin_iter:02d}B" if twin_iter != 1 else "pass01")
                cycle_len = it - twin_iter
                break

            seen[sig] = it
            seen_label[sig] = labelB

            # advance state for next iteration
            prev_B_full_winners = B_winners
            prev_B_full_intervals = B_intervals

            it += 1

        if solved:
            break

        # Not solved after this block
        if args.no_prompt:
            break

        more = prompt_more_iters(default=19)
        if more is None:
            break
        max_iters_current += more

    completion = None
    if solved:
        completion = run_full_chamber_completion(
            outdir=args.outdir,
            candidates_list=candidates_list,
            seats=seats,
            base_groups=base_groups,
            mega_groups=mega_groups,
            party_groups=party_groups,
            party_lists=party_lists,
            prefix_allow=prefix_allow,
            ban=ban
        )

    print(json.dumps({
        "status": "repeat_signature_found" if solved else "not_converged",
        "exit_reason": exit_reason,
        "repeat_at_iteration": solved_iter,
        "twin_iteration": twin_iter,
        "twin_label": twin_label,
        "cycle_length": cycle_len,
        "max_iters_initial": args.max_iters,
        "max_iters_reached": max_iters_current,
        "completion": completion,
        "note": (
            "Signature = Part-B prefix winners until projection > 5/9 (strict). "
            "Stop at first repeat signature (covers convergence and cycles). "
            "If stopped, complete from scratch until rounds>=seats and projection>2/3 (strict)."
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

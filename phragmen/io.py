from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import csv
import math
import os

from .types import ElectionProfile, Group


# ---------------------------
# Small utilities
# ---------------------------

def canon_set(xs: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in xs)))


def total_normal_ballots_weight(base_groups: List[Group]) -> float:
    return float(sum(g.weight for g in base_groups))


def candidates_from_groups(groups: List[Group]) -> Set[str]:
    s: Set[str] = set()
    for g in groups:
        s.update(g.approvals)
    return s


def extract_candidates_from_defs(raw_defs: List[dict]) -> Set[str]:
    """
    Extract candidate names from list definitions that may store candidates under:
      - "candidates"
      - "approvals"
    Used for party + electorate candidate universe expansion before simplification.
    """
    out: Set[str] = set()
    for d in raw_defs or []:
        xs = d.get("candidates", d.get("approvals", [])) or []
        for c in xs:
            out.add(str(c))
    return out


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
# Quota-floor math
# ---------------------------

def projection_total_for_quota_from_base(base_groups: List[Group]) -> int:
    tot = total_normal_ballots_weight(base_groups)
    return int(math.ceil((2.0 / 3.0) * tot - 1e-15))


def derive_quota_and_rel_weight_from_abs(
    abs_weight: float,
    population: float,
    total_voter_ballots: float
) -> Tuple[float, float, float]:
    if population <= 0:
        raise ValueError("population must be > 0")
    share = abs_weight / population
    rel_weight = share * total_voter_ballots
    quota_floor = min((2.0 / 3.0) * share, 1.0 / 3.0)
    return share, rel_weight, quota_floor


def invert_share_from_quota_floor(qf: float) -> Tuple[float, bool]:
    if qf < (1.0 / 3.0) - 1e-15:
        return 1.5 * qf, False
    return 0.5, True


# ---------------------------
# Base ballots
# ---------------------------

def canonicalize_base_ballots(raw_ballots: List[dict], profile: ElectionProfile) -> List[Group]:
    agg: Dict[Tuple[str, ...], float] = {}
    for b in raw_ballots or []:
        approvals = canon_set(b.get("approvals", []))
        if not approvals:
            continue
        w = float(b.get("weight", 1.0))
        w = float(profile.scale_base_weight(w))
        if w <= 0:
            continue
        agg[approvals] = agg.get(approvals, 0.0) + w

    groups: List[Group] = []
    for i, (appr, wsum) in enumerate(agg.items()):
        groups.append(Group(gid=f"B:{i}", kind="base", approvals=appr, weight=wsum))
    return groups


# ---------------------------
# Party ballots
# ---------------------------

def parse_party_ballots(
    raw_parties: List[dict],
    total_voter_ballots: float,
    profile: ElectionProfile,
) -> Tuple[List[Group], Dict[str, List[str]], List[dict], Set[str]]:
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
            saturation_assumed = (abs(qf - (1.0 / 3.0)) <= 1e-12)

        rel_weight = float(profile.scale_party_rel_weight(float(rel_weight)))

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
            share=float(share) if share is not None else None,
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
# Mega ballots
# ---------------------------

def parse_mega_ballots(
    raw_megas: List[dict],
    total_voter_ballots: float,
    candidate_set: Set[str],
    candidate_meta: Dict[str, Dict[str, Set[str]]],
    profile: ElectionProfile,
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
            saturation_assumed = (abs(qf - (1.0 / 3.0)) <= 1e-12)

        rel_weight = float(profile.scale_mega_rel_weight(float(rel_weight)))

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
            share=float(share) if share is not None else None,
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
# Electorate ballots
# ---------------------------

def parse_electorate_ballots(
    raw_electorates: List[dict],
    total_voter_ballots: float,
    candidate_set: Set[str],
    profile: ElectionProfile,
) -> Tuple[List[Group], List[dict]]:
    """
    Electorate ballot groups:
      - behave like mega/party quota groups in the engine (soft quota-floor reserve)
      - approvals are limited to candidates registered in that electorate
      - inputs support abs_weight+population OR quota_floor
      - no auto_include (for now)
    """
    groups: List[Group] = []
    meta_rows: List[dict] = []

    for i, e in enumerate(raw_electorates or []):
        gid = str(e.get("id", f"E:{i}"))

        quota_floor = e.get("quota_floor", None)
        abs_weight = e.get("abs_weight", e.get("weight", None))
        population = e.get("population", None)

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
            saturation_assumed = (abs(qf - (1.0 / 3.0)) <= 1e-12)

        rel_weight = float(profile.scale_electorate_rel_weight(float(rel_weight)))

        raw_cands = e.get("candidates", e.get("approvals", [])) or []
        ordered = [str(x) for x in raw_cands]

        # de-dupe preserve order (for meta only)
        seen: Set[str] = set()
        ordered_dedup: List[str] = []
        for c in ordered:
            if c not in seen:
                seen.add(c)
                ordered_dedup.append(c)

        approvals = tuple(sorted(set(ordered_dedup) & candidate_set))

        if len(approvals) == 0:
            meta_rows.append({
                "id": gid, "kind": "electorate",
                "mode": derived_mode,
                "population": pop_val if pop_val is not None else "",
                "abs_weight": abs_val if abs_val is not None else "",
                "share": share,
                "rel_weight": rel_weight,
                "quota_floor": quota_floor,
                "electorate_candidate_list_len": len(ordered_dedup),
                "approvals_simplified": 0,
                "saturation_assumed_min_share": saturation_assumed,
                "dropped_from_algorithm": True,
            })
            continue

        groups.append(Group(
            gid=gid, kind="electorate",
            approvals=approvals,
            weight=float(rel_weight),
            quota_floor=float(quota_floor),
            population=float(pop_val) if pop_val is not None else None,
            abs_weight=float(abs_val) if abs_val is not None else None,
            share=float(share) if share is not None else None,
        ))

        meta_rows.append({
            "id": gid, "kind": "electorate",
            "mode": derived_mode,
            "population": pop_val if pop_val is not None else "",
            "abs_weight": abs_val if abs_val is not None else "",
            "share": share,
            "rel_weight": rel_weight,
            "quota_floor": quota_floor,
            "electorate_candidate_list_len": len(ordered_dedup),
            "approvals_simplified": len(approvals),
            "saturation_assumed_min_share": saturation_assumed,
            "dropped_from_algorithm": False,
        })

    return groups, meta_rows


# ---------------------------
# Prefix intervention
# ---------------------------

def parse_prefix_intervention(data: dict) -> Tuple[List[str], Set[str]]:
    pi = data.get("prefix_intervention") or {}
    allow = [str(x) for x in (pi.get("allow_only") or [])]
    ban = set(str(x) for x in (pi.get("ban") or []))
    return allow, ban

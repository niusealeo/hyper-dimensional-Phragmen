from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import csv
import math

from .types import Group, ElectionProfile


def _canon_approvals(seq: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in seq if str(x).strip() != "")))


def canonicalize_base_ballots(ballots: List[dict], profile: ElectionProfile) -> List[Group]:
    agg: Dict[Tuple[str, ...], float] = defaultdict(float)
    for b in ballots or []:
        apps = _canon_approvals(b.get("approvals", []))
        w = float(b.get("weight", 1.0))
        if w <= 0 or not apps:
            continue
        agg[apps] += w

    out: List[Group] = []
    for i, (apps, w) in enumerate(sorted(agg.items(), key=lambda x: (-x[1], x[0]))):
        out.append(
            Group(
                gid=f"base_{i}",
                kind="base",
                approvals=apps,
                weight=profile.scale_base_weight(w),
                meta={"raw_weight": w, "approvals_size": len(apps)},
            )
        )
    return out


def total_normal_ballots_weight(base_groups: List[Group]) -> float:
    return sum(g.weight for g in base_groups)


def candidates_from_groups(groups: List[Group]) -> Set[str]:
    s: Set[str] = set()
    for g in groups:
        s.update(g.approvals)
    return s


def parse_candidate_meta(obj: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, v in (obj or {}).items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def candidates_matching_meta(candidate_meta: Dict[str, dict], where: Dict[str, Any]) -> Set[str]:
    if not where:
        return set()

    matched: Set[str] = set()
    for cand, meta in candidate_meta.items():
        ok = True
        for key, want in where.items():
            if key not in meta:
                ok = False
                break
            have = meta[key]
            if isinstance(want, (list, tuple, set)):
                want_set = set(want)
                if isinstance(have, (list, tuple, set)):
                    if not (set(have) & want_set):
                        ok = False
                        break
                else:
                    if have not in want_set:
                        ok = False
                        break
            else:
                if isinstance(have, (list, tuple, set)):
                    if want not in set(have):
                        ok = False
                        break
                else:
                    if have != want:
                        ok = False
                        break
        if ok:
            matched.add(cand)
    return matched


def parse_prefix_intervention(data: dict) -> Tuple[List[str], Set[str]]:
    interventions = data.get("interventions", {}) or {}
    allow = interventions.get("prefix_allow_only", []) or data.get("prefix_allow_only", [])
    ban = interventions.get("ban", []) or data.get("ban", [])
    allow_list = [str(x) for x in allow]
    ban_set = set(str(x) for x in ban)
    return allow_list, ban_set


def quota_floor_from_share(share: float) -> float:
    return min((2.0 / 3.0) * share, 1.0 / 3.0)


def share_from_quota_floor(qf: float) -> float:
    qf = float(qf)
    if qf < (1.0 / 3.0) - 1e-15:
        return 1.5 * qf
    return 0.5


def normalize_rel_weight_from_share(share: float, total_voter_ballots: float) -> float:
    return float(share) * float(total_voter_ballots)


META_FIELDS = [
    "gid",
    "kind",
    "population",
    "abs_weight",
    "share",
    "rel_weight",
    "quota_floor",
    "approvals_size",
]


def write_meta_csv(path: str, groups: List[Group]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for g in groups:
            w.writerow({
                "gid": g.gid,
                "kind": g.kind,
                "population": g.population or "",
                "abs_weight": g.abs_weight or "",
                "share": g.share or "",
                "rel_weight": g.weight,
                "quota_floor": g.quota_floor if g.quota_floor is not None else "",
                "approvals_size": len(g.approvals),
            })


def parse_party_ballots(
    party_defs: List[dict],
    total_voter_ballots: float,
    profile: ElectionProfile,
) -> Tuple[List[Group], Dict[str, List[str]], List[Group], Set[str]]:
    groups: List[Group] = []
    party_lists: Dict[str, List[str]] = {}
    meta: List[Group] = []
    cands: Set[str] = set()

    for i, d in enumerate(party_defs or []):
        gid = str(d.get("id") or f"party_{i}")
        candidates = [str(x) for x in (d.get("candidates") or d.get("list") or [])]
        apps = _canon_approvals(candidates)
        cands.update(apps)
        party_lists[gid] = list(candidates)

        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        if population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
            if qf is None:
                qf = quota_floor_from_share(share)
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
        else:
            rel_w = float(d.get("rel_weight", 0.0))

        g = Group(
            gid=gid,
            kind="party",
            approvals=apps,
            weight=profile.scale_party_rel_weight(rel_w),
            quota_floor=float(qf) if qf is not None else None,
            population=float(population) if population is not None else None,
            abs_weight=float(abs_weight) if abs_weight is not None else None,
            share=float(share) if share is not None else None,
            meta={"source": "party_ballots"},
        )
        groups.append(g)
        meta.append(g)

    return groups, party_lists, meta, cands


def _expand_approvals_with_meta(
    approvals: Set[str],
    candidate_meta: Dict[str, dict],
    where: Optional[Dict[str, Any]],
) -> Set[str]:
    if not where:
        return approvals
    approvals = set(approvals)
    approvals |= candidates_matching_meta(candidate_meta, where)
    return approvals


def parse_mega_ballots(
    mega_defs: List[dict],
    total_voter_ballots: float,
    candidate_set: Set[str],
    candidate_meta: Dict[str, dict],
    profile: ElectionProfile,
) -> Tuple[List[Group], List[Group]]:
    groups: List[Group] = []
    meta: List[Group] = []

    for i, d in enumerate(mega_defs or []):
        gid = str(d.get("id") or f"mega_{i}")

        raw_apps = set(str(x) for x in (d.get("candidates") or []))
        where = d.get("include_where") or d.get("meta_where")
        raw_apps = _expand_approvals_with_meta(raw_apps, candidate_meta, where)
        apps = tuple(sorted(raw_apps & set(candidate_set)))

        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        if population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
            if qf is None:
                qf = quota_floor_from_share(share)
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
        else:
            rel_w = float(d.get("rel_weight", 0.0))

        g = Group(
            gid=gid,
            kind="mega",
            approvals=apps,
            weight=profile.scale_mega_rel_weight(rel_w),
            quota_floor=float(qf) if qf is not None else None,
            population=float(population) if population is not None else None,
            abs_weight=float(abs_weight) if abs_weight is not None else None,
            share=float(share) if share is not None else None,
            meta={"source": "mega_ballots", "where": where},
        )
        groups.append(g)
        meta.append(g)

    return groups, meta


def parse_electorate_ballots(
    elect_defs: List[dict],
    total_voter_ballots: float,
    candidate_set: Set[str],
    profile: ElectionProfile,
) -> Tuple[List[Group], List[Group]]:
    groups: List[Group] = []
    meta: List[Group] = []

    for i, d in enumerate(elect_defs or []):
        gid = str(d.get("id") or f"electorate_{i}")
        raw_apps = set(str(x) for x in (d.get("candidates") or []))
        apps = tuple(sorted(raw_apps & set(candidate_set)))

        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        if population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
            if qf is None:
                qf = quota_floor_from_share(share)
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, total_voter_ballots)
        else:
            rel_w = float(d.get("rel_weight", 0.0))

        g = Group(
            gid=gid,
            kind="electorate",
            approvals=apps,
            weight=profile.scale_electorate_rel_weight(rel_w),
            quota_floor=float(qf) if qf is not None else None,
            population=float(population) if population is not None else None,
            abs_weight=float(abs_weight) if abs_weight is not None else None,
            share=float(share) if share is not None else None,
            meta={"source": "electorate_ballots"},
        )
        groups.append(g)
        meta.append(g)

    return groups, meta


def extract_candidates_from_defs(defs: List[dict]) -> Set[str]:
    s: Set[str] = set()
    for d in defs or []:
        s.update(str(x) for x in (d.get("candidates") or d.get("list") or []))
    return s


def projection_seat_equiv(total_projection: float, seats: int) -> int:
    return int(math.ceil(float(total_projection) * float(seats) - 1e-15))

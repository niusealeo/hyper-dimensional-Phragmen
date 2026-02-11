from __future__ import annotations

from typing import Dict, List, Tuple

from ..parse.numeric import parse_numeric_optional


def _canon_approvals(seq: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in seq if str(x).strip() != "")))


def compute_base_electorate_info(ballots: List[dict]) -> Dict[str, dict]:
    """Compute per-electorate info from raw base voter ballots.

    Returns mapping electorate_id -> {
        'counted_base_ballots': sum of ballot weights (raw, before profile scaling),
        'counted_base_records': number of ballot records (unweighted),
        'candidates': set of candidates appearing in ballots registered to the electorate,
    }

    The electorate id is read from one of: 'electorate', 'electorate_id', 'electorate_gid'.
    Missing electorate ids are mapped to 'electorate_unknown'.
    """
    info: Dict[str, dict] = {}
    for b in ballots or []:
        apps = _canon_approvals(b.get("approvals", []))
        try:
            w = float(parse_numeric_optional(b.get("weight", 1.0), field="ballots.weight") or 0.0)
        except Exception:
            w = 0.0
        if w <= 0 or not apps:
            continue
        eid = b.get("electorate")
        if eid is None:
            eid = b.get("electorate_id")
        if eid is None:
            eid = b.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        if eid not in info:
            info[eid] = {"counted_base_ballots": 0.0, "counted_base_records": 0, "candidates": set()}
        info[eid]["counted_base_ballots"] += w
        info[eid]["counted_base_records"] += 1
        info[eid]["candidates"].update(apps)
    return info


def compute_partyrock_electorate_info(partyrock_defs: List[dict]) -> Dict[str, dict]:
    """Compute per-electorate info from raw PartyRock ballots.

    Returns mapping electorate_id -> {
        'counted_partyrock_abs_weight': sum of PartyRock ballot absolute weights,
        'counted_partyrock_records': number of PartyRock ballot records,
        'counted_partyrock_ballots': a count-like total for denominator use (alias of records),
        'candidates': set of candidates appearing in PartyRock ballots registered to the electorate,
    }

    The electorate id is read from one of: 'electorate', 'electorate_id', 'electorate_gid'.
    Missing electorate ids are mapped to 'electorate_unknown'.
    """
    info: Dict[str, dict] = {}
    for d in partyrock_defs or []:
        if not isinstance(d, dict):
            continue
        apps = _canon_approvals(d.get("approvals", []))
        try:
            w_abs = float(parse_numeric_optional(d.get("weight", 0.0), field="partyrock.weight") or 0.0)
        except Exception:
            w_abs = 0.0
        if w_abs <= 0 or not apps:
            continue

        eid = d.get("electorate")
        if eid is None:
            eid = d.get("electorate_id")
        if eid is None:
            eid = d.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        if eid not in info:
            info[eid] = {
                "counted_partyrock_abs_weight": 0.0,
                "counted_partyrock_records": 0,
                "counted_partyrock_ballots": 0,
                "candidates": set(),
            }
        info[eid]["counted_partyrock_abs_weight"] += w_abs
        info[eid]["counted_partyrock_records"] += 1
        info[eid]["counted_partyrock_ballots"] += 1
        info[eid]["candidates"].update(apps)
    return info


def canonicalize_base_ballots(ballots: List[dict], profile: ElectionProfile) -> List[Group]:
    """Canonicalize and collapse base voter ballots.

    Collapsing key is (electorate_id, approvals) so identical approval lists in different electorates
    remain distinct groups.
    """
    agg: Dict[Tuple[str, Tuple[str, ...]], float] = defaultdict(float)

    for b in ballots or []:
        apps = _canon_approvals(b.get("approvals", []))
        w = float(parse_numeric_optional(b.get("weight", 1.0), field="ballots.weight") or 0.0)
        if w <= 0 or not apps:
            continue

        eid = b.get("electorate")
        if eid is None:
            eid = b.get("electorate_id")
        if eid is None:
            eid = b.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        agg[(eid, apps)] += w

    out: List[Group] = []
    # deterministic order: by weight desc, then electorate id, then approvals
    items = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    for i, ((eid, apps), w) in enumerate(items):
        out.append(
            Group(
                gid=f"base_{i}",
                kind="base",
                approvals=apps,
                weight=profile.scale_base_weight(w),
                meta={"raw_weight": w, "approvals_size": len(apps), "electorate": eid},
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

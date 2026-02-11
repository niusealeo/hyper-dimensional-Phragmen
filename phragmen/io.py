from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import csv
import math

from .types import Group, ElectionProfile


def _clone_group_with_weight(g: Group, new_weight: float, meta_update: Optional[dict] = None) -> Group:
    meta = dict(g.meta or {})
    if meta_update:
        meta.update(meta_update)
    return Group(
        gid=g.gid,
        kind=g.kind,
        approvals=g.approvals,
        weight=float(new_weight),
        quota_floor=g.quota_floor,
        population=g.population,
        abs_weight=g.abs_weight,
        share=g.share,
        meta=meta,
    )


def compute_profile_12_324_totals(data: dict) -> Tuple[float, float, float]:
    """Return (x, y, n) for profile 12 / 324 normalisation.

    x = total absolute voter ballots sum from data['ballots'] (sum of valid ballot weights)
    y = total absolute party ballots sum from data['party_ballots'] (sum of defs[*].weight when present)
    n = optional total ballot capacity (defaults to max(x, y) if missing)

    Accepted keys for n: 'n', 'n_ballots', 'total_ballots', 'ballots_n'.
    """
    ballots = data.get("ballots", []) if isinstance(data, dict) else []
    party_defs = data.get("party_ballots", []) if isinstance(data, dict) else []

    x = 0.0
    for b in ballots or []:
        apps = b.get("approvals", [])
        try:
            w = float(b.get("weight", 1.0))
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        if not apps:
            continue
        if isinstance(apps, (list, tuple, set)) and len(apps) == 0:
            continue
        x += w

    y = 0.0
    for d in party_defs or []:
        if not isinstance(d, dict):
            continue
        if d.get("weight") is not None:
            try:
                w = float(d.get("weight"))
            except Exception:
                w = 0.0
            if w > 0:
                y += w
        elif d.get("abs_weight") is not None:
            try:
                w = float(d.get("abs_weight"))
            except Exception:
                w = 0.0
            if w > 0:
                y += w

    n = None
    for k in ("n", "n_ballots", "total_ballots", "ballots_n"):
        if isinstance(data, dict) and data.get(k) is not None:
            try:
                n = float(data.get(k))
            except Exception:
                n = None
            break
    if n is None or n <= 0:
        n = max(x, y, 0.0)
    # Ensure n is at least max(x,y) (as per spec).
    n = max(n, x, y)
    return float(x), float(y), float(n)


def compute_profile_12_324_multipliers(profile_key: str, x: float, y: float, n: float) -> Tuple[float, float]:
    """Return (voter_mult, party_mult) for profile '12' or '324'."""
    profile_key = str(profile_key)
    x = float(x)
    y = float(y)
    n = float(n)
    if n <= 0:
        return 1.0, 1.0
    delta = (2.0 * x) - y
    if profile_key == "324":
        voter_mult = (n + delta) * 2.0 / (9.0 * n)
        party_mult = (2.0 * n - delta) * 2.0 / (9.0 * n)
    else:  # "12"
        voter_mult = (n + delta) / (3.0 * n)
        party_mult = (2.0 * n - delta) / (3.0 * n)
    return float(voter_mult), float(party_mult)


def apply_constant_multiplier(groups: List[Group], multiplier: float, label: str) -> List[Group]:
    if abs(multiplier - 1.0) <= 1e-18:
        return list(groups)
    out: List[Group] = []
    for g in groups:
        out.append(_clone_group_with_weight(g, g.weight * multiplier, meta_update={f"mult_{label}": multiplier}))
    return out


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
            w = float(b.get("weight", 1.0))
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
            w_abs = float(d.get("weight", 0.0))
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
        w = float(b.get("weight", 1.0))
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
    # Canonical rule: rel_weight = N * quota_floor, where quota_floor = min((2/3)*share, 1/3)
    # This ensures the capped form: rel_weight = min((2/3)*w_abs, N/3) when share = w_abs/N.
    qf = quota_floor_from_share(float(share))
    return float(total_voter_ballots) * float(qf)


def _max_share_with_denoms(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (share_used, denom_used) given pairs of (numerator, denom).

    Each pair is interpreted as share = numerator/denom (with denom>0).
    The max share is selected; ties prefer the larger denom (stable cap behaviour).
    """
    best_share = 0.0
    best_denom = 0.0
    for num, denom in pairs:
        denom = float(denom)
        num = float(num)
        if denom <= 0:
            continue
        s = num / denom
        if s > best_share + 1e-18 or (abs(s - best_share) <= 1e-18 and denom > best_denom + 1e-18):
            best_share = s
            best_denom = denom
    return float(best_share), float(best_denom)


def compute_global_totals(data: dict) -> dict:
    """Compute global election totals.

    Expected input keys (top-level):
      - total_population (wglobal1)
      - total_enrollment (wglobal2)
      - total_turnout (wglobal3)

    Derived:
      - wglobal4 = max(wglobal3, sum(party abs), sum(partyrock abs), sum(base abs))

    If any of wglobal1-3 are missing/invalid, defaults are computed where possible:
      - total_turnout defaults to sum(base abs)
      - total_enrollment defaults to sum(electorate enrollment abs)
      - total_population defaults to total_enrollment
    """
    d = data or {}

    def _f(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    # Raw sums from input.
    base_abs = 0.0
    for b in d.get("ballots", []) or []:
        if not isinstance(b, dict):
            continue
        if not b.get("approvals"):
            continue
        w = _f(b.get("weight", 1.0))
        if w > 0:
            base_abs += w

    party_abs = 0.0
    for p in d.get("party_ballots", []) or []:
        if not isinstance(p, dict):
            continue
        w = _f(p.get("weight", p.get("abs_weight", 0.0)))
        if w > 0:
            party_abs += w

    partyrock_abs = 0.0
    for pr in d.get("partyrock_ballots", []) or []:
        if not isinstance(pr, dict):
            continue
        w = _f(pr.get("weight", 0.0))
        if w > 0:
            partyrock_abs += w

    # Globals.
    wglobal1 = _f(d.get("total_population", d.get("wglobal1")))
    wglobal2 = _f(d.get("total_enrollment", d.get("wglobal2")))
    wglobal3 = _f(d.get("total_turnout", d.get("wglobal3")))

    if wglobal3 <= 0:
        wglobal3 = base_abs
    if wglobal2 <= 0:
        # fallback: sum electorate enrollments if available
        for e in d.get("electorate_ballots", []) or []:
            if not isinstance(e, dict):
                continue
            wglobal2 += _f(e.get("weight", e.get("abs_weight", 0.0)))
    if wglobal1 <= 0:
        wglobal1 = wglobal2 if wglobal2 > 0 else max(wglobal3, 0.0)

    wglobal4 = max(wglobal3, party_abs, partyrock_abs, base_abs)
    return {
        "wglobal1": float(wglobal1),
        "wglobal2": float(wglobal2),
        "wglobal3": float(wglobal3),
        "wglobal4": float(wglobal4),
        "sum_base_abs": float(base_abs),
        "sum_party_abs": float(party_abs),
        "sum_partyrock_abs": float(partyrock_abs),
    }


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
    wglobal5: float,
    wglobal4: float,
    partyrock_abs_by_party: Optional[Dict[str, float]],
    partyrock_norm_sum_by_party: Optional[Dict[str, float]],
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

        # New global-share model (preferred when globals are provided):
        #   wp1 = abs party weight (json)
        #   wp2 = max(wp1, sum(partyrock abs for party))
        #   share1 = wp2 / wglobal4
        #   share2 = wp3 / wglobal5
        #   share_used = max(share1, share2)
        #   rel_weight = N_used * quota_floor(share_used)
        # Legacy support: if explicit population/quota_floor present, respect it.
        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        if (population is None and qf is None) and abs_weight is not None and (wglobal2 > 0 or wglobal4 > 0):
            wp1 = float(abs_weight)
            pr_abs = float((partyrock_abs_by_party or {}).get(gid, 0.0))
            wp2 = max(wp1, pr_abs)

            wp3 = float((partyrock_norm_sum_by_party or {}).get(gid, 0.0))

            share1 = (wp2 / wglobal4) if wglobal4 > 0 else 0.0
            share2 = (wp3 / wglobal5) if wglobal5 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
            # Global quota ballots use arena N = wglobal4 for rel_weight conversion.
            rel_w = normalize_rel_weight_from_share(share_used, wglobal4)
            meta_extra.update({
                "wp1": wp1,
                "wp2": wp2,
                "wp3": wp3,
                "share1": share1,
                "share2": share2,
                "share_used": share_used,
                "N_arena": float(wglobal4),
            })
            population = float(wglobal4)
        elif population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            if qf is None:
                qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, float(population))
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, wglobal4 if wglobal4 > 0 else float(population or 0.0))
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
            meta={"source": "party_ballots", **meta_extra},
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
    wglobal1: float,
    wglobal4: float,
    megarock_abs_by_mega: Optional[Dict[str, float]],
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

        # New global-share model (preferred when globals are provided):
        #   w1 = abs mega weight (json)
        #   w2 = max(w1, sum(megarock abs for mega))
        #   share1 = w1 / wglobal1
        #   share2 = w2 / wglobal1
        #   share_used = max(share1, share2)
        #   rel_weight = wglobal1 * quota_floor(share_used)
        # Legacy support: if explicit population/quota_floor present, respect it.
        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        if (population is None and qf is None) and abs_weight is not None and wglobal1 > 0:
            w1 = float(abs_weight)
            wmr_sum = float((megarock_abs_by_mega or {}).get(gid, 0.0))
            w2 = max(w1, wmr_sum)
            share1 = (w1 / wglobal1) if wglobal1 > 0 else 0.0
            share2 = (w2 / wglobal1) if wglobal1 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
            # Global quota ballots use arena N = wglobal4 for rel_weight conversion.
            rel_w = normalize_rel_weight_from_share(share_used, wglobal4)
            meta_extra.update({
                "w1": w1,
                "w2": w2,
                "share1": share1,
                "share2": share2,
                "share_used": share_used,
                "N_arena": float(wglobal4),
            })
            population = float(wglobal4)
        elif population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            if qf is None:
                qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, float(population))
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, wglobal4 if wglobal4 > 0 else float(population or 0.0))
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
            meta={"source": "mega_ballots", "where": where, **meta_extra},
        )
        groups.append(g)
        meta.append(g)

    return groups, meta


def parse_electorate_ballots(
    elect_defs: List[dict],
    wglobal2: float,
    wglobal4: float,
    candidate_set: Set[str],
    profile: ElectionProfile,
    base_electorate_info: Optional[Dict[str, dict]] = None,
    partyrock_electorate_info: Optional[Dict[str, dict]] = None,
) -> Tuple[List[Group], List[Group]]:
    """Parse electorate ballots and ensure electorate coverage for base voter + PartyRock registrations.

    - Each electorate def may include a turnout attribute (e.g. 'turnout', 'total_turnout', 'voter_turnout');
      this is preserved in meta and not used for weight calculations.
    - Electorate approvals are expanded with any candidates appearing in registered base ballots and PartyRock ballots.
    - If base ballots or PartyRock ballots reference electorates not present in elect_defs, new electorate groups are created with
      rel_weight=0 and quota_floor=None (audit-only placeholders unless user supplies weights).
    - Each electorate group receives meta fields:
        - 'turnout' (if provided)
        - 'counted_base_ballots' (sum of raw base ballot weights registered to that electorate)
        - 'counted_base_records' (number of ballot records registered to that electorate)
        - 'counted_partyrock_ballots' (count of PartyRock ballot records registered to that electorate)
        - 'counted_partyrock_records' (alias of counted_partyrock_ballots)
        - 'counted_partyrock_abs_weight' (sum of PartyRock absolute weights registered to that electorate)
    """
    groups: List[Group] = []
    meta: List[Group] = []

    base_electorate_info = base_electorate_info or {}
    partyrock_electorate_info = partyrock_electorate_info or {}
    # Index provided electorate defs by id
    provided: Dict[str, dict] = {}
    for i, d in enumerate(elect_defs or []):
        gid = str(d.get("id") or f"electorate_{i}")
        provided[gid] = d

    # Ensure we include electorates referenced by base ballots or PartyRock ballots
    all_eids = set(provided.keys()) | set(base_electorate_info.keys()) | set(partyrock_electorate_info.keys())

    for i, gid in enumerate(sorted(all_eids)):
        d = provided.get(gid, {"id": gid})

        # turnout metadata (not used in weight). We preserve the distinct keys when provided.
        turnout = d.get("turnout")
        total_turnout = d.get("total_turnout")
        voter_turnout = d.get("voter_turnout")
        # A convenience alias used by older code paths.
        turnout_alias = turnout if turnout is not None else (total_turnout if total_turnout is not None else voter_turnout)

        # electorate candidates from def + base ballots registered to the electorate
        raw_apps = set(str(x) for x in (d.get("candidates") or []))
        extra_base = base_electorate_info.get(gid, {}).get("candidates", set())
        extra_partyrock = partyrock_electorate_info.get(gid, {}).get("candidates", set())
        raw_apps |= set(str(x) for x in extra_base)
        raw_apps |= set(str(x) for x in extra_partyrock)

        apps = tuple(sorted(raw_apps & set(candidate_set)))

        # New global-share model (preferred when globals are provided):
        #   we1 = abs enrollment weight (json)
        #   we2 = abs electorate turnout (json)
        #   we3 = max(we1, sum(partyrock abs for electorate), sum(base abs for electorate))
        #   share1 = we2 / wglobal2
        #   share2 = we3 / wglobal4
        #   share_used = max(share1, share2)
        #   rel_weight = N_used * quota_floor(share_used)
        # Legacy support: if explicit population/quota_floor present, respect it.
        population = d.get("population")
        abs_weight = d.get("weight")
        qf = d.get("quota_floor")

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        be = base_electorate_info.get(gid, {})
        pr = partyrock_electorate_info.get(gid, {})

        # Derive we2/we3 from turnout + parsed counts (even for legacy, for meta/audit).
        we1 = float(abs_weight) if abs_weight is not None else 0.0
        we2 = float(turnout_alias) if turnout_alias is not None else 0.0
        we3 = max(
            we2,
            float(be.get("counted_base_ballots", 0.0)),
            float(pr.get("counted_partyrock_abs_weight", 0.0)),
        )
        meta_extra.update({"we1": we1, "we2": we2, "we3": we3})

        if (population is None and qf is None) and abs_weight is not None and (wglobal2 > 0 or wglobal4 > 0):
            share1 = (we1 / wglobal2) if wglobal2 > 0 else 0.0
            share2 = (we3 / wglobal4) if wglobal4 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
            # Global quota ballots use arena N = wglobal4 for rel_weight conversion.
            rel_w = normalize_rel_weight_from_share(share_used, wglobal4)
            meta_extra.update({
                "share1": share1,
                "share2": share2,
                "share_used": share_used,
                "N_arena": float(wglobal4),
            })
            population = float(wglobal4)
        elif population is not None and abs_weight is not None:
            share = float(abs_weight) / float(population) if float(population) > 0 else 0.0
            if qf is None:
                qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, float(population))
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = normalize_rel_weight_from_share(share, wglobal4 if wglobal4 > 0 else float(population or 0.0))
        else:
            rel_w = float(d.get("rel_weight", 0.0))

        counted_ballots = float(be.get("counted_base_ballots", 0.0))
        counted_records = int(be.get("counted_base_records", 0))

        counted_pr_abs_weight = float(pr.get("counted_partyrock_abs_weight", 0.0))
        counted_pr_records = int(pr.get("counted_partyrock_records", 0))
        counted_pr_ballots = int(pr.get("counted_partyrock_ballots", counted_pr_records))

        meta_obj = {
            "source": "electorate_ballots" if gid in provided else "electorate_ballots_auto",
            "turnout": turnout_alias,
            "total_turnout": total_turnout,
            "voter_turnout": voter_turnout,
            "counted_base_ballots": counted_ballots,
            "counted_base_records": counted_records,
            "counted_partyrock_abs_weight": counted_pr_abs_weight,
            "counted_partyrock_records": counted_pr_records,
            "counted_partyrock_ballots": counted_pr_ballots,
            **meta_extra,
        }

        g = Group(
            gid=gid,
            kind="electorate",
            approvals=apps,
            weight=profile.scale_electorate_rel_weight(rel_w),
            quota_floor=float(qf) if qf is not None else None,
            population=float(population) if population is not None else None,
            abs_weight=float(abs_weight) if abs_weight is not None else None,
            share=float(share) if share is not None else None,
            meta=meta_obj,
        )
        groups.append(g)
        meta.append(g)

    return groups, meta


def parse_partyrock_ballots(
    partyrock_defs: List[dict],
    electorate_groups: List[Group],
    candidate_set: Set[str],
) -> Tuple[List[Group], List[Group]]:
    """Parse PartyRock ballots.

    PartyRock ballots are parsed and normalized, but are NOT yet used by any profile algorithm.

    Each PartyRock ballot record is expected to include:
      - 'weight' (absolute weight)
      - 'approvals' (candidate approvals)
      - electorate reference: one of 'electorate', 'electorate_id', 'electorate_gid'
      - party reference: one of 'party', 'party_id', 'party_gid'

    PartyRock ballots are local-level ballots.

    Definitions:
      - wpr1 = abs PartyRock weight (json)
      - we1 = electorate abs enrollment (electorate abs_weight)
      - we2 = electorate turnout (electorate meta.turnout alias)
      - we3 = max(we1, sum(partyrock abs for electorate), sum(base abs for electorate))

    Electorate-normalized weight:
        wpr2 = wpr1 * we1 / we3

    Local share (for future quota use):
        share1 = wpr1 / we2
    """
    groups: List[Group] = []
    meta: List[Group] = []

    elect_by_gid: Dict[str, Group] = {g.gid: g for g in electorate_groups}

    def _electorate_we2_we3(e: Optional[Group]) -> Tuple[float, float, float]:
        """Return (we1, we2, we3) from an electorate group."""
        if e is None:
            return 0.0, 0.0, 0.0
        m = e.meta or {}
        we1 = float(e.abs_weight) if e.abs_weight is not None else 0.0
        # we2 is the electorate turnout; prefer derived meta if present.
        we2 = 0.0
        for k in ("we2", "turnout", "total_turnout", "voter_turnout"):
            if m.get(k) is not None:
                try:
                    we2 = float(m.get(k))
                    break
                except Exception:
                    pass
        # we3 is the max-scale electorate competition pool.
        we3 = 0.0
        for k in ("we3",):
            if m.get(k) is not None:
                try:
                    we3 = float(m.get(k))
                except Exception:
                    we3 = 0.0
        if we3 <= 0:
            we3 = max(
                we2,
                float(m.get("counted_base_ballots", 0.0) or 0.0),
                float(m.get("counted_partyrock_abs_weight", 0.0) or 0.0),
            )
        return float(we1), float(we2), float(we3)

    i = 0
    for d in partyrock_defs or []:
        if not isinstance(d, dict):
            continue
        raw_apps = [str(x) for x in (d.get("approvals", []) or [])]
        # Preserve input order for PartyRock ballots (used later for tie-breaking construction).
        seen = set()
        ordered = []
        for x in raw_apps:
            x = str(x)
            if x.strip() == "":
                continue
            if x in seen:
                continue
            seen.add(x)
            ordered.append(x)

        # Candidate universe filter
        cand_set = set(str(x) for x in candidate_set)
        ordered = [c for c in ordered if c in cand_set]
        if not ordered:
            continue

        apps = tuple(sorted(set(ordered)))
        if not apps:
            continue

        try:
            abs_w = float(d.get("weight", 0.0))
        except Exception:
            abs_w = 0.0
        if abs_w <= 0:
            continue

        eid = d.get("electorate")
        if eid is None:
            eid = d.get("electorate_id")
        if eid is None:
            eid = d.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        pid = d.get("party")
        if pid is None:
            pid = d.get("party_id")
        if pid is None:
            pid = d.get("party_gid")
        pid = str(pid) if pid is not None and str(pid).strip() != "" else "party_unknown"

        e = elect_by_gid.get(eid)
        we1, we2, we3 = _electorate_we2_we3(e)
        wpr2 = (abs_w * we1 / we3) if we3 > 0 and we1 > 0 else 0.0
        share1 = (abs_w / we2) if we2 > 0 else 0.0
        qf = quota_floor_from_share(share1) if we2 > 0 else 0.0
        rel_w = normalize_rel_weight_from_share(share1, we3) if we2 > 0 else 0.0

        g = Group(
            gid=f"partyrock_{i}",
            kind="partyrock",
            approvals=apps,
            weight=float(wpr2),
            quota_floor=float(qf),
            population=float(we3) if we3 > 0 else None,
            abs_weight=float(abs_w),
            share=float(share1) if we2 > 0 else None,
            meta={
                "source": "partyrock_ballots",
                "electorate": eid,
                "party": pid,
                "we1": we1,
                "we2": we2,
                "we3": we3,
                "wpr1": float(abs_w),
                "wpr2": float(wpr2),
                "share1": float(share1),
                "rel_weight": float(rel_w),
                "N_arena": float(we3),
                "share_denom": float(denom),
                "N_arena": float(we3),
                "share_denom": float(denom),
                "ordered_candidates": list(ordered),
            },
        )
        groups.append(g)
        meta.append(g)
        i += 1

    return groups, meta


def compute_partyrock_party_sums(partyrock_groups: List[Group]) -> Dict[str, float]:
    """Return mapping party_id -> sum of PartyRock electorate-normalized weights.

    PartyRock groups store their party id in meta['party'].
    """
    sums: Dict[str, float] = defaultdict(float)
    for g in partyrock_groups or []:
        if g.kind != "partyrock":
            continue
        pid = None
        if g.meta:
            pid = g.meta.get("party")
        pid = str(pid) if pid is not None and str(pid).strip() != "" else "party_unknown"
        sums[pid] += float(g.weight)
    return dict(sums)


def compute_partyrock_party_abs_sums(partyrock_defs: List[dict]) -> Dict[str, float]:
    """Return mapping party_id -> sum of PartyRock absolute weights (wpr1) from raw JSON defs."""
    out: Dict[str, float] = defaultdict(float)
    for d in partyrock_defs or []:
        if not isinstance(d, dict):
            continue
        try:
            w = float(d.get("weight", 0.0))
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        pid = d.get("party")
        if pid is None:
            pid = d.get("party_id")
        if pid is None:
            pid = d.get("party_gid")
        pid = str(pid) if pid is not None and str(pid).strip() != "" else "party_unknown"
        out[pid] += w
    return dict(out)


def compute_megarock_abs_sums(megarock_defs: List[dict]) -> Dict[str, float]:
    """Return mapping mega_id -> sum of MegaRock absolute weights (wmr) from raw JSON defs."""
    out: Dict[str, float] = defaultdict(float)
    for d in megarock_defs or []:
        if not isinstance(d, dict):
            continue
        try:
            w = float(d.get("weight", 0.0))
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        mid = d.get("mega")
        if mid is None:
            mid = d.get("mega_id")
        if mid is None:
            mid = d.get("mega_gid")
        mid = str(mid) if mid is not None and str(mid).strip() != "" else "mega_unknown"
        out[mid] += w
    return dict(out)


def parse_megarock_ballots(
    megarock_defs: List[dict],
    electorate_groups: List[Group],
    wglobal1: float,
    wglobal2: float,
    candidate_set: Set[str],
) -> Tuple[List[Group], List[Group]]:
    """Parse MegaRock ballots (parsing-only; not used by any profile algorithm yet).

    Each MegaRock ballot record is expected to include:
      - 'weight' (absolute weight wmr)
      - 'approvals' (candidate approvals)
      - electorate reference: one of 'electorate', 'electorate_id', 'electorate_gid'
      - mega reference: one of 'mega', 'mega_id', 'mega_gid'

    Local share definition:
      - we1 = electorate abs enrollment
      - share1 = wmr / (we1 * wglobal1 / wglobal2)

    Canonical quota mapping (for future use):
      quota_floor = min((2/3)*share1, 1/3)
      rel_weight = denom_used * quota_floor
    """
    groups: List[Group] = []
    meta: List[Group] = []

    elect_by_gid: Dict[str, Group] = {g.gid: g for g in electorate_groups}
    cand_set = set(str(x) for x in candidate_set)

    i = 0
    for d in megarock_defs or []:
        if not isinstance(d, dict):
            continue
        raw_apps = [str(x) for x in (d.get("approvals", []) or [])]
        seen = set()
        ordered: List[str] = []
        for x in raw_apps:
            x = str(x)
            if x.strip() == "" or x in seen:
                continue
            seen.add(x)
            if x in cand_set:
                ordered.append(x)
        if not ordered:
            continue
        apps = tuple(sorted(set(ordered)))
        if not apps:
            continue

        try:
            wmr = float(d.get("weight", 0.0))
        except Exception:
            wmr = 0.0
        if wmr <= 0:
            continue

        eid = d.get("electorate")
        if eid is None:
            eid = d.get("electorate_id")
        if eid is None:
            eid = d.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        mid = d.get("mega")
        if mid is None:
            mid = d.get("mega_id")
        if mid is None:
            mid = d.get("mega_gid")
        mid = str(mid) if mid is not None and str(mid).strip() != "" else "mega_unknown"

        e = elect_by_gid.get(eid)
        we1 = float(e.abs_weight) if (e is not None and e.abs_weight is not None) else 0.0
        # Arena N for local ballots is we3 (electorate competition pool).
        we3 = 0.0
        if e is not None and e.meta is not None and e.meta.get("we3") is not None:
            try:
                we3 = float(e.meta.get("we3"))
            except Exception:
                we3 = 0.0
        if we3 <= 0:
            # Fallback: derive we3 from available electorate meta
            m = e.meta if (e is not None and e.meta is not None) else {}
            we2 = 0.0
            for k in ("we2", "turnout", "total_turnout", "voter_turnout"):
                if m.get(k) is not None:
                    try:
                        we2 = float(m.get(k))
                        break
                    except Exception:
                        pass
            we3 = max(
                float(we2 or 0.0),
                float(m.get("counted_base_ballots", 0.0) or 0.0),
                float(m.get("counted_partyrock_abs_weight", 0.0) or 0.0),
            )
        denom = (we1 * float(wglobal1) / float(wglobal2)) if (we1 > 0 and wglobal1 > 0 and wglobal2 > 0) else 0.0
        share1 = (wmr / denom) if denom > 0 else 0.0
        qf = quota_floor_from_share(share1) if denom > 0 else 0.0
        rel_w = normalize_rel_weight_from_share(share1, we3) if denom > 0 else 0.0

        g = Group(
            gid=f"megarock_{i}",
            kind="megarock",
            approvals=apps,
            weight=float(wmr),
            quota_floor=float(qf) if denom > 0 else None,
            population=float(we3) if we3 > 0 else None,
            abs_weight=float(wmr),
            share=float(share1) if denom > 0 else None,
            meta={
                "source": "megarock_ballots",
                "electorate": eid,
                "mega": mid,
                "we1": we1,
                "denom": float(denom),
                "share1": float(share1),
                "rel_weight": float(rel_w),
                "N_arena": float(we3),
                "share_denom": float(denom),
                "ordered_candidates": list(ordered),
            },
        )
        groups.append(g)
        meta.append(g)
        i += 1

    return groups, meta


def extract_candidates_from_defs(defs: List[dict]) -> Set[str]:
    s: Set[str] = set()
    for d in defs or []:
        s.update(str(x) for x in (d.get("candidates") or d.get("list") or []))
    return s

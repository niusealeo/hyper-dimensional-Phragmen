from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import csv
import json
import math
import ast

from ..types import Group, ElectionProfile
from .numeric import parse_numeric, parse_numeric_optional
from ..derive.normalisation import (
    _clone_group_with_weight,
    compute_profile_12_324_totals,
    compute_profile_12_324_multipliers,
    apply_constant_multiplier,
)
from ..derive.electorate_stats import (
    compute_base_electorate_info,
    compute_partyrock_electorate_info,
)
from ..output.audit_groups import (
    write_meta_csv,
    write_globals_csv,
    write_globals_kv_csv,
    write_groups_audit_csv,
    write_party_lists_csv,
)


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


# ============================================================
# SAFE NUMERIC EXPRESSION PARSER
# ============================================================

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def parse_party_ballots(
    party_defs: List[dict],
    wglobal4: float,
    wglobal5: float,
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
        population = parse_numeric_optional(d.get("population"), field="party.population")
        abs_weight = parse_numeric_optional(d.get("weight", d.get("abs_weight")), field="party.weight")
        share_in = parse_numeric_optional(d.get("share"), field="party.share")
        qf = parse_numeric_optional(d.get("quota_floor"), field="party.quota_floor")
        qf = clamp_quota_floor(qf) if qf is not None else None

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        # Priority: weight -> share -> quota_floor
        if abs_weight is not None and (wglobal5 > 0 or wglobal4 > 0):
            wp1 = float(abs_weight)
            pr_abs = float((partyrock_abs_by_party or {}).get(gid, 0.0))
            wp2 = max(wp1, pr_abs)
            share1 = (wp2 / wglobal4) if wglobal4 > 0 else 0.0
            wp3 = float((partyrock_norm_sum_by_party or {}).get(gid, 0.0))
            share2 = (wp3 / wglobal5) if wglobal5 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
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
        elif abs_weight is not None and population is not None:
            denom = float(population)
            share = float(abs_weight) / denom if denom > 0 else 0.0
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, denom)
        elif share_in is not None:
            share = float(share_in)
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, wglobal4)
            population = float(wglobal4)
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = float(wglobal4) * float(qf)
            population = float(wglobal4)
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
        population = parse_numeric_optional(d.get("population"), field="mega.population")
        abs_weight = parse_numeric_optional(d.get("weight", d.get("abs_weight")), field="mega.weight")
        share_in = parse_numeric_optional(d.get("share"), field="mega.share")
        qf = parse_numeric_optional(d.get("quota_floor"), field="mega.quota_floor")
        qf = clamp_quota_floor(qf) if qf is not None else None

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        # Priority: weight -> share -> quota_floor
        if abs_weight is not None and wglobal1 > 0:
            w1 = float(abs_weight)
            wmr_sum = float((megarock_abs_by_mega or {}).get(gid, 0.0))
            w2 = max(w1, wmr_sum)
            share1 = (w1 / wglobal1) if wglobal1 > 0 else 0.0
            share2 = (w2 / wglobal1) if wglobal1 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
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
        elif abs_weight is not None and population is not None:
            denom = float(population)
            share = float(abs_weight) / denom if denom > 0 else 0.0
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, denom)
        elif share_in is not None:
            share = float(share_in)
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, wglobal4)
            population = float(wglobal4)
        elif qf is not None:
            # Cap already applied above.
            share = share_from_quota_floor(float(qf))
            rel_w = float(wglobal4) * float(qf)
            population = float(wglobal4)
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
    wglobal5: float,
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
        #   share1 = we2 / wglobal5
        #   share2 = we3 / wglobal4
        #   share_used = max(share1, share2)
        #   rel_weight = N_used * quota_floor(share_used)
        # Legacy support: if explicit population/quota_floor present, respect it.
        population = parse_numeric_optional(d.get("population"), field="electorate.population")
        abs_weight = parse_numeric_optional(d.get("weight", d.get("abs_weight")), field="electorate.weight")
        share_in = parse_numeric_optional(d.get("share"), field="electorate.share")
        qf = parse_numeric_optional(d.get("quota_floor"), field="electorate.quota_floor")
        qf = clamp_quota_floor(qf) if qf is not None else None

        share = None
        rel_w = 0.0
        meta_extra: Dict[str, Any] = {}

        be = base_electorate_info.get(gid, {})
        pr = partyrock_electorate_info.get(gid, {})

        # Derive we2/we3 from turnout + parsed counts (even for legacy, for meta/audit).
        we1 = float(abs_weight) if abs_weight is not None else 0.0
        we2 = float(parse_numeric_optional(turnout_alias, field="electorate.turnout") or 0.0)
        we3 = max(
            we2,
            float(parse_numeric_optional(be.get("counted_base_ballots", 0.0), field="electorate.counted_base_ballots") or 0.0),
            float(parse_numeric_optional(pr.get("counted_partyrock_abs_weight", 0.0), field="electorate.counted_partyrock_abs_weight") or 0.0),
        )
        meta_extra.update({"we1": we1, "we2": we2, "we3": we3})

        # Priority: weight -> share -> quota_floor
        if abs_weight is not None and (wglobal5 > 0 or wglobal4 > 0):
            share1 = (we2 / wglobal5) if wglobal5 > 0 else 0.0
            share2 = (we3 / wglobal4) if wglobal4 > 0 else 0.0
            share_used = max(share1, share2)
            share = share_used
            qf = quota_floor_from_share(share_used)
            rel_w = normalize_rel_weight_from_share(share_used, wglobal4)
            meta_extra.update({
        "share1": share1,
        "share2": share2,
        "share_used": share_used,
        "N_arena": float(wglobal4),
            })
            population = float(wglobal4)
        elif abs_weight is not None and population is not None:
            denom = float(population)
            share = float(abs_weight) / denom if denom > 0 else 0.0
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, denom)
        elif share_in is not None:
            share = float(share_in)
            qf = quota_floor_from_share(share)
            rel_w = normalize_rel_weight_from_share(share, wglobal4)
            population = float(wglobal4)
        elif qf is not None:
            share = share_from_quota_floor(float(qf))
            rel_w = float(wglobal4) * float(qf)
            population = float(wglobal4)
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
      - we3 = max(we2, sum(partyrock abs for electorate), sum(base abs for electorate))

    Electorate-normalized weight:
        wpr2 = wpr1 * we1 / we3

    Local share (for quota floor):
        share1 = wpr1 / we2

    Arena N mapping:
        N_arena = we3

    Canonical quota rule:
        quota_floor = min((2/3)*share1, 1/3)
        rel_weight = N_arena * quota_floor
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

        # we2: turnout (prefer explicit turnout meta if present)
        we2 = 0.0
        for k in ("we2", "turnout", "total_turnout", "voter_turnout"):
            if m.get(k) is not None:
                try:
                    we2 = float(m.get(k))
                    break
                except Exception:
                    pass

        # we3: local max-scale competition pool
        we3 = 0.0
        if m.get("we3") is not None:
            try:
                we3 = float(m.get("we3"))
            except Exception:
                we3 = 0.0

        if we3 <= 0:
            we3 = max(
                we2,
                float(m.get("counted_base_ballots", 0.0) or 0.0),
                float(m.get("counted_partyrock_abs_weight", 0.0) or 0.0),
            )

        return float(we1), float(we2), float(we3)

    cand_set = set(str(x) for x in candidate_set)

    i = 0
    for d in partyrock_defs or []:
        if not isinstance(d, dict):
            continue

        # Preserve PartyRock input order for later tie-breaking list work
        raw_apps = [str(x) for x in (d.get("approvals", []) or [])]
        seen = set()
        ordered: List[str] = []
        for x in raw_apps:
            x = str(x)
            if x.strip() == "":
                continue
            if x in seen:
                continue
            seen.add(x)
            ordered.append(x)

        # Candidate universe filter
        ordered = [c for c in ordered if c in cand_set]
        if not ordered:
            continue

        apps = tuple(sorted(set(ordered)))
        if not apps:
            continue

        # Absolute weight (wpr1)
        try:
            abs_w = float(parse_numeric_optional(d.get("weight", 0.0), field="partyrock.weight") or 0.0)
        except Exception:
            abs_w = 0.0
        if abs_w <= 0:
            continue

        # Electorate id
        eid = d.get("electorate")
        if eid is None:
            eid = d.get("electorate_id")
        if eid is None:
            eid = d.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        # Party id
        pid = d.get("party")
        if pid is None:
            pid = d.get("party_id")
        if pid is None:
            pid = d.get("party_gid")
        pid = str(pid) if pid is not None and str(pid).strip() != "" else "party_unknown"

        e = elect_by_gid.get(eid)
        we1, we2, we3 = _electorate_we2_we3(e)

        # wpr2 electorate-normalised
        wpr2 = (abs_w * we1 / we3) if (we3 > 0 and we1 > 0) else 0.0

        # share1 uses we2 as denominator
        share1 = (abs_w / we2) if we2 > 0 else 0.0

        # quota + rel_weight using arena N=we3
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

                # electorate scalars
                "we1": float(we1),
                "we2": float(we2),
                "we3": float(we3),

                # partyrock scalars
                "wpr1": float(abs_w),
                "wpr2": float(wpr2),

                # local share definition
                "share1": float(share1),
                "share_denom": float(we2),   # share1 = wpr1 / we2

                # quota projection info
                "quota_floor": float(qf),
                "N_arena": float(we3),
                "rel_weight": float(rel_w),

                # ordered list for later tie-breaking/list work
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
            w = float(parse_numeric_optional(d.get("weight", 0.0), field="weight") or 0.0)
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
            w = float(parse_numeric_optional(d.get("weight", 0.0), field="weight") or 0.0)
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
    wglobal5: float,
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
      - share1 = wmr / (we1 * wglobal1 / wglobal5)

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
            wmr = float(parse_numeric_optional(d.get("weight", 0.0), field="megarock.weight") or 0.0)
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
        denom = (we1 * float(wglobal1) / float(wglobal5)) if (we1 > 0 and wglobal1 > 0 and wglobal5 > 0) else 0.0
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

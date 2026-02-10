from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import math

from .types import EPS, Group


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
    "spend_mode",
    "dt0_tie_rule",
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
    dt0_tie_rule: str = "party_then_name",
) -> Tuple[str, float, float, float, bool]:
    """
    Select by minimum dt as usual.
    Special: if best dt == 0 and dt0_tie_rule = max_have_then_party_then_name,
    choose among dt==0 ties the candidate with greatest 'have' first.
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
    tied = [(dt, c, have, rate) for (dt, c, have, rate) in scored if abs(dt - best_dt) <= 1e-12]

    if len(tied) == 1:
        dt, c, have, rate = tied[0]
        return c, dt, have, rate, allow_used

    tied_candidates = [c for (_dt, c, _have, _rate) in tied]

    if abs(best_dt) <= 1e-12 and dt0_tie_rule == "max_have_then_party_then_name":
        # Pick candidate with greatest available spending balance ('have') first.
        max_have = max(h for (_dt, _c, h, _r) in tied)
        tied2 = [t for t in tied if abs(t[2] - max_have) <= 1e-12]
        if len(tied2) == 1:
            dt, c, have, rate = tied2[0]
            return c, dt, have, rate, allow_used
        tied_candidates2 = [c for (_dt, c, _have, _rate) in tied2]
        chosen = tie_break_by_party_order(tied_candidates2, party_lists)
        for dt, c, have, rate in tied2:
            if c == chosen:
                return c, dt, have, rate, allow_used
        chosen = min(tied_candidates2)
        for dt, c, have, rate in tied2:
            if c == chosen:
                return c, dt, have, rate, allow_used

    # Default: party tie-break then name
    chosen = tie_break_by_party_order(tied_candidates, party_lists)
    # Return its tuple
    for dt, c, have, rate in tied:
        if c == chosen:
            return c, dt, have, rate, allow_used

    dt, chosen, have, rate = tied[0]
    return chosen, dt, have, rate, allow_used


def compute_projection_delta_for_chosen(
    chosen: str,
    groups: List[Group],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    base_used: List[bool],
) -> float:
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


def spend_winner_reset(
    cand: str,
    groups: List[Group],
    balances: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
) -> None:
    """
    Old behaviour: reset all active supporters' balances to 0.
    Dormant quota balances are not touched because active_mask is False for them.
    """
    for gi in supporters.get(cand, []):
        if active_mask[gi]:
            balances[gi] = 0.0


def _spend_from_indices_proportional(balances: List[float], indices: List[int], amount: float) -> float:
    """
    Spend up to 'amount' from balances[indices], proportionally.
    Returns the amount actually spent (<= amount).
    """
    if amount <= 0:
        return 0.0
    total = sum(max(balances[i], 0.0) for i in indices)
    if total <= EPS:
        return 0.0
    if total <= amount + EPS:
        # take all
        for i in indices:
            balances[i] = 0.0
        return total
    # partial proportional
    frac = (amount / total)
    for i in indices:
        b = max(balances[i], 0.0)
        balances[i] = b * (1.0 - frac)
    return amount


def spend_winner_partial_priority(
    cand: str,
    groups: List[Group],
    balances: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    priority: List[str],
) -> None:
    """
    Spend EXACTLY 1 seat value (or as much as possible if infeasible),
    preserving overshoot leftovers.

    Priority order (your spec):
      base → electorate → party → mega

    Rules:
      - Only groups with active_mask True are spendable this round
        (so dormant quota reserves remain untouched).
      - Within each kind, spending is proportional if partial.
    """
    need = 1.0

    # Partition supporter indices by kind, respecting active_mask
    by_kind: Dict[str, List[int]] = {k: [] for k in priority}
    for gi in supporters.get(cand, []):
        if not active_mask[gi]:
            continue
        k = groups[gi].kind
        if k in by_kind:
            by_kind[k].append(gi)

    for k in priority:
        if need <= EPS:
            break
        idxs = by_kind.get(k, [])
        if not idxs:
            continue
        spent = _spend_from_indices_proportional(balances, idxs, need)
        need -= spent

    # If need > 0 here, it means: despite dt=0 or dt>0, the active supporters couldn't cover 1.
    # That should be rare (and indicates inconsistent dt computation), but we fail safe by not going negative.

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


# -----------------------
# dt / have computations
# -----------------------

def candidate_dt_balances(
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


def candidate_dt_fifo(
    cand: str,
    t_now: float,
    t_start: List[float],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
) -> Optional[Tuple[float, float, float]]:
    have = 0.0
    rate = 0.0
    for gi in supporters.get(cand, []):
        if not active_mask[gi]:
            continue
        w = weights[gi]
        if w <= EPS:
            continue
        b = w * max(0.0, t_now - t_start[gi])
        have += b
        rate += w
    if rate <= EPS:
        return None
    if have >= 1.0 - EPS:
        return (0.0, have, rate)
    return ((1.0 - have) / rate, have, rate)


# -----------------------
# Party tie-break support
# -----------------------

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
    # balances mode
    balances: Optional[List[float]],
    # fifo mode
    t_now: Optional[float],
    t_start: Optional[List[float]],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    party_lists: Dict[str, List[str]],
    allow_only_pool: Optional[Set[str]] = None,
    dt0_tie_rule: str = "party_then_name",
    spend_mode: str = "reset",
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
        if spend_mode == "fifo_time_priority":
            assert t_now is not None and t_start is not None
            res = candidate_dt_fifo(c, t_now, t_start, weights, supporters, active_mask)
        else:
            assert balances is not None
            res = candidate_dt_balances(c, balances, weights, supporters, active_mask)
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

    chosen = tie_break_by_party_order(tied_candidates, party_lists)
    for dt, c, have, rate in tied:
        if c == chosen:
            return c, dt, have, rate, allow_used

    dt, chosen, have, rate = tied[0]
    return chosen, dt, have, rate, allow_used


# -----------------------
# Projection accounting
# -----------------------

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


# -----------------------
# Spending: reset (legacy)
# -----------------------

def spend_winner_reset(
    cand: str,
    balances: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
) -> None:
    for gi in supporters.get(cand, []):
        if active_mask[gi]:
            balances[gi] = 0.0


# -----------------------
# FIFO time-priority spending
# -----------------------

def _available_credit_fifo(t_now: float, t_start: List[float], weights: List[float], indices: List[int]) -> float:
    tot = 0.0
    for i in indices:
        w = weights[i]
        if w <= EPS:
            continue
        dt = t_now - t_start[i]
        if dt > 0:
            tot += w * dt
    return tot


def _fifo_cutoff_tau(
    t_now: float,
    t_start: List[float],
    weights: List[float],
    indices: List[int],
    amount: float,
) -> float:
    """
    Find τ ≤ t_now such that sum_i w_i * max(0, τ - t_start[i]) = amount,
    considering only indices list, and assuming amount <= total available credit up to t_now.
    """
    if amount <= 0:
        return min(t_start[i] for i in indices) if indices else t_now

    items: List[Tuple[float, float]] = []  # (t_start, weight)
    for i in indices:
        w = weights[i]
        if w <= EPS:
            continue
        items.append((t_start[i], w))
    if not items:
        return t_now

    items.sort(key=lambda x: x[0])
    starts = [s for (s, _w) in items]
    ws = [w for (_s, w) in items]

    # Sweep segments
    spent = 0.0
    W = 0.0

    # Start at the smallest start time
    cur = starts[0]
    W = ws[0]
    j = 0

    # If multiple have same start
    while j + 1 < len(starts) and abs(starts[j + 1] - cur) <= 1e-18:
        j += 1
        W += ws[j]

    # Next change points are either next start time or t_now
    while True:
        nxt = t_now
        if j + 1 < len(starts):
            nxt = min(nxt, starts[j + 1])

        if nxt < cur:
            nxt = cur

        cap = W * (nxt - cur)
        if spent + cap >= amount - 1e-15:
            # Solve within this segment
            if W <= EPS:
                return cur
            tau = cur + (amount - spent) / W
            if tau > t_now:
                tau = t_now
            return tau

        spent += cap
        cur = nxt

        if abs(cur - t_now) <= 1e-18:
            return t_now

        # Add next group(s)
        while j + 1 < len(starts) and abs(starts[j + 1] - cur) <= 1e-18:
            j += 1
            W += ws[j]
        if j + 1 < len(starts):
            j += 1
            W += ws[j]
            # absorb any equals
            while j + 1 < len(starts) and abs(starts[j + 1] - cur) <= 1e-18:
                j += 1
                W += ws[j]


def _fifo_spend_exact_amount(
    t_now: float,
    t_start: List[float],
    weights: List[float],
    indices: List[int],
    amount: float,
) -> float:
    """
    Spend exactly 'amount' by FIFO (oldest credit first) within these indices.
    Implemented by computing cutoff τ and advancing start times to τ (or t_now if drained).
    Returns amount actually spent (<= amount).
    """
    if amount <= 0 or not indices:
        return 0.0

    avail = _available_credit_fifo(t_now, t_start, weights, indices)
    if avail <= EPS:
        return 0.0

    if amount >= avail - 1e-15:
        # Drain all: set their start to now
        for i in indices:
            t_start[i] = t_now
        return avail

    tau = _fifo_cutoff_tau(t_now, t_start, weights, indices, amount)

    # Advance start times up to τ where applicable
    spent = 0.0
    for i in indices:
        w = weights[i]
        if w <= EPS:
            continue
        s = t_start[i]
        if tau > s:
            # spent from [s, tau]
            spent_i = w * (tau - s)
            spent += spent_i
            t_start[i] = tau

    # Numerical trim: we may have spent slightly > amount due to floating error; clamp not needed for t_start.
    return min(spent, amount)


def spend_winner_fifo_time_priority(
    cand: str,
    groups: List[Group],
    t_now: float,
    t_start: List[float],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    priority: List[str],
) -> None:
    """
    Spend exactly 1 seat value by FIFO time priority, tiered:
      base → electorate → party → mega

    Only spend from indices that are active this round (active_mask True).
    Dormant quota reserves remain untouched.
    """
    need = 1.0

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
        spent = _fifo_spend_exact_amount(t_now, t_start, weights, idxs, need)
        need -= spent

    # If need > 0, it means the candidate wasn't actually payable; that's a consistency bug upstream.
    # We fail safe by leaving it partially unpaid (no negative effects), but it should not happen.


def fifo_balance_at(
    t_now: float,
    t_start: List[float],
    weights: List[float],
    i: int,
) -> float:
    w = weights[i]
    if w <= EPS:
        return 0.0
    dt = t_now - t_start[i]
    return w * dt if dt > 0 else 0.0

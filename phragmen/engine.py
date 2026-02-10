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
    # Franchise / projection
    "total_voter_ballots",
    "delta_voter_ballots_used",
    "delta_projection",
    "total_projection",
    "used_voter_ballots_cum",
    "projection_interval_prev",
    "projection_interval_curr",
    # Quota activation bookkeeping
    "active_quota_groups",
    # Intervention bookkeeping
    "prefix_allow_pool_size_before",
    "iter_allow_pool_size_before",
    "allow_pool_source",
    "allow_only_used",
    # Settings snapshot
    "spend_mode",
    "dt0_tie_rule",
    "spend_tiers",
    "tier_within_mode",
]

QUOTA_FIELDS = [
    "label",
    "round",
    "chosen",
    "quota_gid",
    "quota_kind",
    "quota_floor",
    "required_by_round",
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


def quota_required(quota_floor: float, current_round: int) -> int:
    return int(math.ceil(quota_floor * current_round - 1e-15))


def compute_quota_active_info(
    quota_groups: List[Group],
    winners_so_far: List[str],
    current_round: int,
) -> Dict[str, Tuple[bool, int, int]]:
    """Quota activation is computed strictly at current_round (no projection look-ahead)."""
    info: Dict[str, Tuple[bool, int, int]] = {}
    winners_set = set(winners_so_far)
    for g in quota_groups:
        qf = float(g.quota_floor or 0.0)
        req = quota_required(qf, current_round)
        if req <= 0:
            info[g.gid] = (False, 0, 0)
            continue
        s = set(g.approvals)
        in_set = sum(1 for w in winners_set if w in s)
        active = in_set < req
        info[g.gid] = (active, in_set, req)
    return info


def fifo_balance_at(t_now: float, t_start: List[float], weights: List[float], i: int) -> float:
    w = weights[i]
    if w <= EPS:
        return 0.0
    dt = t_now - t_start[i]
    return w * dt if dt > 0 else 0.0


def _available_credit_fifo(t_now: float, t_start: List[float], weights: List[float], indices: List[int]) -> float:
    return sum(fifo_balance_at(t_now, t_start, weights, i) for i in indices)


def _fifo_cutoff_tau(
    t_now: float,
    t_start: List[float],
    weights: List[float],
    indices: List[int],
    amount: float,
) -> float:
    if amount <= 0:
        return min((t_start[i] for i in indices), default=t_now)

    items: List[Tuple[float, float]] = []
    for i in indices:
        w = weights[i]
        if w > EPS:
            items.append((t_start[i], w))
    if not items:
        return t_now

    items.sort(key=lambda x: x[0])
    starts = [s for (s, _w) in items]
    ws = [w for (_s, w) in items]

    spent = 0.0
    W = 0.0

    cur = starts[0]
    j = 0
    while j < len(starts) and abs(starts[j] - cur) <= 1e-18:
        W += ws[j]
        j += 1

    while True:
        nxt = t_now
        if j < len(starts):
            nxt = min(nxt, starts[j])
        if nxt < cur:
            nxt = cur

        cap = W * (nxt - cur)
        if spent + cap >= amount - 1e-15:
            if W <= EPS:
                return cur
            return min(cur + (amount - spent) / W, t_now)

        spent += cap
        cur = nxt
        if abs(cur - t_now) <= 1e-18:
            return t_now

        while j < len(starts) and abs(starts[j] - cur) <= 1e-18:
            W += ws[j]
            j += 1


def _fifo_spend_exact_amount(
    t_now: float,
    t_start: List[float],
    weights: List[float],
    indices: List[int],
    amount: float,
) -> float:
    if amount <= 0 or not indices:
        return 0.0
    avail = _available_credit_fifo(t_now, t_start, weights, indices)
    if avail <= EPS:
        return 0.0
    if amount >= avail - 1e-15:
        for i in indices:
            t_start[i] = t_now
        return avail

    tau = _fifo_cutoff_tau(t_now, t_start, weights, indices, amount)
    spent = 0.0
    for i in indices:
        w = weights[i]
        if w <= EPS:
            continue
        s = t_start[i]
        if tau > s:
            spent_i = w * (tau - s)
            spent += spent_i
            t_start[i] = tau
    return min(spent, amount)


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
        have += w * max(0.0, t_now - t_start[gi])
        rate += w
    if rate <= EPS:
        return None
    if have >= 1.0 - EPS:
        return (0.0, have, rate)
    return ((1.0 - have) / rate, have, rate)


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


def choose_candidate_for_round(
    remaining: List[str],
    balances: Optional[List[float]],
    t_now: Optional[float],
    t_start: Optional[List[float]],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    party_lists: Dict[str, List[str]],
    allow_only_pool: Optional[Set[str]] = None,
    dt0_tie_rule: str = "party_then_name",
    spend_mode: str = "fifo_time_priority",
) -> Tuple[str, float, float, float, bool]:
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
        raise ValueError("No candidates affordable (no active supporters).")

    scored.sort(key=lambda x: (x[0], x[1]))
    best_dt = scored[0][0]
    tied = [(dt, c, have, rate) for (dt, c, have, rate) in scored if abs(dt - best_dt) <= 1e-12]

    if len(tied) == 1:
        dt, c, have, rate = tied[0]
        return c, dt, have, rate, allow_used

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

    tied_candidates = [c for (_dt, c, _have, _rate) in tied]
    chosen = tie_break_by_party_order(tied_candidates, party_lists)
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
    balances: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
) -> None:
    for gi in supporters.get(cand, []):
        if active_mask[gi]:
            balances[gi] = 0.0


def parse_spend_tiers(spec: str) -> List[List[str]]:
    valid = {"base", "electorate", "party", "mega"}
    tiers: List[List[str]] = []
    for tier_str in spec.split(">"):
        kinds = [k.strip() for k in tier_str.split(",") if k.strip()]
        if not kinds:
            continue
        for k in kinds:
            if k not in valid:
                raise ValueError(f"Invalid kind '{k}' in spend_tiers. Valid: {sorted(valid)}")
        tiers.append(kinds)
    if not tiers:
        raise ValueError("spend_tiers produced zero tiers.")
    return tiers


def spend_winner_fifo_time_priority_tiers(
    cand: str,
    groups: List[Group],
    t_now: float,
    t_start: List[float],
    weights: List[float],
    supporters: Dict[str, List[int]],
    active_mask: List[bool],
    spend_tiers: List[List[str]],
    tier_within_mode: str = "combined_fifo",
) -> None:
    need = 1.0
    by_kind: Dict[str, List[int]] = {}
    for gi in supporters.get(cand, []):
        if not active_mask[gi]:
            continue
        by_kind.setdefault(groups[gi].kind, []).append(gi)

    for tier in spend_tiers:
        if need <= EPS:
            break

        if tier_within_mode == "separate_by_kind":
            for k in tier:
                if need <= EPS:
                    break
                idxs = by_kind.get(k, [])
                if not idxs:
                    continue
                spent = _fifo_spend_exact_amount(t_now, t_start, weights, idxs, need)
                need -= spent
        else:
            idxs_all: List[int] = []
            for k in tier:
                idxs_all.extend(by_kind.get(k, []))
            if not idxs_all:
                continue
            spent = _fifo_spend_exact_amount(t_now, t_start, weights, idxs_all, need)
            need -= spent

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import math

from .types import EPS, Group


# ---------------------------
# CSV schemas (shared with CLI)
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
# Mechanics
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

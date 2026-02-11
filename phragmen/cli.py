from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import argparse
import csv
import json
import os

from . import io as io_mod
from .engine import (
    run_fifo_ordering_general_alpha,
    ROUND_FIELDS, QUOTA_FIELDS, PROJ_FIELDS,
    build_supporters, choose_candidate_for_round,
    compute_projection_delta_for_chosen,
    compute_quota_active_info,
    fifo_balance_at,
    parse_spend_tiers,
    spend_winner_fifo_time_priority_tiers,
    spend_winner_reset,
)
from .profiles import get_profile, list_profiles
from .types import ElectionProfile, Group


def _build_candidate_discrepancy_map(quota_groups: List[Group], quota_info: Dict[str, Tuple[bool, int, int]]) -> Dict[str, float]:
    """Return mapping candidate -> max (required - in_set) across approving quota groups."""
    out: Dict[str, float] = {}
    for g in quota_groups:
        if g.gid not in quota_info:
            continue
        _active, in_set, req = quota_info[g.gid]
        disc = float(max(int(req) - int(in_set), 0))
        if disc <= 0:
            continue
        for c in g.approvals:
            prev = float(out.get(c, 0.0))
            if disc > prev:
                out[c] = disc
    return out


def open_csv(path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


def choose_profile_interactive() -> ElectionProfile:
    profiles = list_profiles()
    print("Select election profile:")
    for i, p in enumerate(profiles, start=1):
        print(f"  {i}) {p.name}  [{p.key}]")
        print(f"     {p.description}")
    while True:
        s = input(f"Enter 1-{len(profiles)} (default 1): ").strip()
        if s == "":
            return profiles[0]
        try:
            n = int(s)
            if 1 <= n <= len(profiles):
                return profiles[n - 1]
        except ValueError:
            pass
        print("Invalid selection.")


def prompt_more_iters(default: int) -> Optional[int]:
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


def select_pool_by_coverage(
    winners: List[str],
    intervals: List[Tuple[float, float]],
    a: float,
    b: float,
) -> List[str]:
    out: List[str] = []
    for w, (p0, p1) in zip(winners, intervals):
        if p0 < b - 1e-15 and p1 >= a - 1e-15:
            out.append(w)
    return out


def prefix_until_projection_strict_gt(
    winners: List[str],
    intervals: List[Tuple[float, float]],
    target: float,
) -> List[str]:
    out: List[str] = []
    for w, (_p0, p1) in zip(winners, intervals):
        out.append(w)
        if p1 > target + 1e-15:
            break
    return out


def run_sequential(
    label: str,
    candidates: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    electorate_groups: List[Group],
    party_lists: Dict[str, List[str]],
    mega_lists: Optional[Dict[str, List[str]]],
    stop_when_proj_gt: Optional[float],
    prefix_allow_only_init: Optional[List[str]],
    ban_set: Optional[Set[str]],
    iter_allow_only_init: Optional[List[str]],
    rounds_csv_path: Optional[str],
    quota_csv_path: Optional[str],
    proj_csv_path: Optional[str],
    spend_mode: str,
    dt0_tie_rule: str,
    spend_tiers_spec: str,
    tier_within_mode: str,
) -> dict:
    spend_tiers = parse_spend_tiers(spend_tiers_spec)

    groups = base_groups + mega_groups + party_groups + electorate_groups
    weights = [g.weight for g in groups]
    mega_indices = [i for i, g in enumerate(groups) if g.kind == "mega"]
    supporters = build_supporters(groups, candidates)

    gid_to_index = {g.gid: i for i, g in enumerate(groups)}
    quota_groups = [g for g in groups if g.kind in ("mega", "party", "electorate")]

    total_voter_ballots = io_mod.total_normal_ballots_weight(base_groups)

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

    t_now = 0.0
    t_start = [0.0 for _ in groups]

    balances = [0.0 for _ in groups]

    def legacy_apply_time_step(dt: float) -> None:
        if dt <= 0:
            return
        for i in range(len(balances)):
            balances[i] += dt * weights[i]

    try:
        for r in range(1, min(seats, len(candidates)) + 1):
            if stop_when_proj_gt is not None and p_total > stop_when_proj_gt + 1e-15:
                break

            remaining = [c for c in candidates if c not in winners_set and c not in ban]
            if not remaining:
                break

            current_round = r
            quota_info = compute_quota_active_info(quota_groups, winners, current_round)

            # Candidate-level quota dissatisfaction discrepancy maps used for tie-breaking.
            party_discrepancy = _build_candidate_discrepancy_map(party_groups, quota_info)
            mega_discrepancy = _build_candidate_discrepancy_map(mega_groups, quota_info)

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

            weights_for_round = weights
            if profile.key == "324" and norm_ctx is not None and mega_indices:
                n_norm = float(norm_ctx.get("n", 0.0))
                if n_norm > 0:
                    z = sum(weights[i] for i in mega_indices if active_mask[i])
                    if z > (n_norm / 3.0) + 1e-18 and z > 0:
                        mega_mult = n_norm / (3.0 * z)
                        eff = list(weights)
                        for i in mega_indices:
                            if active_mask[i]:
                                eff[i] = weights[i] * mega_mult
                        weights_for_round = eff

            chosen, dt, have, rate, allow_used = choose_candidate_for_round(
                remaining=remaining,
                balances=None if spend_mode == "fifo_time_priority" else balances,
                t_now=t_now if spend_mode == "fifo_time_priority" else None,
                t_start=t_start if spend_mode == "fifo_time_priority" else None,
                weights=weights_for_round,
                supporters=supporters,
                active_mask=active_mask,
                party_lists=party_lists,
                mega_lists=mega_lists or {},
                party_discrepancy=party_discrepancy,
                mega_discrepancy=mega_discrepancy,
                allow_only_pool=pool_to_use,
                dt0_tie_rule=dt0_tie_rule,
                spend_mode=spend_mode,
            )

            if spend_mode == "fifo_time_priority":
                t_now += dt
            else:
                legacy_apply_time_step(dt)
                t_now += dt

            newly_used_ballots = compute_projection_delta_for_chosen(
                chosen, groups, supporters, active_mask, base_used
            )
            delta_p = (newly_used_ballots / total_voter_ballots) if total_voter_ballots > 0 else 0.0
            p_prev = p_total
            p_total += delta_p
            used_voter_ballots_cum += newly_used_ballots

            if spend_mode == "fifo_time_priority":
                spend_winner_fifo_time_priority_tiers(
                    cand=chosen,
                    groups=groups,
                    t_now=t_now,
                    t_start=t_start,
                    weights=weights_for_round,
                    supporters=supporters,
                    active_mask=active_mask,
                    spend_tiers=spend_tiers,
                    tier_within_mode=tier_within_mode,
                )
            else:
                spend_winner_reset(
                    cand=chosen,
                    balances=balances,
                    supporters=supporters,
                    active_mask=active_mask,
                )

            winners.append(chosen)
            winners_set.add(chosen)

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
                    "time": f"{t_now:.12f}",
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
                    "active_quota_groups": int(active_quota_groups),
                    "prefix_allow_pool_size_before": int(prefix_size_before),
                    "iter_allow_pool_size_before": int(iter_size_before),
                    "allow_pool_source": pool_source,
                    "allow_only_used": 1 if allow_used else 0,
                    "spend_mode": spend_mode,
                    "dt0_tie_rule": dt0_tie_rule,
                    "spend_tiers": spend_tiers_spec,
                    "tier_within_mode": tier_within_mode,
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
                    bal_after = fifo_balance_at(t_now, t_start, weights, gi) if spend_mode == "fifo_time_priority" else balances[gi]
                    quota_w.writerow({
                        "label": label,
                        "round": r,
                        "chosen": chosen,
                        "quota_gid": g.gid,
                        "quota_kind": g.kind,
                        "quota_floor": float(g.quota_floor or 0.0),
                        "required_by_round": int(req),
                        "winners_in_set_before": int(in_set),
                        "active_for_race": bool(active),
                        "reserve_balance_after": float(bal_after),
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
        "final_time": t_now,
        "final_projection": p_total,
        "stopped_at_round": len(winners),
    }


def run_full_chamber_completion(
    outdir: str,
    candidates_list: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    electorate_groups: List[Group],
    party_lists: Dict[str, List[str]],
    mega_lists: Dict[str, List[str]],
    prefix_allow: List[str],
    ban: Set[str],
    completion_target: float,
    spend_mode: str,
    dt0_tie_rule: str,
    spend_tiers_spec: str,
    tier_within_mode: str,
) -> dict:
    label = "converged_full"
    rounds_csv = os.path.join(outdir, f"{label}_rounds.csv")
    quota_csv = os.path.join(outdir, f"{label}_quota.csv")
    proj_csv = os.path.join(outdir, f"{label}_projection.csv")

    _ = run_sequential(
        label=label,
        candidates=candidates_list,
        seats=len(candidates_list),
        base_groups=base_groups,
        mega_groups=mega_groups,
        party_groups=party_groups,
        electorate_groups=electorate_groups,
        party_lists=party_lists,
        mega_lists=mega_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=rounds_csv,
        quota_csv_path=quota_csv,
        proj_csv_path=proj_csv,
        spend_mode=spend_mode,
        dt0_tie_rule=dt0_tie_rule,
        spend_tiers_spec=spend_tiers_spec,
        tier_within_mode=tier_within_mode,
    )

    winners, intervals = recompute_intervals_from_rounds_csv(rounds_csv)

    R2 = None
    for i, (_p0, p1) in enumerate(intervals, start=1):
        if i >= seats and p1 > completion_target + 1e-15:
            R2 = i
            break

    if R2 is None:
        R2 = len(winners)
    full_size = min(len(candidates_list), max(seats, R2))

    return {
        "label": label,
        "full_chamber_rounds": full_size,
        "full_chamber_winners": winners[:full_size],
        "full_chamber_projection": intervals[full_size - 1][1] if full_size >= 1 and full_size <= len(intervals) else (intervals[-1][1] if intervals else 0.0),
        "rounds_csv": rounds_csv,
        "quota_csv": quota_csv,
        "projection_csv": proj_csv,
        "note": "full chamber size = max(input seats, first round with projection > 2/3 strict), capped by candidate count",
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Sequential Phragmén FIFO + quota reserve racers + multi-pass convergence.")
    ap.add_argument("input_json", help="Election JSON file.")
    ap.add_argument("--outdir", default="out", help="Directory for CSV outputs.")
    ap.add_argument("--quota_meta_csv", default=None, help="Write normalized quota-group meta CSV (path).")
    ap.add_argument("--max_iters", type=int, default=19, help="Max iterations INCLUDING the first pass.")
    ap.add_argument("--no_prompt", action="store_true", help="Disable interactive prompts.")
    ap.add_argument("--profile", default=None, help="Profile key (default prompts, else general_alpha).")

    ap.add_argument("--spend_mode", choices=["reset", "fifo_time_priority"], default=None, help="Override spend mode.")
    ap.add_argument("--dt0_tie_rule", choices=["party_then_name", "max_have_then_party_then_name"], default=None, help="Override dt=0 tie rule.")
    ap.add_argument("--spend_tiers", default=None, help='Override tier plan, e.g. "base>party>electorate,mega".')
    ap.add_argument("--tier_within_mode", choices=["combined_fifo", "separate_by_kind"], default=None, help="Override within-tier mode.")

    args = ap.parse_args(argv)

    if args.profile is None:
        profile = get_profile("general_alpha") if args.no_prompt else choose_profile_interactive()
    else:
        profile = get_profile(args.profile)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    spend_mode = args.spend_mode or data.get("spend_mode") or profile.spend_mode
    dt0_tie_rule = args.dt0_tie_rule or data.get("dt0_tie_rule") or profile.dt0_tie_rule
    spend_tiers_spec = args.spend_tiers or data.get("spend_tiers") or profile.spend_tiers_default
    tier_within_mode = args.tier_within_mode or data.get("tier_within_mode") or profile.tier_within_mode

    _ = parse_spend_tiers(spend_tiers_spec)

    seats = int(data["seats"])

    # Global totals (wglobal1..4) for uniform quota share calculations.
    globals_ctx = io_mod.compute_global_totals(data)
    wglobal1 = globals_ctx["wglobal1"]
    wglobal2 = globals_ctx["wglobal2"]
    wglobal3 = globals_ctx["wglobal3"]
    wglobal4 = globals_ctx["wglobal4"]
    wglobal5 = globals_ctx["wglobal5"]

    base_electorate_info = io_mod.compute_base_electorate_info(data.get("ballots", []))
    partyrock_electorate_info = io_mod.compute_partyrock_electorate_info(data.get("partyrock_ballots", []))

    # MegaRock sums (parsing-only; used to compute mega share2 = max(w1, sum(wmr))).
    megarock_abs_by_mega = io_mod.compute_megarock_abs_sums(data.get("megarock_ballots", []) or [])

    base_groups = io_mod.canonicalize_base_ballots(data.get("ballots", []), profile=profile)

    # Profile 12/324: constant normalisation multipliers computed from input totals.
    norm_ctx = None  # type: Optional[dict]
    if profile.key in ("12", "324"):
        x, y, n_norm = io_mod.compute_profile_12_324_totals(data)
        voter_mult, party_mult = io_mod.compute_profile_12_324_multipliers(profile.key, x=x, y=y, n=n_norm)
        base_groups = io_mod.apply_constant_multiplier(base_groups, voter_mult, label="voter")
        norm_ctx = {"x": x, "y": y, "n": n_norm, "voter_mult": voter_mult, "party_mult": party_mult}

    total_voter_ballots = io_mod.total_normal_ballots_weight(base_groups)

    candidate_meta = io_mod.parse_candidate_meta(data.get("candidate_meta") or {})

    candidates: Set[str] = set(str(c) for c in data.get("candidates", []))
    candidates |= io_mod.candidates_from_groups(base_groups)
    candidates |= io_mod.extract_candidates_from_defs(data.get("party_ballots", []))
    candidates |= io_mod.extract_candidates_from_defs(data.get("electorate_ballots", []))
    # PartyRock ballots: parse approvals for candidate discovery (not yet used by profiles).
    for _pr in data.get("partyrock_ballots", []) or []:
        if isinstance(_pr, dict):
            for _c in (_pr.get("approvals") or []):
                if str(_c).strip() != "":
                    candidates.add(str(_c))
    # MegaRock ballots: parse approvals for candidate discovery (not yet used by profiles).
    for _mr in data.get("megarock_ballots", []) or []:
        if isinstance(_mr, dict):
            for _c in (_mr.get("approvals") or []):
                if str(_c).strip() != "":
                    candidates.add(str(_c))
    candidates |= io_mod.extract_candidates_from_defs(data.get("mega_ballots", []))

    prefix_allow, ban = io_mod.parse_prefix_intervention(data)

    mega_groups, mega_meta = io_mod.parse_mega_ballots(
        data.get("mega_ballots", []),
        wglobal1=wglobal1,
        wglobal4=wglobal4,
        megarock_abs_by_mega=megarock_abs_by_mega,
        candidate_set=candidates,
        candidate_meta=candidate_meta,
        profile=profile,
    )


    partyrock_abs_by_party = io_mod.compute_partyrock_party_abs_sums(data.get("partyrock_ballots", []) or [])

    # Parse electorates first (PartyRock and MegaRock depend on electorates).
    electorate_groups, electorate_meta = io_mod.parse_electorate_ballots(
        data.get("electorate_ballots", []),
        wglobal2=wglobal2,
        wglobal4=wglobal4,
        wglobal5=wglobal5,
        candidate_set=candidates,
        profile=profile,
        base_electorate_info=base_electorate_info,
        partyrock_electorate_info=partyrock_electorate_info,
    )

    # Parse PartyRock ballots (parsing-only; not yet included in the engine group set).
    partyrock_groups, partyrock_meta = io_mod.parse_partyrock_ballots(
        data.get("partyrock_ballots", []),
        electorate_groups=electorate_groups,
        candidate_set=candidates,
    )

    # Parse MegaRock ballots (parsing-only; not yet included in the engine group set).
    megarock_groups, megarock_meta = io_mod.parse_megarock_ballots(
        data.get("megarock_ballots", []) or [],
        electorate_groups=electorate_groups,
        wglobal1=wglobal1,
        wglobal5=wglobal5,
        candidate_set=candidates,
    )

    # PartyRock -> Party derived attribute: wp3 = sum PartyRock electorate-normalised weights (wpr2) for the party.
    partyrock_norm_sum_by_party = io_mod.compute_partyrock_party_sums(partyrock_groups)

    # Parse parties (after PartyRock so that share2 can use wp3 / wglobal5).
    party_groups, party_lists, party_meta, party_cands = io_mod.parse_party_ballots(
        data.get("party_ballots", []),
        wglobal4=wglobal4,
        wglobal5=wglobal5,
        partyrock_abs_by_party=partyrock_abs_by_party,
        partyrock_norm_sum_by_party=partyrock_norm_sum_by_party,
        profile=profile,
    )
    candidates |= party_cands

    if norm_ctx is not None and profile.key in ("12", "324"):
        party_groups = io_mod.apply_constant_multiplier(party_groups, norm_ctx["party_mult"], label="party")
        party_meta = io_mod.apply_constant_multiplier(party_meta, norm_ctx["party_mult"], label="party")
    # PartyRock -> Party derived attributes (parsing only; does not change allocation semantics).
    # - Auto-create party ballots referenced by PartyRock ballots when missing.
    # - Attach a calculated attribute to party ballots: sum of PartyRock electorate-normalized weights.
    partyrock_party_sums = io_mod.compute_partyrock_party_sums(partyrock_groups)

    # Index existing parties.
    _party_by_gid = {g.gid: g for g in party_groups}
    for pid, sum_norm in sorted(partyrock_party_sums.items(), key=lambda kv: kv[0]):
        if pid in _party_by_gid:
            # Update meta for existing party groups (and meta duplicates).
            for _lst in (party_groups, party_meta):
                for idx, g in enumerate(list(_lst)):
                    if g.gid != pid:
                        continue
                    _lst[idx] = io_mod._clone_group_with_weight(
                        g,
                        g.weight,
                        meta_update={"partyrock_sum_norm_weight": float(sum_norm)},
                    )
        else:
            # Auto-create a placeholder party ballot with weight 0 and no approvals.
            auto = Group(
                gid=str(pid),
                kind="party",
                approvals=tuple(),
                weight=0.0,
                quota_floor=None,
                population=None,
                abs_weight=None,
                share=None,
                meta={"source": "party_ballots_auto", "partyrock_sum_norm_weight": float(sum_norm)},
            )
            party_groups.append(auto)
            party_meta.append(auto)
            if str(pid) not in party_lists:
                party_lists[str(pid)] = []



    # Party list extension from PartyRock ballots (parsing-only, used for tie-breaking order maps).
    # For each party:
    #   - extra candidates = candidates appearing in PartyRock approvals for that party but not already in the party list
    #   - run a mini FIFO sequential Phragmén election (general_alpha-style) over the extra candidates
    #     using:
    #       * base ballots: raw base ballots whose approvals include any extra candidate (restricted to the extra set)
    #       * party ballots (quota): PartyRock ballots for the party (restricted to the extra set)
    #     Mini population = max(sum(abs base weights), sum(abs PartyRock weights))
    #
    # The resulting winner order is appended to the end of each existing party list. Any remaining missing extras
    # (should be none) are appended deterministically.
    partyrock_by_party = {}
    for g in partyrock_groups:
        pid = str((g.meta or {}).get("party") or "party_unknown")
        partyrock_by_party.setdefault(pid, []).append(g)

    # Raw base ballots are used (pre-profile scaling) for the mini election.
    raw_base_ballots = data.get("ballots", []) or []

    for pid, pr_groups in sorted(partyrock_by_party.items(), key=lambda kv: kv[0]):
        existing_list = list(party_lists.get(pid, []))
        existing_set = set(str(x) for x in existing_list)

        # Candidate universe for this extension (PartyRock candidates not already in party list).
        pr_cand_set = set()
        for g in pr_groups:
            pr_cand_set.update(g.approvals)
        extra = [c for c in sorted(pr_cand_set) if c not in existing_set]
        if not extra:
            continue

        extra_set = set(extra)

        # Build mini base groups: raw ballots restricted to extra candidates.
        base_agg = {}
        base_sum = 0.0
        for b in raw_base_ballots:
            if not isinstance(b, dict):
                continue
            apps = [str(x) for x in (b.get("approvals") or [])]
            apps = [a for a in apps if a in extra_set]
            if not apps:
                continue
            try:
                w = float(b.get("weight", 1.0))
            except Exception:
                w = 0.0
            if w <= 0:
                continue
            key = tuple(sorted(set(apps)))
            base_agg[key] = base_agg.get(key, 0.0) + w
            base_sum += w

        mini_base_groups = [
            Group(
                gid=f"mini_base_{pid}_{i}",
                kind="base",
                approvals=apps,
                weight=float(w),
                abs_weight=float(w),
                meta={"source": "mini_party_list_extension", "party": pid},
            )
            for i, (apps, w) in enumerate(base_agg.items())
        ]

        # Build mini quota groups from PartyRock ballots restricted to extra candidates.
        pr_sum_abs = 0.0
        mini_quota_groups = []
        for i, g in enumerate(pr_groups):
            apps = tuple(sorted(set([c for c in g.approvals if c in extra_set])))
            if not apps:
                continue
            abs_w = float(g.abs_weight or 0.0)
            if abs_w <= 0:
                continue
            pr_sum_abs += abs_w
            mini_quota_groups.append((i, apps, abs_w))

        mini_pop = float(max(base_sum, pr_sum_abs, 0.0))
        if mini_pop <= 0:
            ordered_extra = list(extra)
        else:
            quota_groups = []
            for i, apps, abs_w in mini_quota_groups:
                # Mini-election quota mapping:
                #   share = wr / N
                #   quota_floor = min((2/3)*share, 1/3)
                #   rel_weight = N * quota_floor
                share = (abs_w / mini_pop) if mini_pop > 0 else 0.0
                qf = io_mod.quota_floor_from_share(share)
                rel_w = io_mod.normalize_rel_weight_from_share(share, mini_pop)
                quota_groups.append(
                    Group(
                        gid=f"mini_pr_{pid}_{i}",
                        kind="party",
                        approvals=apps,
                        weight=float(rel_w),
                        quota_floor=float(qf),
                        population=float(mini_pop),
                        abs_weight=float(abs_w),
                        share=float(share),
                        meta={"source": "mini_party_list_extension", "party": pid, "mini_pop": mini_pop},
                    )
                )

            ordered_extra = run_fifo_ordering_general_alpha(
                candidates=extra,
                base_groups=mini_base_groups,
                quota_groups=quota_groups,
                party_lists={},
                dt0_tie_rule="party_then_name",
                spend_tiers_str="base>mega",
                tier_within_mode="combined_fifo",
            )

            missing = [c for c in extra if c not in set(ordered_extra)]
            if missing:
                ordered_extra.extend(sorted(missing))

        # Append to party list, preserving original order.
        party_lists[pid] = existing_list + ordered_extra

        # Expand party quota approvals set to include appended candidates.
        extra_tuple = tuple(sorted(set(ordered_extra)))
        if extra_tuple:
            for _lst in (party_groups, party_meta):
                for idx, g in enumerate(list(_lst)):
                    if g.gid != pid:
                        continue
                    new_apps = tuple(sorted(set(g.approvals) | set(extra_tuple)))
                    _lst[idx] = io_mod._clone_group_with_weight(
                        Group(
                            gid=g.gid,
                            kind=g.kind,
                            approvals=new_apps,
                            weight=g.weight,
                            quota_floor=g.quota_floor,
                            population=g.population,
                            abs_weight=g.abs_weight,
                            share=g.share,
                            meta=g.meta,
                        ),
                        g.weight,
                        meta_update={"party_list_extended": True, "party_list_extension_size": len(ordered_extra)},
                    )

    # -------------------------------------------------------------------------
    # Candidate meta -> Mega/MegaRock binding + ordering (parsing-time only).
    #
    # Rules:
    # 1) candidate_meta mega binding only EXTENDS approvals, and only for candidates already
    #    in the candidate universe.
    # 2) MegaRock binding: if a candidate has mega binding AND appears in an electorate's
    #    candidate universe, it is auto-approved onto any MegaRock ballot (E, M) for that
    #    electorate and mega.
    # 3) MegaRock approvals EXTEND Mega approvals for the associated mega.
    #
    # Ordering:
    # - MegaRock: run a mini general_alpha election per MegaRock ballot to produce an
    #   ordered candidate list (stored in meta['ordered_candidates']).
    # - Mega: run a mini general_alpha election per Mega to produce an ordered candidate
    #   list (stored in mega_lists[mega_id] and in meta['ordered_candidates']).
    #
    # These ordered lists are used for tie-breaking later (but do not yet change allocation).
    # -------------------------------------------------------------------------

    # Initial mega_lists (from JSON input order), used as a starting point.
    mega_lists: Dict[str, List[str]] = {}
    for i, d in enumerate(data.get("mega_ballots", []) or []):
        mid = str(d.get("id") or f"mega_{i}")
        raw = [str(x) for x in (d.get("candidates") or []) if str(x).strip() != ""]
        # Preserve order but filter to candidate universe
        seen = set()
        ordered = []
        for c in raw:
            if c in seen:
                continue
            seen.add(c)
            if c in candidates:
                ordered.append(c)
        mega_lists[mid] = ordered

    # candidate -> megas binding (ignore meta-only candidates)
    cand_to_megas: Dict[str, Set[str]] = {}
    for c, m in (candidate_meta or {}).items():
        if c not in candidates:
            continue
        megas = m.get("megas")
        if megas is None:
            continue
        if isinstance(megas, (str, int, float)):
            megas = [str(megas)]
        if not isinstance(megas, list):
            continue
        s = set(str(x) for x in megas if str(x).strip() != "")
        if s:
            cand_to_megas[c] = s

    # Index electorates and megas
    elect_by_gid: Dict[str, Group] = {g.gid: g for g in electorate_groups}
    mega_by_gid: Dict[str, Group] = {g.gid: g for g in mega_groups}

    # 2) MegaRock binding based on electorate candidate universe + candidate_meta.megas
    new_megarock_groups: List[Group] = []
    new_megarock_meta: List[Group] = []
    for g in megarock_groups:
        m = (g.meta or {})
        eid = str(m.get("electorate") or "electorate_unknown")
        mid = str(m.get("mega") or "mega_unknown")
        e = elect_by_gid.get(eid)
        e_cands = set(e.approvals) if e is not None else set()
        add = sorted([c for c in e_cands if mid in cand_to_megas.get(c, set())])
        cur_set = set(g.approvals)
        added = [c for c in add if c not in cur_set]
        new_apps = tuple(sorted(cur_set | set(added)))
        ordered = list((m.get("ordered_candidates") or []))
        for c in added:
            if c not in ordered:
                ordered.append(c)
        meta2 = dict(m)
        if added:
            meta2["auto_approved_from_candidate_meta"] = {"mega": mid, "added": list(added)}
        meta2["ordered_candidates"] = ordered
        g2 = Group(
            gid=g.gid,
            kind=g.kind,
            approvals=new_apps,
            weight=g.weight,
            quota_floor=g.quota_floor,
            population=g.population,
            abs_weight=g.abs_weight,
            share=g.share,
            meta=meta2,
        )
        new_megarock_groups.append(g2)
        new_megarock_meta.append(g2)
    megarock_groups = new_megarock_groups
    megarock_meta = new_megarock_meta

    # 3) MegaRock approvals extend Mega approvals (and will also include meta-bound candidates from above).
    megarock_by_mega: Dict[str, Set[str]] = {}
    for g in megarock_groups:
        mid = str((g.meta or {}).get("mega") or "mega_unknown")
        megarock_by_mega.setdefault(mid, set()).update(g.approvals)

    # 1) candidate_meta mega binding extends Mega approvals (universe-only).
    bound_by_mega: Dict[str, Set[str]] = {}
    for c, mids in cand_to_megas.items():
        for mid in mids:
            bound_by_mega.setdefault(mid, set()).add(c)

    new_mega_groups: List[Group] = []
    new_mega_meta: List[Group] = []
    for g in mega_groups:
        mid = g.gid
        cur = set(g.approvals)
        add1 = set(bound_by_mega.get(mid, set()))
        add2 = set(megarock_by_mega.get(mid, set()))
        added = sorted([c for c in (add1 | add2) if c in candidates and c not in cur])
        new_apps = tuple(sorted(cur | set(added)))

        # Extend mega_lists (preserve existing order; append missing deterministically for now)
        lst = list(mega_lists.get(mid, []))
        for c in added:
            if c not in lst:
                lst.append(c)
        # Ensure list covers approvals (append any remaining missing)
        for c in sorted(new_apps):
            if c not in lst:
                lst.append(c)
        mega_lists[mid] = lst

        meta2 = dict(g.meta or {})
        if added:
            meta2["approvals_extended"] = {
                "from_candidate_meta": sorted(set(bound_by_mega.get(mid, set())) & set(added)),
                "from_megarock": sorted(set(megarock_by_mega.get(mid, set())) & set(added)),
            }
        g2 = Group(
            gid=g.gid,
            kind=g.kind,
            approvals=new_apps,
            weight=g.weight,
            quota_floor=g.quota_floor,
            population=g.population,
            abs_weight=g.abs_weight,
            share=g.share,
            meta=meta2,
        )
        new_mega_groups.append(g2)
        new_mega_meta.append(g2)
    mega_groups = new_mega_groups
    mega_meta = new_mega_meta

    # --- MegaRock mini ordering ---
    partyrock_by_eid: Dict[str, List[Group]] = {}
    for g in partyrock_groups:
        eid = str((g.meta or {}).get("electorate") or "electorate_unknown")
        partyrock_by_eid.setdefault(eid, []).append(g)

    updated_megarock_groups: List[Group] = []
    updated_megarock_meta: List[Group] = []
    for g in megarock_groups:
        m = g.meta or {}
        eid = str(m.get("electorate") or "electorate_unknown")
        candset = set(g.approvals)
        if not candset:
            updated_megarock_groups.append(g)
            updated_megarock_meta.append(g)
            continue

        # Base ballots: only those registered to this electorate, restricted to candset.
        base_agg: Dict[Tuple[str, ...], float] = {}
        base_sum = 0.0
        for b in raw_base_ballots:
            if not isinstance(b, dict):
                continue
            beid = b.get("electorate")
            if beid is None:
                beid = b.get("electorate_id")
            if beid is None:
                beid = b.get("electorate_gid")
            beid = str(beid) if beid is not None and str(beid).strip() != "" else "electorate_unknown"
            if beid != eid:
                continue
            apps = [str(x) for x in (b.get("approvals") or [])]
            apps = [a for a in apps if a in candset]
            if not apps:
                continue
            try:
                w = float(b.get("weight", 1.0))
            except Exception:
                w = 0.0
            if w <= 0:
                continue
            key = tuple(sorted(set(apps)))
            base_agg[key] = base_agg.get(key, 0.0) + w
            base_sum += w

        mini_base_groups = [
            Group(
                gid=f"mini_mr_base_{g.gid}_{i}",
                kind="base",
                approvals=apps,
                weight=float(w),
                abs_weight=float(w),
                meta={"source": "mini_megarock_order", "megarock": g.gid, "electorate": eid},
            )
            for i, (apps, w) in enumerate(base_agg.items())
        ]

        # Quota ballots: PartyRock in the same electorate, restricted to candset.
        pr_sum_abs = 0.0
        pr_items = []
        for i, pr in enumerate(partyrock_by_eid.get(eid, [])):
            apps = tuple(sorted(set([c for c in pr.approvals if c in candset])))
            if not apps:
                continue
            abs_w = float(pr.abs_weight or 0.0)
            if abs_w <= 0:
                continue
            pr_sum_abs += abs_w
            pr_items.append((i, apps, abs_w))

        mini_pop = float(max(base_sum, pr_sum_abs, 0.0))
        if mini_pop <= 0:
            ordered = list(sorted(candset))
        else:
            quota_groups = []
            for i, apps, abs_w in pr_items:
                share = (abs_w / mini_pop) if mini_pop > 0 else 0.0
                qf = io_mod.quota_floor_from_share(share)
                rel_w = io_mod.normalize_rel_weight_from_share(share, mini_pop)
                quota_groups.append(
                    Group(
                        gid=f"mini_mr_pr_{g.gid}_{i}",
                        kind="party",
                        approvals=apps,
                        weight=float(rel_w),
                        quota_floor=float(qf),
                        population=float(mini_pop),
                        abs_weight=float(abs_w),
                        share=float(share),
                        meta={"source": "mini_megarock_order", "megarock": g.gid, "electorate": eid, "mini_pop": mini_pop},
                    )
                )
            ordered = run_fifo_ordering_general_alpha(
                candidates=sorted(candset),
                base_groups=mini_base_groups,
                quota_groups=quota_groups,
                party_lists={},
                dt0_tie_rule="party_then_name",
                spend_tiers_str="base>party",
                tier_within_mode="combined_fifo",
            )
            missing = [c for c in sorted(candset) if c not in set(ordered)]
            if missing:
                ordered.extend(sorted(missing))

        meta2 = dict(m)
        meta2["ordered_candidates"] = list(ordered)
        meta2["mini_pop"] = float(mini_pop)
        g2 = Group(
            gid=g.gid,
            kind=g.kind,
            approvals=g.approvals,
            weight=g.weight,
            quota_floor=g.quota_floor,
            population=g.population,
            abs_weight=g.abs_weight,
            share=g.share,
            meta=meta2,
        )
        updated_megarock_groups.append(g2)
        updated_megarock_meta.append(g2)

    megarock_groups = updated_megarock_groups
    megarock_meta = updated_megarock_meta

    # --- Mega mini ordering ---
    updated_mega_groups: List[Group] = []
    updated_mega_meta: List[Group] = []
    for g in mega_groups:
        mid = g.gid
        candset = set(g.approvals)
        if not candset:
            updated_mega_groups.append(g)
            updated_mega_meta.append(g)
            continue

        # Base ballots: any base ballot that approves any mega candidate (restricted to candset).
        base_agg: Dict[Tuple[str, ...], float] = {}
        base_sum = 0.0
        for b in raw_base_ballots:
            if not isinstance(b, dict):
                continue
            apps = [str(x) for x in (b.get("approvals") or [])]
            apps = [a for a in apps if a in candset]
            if not apps:
                continue
            try:
                w = float(b.get("weight", 1.0))
            except Exception:
                w = 0.0
            if w <= 0:
                continue
            key = tuple(sorted(set(apps)))
            base_agg[key] = base_agg.get(key, 0.0) + w
            base_sum += w

        mini_base_groups = [
            Group(
                gid=f"mini_m_base_{mid}_{i}",
                kind="base",
                approvals=apps,
                weight=float(w),
                abs_weight=float(w),
                meta={"source": "mini_mega_order", "mega": mid},
            )
            for i, (apps, w) in enumerate(base_agg.items())
        ]

        # Quota ballots: MegaRock ballots linked to this mega that approve any mega candidate (restricted to candset).
        mr_sum_abs = 0.0
        mr_items = []
        for i, mr in enumerate(megarock_groups):
            if not isinstance(mr.meta, dict):
                continue
            if str(mr.meta.get("mega")) != str(mid):
                continue
            apps = tuple(sorted(set([c for c in mr.approvals if c in candset])))
            if not apps:
                continue
            abs_w = float(mr.abs_weight or 0.0)
            if abs_w <= 0:
                continue
            mr_sum_abs += abs_w
            mr_items.append((i, apps, abs_w, mr.gid))

        mini_pop = float(max(base_sum, mr_sum_abs, 0.0))
        if mini_pop <= 0:
            ordered = list(sorted(candset))
        else:
            quota_groups = []
            for i, apps, abs_w, gid in mr_items:
                share = (abs_w / mini_pop)
                qf = min((2.0 / 3.0) * share, 1.0 / 3.0)
                rel_w = mini_pop * qf
                quota_groups.append(
                    Group(
                        gid=f"mini_m_mrquota_{mid}_{i}",
                        kind="mega",
                        approvals=apps,
                        weight=float(rel_w),
                        abs_weight=float(abs_w),
                        quota_floor=float(qf),
                        meta={
                            "source": "mini_mega_order_quota_megarock",
                            "mega": mid,
                            "megarock_gid": gid,
                            "share": float(share),
                            "quota_floor": float(qf),
                            "N_arena": float(mini_pop),
                        },
                    )
                )

            ordered = run_fifo_ordering_general_alpha(
                candidates=list(sorted(candset)),
                base_groups=mini_base_groups,
                quota_groups=quota_groups,
                party_lists=None,
                dt0_tie_rule="party_then_name",
                spend_tiers_str="base>party",
                tier_within_mode="combined_fifo",
            )
        if mini_pop <= 0:
            ordered = list(sorted(candset))
        else:
            quota_groups = []
            for i, apps, abs_w, pid in party_items:
                share = (abs_w / mini_pop) if mini_pop > 0 else 0.0
                qf = io_mod.quota_floor_from_share(share)
                rel_w = io_mod.normalize_rel_weight_from_share(share, mini_pop)
                quota_groups.append(
                    Group(
                        gid=f"mini_m_party_{mid}_{i}",
                        kind="party",
                        approvals=apps,
                        weight=float(rel_w),
                        quota_floor=float(qf),
                        population=float(mini_pop),
                        abs_weight=float(abs_w),
                        share=float(share),
                        meta={"source": "mini_mega_order", "mega": mid, "party": pid, "mini_pop": mini_pop},
                    )
                )
            ordered = run_fifo_ordering_general_alpha(
                candidates=sorted(candset),
                base_groups=mini_base_groups,
                quota_groups=quota_groups,
                party_lists={},
                dt0_tie_rule="party_then_name",
                spend_tiers_str="base>party",
                tier_within_mode="combined_fifo",
            )
            missing = [c for c in sorted(candset) if c not in set(ordered)]
            if missing:
                ordered.extend(sorted(missing))

        mega_lists[mid] = list(ordered)
        meta2 = dict(g.meta or {})
        meta2["ordered_candidates"] = list(ordered)
        meta2["mini_pop"] = float(mini_pop)
        g2 = Group(
            gid=g.gid,
            kind=g.kind,
            approvals=g.approvals,
            weight=g.weight,
            quota_floor=g.quota_floor,
            population=g.population,
            abs_weight=g.abs_weight,
            share=g.share,
            meta=meta2,
        )
        updated_mega_groups.append(g2)
        updated_mega_meta.append(g2)

    mega_groups = updated_mega_groups
    mega_meta = updated_mega_meta

    candidates_list = sorted(candidates)
    os.makedirs(args.outdir, exist_ok=True)

    # Audit outputs for all calculated weights/shares.
    io_mod.write_globals_kv_csv(os.path.join(args.outdir, 'audit_globals.csv'), globals_ctx, norm_ctx=norm_ctx)
    io_mod.write_groups_audit_csv(os.path.join(args.outdir, 'audit_groups.csv'), base_groups + mega_meta + party_meta + electorate_meta + partyrock_meta + megarock_meta)
    io_mod.write_party_lists_csv(os.path.join(args.outdir, 'audit_party_lists.csv'), party_lists)
    io_mod.write_party_lists_csv(os.path.join(args.outdir, 'audit_mega_lists.csv'), mega_lists)


    if args.quota_meta_csv:
        io_mod.write_meta_csv(args.quota_meta_csv, mega_meta + party_meta + electorate_meta)

    # PASS 1
    pass1_label = "pass01"
    pass1_rounds = os.path.join(args.outdir, f"{pass1_label}_rounds.csv")
    pass1_quota = os.path.join(args.outdir, f"{pass1_label}_quota.csv")
    pass1_proj = os.path.join(args.outdir, f"{pass1_label}_projection.csv")

    _ = run_sequential(
        label=pass1_label,
        candidates=candidates_list,
        seats=seats,
        base_groups=base_groups,
        mega_groups=mega_groups,
        party_groups=party_groups,
        electorate_groups=electorate_groups,
        party_lists=party_lists,
        mega_lists=mega_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=pass1_rounds,
        quota_csv_path=pass1_quota,
        proj_csv_path=pass1_proj,
        spend_mode=spend_mode,
        dt0_tie_rule=dt0_tie_rule,
        spend_tiers_spec=spend_tiers_spec,
        tier_within_mode=tier_within_mode,
    )

    pass1_winners, pass1_intervals = recompute_intervals_from_rounds_csv(pass1_rounds)
    pass1_sig = tuple(prefix_until_projection_strict_gt(pass1_winners, pass1_intervals, profile.sig_target))

    seen: Dict[Tuple[str, ...], int] = {pass1_sig: 1}
    prev_B_full_winners = pass1_winners
    prev_B_full_intervals = pass1_intervals

    solved = False
    solved_iter = None
    twin_iter = None
    cycle_len = None

    it = 2
    max_iters_current = args.max_iters

    while True:
        while it <= max_iters_current:
            labelA = f"iter{it:02d}A"
            labelB = f"iter{it:02d}B"

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
                electorate_groups=electorate_groups,
                party_lists=party_lists,
                mega_lists=mega_lists,
                stop_when_proj_gt=profile.sig_target,
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_A,
                rounds_csv_path=A_rounds,
                quota_csv_path=A_quota,
                proj_csv_path=A_proj,
                spend_mode=spend_mode,
                dt0_tie_rule=dt0_tie_rule,
                spend_tiers_spec=spend_tiers_spec,
                tier_within_mode=tier_within_mode,
            )

            A_winners, _A_intervals = recompute_intervals_from_rounds_csv(A_rounds)

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
                electorate_groups=electorate_groups,
                party_lists=party_lists,
                mega_lists=mega_lists,
                stop_when_proj_gt=profile.sig_target,
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_B,
                rounds_csv_path=B_rounds,
                quota_csv_path=B_quota,
                proj_csv_path=B_proj,
                spend_mode=spend_mode,
                dt0_tie_rule=dt0_tie_rule,
                spend_tiers_spec=spend_tiers_spec,
                tier_within_mode=tier_within_mode,
            )

            B_winners, B_intervals = recompute_intervals_from_rounds_csv(B_rounds)
            sig = tuple(prefix_until_projection_strict_gt(B_winners, B_intervals, profile.sig_target))

            if sig in seen:
                solved = True
                solved_iter = it
                twin_iter = seen[sig]
                cycle_len = it - twin_iter
                prev_B_full_winners = B_winners
                prev_B_full_intervals = B_intervals
                break

            seen[sig] = it
            prev_B_full_winners = B_winners
            prev_B_full_intervals = B_intervals
            it += 1

        if solved:
            break

        if args.no_prompt:
            break

        more = prompt_more_iters(default=profile.prompt_block)
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
            electorate_groups=electorate_groups,
            party_lists=party_lists,
            mega_lists=mega_lists,
            prefix_allow=prefix_allow,
            ban=ban,
            completion_target=profile.completion_target,
            spend_mode=spend_mode,
            dt0_tie_rule=dt0_tie_rule,
            spend_tiers_spec=spend_tiers_spec,
            tier_within_mode=tier_within_mode,
        )

    print(json.dumps({
        "profile": profile.key,
        "spend_mode": spend_mode,
        "dt0_tie_rule": dt0_tie_rule,
        "spend_tiers": spend_tiers_spec,
        "tier_within_mode": tier_within_mode,
        "status": "repeat_signature_found" if solved else "not_converged",
        "repeat_at_iteration": solved_iter,
        "twin_iteration": twin_iter,
        "cycle_length": cycle_len,
        "signature_definition": "Part-B prefix winners until projection > 5/9 (strict)",
        "completion": completion,
        "iters_seen": len(seen),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

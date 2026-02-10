from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import argparse
import csv
import json
import os

from . import io as io_mod
from .engine import (
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

            chosen, dt, have, rate, allow_used = choose_candidate_for_round(
                remaining=remaining,
                balances=None if spend_mode == "fifo_time_priority" else balances,
                t_now=t_now if spend_mode == "fifo_time_priority" else None,
                t_start=t_start if spend_mode == "fifo_time_priority" else None,
                weights=weights,
                supporters=supporters,
                active_mask=active_mask,
                party_lists=party_lists,
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
                    weights=weights,
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

    base_groups = io_mod.canonicalize_base_ballots(data.get("ballots", []), profile=profile)
    total_voter_ballots = io_mod.total_normal_ballots_weight(base_groups)

    candidate_meta = io_mod.parse_candidate_meta(data.get("candidate_meta") or {})

    candidates: Set[str] = set(str(c) for c in data.get("candidates", []))
    candidates |= io_mod.candidates_from_groups(base_groups)
    candidates |= io_mod.extract_candidates_from_defs(data.get("party_ballots", []))
    candidates |= io_mod.extract_candidates_from_defs(data.get("electorate_ballots", []))
    candidates |= io_mod.extract_candidates_from_defs(data.get("mega_ballots", []))

    prefix_allow, ban = io_mod.parse_prefix_intervention(data)

    party_groups, party_lists, party_meta, party_cands = io_mod.parse_party_ballots(
        data.get("party_ballots", []),
        total_voter_ballots=total_voter_ballots,
        profile=profile,
    )
    candidates |= party_cands

    mega_groups, mega_meta = io_mod.parse_mega_ballots(
        data.get("mega_ballots", []),
        total_voter_ballots=total_voter_ballots,
        candidate_set=candidates,
        candidate_meta=candidate_meta,
        profile=profile,
    )

    electorate_groups, electorate_meta = io_mod.parse_electorate_ballots(
        data.get("electorate_ballots", []),
        total_voter_ballots=total_voter_ballots,
        candidate_set=candidates,
        profile=profile,
    )

    candidates_list = sorted(candidates)
    os.makedirs(args.outdir, exist_ok=True)

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

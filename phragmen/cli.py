from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import argparse
import csv
import json
import os

from . import io as io_mod
from .engine import (
    ROUND_FIELDS, QUOTA_FIELDS, PROJ_FIELDS,
    apply_time_step, build_supporters, choose_candidate_for_round,
    compute_projection_delta_for_chosen, compute_quota_active_info,
    spend_winner
)
from .profiles import get_profile, list_profiles
from .types import ElectionProfile, Group


# ---------------------------
# CSV writers
# ---------------------------

def open_csv(path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


# ---------------------------
# Profile selection
# ---------------------------

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


# ---------------------------
# Helpers for reading intervals from CSV
# ---------------------------

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


def select_pool_by_coverage(winners: List[str], intervals: List[Tuple[float, float]], a: float, b: float) -> List[str]:
    out: List[str] = []
    for w, (p0, p1) in zip(winners, intervals):
        if p0 < b - 1e-15 and p1 >= a - 1e-15:
            out.append(w)
    return out


def prefix_until_projection_strict_gt(winners: List[str], intervals: List[Tuple[float, float]], target: float) -> List[str]:
    out: List[str] = []
    for w, (p0, p1) in zip(winners, intervals):
        out.append(w)
        if p1 > target + 1e-15:
            break
    return out


# ---------------------------
# Core runner (single pass)
# ---------------------------

def run_sequential(
    label: str,
    candidates: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    party_lists: Dict[str, List[str]],
    stop_when_proj_gt: Optional[float],            # stop when p_total > threshold (strict)
    prefix_allow_only_init: Optional[List[str]],   # higher priority pool
    ban_set: Optional[Set[str]],                   # always banned
    iter_allow_only_init: Optional[List[str]],     # lower priority pool
    rounds_csv_path: Optional[str],
    quota_csv_path: Optional[str],
    proj_csv_path: Optional[str],
) -> dict:
    groups = base_groups + mega_groups + party_groups
    weights = [g.weight for g in groups]
    balances = [0.0 for _ in groups]

    supporters = build_supporters(groups, candidates)
    gid_to_index = {g.gid: i for i, g in enumerate(groups)}
    quota_groups = [g for g in groups if g.kind in ("mega", "party")]

    total_voter_ballots = io_mod.total_normal_ballots_weight(base_groups)
    proj_total_for_quota = io_mod.projection_total_for_quota_from_base(base_groups)

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
    time_now = 0.0

    try:
        for r in range(1, seats + 1):
            if stop_when_proj_gt is not None and p_total > stop_when_proj_gt + 1e-15:
                break

            remaining = [c for c in candidates if c not in winners_set and c not in ban]
            if not remaining:
                break

            r_eff = max(r, proj_total_for_quota)
            projection_delta_for_quota = r_eff - r

            quota_info = compute_quota_active_info(quota_groups, winners, r_eff)

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
                remaining,
                balances, weights, supporters, active_mask,
                party_lists,
                allow_only_pool=pool_to_use
            )

            apply_time_step(balances, weights, dt)
            time_now += dt

            newly_used_ballots = compute_projection_delta_for_chosen(
                chosen, groups, supporters, active_mask, base_used
            )
            delta_p = (newly_used_ballots / total_voter_ballots) if total_voter_ballots > 0 else 0.0

            p_prev = p_total
            p_total += delta_p
            used_voter_ballots_cum += newly_used_ballots

            spend_winner(chosen, balances, supporters, active_mask)
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
                    "time": f"{time_now:.12f}",
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
                    "r_eff": r_eff,
                    "projection_total_for_quota": proj_total_for_quota,
                    "projection_delta_for_quota": projection_delta_for_quota,
                    "active_quota_groups": active_quota_groups,
                    "prefix_allow_pool_size_before": prefix_size_before,
                    "iter_allow_pool_size_before": iter_size_before,
                    "allow_pool_source": pool_source,
                    "allow_only_used": 1 if allow_used else 0,
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
                    quota_w.writerow({
                        "label": label,
                        "round": r,
                        "chosen": chosen,
                        "quota_gid": g.gid,
                        "quota_kind": g.kind,
                        "quota_floor": float(g.quota_floor or 0.0),
                        "required_by_r_eff": int(req),
                        "winners_in_set_before": int(in_set),
                        "active_for_race": bool(active),
                        "reserve_balance_after": float(balances[gi]),
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
        "final_time": time_now,
        "final_projection": p_total,
        "stopped_at_round": len(winners),
        "total_voter_ballots": total_voter_ballots,
        "projection_total_for_quota": proj_total_for_quota,
    }


def run_full_chamber_completion(
    outdir: str,
    candidates_list: List[str],
    seats: int,
    base_groups: List[Group],
    mega_groups: List[Group],
    party_groups: List[Group],
    party_lists: Dict[str, List[str]],
    prefix_allow: List[str],
    ban: Set[str],
    completion_target: float,
) -> dict:
    cap = max(seats, len(candidates_list))
    label = "converged_full"
    rounds_csv = os.path.join(outdir, f"{label}_rounds.csv")
    quota_csv = os.path.join(outdir, f"{label}_quota.csv")
    proj_csv = os.path.join(outdir, f"{label}_projection.csv")

    _res = run_sequential(
        label=label,
        candidates=candidates_list,
        seats=cap,
        base_groups=base_groups,
        mega_groups=mega_groups,
        party_groups=party_groups,
        party_lists=party_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=rounds_csv,
        quota_csv_path=quota_csv,
        proj_csv_path=proj_csv,
    )

    winners, intervals = recompute_intervals_from_rounds_csv(rounds_csv)

    R2 = None
    for i, (_p0, p1) in enumerate(intervals, start=1):
        if i >= seats and p1 > completion_target + 1e-15:
            R2 = i
            break

    if R2 is None:
        # If infeasible to meet completion_target or seats, cap at what exists.
        R2 = len(winners)

    return {
        "label": label,
        "cap_run": cap,
        "full_chamber_rounds": R2,
        "full_chamber_winners": winners[:R2],
        "full_chamber_projection": intervals[R2 - 1][1] if R2 >= 1 else 0.0,
        "rounds_csv": rounds_csv,
        "quota_csv": quota_csv,
        "projection_csv": proj_csv,
        "infeasible_full_chamber": (len(winners) < seats),
    }


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Sequential Phragmén (library layout) with profiles.")
    ap.add_argument("input_json", help="Election JSON file.")
    ap.add_argument("--outdir", default="out", help="Output directory for CSVs.")
    ap.add_argument("--quota_meta_csv", default=None, help="Write normalized mega/party meta CSV (path).")
    ap.add_argument("--max_iters", type=int, default=19, help="Max iterations INCLUDING the first pass.")
    ap.add_argument("--no_prompt", action="store_true",
                    help="Disable interactive prompts (profile selection + more-iters).")
    ap.add_argument("--profile", default=None,
                    help="Profile key (e.g. general_alpha). If omitted, prompt (unless --no_prompt).")
    args = ap.parse_args(argv)

    # Choose profile
    if args.profile is None:
        if args.no_prompt:
            profile = get_profile("general_alpha")
        else:
            profile = choose_profile_interactive()
    else:
        try:
            profile = get_profile(args.profile)
        except KeyError as e:
            raise SystemExit(str(e))

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    seats = int(data["seats"])

    # Parse inputs via io layer (profile-aware scaling hooks)
    base_groups = io_mod.canonicalize_base_ballots(data.get("ballots", []), profile=profile)
    total_voter_ballots = io_mod.total_normal_ballots_weight(base_groups)

    candidate_meta = io_mod.parse_candidate_meta(data.get("candidate_meta") or {})

    candidates: Set[str] = set(str(c) for c in data.get("candidates", []))
    candidates.update(io_mod.candidates_from_groups(base_groups))

    prefix_allow, ban = io_mod.parse_prefix_intervention(data)

    party_groups, party_lists, party_meta, party_cands = io_mod.parse_party_ballots(
        data.get("party_ballots", []),
        total_voter_ballots=total_voter_ballots,
        profile=profile,
    )
    candidates.update(party_cands)

    mega_groups, mega_meta = io_mod.parse_mega_ballots(
        data.get("mega_ballots", []),
        total_voter_ballots=total_voter_ballots,
        candidate_set=candidates,
        candidate_meta=candidate_meta,
        profile=profile,
    )

    candidates_list = sorted(candidates)

    if args.quota_meta_csv:
        io_mod.write_meta_csv(args.quota_meta_csv, mega_meta + party_meta)

    os.makedirs(args.outdir, exist_ok=True)

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
        party_lists=party_lists,
        stop_when_proj_gt=None,
        prefix_allow_only_init=prefix_allow,
        ban_set=ban,
        iter_allow_only_init=None,
        rounds_csv_path=pass1_rounds,
        quota_csv_path=pass1_quota,
        proj_csv_path=pass1_proj,
    )

    pass1_winners, pass1_intervals = recompute_intervals_from_rounds_csv(pass1_rounds)
    pass1_prefix = prefix_until_projection_strict_gt(pass1_winners, pass1_intervals, profile.sig_target)

    # Signature map: stop at first repeat (covers convergence + cycles)
    seen: Dict[Tuple[str, ...], int] = {tuple(pass1_prefix): 1}
    seen_label: Dict[Tuple[str, ...], str] = {tuple(pass1_prefix): "pass01"}

    prev_B_full_winners = pass1_winners
    prev_B_full_intervals = pass1_intervals

    solved = False
    solved_iter: Optional[int] = None
    twin_iter: Optional[int] = None
    twin_label: Optional[str] = None
    cycle_len: Optional[int] = None

    it = 2
    max_iters_current = args.max_iters

    while True:
        while it <= max_iters_current:
            labelA = f"iter{it:02d}A"
            labelB = f"iter{it:02d}B"

            pool_A = select_pool_by_coverage(prev_B_full_winners, prev_B_full_intervals, 1 / 9, 5 / 9)

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
                party_lists=party_lists,
                stop_when_proj_gt=profile.sig_target,
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_A,
                rounds_csv_path=A_rounds,
                quota_csv_path=A_quota,
                proj_csv_path=A_proj,
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
                party_lists=party_lists,
                stop_when_proj_gt=profile.sig_target,
                prefix_allow_only_init=prefix_allow,
                ban_set=ban,
                iter_allow_only_init=pool_B,
                rounds_csv_path=B_rounds,
                quota_csv_path=B_quota,
                proj_csv_path=B_proj,
            )

            B_winners, B_intervals = recompute_intervals_from_rounds_csv(B_rounds)
            B_prefix = prefix_until_projection_strict_gt(B_winners, B_intervals, profile.sig_target)
            sig = tuple(B_prefix)

            if sig in seen:
                solved = True
                solved_iter = it
                twin_iter = seen[sig]
                twin_label = seen_label.get(sig, "pass01" if twin_iter == 1 else f"iter{twin_iter:02d}B")
                cycle_len = it - twin_iter
                break

            seen[sig] = it
            seen_label[sig] = labelB

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
            party_lists=party_lists,
            prefix_allow=prefix_allow,
            ban=ban,
            completion_target=profile.completion_target,
        )

    print(json.dumps({
        "profile": profile.key,
        "status": "repeat_signature_found" if solved else "not_converged",
        "repeat_at_iteration": solved_iter,
        "twin_iteration": twin_iter,
        "twin_label": twin_label,
        "cycle_length": cycle_len,
        "max_iters_initial": args.max_iters,
        "max_iters_reached": max_iters_current,
        "completion": completion,
        "signature_definition": f"Part-B prefix winners until projection > {profile.sig_target} (strict)",
    }, ensure_ascii=False, indent=2))

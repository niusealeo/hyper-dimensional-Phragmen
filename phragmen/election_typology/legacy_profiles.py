from __future__ import annotations

from typing import Dict, List
from .types import ElectionProfile


def _identity(x: float) -> float:
    return float(x)


PROFILES: Dict[str, ElectionProfile] = {
    "general_alpha": ElectionProfile(
        key="general_alpha",
        name="General Alpha",
        description=(
            "Default profile for FIFO sequential Phragmén experiments.\n"
            "- Soft quota-floor reserve racers (mega/party/electorate) are active only when unsatisfied\n"
            "- Dormant quota groups accumulate reserve but do not affect selection\n"
            "- FIFO time-priority spending (cutoff τ) with configurable tier priorities\n"
            "- dt=0 ties choose max 'have', then party order, then name\n"
            "- Signature uses prefix until projection > 5/9 (strict)\n"
            "- Full chamber completes to max(seats, projection > 2/3 (strict))"
        ),
        scale_base_weight=_identity,
        scale_mega_rel_weight=_identity,
        scale_party_rel_weight=_identity,
        scale_electorate_rel_weight=_identity,
        sig_target=5.0 / 9.0,
        completion_target=2.0 / 3.0,
        spend_mode="fifo_time_priority",
        dt0_tie_rule="max_have_then_party_then_name",
        spend_tiers_default="base>electorate>party>mega",
        tier_within_mode="combined_fifo",
        prompt_block=19,
    ),

    "12": ElectionProfile(
        key="12",
        name="Profile 12",
        description=(
            "Profile 12: constant ballot normalisation between voter ballots (x) and party ballots (y), "
            "using optional total ballot capacity n (defaults to max(x,y)).\n"
            "Voter multiplier: (n + (2x - y)) / (3n)\n"
            "Party multiplier: (2n - (2x - y)) / (3n)\n"
            "Multipliers are computed once from input totals and remain constant throughout the election."
        ),
        scale_base_weight=_identity,
        scale_mega_rel_weight=_identity,
        scale_party_rel_weight=_identity,
        scale_electorate_rel_weight=_identity,
        sig_target=5.0 / 9.0,
        completion_target=2.0 / 3.0,
        spend_mode="fifo_time_priority",
        dt0_tie_rule="max_have_then_party_then_name",
        spend_tiers_default="base>electorate>party>mega",
        tier_within_mode="combined_fifo",
        prompt_block=19,
    ),
    "324": ElectionProfile(
        key="324",
        name="Profile 324",
        description=(
            "Profile 324: like Profile 12 but with voter/party multipliers scaled by 2/(9n) instead of 1/(3n), "
            "and with dynamic mega reweighting based on active (unsatisfied) mega weight z each round.\n"
            "Voter multiplier: (n + (2x - y)) * 2 / (9n)\n"
            "Party multiplier: (2n - (2x - y)) * 2 / (9n)\n"
            "Mega multiplier each round: if z <= n/3 then 1 else n/(3z), applied to active mega groups only."
        ),
        scale_base_weight=_identity,
        scale_mega_rel_weight=_identity,
        scale_party_rel_weight=_identity,
        scale_electorate_rel_weight=_identity,
        sig_target=5.0 / 9.0,
        completion_target=2.0 / 3.0,
        spend_mode="fifo_time_priority",
        dt0_tie_rule="max_have_then_party_then_name",
        spend_tiers_default="base>electorate>party>mega",
        tier_within_mode="combined_fifo",
        prompt_block=19,
    ),

}



def list_profiles() -> List[ElectionProfile]:
    return [PROFILES[k] for k in sorted(PROFILES.keys())]


def get_profile(key: str) -> ElectionProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(sorted(PROFILES.keys()))}")
    return PROFILES[key]

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
    )
}


def list_profiles() -> List[ElectionProfile]:
    return [PROFILES[k] for k in sorted(PROFILES.keys())]


def get_profile(key: str) -> ElectionProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(sorted(PROFILES.keys()))}")
    return PROFILES[key]

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
            "Default behaviour: identity scaling for all weights; "
            "signature repeats on prefix to projection > 5/9; "
            "completion when projection > 2/3."
        ),
        scale_base_weight=_identity,
        scale_mega_rel_weight=_identity,
        scale_party_rel_weight=_identity,
        sig_target=5 / 9,
        completion_target=2 / 3,
        prompt_block=19,
    )
}


def list_profiles() -> List[ElectionProfile]:
    return [PROFILES[k] for k in sorted(PROFILES.keys())]


def get_profile(key: str) -> ElectionProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(sorted(PROFILES.keys()))}")
    return PROFILES[key]

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


EPS = 1e-12

WeightFn = Callable[[float], float]


@dataclass(frozen=True)
class Group:
    gid: str
    kind: str  # "base" | "mega" | "party" | "electorate"
    approvals: Tuple[str, ...]
    weight: float
    quota_floor: Optional[float] = None
    population: Optional[float] = None
    abs_weight: Optional[float] = None
    share: Optional[float] = None


@dataclass(frozen=True)
class ElectionProfile:
    """
    Election 'settings bundle' (profile) that can later define scaling rules
    without touching the calculation engine.

    For now (general_alpha), these are identity functions.
    """
    key: str
    name: str
    description: str

    # Weight transforms
    scale_base_weight: WeightFn
    scale_mega_rel_weight: WeightFn
    scale_party_rel_weight: WeightFn
    scale_electorate_rel_weight: WeightFn

    # Thresholds (strict comparisons are used: > target)
    sig_target: float
    completion_target: float

    # UX
    prompt_block: int

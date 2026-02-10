from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

EPS: float = 1e-12

WeightFn = Callable[[float], float]

SpendMode = Literal["reset", "fifo_time_priority"]

DT0TieRule = Literal["party_then_name", "max_have_then_party_then_name"]

TierWithinMode = Literal["combined_fifo", "separate_by_kind"]


@dataclass(frozen=True)
class Group:
    gid: str
    kind: str  # "base" | "mega" | "party" | "electorate"
    approvals: Tuple[str, ...]
    weight: float

    quota_floor: Optional[float] = None

    # Audit metadata
    population: Optional[float] = None
    abs_weight: Optional[float] = None
    share: Optional[float] = None
    meta: Optional[dict] = None


@dataclass(frozen=True)
class ElectionProfile:
    key: str
    name: str
    description: str

    scale_base_weight: WeightFn
    scale_mega_rel_weight: WeightFn
    scale_party_rel_weight: WeightFn
    scale_electorate_rel_weight: WeightFn

    sig_target: float              # default 5/9
    completion_target: float       # default 2/3

    spend_mode: SpendMode
    dt0_tie_rule: DT0TieRule

    spend_tiers_default: str       # e.g. "base>party>electorate,mega"
    tier_within_mode: TierWithinMode

    prompt_block: int              # default 19

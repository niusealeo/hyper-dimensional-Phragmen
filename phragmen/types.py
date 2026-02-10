from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

# Numerical tolerance used across the project.
EPS: float = 1e-12

# Weight transforms (profiles can scale weights without changing the engine).
WeightFn = Callable[[float], float]

# Spending modes supported by the engine.
SpendMode = Literal[
    "reset",
    "partial_priority",
    "fifo_time_priority",
]

# dt=0 tie-breaking modes.
DT0TieRule = Literal[
    "party_then_name",
    "max_have_then_party_then_name",
]

# Within-tier behaviour when multiple kinds are grouped in a single tier.
TierWithinMode = Literal[
    "combined_fifo",
    "separate_by_kind",
]


@dataclass(frozen=True)
class Group:
    """
    A canonical approval group used by the engine.

    gid: stable identifier for the group (used for logging).
    kind: one of {"base","mega","party","electorate"}.
    approvals: candidates approved by this group.
    weight: group's effective earning rate / weight used by the engine.

    quota_floor:
      - for mega/party/electorate groups: soft quota-floor proportion in [0,1/3] typically.
      - None for base groups.

    population / abs_weight / share:
      - metadata useful for normalization logs and reproducibility.
      - 'weight' is the effective/relative weight that participates in the race.
    """
    gid: str
    kind: str  # "base" | "mega" | "party" | "electorate"
    approvals: Tuple[str, ...]
    weight: float

    quota_floor: Optional[float] = None

    # Metadata (optional)
    population: Optional[float] = None
    abs_weight: Optional[float] = None
    share: Optional[float] = None


@dataclass(frozen=True)
class ElectionProfile:
    """
    A profile is a named bundle of settings.

    Profiles define:
      - scaling rules (how to transform various weights)
      - signature/convergence targets used by the iterative A/B cycle logic
      - completion target for "full chamber"
      - default spend mode and dt=0 tie behaviour
      - default FIFO tier plan (spend_tiers_default) and within-tier behaviour

    IMPORTANT:
      Profiles are not the calculation engine; they only provide defaults and transforms.
      CLI flags and/or JSON may override these.
    """
    key: str
    name: str
    description: str

    # Weight transforms (applied during canonicalization / parsing)
    scale_base_weight: WeightFn
    scale_mega_rel_weight: WeightFn
    scale_party_rel_weight: WeightFn
    scale_electorate_rel_weight: WeightFn

    # Iteration signature target: stop prefix when projection is strictly > sig_target
    sig_target: float  # default 5/9

    # Completion target: full chamber is max(seats, first round where projection strictly > completion_target)
    completion_target: float  # default 2/3

    # Engine behaviour defaults
    spend_mode: SpendMode
    dt0_tie_rule: DT0TieRule

    # FIFO tier behaviour defaults
    # Syntax: tiers separated by '>', multiple kinds in same tier separated by ','
    # Example: "base>party>electorate,mega"
    spend_tiers_default: str
    tier_within_mode: TierWithinMode

    # How many iterations to propose at a time when prompting user for more.
    prompt_block: int

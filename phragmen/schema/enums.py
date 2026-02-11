from __future__ import annotations

from typing import Literal

# Expand as needed; kept as plain Literals for now to avoid heavy deps.
GroupKind = Literal[
    "base",
    "mega",
    "party",
    "electorate",
    "partyrock",
    "megarock",
]

from __future__ import annotations

"""Election typologies (renamed from 'profiles').

At this refactor stage, we keep the underlying dataclass shape identical to the
previous `ElectionProfile` to minimize churn.
"""

from ..types import ElectionProfile as ElectionTypology

__all__ = ["ElectionTypology"]

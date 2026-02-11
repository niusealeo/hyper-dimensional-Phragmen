from __future__ import annotations

"""Schema-level canonical models.

At this refactor stage, these types are re-exported from the existing `phragmen.types`
module to preserve backwards compatibility.

New code should import from `phragmen.schema.models`.
"""

from ..types import EPS, Group

__all__ = ["EPS", "Group"]

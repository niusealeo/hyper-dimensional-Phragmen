from __future__ import annotations

"""Apply an election typology to parsed election data.

In this refactor pass, typology application still lives in `phragmen.cli`.
This module is a reserved location for future consolidation.
"""

from .typology import ElectionTypology

__all__ = ["ElectionTypology"]

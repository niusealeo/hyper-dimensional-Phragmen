"""Sequential Phragmén (FIFO time-priority) toolkit."""

from .types import Group, ElectionProfile, EPS
from .election_typology.typology import ElectionTypology

__all__ = ["Group", "ElectionProfile", "ElectionTypology", "EPS"]

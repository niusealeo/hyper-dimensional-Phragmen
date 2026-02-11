from __future__ import annotations

"""Backwards-compatible façade.

Profiles have been renamed to election typologies.
"""

from .election_typology.registry import list_profiles, get_profile, TYPOLOGIES as PROFILES

__all__ = ["list_profiles", "get_profile", "PROFILES"]

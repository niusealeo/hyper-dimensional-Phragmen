from __future__ import annotations


class SpecError(Exception):
    """Base error for malformed or inconsistent election specs."""


class ValidationError(SpecError):
    """Raised when input fails validation."""


class NormalizationError(SpecError):
    """Raised when normalization / scaling cannot be applied safely."""

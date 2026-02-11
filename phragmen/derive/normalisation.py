from __future__ import annotations

from typing import List, Optional, Tuple, Any

from ..schema.models import Group
from ..parse.numeric import parse_numeric_optional


def _clone_group_with_weight(g: Group, new_weight: float, meta_update: Optional[dict] = None) -> Group:
    meta = dict(g.meta or {})
    if meta_update:
        meta.update(meta_update)
    return Group(
        gid=g.gid,
        kind=g.kind,
        approvals=g.approvals,
        weight=float(new_weight),
        quota_floor=g.quota_floor,
        population=g.population,
        abs_weight=g.abs_weight,
        share=g.share,
        meta=meta,
    )


def compute_profile_12_324_totals(data: dict) -> Tuple[float, float, float]:
    """Return (x, y, n) for profile 12 / 324 normalisation.

    x = total absolute voter ballots sum from data['ballots'] (sum of valid ballot weights)
    y = total absolute party ballots sum from data['party_ballots'] (sum of defs[*].weight when present)
    n = optional total ballot capacity (defaults to max(x, y) if missing)

    Accepted keys for n: 'n', 'n_ballots', 'total_ballots', 'ballots_n'.
    """
    ballots = data.get("ballots", []) if isinstance(data, dict) else []
    party_defs = data.get("party_ballots", []) if isinstance(data, dict) else []

    x = 0.0
    for b in ballots or []:
        apps = b.get("approvals", [])
        try:
            w = float(parse_numeric_optional(b.get("weight", 1.0), field="ballots.weight") or 0.0)
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        if not apps:
            continue
        if isinstance(apps, (list, tuple, set)) and len(apps) == 0:
            continue
        x += w

    y = 0.0
    for d in party_defs or []:
        if not isinstance(d, dict):
            continue
        if d.get("weight") is not None:
            try:
                w = float(parse_numeric_optional(d.get("weight"), field="ballots.weight") or 0.0)
            except Exception:
                w = 0.0
            if w > 0:
                y += w
        elif d.get("abs_weight") is not None:
            try:
                w = float(d.get("abs_weight"))
            except Exception:
                w = 0.0
            if w > 0:
                y += w

    n = None
    for k in ("n", "n_ballots", "total_ballots", "ballots_n"):
        if isinstance(data, dict) and data.get(k) is not None:
            try:
                n = float(data.get(k))
            except Exception:
                n = None
            break
    if n is None or n <= 0:
        n = max(x, y, 0.0)
    # Ensure n is at least max(x,y) (as per spec).
    n = max(n, x, y)
    return float(x), float(y), float(n)


def compute_profile_12_324_multipliers(profile_key: str, x: float, y: float, n: float) -> Tuple[float, float]:
    """Return (voter_mult, party_mult) for profile '12' or '324'."""
    profile_key = str(profile_key)
    x = float(x)
    y = float(y)
    n = float(n)
    if n <= 0:
        return 1.0, 1.0
    delta = (2.0 * x) - y
    if profile_key == "324":
        voter_mult = (n + delta) * 2.0 / (9.0 * n)
        party_mult = (2.0 * n - delta) * 2.0 / (9.0 * n)
    else:  # "12"
        voter_mult = (n + delta) / (3.0 * n)
        party_mult = (2.0 * n - delta) / (3.0 * n)
    return float(voter_mult), float(party_mult)


def apply_constant_multiplier(groups: List[Group], multiplier: float, label: str) -> List[Group]:
    if abs(multiplier - 1.0) <= 1e-18:
        return list(groups)
    out: List[Group] = []
    for g in groups:
        out.append(_clone_group_with_weight(g, g.weight * multiplier, meta_update={f"mult_{label}": multiplier}))
    return out

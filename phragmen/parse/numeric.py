from __future__ import annotations

from typing import Any, Optional

import ast


def parse_numeric(value: Any, *, field: str = "") -> float:
    """
    Parse numeric inputs that may be numbers or string expressions.

    Supported:
        - int/float
        - scientific notation (1e6, 5e-3)
        - fractions ("5/9", "1000/3")
        - arithmetic expressions with + - * / ** (or ^ mapped to **)
        - parentheses

    Rejected:
        - names
        - function calls
        - comparisons
        - boolean operators
        - bitwise operators
    """

    if value is None:
        raise ValueError(f"Missing numeric value for {field}" if field else "Missing numeric value")

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Unsupported numeric type for {field}: {type(value)}")

    s = value.strip()
    if s == "":
        raise ValueError(f"Empty numeric string for {field}" if field else "Empty numeric string")

    # Map caret exponent to Python exponent
    s = s.replace("^", "**")

    # Try simple float first (fast path)
    try:
        return float(s)
    except Exception:
        pass

    try:
        node = ast.parse(s, mode="eval")
    except Exception as e:
        raise ValueError(f"Invalid numeric expression for {field}: {value!r}") from e

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError(f"Only numeric constants allowed in {field}")

        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Unary operator not allowed in {field}")
            return _ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand))

        if isinstance(n, ast.BinOp):
            if type(n.op) not in _ALLOWED_BINOPS:
                raise ValueError(f"Operator not allowed in {field}")
            left = _eval(n.left)
            right = _eval(n.right)
            result = _ALLOWED_BINOPS[type(n.op)](left, right)
            return result

        raise ValueError(f"Disallowed expression element in {field}: {value!r}")

    result = _eval(node)

    # Reject complex numbers
    if isinstance(result, complex):
        raise ValueError(f"Complex numbers not allowed in {field}")

    return float(result)


def parse_numeric_optional(value: Any, *, field: str = "") -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return parse_numeric(value, field=field)


def clamp_quota_floor(qf: float) -> float:
    """Ensure quota floors never exceed 1/3."""
    return min(float(qf), 1.0 / 3.0)


def quota_floor_from_share(share: float) -> float:
    return min((2.0 / 3.0) * share, 1.0 / 3.0)


def share_from_quota_floor(qf: float) -> float:
    qf = float(qf)
    if qf < (1.0 / 3.0) - 1e-15:
        return 1.5 * qf
    return 0.5


def normalize_rel_weight_from_share(share: float, total_voter_ballots: float) -> float:
    # Canonical rule: rel_weight = N * quota_floor, where quota_floor = min((2/3)*share, 1/3)
    # This ensures the capped form: rel_weight = min((2/3)*w_abs, N/3) when share = w_abs/N.
    qf = quota_floor_from_share(float(share))
    return float(total_voter_ballots) * float(qf)


def _max_share_with_denoms(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (share_used, denom_used) given pairs of (numerator, denom).

    Each pair is interpreted as share = numerator/denom (with denom>0).
    The max share is selected; ties prefer the larger denom (stable cap behaviour).
    """
    best_share = 0.0
    best_denom = 0.0
    for num, denom in pairs:
        denom = float(denom)
        num = float(num)
        if denom <= 0:
            continue
        s = num / denom
        if s > best_share + 1e-18 or (abs(s - best_share) <= 1e-18 and denom > best_denom + 1e-18):
            best_share = s
            best_denom = denom
    return float(best_share), float(best_denom)


def compute_global_totals(data: dict) -> dict:
    """Compute global election totals.

    Expected input keys (top-level):
      - total_population (wglobal1)
      - total_enrollment (wglobal2)
      - total_turnout (wglobal3)

    Derived:
      - wglobal4 = max(wglobal3, sum(party wp1), sum(PartyRock wpr1), sum(base wr))
      - wglobal5 = max(wglobal2, sum(electorate we1))

    If any of wglobal1-3 are missing/invalid, defaults are computed where possible:
      - total_turnout defaults to sum(base wr)
      - total_enrollment defaults to sum(electorate we1)
      - total_population defaults to total_enrollment
    """
    d = data or {}

    def _f(x, field: str = "") -> float:
        try:
            v = parse_numeric_optional(x, field=field)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # Raw sums from input.
    base_abs = 0.0
    for b in d.get("ballots", []) or []:
        if not isinstance(b, dict):
            continue
        if not b.get("approvals"):
            continue
        w = _f(b.get("weight", 1.0), field="ballots.weight")
        if w > 0:
            base_abs += w

    party_abs = 0.0
    for p in d.get("party_ballots", []) or []:
        if not isinstance(p, dict):
            continue
        w = _f(p.get("weight", p.get("abs_weight", 0.0)), field="party_ballots.weight")
        if w > 0:
            party_abs += w

    partyrock_abs = 0.0
    for pr in d.get("partyrock_ballots", []) or []:
        if not isinstance(pr, dict):
            continue
        w = _f(pr.get("weight", 0.0), field="partyrock_ballots.weight")
        if w > 0:
            partyrock_abs += w

    electorate_enroll_abs = 0.0
    for e in d.get("electorate_ballots", []) or []:
        if not isinstance(e, dict):
            continue
        w = _f(e.get("weight", e.get("abs_weight", 0.0)), field="electorate_ballots.weight")
        if w > 0:
            electorate_enroll_abs += w

    # Globals.
    wglobal1 = _f(d.get("total_population", d.get("wglobal1")), field="total_population")
    wglobal2 = _f(d.get("total_enrollment", d.get("wglobal2")), field="total_enrollment")
    wglobal3 = _f(d.get("total_turnout", d.get("wglobal3")), field="total_turnout")

    if wglobal3 <= 0:
        wglobal3 = base_abs
    if wglobal2 <= 0:
        wglobal2 = electorate_enroll_abs
    if wglobal1 <= 0:
        wglobal1 = wglobal2 if wglobal2 > 0 else max(wglobal3, 0.0)

    wglobal4 = max(wglobal3, party_abs, partyrock_abs, base_abs)
    wglobal5 = max(wglobal2, electorate_enroll_abs)

    return {
        "wglobal1": float(wglobal1),
        "wglobal2": float(wglobal2),
        "wglobal3": float(wglobal3),
        "wglobal4": float(wglobal4),
        "wglobal5": float(wglobal5),
        "sum_base_abs": float(base_abs),
        "sum_party_abs": float(party_abs),
        "sum_partyrock_abs": float(partyrock_abs),
        "sum_electorate_enroll_abs": float(electorate_enroll_abs),
    }


META_FIELDS = [
    "gid",
    "kind",
    "population",
    "abs_weight",
    "share",
    "rel_weight",
    "quota_floor",
    "approvals_size",
]

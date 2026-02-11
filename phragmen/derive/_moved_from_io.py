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


def _canon_approvals(seq: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in seq if str(x).strip() != "")))


def compute_base_electorate_info(ballots: List[dict]) -> Dict[str, dict]:
    """Compute per-electorate info from raw base voter ballots.

    Returns mapping electorate_id -> {
        'counted_base_ballots': sum of ballot weights (raw, before profile scaling),
        'counted_base_records': number of ballot records (unweighted),
        'candidates': set of candidates appearing in ballots registered to the electorate,
    }

    The electorate id is read from one of: 'electorate', 'electorate_id', 'electorate_gid'.
    Missing electorate ids are mapped to 'electorate_unknown'.
    """
    info: Dict[str, dict] = {}
    for b in ballots or []:
        apps = _canon_approvals(b.get("approvals", []))
        try:
            w = float(parse_numeric_optional(b.get("weight", 1.0), field="ballots.weight") or 0.0)
        except Exception:
            w = 0.0
        if w <= 0 or not apps:
            continue
        eid = b.get("electorate")
        if eid is None:
            eid = b.get("electorate_id")
        if eid is None:
            eid = b.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        if eid not in info:
            info[eid] = {"counted_base_ballots": 0.0, "counted_base_records": 0, "candidates": set()}
        info[eid]["counted_base_ballots"] += w
        info[eid]["counted_base_records"] += 1
        info[eid]["candidates"].update(apps)
    return info


def compute_partyrock_electorate_info(partyrock_defs: List[dict]) -> Dict[str, dict]:
    """Compute per-electorate info from raw PartyRock ballots.

    Returns mapping electorate_id -> {
        'counted_partyrock_abs_weight': sum of PartyRock ballot absolute weights,
        'counted_partyrock_records': number of PartyRock ballot records,
        'counted_partyrock_ballots': a count-like total for denominator use (alias of records),
        'candidates': set of candidates appearing in PartyRock ballots registered to the electorate,
    }

    The electorate id is read from one of: 'electorate', 'electorate_id', 'electorate_gid'.
    Missing electorate ids are mapped to 'electorate_unknown'.
    """
    info: Dict[str, dict] = {}
    for d in partyrock_defs or []:
        if not isinstance(d, dict):
            continue
        apps = _canon_approvals(d.get("approvals", []))
        try:
            w_abs = float(parse_numeric_optional(d.get("weight", 0.0), field="partyrock.weight") or 0.0)
        except Exception:
            w_abs = 0.0
        if w_abs <= 0 or not apps:
            continue

        eid = d.get("electorate")
        if eid is None:
            eid = d.get("electorate_id")
        if eid is None:
            eid = d.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        if eid not in info:
            info[eid] = {
                "counted_partyrock_abs_weight": 0.0,
                "counted_partyrock_records": 0,
                "counted_partyrock_ballots": 0,
                "candidates": set(),
            }
        info[eid]["counted_partyrock_abs_weight"] += w_abs
        info[eid]["counted_partyrock_records"] += 1
        info[eid]["counted_partyrock_ballots"] += 1
        info[eid]["candidates"].update(apps)
    return info


def canonicalize_base_ballots(ballots: List[dict], profile: ElectionProfile) -> List[Group]:
    """Canonicalize and collapse base voter ballots.

    Collapsing key is (electorate_id, approvals) so identical approval lists in different electorates
    remain distinct groups.
    """
    agg: Dict[Tuple[str, Tuple[str, ...]], float] = defaultdict(float)

    for b in ballots or []:
        apps = _canon_approvals(b.get("approvals", []))
        w = float(parse_numeric_optional(b.get("weight", 1.0), field="ballots.weight") or 0.0)
        if w <= 0 or not apps:
            continue

        eid = b.get("electorate")
        if eid is None:
            eid = b.get("electorate_id")
        if eid is None:
            eid = b.get("electorate_gid")
        eid = str(eid) if eid is not None and str(eid).strip() != "" else "electorate_unknown"

        agg[(eid, apps)] += w

    out: List[Group] = []
    # deterministic order: by weight desc, then electorate id, then approvals
    items = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    for i, ((eid, apps), w) in enumerate(items):
        out.append(
            Group(
                gid=f"base_{i}",
                kind="base",
                approvals=apps,
                weight=profile.scale_base_weight(w),
                meta={"raw_weight": w, "approvals_size": len(apps), "electorate": eid},
            )
        )
    return out


def total_normal_ballots_weight(base_groups: List[Group]) -> float:
    return sum(g.weight for g in base_groups)


def candidates_from_groups(groups: List[Group]) -> Set[str]:
    s: Set[str] = set()
    for g in groups:
        s.update(g.approvals)
    return s



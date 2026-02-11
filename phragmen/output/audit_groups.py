from __future__ import annotations

from typing import Dict, List

import csv

from ..schema.models import Group


def write_meta_csv(path: str, groups: List[Group]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for g in groups:
            w.writerow({
                "gid": g.gid,
                "kind": g.kind,
                "population": g.population or "",
                "abs_weight": g.abs_weight or "",
                "share": g.share or "",
                "rel_weight": g.weight,
                "quota_floor": g.quota_floor if g.quota_floor is not None else "",
                "approvals_size": len(g.approvals),
            })




def write_globals_csv(path: str, globals_ctx: dict, norm_ctx: dict | None = None) -> None:
    """Write computed global totals (and optional profile normalisation context) to a key/value CSV."""
    rows = []
    for k in sorted((globals_ctx or {}).keys()):
        rows.append({"key": k, "value": (globals_ctx or {}).get(k)})
    if norm_ctx:
        for k in sorted(norm_ctx.keys()):
            rows.append({"key": f"norm_{k}", "value": norm_ctx.get(k)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


GROUP_AUDIT_FIELDS = [
    "gid",
    "kind",
    "population",
    "abs_weight",
    "share",
    "quota_floor",
    "rel_weight",
    "approvals_size",
    "meta_json",
]


def write_groups_audit_csv(path: str, groups: List[Group]) -> None:
    """Write per-group computed fields, plus the full meta dict as JSON."""
    import json as _json

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GROUP_AUDIT_FIELDS)
        w.writeheader()
        for g in groups:
            w.writerow({
                "gid": g.gid,
                "kind": g.kind,
                "population": g.population if g.population is not None else "",
                "abs_weight": g.abs_weight if g.abs_weight is not None else "",
                "share": g.share if g.share is not None else "",
                "quota_floor": g.quota_floor if g.quota_floor is not None else "",
                "rel_weight": g.weight,
                "approvals_size": len(g.approvals),
                "meta_json": _json.dumps(g.meta or {}, ensure_ascii=False, sort_keys=True),
            })


def write_party_lists_csv(path: str, party_lists: Dict[str, List[str]]) -> None:
    """Write party lists (including PartyRock extensions) in an explicit stable order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["party_id", "pos", "candidate"])
        w.writeheader()
        for pid in sorted((party_lists or {}).keys()):
            lst = party_lists.get(pid) or []
            for i, c in enumerate(lst, start=1):
                w.writerow({"party_id": pid, "pos": i, "candidate": c})


# Backwards-compatible alias expected by cli.py
def write_globals_kv_csv(path: str, globals_ctx: dict, norm_ctx: dict | None = None) -> None:
    return write_globals_csv(path, globals_ctx, norm_ctx=norm_ctx)

"""Exact change statistics for a LISTING result (no LLM).

A listing of one value per item and period (a measure run over every item)
answers "how has X changed" only if someone does the arithmetic -- dozens of
first/last values and percentages. Models get that wrong, so it is done here:
per item the first and last value over the periods returned, the absolute and
percentage change, and the range; over all items how many rose / fell, the
median change and the largest movers. Domain-neutral: the period and value
columns come from the measure definition or the table's column roles.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from decimal import Decimal
from typing import Any

SKIP_COLUMNS = {"rows_aggregated"}
FLAT_SHARE = 0.005          # |change| below 0.5 % counts as unchanged
TOP_MOVERS = 5


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _key(v: Any) -> str:
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return "" if v is None else str(v)


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e6:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:,.1f}"
    if a >= 1:
        return f"{x:,.2f}"
    return f"{x:.4g}"


def roles_for(meta: dict, query_id: str | None, columns: list[str]) -> tuple[str | None, str | None]:
    """(period column, value column) of a result, or (None, None)."""
    cols = set(columns)
    for it in meta.get("measures") or []:
        if it.get("id") == query_id and isinstance(it.get("measure"), dict):
            m = it["measure"]
            periods = [p for p in m.get("period_columns") or [] if p in cols]
            if periods and m.get("value_column") in cols:
                return periods[-1], m["value_column"]
    by_name = {c.get("name"): c for c in meta.get("columns") or []}
    periods = [c for c in columns if (by_name.get(c) or {}).get("role") == "time"]
    values = [c for c in columns if (by_name.get(c) or {}).get("role") == "measure"]
    if len(periods) == 1 and len(values) == 1:
        return periods[0], values[0]
    return None, None


def change_stats(columns: list[str], rows: list, period_col: str, value_col: str) -> dict | None:
    """Per-item change between the first and last period of each item."""
    if period_col not in columns or value_col not in columns:
        return None
    pi, vi = columns.index(period_col), columns.index(value_col)
    group_idx = [i for i, c in enumerate(columns)
                 if i not in (pi, vi) and c not in SKIP_COLUMNS
                 and not c.startswith(("first_", "last_"))]
    # a column holding one value on every row says nothing about the item
    group_idx = [i for i in group_idx if len({_key(r[i]) for r in rows}) > 1] or group_idx
    series: dict[tuple, list[tuple[str, float]]] = {}
    for r in rows:
        v = _num(r[vi])
        if v is None or r[pi] is None:
            continue
        series.setdefault(tuple(_key(r[i]) for i in group_idx), []).append((_key(r[pi]), v))
    periods = sorted({p for s in series.values() for p, _ in s})
    if len(periods) < 2 or not series:
        return None
    items = []
    for g, pts in series.items():
        pts.sort()
        (p0, v0), (p1, v1) = pts[0], pts[-1]
        vals = [v for _, v in pts]
        pct = (v1 - v0) / abs(v0) * 100 if v0 not in (0, 0.0) and p1 != p0 else None
        items.append({"item": " / ".join(x for x in g if x) or "(all)", "first_period": p0, "first": v0,
                      "last_period": p1, "last": v1, "change": v1 - v0 if p1 != p0 else None,
                      "pct": pct, "min": min(vals), "max": max(vals), "periods": len(pts)})
    with_pct = [it for it in items if it["pct"] is not None]
    up = sum(1 for it in with_pct if it["pct"] > FLAT_SHARE * 100)
    down = sum(1 for it in with_pct if it["pct"] < -FLAT_SHARE * 100)
    ranked = sorted(with_pct, key=lambda it: it["pct"])
    return {
        "period_column": period_col, "value_column": value_col,
        "group_columns": [columns[i] for i in group_idx],
        "periods": [periods[0], periods[-1]], "n_periods": len(periods),
        "items": sorted(items, key=lambda it: it["item"]),
        "rose": up, "fell": down, "unchanged": len(with_pct) - up - down,
        "no_change_computable": len(items) - len(with_pct),
        "median_pct": statistics.median([it["pct"] for it in with_pct]) if with_pct else None,
        "top_risers": [it["item"] for it in reversed(ranked[-TOP_MOVERS:]) if it["pct"] > FLAT_SHARE * 100],
        "top_fallers": [it["item"] for it in ranked[:TOP_MOVERS] if it["pct"] < -FLAT_SHARE * 100],
    }


def stats_text(st: dict) -> str:
    """Compact rendering for the evidence block."""
    head = (f"CHANGE SUMMARY (computed exactly from these rows; use these numbers, do not recompute): "
            f"{len(st['items'])} item(s) by {', '.join(st['group_columns']) or 'row'}, "
            f"{st['value_column']} over {st['period_column']} {st['periods'][0]}..{st['periods'][1]} "
            f"({st['n_periods']} period(s)). First vs last period of each item: {st['rose']} rose, "
            f"{st['fell']} fell, {st['unchanged']} unchanged (within 0.5%)")
    if st["no_change_computable"]:
        head += f", {st['no_change_computable']} with one period or a zero start (no % change)"
    if st["median_pct"] is not None:
        head += f"; median change {st['median_pct']:+.1f}%"
    head += "."
    if st["top_risers"]:
        head += " Largest rises: " + ", ".join(st["top_risers"]) + "."
    if st["top_fallers"]:
        head += " Largest falls: " + ", ".join(st["top_fallers"]) + "."
    lines = ["item | first | last | change | change % | min..max"]
    for it in st["items"]:
        pct = "" if it["pct"] is None else f"{it['pct']:+.1f}%"
        chg = "" if it["change"] is None else ("+" if it["change"] >= 0 else "") + _fmt(it["change"])
        lines.append(f"{it['item']} | {it['first_period']}: {_fmt(it['first'])} | "
                     f"{it['last_period']}: {_fmt(it['last'])} | {chg} | {pct} | "
                     f"{_fmt(it['min'])}..{_fmt(it['max'])}")
    return head + " Per item: " + " ;; ".join(lines)


__all__ = ["change_stats", "roles_for", "stats_text"]

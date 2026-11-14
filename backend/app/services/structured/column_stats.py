"""Measured column statistics and the table's METADATA CARD (no LLM).

Questions about a table itself -- what it holds, its columns, the period it
covers, how many distinct items it tracks -- are answered from its metadata,
not from SQL the model has to invent. The LLM-written metadata says what the
table and its columns MEAN; this module adds what can only be measured:

* per column: non-null rows and distinct values; min/max for dates, numbers
  and time columns;
* per period column: the range per granularity (years 1960..2025, months
  1960M01..2026M10), since one text column may mix annual and monthly
  periods and a plain MIN/MAX of the text would hide that.

Stored in ``meta["column_stats"]``; :func:`metadata_card` renders the card
the answer model reads. Read-only on the table.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text as sql_text

MAX_PERIOD_VALUES = 50000
_RANGE_TYPES = {"date", "timestamp", "numeric", "integer"}
_GRANULARITY = ("year", "quarter", "month", "day")


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def granularity(value: Any) -> str | None:
    """year / quarter / month / day of one period value, or None."""
    from datetime import date

    from backend.app.services.structured.graph_links import period_of

    p = period_of(value)
    if p is None:
        return None
    if isinstance(value, date):
        return "day"
    s = str(value).strip()
    if p[2] is not None:
        is_day = (len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-") or (len(s) == 8 and s.isdigit())
        return "day" if is_day else "month"
    return "quarter" if p[1] is not None else "year"


def period_ranges(values: list[Any]) -> dict[str, list]:
    """{granularity: [earliest, latest]} over the parseable values, in the
    values' own spelling."""
    from backend.app.services.structured.graph_links import period_of

    best: dict[str, list] = {}
    for v in values:
        g = granularity(v)
        if g is None:
            continue
        key = period_of(v)
        lo_hi = best.get(g)
        if lo_hi is None:
            best[g] = [(key, v), (key, v)]
            continue
        if key < lo_hi[0][0]:
            lo_hi[0] = (key, v)
        if key > lo_hi[1][0]:
            lo_hi[1] = (key, v)
    out = {}
    for g in _GRANULARITY:
        if g in best:
            lo, hi = best[g][0][1], best[g][1][1]
            out[g] = [lo.isoformat() if hasattr(lo, "isoformat") else lo,
                      hi.isoformat() if hasattr(hi, "isoformat") else hi]
    return out


def time_like_columns(meta: dict) -> list[str]:
    from backend.app.services.structured.graph_links import period_columns

    cols = list(period_columns(meta))
    for c in meta.get("columns") or []:
        if (c.get("role") == "time" or c.get("type") in {"date", "timestamp"}) and c["name"] not in cols:
            cols.append(c["name"])
    return cols


async def compute_column_stats(session, meta: dict) -> dict[str, dict]:
    from backend.app.services.structured.value_domains import _plain

    table = meta["table_name"]
    ref = f"graphrag.{_q(table)}"
    cols = [c for c in meta.get("columns") or [] if c.get("name")]
    if not cols:
        return {}
    parts = []
    for c in cols:
        n = _q(c["name"])
        parts += [f"COUNT({n})", f"COUNT(DISTINCT {n})"]
        if (c.get("type") or "text") in _RANGE_TYPES:
            parts += [f"MIN({n})", f"MAX({n})"]
    row = (await session.execute(sql_text(f"SELECT {', '.join(parts)} FROM {ref}"))).one()
    stats: dict[str, dict] = {}
    i = 0
    for c in cols:
        d: dict[str, Any] = {"non_null": int(row[i]), "distinct": int(row[i + 1])}
        i += 2
        if (c.get("type") or "text") in _RANGE_TYPES:
            d["min"], d["max"] = _plain(row[i]), _plain(row[i + 1])
            i += 2
        stats[c["name"]] = d
    for name in time_like_columns(meta):
        if name not in stats:
            continue
        vals = (await session.execute(sql_text(
            f"SELECT DISTINCT {_q(name)} FROM {ref} WHERE {_q(name)} IS NOT NULL "
            f"LIMIT {MAX_PERIOD_VALUES}"))).scalars().all()
        ranges = period_ranges(list(vals))
        if ranges:
            stats[name]["periods"] = ranges
    meta["column_stats"] = stats
    return stats


def latest_period(meta: dict, column: str, like: Any) -> Any | None:
    """The latest value of ``column`` at the granularity of ``like`` (a
    parameter value such as '2023' or '2023M01'), or None."""
    g = granularity(like)
    rng = ((meta.get("column_stats") or {}).get(column) or {}).get("periods") or {}
    return (rng.get(g) or [None, None])[1] if g else None


def metadata_card(meta: dict, *, max_values: int = 40) -> str:
    """What the answer model reads about a table: meaning (LLM-written) plus
    measured coverage (exact)."""
    stats = meta.get("column_stats") or {}
    domains = meta.get("value_domains") or {}
    src = (meta.get("source") or {}).get("rel_path")
    lines = [f"Table {meta.get('table_name')}"
             + (f" (source file {src})" if src else "")
             + f": {meta.get('row_count')} rows. {str(meta.get('description') or '').strip()}"]
    grain = meta.get("grain")
    if grain:
        lines.append(f"One row = {grain if isinstance(grain, str) else ', '.join(map(str, grain))}.")
    lines.append("Columns (exact counts and ranges measured on the loaded table):")
    for c in meta.get("columns") or []:
        name = c.get("name")
        s = stats.get(name) or {}
        bits = [f"{c.get('type') or 'text'}", f"{c.get('role') or ''}".strip()]
        cov = []
        if s:
            cov.append(f"{s.get('distinct')} distinct value(s) in {s.get('non_null')} non-empty row(s)")
        if s.get("periods"):
            cov.append("periods " + "; ".join(f"{g} {lo}..{hi}" for g, (lo, hi) in s["periods"].items()))
        elif "min" in s and s.get("min") is not None:
            cov.append(f"range {s['min']}..{s['max']}")
        vals = (domains.get(name) or {}).get("values")
        if vals:
            shown = ", ".join(map(str, vals[:max_values]))
            cov.append(f"values: {shown}" + (f" (+{len(vals) - max_values} more)" if len(vals) > max_values else ""))
        lines.append(f"- {name} ({', '.join(b for b in bits if b)}): {str(c.get('description') or '').strip()}"
                     + (f" [{'; '.join(cov)}]" if cov else ""))
    measures = [m.get("label") for m in meta.get("measures") or [] if m.get("label")]
    if measures:
        lines.append("Measures: " + ", ".join(measures) + ".")
    notes = [n for n in meta.get("usage_notes") or [] if n]
    if notes:
        lines.append("Usage notes: " + " ".join(notes))
    return "\n".join(lines)


__all__ = ["compute_column_stats", "granularity", "latest_period", "metadata_card",
           "period_ranges", "time_like_columns"]

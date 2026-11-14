"""How a loaded table's columns interact -- measured with SQL, no LLM.

A column description says what a column is; it does not say how to query
the table correctly. These facts come from the data itself and hold for any
table:

* **grain** -- a small set of columns that identifies one row (greedy
  search for a unique key).
* **functional dependencies** -- ``a -> b`` when every value of ``a`` has one
  value of ``b`` (a code and its name, a report id and its date).
* **repeated values** -- a measure that stays the same when only some
  columns change (and the columns that depend on them). Those columns
  describe the REPORT or VERSION a value came from, not the quantity: the
  same figure is restated in several reports, so summing across them counts
  it several times. This is how a table of filings, vintages, revisions or
  snapshots behaves, whatever its domain.
* **item columns** -- a text column in the grain whose values name different
  quantities (a long-format table); its most widely shared values are listed
  so the measures can be defined on them.

The result is stored in ``meta["analysis"]`` and drives both the LLM step
that defines measures (``refine``) and the deterministic tests of every
stored query.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text

log = logging.getLogger(__name__)

MAX_KEY_COLUMNS = 8
MAX_CANDIDATES = 16
REPEAT_MIN_SHARE = 0.9        # of multi-row groups where the measure is constant
REPEAT_MIN_GROUPS = 0.02      # multi-row groups / all groups, to count at all
TOP_VALUES = 60
TIMEOUT_MS = 30000


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _cols(cols: list[str]) -> str:
    return ", ".join(_q(c) for c in cols)


async def _scalar(session, sql: str, binds: dict | None = None):
    try:
        async with session.begin_nested():
            await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
            return (await session.execute(sql_text(sql), binds or {})).scalar()
    except Exception as exc:  # a slow or odd column must not stop the analysis
        log.info("analysis query skipped: %s", str(exc).splitlines()[0][:200])
        return None


def candidate_columns(meta: dict) -> list[dict]:
    """Columns that can identify or classify a row (not measures, not free text)."""
    out = []
    for c in meta.get("columns") or []:
        if c.get("role") in {"measure", "text"}:
            continue
        out.append(c)
    return out[:MAX_CANDIDATES * 2]


def measure_columns(meta: dict) -> list[str]:
    return [c["name"] for c in meta.get("columns") or []
            if c.get("role") == "measure" and c.get("type") in {"integer", "numeric"}]


async def _distinct(session, ref: str, cols: list[str]) -> int | None:
    if not cols:
        return 1
    return await _scalar(session, f"SELECT COUNT(*) FROM (SELECT DISTINCT {_cols(cols)} FROM {ref}) d")


async def analyze_table(session, meta: dict) -> dict[str, Any]:
    table = meta["table_name"]
    ref = f"graphrag.{_q(table)}"
    rows = await _scalar(session, f"SELECT COUNT(*) FROM {ref}") or 0
    out: dict[str, Any] = {"row_count": rows, "grain": [], "grain_unique": False,
                           "dependencies": {}, "repeats": {}, "item_columns": {}}
    if rows == 0:
        return out

    cands = candidate_columns(meta)
    distinct = {}
    for c in cands:
        n = await _distinct(session, ref, [c["name"]])
        if n is not None:
            distinct[c["name"]] = n
    # A column unique on (nearly) every row is a row id, not part of a grain.
    usable = [c["name"] for c in cands if c["name"] in distinct and 1 < distinct[c["name"]] < 0.95 * rows]
    usable = sorted(usable, key=lambda n: -distinct[n])[:MAX_CANDIDATES]

    # ---- grain: greedy unique key
    key: list[str] = []
    best = 1
    while len(key) < MAX_KEY_COLUMNS and best < rows:
        step = None
        for c in usable:
            if c in key:
                continue
            n = await _distinct(session, ref, [*key, c])
            if n is not None and n > best and (step is None or n > step[1]):
                step = (c, n)
        if step is None:
            break
        key.append(step[0])
        best = step[1]
    out["grain"] = key
    out["grain_unique"] = best >= rows
    out["grain_distinct"] = best

    # ---- functional dependencies a -> b (only where |b| <= |a|)
    deps: dict[str, list[str]] = {}
    for a in usable:
        for b in usable:
            if a == b or distinct[b] > distinct[a]:
                continue
            n = await _distinct(session, ref, [a, b])
            if n is not None and n == distinct[a]:
                deps.setdefault(a, []).append(b)
    out["dependencies"] = deps

    # ---- repeated measure values across report/version columns
    measures = measure_columns(meta)
    for m in measures:
        found = []
        for c in usable:
            moved = [c, *deps.get(c, [])]
            group = [x for x in usable if x not in moved]
            if not group:
                continue
            stats = await _scalar(session, f"""
                SELECT json_build_object(
                    'multi', COUNT(*) FILTER (WHERE n > 1),
                    'groups', COUNT(*),
                    'same', COUNT(*) FILTER (WHERE n > 1 AND nd = 1))
                  FROM (SELECT COUNT(*) AS n, COUNT(DISTINCT {_q(m)}) AS nd
                          FROM {ref} WHERE {_q(m)} IS NOT NULL GROUP BY {_cols(group)}) g""")
            if not stats or not stats.get("groups"):
                continue
            multi, groups, same = stats["multi"], stats["groups"], stats["same"]
            if multi and multi / groups >= REPEAT_MIN_GROUPS and same / multi >= REPEAT_MIN_SHARE:
                found.append({"column": c, "with": deps.get(c, []),
                              "share_of_groups_repeated": round(multi / groups, 3),
                              "share_identical": round(same / multi, 3)})
        if found:
            out["repeats"][m] = found

    # ---- item columns: text grain columns with many values; their most
    # widely shared values (present for the most entities, then most rows)
    ent = next((c["name"] for c in cands if c.get("role") == "identifier" and c["name"] in usable), None)
    domains = meta.get("value_domains") or {}
    types = {c["name"]: c.get("type") for c in cands}
    for c in key:
        if types.get(c) != "text" or "values" in domains.get(c, {}):
            continue
        order = f"COUNT(DISTINCT {_q(ent)}) DESC, " if ent and ent != c else ""
        try:
            async with session.begin_nested():
                await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
                vals = (await session.execute(sql_text(
                    f"SELECT {_q(c)} FROM {ref} WHERE {_q(c)} IS NOT NULL GROUP BY 1 "
                    f"ORDER BY {order}COUNT(*) DESC, 1 LIMIT {TOP_VALUES}"))).scalars().all()
        except Exception:
            vals = []
        if vals:
            out["item_columns"][c] = {"distinct": distinct.get(c), "top_values": [str(v) for v in vals]}
    return out


def report_columns(analysis: dict, measure: str | None = None) -> list[str]:
    """Columns across which a measure's value is restated (plus the columns
    they determine)."""
    out: list[str] = []
    reps = analysis.get("repeats") or {}
    for m, found in reps.items():
        if measure and m != measure:
            continue
        for f in found:
            for c in [f["column"], *f.get("with", [])]:
                if c not in out:
                    out.append(c)
    return out


def summary_for_prompt(analysis: dict) -> dict:
    """The parts of the analysis the LLM and the selection model need."""
    out = {"grain": analysis.get("grain"), "grain_unique": analysis.get("grain_unique")}
    if analysis.get("repeats"):
        out["values_repeated_across"] = {
            m: sorted({c for f in found for c in [f["column"], *f.get("with", [])]})
            for m, found in analysis["repeats"].items()}
    if analysis.get("dependencies"):
        out["determines"] = analysis["dependencies"]
    if analysis.get("item_columns"):
        out["item_columns"] = analysis["item_columns"]
    return out

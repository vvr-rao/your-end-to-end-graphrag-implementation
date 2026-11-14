"""Valid parameter values, read from the loaded table (no LLM).

A filter or query only works when its parameters hold values that exist in
the column: 'Iron ore, cfr spot' (not 'Iron ore'), '($/dmtu)' (not
'($/dmt)'), 'ifrs-full' (not 'us-gaap' for an IFRS filer). The model that
fills the parameters cannot know them, so after a table is loaded:

* every filterable column gets a DOMAIN in ``meta["value_domains"]``:
  ``{"values": [...]}`` when it is a fixed set (at most ``max_values``
  distinct values that repeat across rows -- a near-unique key or name
  column is ``{"distinct": "unique"}``),
  ``{"min": .., "max": ..}`` for dates/numbers/time columns, else
  ``{"distinct": "many"}`` (too many to list; checked against the table at
  question time instead);
* every parameter of every stored item copies its column's domain
  (``allowed_values`` or ``range``), so the values travel with the query;
* an item whose equality parameters all have listed values also stores
  ``allowed_combinations`` -- the value tuples that actually occur together
  (iron ore is only priced in '($/dmtu)'), when there are at most
  ``max_combos`` of them.

At question time :func:`check_params` validates the model's values against
these lists BEFORE any SQL runs: an exact match passes, a match that differs
only in case/spacing/punctuation is replaced (and recorded), anything else is
a failure that hands the valid values to the one retry.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text as sql_text

MAX_VALUES = 100
MAX_COMBOS = 400
_RANGE_TYPES = {"date", "timestamp", "numeric"}
_LISTABLE_TYPES = {"text", "boolean", "integer"}
_EQ_OPS = {"=", "in"}


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _plain(v: Any):
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def norm(v: Any) -> str:
    """Case, spacing and punctuation-insensitive key ('($/dmtu)' != '($/dmt)')."""
    return re.sub(r"[^0-9a-z]+", "", str(v).casefold())


async def column_domain(session, table: str, column: dict, *, max_values: int = MAX_VALUES,
                        row_count: int | None = None) -> dict:
    name, ctype = column["name"], column.get("type") or "text"
    ref = f"graphrag.{_q(table)}"
    if ctype in _LISTABLE_TYPES and not (ctype == "integer" and column.get("role") == "measure"):
        rows = (await session.execute(sql_text(
            f"SELECT DISTINCT {_q(name)} FROM {ref} WHERE {_q(name)} IS NOT NULL "
            f"ORDER BY 1 LIMIT {max_values + 1}"))).scalars().all()
        if not rows:
            return {"empty": True}
        # A FIXED SET repeats across rows; a column whose values are (nearly)
        # all different (keys, names, addresses) is not one, however short.
        unique = len(rows) > 20 and row_count and len(rows) >= 0.8 * row_count
        if len(rows) <= max_values and not unique:
            return {"values": [_plain(v) for v in rows]}
        if unique:
            return {"distinct": "unique"}
        if ctype == "text" and column.get("role") != "time":
            return {"distinct": "many"}
    if ctype in _RANGE_TYPES | {"integer"} or column.get("role") == "time":
        lo, hi = (await session.execute(sql_text(
            f"SELECT MIN({_q(name)}), MAX({_q(name)}) FROM {ref}"))).one()
        return {"min": _plain(lo), "max": _plain(hi)}
    return {"distinct": "many"}


def _eq_param_columns(item: dict) -> dict[str, str]:
    """param name -> column for the spec's single-table = / in filters."""
    out = {}
    for w in (item.get("query") or {}).get("where") or []:
        if not isinstance(w, dict) or str(w.get("op") or "=").lower() not in _EQ_OPS:
            continue
        col, prm = w.get("column"), w.get("param")
        if isinstance(col, str) and "." not in col and isinstance(prm, str):
            out[prm] = col
    return out


async def attach_value_domains(session, meta: dict, *, max_values: int = MAX_VALUES,
                               max_combos: int = MAX_COMBOS) -> dict[str, int]:
    """Fill ``value_domains``, per-param ``allowed_values``/``range`` and
    per-item ``allowed_combinations`` in place. Read-only on the table."""
    from backend.app.services.structured.metadata import items

    table = meta["table_name"]
    cols = {c["name"]: c for c in meta.get("columns") or []}
    domains: dict[str, dict] = {}
    for c in cols.values():
        if c.get("role") in {"measure", "text"} and c.get("type") != "integer":
            continue      # never a filter value list (prices, free text)
        d = await column_domain(session, table, c, max_values=max_values,
                                row_count=meta.get("row_count"))
        if not d.get("empty"):
            domains[c["name"]] = d
    meta["value_domains"] = domains

    stats = {"listed_columns": sum(1 for d in domains.values() if "values" in d),
             "params_with_values": 0, "items_with_combinations": 0}
    combo_cache: dict[tuple[str, ...], list | None] = {}
    for _sec, it in items(meta):
        for p in it.get("params") or []:
            p.pop("allowed_values", None)
            p.pop("range", None)
            d = domains.get(p.get("column") or "")
            if d and "values" in d:
                p["allowed_values"] = d["values"]
                stats["params_with_values"] += 1
            elif d and "min" in d:
                p["range"] = [d["min"], d["max"]]
        it.pop("allowed_combinations", None)
        eq = {k: v for k, v in _eq_param_columns(it).items()
              if "values" in domains.get(v, {}) and len(domains[v]["values"]) > 1}
        if len(eq) < 2:
            continue
        names = list(eq)
        key = tuple(eq[n] for n in names)
        if key not in combo_cache:
            rows = (await session.execute(sql_text(
                f"SELECT DISTINCT {', '.join(_q(c) for c in key)} FROM graphrag.{_q(table)} "
                f"ORDER BY {', '.join(str(i + 1) for i in range(len(key)))} LIMIT {max_combos + 1}"
            ))).all()
            combo_cache[key] = ([[_plain(v) for v in r] for r in rows]
                                if len(rows) <= max_combos else None)
        if combo_cache[key] is not None:
            it["allowed_combinations"] = {"params": names, "rows": combo_cache[key]}
            stats["items_with_combinations"] += 1
    return stats


# ------------------------------------------------------------------ question time


def _match(value: Any, allowed: list) -> tuple[bool, Any]:
    """(ok, value to use). Exact, then the unique case/punctuation-insensitive match."""
    if value in allowed or str(value) in {str(a) for a in allowed}:
        exact = next((a for a in allowed if str(a) == str(value)), value)
        return True, exact
    hits = [a for a in allowed if norm(a) == norm(value) and norm(value)]
    if len(hits) == 1:
        return True, hits[0]
    return False, value


def check_params(params: dict[str, Any], declared: list[dict] | None,
                 combinations: dict | None = None, *, show: int = 40,
                 ) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Validate parameter values against their stored valid values.

    Returns (params to run with, problems, substitutions). ``problems`` is
    empty when the query may run; otherwise each entry names the param, the
    bad value and the valid values (or valid combinations) for the retry."""
    out = dict(params or {})
    problems: list[dict] = []
    subs: list[dict] = []
    for p in declared or []:
        name, allowed = p.get("name"), p.get("allowed_values")
        if name not in out or not allowed:
            continue
        val = out[name]
        vals = val if isinstance(val, list) else [val]
        fixed = []
        for v in vals:
            ok, use = _match(v, allowed)
            if not ok:
                problems.append({"param": name, "value": v, "column": p.get("column"),
                                 "allowed_values": allowed[:show],
                                 **({"more_values": len(allowed) - show} if len(allowed) > show else {})})
            elif str(use) != str(v):
                subs.append({"param": name, "from": v, "to": use})
            fixed.append(use)
        out[name] = fixed if isinstance(val, list) else fixed[0]
    if not problems and combinations and combinations.get("rows"):
        names = combinations["params"]
        if all(n in out and not isinstance(out[n], list) for n in names):
            want = [str(out[n]) for n in names]
            rows = combinations["rows"]
            if [str(x) for x in want] not in [[str(x) for x in r] for r in rows]:
                # Rows sharing the most values with the request: the closest valid choices.
                best = max(sum(str(a) == b for a, b in zip(r, want, strict=True)) for r in rows)
                near = [dict(zip(names, r, strict=True)) for r in rows
                        if sum(str(a) == b for a, b in zip(r, want, strict=True)) == best][:show]
                problems.append({"combination": dict(zip(names, want, strict=True)),
                                 "error": "these values never occur together in the table",
                                 "valid_combinations": near})
    return out, problems, subs


def spec_declared(built_params: list[dict], domains: dict[str, dict]) -> list[dict]:
    """Declared params for a model-written spec: compiler params + the
    table's stored column domains."""
    out = []
    for p in built_params:
        d = domains.get(p.get("column") or "") or {}
        out.append({**p, **({"allowed_values": d["values"]} if "values" in d else {})})
    return out

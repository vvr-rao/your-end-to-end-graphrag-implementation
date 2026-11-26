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

import json
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


def _fixed_where(item: dict, cols: set[str]) -> list[dict]:
    """The item's constant (param-free) single-table where entries: fixed
    values and population predicates."""
    out = []
    for w in (item.get("query") or {}).get("where") or []:
        c = w.get("column") if isinstance(w, dict) else None
        if isinstance(c, str) and "." not in c and c in cols and w.get("param") in (None, "") \
                and ("value" in w or str(w.get("op")) in {"is_null", "not_null"}):
            out.append({k: w[k] for k in ("column", "op", "value") if k in w})
    return out


def _fixed_sql(table: str, types: dict[str, str], fixed: list[dict]) -> str:
    """' WHERE ...' for constant entries, compiled by the query builder
    (literals only, no binds); '' when there are none."""
    if not fixed:
        return ""
    from backend.app.services.structured.query_builder import build_sql

    built = build_sql({"select": [next(iter(types))], "where": fixed}, table, {table: types})
    i = built.sql.find(" WHERE ")
    if i < 0:
        return ""
    j = built.sql.find(" ORDER BY ", i)
    return built.sql[i:j if j > 0 else None]


async def _scoped_values(session, table: str, types: dict[str, str], column: str, fixed: list[dict],
                         max_values: int) -> list | None:
    where = _fixed_sql(table, types, fixed)
    rows = (await session.execute(sql_text(
        f"SELECT DISTINCT {_q(column)} FROM graphrag.{_q(table)}{where}"
        f"{' AND' if where else ' WHERE'} {_q(column)} IS NOT NULL ORDER BY 1 LIMIT {max_values + 1}"
    ))).scalars().all()
    return [_plain(v) for v in rows] if len(rows) <= max_values else None


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
    from backend.app.services.structured.column_stats import compute_column_stats

    await compute_column_stats(session, meta)       # counts, ranges, period ranges

    stats = {"listed_columns": sum(1 for d in domains.values() if "values" in d),
             "params_with_values": 0, "items_with_combinations": 0}
    combo_cache: dict[tuple, list | None] = {}
    scoped_cache: dict[tuple, list | None] = {}
    types = {c["name"]: c.get("type") or "text" for c in cols.values()}
    for _sec, it in items(meta):
        fixed = _fixed_where(it, set(cols))
        fkey = json.dumps(fixed, sort_keys=True, default=str)
        for p in it.get("params") or []:
            p.pop("allowed_values", None)
            p.pop("range", None)
            d = domains.get(p.get("column") or "")
            if d and "values" in d:
                vals = d["values"]
                if fixed:          # the item's fixed conditions (population) narrow the domain
                    k = (p["column"], fkey)
                    if k not in scoped_cache:
                        scoped_cache[k] = await _scoped_values(session, table, types, p["column"], fixed,
                                                               max_values)
                    vals = scoped_cache[k] if scoped_cache[k] is not None else vals
                p["allowed_values"] = vals
                stats["params_with_values"] += 1
            elif d and "min" in d:
                p["range"] = [d["min"], d["max"]]
        it.pop("allowed_combinations", None)
        eq = {k: v for k, v in _eq_param_columns(it).items()
              if "values" in domains.get(v, {}) and len(domains[v]["values"]) > 1}
        if len(eq) < 2:
            continue
        names = list(eq)
        key = (tuple(eq[n] for n in names), fkey)
        if key not in combo_cache:
            where = _fixed_sql(table, types, fixed)
            rows = (await session.execute(sql_text(
                f"SELECT DISTINCT {', '.join(_q(c) for c in key[0])} FROM graphrag.{_q(table)}{where} "
                f"ORDER BY {', '.join(str(i + 1) for i in range(len(key[0])))} LIMIT {max_combos + 1}"
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


# ------------------------------------------------------------ stored-query calls

_BOOL_WORDS = {"true": True, "t": True, "yes": True, "y": True, "1": True,
               "false": False, "f": False, "no": False, "n": False, "0": False}


def _type_ok(kind: str, v: Any) -> bool:
    """Whether ``v`` can be bound as ``kind`` without guessing."""
    from backend.app.services.structured.sql_exec import _coerce_one

    k = (kind or "text").lower()
    if k.endswith("[]"):
        vals = v if isinstance(v, list) else [v]
        return bool(vals) and all(_type_ok(k[:-2], x) for x in vals)
    if isinstance(v, (dict, list)):
        return False
    if k in {"boolean", "bool"}:
        return isinstance(v, bool) or str(v).strip().lower() in _BOOL_WORDS
    if isinstance(v, bool) and k != "text":
        return False
    out = _coerce_one(k, v)
    if k in {"integer", "int", "bigint", "smallint"}:
        return isinstance(out, int) and not isinstance(out, bool)
    if k in {"numeric", "decimal", "float", "double", "real", "number"}:
        return isinstance(out, Decimal)
    if k == "date":
        return isinstance(out, date)
    if k in {"timestamp", "timestamptz", "datetime"}:
        return isinstance(out, datetime)
    return True


def validate_call_params(params: dict[str, Any] | None, declared: list[dict] | None,
                         combinations: dict | None = None,
                         ) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Check a selected stored query's parameter values against its DECLARED
    parameters before anything runs. Returns (params to bind, problems,
    substitutions); the call may run only when ``problems`` is empty.

    * a name the query does not declare is refused (never silently dropped);
    * a required parameter must have a value (an optional one may be left out:
      its filter then covers every value, or the total row where the table
      has one -- the behaviour compiled into the stored SQL);
    * each value must bind as the declared type without guessing;
    * then the column's valid values and the valid combinations
      (:func:`check_params`)."""
    declared = [p for p in declared or [] if isinstance(p, dict) and p.get("name")]
    names = {str(p["name"]) for p in declared}
    given = {str(k).lstrip(":"): v for k, v in (params or {}).items()}
    problems: list[dict] = []
    for k in sorted(set(given) - names):
        problems.append({"param": k, "error": "not a parameter of this stored query",
                         "declared": sorted(names)})
    clean = {k: v for k, v in given.items() if k in names}
    for p in declared:
        n = str(p["name"])
        if clean.get(n) is None:
            clean.pop(n, None)
            if not p.get("optional"):
                problems.append({"param": n, "error": "required parameter has no value",
                                 "column": p.get("column"), "type": p.get("type")})
            continue
        if not _type_ok(str(p.get("type") or "text"), clean[n]):
            problems.append({"param": n, "value": clean[n], "column": p.get("column"),
                             "error": f"value is not a valid {p.get('type')}"})
    if problems:
        return clean, problems, []
    return check_params(clean, declared, combinations)

"""Units of a SQL result's output columns (no LLM).

The answer model reads a SQL result as column names and values. The unit of
a value lives in the table metadata (a column's ``unit``), in the data (a
column with role ``unit`` that holds each row's unit, e.g. one unit per
indicator), or in a measure's definition -- never in the result itself. This
module resolves, for the ACTUAL statement that ran (the stored SQL, listified
or reshaped), what unit each output column carries:

  known          one unit for the whole column ("USD"; a SUM/AVG/MIN/MAX of
                 USD values is USD)
  dimensionless  a pure number: a COUNT, a ratio of two values in the same
                 unit, a numeric literal
  unknown        the table states no unit for the source column -- NOT the
                 same as dimensionless
  per_row        the unit differs by row and the result carries it in one of
                 its own columns
  mapped         the unit differs by row; the program looked it up for each
                 row from a column of the result that determines it
                 (``row_units``)
  mixed          the unit differs by row and the rows do not say which
  na             not a quantity (a name, code or period): no unit is reported

Varying units are resolved from the data: the unit column is read for the
values the query selected (its equality filters and parameters, or the
determining column's values on the returned rows). The lookups are plain
read-only SELECTs through :func:`sql_exec.run_sql` (the guard applies);
retrieval itself still runs only the stored, verified query.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

KNOWN, DIMENSIONLESS, UNKNOWN = "known", "dimensionless", "unknown"
VARIES, PER_ROW, MAPPED, MIXED, NA = "varies", "per_row", "mapped", "mixed", "na"

# A column 'unit' text that says the unit is not one value ("varies by
# indicator (see unit)", "depends on qtyunitabbr (kg or items)").
_VARIES_TEXT = re.compile(r"\b(vary|varies|varying|depends?|depending|see|per row|mixed|differs?)\b", re.I)
MAX_LISTED_UNITS = 12


def _cols(meta: dict) -> dict[str, dict]:
    return {c["name"]: c for c in (meta or {}).get("columns") or [] if c.get("name")}


def _roles(c: dict) -> set[str]:
    return {str(r).lower() for r in (c.get("roles") or [])} | {str(c.get("role") or "").lower()}


def unit_column_for(meta: dict, column: dict) -> str | None:
    """The column holding ``column``'s unit row by row: a column with role
    'unit' that the column's unit text or description names -- or, when its
    unit text says the unit varies without naming one, the only unit column
    the table has. A column that states no unit is not given another
    column's unit (its unit is unknown)."""
    cands = [n for n, c in _cols(meta).items() if "unit" in _roles(c) and n != column.get("name")]
    if not cands:
        return None
    text = f"{column.get('unit') or ''} {column.get('description') or ''}".lower()
    named = [n for n in cands if re.search(rf"\b{re.escape(n.lower())}\b", text)]
    if len(named) == 1:
        return named[0]
    says_varies = bool(_VARIES_TEXT.search(str(column.get("unit") or "")))
    return cands[0] if len(cands) == 1 and says_varies else None


def _determinants(meta: dict, unit_col: str) -> list[str]:
    """Columns whose value fixes ``unit_col`` (measured functional dependencies)."""
    deps = ((meta or {}).get("analysis") or {}).get("dependencies") or {}
    return sorted(d for d, fixed in deps.items() if unit_col in (fixed or []) and d != unit_col)


def base_column_unit(meta: dict, table: str, name: str, measure: dict | None = None) -> dict:
    """Unit info of a table column (before any aggregation)."""
    c = _cols(meta).get(name)
    if c is None:
        return {"status": UNKNOWN, "table": table, "column": name}
    base = {"table": table, "column": name}
    if "unit" in _roles(c):
        return {"status": NA, **base, "is_unit_column": True}
    is_quantity = str(c.get("role") or "").lower() == "measure" or "quantity" in _roles(c)
    if not is_quantity:
        return {"status": NA, **base}
    text = str(c.get("unit") or "").strip()
    ucol = unit_column_for(meta, c)
    measure_unit = None
    if measure and measure.get("value_column") == name and str(measure.get("unit") or "").strip():
        measure_unit = str(measure["unit"]).strip()
    if ucol and (not text or _VARIES_TEXT.search(text)):
        return {"status": VARIES, **base, "unit_column": ucol, "determined_by": _determinants(meta, ucol),
                **({"measure_unit": measure_unit} if measure_unit else {})}
    if text and not _VARIES_TEXT.search(text):
        return {"status": KNOWN, **base, "unit": text, "source": "column metadata"}
    if measure_unit:
        return {"status": KNOWN, **base, "unit": measure_unit, "source": "measure definition"}
    return {"status": UNKNOWN, **base, **({"note": text} if text else {})}


# ---------------------------------------------------------------- the statement

def _from(sel) -> Any:
    return sel.args.get("from") or sel.args.get("from_")


class _Scope:
    """Sources of one SELECT: base tables and derived (subquery) outputs."""

    def __init__(self, sel, metas: dict[str, dict], measure: dict | None):
        from sqlglot import exp

        self.sources: list[tuple[str, str, Any]] = []      # (alias, kind, payload)
        nodes = []
        f = _from(sel)
        if f is not None:
            nodes.append(f.this)
        nodes += [j.this for j in sel.args.get("joins") or []]
        for n in nodes:
            if isinstance(n, exp.Subquery) and isinstance(n.this, exp.Select):
                self.sources.append((n.alias_or_name, "derived", resolve_select(n.this, metas, measure)))
            elif isinstance(n, exp.Table):
                self.sources.append((n.alias_or_name, "table", n.name))
        self.metas, self.measure = metas, measure

    def lookup(self, qualifier: str, name: str) -> dict:
        order = ([s for s in self.sources if s[0] == qualifier] if qualifier else []) or self.sources
        for _alias, kind, payload in order:
            if kind == "derived":
                hit = payload.get(name) or next((v for k, v in payload.items() if k.lower() == name.lower()), None)
                if hit is not None:
                    return hit
            else:
                meta = self.metas.get(payload)
                if meta is not None and name in _cols(meta):
                    return base_column_unit(meta, payload, name, self.measure)
        return {"status": UNKNOWN, "column": name}

    def star(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for _alias, kind, payload in self.sources:
            if kind == "derived":
                out.update(payload)
            else:
                meta = self.metas.get(payload) or {}
                for n in _cols(meta):
                    out[n] = base_column_unit(meta, payload, n, self.measure)
        return out


def _same(a: dict, b: dict) -> bool:
    return a.get("status") == KNOWN and b.get("status") == KNOWN and \
        str(a.get("unit")).strip().lower() == str(b.get("unit")).strip().lower()


def _expr(e, scope: _Scope) -> dict:
    from sqlglot import exp

    if isinstance(e, (exp.Alias, exp.Paren, exp.Cast, exp.TryCast, exp.Distinct)):
        inner = e.this if not isinstance(e, exp.Distinct) else (e.expressions[0] if e.expressions else None)
        return _expr(inner, scope) if inner is not None else {"status": UNKNOWN}
    if isinstance(e, exp.Column):
        return scope.lookup(e.table, e.name)
    if isinstance(e, exp.Count):
        return {"status": DIMENSIONLESS, "what": "count"}
    if isinstance(e, (exp.Sum, exp.Avg, exp.Min, exp.Max)):
        inner = _expr(e.this, scope)
        if inner.get("status") in (NA, DIMENSIONLESS):
            return inner
        return {**inner, "via": f"{e.key.lower()} of {inner.get('column') or 'a value'}"}
    if isinstance(e, (exp.Round, exp.Abs, exp.Coalesce, exp.Neg)):
        return _expr(e.this, scope)
    if isinstance(e, exp.Literal):
        return {"status": DIMENSIONLESS, "what": "number"} if not e.is_string else {"status": NA}
    if isinstance(e, exp.Div):
        a, b = _expr(e.this, scope), _expr(e.expression, scope)
        if _same(a, b):
            return {"status": DIMENSIONLESS, "what": f"ratio of two {a['unit']} values"}
        if a.get("status") == KNOWN and b.get("status") == DIMENSIONLESS:
            return a
        if a.get("status") == DIMENSIONLESS and b.get("status") == DIMENSIONLESS:
            return {"status": DIMENSIONLESS, "what": "ratio"}
        if a.get("status") == KNOWN and b.get("status") == KNOWN:
            return {"status": KNOWN, "unit": f"{a['unit']} per {b['unit']}", "source": "ratio"}
        return {"status": UNKNOWN, "what": "ratio of values whose units are not both known"}
    if isinstance(e, exp.Mul):
        a, b = _expr(e.this, scope), _expr(e.expression, scope)
        if a.get("status") == DIMENSIONLESS:
            return b
        if b.get("status") == DIMENSIONLESS:
            return a
        return {"status": UNKNOWN, "what": "product of values"}
    if isinstance(e, (exp.Add, exp.Sub)):
        a, b = _expr(e.this, scope), _expr(e.expression, scope)
        if _same(a, b):
            return a
        if a.get("status") == DIMENSIONLESS and b.get("status") == DIMENSIONLESS:
            return a
        return {"status": UNKNOWN, "what": "sum/difference of values whose units differ or are unknown"}
    if e is not None and e.find(exp.Column) is None and e.find(exp.AggFunc) is None:
        return {"status": NA}
    return {"status": UNKNOWN}


def resolve_select(sel, metas: dict[str, dict], measure: dict | None = None) -> dict[str, dict]:
    """{output column: unit info} of one parsed SELECT (follows subqueries)."""
    from sqlglot import exp

    scope = _Scope(sel, metas, measure)
    out: dict[str, dict] = {}
    for s in sel.selects:
        if isinstance(s, exp.Star) or (isinstance(s, exp.Column) and isinstance(s.this, exp.Star)):
            out.update(scope.star())
            continue
        out[s.alias_or_name] = _expr(s, scope)
    return out


def static_units(sql: str, metas: dict[str, dict], measure: dict | None = None) -> dict[str, dict]:
    """Unit info per output column from the statement and the metadata alone
    (varying units stay ``varies``; :func:`result_units` resolves them)."""
    from backend.app.services.structured.sql_guard import parse_one_select

    return resolve_select(parse_one_select(sql), metas, measure)


# ---------------------------------------------------------------- varying units, from the data

def _equality_values(sql: str, params: dict[str, Any], table: str, columns: set[str]) -> dict[str, list]:
    """{column: values} the statement fixes by equality (literal, parameter or
    = ANY(list parameter)) anywhere in it, for columns of ``table``."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import parse_one_select

    found: dict[str, list] = {}

    def value_of(node) -> list | None:
        while isinstance(node, (exp.Cast, exp.Paren, exp.TryCast)):
            node = node.this
        if isinstance(node, exp.Any):
            return value_of(node.this)
        if isinstance(node, exp.Literal):
            return [node.this]
        if isinstance(node, exp.Placeholder):
            v = (params or {}).get(node.name)
            if v is None:
                return None
            return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
        return None

    for eq in parse_one_select(sql).find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if not isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left, right = right, left
        if not isinstance(left, exp.Column) or left.name not in columns:
            continue
        if left.table and left.table != table:
            continue
        vals = value_of(right)
        if vals:
            found.setdefault(left.name, [])
            found[left.name] += [v for v in vals if v not in found[left.name]]
    return found


RunFn = Callable[[str, dict[str, Any], set[str]], Awaitable[list[list[Any]] | None]]


async def _distinct(run: RunFn, table: str, select: list[str], where_col: str | None,
                    values: list | None) -> list[list[Any]] | None:
    cols = ", ".join(f'CAST("{c}" AS text)' for c in select)
    sql = f'SELECT DISTINCT {cols} FROM "{table}"'
    params: dict[str, Any] = {}
    if where_col:
        sql += f' WHERE CAST("{where_col}" AS text) = ANY(CAST(:vals AS text[]))'
        params["vals"] = [str(v) for v in values or []]
    sql += f" LIMIT {MAX_LISTED_UNITS * 50}"
    return await run(sql, params, {table})


async def result_units(sql: str, params: dict[str, Any], columns: list[str], rows: list[list[Any]],
                       metas: dict[str, dict], *, measure: dict | None = None,
                       run: RunFn | None = None) -> dict[str, Any]:
    """{"columns": {output: info}, "row_units": {output: [unit per row]}}.
    ``run(sql, params, tables) -> rows | None`` executes the read-only unit
    lookups (None: lookups skipped, varying units stay unresolved)."""
    infos = static_units(sql, metas, measure)
    by_lower = {c.lower(): i for i, c in enumerate(columns)}
    out: dict[str, dict] = {c: infos.get(c) or next((v for k, v in infos.items() if k.lower() == c.lower()),
                                                    {"status": UNKNOWN}) for c in columns}
    row_units: dict[str, list] = {}
    for name, info in list(out.items()):
        if info.get("status") != VARIES:
            continue
        table, ucol, dets = info.get("table"), info["unit_column"], info.get("determined_by") or []
        # 1. the result carries the unit column itself
        carried = next((c for c, i in out.items()
                        if i.get("table") == table and i.get("column") == ucol), None)
        if carried:
            out[name] = {**info, "status": PER_ROW, "unit_output": carried}
            continue
        # 2. the result carries a column that determines the unit: map each row
        det_out = next(((c, i["column"]) for c, i in out.items()
                        if i.get("table") == table and i.get("column") in dets), None)
        if det_out and run is not None and rows:
            oc, dcol = det_out
            idx = by_lower[oc.lower()]
            vals = sorted({str(r[idx]) for r in rows if r[idx] is not None})
            pairs = await _distinct(run, table, [dcol, ucol], dcol, vals)
            if pairs is not None:
                m: dict[str, set] = {}
                for d, u in pairs:
                    m.setdefault(str(d), set()).add(u)
                if vals and all(len(m.get(v, ())) == 1 for v in vals):
                    per = [next(iter(m[str(r[idx])])) if r[idx] is not None else None for r in rows]
                    distinct = sorted({u for u in per if u is not None})
                    if len(distinct) == 1:
                        out[name] = {**info, "status": KNOWN, "unit": distinct[0],
                                     "source": f"column {ucol} for the selected {dcol}"}
                    else:
                        out[name] = {**info, "status": MAPPED, "by": oc}
                        row_units[name] = per
                    continue
        # 3. the statement fixes the unit, or a column that determines it
        fixed = _equality_values(sql, params, table, {ucol, *dets}) if table else {}
        units: list | None = None
        how = ""
        if fixed.get(ucol):
            units, how = fixed[ucol], f"the query's filter on {ucol}"
        else:
            d = next((d for d in dets if fixed.get(d)), None)
            if d and run is not None:
                got = await _distinct(run, table, [ucol], d, fixed[d])
                if got is not None:
                    units = sorted({r[0] for r in got if r and r[0] is not None})
                    how = f"column {ucol} for {d} = {', '.join(map(str, fixed[d]))}"
        if units and len(units) == 1:
            out[name] = {**info, "status": KNOWN, "unit": units[0], "source": how}
            continue
        if not units and info.get("measure_unit"):
            out[name] = {**info, "status": KNOWN, "unit": info["measure_unit"], "source": "measure definition"}
            continue
        if not units and run is not None:
            got = await _distinct(run, table, [ucol], None, None)
            units = sorted({r[0] for r in got or [] if r and r[0] is not None})
        out[name] = {**info, "status": MIXED, "units": (units or [])[:MAX_LISTED_UNITS]}
    return {"columns": out, "row_units": row_units}


# ---------------------------------------------------------------- text for the answer prompt

def _describe(name: str, info: dict) -> str | None:
    st = info.get("status")
    via = f" ({info['via']})" if info.get("via") else ""
    if st == KNOWN:
        src = f"; from {info['source']}" if info.get("source") not in (None, "column metadata") else ""
        return f"{name} = {info['unit']}{via}{src}"
    if st == DIMENSIONLESS:
        return f"{name} = dimensionless ({info.get('what') or 'a pure number'})"
    if st == UNKNOWN:
        return f"{name} = unit unknown (the table does not state one{'' if not info.get('note') else ': ' + info['note']})"
    if st == PER_ROW:
        return f"{name} = unit differs by row; each row's unit is in its {info['unit_output']} column"
    if st == MAPPED:
        return f"{name} = unit differs by row; shown per row as {name}_unit (looked up from {info['by']})"
    if st == MIXED:
        listed = " | ".join(map(str, info.get("units") or [])) or "not determinable"
        return (f"{name} = unit differs by row ({listed}) and these rows do not say which unit each "
                "value has -- do not state a unit for them")
    return None


def units_line(units: dict | None) -> str:
    """'Units: a = USD (sum of primaryvalue); n = dimensionless (count).' or ''."""
    parts = [p for n, i in ((units or {}).get("columns") or {}).items() if (p := _describe(n, i))]
    return ("Units: " + "; ".join(parts) + ". ") if parts else ""

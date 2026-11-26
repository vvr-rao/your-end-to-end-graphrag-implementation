"""Bounded changes retrieval may make to a STORED query (it never writes SQL).

The stored SQL stays the verified definition of which rows a query reads.
Retrieval may only

* **list values** -- give a parameter that filters by equality
  (``col = CAST(:p AS text)``) a LIST of values; the program rewrites that one
  comparison to ``col = ANY(CAST(:p AS text[]))`` (and its optional
  ``CAST(:p AS text) IS NULL`` guard to the array form). A parameter used any
  other way (a range bound, a pattern, a selected value) cannot take a list.
* **reshape the result** -- aggregate numeric output columns (max, min, avg,
  count), group by categorical output columns, order by numeric output
  columns, and limit the rows. The program wraps the stored statement as a
  subquery: ``SELECT ... FROM (<stored sql>) AS base GROUP BY ... ORDER BY
  ... LIMIT n``. The rows read are exactly the stored query's rows.

sqlglot is used only to CHECK the statement (regenerating SQL with it rewrites
``:param`` placeholders); the rewrite itself is textual on the program-written
stored SQL.
"""

from __future__ import annotations

import re
from typing import Any

MAX_LIST_VALUES = 25
AGG_FUNCS = {"max": "MAX", "min": "MIN", "avg": "AVG", "count": "COUNT"}
NUMERIC_TYPES = {"integer", "int", "bigint", "smallint", "numeric", "decimal", "float",
                 "double", "double precision", "real", "number"}
TIME_ROLES = {"time", "period", "date"}


class ReshapeError(ValueError):
    """The requested change is not one retrieval may make."""


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------- list values

def listify(sql: str, params: dict[str, Any], declared: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Rewrite equality filters whose param was given a list of values.

    Returns (sql, declared params with list types, names of rewritten params).
    A one-value list becomes that scalar. Raises ReshapeError when a listed
    param is not used only as an equality filter (or its optional guard)."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import SqlGuardError, parse_one_select

    lists = {k: v for k, v in (params or {}).items() if isinstance(v, list)}
    kinds = {str(p.get("name")): str(p.get("type") or "text") for p in declared or []}
    todo = [k for k in lists if k in kinds and not kinds[k].endswith("[]")]
    if not todo:
        return sql, declared, []
    try:
        root = parse_one_select(sql)
    except SqlGuardError as exc:
        raise ReshapeError(f"stored query does not parse: {exc}") from exc
    done: list[str] = []
    new_declared = [dict(p) for p in declared]
    for name in todo:
        vals = lists[name]
        if not vals:
            raise ReshapeError(f"param {name}: an empty list")
        if len(vals) > MAX_LIST_VALUES:
            raise ReshapeError(f"param {name}: at most {MAX_LIST_VALUES} values in one list")
        if len(vals) == 1:
            params[name] = vals[0]
            continue
        uses = [ph for ph in root.find_all(exp.Placeholder) if ph.name == name]
        for ph in uses:
            node = ph.parent if isinstance(ph.parent, exp.Cast) else ph
            parent = node.parent
            ok = (isinstance(parent, exp.EQ) and isinstance(
                parent.this if parent.expression is node else parent.expression, exp.Column)) or (
                isinstance(parent, exp.Is) and isinstance(parent.expression, exp.Null))
            if not ok:
                raise ReshapeError(
                    f"param {name} is not an equality filter (it is used in a "
                    f"{type(parent).__name__}), so it cannot take a list of values")
        kind = kinds[name]
        cast = re.compile(r"CAST\(\s*:" + re.escape(name) + r"\s+AS\s+([A-Za-z ]+?)\s*\)", re.I)
        if not cast.search(sql):
            raise ReshapeError(f"param {name} is not bound with a type cast; cannot take a list")
        sql = cast.sub(lambda m, n=name: f"CAST(:{n} AS {m.group(1)}[])", sql)
        arr = re.escape(f"CAST(:{name} AS ") + r"[A-Za-z ]+?\[\]\)"
        sql = re.sub(r"=\s*(" + arr + r")", r"= ANY(\1)", sql)
        for p in new_declared:
            if p.get("name") == name:
                p["type"] = f"{kind}[]"
        done.append(name)
    if done:
        try:
            parse_one_select(sql)
        except SqlGuardError as exc:   # never run a rewrite that does not parse
            raise ReshapeError(f"list rewrite failed to parse: {exc}") from exc
    return sql, new_declared, done


# ------------------------------------------------------------- reshape result

def output_columns(sql: str, col_types: dict[str, str], roles: dict[str, str]) -> dict[str, str]:
    """{output column: 'numeric' | 'time' | 'categorical'} of a stored query."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import parse_one_select

    out: dict[str, str] = {}
    for s in parse_one_select(sql).selects:
        name = s.alias_or_name
        inner = s.this if isinstance(s, exp.Alias) else s
        if inner.find(exp.AggFunc) is not None and not isinstance(inner, exp.Column):
            out[name] = "numeric"
            continue
        base = inner.name if isinstance(inner, exp.Column) else name
        role = (roles.get(base) or "").lower()
        typ = (col_types.get(base) or "text").lower()
        if role in TIME_ROLES:
            out[name] = "time"
        elif typ in NUMERIC_TYPES or role == "measure":
            out[name] = "numeric"
        else:
            out[name] = "categorical"
    return out


def reshape_sql(sql: str, spec: dict[str, Any], kinds: dict[str, str], *, max_rows: int) -> tuple[str, str]:
    """Wrap a stored statement. Returns (sql, one-line description).

    spec: {"aggregate": [{"fn": max|min|avg|count, "column": c}],
           "group_by": [c, ...], "order_by": [{"column": c, "dir": asc|desc}],
           "limit": n}"""
    if not isinstance(spec, dict) or not spec:
        raise ReshapeError("reshape must be a JSON object")
    unknown = set(spec) - {"aggregate", "group_by", "order_by", "limit"}
    if unknown:
        raise ReshapeError(f"reshape keys not allowed: {sorted(unknown)} "
                           "(allowed: aggregate, group_by, order_by, limit)")

    def _col(c: Any, what: str) -> str:
        if not isinstance(c, str) or c not in kinds:
            raise ReshapeError(f"{what}: {c!r} is not an output column of this query "
                               f"(output columns: {', '.join(kinds)})")
        return c

    aggs, names = [], []
    for a in spec.get("aggregate") or []:
        if not isinstance(a, dict) or str(a.get("fn", "")).lower() not in AGG_FUNCS:
            raise ReshapeError(f"aggregate {a!r}: fn must be one of {sorted(AGG_FUNCS)}")
        fn = str(a["fn"]).lower()
        if fn == "count" and not a.get("column"):
            aggs.append("COUNT(*) AS \"count_rows\"")
            names.append("count_rows")
            continue
        c = _col(a.get("column"), f"aggregate {fn}")
        if fn != "count" and kinds[c] != "numeric":
            raise ReshapeError(f"aggregate {fn}: {c!r} is not a numeric column")
        alias = f"{fn}_{c}"
        aggs.append(f"{AGG_FUNCS[fn]}(base.{_q(c)}) AS {_q(alias)}")
        names.append(alias)
    groups = [_col(g, "group_by") for g in spec.get("group_by") or []]
    for g in groups:
        if kinds[g] == "numeric":
            raise ReshapeError(f"group_by: {g!r} is numeric; group by categorical or period columns")
    if groups and not aggs:
        aggs, names = ["COUNT(*) AS \"count_rows\""], ["count_rows"]
    orderable = ({n: "numeric" for n in names} | {g: kinds[g] for g in groups}) if aggs else kinds
    order = []
    for o in spec.get("order_by") or []:
        col = o.get("column") if isinstance(o, dict) else o
        if col not in orderable:
            raise ReshapeError(f"order_by: {col!r} is not a column of the reshaped result "
                               f"({', '.join(orderable)})")
        if orderable[col] == "categorical":
            raise ReshapeError(f"order_by: {col!r} is categorical; order by numeric or period columns")
        d = str((o.get("dir") if isinstance(o, dict) else None) or "asc").lower()
        if d not in ("asc", "desc"):
            raise ReshapeError(f"order_by dir must be asc or desc, not {d!r}")
        ref = _q(col) if aggs else f"base.{_q(col)}"
        order.append(f"{ref} {d.upper()} NULLS LAST")
    lim = spec.get("limit")
    if lim is not None:
        if isinstance(lim, bool) or not isinstance(lim, int) or not 1 <= lim <= max_rows:
            raise ReshapeError(f"limit must be an integer from 1 to {max_rows}")
    if not (aggs or order or lim):
        raise ReshapeError("reshape changes nothing")
    body = sql.strip().rstrip(";")
    select = ", ".join([f"base.{_q(g)}" for g in groups] + aggs) if aggs else "base.*"
    out = f"SELECT {select} FROM ({body}) AS base"
    if groups:
        out += " GROUP BY " + ", ".join(f"base.{_q(g)}" for g in groups)
    if order:
        out += " ORDER BY " + ", ".join(order)
    if lim:
        out += f" LIMIT {lim}"
    desc = []
    if aggs:
        desc.append(", ".join(names) + (" per " + ", ".join(groups) if groups else " over all rows"))
    if order:
        parts = []
        for o in spec.get("order_by") or []:
            col = o.get("column") if isinstance(o, dict) else o
            d = (o.get("dir") if isinstance(o, dict) else None) or "asc"
            parts.append(f"{col} {d}")
        desc.append("ordered by " + ", ".join(parts))
    if lim:
        desc.append(f"first {lim} row(s)")
    return out, "; ".join(desc)

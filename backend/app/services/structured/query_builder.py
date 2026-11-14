"""Deterministic SQL from a JSON QUERY SPEC (no LLM writes SQL text).

Every stored metric/filter/query/lookup keeps its query as a JSON spec plus a
separate parameter list; this module turns the spec into one Postgres SELECT.
The same spec always gives the same SQL. The retrieval model only picks a
stored spec and fills its parameter values; when no stored spec fits it may
return a NEW spec in the same JSON shape, which is compiled here too.

Spec shape::

    {
      "from":     "table",               # optional; defaults to the item's table
      "select":   [COL | {"agg": "sum|avg|min|max|count|count_distinct",
                            "column": COL (omit for count(*)), "as": "alias"}],
      "distinct": false,
      "distinct_on": [COL],              # one row per value (first by order_by)
      "join":     {"table": "other", "on": [{"column": "a", "other_column": "b"}]},
      "where":    [{"column": COL, "op": OP, "param": "name" | ["from", "to"]}
                   | {"column": COL, "op": OP, "value": literal}],
                                         # "optional": true on a single-param filter: a null /
                                         # omitted value drops the filter (every value)
      "group_by": [COL],                 # optional: derived from select
      "order_by": [{"column": COL | "alias", "dir": "asc|desc"}],
      "limit":    10 | {"param": "n"}
    }

    COL = "name" | "table.name" | {"column": "name", "table": "t", "fn": "year|month|quarter|lower"}
    OP  = = != < <= > >= between in contains ilike is_null not_null

Parameter types are taken from the column the parameter is compared with,
and every parameter is bound (``:name``) and cast, never pasted into the SQL.
All identifiers are double-quoted, so a column called ``end`` just works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from backend.app.services.structured.profiling import PG_TYPES
from backend.app.services.structured.sql_guard import PG_RESERVED

AGGS = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT",
        "count_distinct": "COUNT"}
FNS = {"year": "integer", "month": "integer", "quarter": "integer", "lower": "text"}
COMPARE = {"=": "=", "!=": "<>", "<>": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
OPS = set(COMPARE) | {"between", "in", "contains", "ilike", "is_null", "not_null"}
MAX_LIMIT = 1000
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_SPEC_KEYS = {"from", "select", "distinct", "distinct_on", "join", "where", "group_by", "order_by", "limit"}


class QuerySpecError(ValueError):
    """The spec cannot be compiled (unknown column, bad operator, ...)."""


@dataclass
class Built:
    sql: str
    params: list[dict] = field(default_factory=list)   # [{name, type, column}] in order of use
    tables: set[str] = field(default_factory=set)


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if v is None:
        raise QuerySpecError("use op is_null instead of value null")
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return "'" + s.replace("'", "''") + "'"


class _Ctx:
    def __init__(self, table: str, columns: dict[str, dict[str, str]], join: dict | None):
        if table not in columns:
            raise QuerySpecError(f"unknown table '{table}'")
        self.base = table
        self.columns = columns
        self.other: str | None = None
        if join:
            other = str(join.get("table") or "")
            if other not in columns:
                raise QuerySpecError(f"join table '{other}' is not loaded")
            if other == table:
                raise QuerySpecError("a table cannot join itself")
            self.other = other
        self.params: dict[str, dict] = {}
        self._seen: set[str] = set()
        self.aliases: set[str] = set()
        self.fn_alias: dict[str, str] = {}     # raw column sql -> alias of fn(column) in select

    @property
    def tables(self) -> list[str]:
        return [self.base] + ([self.other] if self.other else [])

    def ref(self, raw) -> tuple[str, str, str, str]:
        """COL -> (sql, type, table, output name)."""
        fn = None
        if isinstance(raw, dict):
            name, table, fn = raw.get("column"), raw.get("table"), raw.get("fn")
        else:
            name, table = str(raw or ""), None
            if "." in name:
                table, name = name.split(".", 1)
        name = str(name or "").strip()
        if not name:
            raise QuerySpecError("column reference without a column name")
        if table is not None:
            table = str(table)
            if table not in self.tables:
                raise QuerySpecError(f"column {table}.{name}: table '{table}' is not part of this query")
            if name not in self.columns[table]:
                raise QuerySpecError(f"unknown column {table}.{name}")
        else:
            owners = [t for t in self.tables if name in self.columns[t]]
            if not owners:
                raise QuerySpecError(f"unknown column '{name}' (columns: "
                                     f"{', '.join(sorted(self.columns[self.base]))})")
            table = owners[0]   # the base table wins when both have it
        ctype = self.columns[table][name]
        sql = f"{_q(table)}.{_q(name)}" if self.other else _q(name)
        out = name
        if fn:
            if fn not in FNS:
                raise QuerySpecError(f"fn must be one of {', '.join(FNS)} (got {fn!r})")
            if fn == "lower":
                sql, ctype = f"lower({sql})", "text"
            elif ctype in {"date", "timestamp"}:
                sql, ctype = f"CAST(EXTRACT({fn.upper()} FROM {sql}) AS integer)", "integer"
            elif fn == "year" and ctype in {"text", "integer"}:
                # Text periods come as '2024', '2024-03', '2024M03', '2024Q1':
                # the year is always the first four characters.
                sql, ctype = f"CAST(substr(CAST({sql} AS text), 1, 4) AS integer)", "integer"
            else:
                raise QuerySpecError(f"fn {fn} needs a date column ('{name}' is {ctype}); "
                                     "filter the period text with between instead")
            out = f"{fn}_{name}"
        return sql, ctype, table, out

    def param(self, name, ptype: str, column: str, *, optional: bool = False) -> str:
        name = str(name or "").lstrip(":")
        if not _NAME.match(name):
            raise QuerySpecError(f"bad parameter name {name!r} (use snake_case)")
        if name.lower() in PG_RESERVED:
            raise QuerySpecError(f"parameter name {name!r} is an SQL key word; "
                                 f"use e.g. {name}_{column or 'value'}")
        prev = self.params.get(name)
        if prev and prev["type"] != ptype:
            raise QuerySpecError(f"parameter '{name}' is used as {prev['type']} and as {ptype}")
        self.params.setdefault(name, {"name": name, "type": ptype, "column": column})
        if not optional:
            self.params[name].pop("optional", None)      # required anywhere -> required
        elif name not in self._seen:
            self.params[name]["optional"] = True
        self._seen.add(name)
        base = ptype[:-2] if ptype.endswith("[]") else ptype
        pg = PG_TYPES.get(base, "text") + ("[]" if ptype.endswith("[]") else "")
        return f"CAST(:{name} AS {pg})"


def _select_item(ctx: _Ctx, raw) -> tuple[str, str, bool, str | None]:
    """-> (sql, output name, is_aggregate, group-by expression)."""
    if isinstance(raw, dict) and raw.get("agg"):
        agg = str(raw["agg"]).lower()
        if agg not in AGGS:
            raise QuerySpecError(f"agg must be one of {', '.join(AGGS)} (got {agg!r})")
        alias = str(raw.get("as") or "").strip()
        if agg == "count_distinct" and isinstance(raw.get("columns"), list) and raw["columns"]:
            refs = [ctx.ref(c) for c in raw["columns"]]
            inner = ", ".join(r[0] for r in refs)
            expr = f"COUNT(DISTINCT ({inner}))" if len(refs) > 1 else f"COUNT(DISTINCT {inner})"
            alias = alias or "count_distinct_" + "_".join(r[3] for r in refs)
        elif raw.get("column") in (None, "", "*"):
            if agg != "count":
                raise QuerySpecError(f"{agg} needs a column")
            expr, alias = "COUNT(*)", alias or "row_count"
        else:
            sql, ctype, _t, out = ctx.ref(raw["column"])
            if agg in {"sum", "avg"} and ctype not in {"integer", "numeric"}:
                sql = f"CAST({sql} AS numeric)"
            inner = f"DISTINCT {sql}" if agg == "count_distinct" else sql
            expr, alias = f"{AGGS[agg]}({inner})", alias or f"{agg}_{out}"
        if not _NAME.match(alias):
            raise QuerySpecError(f"bad alias {alias!r} (use snake_case)")
        return f"{expr} AS {_q(alias)}", alias, True, None
    sql, _ct, _t, out = ctx.ref(raw)
    if isinstance(raw, dict) and raw.get("fn"):
        base = ctx.ref({k: v for k, v in raw.items() if k not in {"fn", "as"}})[0]
        ctx.fn_alias.setdefault(base, str(raw.get("as") or "").strip() or out)
    alias = str(raw.get("as") or "").strip() if isinstance(raw, dict) else ""
    if alias and not _NAME.match(alias):
        raise QuerySpecError(f"bad alias {alias!r} (use snake_case)")
    name = alias or out
    needs_alias = bool(alias) or "(" in sql      # fn applied: name the output
    return (f"{sql} AS {_q(name)}" if needs_alias else sql), name, False, sql


def _where(ctx: _Ctx, cond: dict) -> str:
    if not isinstance(cond, dict):
        raise QuerySpecError("each where entry must be an object")
    op = str(cond.get("op") or "=").lower()
    if op not in OPS:
        raise QuerySpecError(f"op must be one of {', '.join(sorted(OPS))} (got {op!r})")
    sql, ctype, _t, out = ctx.ref(cond.get("column"))
    if op in {"is_null", "not_null"}:
        return f"{sql} IS {'NOT ' if op == 'not_null' else ''}NULL"
    has_param, has_value = cond.get("param") not in (None, ""), "value" in cond
    if has_param == has_value:
        raise QuerySpecError(f"where on '{out}': give exactly one of param or value")
    if op in {"contains", "ilike"}:
        rhs = ctx.param(cond["param"], "text", out) if has_param else _lit(cond["value"])
        lhs = sql if ctype == "text" else f"CAST({sql} AS text)"
        if op == "contains":
            return f"{lhs} ILIKE '%' || {rhs} || '%'"
        return f"{lhs} ILIKE {rhs}"
    if op == "between":
        pair = cond.get("param") if has_param else cond.get("value")
        if not isinstance(pair, list) or len(pair) != 2:
            raise QuerySpecError(f"between on '{out}' needs two params, e.g. [\"start\", \"end\"]")
        lo, hi = ((ctx.param(p, ctype, out) for p in pair) if has_param else (_lit(v) for v in pair))
        return f"{sql} BETWEEN {lo} AND {hi}"
    if op == "in":
        if has_param and cond.get("optional"):
            rhs = ctx.param(cond["param"], ctype + "[]", out, optional=True)
            return f"({rhs} IS NULL OR {sql} = ANY({rhs}))"
        if has_param:
            return f"{sql} = ANY({ctx.param(cond['param'], ctype + '[]', out)})"
        vals = cond["value"] if isinstance(cond["value"], list) else [cond["value"]]
        if not vals:
            raise QuerySpecError(f"in on '{out}' needs at least one value")
        return f"{sql} IN ({', '.join(_lit(v) for v in vals)})"
    if has_param and cond.get("optional"):
        if not isinstance(cond["param"], str) or op not in {"=", "in"}:
            raise QuerySpecError(f"optional on '{out}': only a single '=' or 'in' param can be optional")
    if op == "=" and has_param and cond.get("optional"):
        rhs = ctx.param(cond["param"], ctype, out, optional=True)
        return f"({rhs} IS NULL OR {sql} = {rhs})"
    rhs = ctx.param(cond["param"], ctype, out) if has_param else _lit(cond["value"])
    return f"{sql} {COMPARE[op]} {rhs}"


def build_sql(spec: dict, table: str, columns: dict[str, dict[str, str]], *,
              time_columns: dict[str, list[str]] | None = None) -> Built:
    """Compile ``spec`` (reading ``table``) to SQL. ``columns``: table ->
    {column: type} for every table the spec may read.

    Every aggregate query also returns how much it combined, so a total is
    never mistaken for one value: ``rows_aggregated`` (COUNT(*)) and, for each
    of the table's ``time_columns`` that is not grouped by, ``first_<col>`` /
    ``last_<col>`` (the span of periods the aggregate covers)."""
    if not isinstance(spec, dict):
        raise QuerySpecError("query spec must be a JSON object")
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise QuerySpecError(f"unknown spec key(s): {', '.join(sorted(unknown))}")
    # "from" pins the base table (a lookup stored in its partner's metadata
    # keeps the base it was written for).
    table = str(spec.get("from") or table)
    join = spec.get("join") or None
    ctx = _Ctx(table, columns, join)

    sel = spec.get("select") or []
    if not isinstance(sel, list) or not sel:
        raise QuerySpecError("select must list at least one column or aggregate")
    parts, names, plain_groups, has_agg = [], [], [], False
    for raw in sel:
        s, name, is_agg, grp = _select_item(ctx, raw)
        if name in names:
            raise QuerySpecError(f"output column '{name}' appears twice (use \"as\")")
        parts.append(s)
        names.append(name)
        has_agg |= is_agg
        if grp:
            plain_groups.append(grp)
    if has_agg:
        grouped = {ctx.ref(r)[0] for r in spec.get("group_by") or []} | set(plain_groups)
        extra = []
        if "COUNT(*)" not in " ".join(p.split(" AS ")[0] for p in parts):
            extra.append(("COUNT(*)", "rows_aggregated"))
        base_cols = columns.get(table, {})
        for tcol in [c for c in (time_columns or {}).get(table, []) if c in base_cols][:2]:
            expr = f"{_q(table)}.{_q(tcol)}" if ctx.other else _q(tcol)
            if expr not in grouped:
                extra += [(f"MIN({expr})", f"first_{tcol}"), (f"MAX({expr})", f"last_{tcol}")]
        for expr, alias in extra:
            if alias not in names:
                parts.append(f"{expr} AS {_q(alias)}")
                names.append(alias)
    ctx.aliases = set(names)

    distinct_on = [ctx.ref(r)[0] for r in spec.get("distinct_on") or []]
    if distinct_on and has_agg:
        raise QuerySpecError("distinct_on cannot be combined with aggregates")
    head = "SELECT "
    if distinct_on:
        head += f"DISTINCT ON ({', '.join(distinct_on)}) "
    elif spec.get("distinct"):
        head += "DISTINCT "
    sql = head + ", ".join(parts) + f" FROM {_q(table)}"

    if join:
        on = join.get("on") or []
        if not on:
            raise QuerySpecError("join needs at least one on pair")
        conds = []
        for pair in on:
            a, b = pair.get("column"), pair.get("other_column")
            if a not in columns[table] or b not in columns[ctx.other]:
                raise QuerySpecError(f"join columns {table}.{a} = {ctx.other}.{b} do not exist")
            conds.append(f"{_q(table)}.{_q(a)} = {_q(ctx.other)}.{_q(b)}")
        sql += f" JOIN {_q(ctx.other)} ON " + " AND ".join(conds)

    where = [_where(ctx, c) for c in spec.get("where") or []]
    if where:
        sql += " WHERE " + " AND ".join(where)

    groups: list[str] = []
    if has_agg:
        groups = [ctx.ref(r)[0] for r in spec.get("group_by") or []]
        for g in plain_groups:
            if g not in groups:
                groups.append(g)
        if groups:
            sql += " GROUP BY " + ", ".join(groups)
    elif spec.get("group_by"):
        raise QuerySpecError("group_by without an aggregate in select")

    order = []
    for o in spec.get("order_by") or []:
        o = o if isinstance(o, dict) else {"column": o}
        d = str(o.get("dir") or "asc").lower()
        if d not in {"asc", "desc"}:
            raise QuerySpecError(f"dir must be asc or desc (got {d!r})")
        col = o.get("column")
        if isinstance(col, str) and col in ctx.aliases and "." not in col:
            expr = _q(col)
        else:
            expr = ctx.ref(col)[0]
            if has_agg and expr not in groups:
                if expr in ctx.fn_alias:          # year(date) selected: order by that
                    expr = _q(ctx.fn_alias[expr])
                else:
                    raise QuerySpecError(f"order_by {col!r}: in an aggregate query order by a "
                                         "selected/grouped column or an aggregate alias")
        order.append(f"{expr} {d.upper()}" + (" NULLS LAST" if d == "desc" else ""))
    if distinct_on:
        # Postgres: DISTINCT ON expressions must lead the ORDER BY.
        lead = []
        for e in distinct_on:
            hit = next((x for x in order if x.startswith(e + " ")), None)
            if hit is not None:
                order.remove(hit)
            lead.append(hit or f"{e} ASC")
        order = lead + order
    if order:
        sql += " ORDER BY " + ", ".join(order)

    lim = spec.get("limit")
    if lim not in (None, ""):
        if isinstance(lim, dict) and lim.get("param"):
            sql += f" LIMIT {ctx.param(lim['param'], 'integer', '')}"
        elif isinstance(lim, int) and not isinstance(lim, bool) and 0 < lim <= MAX_LIMIT:
            sql += f" LIMIT {lim}"
        else:
            raise QuerySpecError(f"limit must be 1..{MAX_LIMIT} or {{\"param\": \"n\"}}")
    return Built(sql=sql, params=list(ctx.params.values()), tables=set(ctx.tables))


def merge_params(built: Built, declared: list[dict] | None) -> list[dict]:
    """The parameter list to store: names/types/columns from the compiled
    spec (authoritative), descriptions/examples from what the model wrote."""
    by_name = {str(p.get("name") or "").lstrip(":"): p for p in declared or [] if isinstance(p, dict)}
    out = []
    for p in built.params:
        d = by_name.get(p["name"]) or {}
        ex = d.get("example")
        out.append({"name": p["name"], "type": p["type"], "column": p["column"] or None,
                    "description": str(d.get("description") or ""),
                    "example": None if ex in (None, "") else (ex if isinstance(ex, list) else str(ex)),
                    **({"optional": True} if p.get("optional") else {})})
    return out


def spec_join_for(item: dict) -> dict | None:
    """A lookup item's join (``other_table`` + ``join`` pairs) as a spec join."""
    if item.get("other_table") and item.get("join"):
        return {"table": item["other_table"],
                "on": [{"column": j.get("column"), "other_column": j.get("other_column")}
                       for j in item["join"]]}
    return None


_ORDER = ["from", "select", "distinct", "distinct_on", "join", "where", "group_by", "order_by", "limit",
          "table", "column", "fn", "agg", "op", "param", "optional", "value", "as", "dir", "on", "other_column"]


def canonical(node):
    """Keys in reading order (the database's JSONB reorders them)."""
    if isinstance(node, dict):
        rank = {k: i for i, k in enumerate(_ORDER)}
        return {k: canonical(node[k]) for k in sorted(node, key=lambda k: (rank.get(k, len(rank)), k))}
    if isinstance(node, list):
        return [canonical(v) for v in node]
    return node


def compact(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":"))


def example_value(ptype: str):
    """A harmless value of the right type (EXPLAIN without examples)."""
    base = ptype[:-2] if ptype.endswith("[]") else ptype
    v = {"integer": 1, "numeric": 1, "date": date(2000, 1, 1).isoformat(),
         "timestamp": "2000-01-01T00:00:00", "boolean": True}.get(base, "x")
    return [v] if ptype.endswith("[]") else v

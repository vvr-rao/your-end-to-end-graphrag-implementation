"""Build and run a table query with no LLM (CLI ``table-query``).

Given a loaded table and either a stored item id (metric/filter/query/lookup)
or a JSON query spec, plus parameter values: compile the spec to SQL with
:mod:`query_builder`, then run it through the guard (read-only transaction,
statement timeout, row cap). Retrieval uses the same two steps after the
model has picked a query and its parameter values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.services.structured.metadata import column_types, compile_item, find_item, items
from backend.app.services.structured.query_builder import Built, build_sql
from backend.app.services.structured.sql_exec import SqlResult, run_sql


class QueryRunError(ValueError):
    pass


@dataclass
class QueryRun:
    table: str
    query_id: str | None
    built: Built
    declared: list[dict] = field(default_factory=list)
    result: SqlResult | None = None


def parse_param_args(pairs: list[str] | None) -> dict[str, Any]:
    """``["cik=0000811809", "tags=Revenue,ProfitLoss"]`` -> dict. Values stay
    strings; the declared parameter types cast them (a list type splits on ',')."""
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise QueryRunError(f"--param must look like name=value (got {p!r})")
        k, v = p.split("=", 1)
        out[k.strip().lstrip(":")] = v
    return out


async def prepare(table: str, *, query_id: str | None = None, spec: dict | None = None) -> QueryRun:
    """Compile (no database write, no LLM). Raises QueryRunError."""
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import registered_tables

    async with session_scope() as s:
        reg = await registered_tables(s)
    if table not in reg:
        raise QueryRunError(f"table {table!r} is not loaded (loaded: {', '.join(sorted(reg)) or 'none'})")
    meta = reg[table]["meta"]
    types = {n: column_types(r["meta"]) for n, r in reg.items()}
    if (query_id is None) == (spec is None):
        raise QueryRunError("give exactly one of a stored query id or a query spec")
    if query_id is not None:
        hit = find_item(meta, query_id)
        if hit is None:
            ids = ", ".join(it["id"] for _s, it in items(meta))
            raise QueryRunError(f"no stored query {query_id!r} in {table} (ids: {ids})")
        section, item = hit
        built = compile_item(meta, section, item, types)
        declared = item.get("params") if isinstance(item.get("query"), dict) else built.params
        return QueryRun(table=table, query_id=query_id, built=built, declared=declared or [])
    from backend.app.services.structured.metadata import time_columns

    built = build_sql(spec, table, types, time_columns={table: time_columns(meta)})
    return QueryRun(table=table, query_id=None, built=built, declared=built.params)


async def run(table: str, params: dict[str, Any], *, query_id: str | None = None,
              spec: dict | None = None, max_rows: int = 50, timeout_ms: int = 5000) -> QueryRun:
    qr = await prepare(table, query_id=query_id, spec=spec)
    missing = [p["name"] for p in qr.built.params if p["name"] not in params and not p.get("optional")]
    if missing:
        raise QueryRunError(f"missing --param value(s): {', '.join(missing)}")
    qr.result = await run_sql(qr.built.sql, params, qr.built.tables, max_rows=max_rows,
                              timeout_ms=timeout_ms, declared_params=qr.declared)
    return qr

"""Build and run a table query with no LLM (CLI ``table-query``).

Given a loaded table and either a stored item id (metric/filter/query/lookup/
measure) or a JSON query spec, plus parameter values. A stored id runs its
VERIFIED stored SQL exactly as retrieval does (refused, with the rebuild
action, when it is not verified); a spec is an operator diagnostic, compiled
with :mod:`query_builder`. Both run through the guard (read-only transaction,
statement timeout, row cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.services.structured.metadata import column_types, find_item, items
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
    from backend.app.services.structured.analysis import total_markers

    totals = {n: total_markers(r["meta"]) for n, r in reg.items()}
    if (query_id is None) == (spec is None):
        raise QueryRunError("give exactly one of a stored query id or a query spec")
    if query_id is not None:
        hit = find_item(meta, query_id)
        if hit is None:
            ids = ", ".join(it["id"] for _s, it in items(meta))
            raise QueryRunError(f"no stored query {query_id!r} in {table} (ids: {ids})")
        section, item = hit
        # Same as retrieval: the exact SQL ingestion verified, never a recompile.
        from backend.app.services.structured.metadata import item_tables, rebuild_action, selectable

        # a person running one stored query by hand may run one that awaits its
        # semantic review (retrieval may not); the data test still applies
        ok, why = selectable(meta, item, semantic=False)
        if not ok:
            raise QueryRunError(f"stored query {query_id!r} is unavailable: {why}. "
                                f"Rebuild: {rebuild_action(table, why)}")
        built = Built(sql=item["sql"], params=list(item["params"]),
                      tables=set(item.get("tables") or item_tables(meta, section, item)),
                      empty_excluded=list(item.get("empty_excluded") or []),
                      added_conditions=list(item.get("added_conditions") or []))
        return QueryRun(table=table, query_id=query_id, built=built, declared=list(item["params"]))
    from backend.app.services.structured.metadata import time_columns

    built = build_sql(spec, table, types, time_columns={table: time_columns(meta)}, totals=totals)
    return QueryRun(table=table, query_id=None, built=built, declared=built.params)


async def run(table: str, params: dict[str, Any], *, query_id: str | None = None,
              spec: dict | None = None, max_rows: int = 50, timeout_ms: int = 5000) -> QueryRun:
    qr = await prepare(table, query_id=query_id, spec=spec)
    if query_id is not None:
        from backend.app.services.structured.value_domains import validate_call_params

        params, bad, _subs = validate_call_params(params, qr.declared)
        if bad:
            raise QueryRunError("parameter value(s) refused: " + "; ".join(
                f"{b.get('param') or b.get('combination')}: {b.get('error') or 'not a valid value'}"
                for b in bad))
    missing = [p["name"] for p in qr.built.params if p["name"] not in params and not p.get("optional")]
    if missing:
        raise QueryRunError(f"missing --param value(s): {', '.join(missing)}")
    qr.result = await run_sql(qr.built.sql, params, qr.built.tables, max_rows=max_rows,
                              timeout_ms=timeout_ms, declared_params=qr.declared)
    return qr

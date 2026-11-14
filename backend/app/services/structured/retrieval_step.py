"""``query --use-structured-data``: answer from the loaded tables via SQL only.

Runs after the graph walk, before answer synthesis:

1. **Find tables through the graph.** TableMetadata nodes linked to the
   entities the walk reached (``tableEntity`` edges) or to the ontology classes
   the question matched (``tableColumn``/``tableMetric``/``tableFilter``);
   a vector search over the metadata nodes fills the remaining slots.
2. **Decide and pick** (task ``table_query_select``, one call): is table data
   needed, and which STORED metric/filter/query/lookup answers it, with which
   parameter values. The model never writes SQL: when no stored query fits it
   may return a new JSON query spec.
3. **Build and run** deterministically: :mod:`query_builder` compiles the
   stored (or new) spec to SQL, which then runs through the guard
   (``sql_exec``): read-only, timed out, rows capped. A query that errors or
   returns nothing gets one retry call, which is told the error.
4. Each result becomes an evidence item ``kind="sql_result"`` with its own
   citation id. Table data reaches the prompt ONLY through these results.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.predicates import VIAO_NS
from backend.app.services.structured.metadata import (
    catalog_for_prompt,
    column_types,
    compile_item,
    find_item,
)
from backend.app.services.structured.query_builder import QuerySpecError, build_sql
from backend.app.services.structured.sql_exec import SqlResult, run_sql
from backend.app.services.structured.sql_guard import SqlGuardError
from backend.app.services.structured.value_domains import check_params, spec_declared

_TABLE_EDGE_KINDS = ("tableEntity", "tablePlace", "tableColumn", "tableMetric", "tableFilter", "tableTime")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


async def candidate_tables(
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    *,
    max_tables: int = 3,
    vector_max_distance: float | None = None,
    time_ids: list[uuid.UUID] | None = None,
) -> list[dict[str, Any]]:
    """[{id, table, meta, score, via, hits}] best first. ``hits`` records what
    the question reached the table through (entity/place columns, class
    labels, periods) so the table's tagged queries can be ranked."""
    from backend.app.db.session import session_scope

    scored: dict[uuid.UUID, dict[str, Any]] = {}
    async with session_scope() as s:
        if entity_ids or class_ids or time_ids:
            rows = (await s.execute(sql_text("""
                SELECT a.id, gr.target_node_type, gr.relationship_type, gr.extra_metadata, oc.iri, oc.label
                  FROM graphrag.intelligence_artifacts a
                  JOIN graphrag.graph_relationships gr
                    ON gr.source_node_type = 'intelligence_artifact' AND gr.source_node_id = a.id
                   AND gr.relationship_type = ANY(CAST(:kinds AS text[]))
                  LEFT JOIN graphrag.ontology_classes oc
                    ON gr.target_node_type = 'ontology_class' AND oc.id = gr.target_node_id
                 WHERE a.artifact_type = 'TableMetadata' AND a.status = 'ACTIVE'
                   AND ((gr.target_node_type = 'entity' AND gr.target_node_id = ANY(CAST(:ents AS uuid[])))
                     OR (gr.target_node_type = 'ontology_class' AND gr.target_node_id = ANY(CAST(:cls AS uuid[])))
                     OR (gr.target_node_type = 'time_instance' AND gr.target_node_id = ANY(CAST(:tms AS uuid[]))))
            """), {"kinds": list(_TABLE_EDGE_KINDS), "ents": [str(e) for e in entity_ids],
                   "cls": [str(c) for c in class_ids],
                   "tms": [str(t) for t in time_ids or []]})).all()
            counts: dict[tuple[uuid.UUID, str], int] = {}
            for aid, ttype, kind, extra, ciri, clabel in rows:
                d = scored.setdefault(aid, {"score": 0.0, "via": [], "hits": {
                    "columns": set(), "classes": set(), "time": False}})
                extra = extra or {}
                if ttype == "time_instance":
                    if not d["hits"]["time"]:
                        d["score"] += 1.0          # once: a long series covers many periods
                    d["hits"]["time"] = True
                else:
                    d["score"] += 2.0 if ttype == "entity" else 1.0
                if ttype == "entity":
                    d["hits"]["columns"].update(extra.get("columns") or [])
                if ttype == "ontology_class":
                    d["hits"]["classes"].update(x for x in (ciri, clabel) if x)
                key = "place" if kind == "tablePlace" else ttype
                counts[(aid, key)] = counts.get((aid, key), 0) + 1
            for (aid, key), n in counts.items():
                scored[aid]["via"].append(f"{key}:{n}")
        if qvec is not None and len(scored) < max_tables:
            rows = (await s.execute(sql_text("""
                SELECT id, (embedding <-> CAST(:q AS vector)) AS dist
                  FROM graphrag.intelligence_artifacts
                 WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE' AND embedding IS NOT NULL
                 ORDER BY dist LIMIT :n
            """), {"q": _vec(qvec), "n": max_tables})).all()
            for aid, dist in rows:
                if vector_max_distance is not None and float(dist) > vector_max_distance:
                    continue
                if aid not in scored:
                    scored[aid] = {"score": 0.5 / (1.0 + float(dist)), "via": [f"vector:{float(dist):.3f}"],
                                   "hits": {"columns": set(), "classes": set(), "time": False}}
        if not scored:
            return []
        best = sorted(scored.items(), key=lambda kv: -kv[1]["score"])[:max_tables]
        metas = dict((await s.execute(sql_text("""
            SELECT id, extra_metadata FROM graphrag.intelligence_artifacts
             WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": [str(a) for a, _ in best]})).all())
    out = []
    for aid, d in best:
        meta = metas.get(aid) or {}
        if meta.get("table_name"):
            out.append({"id": aid, "table": meta["table_name"], "meta": meta, **d})
    return out


def rank_items(meta: dict, hits: dict | None) -> dict[str, tuple[float, list[str]]]:
    """Score each stored query by the graph tags the question matched: its
    classes among the classes the question reached, its entity/place columns
    among the columns whose values the question named, its period columns
    when the question names a period. Measures get a small edge (tested,
    one value per period)."""
    from backend.app.services.structured.metadata import item_tags, items

    hits = hits or {}
    out: dict[str, tuple[float, list[str]]] = {}
    for sec, it in items(meta):
        tags = it.get("tags") or item_tags(meta, sec, it)
        score, why = 0.0, []
        cls = [c.get("label") or c.get("iri") for c in tags.get("classes") or []
               if (c.get("iri") in hits.get("classes", set()) or c.get("label") in hits.get("classes", set()))]
        if cls:
            score += 2.0 * len(set(cls))
            why.append("class " + ", ".join(sorted(set(cls))))
        cols = sorted(set(tags.get("entity_columns") or []) | set(tags.get("place_columns") or []))
        cols = [c for c in cols if c in hits.get("columns", set())]
        if cols:
            score += 1.5 * len(cols)
            why.append("entity in " + ", ".join(cols))
        if hits.get("time") and tags.get("time_columns"):
            score += 1.0
            why.append("period in " + ", ".join(tags["time_columns"]))
        if score and sec == "measures":
            score += 0.5
        out[it["id"]] = (score, why)
    return out


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}
    return obj if isinstance(obj, dict) else {}


async def table_metas(names: set[str]) -> dict[str, dict]:
    """ACTIVE metadata of the named tables (lookup partners outside the
    candidate set need their column types to compile)."""
    if not names:
        return {}
    from backend.app.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(sql_text("""
            SELECT extra_metadata FROM graphrag.intelligence_artifacts
             WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE'
               AND extra_metadata->>'table_name' = ANY(CAST(:t AS text[]))
        """), {"t": sorted(names)})).scalars().all()
    return {m["table_name"]: m for m in rows if isinstance(m, dict) and m.get("table_name")}


async def _execute(call: dict, by_table: dict[str, dict], *, max_rows: int, timeout_ms: int,
                   all_tables: set[str], types: dict[str, dict[str, str]],
                   ) -> tuple[dict, SqlResult | None, str | None]:
    """Build (deterministically) and run one selected call. Returns (record, result, error)."""
    table = call.get("table")
    params = call.get("params") if isinstance(call.get("params"), dict) else {}
    if call.get("query_id"):
        meta = (by_table.get(table) or {}).get("meta")
        hit = find_item(meta, call["query_id"]) if meta else None
        if hit is None:
            # The model may name the table wrong; search every candidate.
            for cand in by_table.values():
                hit = find_item(cand["meta"], call["query_id"])
                if hit:
                    meta, table = cand["meta"], cand["table"]
                    break
        if hit is None:
            err = f"unknown query_id {call['query_id']!r}"
            return {"table": table, "query_id": call["query_id"], "error": err}, None, err
        section, item = hit
        rec = {"table": meta["table_name"], "query_id": item["id"], "label": item.get("label"),
               "source": "catalog", "section": section}
        try:
            built = compile_item(meta, section, item, types)
        except (QuerySpecError, SqlGuardError) as exc:
            err = f"stored query does not compile: {exc}"
            return {**rec, "error": err}, None, err
        declared = (item.get("params") if isinstance(item.get("query"), dict) else None) or built.params
        params, bad, subs = check_params(params, declared, item.get("allowed_combinations"))
        if subs:
            rec["substituted"] = subs
        if bad:
            err = "parameter value(s) not in the column's valid values (see invalid_values)"
            return {**rec, "params": params, "invalid_values": bad, "error": err}, None, err
        res = await run_sql(built.sql, params, built.tables & (all_tables | {meta["table_name"]}),
                            max_rows=max_rows, timeout_ms=timeout_ms, declared_params=declared)
    elif isinstance(call.get("query"), dict):
        rec = {"table": table, "query_id": None, "label": call.get("purpose") or "new query spec",
               "source": "spec", "query": call["query"], "why_not_catalog": call.get("why_not_catalog")}
        if table not in by_table:
            err = f"table {table!r} is not a candidate table"
            return {**rec, "error": err}, None, err
        try:
            from backend.app.services.structured.metadata import time_columns

            tmeta = (by_table.get(table) or {}).get("meta") or {}
            built = build_sql(call["query"], table, {t: c for t, c in types.items() if t in all_tables},
                              time_columns={table: time_columns(tmeta)})
        except QuerySpecError as exc:
            err = f"query spec does not compile: {exc}"
            return {**rec, "error": err}, None, err
        domains = ((by_table.get(table) or {}).get("meta") or {}).get("value_domains") or {}
        declared = spec_declared(built.params, domains)
        params, bad, subs = check_params(params, declared)
        if subs:
            rec["substituted"] = subs
        if bad:
            err = "parameter value(s) not in the column's valid values (see invalid_values)"
            return {**rec, "params": params, "invalid_values": bad, "error": err}, None, err
        res = await run_sql(built.sql, params, built.tables, max_rows=max_rows, timeout_ms=timeout_ms,
                            declared_params=declared)
    else:
        err = "call has neither query_id nor query"
        return {"table": table, "error": err}, None, err
    rec.update({"sql": res.sql, "params": res.params, "row_count": res.row_count,
                "truncated": res.truncated, "total_rows": res.total_rows, "seconds": res.seconds,
                "error": res.error})
    if res.error:
        return rec, res, res.error
    if res.row_count == 0:
        return rec, res, "returned no rows"
    return rec, res, None


MAX_VALUE_CHECKS = 8
MAX_SUGGESTIONS = 8


def compared_values(sql: str, params: dict, columns: dict[str, dict[str, str]],
                    id_columns: dict[str, set[str]] | None = None) -> list[dict]:
    """``text_column = 'value'`` / ``IN (...)`` comparisons, from literals and
    bound :params: [{table, column, value, context}], where ``context`` holds
    the statement's OTHER single-value equality filters on the same table
    (``cik = '0000811809'``) -- a tag can exist in the table and still not
    exist for that company. ``columns``: table -> {column: type}; only text
    columns of known tables count."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import SqlGuardError, parse_one_select

    try:
        root = parse_one_select(sql)
    except SqlGuardError:
        return []
    alias = {}
    for t in root.find_all(exp.Table):
        if t.name and t.name.lower() in columns:
            alias[t.alias_or_name.lower()] = t.name.lower()
    tables = sorted(set(alias.values()))

    def _table_of(col) -> str | None:
        qual = (col.table or "").lower()
        if qual:
            return alias.get(qual)
        owners = [t for t in tables if col.name.lower() in columns[t]]
        return owners[0] if len(owners) == 1 else None

    def _vals(node) -> list[str]:
        # Compiled specs bind as CAST(:p AS text) and lists as = ANY(CAST(:p AS text[])).
        while isinstance(node, (exp.Cast, exp.Any, exp.Paren)):
            node = node.this
        if isinstance(node, exp.Literal) and node.is_string:
            return [node.this]
        if isinstance(node, exp.Placeholder) and node.name:
            v = (params or {}).get(node.name)
            if isinstance(v, str):
                return [v]
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        return []

    found: list[tuple[str, str, list[str], bool]] = []   # table, column, values, single-EQ
    for node in root.find_all(exp.EQ, exp.In):
        col, rhs = node.this, (node.expressions if isinstance(node, exp.In) else [node.expression])
        if not isinstance(col, exp.Column):
            continue
        t = _table_of(col)
        if t is None or (columns[t].get(col.name.lower()) or "text") != "text":
            continue
        vals = [v for r in rhs for v in _vals(r)]
        if vals:
            found.append((t, col.name.lower(), vals, isinstance(node, exp.EQ) and len(vals) == 1))

    out: list[dict] = []
    for t, c, vals, _single in found:
        # Only identifier filters (cik, lei, id...) scope the check: another
        # compared value may itself be the wrong one, and using it as context
        # would flag every value next to it.
        ids = (id_columns or {}).get(t, set())
        ctx = [(c2, v2[0]) for t2, c2, v2, single in found
               if t2 == t and c2 != c and single and c2 in ids]
        for v in vals:
            item = {"table": t, "column": c, "value": v, "context": ctx}
            if item not in out:
                out.append(item)
    return out[:MAX_VALUE_CHECKS]


async def missing_values(checks: list[dict]) -> list[dict]:
    """Which compared values do not occur in their column (under the
    statement's other equality filters), with the closest values that do. The
    model only ever sees a handful of example values, so it guesses 'Iron ore'
    for 'Iron ore, cfr spot' and US-GAAP tag names for an IFRS filer.
    Read-only, no LLM."""
    if not checks:
        return []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    out = []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for chk in checks:
                table, col, val, ctx = chk["table"], chk["column"], chk["value"], chk.get("context") or []
                ref = f"graphrag.{q(table)}"
                binds = {"v": val, **{f"c{i}": v for i, (_c, v) in enumerate(ctx)}}
                where_ctx = "".join(f" AND {q(c)} = :c{i}" for i, (c, _v) in enumerate(ctx))
                hit = (await s.execute(sql_text(
                    f"SELECT 1 FROM {ref} WHERE {q(col)} = :v{where_ctx} LIMIT 1"), binds)).first()
                if hit:
                    continue
                words = re.findall(r"[A-Za-z0-9]{3,}", val)[:4]
                like = " OR ".join(f"{q(col)} ILIKE :w{i}" for i in range(len(words)))
                binds.update({f"w{i}": f"%{w[:max(3, len(w) - 1)]}%" for i, w in enumerate(words)})
                cond = f"({like}) OR similarity({q(col)}, :v) > 0.3" if like else f"similarity({q(col)}, :v) > 0.3"
                rows = (await s.execute(sql_text(
                    f"SELECT v FROM (SELECT DISTINCT {q(col)} AS v FROM {ref} WHERE TRUE{where_ctx}) d "
                    f"WHERE {cond.replace(q(col), 'v')} "
                    f"ORDER BY similarity(v, :v) DESC, length(v) LIMIT {MAX_SUGGESTIONS}"), binds)).scalars().all()
                out.append({"table": table, "column": col, "value": val,
                            **({"where": {c: v for c, v in ctx}} if ctx else {}),
                            "existing_similar": [str(r) for r in rows]})
        except Exception as exc:  # hints are best effort
            out.append({"error": f"value check failed: {str(exc).splitlines()[0][:200]}"})
        finally:
            await s.rollback()
    return out


async def combination_hints(checks: list[dict]) -> list[dict]:
    """For an EMPTY result whose values all exist on their own: which
    combinations of the other compared columns actually occur with each value
    ('Iron ore, cfr spot' exists, '($/mt)' exists, but not together). Read-only."""
    by_table: dict[str, list[dict]] = {}
    for c in checks:
        by_table.setdefault(c["table"], []).append(c)
    if not any(len({c["column"] for c in cs}) > 1 for cs in by_table.values()):
        return []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    out = []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for table, cs in by_table.items():
                cols = list(dict.fromkeys(c["column"] for c in cs))
                if len(cols) < 2:
                    continue
                # Anchor on the most selective value (fewest combinations):
                # 'Iron ore, cfr spot' says something, 'nominal_usd' does not.
                best = None
                for anchor in cs[:4]:
                    others = [c for c in cols if c != anchor["column"]]
                    rows = (await s.execute(sql_text(
                        f"SELECT DISTINCT {', '.join(q(c) for c in others)} FROM graphrag.{q(table)} "
                        f"WHERE {q(anchor['column'])} = :v LIMIT {MAX_SUGGESTIONS + 1}"),
                        {"v": anchor["value"]})).all()
                    if rows and (best is None or len(rows) < len(best[2])):
                        best = (anchor, others, rows)
                if best is not None:
                    anchor, others, rows = best
                    out.append({"table": table, "where": {anchor["column"]: anchor["value"]},
                                "existing_combinations": [dict(zip(others, map(str, r), strict=True))
                                                          for r in rows[:MAX_SUGGESTIONS]]})
        except Exception as exc:  # hints are best effort
            out.append({"error": f"combination check failed: {str(exc).splitlines()[0][:200]}"})
        finally:
            await s.rollback()
    return out


def _evidence_text(rec: dict, res: SqlResult, meta: dict, char_cap: int) -> str:
    head = (f"SQL result from table {rec['table']}"
            + (f" ({rec['label']})" if rec.get("label") else "")
            + (f" [{(meta.get('source') or {}).get('rel_path')}]" if meta else "") + ". ")
    if rec.get("params"):
        head += "Parameters: " + ", ".join(f"{k}={v}" for k, v in rec["params"].items()) + ". "
    cols = res.columns
    body = []
    used = len(head)
    for r in res.rows:
        row = "; ".join(f"{c}={'' if v is None else v}" for c, v in zip(cols, r, strict=False))
        if used + len(row) + 4 > char_cap - 120:
            body.append(f"... ({len(res.rows) - len(body)} more fetched row(s) not shown)")
            break
        body.append("(" + row + ")")
        used += len(row) + 4
    tail = ""
    if res.truncated:
        tail = (f" Result capped at {res.row_count} of "
                f"{res.total_rows if res.total_rows is not None else 'more'} rows.")
    return head + f"{res.row_count} row(s): " + " ".join(body) + tail


async def structured_evidence(
    question: str,
    *,
    router,
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    max_tables: int = 3,
    max_rows: int = 50,
    timeout_ms: int = 5000,
    max_calls: int = 4,
    char_cap: int = 5000,
    vector_max_distance: float | None = None,
    verbose: bool = False,
    time_ids: list[uuid.UUID] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evidence items (``kind='sql_result'``) + a plan record for persistence."""
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    plan: dict[str, Any] = {"candidates": [], "needed": False, "calls": [], "retried": False}
    cands = await candidate_tables(entity_ids, class_ids, qvec, max_tables=max_tables,
                                   vector_max_distance=vector_max_distance, time_ids=time_ids)
    plan["candidates"] = [{"table": c["table"], "score": round(c["score"], 3), "via": c["via"]}
                          for c in cands]
    rankings = {c["table"]: rank_items(c["meta"], c.get("hits")) for c in cands}
    plan["ranked_queries"] = {t: [i for i, (sc, _w) in sorted(r.items(), key=lambda kv: -kv[1][0]) if sc > 0][:5]
                              for t, r in rankings.items()}
    if not cands:
        plan["reason"] = "no table reachable from the graph walk"
        return [], plan
    by_table = {c["table"]: c for c in cands}
    all_tables = set(by_table)
    for c in cands:  # lookups may join a table outside the candidate set
        for it in c["meta"].get("lookups") or []:
            if it.get("other_table"):
                all_tables.add(it["other_table"])
    partners = await table_metas(all_tables - set(by_table)) if all_tables - set(by_table) else {}
    all_tables = set(by_table) | set(partners)   # a partner that is not loaded cannot be read
    types = {c["table"]: column_types(c["meta"]) for c in cands}
    types.update({n: column_types(m) for n, m in partners.items()})
    catalogs = [catalog_for_prompt(c["meta"], ranking=rankings.get(c["table"])) for c in cands]
    col_types = {c["table"]: {col["name"].lower(): (col.get("type") or "text")
                              for col in c["meta"].get("columns") or []} for c in cands}
    id_cols = {c["table"]: {col["name"].lower() for col in c["meta"].get("columns") or []
                            if col.get("role") == "identifier"} for c in cands}

    async def _value_failures(rec: dict, res: SqlResult | None) -> list[dict]:
        if res is None or res.error:
            return []
        checks = compared_values(res.sql, res.params, col_types, id_cols)
        missing = [m for m in await missing_values(checks) if "value" in m]
        if not missing and res.row_count == 0:
            combos = [h for h in await combination_hints(checks) if "where" in h]
            if combos:
                rec["value_combinations"] = combos
        return missing

    system, user = PROMPTS["table_query_select"](question, catalogs)
    out = await router.chat("table_query_select", system=system, user=user)
    choice = _parse(out.text)
    plan["needed"] = bool(choice.get("needed"))
    plan["reason"] = choice.get("reason")
    if not plan["needed"]:
        return [], plan

    calls = [c for c in (choice.get("calls") or []) if isinstance(c, dict)][:max_calls]
    results: list[tuple[dict, SqlResult]] = []
    failures = []
    for call in calls:
        rec, res, err = await _execute(call, by_table, max_rows=max_rows, timeout_ms=timeout_ms,
                                       all_tables=all_tables, types=types)
        missing = await _value_failures(rec, res)
        if missing:
            rec["missing_values"] = missing
        plan["calls"].append(rec)
        if err or missing:
            failures.append({"call": call, "sql": rec.get("sql"),
                             "error": err or "value(s) not found in their column (see missing_values)",
                             **({"invalid_values": rec["invalid_values"]} if rec.get("invalid_values") else {}),
                             **({"missing_values": missing} if missing else {}),
                             **({"value_combinations": rec["value_combinations"]}
                                if rec.get("value_combinations") else {})})
        if not err and res is not None:
            # A partial hit (one IN value matched, another did not) is still
            # evidence; the retry may add the missing part.
            results.append((rec, res))

    if failures:
        plan["retried"] = True
        system, user = PROMPTS["table_query_select"](question, catalogs, failures)
        out = await router.chat("table_query_select", system=system, user=user)
        retry = _parse(out.text)
        for call in [c for c in (retry.get("calls") or []) if isinstance(c, dict)][:max_calls]:
            rec, res, err = await _execute(call, by_table, max_rows=max_rows, timeout_ms=timeout_ms,
                                           all_tables=all_tables, types=types)
            rec["retry"] = True
            missing = await _value_failures(rec, res)
            if missing:
                rec["missing_values"] = missing
            plan["calls"].append(rec)
            if not err and res is not None:
                results.append((rec, res))

    # The retry may repeat a statement that already succeeded; keep one copy.
    seen_runs: set[str] = set()
    unique = []
    for rec, res in results:
        k = json.dumps([" ".join(res.sql.split()), res.params], sort_keys=True, default=str)
        if k not in seen_runs:
            seen_runs.add(k)
            unique.append((rec, res))
    results = unique

    evidence = []
    for i, (rec, res) in enumerate(results, start=1):
        meta = (by_table.get(rec["table"]) or {}).get("meta") or {}
        node = (by_table.get(rec["table"]) or {}).get("id")
        # Same statement + params -> same id (deep_research rounds can then
        # share it); a different run of the same query gets a different id.
        digest = hashlib.sha256(json.dumps([res.sql, res.params], sort_keys=True,
                                           default=str).encode()).hexdigest()[:8]
        frag = f"SqlResult_{rec['table']}_{rec.get('query_id') or 'q'}_{digest}"
        evidence.append({
            "kind": "sql_result",
            "node_id": node,
            "iri": f"{VIAO_NS}#{re.sub(r'[^A-Za-z0-9_]', '_', frag)}",
            "rank": i, "score": 1.0,
            "text": _evidence_text(rec, res, meta, char_cap),
            "document_title": f"table {rec['table']}",
            "table": rec["table"], "query_id": rec.get("query_id"), "sql": res.sql,
            "params": res.params, "columns": res.columns, "rows": res.rows,
            "row_count": res.row_count, "truncated": res.truncated, "total_rows": res.total_rows,
            "source_file": (meta.get("source") or {}).get("rel_path"),
            "sql_source": rec.get("source"),
        })
    plan["seconds"] = round(time.monotonic() - t0, 3)
    if verbose:
        print(f"[query] structured data: {len(cands)} candidate table(s) "
              f"{[c['table'] for c in cands]}, needed={plan['needed']}, "
              f"{len(evidence)} result(s), retried={plan['retried']}")
    return evidence, plan

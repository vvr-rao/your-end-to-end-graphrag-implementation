"""``query --use-structured-data``: answer from the loaded tables via SQL only.

Runs after the graph walk, before answer synthesis:

1. **Find tables through the graph.** TableMetadata nodes linked to the
   entities the walk reached (``tableEntity`` edges) or to the ontology classes
   the question matched (``tableColumn``/``tableMetric``/``tableFilter``);
   a vector search over the metadata nodes fills the remaining slots.
2. **Decide and pick** (task ``table_query_select``, one call): is table data
   needed, and which STORED metric/filter/query/lookup answers it, with which
   parameter values. New SQL is written only when no stored statement fits.
3. **Run** each statement through the guard (``sql_exec``): read-only, timed
   out, rows capped. A stored statement that errors or returns nothing gets
   one retry call, which is told the error and may write new SQL.
4. Each result becomes an evidence item ``kind="sql_result"`` with its own
   citation id. Table data reaches the prompt ONLY through these results.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.predicates import VIAO_NS
from backend.app.services.structured.metadata import catalog_for_prompt, find_item, item_tables
from backend.app.services.structured.sql_exec import SqlResult, run_sql

_TABLE_EDGE_KINDS = ("tableEntity", "tableColumn", "tableMetric", "tableFilter")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


async def candidate_tables(
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    *,
    max_tables: int = 3,
    vector_max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """[{id, table, meta, score, via}] best first."""
    from backend.app.db.session import session_scope

    scored: dict[uuid.UUID, dict[str, Any]] = {}
    async with session_scope() as s:
        if entity_ids or class_ids:
            rows = (await s.execute(sql_text("""
                SELECT a.id, gr.target_node_type, count(*)
                  FROM graphrag.intelligence_artifacts a
                  JOIN graphrag.graph_relationships gr
                    ON gr.source_node_type = 'intelligence_artifact' AND gr.source_node_id = a.id
                   AND gr.relationship_type = ANY(CAST(:kinds AS text[]))
                 WHERE a.artifact_type = 'TableMetadata' AND a.status = 'ACTIVE'
                   AND ((gr.target_node_type = 'entity' AND gr.target_node_id = ANY(CAST(:ents AS uuid[])))
                     OR (gr.target_node_type = 'ontology_class' AND gr.target_node_id = ANY(CAST(:cls AS uuid[]))))
                 GROUP BY a.id, gr.target_node_type
            """), {"kinds": list(_TABLE_EDGE_KINDS), "ents": [str(e) for e in entity_ids],
                   "cls": [str(c) for c in class_ids]})).all()
            for aid, ttype, n in rows:
                d = scored.setdefault(aid, {"score": 0.0, "via": []})
                d["score"] += (2.0 if ttype == "entity" else 1.0) * float(n)
                d["via"].append(f"{ttype}:{n}")
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
                    scored[aid] = {"score": 0.5 / (1.0 + float(dist)), "via": [f"vector:{float(dist):.3f}"]}
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


async def _execute(call: dict, by_table: dict[str, dict], *, max_rows: int, timeout_ms: int,
                   all_tables: set[str]) -> tuple[dict, SqlResult | None, str | None]:
    """Run one selected call. Returns (record, result, error)."""
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
        res = await run_sql(item["sql"], params, item_tables(meta, section, item) & all_tables
                            if section == "lookups" else {meta["table_name"]},
                            max_rows=max_rows, timeout_ms=timeout_ms, declared_params=item.get("params"))
        rec = {"table": meta["table_name"], "query_id": item["id"], "label": item.get("label"),
               "source": "catalog", "section": section}
    elif call.get("new_sql"):
        res = await run_sql(str(call["new_sql"]), params, all_tables, max_rows=max_rows, timeout_ms=timeout_ms)
        rec = {"table": table, "query_id": None, "label": call.get("purpose") or "generated query",
               "source": "generated", "why_not_catalog": call.get("why_not_catalog")}
    else:
        err = "call has neither query_id nor new_sql"
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evidence items (``kind='sql_result'``) + a plan record for persistence."""
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    plan: dict[str, Any] = {"candidates": [], "needed": False, "calls": [], "retried": False}
    cands = await candidate_tables(entity_ids, class_ids, qvec, max_tables=max_tables,
                                   vector_max_distance=vector_max_distance)
    plan["candidates"] = [{"table": c["table"], "score": round(c["score"], 3), "via": c["via"]}
                          for c in cands]
    if not cands:
        plan["reason"] = "no table reachable from the graph walk"
        return [], plan
    by_table = {c["table"]: c for c in cands}
    all_tables = set(by_table)
    for c in cands:  # lookups may join a table outside the candidate set
        for it in c["meta"].get("lookups") or []:
            if it.get("other_table"):
                all_tables.add(it["other_table"])
    catalogs = [catalog_for_prompt(c["meta"]) for c in cands]
    col_types = {c["table"]: {col["name"].lower(): (col.get("type") or "text")
                              for col in c["meta"].get("columns") or []} for c in cands}
    id_cols = {c["table"]: {col["name"].lower() for col in c["meta"].get("columns") or []
                            if col.get("role") == "identifier"} for c in cands}

    async def _value_failures(rec: dict, res: SqlResult | None) -> list[dict]:
        if res is None or res.error:
            return []
        return [m for m in await missing_values(compared_values(res.sql, res.params, col_types, id_cols))
                if "value" in m]

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
                                       all_tables=all_tables)
        missing = await _value_failures(rec, res)
        if missing:
            rec["missing_values"] = missing
        plan["calls"].append(rec)
        if err or missing:
            failures.append({"call": call, "sql": rec.get("sql"),
                             "error": err or "value(s) not found in their column (see missing_values)",
                             **({"missing_values": missing} if missing else {})})
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
                                           all_tables=all_tables)
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
        frag = f"SqlResult_{rec['table']}_{rec.get('query_id') or 'q'}_{i}"
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

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
from datetime import date
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.predicates import VIAO_NS
from backend.app.services.structured.metadata import (
    catalog_for_prompt,
    column_types,
    compile_item,
    find_item,
    item_spec,
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


async def table_inventory() -> dict[str, dict]:
    """Every ACTIVE table: {table: {id, iri, meta}} (for the ALL TABLES list
    and for describing a table that is not a candidate)."""
    from backend.app.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(sql_text("""
            SELECT id, artifact_identifier, extra_metadata FROM graphrag.intelligence_artifacts
             WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE'
             ORDER BY extra_metadata->>'table_name'
        """))).all()
    return {m["table_name"]: {"id": aid, "iri": iri, "meta": m}
            for aid, iri, m in rows if (m or {}).get("table_name")}


def inventory_for_prompt(inv: dict[str, dict], limit: int = 60) -> list[dict]:
    out = []
    for name, rec in list(inv.items())[:limit]:
        desc = str(rec["meta"].get("description") or "").strip()
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:220]
        out.append({"table": name, "rows": rec["meta"].get("row_count"), "about": first})
    return out


# ------------------------------------------------------------ relative periods

_RELATIVE = re.compile(
    r"\b(past|last|previous|recent|latest|trailing|preceding)\b|\b(this|current)\s+(year|quarter|month)\b"
    r"|\byear[- ]to[- ]date\b|\bytd\b|\bso far\b", re.I)
_CURRENT_YEAR = re.compile(r"\b(?:current|present)\s+year\s+is\s+(\d{4})|\b(?:it|today)\s+is\s+(?:\w+\s+)*?(\d{4})", re.I)


def relative_period_question(question: str, today_year: int) -> bool:
    """The question counts periods back from 'now' and names no period of its
    own (a year other than the stated current one fixes the window itself)."""
    if not _RELATIVE.search(question or ""):
        return False
    current = {today_year} | {int(y) for m in _CURRENT_YEAR.finditer(question) for y in m.groups() if y}
    named = {int(y) for y in re.findall(r"\b(1[6-9]\d\d|2[01]\d\d)\b", question)}
    return not (named - current)


def _period_index(value: Any) -> tuple[str, int] | None:
    from backend.app.services.structured.column_stats import granularity
    from backend.app.services.structured.graph_links import period_of

    g, p = granularity(value), period_of(value)
    if p is None or g not in {"year", "quarter", "month"}:
        return None
    y, q, m = p
    return g, (y if g == "year" else y * 4 + (q - 1) if g == "quarter" else y * 12 + (m - 1))


def _period_value(like: Any, gran: str, idx: int) -> Any:
    """Render period index ``idx`` in the spelling of ``like``."""
    s = str(like).strip()
    if gran == "year":
        return idx if isinstance(like, int) else str(idx)
    if gran == "quarter":
        y, q = divmod(idx, 4)
        m = re.match(r"^(\d{4})(\D*)([1-4])$", s)
        return f"{y}{m.group(2)}{q + 1}" if m else f"{y}Q{q + 1}"
    y, mo = divmod(idx, 12)
    m = re.match(r"^\d{4}(\D*)\d{1,2}$", s)
    sep = m.group(1) if m else "-"
    return f"{y}{sep}{mo + 1:02d}"


def between_params(spec: dict | None) -> list[tuple[str, str, str]]:
    """(column, from-param, to-param) of the spec's parameterised ranges."""
    out = []
    for w in (spec or {}).get("where") or []:
        if (isinstance(w, dict) and str(w.get("op") or "").lower() == "between"
                and isinstance(w.get("param"), list) and len(w["param"]) == 2
                and isinstance(w.get("column"), str)):
            out.append((w["column"].split(".")[-1], str(w["param"][0]), str(w["param"][1])))
    return out


def anchor_relative_periods(question: str, call: dict, meta: dict, *, today_year: int) -> dict | None:
    """When the question counts back from 'now' ('past 3 years', 'recent'),
    move a call's period window so it ends at the LATEST period the table
    holds at that granularity, keeping its length. Returns the adjustment
    (and changes ``call['params']``) or None. Deterministic; any table."""
    from backend.app.services.structured.column_stats import latest_period

    if not relative_period_question(question, today_year):
        return None
    if call.get("query_id"):
        found = find_item(meta, call["query_id"])
        spec = item_spec(found[0], found[1]) if found else None
    else:
        spec = call.get("query")
    params = call.get("params") or {}
    for col, p_from, p_to in between_params(spec):
        lo, hi = params.get(p_from), params.get(p_to)
        if lo is None or hi is None:
            continue
        a, b = _period_index(lo), _period_index(hi)
        latest = latest_period(meta, col, hi)
        c = _period_index(latest) if latest is not None else None
        if not a or not b or not c or a[0] != b[0] or b[0] != c[0] or b[1] == c[1] or b[1] < a[1]:
            continue
        span = b[1] - a[1]
        new_lo, new_hi = _period_value(lo, a[0], c[1] - span), _period_value(hi, a[0], c[1])
        call["params"] = {**params, p_from: new_lo, p_to: new_hi}
        return {"column": col, "was": [lo, hi], "now": [new_lo, new_hi], "latest": latest}
    return None


def describe_evidence(names: list[str], inv: dict[str, dict], start_rank: int) -> list[dict]:
    from backend.app.services.structured.column_stats import metadata_card

    out = []
    for i, name in enumerate(n for n in dict.fromkeys(names) if n in inv):
        rec = inv[name]
        iri = rec.get("iri") or f"{VIAO_NS}#TableMetadata_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
        out.append({"kind": "table_metadata", "node_id": rec["id"], "iri": iri,
                    "rank": start_rank + i, "score": 1.0, "text": metadata_card(rec["meta"]),
                    "char_cap": 12000, "document_title": f"table {name}", "table": name,
                    "source_file": (rec["meta"].get("source") or {}).get("rel_path")})
    return out


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


def _listing_text(head: str, res: SqlResult, char_cap: int, summary: str = "") -> str:
    """Compact layout for a LISTING result: the header once, then one line of
    values per row (about half the size of name=value pairs). ``summary`` (the
    exact change statistics) goes first, so the row cap never cuts it."""
    if summary:
        head += summary + " "
    def cell(v) -> str:
        return "" if v is None else str(v).replace("|", "/")
    lines = [" | ".join(res.columns)]
    used = len(head) + len(lines[0])
    shown = 0
    for r in res.rows:
        row = " | ".join(cell(v) for v in r)
        if used + len(row) + 3 > char_cap - 160:
            break
        lines.append(row)
        used += len(row) + 3
        shown += 1
    note = f"{res.row_count} row(s)" + (f" of {res.total_rows if res.total_rows is not None else 'more'}"
                                         " (capped)" if res.truncated else "")
    if shown < len(res.rows):
        note += f"; {shown} shown here, {len(res.rows) - shown} more in the returned data"
    return head + f"LISTING, {note}. Rows: " + " ;; ".join(lines)


def _listing_stats(rec: dict, res: SqlResult, meta: dict) -> dict | None:
    if not rec.get("listing"):
        return None
    from backend.app.services.structured.listing_stats import change_stats, roles_for

    period_col, value_col = roles_for(meta or {}, rec.get("query_id"), res.columns)
    return change_stats(res.columns, res.rows, period_col, value_col) if period_col else None


def _evidence_text(rec: dict, res: SqlResult, meta: dict, char_cap: int, stats: dict | None = None) -> str:
    head = (f"SQL result from table {rec['table']}"
            + (f" ({rec['label']})" if rec.get("label") else "")
            + (f" [{(meta.get('source') or {}).get('rel_path')}]" if meta else "") + ". ")
    if rec.get("params"):
        head += "Parameters: " + ", ".join(f"{k}={v}" for k, v in rec["params"].items()) + ". "
    if rec.get("listing"):
        from backend.app.services.structured.listing_stats import stats_text

        return _listing_text(head, res, char_cap, stats_text(stats) if stats else "")
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
    listing_max_rows: int = 2000,
    listing_char_cap: int = 60000,
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
    inv = await table_inventory()
    inventory = inventory_for_prompt(inv)
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

    system, user = PROMPTS["table_query_select"](question, catalogs, max_calls=max_calls,
                                                  inventory=inventory)
    out = await router.chat("table_query_select", system=system, user=user)
    choice = _parse(out.text)
    if "needed" not in choice:
        # cut off at max_tokens or not JSON: an error, never a silent "not needed"
        plan["select_error"] = (f"selection reply unreadable (finish_reason={getattr(out, "finish_reason", None)}, "
                                f"{len(out.text or '')} chars)")
        plan["retried"] = True
        system, user = PROMPTS["table_query_select"](question, catalogs, [{
            "error": "your previous reply was cut off or was not valid JSON. Reply again with FEWER "
                     "calls: one call per part of the question; a question that lists many items is "
                     "ONE listing query, never one call per item."}], max_calls=max_calls,
            inventory=inventory)
        out = await router.chat("table_query_select", system=system, user=user)
        choice = _parse(out.text)
        if "needed" not in choice:
            plan["select_error"] += f"; retry also unreadable (finish_reason={getattr(out, "finish_reason", None)})"
    describe = [t for t in (choice.get("describe") or []) if isinstance(t, str) and t in inv]
    plan["describe"] = describe
    plan["needed"] = bool(choice.get("needed")) or bool(describe)
    plan["reason"] = choice.get("reason")
    if not plan["needed"]:
        return [], plan

    calls = [c for c in (choice.get("calls") or []) if isinstance(c, dict)][:max_calls]
    today_year = date.today().year
    for call in calls:   # 'past 3 years' ends at the table's latest period, not the model's guess
        meta_c = (by_table.get(call.get("table")) or {}).get("meta")
        adj = anchor_relative_periods(question, call, meta_c, today_year=today_year) if meta_c else None
        if adj:
            plan.setdefault("period_adjustments", []).append({"table": call.get("table"), **adj})
    results: list[tuple[dict, SqlResult]] = []
    failures = []
    def _rows_for(call: dict) -> int:
        return listing_max_rows if call.get("listing") else max_rows

    for call in calls:
        rec, res, err = await _execute(call, by_table, max_rows=_rows_for(call), timeout_ms=timeout_ms,
                                       all_tables=all_tables, types=types)
        rec["listing"] = bool(call.get("listing"))
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
        system, user = PROMPTS["table_query_select"](question, catalogs, failures, max_calls=max_calls,
                                                     inventory=inventory)
        out = await router.chat("table_query_select", system=system, user=user)
        retry = _parse(out.text)
        for call in [c for c in (retry.get("calls") or []) if isinstance(c, dict)][:max_calls]:
            meta_c = (by_table.get(call.get("table")) or {}).get("meta")
            adj = anchor_relative_periods(question, call, meta_c, today_year=today_year) if meta_c else None
            if adj:
                plan.setdefault("period_adjustments", []).append({"table": call.get("table"), **adj})
            rec, res, err = await _execute(call, by_table, max_rows=_rows_for(call), timeout_ms=timeout_ms,
                                           all_tables=all_tables, types=types)
            rec["retry"] = True
            rec["listing"] = bool(call.get("listing"))
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
        stats = _listing_stats(rec, res, meta)
        evidence.append({
            "kind": "sql_result",
            "node_id": node,
            "iri": f"{VIAO_NS}#{re.sub(r'[^A-Za-z0-9_]', '_', frag)}",
            "rank": i, "score": 1.0,
            "text": _evidence_text(rec, res, meta, listing_char_cap if rec.get("listing") else char_cap, stats),
            "change_stats": stats,
            "char_cap": listing_char_cap if rec.get("listing") else char_cap,
            "listing": bool(rec.get("listing")),
            "document_title": f"table {rec['table']}",
            "table": rec["table"], "query_id": rec.get("query_id"), "sql": res.sql,
            "params": res.params, "columns": res.columns, "rows": res.rows,
            "row_count": res.row_count, "truncated": res.truncated, "total_rows": res.total_rows,
            "source_file": (meta.get("source") or {}).get("rel_path"),
            "sql_source": rec.get("source"),
        })
    evidence += describe_evidence(describe, inv, len(evidence) + 1)
    plan["seconds"] = round(time.monotonic() - t0, 3)
    if verbose:
        print(f"[query] structured data: {len(cands)} candidate table(s) "
              f"{[c['table'] for c in cands]}, needed={plan['needed']}, "
              f"{len(evidence)} result(s), retried={plan['retried']}")
    return evidence, plan

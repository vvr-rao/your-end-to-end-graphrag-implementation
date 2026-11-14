"""Second metadata pass on a LOADED table: measure its structure, define its
measures, then test every stored query against the data.

``profiler`` writes the first metadata from a profile and a sample (no
database). Here the table is in Postgres, so the metadata can be checked
against how the data actually behaves:

1. **Analyse** (:mod:`analysis`, SQL only): grain, functional dependencies,
   columns across which a measure repeats (report/version columns), item
   columns of long tables.
2. **Define** (task ``table_refine``, one LLM call with a domain-neutral
   prompt): column corrections, MEASURES (value column, item values, entity /
   period / qualifier columns, which report wins), usage notes, typical
   questions.
3. **Build** measure queries deterministically from each definition: one
   value per entity, period and qualifier -- repeats across report columns
   collapse to the latest report -- with the entity, qualifiers and a period
   range as parameters.
4. **Test** every stored item with parameter values read from the table:
   it must run and return rows; an aggregate must not add the same reported
   value twice (rows that differ only in report columns) nor combine several
   periods while grouping by a report column. Failures get one repair call
   (``table_sql_repair``); what still fails is dropped and recorded. Typical
   questions are run too and kept only when they return rows.

The refined metadata replaces the node's metadata and the cache entry
(``refined.version``), so later runs make no LLM call.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.structured.analysis import analyze_table, report_columns, summary_for_prompt
from backend.app.services.structured.metadata import (
    COLUMN_ROLES,
    SECTIONS,
    column_types,
    compile_item,
    finalize_item,
    item_spec,
    items,
    time_columns,
)
from backend.app.services.structured.query_builder import QuerySpecError, build_sql, compact
from backend.app.services.structured.sql_exec import run_sql
from backend.app.services.structured.sql_guard import SqlGuardError

REFINE_VERSION = "tr-1"
MAX_MEASURES = 15
SAMPLE_ROWS = 8
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


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


# ------------------------------------------------------------------ measures


def normalize_measure(raw: Any, meta: dict) -> tuple[dict | None, str | None]:
    """Validate one measure definition against the table. (measure, error)."""
    if not isinstance(raw, dict):
        return None, "not an object"
    cols = {c["name"]: c for c in meta.get("columns") or []}
    mid = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id") or raw.get("label") or "").lower()).strip("_")[:40]
    if not mid:
        return None, "missing id"
    vcol = raw.get("value_column")
    if vcol not in cols or cols[vcol].get("type") not in {"integer", "numeric"}:
        return None, f"value_column {vcol!r} is not a numeric column"
    item = raw.get("item")
    if item:
        if not isinstance(item, dict) or item.get("column") not in cols:
            return None, f"item column {item!r} does not exist"
        vals = item.get("values") if isinstance(item.get("values"), list) else [item.get("value")]
        vals = [str(v) for v in vals if v not in (None, "")]
        if not vals:
            return None, "item has no values"
        item = {"column": item["column"], "values": list(dict.fromkeys(vals))}

    def _list(key: str) -> list[str]:
        return [c for c in dict.fromkeys(raw.get(key) or []) if isinstance(c, str)]

    ents, periods, quals = _list("entity_columns"), _list("period_columns"), _list("qualifier_columns")
    bad = [c for c in ents + periods + quals if c not in cols]
    if bad:
        return None, f"unknown column(s): {', '.join(bad)}"
    if not ents:
        return None, "needs at least one entity column"
    latest = raw.get("latest_by") if raw.get("latest_by") in cols else None
    return {"id": mid, "label": str(raw.get("label") or mid), "description": str(raw.get("description") or ""),
            "value_column": vcol, "item": item, "entity_columns": ents, "period_columns": periods,
            "qualifier_columns": [q for q in quals if q not in ents + periods],
            "latest_by": latest, "additive_over_time": bool(raw.get("additive_over_time"))}, None


def measure_item(m: dict, meta: dict) -> dict:
    """One stored query per measure, built without an LLM: one value per
    entity x period x qualifiers (repeats across reports collapse to the
    latest), entity + listed qualifiers + a period range as parameters."""
    domains = meta.get("value_domains") or {}
    key = m["entity_columns"][0]
    sel = list(m["entity_columns"]) + m["period_columns"] + m["qualifier_columns"]
    if m["item"] and len(m["item"]["values"]) > 1:
        sel.append(m["item"]["column"])
    sel.append(m["value_column"])
    if m["latest_by"] and m["latest_by"] not in sel:
        sel.append(m["latest_by"])
    where: list[dict] = []
    if m["item"]:
        vals = m["item"]["values"]
        where.append({"column": m["item"]["column"], "op": "in" if len(vals) > 1 else "=",
                      "value": vals if len(vals) > 1 else vals[0]})
    where.append({"column": m["value_column"], "op": "not_null"})
    where.append({"column": key, "op": "=", "param": key})
    for q in m["qualifier_columns"]:
        if len((domains.get(q) or {}).get("values") or []) > 1:
            where.append({"column": q, "op": "=", "param": q})
    if m["period_columns"]:
        p = m["period_columns"][-1]
        where.append({"column": p, "op": "between", "param": [f"{p}_from", f"{p}_to"]})
    spec: dict = {"select": sel, "where": where}
    one_per = [key] + m["period_columns"] + m["qualifier_columns"]
    spec["distinct_on"] = one_per
    order = [{"column": c, "dir": "asc"} for c in one_per]
    if m["latest_by"]:
        order.append({"column": m["latest_by"], "dir": "desc"})
    spec["order_by"] = order
    return {"id": f"measure_{m['id']}", "label": m["label"],
            "description": (m["description"] + " One value per "
                            + ", ".join(one_per)
                            + (f"; when a value is restated, the latest by {m['latest_by']}" if m["latest_by"] else "")
                            + ".").strip(),
            "columns": sel, "class_iri": None, "class_label": None,
            "query": spec, "params": [], "sql": "", "measure": m}


# ------------------------------------------------------------------ tests


def _base_where(spec: dict, table: str, cols: set[str]) -> list[dict]:
    """The spec's constant filters on the base table (no params)."""
    out = []
    for w in spec.get("where") or []:
        c = w.get("column")
        name = c if isinstance(c, str) else (c or {}).get("column")
        if isinstance(name, str) and "." in name:
            t, name = name.split(".", 1)
            if t != table:
                continue
        if name in cols and ("value" in w or w.get("op") in {"is_null", "not_null"}):
            out.append(w)
    return out


async def example_params(meta: dict, section: str, item: dict, built) -> dict[str, Any] | None:
    """Parameter values read from ONE real row that satisfies the item's
    constant filters (between params get that row's value at both ends)."""
    table = meta["table_name"]
    types = column_types(meta)
    spec = item_spec(section, item) or {}
    if spec.get("from") and spec["from"] != table:
        return None
    pcols = {}
    for p in built.params:
        col = p.get("column") or ""
        base = re.sub(r"^(year|month|quarter|lower)_", "", col) if col not in types else col
        if base in types:
            pcols[p["name"]] = base
    need = sorted(set(pcols.values()))
    if not need:
        return {p["name"]: 5 for p in built.params}
    probe = {"select": need,
             "where": _base_where(spec, table, set(types)) + [{"column": c, "op": "not_null"} for c in need],
             "limit": 1}
    try:
        pb = build_sql(probe, table, {table: types})
    except QuerySpecError:
        return None
    res = await run_sql(pb.sql, {}, {table}, max_rows=1, timeout_ms=20000)
    if res.error or not res.rows:
        return None
    row = dict(zip(res.columns, res.rows[0], strict=False))
    out: dict[str, Any] = {}
    for p in built.params:
        col = pcols.get(p["name"])
        if col is None:
            out[p["name"]] = 5          # a limit / top-n
            continue
        v = row.get(col)
        if (p.get("column") or "").startswith("year_") and v:
            v = int(str(v)[:4])
        out[p["name"]] = [v] if p["type"].endswith("[]") else v
    return out


def _grouped_and_filtered(spec: dict) -> set[str]:
    out = set()
    for r in (spec.get("group_by") or []) + [s for s in spec.get("select") or [] if not (isinstance(s, dict) and s.get("agg"))]:
        out.add(r if isinstance(r, str) else (r or {}).get("column"))
    for w in spec.get("where") or []:
        if str(w.get("op") or "=") == "=":
            c = w.get("column")
            out.add(c if isinstance(c, str) else (c or {}).get("column"))
    return {c for c in out if c}


async def test_item(meta: dict, section: str, item: dict, analysis: dict,
                    types_by_table: dict[str, dict[str, str]]) -> dict:
    """{"ok": bool, "rows": n, "error"?: str, "params"?: {...}}"""
    try:
        built = compile_item(meta, section, item, types_by_table)
    except (QuerySpecError, SqlGuardError) as exc:
        return {"ok": False, "error": f"does not compile: {exc}"}
    params = await example_params(meta, section, item, built)
    if params is None:
        return {"ok": True, "rows": None, "note": "not run (no example row found for its parameters)"}
    res = await run_sql(built.sql, params, built.tables, max_rows=50, timeout_ms=20000,
                        declared_params=built.params)
    if res.error:
        return {"ok": False, "error": res.error, "params": params}
    if not res.rows:
        return {"ok": False, "error": "returned no rows with parameter values taken from the table",
                "params": params}
    out = {"ok": True, "rows": res.row_count, "params": params}
    spec = item_spec(section, item) or {}
    aggs = [s for s in spec.get("select") or [] if isinstance(s, dict)
            and str(s.get("agg") or "") in {"sum", "avg"}]
    if not aggs or spec.get("join"):
        return out
    table = meta["table_name"]
    measures = {s.get("column") for s in aggs}
    report = [c for m in measures for c in report_columns(analysis, m)]
    keyed = _grouped_and_filtered(spec)
    groups = [s for s in spec.get("select") or [] if not (isinstance(s, dict) and s.get("agg"))]
    where = spec.get("where") or []
    if report:
        identity = [c for c in analysis.get("grain") or [] if c not in report]
        if identity:
            diag = {"select": [*groups, {"agg": "count", "as": "n"}, {"agg": "count_distinct", "columns": identity, "as": "k"}],
                    "where": where, "group_by": spec.get("group_by") or []}
            r = await _run_diag(diag, table, params, types_by_table)
            if r and any(int(x["n"]) > int(x["k"]) for x in r):
                worst = max(r, key=lambda x: int(x["n"]) - int(x["k"]))
                return {**out, "ok": False, "error": (
                    f"aggregates the same reported value more than once: {worst['n']} rows but only "
                    f"{worst['k']} distinct values of {', '.join(identity)}; rows differ only in "
                    f"{', '.join(report)}, which identify the report the value came from. Filter "
                    "those to one report, or use a measure query.")}
    periods = [c for c in time_columns(meta) if c not in report]
    if periods and report and (keyed & set(report)) and not (keyed & set(periods)):
        diag = {"select": [*groups, {"agg": "count_distinct", "columns": periods, "as": "p"}],
                "where": where, "group_by": spec.get("group_by") or []}
        r = await _run_diag(diag, table, params, types_by_table)
        if r and any(int(x["p"]) > 1 for x in r):
            n = max(int(x["p"]) for x in r)
            return {**out, "ok": False, "error": (
                f"groups by {', '.join(sorted(keyed & set(report)))}, which identify the REPORT, not "
                f"the period: one result row combines {n} different periods "
                f"({', '.join(periods)}). Group or filter by the period columns instead.")}
    return out


async def _run_diag(spec: dict, table: str, params: dict, types_by_table) -> list[dict] | None:
    try:
        b = build_sql(spec, table, {table: types_by_table[table]})
    except QuerySpecError:
        return None
    res = await run_sql(b.sql, params, {table}, max_rows=200, timeout_ms=20000, declared_params=b.params)
    if res.error:
        return None
    return [dict(zip(res.columns, r, strict=False)) for r in res.rows]


# ------------------------------------------------------------------ driver


async def _sample_rows(meta: dict) -> list[dict]:
    from backend.app.db.session import session_scope

    cols = [c["name"] for c in meta.get("columns") or []]
    async with session_scope() as s:
        rows = (await s.execute(sql_text(
            f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM graphrag."{meta["table_name"]}" '
            f"TABLESAMPLE SYSTEM (5) LIMIT {SAMPLE_ROWS}"))).all()
        if len(rows) < SAMPLE_ROWS:
            rows = (await s.execute(sql_text(
                f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM graphrag."{meta["table_name"]}" '
                f"LIMIT {SAMPLE_ROWS}"))).all()
    return [{c: (None if v is None else str(v)[:80]) for c, v in zip(cols, r, strict=True)} for r in rows]


def _prompt_table(meta: dict, analysis: dict, samples: list[dict]) -> dict:
    domains = meta.get("value_domains") or {}
    cols = []
    for c in meta.get("columns") or []:
        d = domains.get(c["name"]) or {}
        e = {"name": c["name"], "type": c.get("type"), "role": c.get("role"),
             "description": c.get("description")}
        if "values" in d:
            e["values"] = d["values"][:40] + (["..."] if len(d["values"]) > 40 else [])
        elif "min" in d:
            e["range"] = [d["min"], d["max"]]
        cols.append(e)
    catalog = [{"id": it["id"], "section": sec, "label": it.get("label"), "query": compact(it["query"])}
               for sec, it in items(meta) if isinstance(it.get("query"), dict) and sec != "measures"]
    return {"table": meta["table_name"], "description": meta.get("description"),
            "grain_description": meta.get("grain"), "row_count": analysis.get("row_count"),
            "columns": cols, "structure": summary_for_prompt(analysis),
            "sample_rows": samples, "catalog": catalog}


async def refine_table(meta: dict, router, *, all_types: dict[str, dict[str, str]],
                       log: Callable[[str], None] = print) -> dict:
    """Refine ``meta`` in place (tables must be loaded). Returns a report."""
    from backend.app.db.session import session_scope
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    table = meta["table_name"]
    async with session_scope() as s:
        analysis = await analyze_table(s, meta)
    meta["analysis"] = analysis
    log(f"[structured] {table}: grain {analysis.get('grain')} "
        f"(unique={analysis.get('grain_unique')}), repeated across "
        f"{summary_for_prompt(analysis).get('values_repeated_across') or 'nothing'}")

    system, user = PROMPTS["table_refine"](_prompt_table(meta, analysis, await _sample_rows(meta)))
    reply = _parse((await router.chat("table_refine", system=system, user=user)).text)
    calls = 1

    # column corrections
    by_name = {c["name"]: c for c in meta["columns"]}
    changed_cols = []
    for c in reply.get("columns") or []:
        if not isinstance(c, dict) or c.get("name") not in by_name:
            continue
        col = by_name[c["name"]]
        if c.get("role") in COLUMN_ROLES and c["role"] != col.get("role"):
            col["role"] = c["role"]
            changed_cols.append(c["name"])
        if c.get("description"):
            col["description"] = str(c["description"]).strip()
    # measures -> deterministic queries
    rejected = []
    meta["measures"] = []
    seen = set()
    for raw in (reply.get("measures") or [])[:MAX_MEASURES]:
        m, err = normalize_measure(raw, meta)
        if err or m["id"] in seen:
            rejected.append({"measure": raw.get("id") if isinstance(raw, dict) else raw,
                             "error": err or "duplicate id"})
            continue
        seen.add(m["id"])
        it = measure_item(m, meta)
        err = finalize_item(meta, "measures", it, all_types)
        if err:
            rejected.append({"measure": m["id"], "error": err})
            continue
        meta["measures"].append(it)
    meta["usage_notes"] = [str(n) for n in reply.get("usage_notes") or [] if n][:8]

    # test everything; one repair round for the stored specs that fail
    from backend.app.services.structured.value_domains import attach_value_domains

    async with session_scope() as s:
        await attach_value_domains(s, meta)
    results: dict[str, dict] = {}
    failures = []
    for sec, it in list(items(meta)):
        r = await test_item(meta, sec, it, analysis, all_types)
        results[it["id"]] = r
        if not r["ok"]:
            failures.append((sec, it, r))
    repaired, dropped = [], []
    if failures:
        fixable = [(sec, it, r) for sec, it, r in failures if sec != "measures"]
        fixed: dict[str, dict] = {}
        if fixable:
            system, user = PROMPTS["table_sql_repair"](
                table, _repair_context(meta, analysis),
                [{"id": it["id"], "error": r["error"], "params": it.get("params"), "query": it.get("query")}
                 for _s, it, r in fixable])
            calls += 1
            out = _parse((await router.chat("table_refine", system=system, user=user)).text)
            fixed = {str(x.get("id")): x for x in out.get("items") or [] if isinstance(x, dict)}
        for sec, it, r in failures:
            fx = fixed.get(it["id"])
            if fx and isinstance(fx.get("query"), dict):
                trial = {**it, "query": fx["query"], "params": fx.get("params") or it.get("params")}
                if not finalize_item(meta, sec, trial, all_types):
                    r2 = await test_item(meta, sec, trial, analysis, all_types)
                    if r2["ok"]:
                        it.update(trial)
                        it["repaired"] = True
                        results[it["id"]] = r2
                        repaired.append(it["id"])
                        continue
            meta[sec].remove(it)
            dropped.append({"id": it["id"], "section": sec, "error": r["error"]})
            meta.setdefault("dropped", []).append({"id": it["id"], "section": sec,
                                                   "query": compact(it.get("query") or {}),
                                                   "error": f"failed its data test: {r['error']}"})
    async with session_scope() as s:
        await attach_value_domains(s, meta)

    # typical questions: keep the ones whose query runs and returns rows
    meta["common_questions"] = await _test_questions(meta, reply.get("common_questions") or [], all_types)

    report = {"version": REFINE_VERSION, "llm_calls": calls,
              "measures": [it["id"] for it in meta["measures"]], "rejected_measures": rejected,
              "columns_changed": changed_cols, "repaired": repaired, "dropped": dropped,
              "tested": {k: {kk: vv for kk, vv in v.items() if kk != "params"} for k, v in results.items()},
              "questions": len(meta["common_questions"]), "seconds": round(time.monotonic() - t0, 1)}
    meta["refined"] = {"version": REFINE_VERSION, "model": _model(router), **{
        k: report[k] for k in ("measures", "rejected_measures", "columns_changed", "repaired", "dropped",
                               "questions")}}
    log(f"[structured] {table}: {len(meta['measures'])} measure(s), {len(repaired)} repaired, "
        f"{len(dropped)} dropped after testing, {len(meta['common_questions'])} typical question(s), "
        f"{calls} LLM call(s), {report['seconds']}s")
    return report


def _model(router) -> str:
    try:
        spec = router.task_spec("table_refine")
        return f"{spec.get('provider')}:{spec.get('model')}"
    except Exception:
        return "unknown"


def _repair_context(meta: dict, analysis: dict) -> str:
    lines = [f"- {c['name']} [{c.get('type')}, {c.get('role')}]: {c.get('description') or ''}"
             for c in meta["columns"]]
    return "\n".join(lines) + "\n\nSTRUCTURE (measured on the data):\n" + json.dumps(
        summary_for_prompt(analysis), ensure_ascii=False, default=str)


async def _test_questions(meta: dict, raw: list, all_types) -> list[dict]:
    from backend.app.services.structured.metadata import find_item
    from backend.app.services.structured.value_domains import check_params

    out = []
    measure_ids = {it["measure"]["id"]: it["id"] for it in meta.get("measures") or []}
    for q in raw[:12]:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        qid = measure_ids.get(str(q.get("query_id")), str(q.get("query_id")))
        hit = find_item(meta, qid)
        if hit is None:
            continue
        sec, it = hit
        params = q.get("params") if isinstance(q.get("params"), dict) else {}
        params, bad, _subs = check_params(params, it.get("params"), it.get("allowed_combinations"))
        if bad:
            continue
        try:
            built = compile_item(meta, sec, it, all_types)
        except (QuerySpecError, SqlGuardError):
            continue
        if any(p["name"] not in params for p in built.params):
            continue
        res = await run_sql(built.sql, params, built.tables, max_rows=50, timeout_ms=20000,
                            declared_params=it.get("params") or built.params)
        if res.error or not res.rows:
            continue
        out.append({"question": str(q["question"]), "query_id": qid, "params": params,
                    "rows": res.row_count})
    return out


async def refine_registered(router, *, tables: list[str] | None = None, refresh: bool = False,
                            cache=None, embed: EmbedFn | None = None, link_entities: bool = True,
                            log: Callable[[str], None] = print) -> list[dict]:
    """Refine loaded tables (all, or ``tables``) whose metadata is not yet
    refined at :data:`REFINE_VERSION`; update node, embedding, edges, cache."""
    from backend.app.db.graph_version import bump_version, current_version
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import rebuild_edges, registered_tables, upsert_node
    from backend.app.services.structured.metadata import metadata_text

    async with session_scope() as s:
        reg = await registered_tables(s)
    names = [n for n in (tables or sorted(reg)) if n in reg]
    all_types = {n: column_types(r["meta"]) for n, r in reg.items()}
    reports = []
    for name in names:
        meta = reg[name]["meta"]
        if not refresh and (meta.get("refined") or {}).get("version") == REFINE_VERSION:
            log(f"[structured] {name}: already refined ({REFINE_VERSION}), no LLM call")
            continue
        try:
            rep = await refine_table(meta, router, all_types=all_types, log=log)
        except Exception as exc:  # one table must not stop the rest
            log(f"[structured] FAILED refining {name}: {type(exc).__name__}: {exc}")
            reports.append({"table": name, "error": str(exc)})
            continue
        vec = (await embed([metadata_text(meta)]))[0] if embed is not None else None
        async with session_scope() as s:
            gv = await current_version(s)
            node_id = await upsert_node(s, meta, vec, gv, reg[name]["id"])
            await rebuild_edges(s, node_id, meta, gv, link_entities=link_entities)
        if cache is not None and meta.get("cache_key"):
            base = cache.get(meta["cache_key"], name) or {}
            entry = {**base, **meta}
            entry.setdefault("status", "complete")
            if entry.get("format"):
                cache.put(entry)
        reports.append({"table": name, **rep})
    if reports:
        async with session_scope() as s:
            await bump_version(s)
    return reports


__all__ = ["REFINE_VERSION", "measure_item", "normalize_measure", "refine_registered", "refine_table",
           "test_item"]
_ = SECTIONS  # measures is one of the item sections

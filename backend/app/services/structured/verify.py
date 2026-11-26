"""Independent verification of a LOADED table's metadata (task ``table_verify``).

Runs after :mod:`refine` has tested every stored query against the data, in
addition to those checks (which prove a query runs, returns rows and does not
double count -- not that it does what its words say). A fresh model call,
which did not write the metadata, checks it in two steps:

1. **Columns** (one call): each column's description, role and unit against
   what is MEASURED on the data -- rows with a value, distinct values, range,
   valid values and ``filled_only_where`` (the values of the period /
   report columns on the rows where the column is filled, so a column that
   only covers some periods is visible). Corrections are applied first, so
   step 2 reads verified column documentation.
2. **Stored queries** (batched calls): per item its label, description and
   parameters next to the compiled SQL, a SHAPE line written by
   :func:`describe_spec` from the spec itself (exact; independent of the
   words) and a SAMPLE result run with parameter values read from the table.
   Verdicts: ``ok``; ``fix_text`` (new label/description from what the SQL
   does); ``fix_query`` (corrected spec -- compiled, then put through every
   existing data test; a fix that fails is dropped); ``drop``.

Verdicts are tracked PER ITEM (``item["verify_verdict"]``). Allowed verdicts
depend on the item type and are enforced here, not only asked for: measures
are built deterministically from their definitions, so they get ``ok`` /
``fix_text`` / ``drop`` only (a ``split`` or ``fix_query`` on a measure is not
applied -- it can never create a catalog metric as a side effect). An item
whose verdict is missing or not allowed is asked again, within
:data:`MAX_VERDICT_RETRIES` extra calls; what is still unresolved stays
``pending`` and a later run verifies only those items. Every corrected or
added query is compiled and put through the data tests before retrieval may
select it. Nothing in the prompts names a dataset, a column or a value. The
result is stored as ``meta["verified"]``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.structured.metadata import (
    COLUMN_ROLES,
    compile_item,
    finalize_item,
    item_spec,
    items,
    spec_columns,
    sql_sha,
    time_columns,
)
from backend.app.services.structured.query_builder import QuerySpecError, compact
from backend.app.services.structured.sql_exec import run_sql
from backend.app.services.structured.sql_guard import SqlGuardError

VERIFY_VERSION = "tv-3"      # tv-3: fix_definition verdict for measures
VERDICTS = ("ok", "fix_text", "fix_query", "split", "drop")
# a measure is built from its definition: the verifier may ask for the
# DEFINITION to be repaired (bounded repair + retest), never for new SQL
ALLOWED_VERDICTS = {"measures": {"ok", "fix_text", "fix_definition", "drop"}}  # others: VERDICTS
MAX_VERDICT_RETRIES = 2         # extra calls for items left without an allowed verdict
ITEMS_PER_CALL = 8
SAMPLE_RESULT_ROWS = 5
MAX_CONTEXT_COLUMNS = 4       # period/report columns used for filled_only_where
MAX_CONTEXT_DISTINCT = 12     # ... only when they have this few distinct values
MAX_FILLED_VALUES = 8
EMPTY = "(empty)"


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


# ------------------------------------------------------------------ measured evidence


def context_columns(meta: dict) -> list[str]:
    """Columns that say WHICH rows a value belongs to: time columns and
    low-cardinality identifiers (report / file / version columns)."""
    stats = meta.get("column_stats") or {}
    out = []
    for name in time_columns(meta) + [c["name"] for c in meta.get("columns") or []
                                      if c.get("role") == "identifier"]:
        st = stats.get(name) or {}
        if name not in out and 1 < int(st.get("distinct") or 0) <= MAX_CONTEXT_DISTINCT:
            out.append(name)
    return out[:MAX_CONTEXT_COLUMNS]


async def filled_only_where(session, meta: dict) -> dict[str, dict[str, list]]:
    """{column: {context column: [values on the rows where column is filled]}},
    only where that is a strict subset of the context column's values (empty
    included). One query per context column, all columns at once."""
    table = meta["table_name"]
    ref = f"graphrag.{_q(table)}"
    cols = [c["name"] for c in meta.get("columns") or []]
    out: dict[str, dict[str, list]] = {}
    for ctx in context_columns(meta):
        others = [c for c in cols if c != ctx]
        if not others:
            continue
        cv = f"COALESCE(CAST({_q(ctx)} AS text), '{EMPTY}')"
        parts = [f"array_agg(DISTINCT {cv})"] + [
            f"array_agg(DISTINCT {cv}) FILTER (WHERE {_q(c)} IS NOT NULL)" for c in others]
        row = (await session.execute(sql_text(f"SELECT {', '.join(parts)} FROM {ref}"))).one()
        every = set(row[0] or [])
        for c, vals in zip(others, row[1:], strict=True):
            got = set(vals or [])
            if got and got != every:
                out.setdefault(c, {})[ctx] = sorted(got)[:MAX_FILLED_VALUES]
    return out


def _column_entry(meta: dict, c: dict, filled: dict) -> dict:
    stats = (meta.get("column_stats") or {}).get(c["name"]) or {}
    dom = (meta.get("value_domains") or {}).get(c["name"]) or {}
    e: dict[str, Any] = {"name": c["name"], "type": c.get("type"), "role": c.get("role"),
                         "unit": c.get("unit"), "description": c.get("description"),
                         "rows_with_value": stats.get("non_null"), "distinct": stats.get("distinct")}
    if "min" in stats:
        e["min"], e["max"] = stats["min"], stats["max"]
    if dom.get("values"):
        e["values"] = dom["values"][:15] + (["..."] if len(dom["values"]) > 15 else [])
    elif c.get("examples"):
        e["examples"] = c["examples"][:5]
    if filled.get(c["name"]):
        e["filled_only_where"] = filled[c["name"]]
    return e


# ------------------------------------------------------------------ SHAPE (deterministic)


def _col(raw) -> str:
    if isinstance(raw, dict):
        name = str(raw.get("column") or "")
        return f"{raw['fn']}({name})" if raw.get("fn") else name
    return str(raw)


def describe_spec(spec: dict | None, *, limit_note: bool = True) -> str:
    """What a query spec does, in plain words, from the spec alone."""
    if not isinstance(spec, dict):
        return "stored SQL (no spec)"
    sel = spec.get("select") or []
    aggs = [s for s in sel if isinstance(s, dict) and s.get("agg")]
    plain = [_col(s) for s in sel if not (isinstance(s, dict) and s.get("agg"))]
    groups = list(dict.fromkeys([_col(g) for g in spec.get("group_by") or []] + (plain if aggs else [])))
    parts = []
    if aggs:
        calc = []
        for a in aggs:
            cols = a.get("columns") or ([a.get("column")] if a.get("column") not in (None, "", "*") else [])
            calc.append(f"{str(a['agg']).upper()}({', '.join(_col(c) for c in cols) or '*'})"
                        + (f" as {a['as']}" if a.get("as") else ""))
        parts.append(("one row per " + ", ".join(groups) if groups else "ONE row over all matching rows")
                     + "; computes " + ", ".join(calc))
    elif spec.get("distinct_on"):
        parts.append(f"one row per {', '.join(_col(c) for c in spec['distinct_on'])} (the first by the "
                     "order below); returns " + ", ".join(plain))
    elif spec.get("distinct"):
        parts.append("the distinct values of " + ", ".join(plain))
    else:
        parts.append("returns the matching rows' " + ", ".join(plain))
    if spec.get("join"):
        j = spec["join"]
        parts.append(f"joined to {j.get('table')} on "
                     + ", ".join(f"{p.get('column')} = {j.get('table')}.{p.get('other_column')}"
                                 for p in j.get("on") or []))
    conds = []
    for w in spec.get("where") or []:
        if not isinstance(w, dict):
            continue
        op = str(w.get("op") or "=")
        col = _col(w.get("column"))
        if op in {"is_null", "not_null"}:
            conds.append(f"{col} {'is empty' if op == 'is_null' else 'has a value'} (fixed)")
        elif w.get("param") not in (None, ""):
            p = w["param"]
            p = " and ".join(f":{x}" for x in p) if isinstance(p, list) else f":{p}"
            conds.append(f"{col} {op} {p} (user's choice{', optional' if w.get('optional') else ''})")
        else:
            conds.append(f"{col} {op} {json.dumps(w.get('value'), ensure_ascii=False)} (FIXED)")
    parts.append("filters: " + ("; ".join(conds) if conds else "none (all rows)"))
    order = [f"{_col(o.get('column') if isinstance(o, dict) else o)} "
             f"{(o.get('dir') if isinstance(o, dict) else None) or 'asc'}"
             for o in spec.get("order_by") or []]
    if order:
        parts.append("ordered by " + ", ".join(order))
    lim = spec.get("limit")
    if lim not in (None, ""):
        n = f":{lim['param']}" if isinstance(lim, dict) else str(lim)
        parts.append(f"only the first {n} rows" + (" (NOT a complete list or total)" if limit_note else ""))
    return "; ".join(parts)


# ------------------------------------------------------------------ helpers


def _parse(text: str) -> dict:
    from backend.app.services.structured.refine import _parse as parse

    return parse(text)


def _model(router) -> str:
    try:
        spec = router.task_spec("table_verify")
        return f"{spec.get('provider')}:{spec.get('model')}"
    except Exception:
        return "unknown"


def verify_enabled(router) -> bool:
    """On unless the ``table_verify`` task is missing or ``enabled: false``."""
    try:
        spec = router.task_spec("table_verify")
    except Exception:
        return False
    return bool(spec) and spec.get("enabled", True) is not False


async def _sample_result(meta: dict, section: str, item: dict, all_types) -> dict:
    """Run the item's STORED SQL (the text retrieval runs) on example values."""
    from types import SimpleNamespace

    from backend.app.services.structured.refine import example_params

    sql = item.get("sql")
    if not sql:
        try:
            built = compile_item(meta, section, item, all_types)
        except (QuerySpecError, SqlGuardError) as exc:
            return {"sql": None, "error": f"does not compile: {exc}"}
        sql, declared, added, tables = built.sql, built.params, built.added_conditions, built.tables
    else:
        declared, added = item.get("params") or [], item.get("added_conditions") or []
        tables = set(item.get("tables") or []) or {meta["table_name"]} | (
            {item["other_table"]} if item.get("other_table") else set())
    params = await example_params(meta, section, item, SimpleNamespace(params=declared))
    if params is None:
        return {"sql": sql, "added_conditions": list(added),
                "note": "not run (no example row for its parameters)"}
    res = await run_sql(sql, params, tables, max_rows=SAMPLE_RESULT_ROWS, timeout_ms=20000,
                        declared_params=declared)
    out: dict[str, Any] = {"sql": sql, "params_used": params, "added_conditions": list(added)}
    if res.error:
        out["error"] = res.error
    else:
        out["columns"] = res.columns
        out["rows"] = res.rows[:SAMPLE_RESULT_ROWS]
        out["more_rows"] = bool(res.truncated)
    return out


def allowed_verdicts(section: str) -> set[str]:
    return ALLOWED_VERDICTS.get(section, set(VERDICTS))


def pending_items(meta: dict) -> list[tuple[str, dict]]:
    """Items without an accepted verdict at the current VERIFY_VERSION."""
    out = []
    for sec, it in items(meta):
        v = it.get("verify_verdict") or {}
        if v.get("version") != VERIFY_VERSION or v.get("verdict") in (None, "pending", "stale") or (
                v.get("sql_sha") and it.get("sql") and v["sql_sha"] != sql_sha(it["sql"])):
            out.append((sec, it))
    return out


def _item_entry(meta: dict, section: str, item: dict, sample: dict,
                other_columns: dict[str, list[dict]] | None = None) -> dict:
    spec = item_spec(section, item)
    e = {"id": item["id"], "section": section, "fixable": section != "measures",
         "allowed_verdicts": sorted(allowed_verdicts(section)),
         "label": item.get("label"), "description": item.get("description"),
         "params": [{k: p.get(k) for k in ("name", "type", "column", "description") if p.get(k)}
                    for p in item.get("params") or []],
         "sql": sample.get("sql") or item.get("sql"), "shape": describe_spec(spec)}
    if sample.get("added_conditions"):
        e["added_by_program"] = sample["added_conditions"]
    if section == "lookups" and item.get("other_table"):
        e["other_table"] = item["other_table"]
        e["other_table_columns"] = (other_columns or {}).get(item["other_table"]) or []
    if section != "measures" and isinstance(item.get("query"), dict):
        e["query"] = item["query"]
    if sample.get("error"):
        e["sample"] = {"error": sample["error"]}
    elif "rows" in sample:
        e["sample"] = {"params": sample.get("params_used"), "columns": sample["columns"],
                       "rows": sample["rows"], "more_rows": sample.get("more_rows")}
    else:
        e["sample"] = {"note": sample.get("note")}
    return e


# ------------------------------------------------------------------ driver


async def verify_table(meta: dict, router, *, analysis: dict, all_types: dict[str, dict[str, str]],
                       partners: dict[str, dict] | None = None, only_pending: bool = False,
                       log: Callable[[str], None] = print) -> dict:
    """Verify ``meta`` in place (tables loaded, items already data-tested).
    ``only_pending``: skip the column step and verify only the items a
    previous run left without a verdict. Returns a report; also stored as
    ``meta["verified"]``."""
    from backend.app.db.session import session_scope
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.structured.refine import _sample_rows, test_item

    t0 = time.monotonic()
    table = meta["table_name"]
    calls = 0

    # 1. columns
    async with session_scope() as s:
        filled = await filled_only_where(s, meta)
    reply: dict = {}
    if not only_pending:
        cols_in = [_column_entry(meta, c, filled) for c in meta.get("columns") or []]
        system, user = PROMPTS["table_verify_columns"]({
            "table": table, "description": meta.get("description"), "grain": meta.get("grain"),
            "row_count": (analysis or {}).get("row_count") or meta.get("row_count"),
            "columns": cols_in, "sample_rows": await _sample_rows(meta)})
        reply = _parse((await router.chat("table_verify", system=system, user=user)).text)
        calls += 1
    by_name = {c["name"]: c for c in meta.get("columns") or []}
    column_fixes = []
    for c in reply.get("columns") or []:
        if not isinstance(c, dict) or c.get("name") not in by_name:
            continue
        col, before = by_name[c["name"]], {k: by_name[c["name"]].get(k) for k in ("role", "description", "unit")}
        if c.get("description"):
            col["description"] = str(c["description"]).strip()
        if c.get("role") in COLUMN_ROLES:
            col["role"] = c["role"]
        if "unit" in c and (c["unit"] is None or isinstance(c["unit"], str)):
            col["unit"] = c["unit"] or None
        after = {k: col.get(k) for k in before}
        if after != before:
            column_fixes.append({"column": c["name"], "problem": str(c.get("problem") or ""),
                                 "before": before, "after": after})
    table_fixes = {}
    for k, key in (("table_description", "description"), ("grain", "grain")):
        v = reply.get(k)
        if isinstance(v, str) and v.strip() and v.strip() != (meta.get(key) or ""):
            table_fixes[key] = {"before": meta.get(key), "after": v.strip()}
            meta[key] = v.strip()

    # 2. stored queries, against the verified columns
    verified_cols = [_column_entry(meta, c, filled) for c in meta.get("columns") or []]
    ctx = {"table": table, "description": meta.get("description"), "grain": meta.get("grain"),
           "usage_notes": meta.get("usage_notes") or [], "columns": verified_cols}
    todo = pending_items(meta) if only_pending else [(sec, it) for sec, it in items(meta)]
    # a lookup reads its partner table too: show that table's documented columns
    others: dict[str, list[dict]] = {}
    partners = {it.get("other_table") for sec, it in todo if sec == "lookups" and it.get("other_table")}
    if partners:
        from backend.app.services.structured.db_load import registered_tables

        async with session_scope() as s:
            reg = await registered_tables(s)
        for name in partners:
            pm = (reg.get(name) or {}).get("meta") or {}
            others[name] = [{k: c.get(k) for k in ("name", "type", "role", "unit", "description")
                             if c.get(k) is not None} for c in pm.get("columns") or []]
    verdicts: dict[str, dict] = {}
    rejected_verdicts: list[dict] = []
    samples: dict[str, dict] = {}

    async def _ask(batch: list[tuple[str, dict]], note: str | None = None) -> None:
        nonlocal calls
        entries = []
        for sec, it in batch:
            if it["id"] not in samples:
                samples[it["id"]] = await _sample_result(meta, sec, it, all_types)
            entries.append(_item_entry(meta, sec, it, samples[it["id"]], others))
        used = set().union(*(spec_columns(item_spec(sec, it), table) for sec, it in batch)) \
            if batch else set()
        ctx_b = {**ctx, "columns": [c for c in verified_cols if c["name"] in used],
                 "other_columns": [{k: c.get(k) for k in ("name", "type", "role", "unit", "description")
                                    if c.get(k) is not None}
                                   for c in verified_cols if c["name"] not in used]}
        if note:
            ctx_b["note"] = note
        system, user = PROMPTS["table_verify_items"](ctx_b, entries)
        out = _parse((await router.chat("table_verify", system=system, user=user)).text)
        calls += 1
        secs = {it["id"]: sec for sec, it in batch}
        for v in out.get("items") or []:
            if not isinstance(v, dict) or str(v.get("id")) not in secs:
                continue
            vid = str(v["id"])
            if v.get("verdict") in allowed_verdicts(secs[vid]):
                verdicts[vid] = v
            else:
                rejected_verdicts.append({"id": vid, "verdict": v.get("verdict"),
                                          "allowed": sorted(allowed_verdicts(secs[vid]))})

    for i in range(0, len(todo), ITEMS_PER_CALL):
        await _ask(todo[i:i + ITEMS_PER_CALL])
    for _r in range(MAX_VERDICT_RETRIES):
        missing = [(sec, it) for sec, it in todo if it["id"] not in verdicts]
        if not missing:
            break
        for i in range(0, len(missing), ITEMS_PER_CALL):
            await _ask(missing[i:i + ITEMS_PER_CALL],
                       note=("These items got no verdict, or a verdict their type does not allow, in the "
                             "previous reply. Give exactly one verdict per item, chosen from its "
                             "allowed_verdicts."))

    ok, text_fixed, query_fixed, dropped, unverified = [], [], [], [], []
    to_repair: list[tuple[dict, dict]] = []
    renames, added, add_failed = [], [], []
    for sec, it in todo:
        v = verdicts.get(it["id"])
        if v is None:
            unverified.append(it["id"])
            it["verify_verdict"] = {"verdict": "pending", "version": VERIFY_VERSION}
            continue
        verdict, reason = v["verdict"], str(v.get("reason") or "")
        it["verify_verdict"] = {"verdict": verdict, "version": VERIFY_VERSION, "reason": reason[:500]}
        renamed = None
        if verdict in {"fix_text", "fix_query", "split", "ok"} and sec != "measures" and v.get("new_id"):
            renamed = _rename(meta, it, v["new_id"])
            if renamed:
                renames.append({"from": it["aliases"][-1], "to": renamed, "reason": reason})
        if verdict == "split":
            # never reached for a measure (not an allowed verdict there)
            target = sec if sec in {"metrics", "filters", "queries"} else "metrics"
            add, err = await _add_item(meta, target, v.get("add") or {}, analysis, all_types,
                                       partners=partners)
            if add:
                added.append({"id": add["id"], "from": it["id"], "reason": reason,
                              "query": compact(add["query"])})
            else:
                add_failed.append({"from": it["id"], "reason": reason, "error": err})
            verdict = "fix_text"
        if verdict == "ok":
            ok.append(it["id"])
        elif verdict == "fix_text":
            change = {"id": it["id"], "reason": reason,
                      "before": {"label": it.get("label"), "description": it.get("description")}}
            if v.get("label"):
                it["label"] = str(v["label"]).strip()
            if v.get("description"):
                it["description"] = str(v["description"]).strip()
            change["after"] = {"label": it.get("label"), "description": it.get("description")}
            it["verified"] = "fix_text"
            text_fixed.append(change)
        elif verdict == "fix_definition":
            to_repair.append((it, {"error": f"verifier: {reason}", "stage": "verification"}))
        elif verdict == "fix_query":
            err = "no corrected query given"
            if isinstance(v.get("query"), dict):
                trial = {**it, "query": v["query"], "params": v.get("params") or it.get("params"),
                         "label": str(v.get("label") or it.get("label") or ""),
                         "description": str(v.get("description") or it.get("description") or "")}
                trial.pop("verification", None)
                r = await test_item(meta, sec, trial, analysis or {}, all_types, partners=partners)
                err = None if r["status"] == "passed" else r.get("error") or f"data test {r['status']}"
                if not err:
                    query_fixed.append({"id": it["id"], "reason": reason,
                                        "before": {"label": it.get("label"),
                                                   "query": compact(it.get("query") or {})},
                                        "after": {"label": trial["label"],
                                                  "query": compact(trial["query"])}})
                    it.update(trial)
                    it["verified"] = "fix_query"
                    continue
            _drop(meta, sec, it, f"{reason} (fix failed: {err})", verdict="fix_query")
            dropped.append({"id": it["id"], "section": sec, "reason": reason, "fix_error": err})
        else:  # drop
            _drop(meta, sec, it, reason, verdict="drop")
            dropped.append({"id": it["id"], "section": sec, "reason": reason})

    definition_repairs: list[dict] = []
    if to_repair:
        # repairable findings: bounded definition repair, rebuilt and re-tested;
        # what stays unresolved moves to failed_measures with the finding
        from backend.app.services.structured.refine import repair_measures

        rcalls, fixed = await repair_measures(meta, router, to_repair, analysis=analysis or {},
                                              all_types=all_types, partners=partners, results={})
        calls += rcalls
        for it, r in to_repair:
            definition_repairs.append({"id": it["id"], "finding": r["error"],
                                       "outcome": "repaired" if it["id"] in fixed else "unresolved"})
            if it["id"] not in fixed:
                dropped.append({"id": it["id"], "section": "measures", "reason": r["error"],
                                "stage": "verification", "repair": "unresolved"})

    report = {"version": VERIFY_VERSION, "model": _model(router), "llm_calls": calls,
              "definition_repairs": definition_repairs,
              "columns_fixed": column_fixes, "table_fixed": table_fixes,
              "ok": ok, "text_fixed": text_fixed, "query_fixed": query_fixed,
              "dropped": dropped, "unverified": unverified,
              "renamed": renames, "added": added, "add_failed": add_failed,
              "rejected_verdicts": rejected_verdicts, "only_pending": only_pending,
              "seconds": round(time.monotonic() - t0, 1)}
    for _sec, it in items(meta):       # a verdict holds for the SQL it reviewed
        vv = it.get("verify_verdict")
        if isinstance(vv, dict) and vv.get("verdict") not in (None, "pending", "stale") \
                and vv.get("version") == VERIFY_VERSION and it.get("sql") and not vv.get("sql_sha"):
            vv["sql_sha"] = sql_sha(it["sql"])
    prev = meta.get("verified") or {}
    stored = {k: v for k, v in report.items() if k != "seconds"}
    if only_pending and prev.get("version") == VERIFY_VERSION:
        # merge: earlier verdicts stand, this run resolved (some of) the pending items
        for k in ("columns_fixed", "ok", "text_fixed", "query_fixed", "dropped", "renamed", "added",
                  "add_failed", "rejected_verdicts"):
            stored[k] = [*(prev.get(k) or []), *stored[k]]
        stored["table_fixed"] = {**(prev.get("table_fixed") or {}), **stored["table_fixed"]}
        stored["llm_calls"] = int(prev.get("llm_calls") or 0) + calls
    stored["pending"] = [it["id"] for _s, it in pending_items(meta)]
    meta["verified"] = stored
    log(f"[structured] {table}: verified -- {len(column_fixes)} column(s) corrected, "
        f"{len(ok)} ok, {len(text_fixed)} re-described, {len(query_fixed)} query fixed, "
        f"{len(dropped)} dropped, {len(renames)} renamed, {len(added)} added, "
        f"{len(unverified)} still pending; {calls} LLM call(s), "
        f"{report['seconds']}s")
    return report


def _rename(meta: dict, item: dict, new_id: Any) -> str | None:
    """Give ``item`` a new id (its old id becomes an alias, so references and
    stored questions keep working). Returns the new id, or None."""
    from backend.app.services.structured.profiler import _norm_id

    if not isinstance(new_id, str) or not new_id.strip():
        return None
    used = {it["id"] for _s, it in items(meta)} | {a for _s, it in items(meta) for a in it.get("aliases") or []}
    old = item["id"]
    want = _norm_id(new_id, set())
    if want == old:
        return None
    nid = _norm_id(want, used)
    item["aliases"] = [*(item.get("aliases") or []), old]
    item["id"] = nid
    for q in meta.get("common_questions") or []:
        if q.get("query_id") == old:
            q["query_id"] = nid
    return nid


async def _add_item(meta: dict, section: str, raw: dict, analysis: dict, all_types, *,
                    partners: dict[str, dict] | None = None) -> tuple[dict | None, str | None]:
    """A new item the verification asked for (split of a metric/filter/query/
    lookup): compiled, then put through every data test; it is published
    only when it passed. Returns (item, None) or (None, error)."""
    if section == "measures":
        return None, "a split never creates a measure"
    from backend.app.services.structured.profiler import _norm_id
    from backend.app.services.structured.refine import test_item

    if not isinstance(raw, dict) or not isinstance(raw.get("query"), dict):
        return None, "no query given"
    used = {it["id"] for _s, it in items(meta)} | {a for _s, it in items(meta) for a in it.get("aliases") or []}
    new = {"id": _norm_id(raw.get("id") or raw.get("label") or "query", used),
           "label": str(raw.get("label") or "").strip(), "description": str(raw.get("description") or "").strip(),
           "query": raw["query"], "params": raw.get("params") or [], "verified": "added",
           "verify_verdict": {"verdict": "added", "version": VERIFY_VERSION}}
    err = finalize_item(meta, section, new, all_types, partners=partners)
    if not err:
        r = await test_item(meta, section, new, analysis or {}, all_types, partners=partners)
        err = None if r["status"] == "passed" else r.get("error") or f"data test {r['status']}"
    if err:
        return None, err
    meta.setdefault(section, []).append(new)
    return new, None


def _drop(meta: dict, section: str, item: dict, reason: str, *, verdict: str | None = None) -> None:
    from backend.app.services.structured.metadata import record_rejection

    record_rejection(meta, section, item, stage="verification", reason=reason, verdict=verdict,
                     extra={"verify_version": VERIFY_VERSION})


__all__ = [
    "VERIFY_VERSION",
    "context_columns",
    "describe_spec",
    "filled_only_where",
    "verify_enabled",
    "verify_table",
]

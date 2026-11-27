"""The per-table METADATA FILE (a JSON dict) and helpers around it.

Shape (``format`` 2)::

    {
      "format": 1, "status": "complete", "cache_key": "...", "profile_version": "tp-1",
      "table_name": "prices",
      "source": {"rel_path": ..., "fmt": "csv", "file_sha": ..., "dataset": ..., "snapshot": ...},
      "row_count": 58822,
      "description": "...", "grain": "one row per commodity per month",
      "columns": [{"name", "source_name", "type", "role", "description", "unit",
                   "entity_kind", "examples", "class_iri", "class_label"}],
      "metrics":  [ITEM], "filters": [ITEM], "queries": [ITEM], "lookups": [ITEM],
      "ontology": {"MATCHES FOUND": [...], "MATCH NOT FOUND": [...]},
      "dropped": [{"id", "section", "sql", "error"}],
      "model": "openai:gpt-4.1", "prompt_version": "...", "edited": false,
      "profile": {TableProfile}
    }

Every ITEM keeps its query as a JSON SPEC with the parameters in a separate
list; the SQL is compiled from the spec by :mod:`query_builder` (deterministic,
no LLM)::

    {"id", "label", "description",
     "query": {QUERY SPEC},                      # the source of truth
     "params": [{"name", "type", "column", "description", "example"}],
     "sql": "SELECT ...",                         # compiled from "query", for reading only
     "columns": [...], "class_iri", "class_label",
     # lookups only:
     "other_table", "join": [{"column", "other_column"}]}

Metric/filter/query items read only their own table; a lookup reads its own
table and ``other_table``. Each lookup is stored in BOTH tables' files.

VERIFIED CATALOG (``catalog_version`` :data:`CATALOG_VERSION`). The spec is
kept for ingestion, repair, inspection and rebuilding; RETRIEVAL runs the
stored ``sql`` text itself, never a recompiled one. Ingestion compiles the
spec once all the metadata it depends on exists (the table analysis; for a
lookup the total-row markers of BOTH tables, kept in ``partner_totals``), runs
the data tests on that exact text and records the outcome::

    "verification": {"status": "passed" | "failed" | "not_exercised",
                     "sql_sha": sha256 of the tested sql, "catalog_version": ...,
                     "error"?: ..., "checks"?: [...]}

Only an item whose status is ``passed`` for its current ``sql`` (same hash)
in a table at the current catalog version can be selected
(:func:`selectable`). Older entries are reported unavailable with the rebuild
action (:func:`rebuild_action`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from backend.app.services.structured.query_builder import (
    Built,
    QuerySpecError,
    build_sql,
    spec_join_for,
)
from backend.app.services.structured.sql_guard import SqlGuardError, check_sql, rename_tables

CATALOG_VERSION = "cv-1"
TEST_STATUSES = ("passed", "failed", "not_exercised")
SECTIONS = ("measures", "metrics", "filters", "queries", "lookups")
SECTION_TITLES = {"measures": "Measures", "metrics": "Metrics", "filters": "Filters", "queries": "Queries", "lookups": "Lookups"}
COLUMN_ROLES = ("identifier", "dimension", "measure", "time", "attribute", "text")


def items(meta: dict, sections: tuple[str, ...] = SECTIONS) -> Iterator[tuple[str, dict]]:
    for sec in sections:
        for it in meta.get(sec) or []:
            yield sec, it


def find_item(meta: dict, item_id: str) -> tuple[str, dict] | None:
    """By id, else by a former id (``aliases``: ids the verification renamed)."""
    hit = next(((s, it) for s, it in items(meta) if it.get("id") == item_id), None)
    return hit or next(((s, it) for s, it in items(meta) if item_id in (it.get("aliases") or [])), None)


def column_names(meta: dict) -> set[str]:
    return {c["name"] for c in meta.get("columns") or []}


def item_tables(meta: dict, section: str, item: dict) -> set[str]:
    tables = {meta["table_name"]}
    if section == "lookups" and item.get("other_table"):
        tables.add(item["other_table"])
    return tables


def time_columns(meta: dict) -> list[str]:
    """Columns that place a row in time (role time first, then date types)."""
    cols = meta.get("columns") or []
    first = [c["name"] for c in cols if c.get("role") == "time"]
    return first + [c["name"] for c in cols
                    if c.get("type") in {"date", "timestamp"} and c["name"] not in first]


def column_types(meta: dict) -> dict[str, str]:
    return {c["name"]: c.get("type") or "text" for c in meta.get("columns") or []}


def item_spec(section: str, item: dict) -> dict | None:
    """The item's query spec, with a lookup's join filled in from the item."""
    spec = item.get("query")
    if not isinstance(spec, dict):
        return None
    if section == "lookups" and not spec.get("join") and not spec.get("from"):
        join = spec_join_for(item)
        if join:
            spec = {**spec, "join": join}
    return spec


def compile_item(meta: dict, section: str, item: dict,
                 types_by_table: dict[str, dict[str, str]], *,
                 totals: dict[str, dict[str, str]] | None = None) -> Built:
    """Deterministic SQL for one item (raises QuerySpecError/SqlGuardError).
    ``types_by_table``: table -> {column: type} for the item's table(s).
    ``totals``: table -> {column: total value}; defaults to this table's own
    (from its analysis; a lookup adds its stored ``partner_totals``)."""
    spec = item_spec(section, item)
    tables = item_tables(meta, section, item)
    if spec is None:  # legacy item: stored SQL
        res = check_sql(item.get("sql") or "", tables)
        return Built(sql=res.sql, params=[{"name": p, "type": "text", "column": None}
                                          for p in res.params], tables=res.tables)
    types = {t: types_by_table[t] for t in tables if t in types_by_table}
    types.setdefault(meta["table_name"], column_types(meta))
    if totals is None:
        totals = item_totals(meta, section, item)
    from backend.app.services.structured.analysis import total_groups

    built = build_sql(spec, meta["table_name"], types,
                      time_columns={meta["table_name"]: time_columns(meta)}, totals=totals,
                      total_groups={meta["table_name"]: total_groups(meta)})
    check_sql(built.sql, tables)      # defence in depth: the guard still applies
    return built


def item_totals(meta: dict, section: str, item: dict) -> dict[str, dict[str, str]]:
    """Total-row markers for every table the item reads: its own table's (from
    the analysis) and, for a lookup, the partner's (``partner_totals``, set
    when the lookup is finalized with the partner's metadata at hand)."""
    from backend.app.services.structured.analysis import total_markers

    out = {meta["table_name"]: total_markers(meta)}
    if section == "lookups" and item.get("other_table"):
        out[item["other_table"]] = dict(item.get("partner_totals") or {})
    return out


def sql_sha(sql: str | None) -> str:
    return hashlib.sha256((sql or "").encode("utf-8")).hexdigest()


def rename_params(values: dict, ren: dict[str, str]) -> dict:
    """Apply parameter renames to a {name: value} map; a range parameter
    split into two bounds ("a" -> "a_from,a_to") takes a 2-value list."""
    out = {}
    for k, v in values.items():
        new = ren.get(k, k)
        if "," in new:
            lo, hi = new.split(",", 1)
            if isinstance(v, list) and len(v) == 2:
                out[lo], out[hi] = v
            else:
                out[lo] = out[hi] = v
        else:
            out[new] = v
    return out


def finalize_item(meta: dict, section: str, item: dict,
                  types_by_table: dict[str, dict[str, str]], *,
                  partners: dict[str, dict] | None = None) -> str | None:
    """Compile a spec item in place: ``sql`` <- compiled statement, ``params``
    <- the parameters it binds (names/types/columns from the compiler,
    descriptions/examples kept). Returns an error message instead on failure.

    ``partners`` (table -> metadata) supplies a lookup's partner table, whose
    total-row markers become part of the SQL. A change of SQL text clears the
    item's verification: it must be tested again before retrieval may use it."""
    from backend.app.services.structured.analysis import total_markers
    from backend.app.services.structured.query_builder import merge_params

    if not isinstance(item.get("query"), dict):
        return "item has no query spec"
    from backend.app.services.structured.query_builder import sanitize_param_names

    ren = sanitize_param_names(item)       # mechanical name fixes, before compiling
    if ren:
        for q in meta.get("common_questions") or []:
            if isinstance(q, dict) and q.get("query_id") == item.get("id") and isinstance(q.get("params"), dict):
                q["params"] = rename_params(q["params"], ren)
    if section == "lookups" and item.get("other_table") and partners \
            and item["other_table"] in partners:
        item["partner_totals"] = total_markers(partners[item["other_table"]])
    try:
        built = compile_item(meta, section, item, types_by_table)
    except (QuerySpecError, SqlGuardError) as exc:
        return str(exc)
    item["params"] = merge_params(built, item.get("params"))
    if item.get("sql") != built.sql:
        item.pop("verification", None)
    item["sql"] = built.sql
    item["tables"] = sorted(built.tables)
    item["empty_excluded"] = list(built.empty_excluded)
    item["added_conditions"] = list(built.added_conditions)
    return None


def record_test(item: dict, status: str, **detail: Any) -> dict:
    """Store a data-test outcome for the item's CURRENT sql."""
    if status not in TEST_STATUSES:
        raise ValueError(f"unknown test status {status!r}")
    item["verification"] = {"status": status, "sql_sha": sql_sha(item.get("sql")),
                            "catalog_version": CATALOG_VERSION,
                            **{k: v for k, v in detail.items() if v not in (None, [], {})}}
    return item["verification"]


def rebuild_action(table: str, reason: str | None = None) -> str:
    """The command that can make an unavailable item available again."""
    if reason and reason.startswith("semantic review"):
        return (f"uv run python -m backend.app.cli table-metadata-refine --table {table} "
                "(paid: verifies the items awaiting a semantic verdict; no re-profiling)")
    if reason and reason.startswith("held:"):
        return (f"review the hold, then uv run python -m backend.app.cli table-metadata-export "
                f"--table {table} / table-metadata-import")
    return (f"uv run python -m backend.app.cli table-metadata-refine --table {table} --relink "
            "(no LLM: re-compiles and re-tests the stored queries)")


def selectable(meta: dict, item: dict, versions: dict[str, str | None] | None = None,
               *, semantic: bool = True) -> tuple[bool, str | None]:
    """May retrieval run this item? (ok, reason when not). ``versions``
    (table -> current data version): an item whose test read a different
    version of any table is stale. ``semantic=False`` checks everything but
    the semantic review (for structural uses that must survive a pending
    review: common questions, concept mapping, the coverage ledger)."""
    if meta.get("catalog_version") != CATALOG_VERSION:
        return False, (f"the table's catalog is version {meta.get('catalog_version') or 'none'}, "
                       f"not {CATALOG_VERSION} (stored SQL not verified)")
    v = item.get("verification") or {}
    if not item.get("sql"):
        return False, "no stored SQL"
    if not isinstance(item.get("params"), list) or any(
            not isinstance(p, dict) or not p.get("name") or not p.get("type") for p in item["params"]):
        return False, "no usable parameter declaration"
    if v.get("status") != "passed":
        return False, f"data test {v.get('status') or 'not run'}" + (
            f": {v['error']}" if v.get("error") else "")
    if v.get("sql_sha") != sql_sha(item["sql"]):
        return False, "stored SQL changed after it was tested"
    for t, was in (v.get("data_versions") or {}).items():
        now = (versions or {}).get(t)
        if versions is not None and now and was and now != was:
            return False, f"table {t} changed after the query was tested"
    hold = item.get("hold")
    if hold:
        return False, "held: " + str((hold.get("reason") if isinstance(hold, dict) else hold) or "no reason given")
    sem = semantic_status(meta, item) if semantic else "not_required"
    if sem in {"pending", "unreviewed"}:
        return False, ("semantic review pending (passed its data test; its label/description have not been "
                       "checked against the SQL)")
    if sem == "stale":
        return False, "semantic review stale (the SQL changed after its label/description were reviewed)"
    return True, None


def semantic_review_required(meta: dict) -> bool:
    """Does this table's catalog need a semantic verdict before an item may
    be selected? Recorded per table by the refine run (``semantic_review``,
    from whether the ``table_verify`` task is enabled); a table verified
    earlier counts as requiring it."""
    sr = meta.get("semantic_review")
    if isinstance(sr, dict) and "required" in sr:
        return bool(sr["required"])
    return bool(meta.get("verified"))


def record_rejection(meta: dict, section: str, item: dict, *, stage: str, reason: str,
                     verdict: str | None = None, extra: dict | None = None) -> dict:
    """Remove ``item`` from its section and keep a DURABLE record of it in
    ``meta["dropped"]``: everything needed to repair or re-test it later
    without reconstructing it from another item (label, description, query
    spec, the measure DEFINITION, stored SQL, parameters, the verdict and its
    reason, the stage that rejected it, when)."""
    import datetime as _dt

    from backend.app.services.structured.query_builder import compact

    if item in meta.get(section) or []:
        meta[section].remove(item)
    rec = {"id": item.get("id"), "section": section, "stage": stage, "reason": reason,
           "error": reason if reason.startswith(stage) else f"{stage}: {reason}",
           "at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
           "label": item.get("label"), "description": item.get("description"),
           "query": compact(item.get("query") or {}), "sql": item.get("sql"),
           "params": item.get("params") or []}
    if isinstance(item.get("measure"), dict):
        rec["definition"] = item["measure"]
    if verdict:
        rec["verdict"] = verdict
    if isinstance(item.get("verification"), dict):
        rec["data_test"] = {k: item["verification"].get(k) for k in ("status", "error", "note")
                            if item["verification"].get(k) is not None}
    rec.update(extra or {})
    meta.setdefault("dropped", []).append(rec)
    return rec


_REPORT_VERDICTS = (("ok", "ok"), ("text_fixed", "fix_text"), ("query_fixed", "fix_query"))


def pin_reviewed_sql(meta: dict) -> int:
    """Before anything rebuilds or re-tests a catalog: every item with a
    semantic verdict (an older catalog's verdict comes from its verification
    report) records the SQL that verdict was given on, so a later change of
    that SQL makes the verdict stale. Returns the number of verdicts pinned."""
    n = 0
    for _s, it in items(meta):
        if not it.get("sql"):
            continue
        v = it.get("verify_verdict")
        if not (isinstance(v, dict) and v.get("verdict")):
            legacy = semantic_status(meta, it)
            if legacy in {"not_required", "unreviewed", "pending", "stale"}:
                continue
            v = it["verify_verdict"] = {"verdict": legacy, "source": "verification report (legacy catalog)"}
        if v.get("verdict") in {"pending", "stale"} or v.get("sql_sha"):
            continue
        v["sql_sha"] = sql_sha(it["sql"])
        n += 1
    return n


def semantic_status(meta: dict, item: dict) -> str:
    """not_required / pending / unreviewed / stale / <verdict> (ok, fix_text, ...).

    ``unreviewed``: the table requires review and the item has no verdict
    (added after the last verification). ``stale``: the verdict was given on
    SQL that has changed since. A catalog verified before verdicts carried
    their SQL falls back to the table's verification report."""
    if not semantic_review_required(meta):
        return "not_required"
    v = item.get("verify_verdict")
    if isinstance(v, dict) and v.get("verdict"):
        if v.get("sql_sha") and item.get("sql") and v["sql_sha"] != sql_sha(item["sql"]):
            return "stale"
        return str(v["verdict"])
    rep = meta.get("verified") or {}
    for key, verdict in _REPORT_VERDICTS:
        for e in rep.get(key) or []:
            if (e.get("id") if isinstance(e, dict) else e) == item.get("id"):
                return verdict
    return "unreviewed"


def selectable_items(meta: dict, versions: dict[str, str | None] | None = None
                     ) -> tuple[list[tuple[str, dict]], list[dict]]:
    """(selectable (section, item) pairs, [unavailable {query_id, reason}])."""
    ok, bad = [], []
    for sec, it in items(meta):
        good, why = selectable(meta, it, versions)
        if good:
            ok.append((sec, it))
        else:
            bad.append({"table": meta["table_name"], "query_id": it.get("id"), "reason": why})
    return ok, bad


def validate_item(meta: dict, section: str, item: dict,
                  columns_by_table: dict[str, set[str]] | dict[str, dict[str, str]]) -> str | None:
    """Return an error message, or None when the item compiles and passes the
    guard. ``columns_by_table`` may map to column sets or {column: type}."""
    if isinstance(item.get("query"), dict):
        types = {t: (c if isinstance(c, dict) else {n: "text" for n in c})
                 for t, c in columns_by_table.items()}
        types[meta["table_name"]] = column_types(meta) or types.get(meta["table_name"], {})
        try:
            built = compile_item(meta, section, item, types)
        except (QuerySpecError, SqlGuardError) as exc:
            return str(exc)
        declared = [p.get("name") for p in item.get("params") or []]
        missing = [p["name"] for p in built.params if p["name"] not in declared]
        if missing:
            return f"parameter(s) used in the query but not in params: {', '.join(missing)}"
        unused = [p for p in declared if p not in {b["name"] for b in built.params}]
        if unused:
            return f"declared param(s) not used in the query: {', '.join(unused)}"
        return None
    sql = item.get("sql") or ""
    try:
        res = check_sql(sql, item_tables(meta, section, item),
                        columns={t: set(c) for t, c in columns_by_table.items()})
    except SqlGuardError as exc:
        return str(exc)
    declared = [p.get("name") for p in item.get("params") or []]
    missing = [p for p in res.params if p not in declared]
    if missing:
        return f"placeholder(s) not declared in params: {', '.join(missing)}"
    unused = [p for p in declared if p not in res.params]
    if unused:
        return f"declared param(s) not used in the SQL: {', '.join(unused)}"
    return None


def validate_all(meta: dict, columns_by_table: dict[str, set[str]]) -> list[tuple[str, dict, str]]:
    return [(s, it, err) for s, it in items(meta)
            if (err := validate_item(meta, s, it, columns_by_table))]


def rename_table(meta: dict, old: str, new: str) -> dict:
    """Return a copy whose table name and every SQL reference is ``new``."""
    if old == new:
        return meta
    out = json.loads(json.dumps(meta))
    if out.get("table_name") == old:
        out["table_name"] = new
    for _sec, it in items(out):
        if it.get("sql"):
            it["sql"] = rename_tables(it["sql"], {old: new})
        if it.get("other_table") == old:
            it["other_table"] = new
        spec = it.get("query")
        if isinstance(spec, dict):
            _rename_spec(spec, old, new)
    return out


def _rename_spec(node, old: str, new: str) -> None:
    """Rewrite ``table`` keys and ``table.column`` strings inside a spec."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "table" and v == old:
                node[k] = new
            elif isinstance(v, str) and v.startswith(old + "."):
                node[k] = new + v[len(old):]
            else:
                _rename_spec(v, old, new)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str) and v.startswith(old + "."):
                node[i] = new + v[len(old):]
            else:
                _rename_spec(v, old, new)


def metadata_text(meta: dict) -> str:
    """What the metadata node embeds and stores as ``text`` (no table data)."""
    lines = [f"Table {meta['table_name']}: {meta.get('description') or ''}".strip()]
    if meta.get("grain"):
        lines.append(f"One row = {meta['grain']}.")
    cols = []
    for c in meta.get("columns") or []:
        desc = (c.get("description") or "").strip()
        cols.append(f"{c['name']} ({c.get('role') or c.get('type')}{': ' + desc if desc else ''})")
    if cols:
        lines.append("Columns: " + "; ".join(cols))
    for sec in SECTIONS:
        labels = [it.get("label") or it.get("id") for it in meta.get(sec) or []]
        if labels:
            lines.append(f"{SECTION_TITLES[sec]}: " + "; ".join(str(x) for x in labels))
    return "\n".join(lines)


def _col_name(raw: Any) -> str | None:
    if isinstance(raw, dict):
        return raw.get("column") if not raw.get("fn") else f"{raw['fn']}({raw.get('column')})"
    return str(raw) if raw not in (None, "") else None


def _base_col(raw: Any) -> str | None:
    c = raw.get("column") if isinstance(raw, dict) else raw
    return str(c).split(".", 1)[-1] if c not in (None, "", "*") else None


def describe_query(meta: dict, section: str, item: dict) -> dict:
    """Machine-readable description of what an item's FINAL query does,
    derived from its spec, parameters and the table metadata (no LLM):
    quantity, operation, output grain, grouping columns, fixed conditions,
    bindable parameters (with what leaving an optional one out does), units,
    period behaviour, ordering and limit. Selection, verification and the
    ontology roles read this instead of guessing from labels."""
    spec = item_spec(section, item) or {}
    cols = {c["name"]: c for c in meta.get("columns") or []}
    times = set(time_columns(meta))
    totals = {}
    for _t, m in item_totals(meta, section, item).items():
        for c, v in (m or {}).items():
            totals[c] = v
    select = spec.get("select") or []
    aggs = [s for s in select if isinstance(s, dict) and s.get("agg")]
    plain = [s for s in select if not (isinstance(s, dict) and s.get("agg"))]
    m = item.get("measure") if isinstance(item.get("measure"), dict) else None
    out: dict[str, Any] = {"kind": "measure" if m else section[:-1] if section != "queries" else "query"}
    if m:
        from backend.app.services.structured.refine import observation_key

        out["quantity"] = [m.get("value_column")]
        out["operation"] = "one value per observation (no aggregation)"
        out["output_grain"] = observation_key(m)
        out["grouping_columns"] = []
        order = m.get("version_order") or (
            [{"column": m["version_column"], "direction": "desc", "nulls": "last", "basis": "precedence"}]
            if m.get("version_column") else [])
        if order:
            out["version_rule"] = {"order": order, "reason": m.get("version_reason") or None,
                                   **({"tie_policy": m["tie_policy"]} if m.get("tie_policy") else {})}
        if m.get("subject_columns"):
            out["subject_columns"] = m["subject_columns"]
        if m.get("dimension_columns"):
            out["dimension_columns"] = m["dimension_columns"]
        pop = m.get("population") or {}
        if pop.get("include") or pop.get("exclude"):
            out["population"] = pop
        if m.get("unit"):
            out["unit"] = m["unit"]
        if (m.get("item") or {}).get("values"):
            out["item_values"] = {"column": m["item"]["column"], "values": m["item"]["values"]}
        for k in ("aggregation_across_entities", "aggregation_across_time", "weight_column",
                  "denominator_column"):
            if m.get(k):
                out[k] = m[k]
    else:
        out["quantity"] = [_base_col(a.get("column")) or "rows" for a in aggs] or None
        out["operation"] = ([f"{a['agg']}({_col_name(a.get('column')) or '*'})" for a in aggs]
                            or ("distinct rows" if spec.get("distinct") else "rows"))
        grp = spec.get("group_by") or ([_col_name(s) for s in plain] if aggs else [])
        out["grouping_columns"] = [g if isinstance(g, str) else _col_name(g) for g in grp]
        out["output_grain"] = (out["grouping_columns"] or ["one row (whole population)"]) if aggs else \
            ([_col_name(c) for c in spec.get("distinct_on") or []] or ["one row per matching source row"])
    out["returns"] = [(s.get("as") or f"{s['agg']}_{_base_col(s.get('column')) or 'rows'}")
                      if isinstance(s, dict) and s.get("agg") else _col_name(s) for s in select]
    fixed, params_by_name = [], {}
    for w in spec.get("where") or []:
        if not isinstance(w, dict):
            continue
        col = _col_name(w.get("column"))
        if "value" in w:
            fixed.append({"column": col, "op": w.get("op"), "value": w["value"]})
        elif isinstance(w.get("param"), list):
            for n in w["param"]:
                params_by_name[n] = {"column": col, "op": w.get("op"), "optional": False}
        elif w.get("param"):
            params_by_name[w["param"]] = {"column": col, "op": w.get("op"), "optional": bool(w.get("optional"))}
    fixed += [{"condition": c, "added_by_program": True} for c in item.get("added_conditions") or []]
    out["fixed_conditions"] = fixed
    plist = []
    for p in item.get("params") or []:
        info = params_by_name.get(p.get("name"), {})
        col = p.get("column") or info.get("column")
        e = {"name": p.get("name"), "type": p.get("type"), "column": col, "op": info.get("op"),
             "optional": bool(p.get("optional"))}
        if e["optional"]:
            base = _base_col(col)
            if base in totals:
                e["when_omitted"] = (f"reads the published total row ({base} = {totals[base]!r}) where its "
                                     "group has one, the other rows where it has none")
            else:
                e["when_omitted"] = f"no filter on {base}: every value"
        plist.append(e)
    out["parameters"] = plist
    out["filter_columns"] = sorted({_base_col(e["column"]) for e in plist if e["column"]}
                                   | {_base_col(f["column"]) for f in fixed if f.get("column")})
    q_cols = [c for c in (out.get("quantity") or []) if c in cols]
    units = {c: cols[c].get("unit") for c in q_cols if cols[c].get("unit")}
    if units:
        out["units"] = units
    unit_cols = [c for c in (m or {}).get("qualifier_columns") or [] if c in cols]
    if unit_cols:
        out["qualifier_columns"] = unit_cols
    period_params = [e for e in plist if _base_col(e["column"]) in times]
    order = spec.get("order_by") or []
    lim = spec.get("limit")
    if period_params:
        out["period_behavior"] = "explicit period range/value: " + ", ".join(
            f"{e['name']} on {e['column']}" for e in period_params)
    elif any(_base_col(o.get("column")) in times and str(o.get("dir")).lower() == "desc"
             for o in order if isinstance(o, dict)) and (lim or spec.get("distinct_on")):
        out["period_behavior"] = "latest period available in the data"
    elif any(_base_col(g) in times for g in out.get("grouping_columns") or [] if g) or \
            any(c in times for c in out.get("output_grain") or []):
        out["period_behavior"] = "one result per period (every period)"
    else:
        out["period_behavior"] = "all periods together"
    if order:
        out["ordering"] = [{"column": _col_name(o.get("column")), "dir": o.get("dir", "asc")}
                           for o in order if isinstance(o, dict)]
    if lim is not None:
        out["limit"] = lim
    if section == "lookups":
        out["joins"] = {"other_table": item.get("other_table"), "on": item.get("join"),
                        "cardinality": item.get("cardinality")}
    return out


def _annotate_published_totals(cat: dict, tot: list[dict]) -> None:
    """Point the selection at the published total rows: each total names the
    offered queries that read it (an equality param on its column), and a query
    that sums a measure while leaving the total row out says it computes a sum
    of the listed parts -- which can differ from the published total (rounding,
    members the table does not list)."""
    def _cap(q: dict) -> dict:
        return q.get("capability") if isinstance(q.get("capability"), dict) else {}

    for t in tot:
        col, val = t["column"], str(t["value"])
        t["read_with"] = [q["id"] for q in cat["queries"]
                          if any(p.get("column") == col and p.get("op") in ("=", None) and p.get("name")
                                 for p in _cap(q).get("parameters") or [])]
        for q in cat["queries"]:
            c = _cap(q)
            ops = c.get("operation") if isinstance(c.get("operation"), list) else []
            sums = any(str(o).lower().startswith("sum(") for o in ops)
            # broken down BY the column (one row per part): each row is a part, not a sum of parts
            per_part = col in [str(g).split(".")[-1] for g in c.get("output_grain") or []]
            excl = any(f.get("column") == col and str(f.get("op")) in ("!=", "<>", "not_in")
                       and (val == str(f.get("value")) or val in [str(v) for v in f.get("value") or []
                                                                  if isinstance(f.get("value"), list)])
                       for f in c.get("fixed_conditions") or [] if isinstance(f, dict))
            if sums and excl and not per_part:
                q["excludes_published_total"] = (
                    f"adds up the listed parts and leaves out the published total rows ({col} = {val!r}); "
                    "the sum can differ from the published total. For the total itself read the total row"
                    + (f" with {', '.join(t['read_with'])} ({col} = {val!r})" if t["read_with"] else ""))


def catalog_for_prompt(meta: dict, max_items: int = 40,
                       ranking: dict[str, tuple[float, list[str]]] | None = None,
                       *, stable: bool = False, only_selectable: bool = False,
                       versions: dict[str, str | None] | None = None) -> dict:
    """Compact view the retrieval model chooses queries from.

    Columns carry their VALID VALUES (every distinct value when there are few,
    else the min/max) from ``value_domains``; value combinations that occur
    together are listed once per table and referenced by index from the
    queries that filter on them. Each query shows its spec as a read-only
    ``definition`` (what it does); the model returns only its id and params.
    ``only_selectable`` leaves out items retrieval may not run (see
    :func:`selectable`)."""
    domains = meta.get("value_domains") or {}

    def _col(c: dict) -> dict:
        d = domains.get(c["name"]) or {}
        out = {"name": c["name"], "type": c.get("type"), "role": c.get("role"),
               "description": c.get("description")}
        if "values" in d:
            out["values"] = d["values"]
        elif "min" in d:
            out["range"] = [d["min"], d["max"]]
        elif c.get("examples"):
            out["examples"] = c["examples"][:5]
        periods = ((meta.get("column_stats") or {}).get(c["name"]) or {}).get("periods")
        if periods:
            out["periods"] = periods            # earliest..latest per granularity
        return out

    cat = {"table": meta["table_name"], "description": meta.get("description"),
           "grain": meta.get("grain"), "row_count": meta.get("row_count"),
           "columns": [_col(c) for c in meta.get("columns") or []],
           "queries": []}
    tot = [{"column": t["column"], "value": t["value"], "per": t.get("groups_by")}
           for t in (meta.get("analysis") or {}).get("total_rows") or [] if isinstance(t, dict)]
    if tot:
        cat["total_rows"] = tot      # published totals: read them, do not re-add the parts
    combos: list[dict] = []
    allowed = ({id(it) for _s, it in selectable_items(meta, versions)[0]} if only_selectable else None)

    def _all() -> list[tuple[str, dict]]:
        return [si for si in items(meta) if allowed is None or id(si[1]) in allowed]

    ordered = _all()
    if ranking:
        # most relevant first (graph tags matched by the question), so the
        # cap keeps them; ties keep the stored order (measures first)
        ordered.sort(key=lambda si: -(ranking.get(si[1]["id"]) or (0.0, []))[0])
    if stable:
        # Same text for every question (prompt-cache friendly): the ranking only
        # decides which items survive the cap; order and content are the stored
        # ones, and the per-question matches go in a separate block.
        keep = {it["id"] for _s, it in ordered[:max_items]}
        ordered = [si for si in _all() if si[1]["id"] in keep]
        ranking = None
    for sec, it in ordered:
        if len(cat["queries"]) >= max_items:
            break
        combo_ref = {}
        ac = it.get("allowed_combinations")
        if ac and ac.get("rows"):
            cols = [next((p.get("column") for p in it.get("params") or [] if p.get("name") == n), n)
                    for n in ac["params"]]
            key = {"columns": cols, "rows": ac["rows"]}
            if key not in combos:
                combos.append(key)
            combo_ref = {"valid_combinations": {"params": ac["params"], "see": combos.index(key)}}
        why = (ranking or {}).get(it["id"])
        if why and why[0] > 0:
            combo_ref = {**combo_ref, "matched": why[1]}
        cat["queries"].append({**combo_ref,
            "id": it["id"], "kind": sec[:-1] if sec != "queries" else "query",
            "label": it.get("label"), "description": it.get("description"),
            "params": [{k: p.get(k) for k in ("name", "type", "column", "example", "optional")
                        if k != "optional" or p.get(k)}
                       for p in it.get("params") or []],
            "capability": it.get("capability") or describe_query(meta, sec, it),
            **({"other_table": it["other_table"]} if it.get("other_table") else {}),
        })
    if tot:
        _annotate_published_totals(cat, tot)
    if combos:
        cat["value_combinations"] = combos
    if meta.get("usage_notes"):
        cat["usage_notes"] = meta["usage_notes"]
    if meta.get("analysis"):
        from backend.app.services.structured.analysis import summary_for_prompt

        st = summary_for_prompt(meta["analysis"])
        st.pop("item_columns", None)     # the measures already name their items
        cat["structure"] = st
    if meta.get("common_questions"):
        # examples only point at queries this catalog offers (the same eligible ids)
        offered = {q["id"] for q in cat["queries"]}
        cat["examples"] = [{k: q.get(k) for k in ("question", "query_id", "params")}
                           for q in meta["common_questions"] if q.get("query_id") in offered][:8]
    if meta.get("coverage_gaps"):
        cat["coverage_gaps"] = meta["coverage_gaps"][:20]
    return cat


def spec_columns(spec: Any, table: str) -> set[str]:
    """Column names a query spec references (base-table names unqualified,
    other tables' as ``table.column``)."""
    out: set[str] = set()

    def col(c):
        if isinstance(c, dict):
            c = c.get("column")
        if isinstance(c, str) and c:
            out.add(c[len(table) + 1:] if c.startswith(table + ".") else c)

    if not isinstance(spec, dict):
        return out
    for key in ("select", "group_by", "distinct_on"):
        for c in spec.get(key) or []:
            if isinstance(c, dict) and "columns" in c:
                for x in c["columns"] or []:
                    col(x)
            col(c)
    for w in spec.get("where") or []:
        col(w.get("column") if isinstance(w, dict) else None)
    for o in spec.get("order_by") or []:
        col(o)
    return out


def item_tags(meta: dict, section: str, item: dict) -> dict:
    """What a stored query is ABOUT, so retrieval can pick it from the graph:
    the ontology classes of the columns it uses (and its own class), the
    entity / place columns its parameters take, the period columns it reads."""
    from backend.app.services.structured.graph_links import period_columns, place_level

    table = meta["table_name"]
    cols = {c["name"]: c for c in meta.get("columns") or []}
    used = spec_columns(item_spec(section, item), table)
    classes: list[dict] = []

    def add(iri, label):
        if (iri or label) and not any(c["iri"] == iri and c["label"] == label for c in classes):
            classes.append({"iri": iri, "label": label})

    add(item.get("class_iri"), item.get("class_label"))
    for name in sorted(used):
        c = cols.get(name)
        if c is not None:
            add(c.get("class_iri"), c.get("class_label"))
    param_cols = {p.get("column") for p in item.get("params") or [] if p.get("column") in cols}
    ent = set(entity_key_columns(meta))
    places = {n for n, c in cols.items() if place_level(c) is not None}
    periods = set(period_columns(meta)) or set(time_columns(meta))
    return {"classes": classes,
            "entity_columns": sorted((param_cols | used) & ent - places),
            "place_columns": sorted((param_cols | used) & places),
            "time_columns": sorted(used & periods)}


def tag_items(meta: dict) -> dict:
    """Set ``tags`` on every stored item (in place); returns meta."""
    for sec, it in items(meta):
        it["tags"] = item_tags(meta, sec, it)
    return meta


def class_refs(meta: dict) -> list[dict]:
    """Every (kind, name, class_iri, class_label) the node should link to."""
    out = []
    for c in meta.get("columns") or []:
        if c.get("class_iri") or c.get("class_label"):
            out.append({"kind": "tableColumn", "name": c["name"],
                        "class_iri": c.get("class_iri"), "class_label": c.get("class_label")})
    for sec, kind in (("measures", "tableMetric"), ("metrics", "tableMetric"), ("filters", "tableFilter")):
        for it in meta.get(sec) or []:
            if it.get("class_iri") or it.get("class_label"):
                out.append({"kind": kind, "name": it["id"],
                            "class_iri": it.get("class_iri"), "class_label": it.get("class_label")})
    # concepts of the final verified catalog (structured.concepts), one per key
    from backend.app.services.structured.concepts import concept_class_refs

    have = {(r["kind"], r.get("class_iri")) for r in out}
    out += [r for r in concept_class_refs(meta) if (r["kind"], r["class_iri"]) not in have]
    return out


def entity_key_columns(meta: dict) -> list[str]:
    """Text columns whose values name things (companies, drugs, places...)."""
    out = []
    for c in meta.get("columns") or []:
        if c.get("type", "text") != "text":
            continue
        if c.get("entity_kind") or c.get("role") in {"identifier", "dimension"}:
            out.append(c["name"])
    return out

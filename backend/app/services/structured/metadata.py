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

An item without ``query`` (metadata written before format 2) runs its stored
``sql`` as before. Metric/filter/query items read only their own table; a
lookup reads its own table and ``other_table``. Each lookup is stored in BOTH
tables' files.
"""

from __future__ import annotations

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

SECTIONS = ("measures", "metrics", "filters", "queries", "lookups")
SECTION_TITLES = {"measures": "Measures", "metrics": "Metrics", "filters": "Filters", "queries": "Queries", "lookups": "Lookups"}
COLUMN_ROLES = ("identifier", "dimension", "measure", "time", "attribute", "text")


def items(meta: dict, sections: tuple[str, ...] = SECTIONS) -> Iterator[tuple[str, dict]]:
    for sec in sections:
        for it in meta.get(sec) or []:
            yield sec, it


def find_item(meta: dict, item_id: str) -> tuple[str, dict] | None:
    return next(((s, it) for s, it in items(meta) if it.get("id") == item_id), None)


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
    (from its analysis)."""
    from backend.app.services.structured.analysis import total_markers

    spec = item_spec(section, item)
    tables = item_tables(meta, section, item)
    if spec is None:  # legacy item: stored SQL
        res = check_sql(item.get("sql") or "", tables)
        return Built(sql=res.sql, params=[{"name": p, "type": "text", "column": None}
                                          for p in res.params], tables=res.tables)
    types = {t: types_by_table[t] for t in tables if t in types_by_table}
    types.setdefault(meta["table_name"], column_types(meta))
    if totals is None:
        totals = {meta["table_name"]: total_markers(meta)}
    built = build_sql(spec, meta["table_name"], types,
                      time_columns={meta["table_name"]: time_columns(meta)}, totals=totals)
    check_sql(built.sql, tables)      # defence in depth: the guard still applies
    return built


def finalize_item(meta: dict, section: str, item: dict,
                  types_by_table: dict[str, dict[str, str]]) -> str | None:
    """Compile a spec item in place: ``sql`` <- compiled statement, ``params``
    <- the parameters it binds (names/types/columns from the compiler,
    descriptions/examples kept). Returns an error message instead on failure."""
    from backend.app.services.structured.query_builder import merge_params

    if not isinstance(item.get("query"), dict):
        return "item has no query spec"
    try:
        built = compile_item(meta, section, item, types_by_table)
    except (QuerySpecError, SqlGuardError) as exc:
        return str(exc)
    item["params"] = merge_params(built, item.get("params"))
    item["sql"] = built.sql
    return None


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


def catalog_for_prompt(meta: dict, max_items: int = 40,
                       ranking: dict[str, tuple[float, list[str]]] | None = None,
                       *, stable: bool = False) -> dict:
    """Compact view the retrieval model chooses queries from.

    Columns carry their VALID VALUES (every distinct value when there are few,
    else the min/max) from ``value_domains``; value combinations that occur
    together are listed once per table and referenced by index from the
    queries that filter on them."""
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
    combos: list[dict] = []
    ordered = list(items(meta))
    if ranking:
        # most relevant first (graph tags matched by the question), so the
        # cap keeps them; ties keep the stored order (measures first)
        ordered.sort(key=lambda si: -(ranking.get(si[1]["id"]) or (0.0, []))[0])
    if stable:
        # Same text for every question (prompt-cache friendly): the ranking only
        # decides which items survive the cap; order and content are the stored
        # ones, and the per-question matches go in a separate block.
        keep = {it["id"] for _s, it in ordered[:max_items]}
        ordered = [si for si in items(meta) if si[1]["id"] in keep]
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
            **({"query": it["query"]} if isinstance(it.get("query"), dict) else {"sql": it.get("sql")}),
            **({"other_table": it["other_table"]} if it.get("other_table") else {}),
        })
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
        cat["examples"] = [{k: q.get(k) for k in ("question", "query_id", "params")}
                           for q in meta["common_questions"][:8]]
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

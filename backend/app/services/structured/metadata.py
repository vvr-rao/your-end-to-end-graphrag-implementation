"""The per-table METADATA FILE (a JSON dict) and helpers around it.

Shape (``format`` 1)::

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

Every ITEM is one well-formed Postgres statement with named ``:params``::

    {"id", "label", "description", "sql",
     "params": [{"name", "type", "column", "description", "example"}],
     "columns": [...], "class_iri", "class_label",
     # lookups only:
     "other_table", "join": [{"column", "other_column"}]}

Metric/filter/query items read only their own table; a lookup reads its own
table and ``other_table``. Each lookup is stored in BOTH tables' files.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from backend.app.services.structured.sql_guard import SqlGuardError, check_sql, rename_tables

SECTIONS = ("metrics", "filters", "queries", "lookups")
SECTION_TITLES = {"metrics": "Metrics", "filters": "Filters", "queries": "Queries", "lookups": "Lookups"}
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


def validate_item(meta: dict, section: str, item: dict,
                  columns_by_table: dict[str, set[str]]) -> str | None:
    """Return an error message, or None when the item's SQL is acceptable."""
    sql = item.get("sql") or ""
    try:
        res = check_sql(sql, item_tables(meta, section, item), columns=columns_by_table)
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
    return out


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


def catalog_for_prompt(meta: dict, max_items: int = 40) -> dict:
    """Compact view the retrieval model chooses queries from."""
    cat = {"table": meta["table_name"], "description": meta.get("description"),
           "grain": meta.get("grain"), "row_count": meta.get("row_count"),
           "columns": [{"name": c["name"], "type": c.get("type"), "role": c.get("role"),
                        "description": c.get("description"),
                        **({"examples": c["examples"][:5]} if c.get("examples") else {})}
                       for c in meta.get("columns") or []],
           "queries": []}
    for sec, it in items(meta):
        if len(cat["queries"]) >= max_items:
            break
        cat["queries"].append({
            "id": it["id"], "kind": sec[:-1] if sec != "queries" else "query",
            "label": it.get("label"), "description": it.get("description"),
            "params": [{k: p.get(k) for k in ("name", "type", "column", "example")}
                       for p in it.get("params") or []],
            "sql": it.get("sql"),
            **({"other_table": it["other_table"]} if it.get("other_table") else {}),
        })
    return cat


def class_refs(meta: dict) -> list[dict]:
    """Every (kind, name, class_iri, class_label) the node should link to."""
    out = []
    for c in meta.get("columns") or []:
        if c.get("class_iri") or c.get("class_label"):
            out.append({"kind": "tableColumn", "name": c["name"],
                        "class_iri": c.get("class_iri"), "class_label": c.get("class_label")})
    for sec, kind in (("metrics", "tableMetric"), ("filters", "tableFilter")):
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

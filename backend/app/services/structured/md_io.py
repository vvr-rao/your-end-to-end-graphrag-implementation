"""Metadata file <-> Markdown, for human review (``table-metadata-export`` /
``table-metadata-import``).

The Markdown is meant to be edited: table description and grain, the column
descriptions/roles/units/entity kinds/classes, and every metric, filter, query
and lookup (label, description, columns, class, parameter descriptions and
examples, and the JSON query spec). Each item shows its query spec in a
```json block (the one to edit) and the SQL compiled from it in a ```sql
block (read-only: it is rebuilt from the spec on import, edits to it are
ignored). Parameter names, types and columns are rebuilt from the spec too. Items can be
added or deleted by adding or deleting their ``### `id` — label`` block. Column
names and types are fixed by the loaded table and must not change.

The parser is strict: anything it cannot read is reported with its line number
and nothing is applied.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

from backend.app.services.structured.metadata import SECTION_TITLES, SECTIONS
from backend.app.services.structured.query_builder import canonical

MARKER_RE = re.compile(r"^<!-- yegi-table-metadata format=(\d+) cache_key=([0-9a-f]+) -->$")
TITLE_RE = re.compile(r"^# Table `([^`]+)`\s*$")
ITEM_RE = re.compile(r"^### `([A-Za-z0-9_\-.]+)`(?:\s+—\s+(.*))?\s*$")
FIELD_RE = re.compile(r"^- \*\*([^*]+):\*\*\s?(.*)$")
CLASS_RE = re.compile(r"^(.*?)\s*<([^<>\s]+)>\s*$")
_TITLE_TO_SECTION = {v: k for k, v in SECTION_TITLES.items()}
COLUMN_HEADER = ["Column", "Type", "Role", "Unit", "Entity kind", "Description", "Class"]
PARAM_HEADER = ["Name", "Type", "Column", "Description", "Example"]


class MetadataParseError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


# ------------------------------------------------------------------ export


def _cell(v) -> str:
    s = "" if v is None else str(v)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _code(v) -> str:
    return f"`{v}`" if v not in (None, "") else ""


def _class(label, iri) -> str:
    if iri:
        return f"{label or ''} <{iri}>".strip()
    return label or ""


def _one_line(v) -> str:
    return " ".join(str(v or "").split())


def to_markdown(meta: dict) -> str:
    out: list[str] = []
    w = out.append
    w(f"# Table `{meta['table_name']}`")
    w("")
    w(f"<!-- yegi-table-metadata format={meta.get('format', 1)} cache_key={meta.get('cache_key', '0')} -->")
    w("")
    src = meta.get("source") or {}
    w(f"- **Source file:** {_code(src.get('rel_path'))}")
    w(f"- **Rows:** {meta.get('row_count', '')}")
    w(f"- **Grain:** {_one_line(meta.get('grain'))}")
    w(f"- **Model:** {meta.get('model', '')}{' (edited)' if meta.get('edited') else ''}")
    w("")
    w("## Description")
    w("")
    w((meta.get("description") or "").strip())
    w("")
    w("## Columns")
    w("")
    w("| " + " | ".join(COLUMN_HEADER) + " |")
    w("|" + "---|" * len(COLUMN_HEADER))
    for c in meta.get("columns") or []:
        w("| " + " | ".join([
            _code(c["name"]), _cell(c.get("type")), _cell(c.get("role")), _cell(c.get("unit")),
            _cell(c.get("entity_kind")), _cell(c.get("description")),
            _cell(_class(c.get("class_label"), c.get("class_iri"))),
        ]) + " |")
    for sec in SECTIONS:
        w("")
        w(f"## {SECTION_TITLES[sec]}")
        for it in meta.get(sec) or []:
            w("")
            label = _one_line(it.get("label"))
            w(f"### `{it['id']}`" + (f" — {label}" if label else ""))
            w("")
            w(f"- **Description:** {_one_line(it.get('description'))}")
            if sec == "lookups":
                w(f"- **Other table:** {_code(it.get('other_table'))}")
                joins = "; ".join(f"`{j.get('column')}` = `{it.get('other_table')}.{j.get('other_column')}`"
                                  for j in it.get("join") or [])
                w(f"- **Join:** {joins}")
            w(f"- **Columns:** {', '.join(_code(c) for c in it.get('columns') or [])}")
            w(f"- **Class:** {_class(it.get('class_label'), it.get('class_iri'))}")
            params = it.get("params") or []
            if not params:
                w("- **Parameters:** none")
            else:
                w("- **Parameters:**")
                w("")
                w("| " + " | ".join(PARAM_HEADER) + " |")
                w("|" + "---|" * len(PARAM_HEADER))
                for p in params:
                    w("| " + " | ".join([
                        _code(p.get("name")), _cell(p.get("type")), _code(p.get("column")),
                        _cell(p.get("description")), _cell(p.get("example")),
                    ]) + " |")
            if isinstance(it.get("query"), dict):
                w("")
                w("```json")
                w(json.dumps(canonical(it["query"]), ensure_ascii=False, indent=2))
                w("```")
                w("")
                w("```sql")
                w("-- compiled from the query spec above (read-only)")
                w((it.get("sql") or "").strip())
                w("```")
            else:
                w("")
                w("```sql")
                w((it.get("sql") or "").strip())
                w("```")
    w("")
    return "\n".join(out)


# ------------------------------------------------------------------ import


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if not (body.startswith("|") and body.endswith("|")) or body == "|":
        raise ValueError("table row must start and end with |")
    body = body[1:-1]
    cells, cur, i = [], [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] in "|\\":
            cur.append(body[i + 1])
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return [c.strip().replace("<br>", "\n") for c in cells]


def _uncode(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s.startswith("`") and s.endswith("`"):
        return s[1:-1]
    return s


def _parse_class(s: str) -> tuple[str | None, str | None]:
    s = s.strip()
    if not s:
        return None, None
    m = CLASS_RE.match(s)
    if m:
        return (m.group(1).strip() or None), m.group(2)
    return s, None


def _none(s: str) -> str | None:
    s = s.strip()
    return s or None


@dataclass
class _Doc:
    table_name: str = ""
    cache_key: str = ""
    grain: str | None = None
    description: str = ""
    columns: list[dict] = field(default_factory=list)
    sections: dict[str, list[dict]] = field(default_factory=dict)


def parse_markdown(text: str) -> _Doc:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    errors: list[str] = []
    doc = _Doc()
    i, n = 0, len(lines)

    def err(lineno: int, msg: str) -> None:
        errors.append(f"line {lineno}: {msg}")

    def skip_blank() -> None:
        nonlocal i
        while i < n and not lines[i].strip():
            i += 1

    skip_blank()
    m = TITLE_RE.match(lines[i]) if i < n else None
    if not m:
        raise MetadataParseError([f"line {i + 1}: expected '# Table `name`'"])
    doc.table_name = m.group(1)
    i += 1
    section: str | None = None
    item: dict | None = None
    seen_sections: set[str] = set()

    def read_table(header: list[str]) -> list[list[str]]:
        nonlocal i
        skip_blank()
        rows: list[list[str]] = []
        if i >= n or not lines[i].strip().startswith("|"):
            err(i + 1, f"expected a table with header {' | '.join(header)}")
            return rows
        try:
            got = _split_row(lines[i])
        except ValueError as exc:
            err(i + 1, str(exc))
            got = []
        if got != header:
            err(i + 1, f"table header must be: | {' | '.join(header)} |")
        i += 1
        if i < n and re.match(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$", lines[i].strip()):
            i += 1
        else:
            err(i + 1, "missing |---| separator row")
        while i < n and lines[i].strip().startswith("|"):
            try:
                row = _split_row(lines[i])
            except ValueError as exc:
                err(i + 1, str(exc))
                i += 1
                continue
            if len(row) != len(header):
                err(i + 1, f"expected {len(header)} cells, got {len(row)}")
            else:
                rows.append(row)
            i += 1
        return rows

    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if (mm := MARKER_RE.match(s)):
            doc.cache_key = mm.group(2)
            i += 1
            continue
        if s.startswith("## "):
            title = s[3:].strip()
            item = None
            if title == "Description":
                section = "description"
                i += 1
                buf = []
                while i < n and not lines[i].startswith("## "):
                    buf.append(lines[i])
                    i += 1
                doc.description = "\n".join(buf).strip()
                continue
            if title == "Columns":
                section = "columns"
                i += 1
                for row in read_table(COLUMN_HEADER):
                    label, iri = _parse_class(row[6])
                    doc.columns.append({"name": _uncode(row[0]), "type": row[1] or None,
                                        "role": _none(row[2]), "unit": _none(row[3]),
                                        "entity_kind": _none(row[4]), "description": row[5],
                                        "class_label": label, "class_iri": iri})
                continue
            if title in _TITLE_TO_SECTION:
                section = _TITLE_TO_SECTION[title]
                if section in seen_sections:
                    err(i + 1, f"section '## {title}' appears twice")
                seen_sections.add(section)
                doc.sections.setdefault(section, [])
                i += 1
                continue
            err(i + 1, f"unknown section '## {title}'")
            i += 1
            continue
        if s.startswith("### "):
            mm = ITEM_RE.match(s)
            if not mm or section not in SECTIONS:
                err(i + 1, "item heading must look like ### `id` — label, inside "
                           "Metrics/Filters/Queries/Lookups")
                i += 1
                continue
            item = {"id": mm.group(1), "label": (mm.group(2) or "").strip() or None,
                    "description": "", "columns": [], "params": [], "sql": "",
                    "class_iri": None, "class_label": None, "_line": i + 1}
            if section == "lookups":
                item.update({"other_table": None, "join": []})
            doc.sections[section].append(item)
            i += 1
            continue
        if (fm := FIELD_RE.match(s)):
            key, val = fm.group(1).strip(), fm.group(2).strip()
            if item is None:
                if key == "Grain":
                    doc.grain = val or None
                elif key not in {"Source file", "Rows", "Model"}:
                    err(i + 1, f"unknown field '{key}'")
                i += 1
                continue
            i += 1
            if key == "Description":
                item["description"] = val
            elif key == "Columns":
                item["columns"] = [_uncode(c) for c in val.split(",") if c.strip()]
            elif key == "Class":
                item["class_label"], item["class_iri"] = _parse_class(val)
            elif key == "Other table" and section == "lookups":
                item["other_table"] = _uncode(val) or None
            elif key == "Join" and section == "lookups":
                joins = []
                for part in [p for p in val.split(";") if p.strip()]:
                    jm = re.match(r"^\s*`([^`]+)`\s*=\s*`([^`.]+)\.([^`]+)`\s*$", part)
                    if not jm:
                        err(i, f"join must look like `col` = `other_table.col`: {part.strip()}")
                        continue
                    joins.append({"column": jm.group(1), "other_column": jm.group(3)})
                item["join"] = joins
            elif key == "Parameters":
                if val.lower() == "none":
                    item["params"] = []
                elif val:
                    err(i, "Parameters must be 'none' or followed by a table")
                else:
                    item["params"] = [{"name": _uncode(r[0]), "type": r[1] or "text",
                                       "column": _uncode(r[2]) or None, "description": r[3],
                                       "example": r[4] or None}
                                      for r in read_table(PARAM_HEADER)]
            else:
                err(i, f"unknown field '{key}'")
            continue
        if s.startswith("```"):
            fence_line = i + 1
            lang = s[3:].strip().lower()
            if item is None:
                err(fence_line, "code block outside an item")
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            if i >= n:
                err(fence_line, "unterminated ``` block")
            i += 1
            if item is not None:
                body = "\n".join(buf).strip()
                if lang == "json":
                    if "query" in item:
                        err(fence_line, f"item `{item['id']}` has more than one ```json block")
                    try:
                        spec = json.loads(body)
                    except json.JSONDecodeError as exc:
                        err(fence_line + 1 + exc.lineno - 1, f"query spec is not valid JSON: {exc.msg}")
                        spec = {}
                    if not isinstance(spec, dict):
                        err(fence_line, "query spec must be a JSON object")
                        spec = {}
                    item["query"] = spec
                elif lang == "sql":
                    if item["sql"]:
                        err(fence_line, f"item `{item['id']}` has more than one SQL block")
                    item["sql"] = body
                else:
                    err(fence_line, "code blocks must be ```json (query spec) or ```sql")
            continue
        err(i + 1, f"unexpected text: {s[:60]}")
        i += 1

    for sec in SECTIONS:
        for it in doc.sections.get(sec, []):
            if "query" in it:
                it["sql"] = ""          # compiled from the spec on import
            elif not it["sql"]:
                err(it["_line"], f"item `{it['id']}` has no ```json query spec")
    ids = [it["id"] for sec in SECTIONS for it in doc.sections.get(sec, [])]
    dups = sorted({x for x in ids if ids.count(x) > 1})
    if dups:
        errors.append(f"duplicate item id(s): {', '.join(dups)}")
    if errors:
        raise MetadataParseError(errors)
    for sec in SECTIONS:
        for it in doc.sections.get(sec, []):
            it.pop("_line", None)
    return doc


_COLUMN_EDITABLE = ("role", "unit", "entity_kind", "description", "class_label", "class_iri")


def apply_markdown(meta: dict, text: str) -> dict:
    """Return a copy of ``meta`` with the Markdown's edits applied.

    Raises :class:`MetadataParseError` for unreadable Markdown or for edits
    that would not match the loaded table (renamed/added/removed columns,
    changed types, a different table).
    """
    doc = parse_markdown(text)
    errors: list[str] = []
    if doc.table_name != meta["table_name"]:
        errors.append(f"file is for table `{doc.table_name}`, not `{meta['table_name']}`")
    if doc.cache_key and meta.get("cache_key") and doc.cache_key != meta["cache_key"]:
        errors.append("cache_key differs: the table's file changed since this Markdown was exported")
    cur_cols = [c["name"] for c in meta.get("columns") or []]
    new_cols = [c["name"] for c in doc.columns]
    if cur_cols != new_cols:
        errors.append(f"columns must stay {cur_cols} (got {new_cols}); edit descriptions, not names")
    else:
        for old, new in zip(meta["columns"], doc.columns, strict=True):
            if new.get("type") and new["type"] != old.get("type"):
                errors.append(f"column `{old['name']}`: type is fixed ({old.get('type')})")
    if errors:
        raise MetadataParseError(errors)
    out = copy.deepcopy(meta)
    out["description"] = doc.description
    out["grain"] = doc.grain
    for col, new in zip(out["columns"], doc.columns, strict=True):
        for k in _COLUMN_EDITABLE:
            col[k] = new.get(k) or None
        col["description"] = new.get("description") or ""
    for sec in SECTIONS:
        out[sec] = doc.sections.get(sec, [])
    return out

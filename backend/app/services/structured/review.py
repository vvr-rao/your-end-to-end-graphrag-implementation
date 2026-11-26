"""``table-metadata-export`` / ``table-metadata-import``: review and edit the
table metadata stored in the database as Markdown.

Export writes one ``<table>.md`` per loaded table (from the ACTIVE
TableMetadata node). Import reads edited files back and, per table:

1. applies the edits strictly (:func:`md_io.apply_markdown` -- column names
   and types are fixed by the loaded table);
2. compiles every item's JSON query spec to SQL (:mod:`query_builder`),
   then checks it with the guard and ``EXPLAIN`` against the loaded table
   (read-only transaction), so a query that would fail at question time is
   refused now;
3. shows a unified diff of the Markdown;
4. unless ``dry_run``: updates the node (text, metadata, embedding), rebuilds
   its class/entity edges from the edited metadata, and writes the edited
   entry to the user cache with ``edited: true`` so later profiling runs keep
   it.

Nothing is applied for a table whose file has any error.
"""

from __future__ import annotations

import difflib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text as sql_text

from backend.app.services.structured.cache import MetadataCache
from backend.app.services.structured.db_load import (
    rebuild_edges,
    registered_tables,
    table_columns,
    upsert_node,
)
from backend.app.services.structured.md_io import (
    MetadataParseError,
    apply_markdown,
    parse_markdown,
    to_markdown,
)
from backend.app.services.structured.metadata import (
    column_types,
    compile_item,
    finalize_item,
    items,
    metadata_text,
    tag_items,
    validate_all,
)
from backend.app.services.structured.query_builder import QuerySpecError, example_value
from backend.app.services.structured.sql_exec import _db_error, coerce_params
from backend.app.services.structured.sql_guard import SqlGuardError

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class ImportReport:
    path: Path
    table: str | None = None
    errors: list[str] = field(default_factory=list)
    diff: str = ""
    applied: bool = False
    class_edges: int = 0
    entity_edges: int = 0
    tested: dict = field(default_factory=dict)      # data-test status counts after the import

    @property
    def changed(self) -> bool:
        return bool(self.diff)


def md_filename(table: str) -> str:
    return f"{table}.md"


async def export_tables(out_dir: Path, tables: list[str] | None = None) -> tuple[list[Path], list[str]]:
    """Write ``<table>.md`` for each ACTIVE table (all when ``tables`` is None).
    Returns (written paths, requested names that are not loaded)."""
    from backend.app.db.session import session_scope

    async with session_scope() as s:
        reg = await registered_tables(s)
    names = sorted(reg) if not tables else list(dict.fromkeys(tables))
    missing = [n for n in names if n not in reg]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n in names:
        if n in reg:
            p = out_dir / md_filename(n)
            p.write_text(to_markdown(reg[n]["meta"]), encoding="utf-8", newline="\n")
            written.append(p)
    return written, missing


def _example_params(item: dict, built_params: list[dict]) -> dict:
    declared = {p.get("name"): p for p in item.get("params") or []}
    vals = {}
    for p in built_params:
        ex = (declared.get(p["name"]) or {}).get("example")
        vals[p["name"]] = ex if ex not in ("", None) else example_value(p.get("type") or "text")
    return coerce_params(vals, built_params)


async def explain_errors(meta: dict, allowed: set[str],
                         types_by_table: dict[str, dict[str, str]] | None = None) -> list[str]:
    """Compile each item and ``EXPLAIN`` it (example values bound) in a
    read-only transaction. Catches what the compiler cannot: type mismatches,
    a lookup partner that is not loaded."""
    from backend.app.db.engine import get_sessionmaker

    types = dict(types_by_table or {})
    types.setdefault(meta["table_name"], column_types(meta))
    errors = []
    for section, item in items(meta):
        try:
            built = compile_item(meta, section, item, types)
        except (QuerySpecError, SqlGuardError) as exc:
            errors.append(f"{section} `{item.get('id')}`: {exc}")
            continue
        missing = sorted(built.tables - allowed - {meta["table_name"]})
        if missing:
            errors.append(f"{section} `{item.get('id')}`: table(s) not loaded: {', '.join(missing)}")
            continue
        bound = _example_params(item, built.params)
        async with get_sessionmaker()() as s:
            try:
                await s.execute(sql_text("SET TRANSACTION READ ONLY"))
                await s.execute(sql_text("SET LOCAL statement_timeout = 5000"))
                await s.execute(sql_text("SET LOCAL search_path TO graphrag"))
                await s.execute(sql_text("EXPLAIN " + built.sql), bound)
            except Exception as exc:
                errors.append(f"{section} `{item.get('id')}`: {_db_error(exc)}")
            finally:
                await s.rollback()
    return errors


def _md_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(p.glob("*.md")) if p.is_dir() else [p])
    return out


async def import_markdown(
    paths: list[Path],
    *,
    dry_run: bool = False,
    embed: EmbedFn | None = None,
    cache: MetadataCache | None = None,
    link_entities: bool = True,
) -> list[ImportReport]:
    from backend.app.db.graph_version import bump_version, current_version
    from backend.app.db.session import session_scope

    reports: list[ImportReport] = []
    async with session_scope() as s:
        reg = await registered_tables(s)
        cols = await table_columns(s, sorted(reg))
    for path in _md_files(paths):
        rep = ImportReport(path=path)
        reports.append(rep)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            rep.errors.append(f"cannot read: {exc}")
            continue
        try:
            rep.table = parse_markdown(text).table_name
        except MetadataParseError as exc:
            rep.errors.extend(exc.errors)
            continue
        cur = reg.get(rep.table)
        if cur is None:
            rep.errors.append(f"table `{rep.table}` is not loaded (no ACTIVE TableMetadata node)")
            continue
        old = cur["meta"]
        try:
            new = apply_markdown(old, text)
        except MetadataParseError as exc:
            rep.errors.extend(exc.errors)
            continue
        types = {n: column_types(r["meta"]) for n, r in reg.items()}
        for sec, it in items(new):
            if isinstance(it.get("query"), dict):
                err = finalize_item(new, sec, it, types)
                if err:
                    rep.errors.append(f"{sec} `{it.get('id')}`: {err}")
        if rep.errors:
            continue
        rep.diff = "".join(difflib.unified_diff(
            to_markdown(old).splitlines(keepends=True), to_markdown(new).splitlines(keepends=True),
            fromfile=f"{rep.table} (database)", tofile=f"{rep.table} ({path.name})"))
        if not rep.diff:
            continue
        rep.errors.extend(f"{sec} `{it.get('id')}`: {err}" for sec, it, err in validate_all(new, cols))
        if not rep.errors:
            rep.errors.extend(await explain_errors(new, set(reg), types))
        if rep.errors or dry_run:
            continue
        new["edited"] = True
        # An edited query runs at retrieval only once its exact SQL passed the
        # data tests (unchanged items keep their verification).
        from backend.app.services.structured.refine import ensure_tested

        rep.tested = await ensure_tested(new, analysis=new.get("analysis") or {}, all_types=types,
                                         partners={n: r["meta"] for n, r in reg.items()})
        vec = (await embed([metadata_text(new)]))[0] if embed is not None else None
        async with session_scope() as s:
            from backend.app.services.structured.value_domains import attach_value_domains

            await attach_value_domains(s, new)
            tag_items(new)
            gv = await current_version(s)
            node_id: uuid.UUID = await upsert_node(s, new, vec, gv, cur["id"])
            rep.class_edges, rep.entity_edges = await rebuild_edges(
                s, node_id, new, gv, link_entities=link_entities, embed=embed)
        if cache is not None:
            # Keep the profile (sample rows, stats) of the cached entry; the
            # node in the database never stores it.
            base = cache.get(new.get("cache_key", ""), rep.table) or {}
            entry = {**base, **new}
            entry.setdefault("status", "complete")
            if entry.get("cache_key") and entry.get("format"):
                cache.put(entry)
        rep.applied = True
    if any(r.applied for r in reports):
        async with session_scope() as s:
            await bump_version(s)
    return reports

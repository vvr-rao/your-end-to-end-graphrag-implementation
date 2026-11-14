"""Load profiled tables into Postgres and connect their metadata to the graph.

For every table of a :class:`ProfileRun`:

1. **Data table** ``graphrag."<table_name>"`` with the inferred column types,
   filled by a streamed ``COPY`` (asyncpg). Tables are marked with a comment
   ``yegi-structured-data:<source path>`` so wipes and safety checks can tell
   them from the graph's own tables. An unchanged file (same sha256) is not
   reloaded; a changed file is dropped and reloaded in one transaction.
2. **Metadata node**: one ``intelligence_artifacts`` row, type
   ``TableMetadata``, ``text`` = description/columns/metric names (embedded),
   ``extra_metadata`` = the metadata file (SQL catalog included).
3. **Edges** (no LLM):
   * node -> ``ontology_class`` for each column/metric/filter with a class
     (``relationship_type`` tableColumn / tableMetric / tableFilter);
   * node -> ``entity`` (``viao:assertsAbout``) for key/dimension values that
     name an existing entity (exact normalised name, short-form variant, then
     pg_trgm >= 0.85). No entities are created.

The same edge builder is used by ``table-metadata-import`` to rebuild a
table's edges after its metadata was edited.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.predicates import VIAO_ASSERTS_ABOUT, VIAO_NS
from backend.app.services.structured.discovery import fit_identifier, sanitize_identifier
from backend.app.services.structured.metadata import (
    class_refs,
    entity_key_columns,
    items,
    metadata_text,
    rename_table,
    tag_items,
)
from backend.app.services.structured.profiling import PG_TYPES, convert_value, iter_records

SCHEMA = "graphrag"
COMMENT_PREFIX = "yegi-structured-data:"
TABLE_EDGE_KINDS = ("tableColumn", "tableMetric", "tableFilter", "tableEntity", "tableTime", "tablePlace")
COPY_BATCH = 5_000
MAX_VALUES_PER_COLUMN = 5_000
MAX_FUZZY_VALUES = 2_000
MAX_ENTITY_EDGES = 2_000
MAX_INDEXES = 8
FUZZY_THRESHOLD = 0.85


@dataclass
class LoadSummary:
    loaded: list[str] = field(default_factory=list)       # (re)created + filled
    unchanged: list[str] = field(default_factory=list)    # same file sha: no reload
    renamed: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    rows: int = 0
    class_edges: int = 0
    entity_edges: int = 0
    nodes: dict[str, str] = field(default_factory=dict)   # table -> artifact id


def q(ident: str) -> str:
    """Quote an identifier we generated (lower snake); refuse anything else."""
    if not ident or not all(ch.isalnum() or ch == "_" for ch in ident) or ident != ident.lower():
        raise ValueError(f"unsafe identifier {ident!r}")
    return f'"{ident}"'


def node_iri(table_name: str, file_sha: str) -> str:
    return f"{VIAO_NS}#TableMetadata_{table_name}_{file_sha[:12]}"


def node_payload(meta: dict) -> dict:
    """What goes into extra_metadata: the metadata file without the profile
    (whose sample rows are table data and are not needed after profiling)."""
    out = {k: v for k, v in meta.items() if k != "profile"}
    return json.loads(json.dumps(out, default=str))


# ------------------------------------------------------------------ registry


async def registered_tables(session: AsyncSession, *, status: str | None = "ACTIVE") -> dict[str, dict]:
    """table_name -> {id, file_sha, rel_path, meta, status} for TableMetadata nodes."""
    where = "AND status = :st" if status else ""
    rows = await session.execute(sql_text(f"""
        SELECT id, status, extra_metadata FROM {SCHEMA}.intelligence_artifacts
         WHERE artifact_type = 'TableMetadata' {where}
         ORDER BY created_at
    """), {"st": status} if status else {})
    out: dict[str, dict] = {}
    for aid, st, meta in rows.all():
        meta = meta or {}
        out[meta.get("table_name", "")] = {
            "id": aid, "status": st, "meta": meta,
            "file_sha": (meta.get("source") or {}).get("file_sha"),
            "rel_path": (meta.get("source") or {}).get("rel_path"),
        }
    return out


async def table_exists(session: AsyncSession, name: str) -> tuple[bool, str | None]:
    row = (await session.execute(sql_text("""
        SELECT c.oid, obj_description(c.oid, 'pg_class')
          FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = :s AND c.relname = :t
    """), {"s": SCHEMA, "t": name})).first()
    return (row is not None, row[1] if row else None)


async def data_table_names(session: AsyncSession) -> list[str]:
    """Every table this module created (identified by its comment)."""
    rows = await session.execute(sql_text("""
        SELECT c.relname FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = :s AND c.relkind = 'r'
           AND obj_description(c.oid, 'pg_class') LIKE :p
         ORDER BY 1
    """), {"s": SCHEMA, "p": COMMENT_PREFIX + "%"})
    return [r[0] for r in rows.all()]


async def drop_data_tables(session: AsyncSession) -> list[str]:
    names = await data_table_names(session)
    for n in names:
        await session.execute(sql_text(f"DROP TABLE IF EXISTS {SCHEMA}.{q(n)}"))
    return names


async def table_columns(session: AsyncSession, names: list[str]) -> dict[str, set[str]]:
    rows = await session.execute(sql_text("""
        SELECT table_name, column_name FROM information_schema.columns
         WHERE table_schema = :s AND table_name = ANY(CAST(:t AS text[]))
    """), {"s": SCHEMA, "t": names})
    out: dict[str, set[str]] = {}
    for t, c in rows.all():
        out.setdefault(t, set()).add(c)
    return out


# ------------------------------------------------------------------ data


def _records(src, meta: dict):
    cols = meta["columns"]
    for rec in iter_records(src):
        row = []
        for c in cols:
            v = convert_value(c["type"], rec.get(c["source_name"]))
            if isinstance(v, datetime) and v.tzinfo is None:
                v = v.replace(tzinfo=UTC)
            row.append(v)
        yield tuple(row)


async def _create_and_fill(session: AsyncSession, src, meta: dict) -> int:
    name = meta["table_name"]
    cols_sql = ", ".join(f"{q(c['name'])} {PG_TYPES[c['type']]}" for c in meta["columns"])
    await session.execute(sql_text(f"DROP TABLE IF EXISTS {SCHEMA}.{q(name)}"))
    await session.execute(sql_text(f"CREATE TABLE {SCHEMA}.{q(name)} ({cols_sql})"))
    comment = (COMMENT_PREFIX + (meta.get("source") or {}).get("rel_path", "")).replace("'", "''")
    await session.execute(sql_text(f"COMMENT ON TABLE {SCHEMA}.{q(name)} IS '{comment}'"))
    conn = await session.connection()
    raw = await conn.get_raw_connection()
    apg = raw.driver_connection
    columns = [c["name"] for c in meta["columns"]]
    total, batch = 0, []
    for rec in _records(src, meta):
        batch.append(rec)
        if len(batch) >= COPY_BATCH:
            await apg.copy_records_to_table(name, schema_name=SCHEMA, columns=columns, records=batch)
            total += len(batch)
            batch = []
    if batch:
        await apg.copy_records_to_table(name, schema_name=SCHEMA, columns=columns, records=batch)
        total += len(batch)
    for col in _index_columns(meta):
        idx = fit_identifier(f"{name}_{col}_idx")
        await session.execute(sql_text(
            f"CREATE INDEX IF NOT EXISTS {q(idx)} ON {SCHEMA}.{q(name)} ({q(col)})"))
    await session.execute(sql_text(f"ANALYZE {SCHEMA}.{q(name)}"))
    return total


def _index_columns(meta: dict) -> list[str]:
    names = {c["name"] for c in meta["columns"]}
    want: list[str] = []

    def add(c):
        if c in names and c not in want:
            want.append(c)

    for c in meta["columns"]:
        if c.get("role") in {"identifier", "time"}:
            add(c["name"])
    for it in meta.get("filters") or []:
        for c in it.get("columns") or []:
            add(c)
    for it in meta.get("lookups") or []:
        for j in it.get("join") or []:
            add(j.get("column"))
    return want[:MAX_INDEXES]


# ------------------------------------------------------------------ node + edges


async def upsert_node(session: AsyncSession, meta: dict, embedding: list[float] | None,
                      gv: int, existing_id: uuid.UUID | None = None) -> uuid.UUID:
    payload = node_payload(meta)
    text = metadata_text(meta)
    vec = None if embedding is None else "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"
    if existing_id is not None:
        await session.execute(sql_text(f"""
            UPDATE {SCHEMA}.intelligence_artifacts
               SET text = :text, title = :title, extra_metadata = CAST(:meta AS jsonb),
                   embedding = COALESCE(CAST(:vec AS vector), embedding), model_name = :model,
                   prompt_version = :pv, graph_version = :gv, status = 'ACTIVE'
             WHERE id = :id
        """), {"text": text, "title": meta["table_name"], "meta": json.dumps(payload),
               "vec": vec, "model": meta.get("model"), "pv": meta.get("prompt_version"),
               "gv": gv, "id": existing_id})
        return existing_id
    new_id = uuid.uuid4()
    await session.execute(sql_text(f"""
        INSERT INTO {SCHEMA}.intelligence_artifacts
            (id, artifact_identifier, artifact_type, title, text, model_name, prompt_version,
             embedding, graph_version, status, extra_metadata)
        VALUES (:id, :iri, 'TableMetadata', :title, :text, :model, :pv,
                CAST(:vec AS vector), :gv, 'ACTIVE', CAST(:meta AS jsonb))
        ON CONFLICT (artifact_identifier) DO UPDATE
           SET text = EXCLUDED.text, extra_metadata = EXCLUDED.extra_metadata,
               embedding = EXCLUDED.embedding, status = 'ACTIVE', graph_version = EXCLUDED.graph_version
        RETURNING id
    """), {"id": new_id, "iri": node_iri(meta["table_name"], (meta.get("source") or {}).get("file_sha", "")),
           "title": meta["table_name"], "text": text, "model": meta.get("model"),
           "pv": meta.get("prompt_version"), "vec": vec, "gv": gv, "meta": json.dumps(payload)})
    row = (await session.execute(sql_text(
        f"SELECT id FROM {SCHEMA}.intelligence_artifacts WHERE artifact_identifier = :iri"),
        {"iri": node_iri(meta["table_name"], (meta.get("source") or {}).get("file_sha", ""))})).first()
    return row[0]


async def _entity_lookup(session: AsyncSession) -> dict[str, uuid.UUID]:
    """normalised name (and unambiguous short forms) -> entity id."""
    from backend.app.services.db_entity_extract import _short_form_variants

    rows = (await session.execute(sql_text(
        f"SELECT id, normalized_name FROM {SCHEMA}.entities WHERE status = 'ACTIVE'"))).all()
    by_norm: dict[str, uuid.UUID] = {}
    canonical: set[str] = set()
    for eid, norm in rows:
        if norm:
            by_norm.setdefault(norm, eid)
            canonical.add(norm)
    ambiguous: set[str] = set()
    for eid, norm in rows:
        for v in _short_form_variants(norm or ""):
            if v == norm or v in canonical or v in ambiguous:
                continue
            cur = by_norm.get(v)
            if cur is None:
                by_norm[v] = eid
            elif cur != eid:
                del by_norm[v]
                ambiguous.add(v)
    return by_norm


async def _class_lookup(session: AsyncSession, refs: list[dict]) -> dict[int, uuid.UUID]:
    iris = sorted({r["class_iri"] for r in refs if r.get("class_iri")})
    labels = sorted({(r.get("class_label") or "").lower() for r in refs if r.get("class_label")})
    rows = (await session.execute(sql_text(f"""
        SELECT id, iri, lower(label) FROM {SCHEMA}.ontology_classes
         WHERE iri = ANY(CAST(:iris AS text[])) OR lower(label) = ANY(CAST(:labels AS text[]))
    """), {"iris": iris, "labels": labels})).all()
    by_iri = {iri: cid for cid, iri, _l in rows}
    by_label: dict[str, uuid.UUID] = {}
    for cid, _iri, lab in rows:
        if lab:
            by_label.setdefault(lab, cid)
    out = {}
    for i, r in enumerate(refs):
        cid = by_iri.get(r.get("class_iri") or "") or by_label.get((r.get("class_label") or "").lower())
        if cid:
            out[i] = cid
    return out


def _edge(src_id, tgt_type: str, tgt_id, kind: str, gv: int, extra: dict) -> dict:
    return {
        "source_node_type": "intelligence_artifact", "source_node_id": src_id,
        "target_node_type": tgt_type, "target_node_id": tgt_id,
        "predicate_iri": VIAO_ASSERTS_ABOUT, "predicate_label": "viao:assertsAbout",
        "relationship_type": kind, "relationship_source": "DOCUMENT_EXTRACTION",
        "is_authoritative": True, "source_artifact_id": src_id, "graph_version": gv,
        "extra_metadata": json.dumps(extra, default=str),
    }


async def rebuild_edges(session: AsyncSession, node_id: uuid.UUID, meta: dict, gv: int,
                        entity_lookup: dict[str, uuid.UUID] | None = None,
                        *, link_entities: bool = True, embed=None,
                        log: Callable[[str], None] | None = None) -> tuple[int, int]:
    """Delete the node's table edges and re-derive them from ``meta``:
    classes, entities, and (``graph_links``) the periods and places the
    table's values cover."""
    await session.execute(sql_text(f"""
        DELETE FROM {SCHEMA}.graph_relationships
         WHERE source_node_type = 'intelligence_artifact' AND source_node_id = :id
           AND relationship_type = ANY(CAST(:kinds AS text[]))
    """), {"id": node_id, "kinds": list(TABLE_EDGE_KINDS)})
    edges: list[dict] = []

    refs = class_refs(meta)
    resolved = await _class_lookup(session, refs) if refs else {}
    per_class: dict[tuple[uuid.UUID, str], list[str]] = {}
    tagged: dict[uuid.UUID, list[str]] = {}       # class -> stored queries tagged with it
    for i, r in enumerate(refs):
        cid = resolved.get(i)
        if cid:
            per_class.setdefault((cid, r["kind"]), []).append(r["name"])
            for _sec, it in items(meta):
                if any((t.get("iri") and t["iri"] == r.get("class_iri"))
                       or (t.get("label") and t["label"] == r.get("class_label"))
                       for t in (it.get("tags") or {}).get("classes") or []):
                    if it["id"] not in tagged.setdefault(cid, []):
                        tagged[cid].append(it["id"])
    for (cid, kind), names in per_class.items():
        edges.append(_edge(node_id, "ontology_class", cid, kind, gv,
                           {"table": meta["table_name"], "names": names,
                            "queries": tagged.get(cid, [])}))
    n_class = len(edges)

    n_ent = 0
    if link_entities:
        from backend.app.services.db_entity_extract import _normalize_name, _short_form_variants

        lookup = entity_lookup if entity_lookup is not None else await _entity_lookup(session)
        hits: dict[uuid.UUID, dict] = {}
        misses: dict[str, tuple[str, str]] = {}
        for col in entity_key_columns(meta):
            vals = (await session.execute(sql_text(
                f"SELECT DISTINCT {q(col)} FROM {SCHEMA}.{q(meta['table_name'])} "
                f"WHERE {q(col)} IS NOT NULL LIMIT :n"), {"n": MAX_VALUES_PER_COLUMN})).scalars().all()
            for v in vals:
                norm = _normalize_name(str(v))
                if not norm or len(norm) < 2:
                    continue
                eid = lookup.get(norm)
                match = "exact"
                if eid is None:
                    # Tables carry full legal names ("BHP GROUP LIMITED"), the
                    # text carries the short one ("BHP Group"): strip the
                    # corporate suffixes on the TABLE side too, longest first.
                    for short in _short_form_variants(norm)[1:]:
                        eid = lookup.get(short)
                        if eid is not None:
                            match = "short_form"
                            break
                if eid is not None:
                    h = hits.setdefault(eid, {"columns": set(), "values": [], "match": match})
                    h["columns"].add(col)
                    if len(h["values"]) < 5 and str(v) not in h["values"]:
                        h["values"].append(str(v))
                elif len(misses) < MAX_FUZZY_VALUES and not norm.isdigit():
                    misses.setdefault(norm, (col, str(v)))
        if misses:
            rows = (await session.execute(sql_text(f"""
                SELECT q.v, e.id FROM unnest(CAST(:vals AS text[])) AS q(v)
                CROSS JOIN LATERAL (
                    SELECT id FROM {SCHEMA}.entities e
                     WHERE e.status = 'ACTIVE' AND e.normalized_name % q.v
                       AND similarity(e.normalized_name, q.v) >= :th
                     ORDER BY similarity(e.normalized_name, q.v) DESC LIMIT 1) e
            """), {"vals": list(misses), "th": FUZZY_THRESHOLD})).all()
            for norm, eid in rows:
                col, raw = misses[norm]
                h = hits.setdefault(eid, {"columns": set(), "values": [], "match": "fuzzy"})
                h["columns"].add(col)
                if len(h["values"]) < 5:
                    h["values"].append(raw)
        for eid, h in list(hits.items())[:MAX_ENTITY_EDGES]:
            edges.append(_edge(node_id, "entity", eid, "tableEntity", gv,
                               {"table": meta["table_name"], "columns": sorted(h["columns"]),
                                "values": h["values"], "match": h["match"]}))
        n_ent = len(edges) - n_class

        from backend.app.services.structured.graph_links import time_and_place_edges

        tp_edges, tp = await time_and_place_edges(session, node_id, meta, gv, embed=embed)
        edges += tp_edges
        n_ent += tp["places"]
        if log is not None and (tp["periods"] or tp["places"]):
            log(f"[structured] {meta['table_name']}: {tp['periods']} period(s) at {tp['time_level']} "
                f"level from {tp['time_columns']}, {tp['places']} place(s) "
                f"({tp['places_minted']} new), {tp['containment']} containment edge(s)")

    for i in range(0, len(edges), 500):
        batch = edges[i:i + 500]
        await session.execute(sql_text(f"""
            INSERT INTO {SCHEMA}.graph_relationships
                (source_node_type, source_node_id, target_node_type, target_node_id,
                 predicate_iri, predicate_label, relationship_type, relationship_source,
                 is_authoritative, source_artifact_id, graph_version, extra_metadata)
            VALUES (:source_node_type, :source_node_id, :target_node_type, :target_node_id,
                    :predicate_iri, :predicate_label, :relationship_type, :relationship_source,
                    :is_authoritative, :source_artifact_id, :graph_version, CAST(:extra_metadata AS jsonb))
        """), batch)
    return n_class, n_ent


# ------------------------------------------------------------------ driver


EmbedFn = Callable[[list[str]], Any]


async def load_and_link(run, *, root_name: str, cache=None, embed: EmbedFn | None = None,
                        link_entities: bool = True, log: Callable[[str], None] = print) -> LoadSummary:
    """Load every table of ``run`` and (re)build its node and edges.

    ``embed`` is an async callable ``list[str] -> list[vector]`` (``Embedder.embed``);
    ``None`` stores nodes without embeddings (tests, or no embedding key).
    One transaction per table, so one bad file does not undo the others.
    """
    from backend.app.db.graph_version import bump_version, current_version
    from backend.app.db.session import session_scope

    summary = LoadSummary()
    prefix = sanitize_identifier(root_name)
    async with session_scope() as s:
        lookup = await _entity_lookup(s) if link_entities else {}
        gv = await current_version(s)

    for meta in list(run.tables):
        src = run.sources.get(meta["table_name"])
        if src is None:
            summary.failed.append((meta["table_name"], "source file not found"))
            continue
        try:
            async with session_scope() as s:
                reg = await registered_tables(s)
                name = meta["table_name"]
                cur = reg.get(name)
                if cur and cur["rel_path"] != meta["source"]["rel_path"] and cur["file_sha"] != src.file_sha:
                    new = fit_identifier(f"{prefix}_{name}")
                    if new in reg and reg[new]["rel_path"] != meta["source"]["rel_path"]:
                        raise RuntimeError(f"table name {name} is taken by {cur['rel_path']} "
                                           f"and {new} by {reg[new]['rel_path']}")
                    log(f"[structured] {name} already holds {cur['rel_path']}; loading "
                        f"{meta['source']['rel_path']} as {new}")
                    meta = rename_table(meta, name, new)
                    summary.renamed.append((name, new))
                    run.sources[new] = src
                    if cache is not None:
                        cache.put(meta)
                    name, cur = new, reg.get(new)
                exists, comment = await table_exists(s, name)
                if exists and not (comment or "").startswith(COMMENT_PREFIX):
                    raise RuntimeError(f"graphrag.{name} exists and is not a structured-data table")
                if cur and cur["file_sha"] == src.file_sha and exists:
                    summary.unchanged.append(name)
                    existing_id = cur["id"]
                else:
                    if cur:
                        await s.execute(sql_text(
                            f"UPDATE {SCHEMA}.intelligence_artifacts SET status = 'STALE' WHERE id = :id"),
                            {"id": cur["id"]})
                    n = await _create_and_fill(s, src, meta)
                    summary.rows += n
                    summary.loaded.append(name)
                    existing_id = None
                # Valid values per filter column / parameter, read from the
                # table just loaded (no LLM): the selection model sees them and
                # parameter values are checked against them before SQL runs.
                from backend.app.services.structured.value_domains import attach_value_domains

                await attach_value_domains(s, meta)
                tag_items(meta)
                vec = None
                if embed is not None:
                    vec = (await embed([metadata_text(meta)]))[0]
                node_id = await upsert_node(s, meta, vec, gv, existing_id)
                nc, ne = await rebuild_edges(s, node_id, meta, gv, lookup, link_entities=link_entities,
                                               embed=embed, log=log)
                summary.class_edges += nc
                summary.entity_edges += ne
                summary.nodes[name] = str(node_id)
                log(f"[structured] {name}: "
                    f"{'loaded ' + str(n) + ' rows' if name in summary.loaded else 'unchanged'}, "
                    f"{nc} class edge(s), {ne} entity edge(s)")
        except Exception as exc:  # one table must not stop the rest
            summary.failed.append((meta["table_name"], f"{type(exc).__name__}: {exc}"))
            log(f"[structured] FAILED {meta['table_name']}: {exc}")

    if summary.loaded or summary.nodes:
        async with session_scope() as s:
            await bump_version(s)
    return summary


def copy_meta(meta: dict) -> dict:
    return copy.deepcopy(meta)

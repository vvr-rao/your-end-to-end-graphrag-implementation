"""SQL helpers for Milestone F retrieval pipeline.

Three responsibilities:
  1. `bfs_expand`            -- step 7: walk `graph_relationships`
                                from seed nodes up to `hops`, return
                                a {(node_id, node_type) -> score} map.
  2. `fetch_candidate_*`     -- step 8: pull chunks/docs/artifacts
                                touched by the expanded node set.
  3. `vector_rerank`         -- step 9c: order a candidate set by
                                L2 distance from a probe embedding,
                                WITHOUT scanning the full table.
  4. `fetch_class_subtree`   -- subClassOf walk (used by exhaustive
                                mode to expand class constraints).
  5. `fetch_exhaustive_*`    -- step 8 for exhaustive_search: hard
                                intersection of constraint groups.

Pure SQL; no LLM calls. Designed to keep BFS bounded -- we cap hops,
deduplicate within the recursion, and rely on Postgres' CYCLE clause
to avoid infinite loops on densely-connected ontology fragments.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession


def _vec_str(v: list[float]) -> str:
    """Render a Python float list as pgvector's text form `[x,y,z]`.

    asyncpg + raw `text()` SQL can't bind a Python list to a `vector`
    column without help -- pgvector's Python adapter only kicks in for
    ORM-mapped columns. We pass the literal string and `CAST(:probe
    AS vector)` does the rest. format compact enough to keep query
    sizes tame on 1024-dim vectors.
    """
    return "[" + ",".join(format(x, ".6f") for x in v) + "]"


_BFS_SQL = sql_text("""
WITH RECURSIVE bfs(node_id, node_type, hop, score) AS (
    SELECT seed.id, seed.type, 0, 1.0::float
      FROM unnest(
             CAST(:seed_ids   AS uuid[]),
             CAST(:seed_types AS text[])
           ) AS seed(id, type)
  UNION ALL
    SELECT
        CASE
            WHEN gr.source_node_id = b.node_id AND gr.source_node_type = b.node_type
            THEN gr.target_node_id ELSE gr.source_node_id END,
        CASE
            WHEN gr.source_node_id = b.node_id AND gr.source_node_type = b.node_type
            THEN gr.target_node_type ELSE gr.source_node_type END,
        b.hop + 1,
        b.score * CAST(:decay AS float)
      FROM bfs b
      JOIN graphrag.graph_relationships gr
        ON  ((gr.source_node_id = b.node_id AND gr.source_node_type = b.node_type)
          OR (gr.target_node_id = b.node_id AND gr.target_node_type = b.node_type))
     WHERE b.hop < CAST(:max_hops AS int)
) CYCLE node_id, node_type SET is_cycle USING path
SELECT node_id, node_type, max(score) AS best_score, min(hop) AS min_hop
  FROM bfs
 WHERE NOT is_cycle
 GROUP BY node_id, node_type
""")


async def bfs_expand(
    session: AsyncSession,
    seeds: list[tuple[uuid.UUID, str]],
    *,
    max_hops: int = 2,
    decay: float = 0.7,
) -> dict[tuple[uuid.UUID, str], dict[str, Any]]:
    """Walk `graph_relationships` outward from seeds. Returns a map of
    `(node_id, node_type) -> {score, hop}`.

    `decay` shrinks score by this factor each hop (0.7 by default).
    `max_hops` caps recursion depth (2 by default; configurable).
    """
    if not seeds:
        return {}
    seed_ids = [str(sid) for sid, _ in seeds]
    seed_types = [stype for _, stype in seeds]
    result = await session.execute(
        _BFS_SQL,
        {
            "seed_ids": seed_ids,
            "seed_types": seed_types,
            "max_hops": max_hops,
            "decay": decay,
        },
    )
    out: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
    for nid, ntype, score, hop in result.all():
        out[(nid, ntype)] = {"score": float(score), "hop": int(hop)}
    return out


async def fetch_candidate_chunks_for_entities(
    session: AsyncSession,
    entity_ids: list[uuid.UUID],
    *,
    limit: int = 500,
) -> list[tuple[uuid.UUID, float]]:
    """Chunks that assertAbout any of the given entities. Ordered by
    how many of the listed entities each chunk asserts about."""
    if not entity_ids:
        return []
    result = await session.execute(
        sql_text("""
        SELECT gr.source_chunk_id, count(*) AS hits
          FROM graphrag.graph_relationships gr
         WHERE gr.predicate_label = 'viao:assertsAbout'
           AND gr.relationship_source = 'DOCUMENT_EXTRACTION'
           AND gr.target_node_type = 'entity'
           AND gr.target_node_id = ANY(CAST(:entity_ids AS uuid[]))
           AND gr.source_chunk_id IS NOT NULL
         GROUP BY gr.source_chunk_id
         ORDER BY hits DESC
         LIMIT :limit
        """),
        {"entity_ids": [str(eid) for eid in entity_ids], "limit": limit},
    )
    return [(cid, float(hits)) for cid, hits in result.all()]


async def fetch_candidate_chunks_for_time_instances(
    session: AsyncSession,
    time_instance_ids: list[uuid.UUID],
    *,
    limit: int = 500,
) -> list[tuple[uuid.UUID, float]]:
    """Chunks linked to any of the given time_instances via time:hasTime."""
    if not time_instance_ids:
        return []
    result = await session.execute(
        sql_text("""
        SELECT gr.source_chunk_id, count(*) AS hits
          FROM graphrag.graph_relationships gr
         WHERE gr.predicate_label = 'time:hasTime'
           AND gr.target_node_type = 'time_instance'
           AND gr.target_node_id = ANY(CAST(:time_ids AS uuid[]))
           AND gr.source_chunk_id IS NOT NULL
         GROUP BY gr.source_chunk_id
         ORDER BY hits DESC
         LIMIT :limit
        """),
        {"time_ids": [str(tid) for tid in time_instance_ids], "limit": limit},
    )
    return [(cid, float(hits)) for cid, hits in result.all()]


async def fetch_candidate_artifacts_for_entities(
    session: AsyncSession,
    entity_ids: list[uuid.UUID],
    *,
    artifact_types: tuple[str, ...] | None = None,
    limit: int = 200,
) -> list[tuple[uuid.UUID, float]]:
    """Intelligence artifacts asserting about any of the given entities."""
    if not entity_ids:
        return []
    type_clause = ""
    params: dict[str, Any] = {
        "entity_ids": [str(eid) for eid in entity_ids],
        "limit": limit,
    }
    if artifact_types is not None:
        type_clause = "AND a.artifact_type = ANY(CAST(:types AS text[]))"
        params["types"] = list(artifact_types)
    result = await session.execute(
        sql_text(f"""
        SELECT a.id, count(*) AS hits
          FROM graphrag.intelligence_artifacts a
          JOIN graphrag.graph_relationships gr
            ON gr.source_node_id = a.id
           AND gr.source_node_type = 'intelligence_artifact'
           AND gr.predicate_label = 'viao:assertsAbout'
           AND gr.target_node_type = 'entity'
           AND gr.target_node_id = ANY(CAST(:entity_ids AS uuid[]))
         WHERE a.status = 'ACTIVE'
           {type_clause}
         GROUP BY a.id
         ORDER BY hits DESC
         LIMIT :limit
        """),
        params,
    )
    return [(aid, float(hits)) for aid, hits in result.all()]


async def fetch_table_artifacts_for_chunks(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
    *,
    limit: int = 200,
) -> list[tuple[uuid.UUID, float]]:
    """StructuredTable artifacts derived from documents that own any of
    the given chunks.

    Path: chunks -> document_id -> intelligence_artifacts where
    artifact_type='StructuredTable' AND graph_relationships ties the
    table to the document via 'viao:derivedFromDocument'.

    Why this exists: when a user asks "what was BHP's 2025 revenue?",
    we reach BHP-related chunks via the chunk->entity edges, but the
    actual revenue table in the BHP 10-K won't have a direct
    table->entity edge (table cells contain numbers, not "BHP" text).
    Pulling tables document-mediated -- "for every doc whose chunks we
    found, include its tables" -- closes that gap.

    Scoring: hits = number of distinct candidate chunks owned by the
    table's source document. So a doc with many relevant chunks pulls
    its tables higher than a doc with one tangential chunk.
    """
    if not chunk_ids:
        return []
    result = await session.execute(
        sql_text("""
        WITH candidate_docs AS (
            SELECT c.document_id, count(*) AS n_chunks
              FROM graphrag.chunks c
             WHERE c.id = ANY(CAST(:chunk_ids AS uuid[]))
               AND c.document_id IS NOT NULL
             GROUP BY c.document_id
        )
        SELECT a.id, cd.n_chunks::float AS hits
          FROM graphrag.intelligence_artifacts a
          JOIN graphrag.graph_relationships gr
            ON gr.source_node_id = a.id
           AND gr.source_node_type = 'intelligence_artifact'
           AND gr.predicate_label = 'viao:derivedFromDocument'
           AND gr.target_node_type = 'document'
          JOIN candidate_docs cd ON cd.document_id = gr.target_node_id
         WHERE a.status = 'ACTIVE'
           AND a.artifact_type = 'StructuredTable'
         ORDER BY hits DESC
         LIMIT :limit
        """),
        {
            "chunk_ids": [str(cid) for cid in chunk_ids],
            "limit": limit,
        },
    )
    return [(aid, float(hits)) for aid, hits in result.all()]


async def fetch_fulltext_chunks_for_chunks(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
    *,
    limit: int = 500,
) -> list[tuple[uuid.UUID, float, uuid.UUID]]:
    """Full-text chunks (kind='fulltext') belonging to the documents that own
    any of the given (summary) chunks.

    Mirrors `fetch_table_artifacts_for_chunks`: the graph reaches a document via
    its summary chunks' entity edges, but those edges point at summary chunks.
    When a document also carries verbatim full-text chunks (ingested with
    --full-text-chunks), this swaps them into the retrieval candidate pool so
    vector rerank surfaces the exact passage and citations are verbatim.

    Returns (fulltext_chunk_id, hits, document_id) where hits = the number of
    candidate summary chunks owned by that document — i.e. the document's graph
    score, propagated to each of its full-text chunks. Empty when no full-text
    chunks exist (→ caller keeps today's summary-chunk behavior)."""
    if not chunk_ids:
        return []
    result = await session.execute(
        sql_text("""
        WITH candidate_docs AS (
            SELECT c.document_id, count(*) AS n_chunks
              FROM graphrag.chunks c
             WHERE c.id = ANY(CAST(:chunk_ids AS uuid[]))
               AND c.document_id IS NOT NULL
             GROUP BY c.document_id
        )
        SELECT ft.id, cd.n_chunks::float AS hits, ft.document_id
          FROM graphrag.chunks ft
          JOIN candidate_docs cd ON cd.document_id = ft.document_id
         WHERE ft.kind = 'fulltext'
           AND ft.status = 'ACTIVE'
           AND ft.embedding IS NOT NULL
         ORDER BY hits DESC, ft.chunk_index
         LIMIT :limit
        """),
        {"chunk_ids": [str(cid) for cid in chunk_ids], "limit": limit},
    )
    return [(cid, float(hits), did) for cid, hits, did in result.all()]


async def fetch_chunk_document_ids(
    session: AsyncSession, chunk_ids: list[uuid.UUID]
) -> dict[uuid.UUID, uuid.UUID]:
    """Lightweight chunk_id -> document_id map (no joins, no text)."""
    if not chunk_ids:
        return {}
    result = await session.execute(
        sql_text("""
        SELECT c.id, c.document_id
          FROM graphrag.chunks c
         WHERE c.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": [str(cid) for cid in chunk_ids]},
    )
    return {cid: did for cid, did in result.all()}


async def vector_rerank_chunks(
    session: AsyncSession,
    candidate_chunk_ids: list[uuid.UUID],
    probe_embedding: list[float],
    *,
    top_k: int = 50,
) -> list[tuple[uuid.UUID, float]]:
    """Order `candidate_chunk_ids` by L2 distance from `probe_embedding`.
    Returns (chunk_id, distance) pairs; smaller distance = more relevant."""
    if not candidate_chunk_ids:
        return []
    result = await session.execute(
        sql_text("""
        SELECT c.id, (c.embedding <-> CAST(:probe AS vector)) AS dist
          FROM graphrag.chunks c
         WHERE c.id = ANY(CAST(:ids AS uuid[]))
           AND c.embedding IS NOT NULL
         ORDER BY dist
         LIMIT :limit
        """),
        {
            "ids": [str(cid) for cid in candidate_chunk_ids],
            "probe": _vec_str(probe_embedding),
            "limit": top_k,
        },
    )
    return [(cid, float(dist)) for cid, dist in result.all()]


async def vector_rerank_artifacts(
    session: AsyncSession,
    candidate_artifact_ids: list[uuid.UUID],
    probe_embedding: list[float],
    *,
    top_k: int = 50,
) -> list[tuple[uuid.UUID, float]]:
    """Vector-rerank artifacts by L2 distance from the probe embedding."""
    if not candidate_artifact_ids:
        return []
    result = await session.execute(
        sql_text("""
        SELECT a.id, (a.embedding <-> CAST(:probe AS vector)) AS dist
          FROM graphrag.intelligence_artifacts a
         WHERE a.id = ANY(CAST(:ids AS uuid[]))
           AND a.embedding IS NOT NULL
         ORDER BY dist
         LIMIT :limit
        """),
        {
            "ids": [str(aid) for aid in candidate_artifact_ids],
            "probe": _vec_str(probe_embedding),
            "limit": top_k,
        },
    )
    return [(aid, float(dist)) for aid, dist in result.all()]


async def fetch_class_subtree(
    session: AsyncSession,
    root_class_ids: list[uuid.UUID],
    *,
    max_depth: int = 10,
) -> set[uuid.UUID]:
    """Return the set of class_ids that are `root_class_ids` themselves
    plus every transitive descendant via rdfs:subClassOf in
    graph_relationships (source='ONTOLOGY')."""
    if not root_class_ids:
        return set()
    result = await session.execute(
        sql_text("""
        WITH RECURSIVE descendants(class_id, depth) AS (
            SELECT id, 0 FROM unnest(CAST(:roots AS uuid[])) AS id
          UNION
            SELECT gr.source_node_id, d.depth + 1
              FROM descendants d
              JOIN graphrag.graph_relationships gr
                ON gr.target_node_id   = d.class_id
               AND gr.target_node_type = 'ontology_class'
               AND gr.source_node_type = 'ontology_class'
               AND gr.predicate_label  = 'rdfs:subClassOf'
               AND gr.relationship_source = 'ONTOLOGY'
             WHERE d.depth < CAST(:max_depth AS int)
        )
        SELECT DISTINCT class_id FROM descendants
        """),
        {"roots": [str(rid) for rid in root_class_ids], "max_depth": max_depth},
    )
    return {row[0] for row in result.all()}


async def fetch_exhaustive_intersection(
    session: AsyncSession,
    entity_id_groups: list[list[uuid.UUID]],
    *,
    limit: int = 1000,
) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """Return (document_id, chunk_id) pairs where the chunk asserts about
    at least ONE entity from EACH group. Each group is an alternative-set
    representing one constraint of the query.

    Example: query "regulations about EV production in Asia" yields
    three groups: [Regulation*], [EV Production*], [Asia countries*].
    A chunk qualifies only if it links to at least one entity from
    each group (logical AND of ORs).
    """
    if not entity_id_groups or not all(entity_id_groups):
        return []

    # Build the WHERE chain dynamically.
    where_clauses = []
    params: dict[str, Any] = {"limit": limit}
    for i, group in enumerate(entity_id_groups):
        params[f"group_{i}"] = [str(eid) for eid in group]
        where_clauses.append(f"""
        c.id IN (
          SELECT gr.source_chunk_id
            FROM graphrag.graph_relationships gr
           WHERE gr.predicate_label = 'viao:assertsAbout'
             AND gr.relationship_source = 'DOCUMENT_EXTRACTION'
             AND gr.target_node_type = 'entity'
             AND gr.target_node_id = ANY(CAST(:group_{i} AS uuid[]))
             AND gr.source_chunk_id IS NOT NULL
        )
        """)
    where_chain = " AND ".join(where_clauses)

    result = await session.execute(
        sql_text(f"""
        SELECT c.document_id, c.id
          FROM graphrag.chunks c
         WHERE c.status = 'ACTIVE'
           AND {where_chain}
         ORDER BY c.document_id, c.chunk_index
         LIMIT :limit
        """),
        params,
    )
    return [(did, cid) for did, cid in result.all()]


async def fetch_chunk_text(
    session: AsyncSession, chunk_ids: list[uuid.UUID]
) -> dict[uuid.UUID, dict[str, Any]]:
    """Bulk-load chunk rows for prompt-context assembly."""
    if not chunk_ids:
        return {}
    result = await session.execute(
        sql_text("""
        SELECT c.id, c.chunk_identifier, c.text, c.document_id,
               d.title, d.document_identifier
          FROM graphrag.chunks c
          JOIN graphrag.documents d ON d.id = c.document_id
         WHERE c.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": [str(cid) for cid in chunk_ids]},
    )
    return {
        cid: {
            "iri": ciri, "text": text,
            "document_id": did, "document_title": dtitle,
            "document_iri": diri,
        }
        for cid, ciri, text, did, dtitle, diri in result.all()
    }


async def fetch_artifact_rows(
    session: AsyncSession, artifact_ids: list[uuid.UUID]
) -> dict[uuid.UUID, dict[str, Any]]:
    """Bulk-load artifact rows for prompt-context assembly."""
    if not artifact_ids:
        return {}
    result = await session.execute(
        sql_text("""
        SELECT a.id, a.artifact_identifier, a.artifact_type, a.text, a.confidence
          FROM graphrag.intelligence_artifacts a
         WHERE a.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": [str(aid) for aid in artifact_ids]},
    )
    return {
        aid: {
            "iri": airi, "type": atype, "text": atext,
            "confidence": float(conf) if conf is not None else None,
        }
        for aid, airi, atype, atext, conf in result.all()
    }

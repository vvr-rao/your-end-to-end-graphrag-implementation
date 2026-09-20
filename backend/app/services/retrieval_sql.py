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


# The same walk restricted to the edges that carry meaning between instances:
# entity -> entity relationships, plus the temporal hierarchy. Class membership
# (rdf:type / subClassOf) and chunk mentions are not traversed, so a seed
# reaches only what it is RELATED to, not everything sharing a class with it.
# Measured on 182 MultiHop-RAG questions (multihop-rag-subset, 515
# relationships): walking every edge reached 40/40 documents on every question
# (7% precision); entity relationships only reached ~7-9 with 98-100% recall
# on questions that name their subjects.
#
# The temporal hierarchy is NOT walked here even though it is instance-level:
# `intervalDuring` edges are directional and this walk is not, so admitting
# them would let a MONTH_2023_10 seed climb to YEAR_2023 and then descend into
# every other month of 2023. `expand_time_instances` below walks it properly.
_ENTITY_BFS_SQL = sql_text(
    _BFS_SQL.text.replace(
        "WHERE b.hop < CAST(:max_hops AS int)",
        "WHERE b.hop < CAST(:max_hops AS int)\n"
        "       AND gr.source_node_type = 'entity'\n"
        "       AND gr.target_node_type = 'entity'",
    )
)


async def bfs_expand(
    session: AsyncSession,
    seeds: list[tuple[uuid.UUID, str]],
    *,
    max_hops: int = 2,
    decay: float = 0.7,
    entity_relationships_only: bool = False,
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
        _ENTITY_BFS_SQL if entity_relationships_only else _BFS_SQL,
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


async def fetch_relationships_among_entities(
    session: AsyncSession,
    entity_ids: list[uuid.UUID],
    *,
    limit: int = 40,
    scope: str = "either",
) -> list[dict[str, Any]]:
    """The entity -> entity edges BFS reached, rendered as readable triples.

    WHY THIS EXISTS. Until now the graph could only influence an answer
    INDIRECTLY: BFS expanded to entities, step 8 projected those entities down
    to chunks, and RRF fused chunk-id lists. The relationship itself was
    discarded at the projection -- `rrf_fuse` takes `list[list[UUID]]`, a
    single flat id space, so a triple has no representation there. The
    consequence was measurable: on a 60-doc corpus holding 14 defeated /
    advancedOver edges, "which teams defeated which other teams" answered
    "the retrieved evidence contains no information", because the model is
    shown only chunk text and no chunk states the full picture.

    So this deliberately BYPASSES fusion. Triples are not ranked against
    chunks -- they are a different kind of thing, and forcing them into one
    ranked list is what lost them. They are fetched for the entities BFS
    already found and appended to the evidence packet directly.

    `scope` decides how strictly an edge must sit inside the neighbourhood:

      "both"   -- both endpoints reached by BFS. Safe but lossy: measured on
                  "which players signed with or transferred to new teams",
                  only 44 of 132 edges qualified and all three `signedWith`
                  edges were excluded because one endpoint (the club) was not
                  reached, so the question was answered "no information" while
                  the exact facts sat in the graph.
      "either" -- one endpoint suffices; the same question then sees 111 of
                  132 and recovers all three. This is only safe because the
                  caller RANKS by relevance before capping -- with the old
                  question-blind cut a wider pool would just have admitted
                  more noise.

    Ordering is support-first here; the caller re-ranks against the question.
    """
    if not entity_ids:
        return []
    _match = (
        "(gr.source_node_id = ANY(CAST(:ids AS uuid[]))"
        " OR gr.target_node_id = ANY(CAST(:ids AS uuid[])))"
        if scope == "either" else
        "gr.source_node_id = ANY(CAST(:ids AS uuid[]))"
        " AND gr.target_node_id = ANY(CAST(:ids AS uuid[]))"
    )
    result = await session.execute(
        sql_text("""
        SELECT s.name AS subject,
               -- A graphrag#relatedTo edge carries its relation as a phrase
               -- ("was born in"); show and rank on that, not on "relatedTo".
               coalesce(gr.extra_metadata ->> 'relation', p.label,
                        split_part(gr.predicate_iri, '#', 2)) AS predicate,
               o.name AS object,
               gr.extra_metadata ->> 'evidence' AS evidence,
               coalesce((gr.extra_metadata ->> 'support_count')::int, 1) AS support,
               gr.predicate_iri,
               coalesce(gr.extra_metadata ->> 'evidence_kind', '') AS evidence_kind
          FROM graphrag.graph_relationships gr
          JOIN graphrag.entities s ON s.id = gr.source_node_id
          JOIN graphrag.entities o ON o.id = gr.target_node_id
          LEFT JOIN graphrag.ontology_object_properties p
                 ON p.iri = gr.predicate_iri
         WHERE gr.source_node_type = 'entity'
           AND gr.target_node_type = 'entity'
           -- World-knowledge edges (enrich-geo's "Bangalore is in India")
           -- are admitted but MARKED, never silently mixed in with
           -- document-sourced claims.
           --
           -- They were excluded outright at first, reasoning that no answer
           -- should rest on a sentence no document wrote. That is right for
           -- most questions and wrong for the one kind where the containment
           -- IS the answer: measured 2026-09-20, "Which cities are in India?"
           -- walked Bangalore -> Republic of India, reached both cities, and
           -- then answered "the retrieved evidence contains no specific
           -- cities in India" because the only edge that said so was filtered
           -- out. A system that holds the answer and declines to give it is
           -- the worse failure; provenance is preserved by labelling instead.
           AND __MATCH__
         ORDER BY support DESC, s.name
         LIMIT :limit
        """.replace("__MATCH__", _match)),
        {"ids": [str(e) for e in entity_ids], "limit": limit},
    )
    out: list[dict[str, Any]] = []
    for subj, pred, obj, ev, support, pred_iri, ev_kind in result.all():
        if not subj or not obj:
            continue
        out.append({
            "subject": subj,
            "predicate": pred or "related to",
            "object": obj,
            "evidence": (ev or "").strip(),
            "support": support,
            "predicate_iri": pred_iri,
            # True where no document states this -- enrich-geo containment.
            # The caller labels it so an answer can use it without presenting
            # it as a corpus claim.
            "world_knowledge": ev_kind == "world_knowledge",
        })
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
    per_document_limit: int = 50,
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
    chunks exist (→ caller keeps today's summary-chunk behavior).

    `per_document_limit` is what stops ONE document eating the whole budget.
    `hits` is a per-DOCUMENT constant, so a plain `ORDER BY hits DESC LIMIT
    500` emitted every chunk of the top-scoring document first: a 600-chunk
    label claimed all 500 slots and no other document appeared at all. That
    is half of how a single document came to supply 30 of 30 evidence items.
    The window keeps each document's best `chunk_index` prefix and leaves
    room for the rest; the global `limit` stays as a backstop."""
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
        ),
        ranked AS (
            SELECT ft.id, cd.n_chunks::float AS hits, ft.document_id,
                   ft.chunk_index,
                   row_number() OVER (
                       PARTITION BY ft.document_id ORDER BY ft.chunk_index
                   ) AS rn
              FROM graphrag.chunks ft
              JOIN candidate_docs cd ON cd.document_id = ft.document_id
             WHERE ft.kind = 'fulltext'
               AND ft.status = 'ACTIVE'
               AND ft.embedding IS NOT NULL
        )
        SELECT id, hits, document_id
          FROM ranked
         WHERE rn <= :per_doc
         ORDER BY hits DESC, chunk_index
         LIMIT :limit
        """),
        {
            "chunk_ids": [str(cid) for cid in chunk_ids],
            "limit": limit,
            "per_doc": per_document_limit,
        },
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


async def document_sources(session: AsyncSession) -> list[str]:
    """Publication names from each document's header line ("Source: TechCrunch").

    Used to keep publications out of graph seeding: "as reported by
    TechCrunch" names a source to filter on, not an entity to start from --
    and a publication entity, if extracted at all, would link to every one of
    its articles.
    """
    result = await session.execute(sql_text("""
        SELECT DISTINCT btrim(substring(text FROM
               'Source:[ \\t]*([^\\r\\n]+)'))
          FROM graphrag.chunks
         WHERE chunk_index = 0 AND status = 'ACTIVE'
    """))
    return [r[0] for r in result.all() if r[0]]


async def match_entities_by_name_or_alias(
    session: AsyncSession, term: str, *, min_similarity: float = 0.4, limit: int = 10,
) -> list[uuid.UUID]:
    """Entities named `term`: trigram on the normalised name, OR an exact
    (case-insensitive) stored alias. The alias arm is what lets "FTX" reach
    `FTX Trading Ltd.`, whose name similarity (0.25) is under the threshold."""
    result = await session.execute(sql_text("""
        SELECT id FROM (
          SELECT e.id,
                 GREATEST(similarity(e.normalized_name, lower(:t)),
                          CASE WHEN jsonb_typeof(e.extra_metadata -> 'aliases') = 'array'
                                AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(
                                               e.extra_metadata -> 'aliases') a
                                             WHERE lower(a) = lower(:t))
                               THEN 1.0 ELSE 0.0 END) AS score
            FROM graphrag.entities e
        ) x
         WHERE score >= :min
         ORDER BY score DESC
         LIMIT :limit
    """), {"t": term, "min": min_similarity, "limit": limit})
    return [r[0] for r in result.all()]


async def entities_in_chunks(
    session: AsyncSession, chunk_ids: list[uuid.UUID]
) -> list[uuid.UUID]:
    if not chunk_ids:
        return []
    result = await session.execute(sql_text("""
        SELECT DISTINCT target_node_id FROM graphrag.graph_relationships
         WHERE predicate_label = 'viao:assertsAbout' AND source_node_type = 'chunk'
           AND target_node_type = 'entity'
           AND source_node_id = ANY(CAST(:c AS uuid[]))
    """), {"c": [str(c) for c in chunk_ids]})
    return [r[0] for r in result.all()]


async def resolve_class_closure(
    session: AsyncSession,
    terms: list[str],
    term_embeddings: list[list[float]],
    *,
    classes_per_term: int = 5,
    max_depth: int = 3,
) -> dict[uuid.UUID, int]:
    """Classes matching the question's class terms, plus their subclasses,
    as `class_id -> number of entities typed with it`.

    Shared by the two class-driven seeding paths so they see exactly the same
    set of classes: the chunk-mediated one (which caps class size, because
    500 undifferentiated members are noise) and the relation-filtered one
    (which does not, because the relation does the narrowing).
    """
    class_ids: set[uuid.UUID] = set()
    for term, vec in zip(terms, term_embeddings, strict=False):
        r = await session.execute(sql_text("""
            SELECT id FROM graphrag.ontology_classes
             WHERE embedding IS NOT NULL
             ORDER BY embedding <-> CAST(:v AS vector) LIMIT :k
        """), {"v": _vec_str(vec), "k": classes_per_term})
        class_ids |= {row[0] for row in r.all()}
        r = await session.execute(sql_text("""
            SELECT id FROM graphrag.ontology_classes
             WHERE label IS NOT NULL
               AND (lower(label) = lower(:t)
                    OR lower(replace(label, ' ', '')) = lower(replace(:t, ' ', '')))
        """), {"t": term})
        class_ids |= {row[0] for row in r.all()}
    if not class_ids:
        return {}
    r = await session.execute(sql_text("""
        WITH RECURSIVE down(id, depth) AS (
            SELECT id, 0 FROM graphrag.ontology_classes WHERE id = ANY(CAST(:ids AS uuid[]))
          UNION
            SELECT gr.source_node_id, down.depth + 1
              FROM down JOIN graphrag.graph_relationships gr
                ON gr.target_node_id = down.id
               AND gr.source_node_type = 'ontology_class'
               AND gr.target_node_type = 'ontology_class'
               AND gr.predicate_label = 'rdfs:subClassOf'
             WHERE down.depth < :depth
        )
        SELECT e.class_id, count(*) FROM graphrag.entities e
         WHERE e.class_id IN (SELECT id FROM down)
         GROUP BY e.class_id
    """), {"ids": [str(c) for c in class_ids], "depth": max_depth})
    return {cid: int(n) for cid, n in r.all()}


async def class_seed_entities(
    session: AsyncSession,
    terms: list[str],
    term_embeddings: list[list[float]],
    question_embedding: list[float],
    *,
    classes_per_term: int = 5,
    max_members: int = 50,
    top_chunks: int = 3,
) -> tuple[list[uuid.UUID], int]:
    """Seed entities from the classes closest to the question's class terms.

    Per term: the nearest classes by embedding plus an exact label match, and
    their subclasses. Classes with more than `max_members` entities are
    skipped -- "individual" would otherwise pick `Person` and reintroduce the
    hub this traversal avoids. The chunks mentioning the remaining classes'
    members are ranked against the QUESTION and the top few supply the seeds.
    Returns (entity ids, number of classes kept).
    """
    counts = await resolve_class_closure(
        session, terms, term_embeddings, classes_per_term=classes_per_term,
    )
    if not counts:
        return [], 0
    keep = [cid for cid, n in counts.items() if 1 <= n <= max_members]
    if not keep:
        return [], 0
    r = await session.execute(sql_text("""
        SELECT c.id FROM graphrag.chunks c
         WHERE c.embedding IS NOT NULL AND c.status = 'ACTIVE'
           AND c.id IN (
             SELECT gr.source_node_id FROM graphrag.graph_relationships gr
               JOIN graphrag.entities e ON e.id = gr.target_node_id
              WHERE gr.predicate_label = 'viao:assertsAbout'
                AND gr.source_node_type = 'chunk'
                AND e.class_id = ANY(CAST(:k AS uuid[])))
         ORDER BY c.embedding <-> CAST(:q AS vector)
         LIMIT :n
    """), {"k": [str(c) for c in keep], "q": _vec_str(question_embedding), "n": top_chunks})
    chunk_ids = [row[0] for row in r.all()]
    return await entities_in_chunks(session, chunk_ids), len(keep)


async def artifact_seed_entities(
    session: AsyncSession,
    question_embedding: list[float],
    *,
    top_artifacts: int = 5,
    top_chunks: int = 3,
) -> list[uuid.UUID]:
    """Seed entities from the artifacts closest to the question -- the
    entities those claims/findings/summaries are about. Falls back to the
    closest chunks' entities when no artifact is linked to any entity."""
    r = await session.execute(sql_text("""
        SELECT DISTINCT gr.target_node_id
          FROM (SELECT id FROM graphrag.intelligence_artifacts
                 WHERE embedding IS NOT NULL AND status = 'ACTIVE'
                 ORDER BY embedding <-> CAST(:q AS vector) LIMIT :n) a
          JOIN graphrag.graph_relationships gr
            ON gr.source_node_id = a.id
           AND gr.source_node_type = 'intelligence_artifact'
           AND gr.predicate_label = 'viao:assertsAbout'
           AND gr.target_node_type = 'entity'
    """), {"q": _vec_str(question_embedding), "n": top_artifacts})
    ids = [row[0] for row in r.all()]
    if ids:
        return ids
    r = await session.execute(sql_text("""
        SELECT id FROM graphrag.chunks
         WHERE embedding IS NOT NULL AND status = 'ACTIVE'
         ORDER BY embedding <-> CAST(:q AS vector) LIMIT :n
    """), {"q": _vec_str(question_embedding), "n": top_chunks})
    return await entities_in_chunks(session, [row[0] for row in r.all()])


async def vector_search_all_artifacts(
    session: AsyncSession,
    probe_embedding: list[float],
    *,
    top_k: int = 100,
) -> list[uuid.UUID]:
    """Global HNSW nearest-neighbor over ALL active artifacts (no id filter, so
    the vector index IS used -> fast even over the whole artifact layer). Used by
    artifact_only mode to bring every artifact type into scope, including
    Insights / Recommendations / Summaries that have no entity edge."""
    result = await session.execute(
        sql_text("""
        SELECT id FROM graphrag.intelligence_artifacts
         WHERE embedding IS NOT NULL AND status = 'ACTIVE'
         ORDER BY embedding <-> CAST(:probe AS vector)
         LIMIT :limit
        """),
        {"probe": _vec_str(probe_embedding), "limit": top_k},
    )
    return [row[0] for row in result.all()]


async def vector_search_documents(
    session: AsyncSession,
    probe_embedding: list[float],
    *,
    top_k: int = 10,
) -> list[uuid.UUID]:
    """Global HNSW nearest-neighbor over ALL active documents.

    Mirrors `vector_search_all_artifacts` -- unfiltered, so the
    `documents_embedding_idx` HNSW index is actually used.

    `documents.embedding` has been written at ingest and indexed since
    0001, but until now nothing read it. It gives retrieval a
    document-level recall path: a document that is ABOUT a topic
    surfaces even when no individual chunk matches the query's
    vocabulary, which is the same failure mode brand/generic names
    cause at chunk level.
    """
    if top_k <= 0:
        return []
    result = await session.execute(
        sql_text("""
        SELECT id FROM graphrag.documents
         WHERE embedding IS NOT NULL AND status = 'ACTIVE'
         ORDER BY embedding <-> CAST(:probe AS vector)
         LIMIT :limit
        """),
        {"probe": _vec_str(probe_embedding), "limit": top_k},
    )
    return [row[0] for row in result.all()]


async def fetch_chunks_for_documents(
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    *,
    limit: int = 300,
    per_document_limit: int = 50,
) -> list[tuple[uuid.UUID, float]]:
    """Chunks belonging to `document_ids`, best-first by document rank.

    Prefers `kind='fulltext'` chunks for documents that have them (the
    same preference the step-8.5 bridge applies), so citations land on
    verbatim text. Returns (chunk_id, score) where score decays with the
    document's position in `document_ids` -- so a chunk from the
    top-ranked document outranks one from the tenth.

    Two things here are load-bearing and were previously wrong:

    * The `LIMIT` had NO `ORDER BY`, so which rows survived truncation was
      whatever order the scan produced -- in practice clustered by
      document. A single document with >= `limit` chunks consumed the
      entire budget and the other nine contributed nothing.
    * The document-rank decay is applied in Python AFTER the query, so it
      could never rescue a document the SQL had already truncated away.

    `per_document_limit` bounds each document's share inside the query,
    where it can actually take effect.
    """
    if not document_ids:
        return []
    rank_of = {did: i for i, did in enumerate(document_ids)}
    result = await session.execute(
        sql_text("""
        WITH eligible AS (
            SELECT c.id, c.document_id, c.chunk_index,
                   row_number() OVER (
                       PARTITION BY c.document_id ORDER BY c.chunk_index
                   ) AS rn
              FROM graphrag.chunks c
             WHERE c.document_id = ANY(CAST(:ids AS uuid[]))
               AND c.status = 'ACTIVE'
               AND c.embedding IS NOT NULL
               AND (c.kind = 'fulltext' OR NOT EXISTS (
                     SELECT 1 FROM graphrag.chunks f
                      WHERE f.document_id = c.document_id
                        AND f.kind = 'fulltext'
                        AND f.status = 'ACTIVE'))
        )
        SELECT id, document_id
          FROM eligible
         WHERE rn <= :per_doc
         -- Order by the caller's document ranking BEFORE truncating, so
         -- the limit trims the least-relevant documents rather than an
         -- arbitrary slice.
         ORDER BY array_position(CAST(:ids AS uuid[]), document_id), chunk_index
         LIMIT :limit
        """),
        {
            "ids": [str(d) for d in document_ids],
            "limit": limit,
            "per_doc": per_document_limit,
        },
    )
    out: list[tuple[uuid.UUID, float]] = []
    for cid, did in result.all():
        out.append((cid, 1.0 / (1.0 + rank_of.get(did, len(document_ids)))))
    out.sort(key=lambda t: -t[1])
    return out


async def fetch_chunk_entity_names(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
    *,
    per_chunk_limit: int = 6,
) -> dict[uuid.UUID, list[str]]:
    """Entity names each chunk assertsAbout, for evidence attribution.

    Run over the FINAL top-k only (~30 rows), not the candidate pool, so
    it is one cheap query. Lets the synthesis prompt label each passage
    with the entity it actually concerns -- the guard against reporting
    one drug's dose under another drug's name.
    """
    if not chunk_ids:
        return {}
    result = await session.execute(
        sql_text("""
        SELECT gr.source_chunk_id, e.name
          FROM graphrag.graph_relationships gr
          JOIN graphrag.entities e ON e.id = gr.target_node_id
         WHERE gr.predicate_label = 'viao:assertsAbout'
           AND gr.target_node_type = 'entity'
           AND gr.source_chunk_id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": [str(cid) for cid in chunk_ids]},
    )
    out: dict[uuid.UUID, list[str]] = {}
    for cid, name in result.all():
        if not name:
            continue
        names = out.setdefault(cid, [])
        if name not in names and len(names) < per_chunk_limit:
            names.append(name)
    return out


# Trigram floor for the misspelling fallback below. Measured against this
# corpus's drug vocabulary:
#     typos           similarity 0.50 - 0.77  (ozempick/ozempic = 0.700)
#     different drugs similarity 0.00 - 0.14  (losartan/lisinopril = 0.053)
# 0.45 sits clear of both, so a typo resolves while two genuinely different
# drugs never collapse into one another -- which would reintroduce the exact
# wrong-entity failure this whole feature exists to prevent.
_ALIAS_FUZZY_THRESHOLD = 0.45

_ALIAS_EXACT_SQL = """
SELECT surface, occurrences FROM (
    SELECT surface_b AS surface, occurrences
      FROM graphrag.term_aliases WHERE term_a = :t
    UNION ALL
    SELECT surface_a AS surface, occurrences
      FROM graphrag.term_aliases WHERE term_b = :t
) s
 -- A synonym USED AS A PROBE has to be a name, not a phrase. Longer runs
 -- are label boilerplate that survived mining ("Medication Guide TRULICITY
 -- TRU-li-si-tee"); injecting one into probe text would poison the
 -- embedding it rides on.
 --
 -- The ceiling is 6, not 3, because acronym EXPANSIONS are legitimately
 -- longer ("maximum recommended human dose", "Non-Steroidal Anti
 -- Inflammatory Drugs") and a 3-word cap silently dropped them -- which is
 -- half of why MRHD never fired in the 2026-08-19 audit. `_expand_aliases`
 -- tightens back to 3 for non-acronym pairs, where it has the per-term
 -- context to tell the difference.
 WHERE array_length(string_to_array(btrim(surface), ' '), 1) <= 6
 ORDER BY occurrences DESC, length(surface) ASC, surface
 LIMIT :limit
"""

# Fallback for misspelled query terms. Every other stage of the pipeline
# tolerates typos -- question_parse and query_decompose are LLM calls,
# entity seeding is already trigram-matched, and embeddings degrade
# gracefully -- so an exact-match alias lookup was the single brittle link
# in the chain: one wrong character silently disabled synonym expansion
# with no signal that it had happened.
_ALIAS_FUZZY_SQL = """
SELECT surface, occurrences, sim FROM (
    SELECT surface_b AS surface, occurrences, similarity(term_a, :t) AS sim
      FROM graphrag.term_aliases WHERE similarity(term_a, :t) >= :thr
    UNION ALL
    SELECT surface_a AS surface, occurrences, similarity(term_b, :t) AS sim
      FROM graphrag.term_aliases WHERE similarity(term_b, :t) >= :thr
) s
 WHERE array_length(string_to_array(btrim(surface), ' '), 1) <= 3
 ORDER BY sim DESC, occurrences DESC, length(surface) ASC, surface
 LIMIT :limit
"""


async def fetch_all_alias_terms(
    session: AsyncSession, *, limit: int = 5000
) -> list[tuple[str, str, str, str]]:
    """Every mined pair as (term_a, term_b, surface_a, surface_b).

    Small by construction -- it is corpus VOCABULARY, not corpus size
    (a few hundred rows), so pulling the lot and matching in Python is
    cheaper than a query per candidate phrase.

    Supports the literal-question scan in `_expand_aliases_from_question`,
    which exists because the question parser reduces multi-word technical
    phrases to a head noun: "maximum recommended human dose" comes back as
    just "dose", so a pair that IS in the table never gets looked up.
    """
    result = await session.execute(
        sql_text("""
        SELECT term_a, term_b, surface_a, surface_b
          FROM graphrag.term_aliases
         ORDER BY occurrences DESC
         LIMIT :limit
        """),
        {"limit": limit},
    )
    return [(a, b, sa, sb) for a, b, sa, sb in result.all()]


def _render_time_span(
    labels_by_date: list[tuple[str, Any]], per_chunk_limit: int
) -> str:
    """Render a chunk's periods as a bounded, DIRECTIONALLY NEUTRAL string.

    Listing the first N periods in date order is a trap. A chunk tagged
    2011,2013,2014,2019,2020,2022,2023,2024 rendered as
    "2011, 2013, 2014, 2019" -- the recent end truncated away -- which
    made every chunk look in-window for a backward bound ("before 2020")
    and out-of-window for a forward one ("after 2023"). That is exactly
    the asymmetry reported on 2026-08-21: forward-bounded questions
    returned confident false denials while their backward twins passed.

    So when there are more periods than fit, show BOTH endpoints and the
    count. Neither direction is starved, and nothing is silently dropped.
    """
    if not labels_by_date:
        return ""
    if len(labels_by_date) <= per_chunk_limit:
        return ", ".join(label for label, _ in labels_by_date)
    earliest = labels_by_date[0][0]
    latest = labels_by_date[-1][0]
    return f"{earliest} ... {latest} ({len(labels_by_date)} periods)"


async def fetch_chunk_time_labels(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
    *,
    per_chunk_limit: int = 4,
) -> dict[uuid.UUID, str]:
    """Time periods each chunk is linked to, for date-scoped questions.

    Mirrors `fetch_chunk_entity_names`: one query over the FINAL top-k
    only, walking `time:hasTime` edges to `time_instances`.

    Without this the synthesis cannot honour a date constraint even when
    told to -- the evidence block carries no per-item date, so "by 2025"
    was answered with a 2026 approval.

    Returns a rendered string per chunk rather than a list, because the
    rendering has to see ALL of a chunk's periods to place both endpoints
    (see `_render_time_span`). Ordered by date, oldest first, so the span
    reads naturally.
    """
    if not chunk_ids:
        return {}
    result = await session.execute(
        sql_text("""
        SELECT gr.source_chunk_id, t.display_label, t.start_date
          FROM graphrag.graph_relationships gr
          JOIN graphrag.time_instances t ON t.id = gr.target_node_id
         WHERE gr.predicate_label = 'time:hasTime'
           AND gr.target_node_type = 'time_instance'
           AND gr.source_chunk_id = ANY(CAST(:ids AS uuid[]))
         ORDER BY gr.source_chunk_id, t.start_date
        """),
        {"ids": [str(cid) for cid in chunk_ids]},
    )
    collected: dict[uuid.UUID, list[tuple[str, Any]]] = {}
    for cid, label, start in result.all():
        if not label:
            continue
        bucket = collected.setdefault(cid, [])
        if all(label != seen for seen, _ in bucket):
            bucket.append((label, start))
    return {
        cid: rendered
        for cid, rows in collected.items()
        if (rendered := _render_time_span(rows, per_chunk_limit))
    }


async def fetch_aliases(
    session: AsyncSession,
    normalized_terms: list[str],
    *,
    per_term_limit: int = 6,
) -> dict[str, list[str]]:
    """Corpus-mined synonyms for each normalized query term.

    Pure indexed lookup against `term_aliases` -- no scan, no LLM. Pairs
    are stored with `term_a < term_b`, so one row answers a lookup from
    either side; hence the UNION.

    Returns {normalized_term: [surface form, ...]}, best-attested first.
    A term with no mined alias is simply absent, and retrieval then
    behaves exactly as it did before this feature existed.

    Ties on `occurrences` break toward the SHORTER surface form: clinical
    tables produce both "Mounjaro" and header runs like "Dose NDC
    Mounjaro" at identical counts, and the bare name is the one worth
    embedding.
    """
    if not normalized_terms:
        return {}
    out: dict[str, list[str]] = {}
    for term in normalized_terms:
        if not term:
            continue
        result = await session.execute(
            sql_text(_ALIAS_EXACT_SQL), {"t": term, "limit": per_term_limit}
        )
        rows = result.all()
        if not rows:
            # Exact miss: the term may simply be misspelled. The table is
            # small (corpus vocabulary, not corpus size), so the seq scan
            # this costs is negligible and only runs on the miss path.
            result = await session.execute(
                sql_text(_ALIAS_FUZZY_SQL),
                {
                    "t": term,
                    "limit": per_term_limit,
                    "thr": _ALIAS_FUZZY_THRESHOLD,
                },
            )
            rows = [(r[0], r[1]) for r in result.all()]
        seen: set[str] = set()
        surfaces: list[str] = []
        for surface, _occ in rows:
            key = (surface or "").strip().lower()
            if not key or key == term or key in seen:
                continue
            seen.add(key)
            surfaces.append(surface.strip())
        if surfaces:
            out[term] = surfaces
    return out


async def same_type_neighbor_edges(
    session: AsyncSession,
    candidate_ids: list[uuid.UUID],
    artifact_type: str,
    *,
    threshold: float,
    max_neighbors: int = 25,
) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """All same-type similarity edges among `candidate_ids`, within L2 distance
    `threshold`. Used by the artifact-rollup clustering stage
    (db_artifact_rollup) to build a graph for union-find grouping.

    Returns (source_id, neighbor_id) pairs where source is one of `candidate_ids`
    and neighbor is its nearest-<=threshold same-type artifact (the caller drops
    neighbors outside the candidate set). Done in ONE batched query -- a LATERAL
    top-k over each candidate whose inner ORDER BY uses the HNSW index -- rather
    than one round-trip per candidate, which does not scale on a latency-bound
    pooled Postgres (thousands of sequential round-trips time out)."""
    if not candidate_ids:
        return []
    result = await session.execute(
        sql_text("""
        SELECT a.id, n.id
          FROM graphrag.intelligence_artifacts a
         CROSS JOIN LATERAL (
              SELECT x.id, (x.embedding <-> a.embedding) AS dist
                FROM graphrag.intelligence_artifacts x
               WHERE x.status = 'ACTIVE'
                 AND x.artifact_type = :atype
                 AND x.embedding IS NOT NULL
                 AND x.id <> a.id
               ORDER BY x.embedding <-> a.embedding
               LIMIT :k
         ) n
         WHERE a.id = ANY(CAST(:cand AS uuid[]))
           AND a.embedding IS NOT NULL
           AND n.dist <= :threshold
        """),
        {
            "cand": [str(i) for i in candidate_ids],
            "atype": artifact_type,
            "k": max_neighbors,
            "threshold": threshold,
        },
    )
    return [(a, b) for a, b in result.all()]


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


# ---------------------------------------------------------------------------
# Relationship-aware traversal (0008 `graph_relationships.embedding`)
# ---------------------------------------------------------------------------
#
# The plain BFS treats every edge out of a node as equally relevant: a seed's
# neighbourhood at 3 hops is whatever it happens to touch. That is right when
# the question names its subjects and asks for context, and wrong when the
# question names a RELATION ("who was accused of fraud", "which company is in
# Germany") -- there the relation is the whole constraint and the graph has the
# answer, just diluted by every unrelated edge on the same nodes.
#
# These helpers score edges by embedding similarity to the relation phrases
# parsed out of the question, so a walk follows only the edges that MEAN what
# was asked. Edges with no vector (ontology TBox rows, graphs built before
# 0008) are invisible here; callers fall back to `bfs_expand`.

_REL_HOP_SQL = sql_text("""
SELECT g.id, g.source_node_id, g.target_node_id,
       1 - (g.embedding <=> CAST(:probe AS vector)) AS sim
  FROM graphrag.graph_relationships g
 WHERE g.embedding IS NOT NULL
   AND g.source_node_type = 'entity'
   AND g.target_node_type = 'entity'
   AND (g.source_node_id = ANY(CAST(:frontier AS uuid[]))
     OR g.target_node_id = ANY(CAST(:frontier AS uuid[])))
   AND 1 - (g.embedding <=> CAST(:probe AS vector)) >= CAST(:threshold AS float)
 ORDER BY sim DESC
 LIMIT :limit
""")

_REL_GLOBAL_SQL = sql_text("""
SELECT g.id, g.source_node_id, g.target_node_id,
       1 - (g.embedding <=> CAST(:probe AS vector)) AS sim
  FROM graphrag.graph_relationships g
 WHERE g.embedding IS NOT NULL
   AND g.source_node_type = 'entity'
   AND g.target_node_type = 'entity'
   AND 1 - (g.embedding <=> CAST(:probe AS vector)) >= CAST(:threshold AS float)
 ORDER BY sim DESC
 LIMIT :limit
""")


async def relationship_hop(
    session: AsyncSession,
    frontier: list[uuid.UUID],
    probes: list[list[float]],
    *,
    threshold: float = 0.55,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Edges touching `frontier` that match ANY probe relation, best first.

    Scoring against the MAX over probes (rather than one vector) is what lets
    a chain whose rungs differ work: "who founded the company that acquired X"
    parses to two relations, hop 1 matches on `acquired` and hop 2 on
    `founded`, and neither hop needs to know which rung it is on. It is
    order-free, so a question that states the chain backwards behaves the same.
    """
    if not frontier or not probes:
        return []
    best: dict[uuid.UUID, dict[str, Any]] = {}
    fr = [str(x) for x in frontier]
    for probe in probes:
        rows = await session.execute(
            _REL_HOP_SQL,
            {"probe": _vec_str(probe), "frontier": fr,
             "threshold": threshold, "limit": limit},
        )
        for rid, src, tgt, sim in rows.all():
            cur = best.get(rid)
            if cur is None or float(sim) > cur["sim"]:
                best[rid] = {"id": rid, "source_node_id": src,
                             "target_node_id": tgt, "sim": float(sim)}
    return sorted(best.values(), key=lambda r: r["sim"], reverse=True)[:limit]


async def relationship_vector_search(
    session: AsyncSession,
    probes: list[list[float]],
    *,
    threshold: float = 0.55,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """The same match with no anchor: the corpus's best edges for a relation.

    This is the path for a question that names a relation but no entity the
    graph knows ("which executives were charged?"). Measured on the 182
    MultiHop-RAG questions, the unanchored search covered 58% of questions at
    threshold 0.55 where the anchored hop fired on only 10%.
    """
    if not probes:
        return []
    best: dict[uuid.UUID, dict[str, Any]] = {}
    for probe in probes:
        rows = await session.execute(
            _REL_GLOBAL_SQL,
            {"probe": _vec_str(probe), "threshold": threshold, "limit": limit},
        )
        for rid, src, tgt, sim in rows.all():
            cur = best.get(rid)
            if cur is None or float(sim) > cur["sim"]:
                best[rid] = {"id": rid, "source_node_id": src,
                             "target_node_id": tgt, "sim": float(sim)}
    return sorted(best.values(), key=lambda r: r["sim"], reverse=True)[:limit]


_CLASS_BY_RELATION_SQL = sql_text("""
SELECT e.id, max(1 - (g.embedding <=> CAST(:probe AS vector))) AS sim
  FROM graphrag.graph_relationships g
  JOIN graphrag.entities e
    ON (e.id = g.source_node_id OR e.id = g.target_node_id)
 WHERE g.embedding IS NOT NULL
   AND g.source_node_type = 'entity'
   AND g.target_node_type = 'entity'
   AND e.class_id = ANY(CAST(:class_ids AS uuid[]))
   AND e.status = 'ACTIVE'
   AND 1 - (g.embedding <=> CAST(:probe AS vector)) >= CAST(:threshold AS float)
 GROUP BY e.id
 ORDER BY sim DESC
 LIMIT :limit
""")


async def entities_of_class_by_relation(
    session: AsyncSession,
    class_ids: list[uuid.UUID],
    probes: list[list[float]],
    *,
    threshold: float = 0.55,
    limit: int = 25,
) -> list[tuple[uuid.UUID, float]]:
    """Members of `class_ids` ranked by how well their edges match a relation.

    The vague-entity path: "who was accused of fraud" names no entity, so
    seeding can only fall back to whichever chunks the question's wording
    happens to hit. Here the CLASS supplies the type constraint (Person) and
    the relation phrase supplies the rest -- scored against the stored
    evidence quote, which is where "accused of defrauding customers" actually
    appears. It is also what makes a big class usable: the member cap that
    normally drops Person/Organization from class seeding exists because 500
    undifferentiated members are noise, and a relation filter differentiates
    them.
    """
    if not class_ids or not probes:
        return []
    best: dict[uuid.UUID, float] = {}
    cids = [str(c) for c in class_ids]
    for probe in probes:
        rows = await session.execute(
            _CLASS_BY_RELATION_SQL,
            {"probe": _vec_str(probe), "class_ids": cids,
             "threshold": threshold, "limit": limit},
        )
        for eid, sim in rows.all():
            if float(sim) > best.get(eid, 0.0):
                best[eid] = float(sim)
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Temporal expansion
# ---------------------------------------------------------------------------

_TIME_DESCEND_SQL = sql_text("""
WITH RECURSIVE down(id, depth) AS (
    SELECT CAST(x AS uuid), 0
      FROM unnest(CAST(:seed_ids AS uuid[])) AS x
  UNION ALL
    SELECT gr.source_node_id, d.depth + 1
      FROM down d
      JOIN graphrag.graph_relationships gr
        ON gr.target_node_id = d.id
       AND gr.target_node_type = 'time_instance'
       AND gr.source_node_type = 'time_instance'
       AND gr.predicate_label = 'time:intervalDuring'
     WHERE d.depth < CAST(:max_depth AS int)
) CYCLE id SET is_cycle USING path
SELECT id, min(depth) FROM down WHERE NOT is_cycle GROUP BY id
""")

_TIME_ASCEND_SQL = sql_text("""
WITH RECURSIVE up(id, depth) AS (
    SELECT CAST(x AS uuid), 0
      FROM unnest(CAST(:seed_ids AS uuid[])) AS x
  UNION ALL
    SELECT gr.target_node_id, u.depth + 1
      FROM up u
      JOIN graphrag.graph_relationships gr
        ON gr.source_node_id = u.id
       AND gr.source_node_type = 'time_instance'
       AND gr.target_node_type = 'time_instance'
       AND gr.predicate_label = 'time:intervalDuring'
     WHERE u.depth < CAST(:max_depth AS int)
) CYCLE id SET is_cycle USING path
SELECT id, min(depth) FROM up WHERE NOT is_cycle GROUP BY id
""")


async def expand_time_instances(
    session: AsyncSession,
    seed_ids: list[uuid.UUID],
    *,
    max_depth: int = 3,
    include_ancestors: bool = True,
) -> dict[uuid.UUID, float]:
    """Periods implied by the ones a question named, as `id -> score`.

    Direction matters, which is why this is not part of the generic BFS.

      down  a question about 2023 is about every month and day inside 2023,
            so descendants come in at full weight.
      up    a question about October 2023 is also, weakly, about a chunk that
            only says "2023" -- the year contains the month. Ancestors come in
            at half weight and are NOT re-descended, or "October" would drag
            in all eleven other months.
    """
    if not seed_ids:
        return {}
    sids = [str(s) for s in seed_ids]
    out: dict[uuid.UUID, float] = {}
    rows = await session.execute(
        _TIME_DESCEND_SQL, {"seed_ids": sids, "max_depth": max_depth}
    )
    for tid, depth in rows.all():
        out[tid] = max(out.get(tid, 0.0), 0.8 ** int(depth))
    if include_ancestors:
        rows = await session.execute(
            _TIME_ASCEND_SQL, {"seed_ids": sids, "max_depth": max_depth}
        )
        for tid, depth in rows.all():
            if int(depth) == 0:
                continue
            out[tid] = max(out.get(tid, 0.0), 0.5 * (0.8 ** int(depth)))
    return out


async def time_instances_by_identifier(
    session: AsyncSession, identifiers: list[str],
) -> list[tuple[uuid.UUID, str]]:
    """Look up minted periods by canonical identifier (YEAR_2023, Q3_2023,
    MONTH_2023_10, DAY_2023_10_15)."""
    if not identifiers:
        return []
    rows = await session.execute(
        sql_text(
            "SELECT id, display_label FROM graphrag.time_instances "
            "WHERE time_identifier = ANY(CAST(:idents AS text[]))"
        ),
        {"idents": list(identifiers)},
    )
    return [(r[0], r[1]) for r in rows.all()]


async def entities_without_relationships(
    session: AsyncSession, entity_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Of `entity_ids`, those on no entity -> entity edge at all.

    A seed like this contributes nothing to any graph walk: it is a node with
    no arcs, so hops 1..N reach exactly itself. Retrieval uses this to decide
    whether the document arm is needed -- for these seeds the graph genuinely
    has nothing to give, and a document-level vector match is the only way to
    recover their context.
    """
    if not entity_ids:
        return []
    r = await session.execute(sql_text("""
        SELECT x FROM unnest(CAST(:ids AS uuid[])) AS x
         WHERE NOT EXISTS (
            SELECT 1 FROM graphrag.graph_relationships g
             WHERE g.source_node_type = 'entity'
               AND g.target_node_type = 'entity'
               AND (g.source_node_id = x OR g.target_node_id = x))
    """), {"ids": [str(e) for e in entity_ids]})
    return [row[0] for row in r.all()]


_ARTIFACT_CHUNKS_SQL = sql_text("""
SELECT asrc.artifact_id, asrc.chunk_id
  FROM graphrag.artifact_sources asrc
  JOIN graphrag.chunks c ON c.id = asrc.chunk_id
 WHERE asrc.artifact_id = ANY(CAST(:ids AS uuid[]))
   AND c.status = 'ACTIVE'
   AND c.embedding IS NOT NULL
 LIMIT :limit
""")


async def fetch_chunks_for_artifacts(
    session: AsyncSession,
    artifact_ids: list[uuid.UUID],
    *,
    limit: int = 300,
) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """(artifact_id, chunk_id) for the chunks each artifact was derived from.

    The reverse of `fetch_table_artifacts_for_chunks`, and the half of the
    Milestone-H traceability chain (answer -> artifact -> chunk -> document)
    that was written at ingestion but never read at query time:
    `artifact_sources` holds a row for 1,277 of 1,278 artifacts on the
    websearch-geo-time build, and nothing in retrieval touched it.

    An artifact is a distillation of its chunk, so bringing the chunk back is
    not free -- it spends a chunk slot on text the artifact already condensed.
    Whether that is worth it is what `qa.artifact_chunk_bridge` selects
    between.
    """
    if not artifact_ids:
        return []
    rows = await session.execute(
        _ARTIFACT_CHUNKS_SQL,
        {"ids": [str(a) for a in artifact_ids], "limit": limit},
    )
    return [(r[0], r[1]) for r in rows.all()]

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

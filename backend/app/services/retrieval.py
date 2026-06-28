"""Milestone F: retrieval pipeline + answer synthesis.

Driver function `retrieve_and_answer(question, mode, ...)` runs the
12-step pipeline as designed in the plan:

  1. (G only) follow-up resolution
  2. mode selected by caller
  3. question_parse -> entities/classes/time/intent
  4. ontology match (vector + trgm)
  5. concept expansion
  6. seed nodes union
  7. graph BFS (depth `hops`)
  8. candidate retrieval (chunks + artifacts)
  9. multi-probe vector rerank
     a. query_decompose -> sub-questions
     b. embed all probes
     c. per-probe SQL rerank against the step-8 candidate set
     d. per-candidate, per-probe scores
 10. RRF fusion of probe rankings + graph signals
 11. context engineering: pack top-K
 12. answer generation (mode-specific prompt)

Persists `retrieval_runs` (1 row) + `retrieval_evidence` (top-K rows).

Two modes (2026-06-13 redesign):
  - simple_qa     -- tight 1-3 sentence direct answer; top_k=20.
  - deep_research -- structured 7-section output (SPECIFICS / ANALYSIS
                     / ANSWER / CONTRADICTIONS / KEY CLAIMS /
                     COVERAGE IMBALANCE / KEY INSIGHTS); top_k=30;
                     default mode.

The summarize/insights/knowledge_gaps/exhaustive_search modes were
removed in the redesign in favor of the structured deep_research
output. The two remaining modes share the same pipeline up through
step 10; they diverge at steps 11 + 12 only on prompt choice + model.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.db.graph_version import current_version
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.entities import Entity, TimeInstance
from backend.app.db.models.ontology import OntologyClass
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS
from backend.app.services import retrieval_sql
from backend.app.services.retrieval_ranking import rrf_fuse
from backend.app.services.retrieval_sql import _vec_str


_DEFAULT_HOPS_FALLBACK = 3


def default_hops() -> int:
    """Graph-BFS depth (step 7) default, sourced from config.yaml's
    `qa.hops`. Falls back to 3 if the key is missing or config can't be
    read. This is the single source of truth -- the CLI and the
    conversation path both pass `hops=None` so they inherit it."""
    try:
        value = get_settings().app_config.get("qa", {}).get("hops")
        return int(value) if value is not None else _DEFAULT_HOPS_FALLBACK
    except Exception:
        return _DEFAULT_HOPS_FALLBACK


_VALID_MODES = ("simple_qa", "deep_research")
# deep_research is the default workhorse mode; simple_qa is the tight
# direct-answer mode. Other modes (summarize, insights, knowledge_gaps,
# exhaustive_search) were removed 2026-06-13 in favor of the structured
# deep_research output. See plan: "QA + artifact redesign: 2-mode ...".


@dataclass
class RetrievalResult:
    answer: str | None
    mode: str
    resolved_query: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieval_run_id: uuid.UUID | None = None
    parsed: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    graph_version: int = 0


async def retrieve_and_answer(
    question: str,
    *,
    mode: str = "deep_research",
    top_k: int | None = None,
    hops: int | None = None,
    max_cost_usd: float = 1.0,
    decompose: bool = True,
    max_probes: int = 5,
    conversation_turn_id: uuid.UUID | None = None,
    resolved_query: str | None = None,
    verbose: bool = False,
) -> RetrievalResult:
    """Driver. `resolved_query` skips the follow-up resolution step
    (the caller -- typically Milestone G -- already resolved it).

    `top_k` defaults to 30 for deep_research (more breadth for the
    seven-section output) and 20 for simple_qa.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"invalid mode: {mode}; must be one of {_VALID_MODES}. "
            "The summarize/insights/knowledge_gaps/exhaustive_search "
            "modes were removed -- use deep_research for structured "
            "answers or simple_qa for direct factoids."
        )
    if top_k is None:
        top_k = 30 if mode == "deep_research" else 20
    if hops is None:
        hops = default_hops()
    t0 = time.time()
    router = LLMRouter()
    cost_before = router.total_cost_usd

    resolved_query = resolved_query or question

    # -------- step 3: parse the question --------
    parsed = await _question_parse(router, resolved_query)
    if verbose:
        print(f"[query] parsed: {parsed}")

    # -------- step 4: ontology match --------
    embedder = Embedder()
    qvec = (await embedder.embed([resolved_query]))[0]
    seeds, matched_classes, matched_entities, matched_times = await _ontology_match(
        parsed, qvec
    )
    if verbose:
        print(
            f"[query] matched: classes={len(matched_classes)} "
            f"entities={len(matched_entities)} times={len(matched_times)} "
            f"seeds={len(seeds)}"
        )

    # -------- step 5: concept expansion --------
    expanded_class_ids = await _concept_expansion(
        router, resolved_query, matched_classes
    )
    seeds.extend((cid, "ontology_class") for cid in expanded_class_ids)
    seeds = list(set(seeds))   # dedupe

    # -------- step 6+7: BFS --------
    async with session_scope() as session:
        bfs_nodes = await retrieval_sql.bfs_expand(
            session, seeds, max_hops=hops, decay=0.7
        )
    expanded_entity_ids = [
        nid for (nid, ntype), _ in bfs_nodes.items() if ntype == "entity"
    ]
    # Seeds may include entities already; merge them in.
    for nid, ntype in seeds:
        if ntype == "entity" and nid not in expanded_entity_ids:
            expanded_entity_ids.append(nid)
    expanded_time_ids = [
        nid for (nid, ntype), _ in bfs_nodes.items() if ntype == "time_instance"
    ]
    for nid, ntype in seeds:
        if ntype == "time_instance" and nid not in expanded_time_ids:
            expanded_time_ids.append(nid)
    if verbose:
        print(
            f"[query] BFS yielded {len(bfs_nodes)} nodes "
            f"({len(expanded_entity_ids)} entities, "
            f"{len(expanded_time_ids)} time_instances)"
        )

    # -------- step 8: candidate retrieval --------
    async with session_scope() as session:
        ent_chunks = await retrieval_sql.fetch_candidate_chunks_for_entities(
            session, expanded_entity_ids, limit=500,
        )
        time_chunks = await retrieval_sql.fetch_candidate_chunks_for_time_instances(
            session, expanded_time_ids, limit=200,
        )
        artifact_candidates = await retrieval_sql.fetch_candidate_artifacts_for_entities(
            session, expanded_entity_ids, limit=200,
        )

    candidate_chunk_ids = list({cid for cid, _ in ent_chunks + time_chunks})
    candidate_artifact_ids = [aid for aid, _ in artifact_candidates]

    # Document-mediated table inclusion: pull every StructuredTable
    # artifact derived from a document that owns at least one of our
    # candidate chunks. Without this step, revenue / financial-data
    # tables that don't repeat the company name inside their cells get
    # missed -- the entity-anchored artifact path only catches tables
    # with direct entity edges (~10-15% of tables in practice). With
    # this step, asking "BHP's 2025 revenue" pulls every table from the
    # BHP 10-K into the candidate pool, where vector rerank then
    # surfaces the actually-relevant revenue table by similarity to the
    # query probes. Bounded at 200 tables to keep Stage-9 vector-rerank
    # cheap; the limit only bites for queries that match very many
    # documents simultaneously.
    if candidate_chunk_ids:
        async with session_scope() as session:
            doc_table_candidates = (
                await retrieval_sql.fetch_table_artifacts_for_chunks(
                    session, candidate_chunk_ids, limit=200,
                )
            )
        # Avoid duplicates if a table was already in candidate_artifact_ids
        # via a direct entity edge.
        already_in = set(candidate_artifact_ids)
        for aid, _ in doc_table_candidates:
            if aid not in already_in:
                candidate_artifact_ids.append(aid)
                already_in.add(aid)
        if verbose:
            n_added = len(candidate_artifact_ids) - len(artifact_candidates)
            print(
                f"[query] document-mediated tables added: {n_added} "
                f"StructuredTable artifact(s) (now {len(candidate_artifact_ids)} "
                f"total artifact candidates)"
            )

    # -------- step 8.5: full-text bridge --------
    # When documents owning our candidate (summary) chunks also carry verbatim
    # full-text chunks (ingested with --full-text-chunks), swap them into the
    # candidate pool so retrieval runs over verbatim text (better recall + exact
    # citations). The graph entered via summary-chunk entity edges; here we exit
    # into full text, document-mediated (mirrors the table bridge above). The
    # graph-coverage signal is propagated to the fulltext chunks by document.
    # No-op when no full-text chunks exist → identical to prior behavior.
    graph_chunk_ranking: list[uuid.UUID] = [cid for cid, _ in ent_chunks]
    if candidate_chunk_ids:
        ft_rows: list[tuple[uuid.UUID, float, uuid.UUID]] = []
        chunk_doc_map: dict[uuid.UUID, uuid.UUID] = {}
        async with session_scope() as session:
            ft_rows = await retrieval_sql.fetch_fulltext_chunks_for_chunks(
                session, candidate_chunk_ids, limit=500,
            )
            if ft_rows:
                chunk_doc_map = await retrieval_sql.fetch_chunk_document_ids(
                    session, candidate_chunk_ids,
                )
        if ft_rows:
            ft_doc_ids = {did for _, _, did in ft_rows}
            # Keep summary chunks only for docs that have NO full-text chunks.
            summary_keep = [
                cid for cid in candidate_chunk_ids
                if chunk_doc_map.get(cid) not in ft_doc_ids
            ]
            ft_ids = [cid for cid, _, _ in ft_rows]
            candidate_chunk_ids = summary_keep + ft_ids
            # Propagate graph scores: fulltext chunks inherit their document's
            # candidate-chunk count; kept summary chunks keep their entity score.
            ent_score = {cid: sc for cid, sc in ent_chunks}
            scored = [(cid, ent_score.get(cid, 0.0)) for cid in summary_keep]
            scored += [(cid, hits) for cid, hits, _ in ft_rows]
            scored.sort(key=lambda t: t[1], reverse=True)
            graph_chunk_ranking = [cid for cid, _ in scored]
            if verbose:
                print(
                    f"[query] full-text bridge: {len(ft_ids)} fulltext chunk(s) "
                    f"across {len(ft_doc_ids)} doc(s); candidate pool now "
                    f"{len(candidate_chunk_ids)}"
                )

    if not candidate_chunk_ids and not candidate_artifact_ids:
        if verbose:
            print("[query] zero candidates from graph; falling back to global vector search")
        # Pure vector fallback. Helps when the query has no entity/class
        # match (e.g. abstract questions about the corpus). Prefer full-text
        # chunks globally when any exist (verbatim citations); else summary.
        async with session_scope() as session:
            r = await session.execute(
                sql_text("""
                SELECT id FROM graphrag.chunks
                 WHERE embedding IS NOT NULL AND status='ACTIVE'
                   AND (kind = 'fulltext' OR NOT EXISTS (
                         SELECT 1 FROM graphrag.chunks
                          WHERE kind = 'fulltext' AND status = 'ACTIVE'))
                 ORDER BY embedding <-> CAST(:probe AS vector)
                 LIMIT :limit
                """),
                {"probe": _vec_str(qvec), "limit": top_k * 2},
            )
            candidate_chunk_ids = [row[0] for row in r.all()]

    if verbose:
        print(
            f"[query] candidates: chunks={len(candidate_chunk_ids)} "
            f"artifacts={len(candidate_artifact_ids)}"
        )

    # -------- step 9: multi-probe vector rerank --------
    probes = [resolved_query]
    if decompose:
        sub_qs = await _query_decompose(router, resolved_query)
        # Keep up to `max_probes` total. Always include original.
        for sq in sub_qs:
            if len(probes) >= max_probes:
                break
            if sq.strip() and sq.strip() != resolved_query.strip():
                probes.append(sq)
    if verbose:
        print(f"[query] probes ({len(probes)}): {probes}")

    probe_vecs = await embedder.embed(probes)

    chunk_rankings: list[list[uuid.UUID]] = []
    artifact_rankings: list[list[uuid.UUID]] = []
    async with session_scope() as session:
        for pvec in probe_vecs:
            ranked = await retrieval_sql.vector_rerank_chunks(
                session, candidate_chunk_ids, pvec, top_k=top_k * 3,
            )
            chunk_rankings.append([cid for cid, _ in ranked])
            if candidate_artifact_ids:
                ranked_a = await retrieval_sql.vector_rerank_artifacts(
                    session, candidate_artifact_ids, pvec, top_k=top_k,
                )
                artifact_rankings.append([aid for aid, _ in ranked_a])

    # Graph-distance ranking: BFS score per chunk (via its entities). Uses the
    # entity-coverage ranking from step 8, propagated to full-text chunks by the
    # step-8.5 bridge when present (else identical to `ent_chunks`).
    chunk_rankings.append(graph_chunk_ranking)
    if artifact_candidates:
        artifact_rankings.append([aid for aid, _ in artifact_candidates])

    # -------- step 10: RRF fusion --------
    fused_chunks = rrf_fuse(chunk_rankings)[:top_k]
    fused_artifacts = rrf_fuse(artifact_rankings)[:top_k // 2]

    # -------- step 11: pack context --------
    async with session_scope() as session:
        chunk_rows = await retrieval_sql.fetch_chunk_text(
            session, [cid for cid, _ in fused_chunks]
        )
        artifact_rows = await retrieval_sql.fetch_artifact_rows(
            session, [aid for aid, _ in fused_artifacts]
        )

    evidence: list[dict[str, Any]] = []
    for rank, (cid, score) in enumerate(fused_chunks, start=1):
        info = chunk_rows.get(cid)
        if info is None:
            continue
        evidence.append({
            "kind": "chunk",
            # `node_id` is the chunk's primary-key UUID; persisted as
            # retrieval_evidence.evidence_uuid so downstream consumers can
            # JOIN back to chunks.id without parsing the IRI string.
            "node_id": cid,
            "iri": info["iri"],
            "rank": rank,
            "score": score,
            "text": info["text"],
            "document_iri": info["document_iri"],
            "document_title": info["document_title"],
        })
    for rank, (aid, score) in enumerate(fused_artifacts, start=1):
        info = artifact_rows.get(aid)
        if info is None:
            continue
        evidence.append({
            "kind": "artifact",
            # `node_id` is the artifact's primary-key UUID; persisted as
            # retrieval_evidence.evidence_uuid so JOINs against
            # intelligence_artifacts.id work directly.
            "node_id": aid,
            "iri": info["iri"],
            "artifact_type": info["type"],
            "rank": rank + len(fused_chunks),
            "score": score,
            "text": info["text"],
            "confidence": info["confidence"],
        })

    # -------- step 12: answer generation --------
    answer_task_name = _ANSWER_TASK_BY_MODE[mode]
    answer_prompt_fn = PROMPTS[answer_task_name]
    sys_p, user_p = answer_prompt_fn(resolved_query, evidence)
    try:
        out = await router.chat(answer_task_name, system=sys_p, user=user_p)
        answer = out.text.strip()
    except Exception as exc:
        answer = f"(answer generation failed: {exc})"

    cost = router.total_cost_usd - cost_before
    result = RetrievalResult(
        answer=answer,
        mode=mode,
        resolved_query=resolved_query,
        evidence=evidence,
        parsed=parsed,
        cost_usd=cost,
        wall_seconds=time.time() - t0,
    )
    await _persist_run(
        result, mode=mode, resolved_query=resolved_query,
        matched_classes=matched_classes, matched_entities=matched_entities,
        matched_times=matched_times, graph_hops=hops,
        conversation_turn_id=conversation_turn_id,
    )
    return result


_ANSWER_TASK_BY_MODE = {
    "simple_qa": "answer_simple_qa",
    "deep_research": "answer_deep_research",
}


# --------------- helper steps ---------------


async def _question_parse(router: LLMRouter, q: str) -> dict[str, Any]:
    system, user = PROMPTS["question_parse"](q)
    try:
        out = await router.chat("question_parse", system=system, user=user)
        parsed = _extract_json(out.text) or {}
    except Exception:
        parsed = {}
    return {
        "entities": parsed.get("entities") or [],
        "classes": parsed.get("classes") or [],
        "time_terms": parsed.get("time_terms") or [],
        "intent": parsed.get("intent") or "factoid",
    }


async def _ontology_match(
    parsed: dict[str, Any], qvec: list[float]
) -> tuple[
    list[tuple[uuid.UUID, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Vector + trgm match the parsed terms against ontology + entity
    rows. Returns (seeds, matched_classes, matched_entities,
    matched_times)."""
    seeds: list[tuple[uuid.UUID, str]] = []
    matched_classes: list[dict[str, Any]] = []
    matched_entities: list[dict[str, Any]] = []
    matched_times: list[dict[str, Any]] = []

    async with session_scope() as session:
        # Vector-match classes against the resolved-query embedding.
        r = await session.execute(
            sql_text("""
            SELECT id, iri, label, (embedding <-> CAST(:probe AS vector)) AS dist
              FROM graphrag.ontology_classes
             WHERE embedding IS NOT NULL
             ORDER BY dist LIMIT 30
            """),
            {"probe": _vec_str(qvec)},
        )
        for cid, iri, label, dist in r.all():
            matched_classes.append(
                {"id": cid, "iri": iri, "label": label or "", "dist": float(dist)}
            )
            seeds.append((cid, "ontology_class"))

        # Vector + trgm entity match per parsed entity term.
        for term in parsed.get("entities", [])[:10]:
            if not term.strip():
                continue
            r = await session.execute(
                sql_text("""
                SELECT id, name,
                       similarity(normalized_name, lower(:t)) AS sim
                  FROM graphrag.entities
                 WHERE similarity(normalized_name, lower(:t)) >= 0.4
                 ORDER BY sim DESC LIMIT 10
                """),
                {"t": term},
            )
            for eid, name, sim in r.all():
                matched_entities.append(
                    {"id": eid, "name": name, "sim": float(sim)}
                )
                seeds.append((eid, "entity"))

        # If no entity term parsed out, still try a global vector pass.
        if not matched_entities:
            r = await session.execute(
                sql_text("""
                SELECT id, name, (embedding <-> CAST(:probe AS vector)) AS dist
                  FROM graphrag.entities
                 WHERE embedding IS NOT NULL
                 ORDER BY dist LIMIT 15
                """),
                {"probe": _vec_str(qvec)},
            )
            for eid, name, dist in r.all():
                matched_entities.append(
                    {"id": eid, "name": name, "sim": 1.0 - float(dist) / 2}
                )
                seeds.append((eid, "entity"))

        # Time term matching: try YEAR_YYYY / MONTH_YYYY_MM etc identifiers.
        for term in parsed.get("time_terms", [])[:5]:
            m = re.search(r"\b(19|20)\d{2}\b", term)
            if not m:
                continue
            year = m.group(0)
            ident = f"YEAR_{year}"
            r = await session.execute(
                sql_text(
                    "SELECT id, display_label FROM graphrag.time_instances "
                    "WHERE time_identifier = :ident"
                ),
                {"ident": ident},
            )
            row = r.first()
            if row:
                matched_times.append(
                    {"id": row[0], "display_label": row[1]}
                )
                seeds.append((row[0], "time_instance"))

    return seeds, matched_classes, matched_entities, matched_times


async def _concept_expansion(
    router: LLMRouter, q: str, matched_classes: list[dict[str, Any]]
) -> list[uuid.UUID]:
    """LLM picks 0-15 additional class IRIs from the matched list."""
    if not matched_classes:
        return []
    listing = [(c["iri"], c["label"]) for c in matched_classes[:20]]
    sys_p, user_p = PROMPTS["concept_expansion"](q, listing)
    try:
        out = await router.chat("concept_expansion", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception:
        return []
    related = parsed.get("related_classes") or []
    if not related:
        return []
    # Resolve IRIs back to IDs in our matched set (we don't trust
    # the LLM to invent IRIs).
    iri_to_id = {c["iri"]: c["id"] for c in matched_classes}
    return [iri_to_id[iri] for iri in related if iri in iri_to_id]


async def _query_decompose(router: LLMRouter, q: str) -> list[str]:
    sys_p, user_p = PROMPTS["query_decompose"](q)
    try:
        out = await router.chat("query_decompose", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception:
        return []
    sqs = parsed.get("sub_questions") or []
    return [s for s in sqs if isinstance(s, str) and s.strip()]


# --------------- persistence ---------------


async def _persist_run(
    result: RetrievalResult,
    *,
    mode: str,
    resolved_query: str,
    matched_classes: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    matched_times: list[dict[str, Any]],
    graph_hops: int,
    conversation_turn_id: uuid.UUID | None,
) -> None:
    """INSERT retrieval_runs + retrieval_evidence rows."""
    run_id = uuid.uuid4()
    async with session_scope() as session:
        gv = await current_version(session)
        result.graph_version = gv
        await session.execute(
            sql_text("""
            INSERT INTO graphrag.retrieval_runs (
              id, conversation_turn_id, resolved_query, retrieval_mode,
              matched_classes, matched_entities, matched_time_instances,
              graph_hops, retrieval_plan, graph_version, created_at
            ) VALUES (
              :id, :ctid, :q, :mode,
              CAST(:mc AS jsonb), CAST(:me AS jsonb), CAST(:mt AS jsonb),
              :hops, CAST(:plan AS jsonb), :gv, now()
            )
            """),
            {
                "id": run_id, "ctid": conversation_turn_id,
                "q": resolved_query, "mode": mode,
                "mc": json.dumps([
                    {"iri": c["iri"], "label": c.get("label", "")}
                    for c in matched_classes[:20]
                ]),
                "me": json.dumps([
                    {"name": e["name"]} for e in matched_entities[:20]
                ]),
                "mt": json.dumps([
                    {"label": t["display_label"]} for t in matched_times[:10]
                ]),
                "hops": graph_hops,
                "plan": json.dumps({"intent": result.parsed.get("intent", "")}),
                "gv": gv,
            },
        )

        # Evidence rows. evidence_uuid carries the source node's primary
        # key (chunk_id for kind='chunk', artifact_id for kind='artifact')
        # so downstream queries can JOIN re.evidence_uuid against
        # graphrag.chunks.id / graphrag.intelligence_artifacts.id without
        # parsing the IRI string. Falls back to a fresh UUID only if a
        # caller produced an evidence dict without `node_id` (legacy).
        ev_rows = []
        for ev in result.evidence[:50]:
            node_id = ev.get("node_id")
            if node_id is None:
                node_id = uuid.uuid4()
            ev_rows.append({
                "id": uuid.uuid4(),
                "retrieval_run_id": run_id,
                "evidence_kind": ev["kind"],
                "evidence_uuid": node_id,
                "evidence_iri": ev["iri"],
                "rank": ev["rank"],
                "score": float(ev["score"]),
                "snippet": (ev.get("text") or "")[:500],
            })
        if ev_rows:
            await session.execute(
                sql_text("""
                INSERT INTO graphrag.retrieval_evidence (
                  id, retrieval_run_id, evidence_kind, evidence_uuid,
                  evidence_iri, rank, score, snippet, created_at
                ) VALUES (
                  :id, :retrieval_run_id, :evidence_kind, :evidence_uuid,
                  :evidence_iri, :rank, :score, :snippet, now()
                )
                """),
                ev_rows,
            )

    result.retrieval_run_id = run_id

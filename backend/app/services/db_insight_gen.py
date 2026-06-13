"""Insight artifact generation -- cross-claim synthesis per ontology class.

Pipeline:
  1. Find every ontology_class C whose entities have at least
     `min_claims_per_class` Claim+Finding artifacts asserting about
     them (via Artifact -> viao:assertsAbout -> Entity ->
     instanceOf -> C).
  2. For each C, gather top-25 Claim+Finding artifacts by confidence.
  3. One gpt-4.1 call per cluster (`insight_gen` prompt) returning
     1-3 Insights as JSON {text, confidence}.
  4. Per Insight: INSERT intelligence_artifacts row, ArtifactSource
     rows for every chunk that fed the cluster, edges:
       - Insight -viao:insightBasedOn-> source Claim/Finding artifacts
       - Insight -viao:derivedFromChunk-> shared source chunks
     Embed Insight text.
  5. Bump graph_version.

Idempotent: skips classes that already have any Insight attached
(via insightBasedOn link). Generic: no corpus-specific assumptions.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import ArtifactSource, IntelligenceArtifact
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import (
    VIAO_DERIVED_FROM_CHUNK,
    VIAO_INSIGHT_BASED_ON,
)
from backend.app.services.prompts import PROMPTS

_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"


@dataclass
class InsightGenSummary:
    classes_scanned: int = 0
    clusters_processed: int = 0
    insights_inserted: int = 0
    based_on_edges: int = 0
    derived_from_chunk_edges: int = 0
    artifact_source_rows: int = 0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _insight_iri() -> str:
    return f"{_VIAO_NS}#Insight_{uuid.uuid4().hex[:16]}"


async def generate_insights(
    *,
    min_claims_per_class: int = 10,
    limit: int | None = None,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
    verbose: bool = False,
) -> InsightGenSummary:
    """Synthesize Insights from per-class Claim+Finding clusters."""
    t0 = time.time()
    summary = InsightGenSummary()

    # 1. Find candidate clusters: ontology classes with >=N Claims/Findings.
    async with session_scope() as session:
        skip_subq = sql_text("""
            SELECT DISTINCT ec.id
              FROM graphrag.ontology_classes ec
              JOIN graphrag.entities e ON e.class_id = ec.id
              JOIN graphrag.graph_relationships gr ON gr.target_node_id = e.id
                AND gr.target_node_type = 'entity'
                AND gr.predicate_label = 'viao:assertsAbout'
              JOIN graphrag.intelligence_artifacts ins ON ins.id = gr.source_node_id
                AND ins.artifact_type = 'Insight'
            """)
        # Find classes with >=N Claim+Finding hits, skipping those already
        # with an Insight via the above chain.
        r = await session.execute(
            sql_text("""
            SELECT c.id AS class_id, c.label AS class_label, count(*) AS n_arts
              FROM graphrag.ontology_classes c
              JOIN graphrag.entities e ON e.class_id = c.id
              JOIN graphrag.graph_relationships gr ON gr.target_node_id = e.id
                AND gr.target_node_type = 'entity'
                AND gr.predicate_label = 'viao:assertsAbout'
              JOIN graphrag.intelligence_artifacts a ON a.id = gr.source_node_id
                AND a.artifact_type IN ('Claim','Finding')
                AND a.status = 'ACTIVE'
             WHERE c.id NOT IN (
                SELECT ec.id
                  FROM graphrag.ontology_classes ec
                  JOIN graphrag.entities e2 ON e2.class_id = ec.id
                  JOIN graphrag.graph_relationships gr2
                    ON gr2.target_node_id = e2.id
                   AND gr2.target_node_type = 'entity'
                   AND gr2.predicate_label = 'viao:assertsAbout'
                  JOIN graphrag.intelligence_artifacts ins
                    ON ins.id = gr2.source_node_id
                   AND ins.artifact_type = 'Insight'
             )
             GROUP BY c.id, c.label
             HAVING count(*) >= :min_n
             ORDER BY n_arts DESC
            """),
            {"min_n": min_claims_per_class},
        )
        clusters = r.all()
    summary.classes_scanned = len(clusters)
    if limit is not None:
        clusters = clusters[:limit]
    if not clusters:
        print(
            f"[insight-gen] no classes met the threshold "
            f"(min_claims_per_class={min_claims_per_class})"
        )
        return summary

    print(
        f"[insight-gen] {len(clusters)} class cluster(s) to process "
        f"(min_claims_per_class={min_claims_per_class}, concurrency={concurrency})"
    )

    router = LLMRouter()
    cost_before = router.total_cost_usd
    sem = asyncio.Semaphore(concurrency)
    cost_limit_hit = asyncio.Event()

    # 2. Per cluster: gather top-25 Claim+Finding artifacts + their
    #    chunks. Then one gpt-4.1 call to synthesize insights.
    async def _one_cluster(class_id, class_label, n_arts):
        if cost_limit_hit.is_set():
            return None
        async with sem:
            if cost_limit_hit.is_set():
                return None
            async with session_scope() as session:
                # Pull top artifacts attached to entities of this class.
                r = await session.execute(
                    sql_text("""
                    SELECT a.id, a.artifact_type, a.text, a.confidence
                      FROM graphrag.intelligence_artifacts a
                      JOIN graphrag.graph_relationships gr ON gr.source_node_id = a.id
                       AND gr.source_node_type = 'intelligence_artifact'
                       AND gr.predicate_label = 'viao:assertsAbout'
                      JOIN graphrag.entities e ON e.id = gr.target_node_id
                       AND e.class_id = :cls
                     WHERE a.artifact_type IN ('Claim','Finding')
                       AND a.status = 'ACTIVE'
                     ORDER BY coalesce(a.confidence, 0) DESC, a.created_at DESC
                     LIMIT 25
                    """),
                    {"cls": class_id},
                )
                arts = r.all()
                if len(arts) < 2:
                    return None

                # Collect the chunks these artifacts derived from
                # (we'll attach the Insight to them too).
                art_ids = [aid for aid, _, _, _ in arts]
                r = await session.execute(
                    sql_text("""
                    SELECT DISTINCT chunk_id
                      FROM graphrag.artifact_sources
                     WHERE artifact_id = ANY(CAST(:aids AS uuid[]))
                    """),
                    {"aids": [str(aid) for aid in art_ids]},
                )
                source_chunk_ids = [row[0] for row in r.all()]

            # gpt-4.1 synthesis call.
            payload = [
                {"type": atype, "text": atext,
                 "confidence": float(conf) if conf is not None else None}
                for _, atype, atext, conf in arts
            ]
            system, user = PROMPTS["insight_gen"](class_label or "", payload)
            try:
                out = await router.chat("insight_gen", system=system, user=user)
            except Exception as exc:
                print(f"[insight-gen] cluster '{class_label}' failed: {exc}")
                return None
            parsed = _extract_json(out.text) or {}
            insights = parsed.get("insights") or []
            kept = []
            for it in insights:
                if not isinstance(it, dict):
                    continue
                t = (it.get("text") or "").strip()
                if not t:
                    continue
                try:
                    c = float(it.get("confidence")) if it.get("confidence") is not None else None
                except (TypeError, ValueError):
                    c = None
                kept.append({"text": t, "confidence": c})
            if not kept:
                return None
            if router.total_cost_usd - cost_before > max_cost_usd:
                if not cost_limit_hit.is_set():
                    cost_limit_hit.set()
                    print(f"[insight-gen] HALT: cost cap ${max_cost_usd:.2f} reached")
            return {
                "class_id": class_id,
                "class_label": class_label,
                "source_artifact_ids": art_ids,
                "source_chunk_ids": source_chunk_ids,
                "insights": kept,
            }

    cluster_results = await asyncio.gather(*[
        _one_cluster(cid, clabel, narts) for cid, clabel, narts in clusters
    ])
    cluster_results = [c for c in cluster_results if c is not None]
    summary.llm_cost_usd = router.total_cost_usd - cost_before
    summary.clusters_processed = len(cluster_results)
    print(
        f"[insight-gen] LLM done: ${summary.llm_cost_usd:.4f}, "
        f"{summary.clusters_processed} cluster(s) succeeded"
    )

    if not cluster_results:
        summary.total_cost_usd = summary.llm_cost_usd
        summary.wall_seconds = time.time() - t0
        return summary

    # 3. Insert artifacts + collect IDs.
    artifact_payloads: list[dict[str, Any]] = []
    artifact_iris: list[str] = []
    # Track which cluster each new Insight came from for edge building.
    insight_to_cluster: list[tuple[str, dict[str, Any]]] = []
    embed_texts: list[str] = []
    async with session_scope() as session:
        gv = await current_version(session)
    for cluster in cluster_results:
        for ins in cluster["insights"]:
            airi = _insight_iri()
            artifact_iris.append(airi)
            artifact_payloads.append({
                "artifact_identifier": airi,
                "artifact_type": "Insight",
                "title": cluster.get("class_label") or None,
                "text": ins["text"],
                "confidence": ins["confidence"],
                "model_name": "gpt-4.1",
                "prompt_version": "insight_gen@v1",
                "status": "ACTIVE",
                "graph_version": gv,
                "extra_metadata": {"source_class_id": str(cluster["class_id"])},
            })
            insight_to_cluster.append((airi, cluster))
            embed_texts.append(ins["text"])

    embedder = Embedder()
    embeds = await embedder.embed(embed_texts) if embed_texts else []
    summary.embedding_cost_usd = embedder.total_cost_usd
    for p, vec in zip(artifact_payloads, embeds, strict=False):
        p["embedding"] = vec

    BATCH = 100
    async with session_scope() as session:
        for i in range(0, len(artifact_payloads), BATCH):
            await session.execute(
                pg_insert(IntelligenceArtifact).values(
                    artifact_payloads[i : i + BATCH]
                )
            )
        r = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(artifact_iris))
        )
        iri_to_id = {iri: aid for aid, iri in r.all()}
    summary.insights_inserted = len(artifact_payloads)

    # 4. Build edges + sources for each Insight.
    source_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    sample_buf: list[dict[str, Any]] = []
    for airi, cluster in insight_to_cluster:
        aid = iri_to_id.get(airi)
        if not aid:
            continue
        # Insight -> insightBasedOn -> source Claim/Finding
        for src_aid in cluster["source_artifact_ids"]:
            edge_rows.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": aid,
                "target_node_type": "intelligence_artifact",
                "target_node_id": src_aid,
                "predicate_iri": VIAO_INSIGHT_BASED_ON,
                "predicate_label": "viao:insightBasedOn",
                "relationship_type": "insightBasedOn",
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": True,
                "source_chunk_id": None,
                "source_document_id": None,
                "source_artifact_id": aid,
                "graph_version": gv,
                "extra_metadata": {},
            })
        # Insight -> derivedFromChunk -> shared source chunks (also
        # gets us into artifact_sources for the traceability M2M).
        for cid in cluster["source_chunk_ids"]:
            source_rows.append({"artifact_id": aid, "chunk_id": cid})
            edge_rows.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": aid,
                "target_node_type": "chunk",
                "target_node_id": cid,
                "predicate_iri": VIAO_DERIVED_FROM_CHUNK,
                "predicate_label": "viao:derivedFromChunk",
                "relationship_type": "derivedFromChunk",
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": True,
                "source_chunk_id": cid,
                "source_document_id": None,
                "source_artifact_id": aid,
                "graph_version": gv,
                "extra_metadata": {},
            })
        if len(sample_buf) < 5:
            sample_buf.append({
                "class": cluster.get("class_label"),
                "insight": next(
                    (p["text"] for p in artifact_payloads
                     if p["artifact_identifier"] == airi),
                    "",
                )[:160],
            })

    summary.based_on_edges = sum(
        1 for e in edge_rows
        if e["predicate_label"] == "viao:insightBasedOn"
    )
    summary.derived_from_chunk_edges = sum(
        1 for e in edge_rows
        if e["predicate_label"] == "viao:derivedFromChunk"
    )
    summary.artifact_source_rows = len(source_rows)

    # 5. Persist.
    async with session_scope() as session:
        for i in range(0, len(source_rows), 500):
            await session.execute(
                pg_insert(ArtifactSource).values(source_rows[i : i + 500])
            )
        for i in range(0, len(edge_rows), 500):
            await session.execute(
                pg_insert(GraphRelationship).values(edge_rows[i : i + 500])
            )
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    summary.samples = sample_buf

    print(
        f"[insight-gen] DONE: insights={summary.insights_inserted}, "
        f"insight_based_on_edges={summary.based_on_edges}, "
        f"derived_from_chunk_edges={summary.derived_from_chunk_edges}, "
        f"artifact_sources={summary.artifact_source_rows}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )
    return summary

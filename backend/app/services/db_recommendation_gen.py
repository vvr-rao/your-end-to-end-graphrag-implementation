"""Recommendation artifact generation -- actionable advice from Insights.

Pipeline (v0 -- single-theme, no clustering):
  1. SELECT top-M Insights by confidence (default M=15).
  2. One gpt-4.1 call (`recommendation_gen` prompt) with all M
     Insights -> JSON {recommendations: [{text, confidence}]}.
  3. Per Recommendation:
       - INSERT intelligence_artifacts row (artifact_type='Recommendation').
       - INSERT graph_relationships edges Recommendation
         -viao:recommendationBasedOn-> each source Insight.
       - Embed text.
  4. Bump graph_version.

Idempotent: skips Insights that already have a Recommendation
attached (via recommendationBasedOn). Generic.

Future v1: cluster Insights via k-means on embeddings and run one
synthesis call per cluster. v0 is the single-theme version since
the corpus is small.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import IntelligenceArtifact
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import VIAO_RECOMMENDATION_BASED_ON
from backend.app.services.prompts import PROMPTS


_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"


@dataclass
class RecommendationGenSummary:
    insights_considered: int = 0
    recommendations_inserted: int = 0
    based_on_edges: int = 0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _recommendation_iri() -> str:
    return f"{_VIAO_NS}#Recommendation_{uuid.uuid4().hex[:16]}"


async def generate_recommendations(
    *,
    top_insights: int = 15,
    max_cost_usd: float = 2.0,
    theme_label: str = "Corpus-wide synthesis",
    verbose: bool = False,
) -> RecommendationGenSummary:
    t0 = time.time()
    summary = RecommendationGenSummary()

    # Pull top-M Insights not already referenced by a Recommendation.
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT a.id, a.artifact_identifier, a.text,
                   a.confidence, a.embedding IS NOT NULL AS has_emb
              FROM graphrag.intelligence_artifacts a
             WHERE a.artifact_type = 'Insight'
               AND a.status = 'ACTIVE'
               AND a.id NOT IN (
                  SELECT DISTINCT gr.target_node_id
                    FROM graphrag.graph_relationships gr
                   WHERE gr.predicate_label = 'viao:recommendationBasedOn'
                     AND gr.target_node_type = 'intelligence_artifact'
               )
             ORDER BY coalesce(a.confidence, 0) DESC, a.created_at DESC
             LIMIT :n
            """),
            {"n": top_insights},
        )
        rows = r.all()
    summary.insights_considered = len(rows)
    if len(rows) < 2:
        print(
            f"[recommendation-gen] not enough new Insights "
            f"(found {len(rows)}); nothing to do"
        )
        return summary

    print(
        f"[recommendation-gen] considering top {len(rows)} Insight(s); "
        f"theme='{theme_label}'"
    )

    insights_payload = [
        {"text": text,
         "confidence": float(conf) if conf is not None else None}
        for _, _, text, conf, _ in rows
    ]
    source_insight_ids = [aid for aid, _, _, _, _ in rows]

    router = LLMRouter()
    cost_before = router.total_cost_usd
    system, user = PROMPTS["recommendation_gen"](theme_label, insights_payload)
    try:
        out = await router.chat("recommendation_gen", system=system, user=user)
    except Exception as exc:
        print(f"[recommendation-gen] LLM call failed: {exc}")
        return summary
    parsed = _extract_json(out.text) or {}
    recs_raw = parsed.get("recommendations") or []
    kept = []
    for r in recs_raw:
        if not isinstance(r, dict):
            continue
        t = (r.get("text") or "").strip()
        if not t:
            continue
        try:
            c = float(r.get("confidence")) if r.get("confidence") is not None else None
        except (TypeError, ValueError):
            c = None
        kept.append({"text": t, "confidence": c})
    summary.llm_cost_usd = router.total_cost_usd - cost_before
    print(
        f"[recommendation-gen] LLM done: ${summary.llm_cost_usd:.4f}, "
        f"{len(kept)} recommendation(s)"
    )
    if not kept:
        return summary

    async with session_scope() as session:
        gv = await current_version(session)

    artifact_payloads = []
    artifact_iris = []
    embed_texts = []
    for rec in kept:
        airi = _recommendation_iri()
        artifact_iris.append(airi)
        artifact_payloads.append({
            "artifact_identifier": airi,
            "artifact_type": "Recommendation",
            "title": theme_label,
            "text": rec["text"],
            "confidence": rec["confidence"],
            "model_name": "gpt-4.1",
            "prompt_version": "recommendation_gen@v1",
            "status": "ACTIVE",
            "graph_version": gv,
            "extra_metadata": {"theme": theme_label},
        })
        embed_texts.append(rec["text"])

    embedder = Embedder()
    embeds = await embedder.embed(embed_texts) if embed_texts else []
    summary.embedding_cost_usd = embedder.total_cost_usd
    for p, vec in zip(artifact_payloads, embeds, strict=False):
        p["embedding"] = vec

    async with session_scope() as session:
        await session.execute(
            pg_insert(IntelligenceArtifact).values(artifact_payloads)
        )
        r = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(artifact_iris))
        )
        iri_to_id = {iri: aid for aid, iri in r.all()}

    summary.recommendations_inserted = len(artifact_payloads)

    edge_payloads = []
    sample_buf = []
    for rec_iri, payload in zip(artifact_iris, artifact_payloads, strict=True):
        rec_aid = iri_to_id.get(rec_iri)
        if not rec_aid:
            continue
        for src_iid in source_insight_ids:
            edge_payloads.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": rec_aid,
                "target_node_type": "intelligence_artifact",
                "target_node_id": src_iid,
                "predicate_iri": VIAO_RECOMMENDATION_BASED_ON,
                "predicate_label": "viao:recommendationBasedOn",
                "relationship_type": "recommendationBasedOn",
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": True,
                "source_chunk_id": None,
                "source_document_id": None,
                "source_artifact_id": rec_aid,
                "graph_version": gv,
                "extra_metadata": {},
            })
        if len(sample_buf) < 5:
            sample_buf.append(
                {"recommendation": payload["text"][:200]}
            )

    summary.based_on_edges = len(edge_payloads)

    async with session_scope() as session:
        for i in range(0, len(edge_payloads), 500):
            await session.execute(
                pg_insert(GraphRelationship).values(edge_payloads[i : i + 500])
            )
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    summary.samples = sample_buf

    print(
        f"[recommendation-gen] DONE: recommendations={summary.recommendations_inserted}, "
        f"based_on_edges={summary.based_on_edges}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )
    return summary

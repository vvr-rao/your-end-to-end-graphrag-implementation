"""Milestone E: generate intelligence artifacts from chunks + documents.

Two pipelines:

  1. Per-chunk extraction (default types: Claim, Finding, Observation)
     - One LLM call per chunk to `artifact_chunk_extract` (gpt-4o-mini)
     - Returns JSON {claims, findings, observations} with text + confidence
     - Each item -> one IntelligenceArtifact row
       + one ArtifactSource (artifact <-> chunk)
       + one GraphRelationship (artifact -derivedFromChunk-> chunk)
     - Embeds artifact text via the shared Embedder
     - Idempotent: skips chunks already processed (per type)

  2. Per-document Summary
     - SELECT docs missing a Summary artifact
     - Concatenate chunks' text (or use documents.text_summary)
     - One LLM call to `artifact_document_summary` (gpt-4o-mini)
     - One IntelligenceArtifact (Summary) row
       + ArtifactSource for every chunk
       + GraphRelationship (artifact -summarizes-> doc)

All edges use predicate IRIs from the imported VIAO vocabulary:
viao:derivedFromChunk + viao:summarizes (verified present in
ontology_object_properties at import time).

Generic: works on any corpus that has been ingested via Milestone B.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import ArtifactSource, IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import (
    VIAO_DERIVED_FROM_CHUNK,
    VIAO_DERIVED_FROM_DOCUMENT,
    VIAO_SUMMARIZES,
)
from backend.app.services.prompts import PROMPTS

_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"
_DEFAULT_PER_CHUNK_TYPES = ("Claim", "Finding", "Observation")


@dataclass
class ArtifactGenSummary:
    chunks_scanned: int = 0
    chunks_skipped_already_processed: int = 0
    chunks_failed: int = 0
    artifacts_inserted: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    edges_inserted: int = 0
    sources_inserted: int = 0
    docs_summarized: int = 0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _artifact_iri(artifact_type: str) -> str:
    return f"{_VIAO_NS}#{artifact_type}_{uuid.uuid4().hex[:16]}"


def _extract_json(text: str) -> Any:
    """Permissive JSON extractor: tries full string, then locates the
    first {...} block. Returns parsed object or None on failure."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def generate_per_chunk_artifacts(
    *,
    scope_document_iri: str | None = None,
    limit: int | None = None,
    types: tuple[str, ...] = _DEFAULT_PER_CHUNK_TYPES,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
) -> ArtifactGenSummary:
    """Drive per-chunk Claim+Finding+Observation extraction.

    `scope_document_iri` (optional) limits to a single doc.
    `limit` caps how many chunks we process (smoke test guard).
    `max_cost_usd` is a hard ceiling that aborts mid-run.

    Idempotent: skips chunks that already have ANY of the target
    artifact types attached (avoids re-billing on re-run).
    """
    t0 = time.time()
    summary = ArtifactGenSummary()
    summary.by_type = {t: 0 for t in types}

    async with session_scope() as session:
        already_processed_subq = (
            select(ArtifactSource.chunk_id)
            .join(
                IntelligenceArtifact,
                IntelligenceArtifact.id == ArtifactSource.artifact_id,
            )
            .where(IntelligenceArtifact.artifact_type.in_(types))
        )
        stmt = (
            select(Chunk.id, Chunk.chunk_identifier, Chunk.text, Chunk.document_id)
            .where(
                Chunk.status == "ACTIVE",
                Chunk.id.notin_(already_processed_subq),
            )
            .order_by(Chunk.created_at)
        )
        if scope_document_iri is not None:
            doc_row = await session.execute(
                select(Document.id).where(
                    Document.document_identifier == scope_document_iri
                )
            )
            doc_id = doc_row.scalar_one_or_none()
            if doc_id is None:
                raise ValueError(f"document not found: {scope_document_iri}")
            stmt = stmt.where(Chunk.document_id == doc_id)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        chunks = result.all()

    if not chunks:
        print("[generate-artifacts] no chunks to process")
        return summary

    print(
        f"[generate-artifacts] {len(chunks)} chunk(s) to process "
        f"(types={','.join(types)}, concurrency={concurrency})"
    )

    router = LLMRouter()
    cost_before = router.total_cost_usd
    sem = asyncio.Semaphore(concurrency)
    # Each task returns (chunk_id, chunk_iri, doc_id, parsed_dict or None)
    results: list[tuple[Any, str, Any, dict[str, Any] | None]] = [None] * len(chunks)  # type: ignore[list-item]
    cost_limit_hit = asyncio.Event()

    async def _one(idx: int, chunk_id: Any, chunk_iri: str, text: str, doc_id: Any) -> None:
        if cost_limit_hit.is_set():
            return
        async with sem:
            if cost_limit_hit.is_set():
                return
            system, user = PROMPTS["artifact_chunk_extract"](text)
            try:
                out = await router.chat("artifact_chunk_extract", system=system, user=user)
            except Exception as exc:
                print(f"[generate-artifacts] chunk {chunk_iri} LLM failed: {exc}")
                summary.chunks_failed += 1
                return
            parsed = _extract_json(out.text)
            if not isinstance(parsed, dict):
                print(f"[generate-artifacts] chunk {chunk_iri} unparseable response")
                summary.chunks_failed += 1
                return
            results[idx] = (chunk_id, chunk_iri, doc_id, parsed)

            if router.total_cost_usd - cost_before > max_cost_usd:
                if not cost_limit_hit.is_set():
                    cost_limit_hit.set()
                    print(
                        f"[generate-artifacts] HALT: cost ceiling "
                        f"${max_cost_usd:.2f} reached"
                    )

    tasks = [
        _one(i, cid, ciri, txt, did)
        for i, (cid, ciri, txt, did) in enumerate(chunks)
    ]
    await asyncio.gather(*tasks)
    summary.llm_cost_usd = router.total_cost_usd - cost_before
    summary.chunks_scanned = sum(1 for r in results if r is not None)
    print(
        f"[generate-artifacts] LLM done: ${summary.llm_cost_usd:.4f}, "
        f"{summary.chunks_scanned} success / {summary.chunks_failed} failed"
    )

    # Build artifact payloads + their source links
    artifact_payloads: list[dict[str, Any]] = []
    artifact_iris: list[str] = []
    artifact_to_chunk: list[tuple[str, Any, Any]] = []  # (artifact_iri, chunk_id, doc_id)
    embed_texts: list[str] = []

    for tup in results:
        if tup is None:
            continue
        chunk_id, chunk_iri, doc_id, parsed = tup
        for artifact_type in types:
            # Map type to JSON key (lowercased plural).
            key = artifact_type.lower() + "s"
            items = parsed.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                conf = item.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None

                airi = _artifact_iri(artifact_type)
                artifact_iris.append(airi)
                artifact_payloads.append({
                    "artifact_identifier": airi,
                    "artifact_type": artifact_type,
                    "title": None,
                    "text": text,
                    "confidence": conf,
                    "model_name": "gpt-4o-mini",
                    "prompt_version": "artifact_chunk_extract@v1",
                    "status": "ACTIVE",
                    "graph_version": 0,  # filled in below
                    "extra_metadata": {},
                })
                artifact_to_chunk.append((airi, chunk_id, doc_id))
                embed_texts.append(text)
                summary.by_type[artifact_type] += 1

    if not artifact_payloads:
        print("[generate-artifacts] LLM returned no artifacts; nothing to insert")
        summary.total_cost_usd = summary.llm_cost_usd
        summary.wall_seconds = time.time() - t0
        return summary

    # Embed
    embedder = Embedder()
    embeds = await embedder.embed(embed_texts)
    summary.embedding_cost_usd = embedder.total_cost_usd
    print(
        f"[generate-artifacts] embedded {len(embeds)} artifact(s): "
        f"${summary.embedding_cost_usd:.4f}"
    )

    async with session_scope() as session:
        gv = await current_version(session)
    for p in artifact_payloads:
        p["graph_version"] = gv
    for p, vec in zip(artifact_payloads, embeds, strict=False):
        p["embedding"] = vec

    # Insert artifacts
    ART_BATCH = 200
    async with session_scope() as session:
        for i in range(0, len(artifact_payloads), ART_BATCH):
            await session.execute(
                pg_insert(IntelligenceArtifact).values(
                    artifact_payloads[i : i + ART_BATCH]
                )
            )
        result = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(artifact_iris))
        )
        iri_to_id = {iri: aid for aid, iri in result.all()}

    summary.artifacts_inserted = len(artifact_payloads)

    # Insert artifact_sources + graph_relationships
    source_payloads = []
    edge_payloads = []
    for airi, chunk_id, doc_id in artifact_to_chunk:
        aid = iri_to_id.get(airi)
        if not aid:
            continue
        source_payloads.append({"artifact_id": aid, "chunk_id": chunk_id})
        edge_payloads.append({
            "source_node_type": "intelligence_artifact",
            "source_node_id": aid,
            "target_node_type": "chunk",
            "target_node_id": chunk_id,
            "predicate_iri": VIAO_DERIVED_FROM_CHUNK,
            "predicate_label": "viao:derivedFromChunk",
            "relationship_type": "derivedFromChunk",
            "relationship_source": "LLM_INFERENCE",
            "is_authoritative": True,
            "source_chunk_id": chunk_id,
            "source_document_id": doc_id,
            "source_artifact_id": aid,
            "graph_version": gv,
            "extra_metadata": {},
        })

    BATCH = 500
    async with session_scope() as session:
        for i in range(0, len(source_payloads), BATCH):
            await session.execute(
                pg_insert(ArtifactSource).values(
                    source_payloads[i : i + BATCH]
                )
            )
        for i in range(0, len(edge_payloads), BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(
                    edge_payloads[i : i + BATCH]
                )
            )
    summary.sources_inserted = len(source_payloads)
    summary.edges_inserted = len(edge_payloads)

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    summary.samples = [
        {
            "type": p["artifact_type"],
            "text": p["text"][:120],
            "confidence": float(p["confidence"]) if p["confidence"] is not None else None,
        }
        for p in artifact_payloads[:5]
    ]

    print(
        f"[generate-artifacts] DONE: "
        f"artifacts={summary.artifacts_inserted} "
        f"({', '.join(f'{t}={n}' for t, n in summary.by_type.items())}), "
        f"sources={summary.sources_inserted}, edges={summary.edges_inserted}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )

    return summary


async def generate_document_summaries(
    *,
    scope_document_iri: str | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
) -> ArtifactGenSummary:
    """Per-document Summary artifact. One LLM call per document.

    Idempotent: skips docs that already have a Summary artifact.
    """
    t0 = time.time()
    summary = ArtifactGenSummary()
    summary.by_type = {"Summary": 0}

    async with session_scope() as session:
        # Subquery: docs that already have a Summary artifact via
        # any of their chunks.
        already_summarized_subq = (
            select(Chunk.document_id)
            .join(ArtifactSource, ArtifactSource.chunk_id == Chunk.id)
            .join(
                IntelligenceArtifact,
                IntelligenceArtifact.id == ArtifactSource.artifact_id,
            )
            .where(IntelligenceArtifact.artifact_type == "Summary")
            .distinct()
        )
        stmt = (
            select(
                Document.id,
                Document.document_identifier,
                Document.title,
                Document.text_summary,
            )
            .where(
                Document.status == "ACTIVE",
                Document.id.notin_(already_summarized_subq),
            )
            .order_by(Document.created_at)
        )
        if scope_document_iri is not None:
            stmt = stmt.where(Document.document_identifier == scope_document_iri)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        docs = result.all()

    if not docs:
        print("[generate-artifacts] no documents needing Summary")
        return summary

    print(f"[generate-artifacts] {len(docs)} doc(s) needing Summary")

    router = LLMRouter()
    cost_before = router.total_cost_usd
    sem = asyncio.Semaphore(concurrency)
    doc_results: list[tuple[Any, str, str, str | None] | None] = [None] * len(docs)
    cost_limit_hit = asyncio.Event()

    async def _one(idx: int, doc_id: Any, diri: str, title: str, text_summary: str | None) -> None:
        if cost_limit_hit.is_set():
            return
        body = text_summary or ""
        if not body.strip():
            # Fall back to chunk concatenation.
            async with session_scope() as session:
                r = await session.execute(
                    select(Chunk.text).where(
                        Chunk.document_id == doc_id,
                        Chunk.status == "ACTIVE",
                    ).order_by(Chunk.chunk_index)
                )
                body = "\n\n".join(t for (t,) in r.all())
        if not body.strip():
            return
        async with sem:
            if cost_limit_hit.is_set():
                return
            system, user = PROMPTS["artifact_document_summary"](body)
            try:
                out = await router.chat("artifact_document_summary", system=system, user=user)
            except Exception as exc:
                print(f"[generate-artifacts] doc {diri} LLM failed: {exc}")
                return
            doc_results[idx] = (doc_id, diri, title, out.text.strip())
            if router.total_cost_usd - cost_before > max_cost_usd:
                if not cost_limit_hit.is_set():
                    cost_limit_hit.set()
                    print(
                        f"[generate-artifacts] HALT: cost ceiling "
                        f"${max_cost_usd:.2f} reached"
                    )

    await asyncio.gather(*[
        _one(i, did, diri, title, ts)
        for i, (did, diri, title, ts) in enumerate(docs)
    ])
    summary.llm_cost_usd = router.total_cost_usd - cost_before

    # Build payloads
    art_payloads = []
    art_iris = []
    art_to_doc: list[tuple[str, Any]] = []
    embed_texts = []
    for tup in doc_results:
        if tup is None:
            continue
        doc_id, diri, title, text = tup
        if not text:
            continue
        airi = _artifact_iri("Summary")
        art_iris.append(airi)
        art_payloads.append({
            "artifact_identifier": airi,
            "artifact_type": "Summary",
            "title": title,
            "text": text,
            "confidence": None,
            "model_name": "gpt-4o-mini",
            "prompt_version": "artifact_document_summary@v1",
            "status": "ACTIVE",
            "graph_version": 0,
            "extra_metadata": {"source_document_iri": diri},
        })
        art_to_doc.append((airi, doc_id))
        embed_texts.append(text)
        summary.by_type["Summary"] += 1

    if not art_payloads:
        summary.total_cost_usd = summary.llm_cost_usd
        summary.wall_seconds = time.time() - t0
        return summary

    embedder = Embedder()
    embeds = await embedder.embed(embed_texts)
    summary.embedding_cost_usd = embedder.total_cost_usd

    async with session_scope() as session:
        gv = await current_version(session)
    for p in art_payloads:
        p["graph_version"] = gv
    for p, vec in zip(art_payloads, embeds, strict=False):
        p["embedding"] = vec

    BATCH = 200
    async with session_scope() as session:
        for i in range(0, len(art_payloads), BATCH):
            await session.execute(
                pg_insert(IntelligenceArtifact).values(
                    art_payloads[i : i + BATCH]
                )
            )
        result = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(art_iris))
        )
        iri_to_id = {iri: aid for aid, iri in result.all()}

    summary.artifacts_inserted = len(art_payloads)
    summary.docs_summarized = len(art_payloads)

    # ArtifactSource for every chunk of each doc + summarizes edge
    source_payloads = []
    edge_payloads = []
    for airi, doc_id in art_to_doc:
        aid = iri_to_id.get(airi)
        if not aid:
            continue
        async with session_scope() as session:
            r = await session.execute(
                select(Chunk.id).where(
                    Chunk.document_id == doc_id, Chunk.status == "ACTIVE"
                )
            )
            chunk_ids = [cid for (cid,) in r.all()]
        for cid in chunk_ids:
            source_payloads.append({"artifact_id": aid, "chunk_id": cid})
        edge_payloads.append({
            "source_node_type": "intelligence_artifact",
            "source_node_id": aid,
            "target_node_type": "document",
            "target_node_id": doc_id,
            "predicate_iri": VIAO_SUMMARIZES,
            "predicate_label": "viao:summarizes",
            "relationship_type": "summarizes",
            "relationship_source": "LLM_INFERENCE",
            "is_authoritative": True,
            "source_chunk_id": None,
            "source_document_id": doc_id,
            "source_artifact_id": aid,
            "graph_version": gv,
            "extra_metadata": {},
        })

    async with session_scope() as session:
        for i in range(0, len(source_payloads), 500):
            await session.execute(
                pg_insert(ArtifactSource).values(source_payloads[i : i + 500])
            )
        for i in range(0, len(edge_payloads), 500):
            await session.execute(
                pg_insert(GraphRelationship).values(edge_payloads[i : i + 500])
            )

    summary.sources_inserted = len(source_payloads)
    summary.edges_inserted = len(edge_payloads)

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    summary.samples = [
        {"type": "Summary", "title": p["title"], "text": p["text"][:120]}
        for p in art_payloads[:3]
    ]

    print(
        f"[generate-artifacts] Summary DONE: "
        f"docs={summary.docs_summarized}, "
        f"sources={summary.sources_inserted}, edges={summary.edges_inserted}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )
    return summary

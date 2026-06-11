"""Milestone B driver: ingest a folder of documents into the graphrag schema.

Per-doc pipeline:
  1. Read file via document_io.load_document (PDF + TXT today)
  2. Compute sha256 of the ORIGINAL text (the citation anchor)
  3. Dup check: skip if an ACTIVE row with this hash already exists
  4. Summarize oversize docs via Phase 1 `summarize_long_documents_async`
     (warm disk cache at ~/.cache/your-personal-knowledge-graph-creator/doc_summaries/)
  5. Chunk via Phase 1 `chunk_documents`
  6. Embed chunks + doc-level summary via the shared Embedder
  7. Insert documents + chunks rows
  8. Insert chunk -> viao:chunkOf -> document edges
     (DOCUMENT_EXTRACTION source; predicate from `ontology_object_properties`)
  9. Bump graph_version once

Idempotent: re-running over the same folder is a no-op (skip-by-hash).
Generic: no corpus-specific paths or assumptions; the only input is `folder`.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.chunking import chunk_documents
from backend.app.services.document_io import load_documents
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.pipeline_llm import summarize_long_documents_async
from backend.app.services.predicates import VIAO_CHUNK_OF

_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"


@dataclass
class IngestSummary:
    docs_found: int = 0
    docs_skipped_existing: int = 0
    docs_inserted: int = 0
    chunks_inserted: int = 0
    edges_inserted: int = 0
    summarization_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    per_doc_chunks: list[int] = field(default_factory=list)


def _document_iri(doc_hash: str) -> str:
    return f"{_VIAO_NS}#Document_{doc_hash[:16]}"


def _chunk_iri(doc_hash: str, idx: int) -> str:
    return f"{_VIAO_NS}#Chunk_{doc_hash[:16]}_{idx:04d}"


async def ingest_documents_folder(
    folder: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    summarization_threshold: int = 2000,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    concurrency: int = 4,
) -> IngestSummary:
    """Ingest every supported file in `folder`. See module docstring."""
    t0 = time.time()
    summary = IngestSummary()

    docs = list(load_documents(folder))
    summary.docs_found = len(docs)
    if limit is not None:
        docs = docs[:limit]

    docs = [d for d in docs if d.text.strip()]
    if not docs:
        print(f"[ingest] no documents to process in {folder}")
        return summary

    hashes = [hashlib.sha256(d.text.encode("utf-8")).hexdigest() for d in docs]
    iris = [_document_iri(h) for h in hashes]

    # Within-batch dedup: corpora sometimes contain duplicate files
    # (e.g. "foo.txt" + "foo (1).txt" with identical contents). Keep
    # the first occurrence so the INSERT doesn't blow up on its own
    # batch.
    seen_in_batch: set[str] = set()
    intra_batch_dups = 0
    keep_for_batch = []
    for i, iri in enumerate(iris):
        if iri in seen_in_batch:
            intra_batch_dups += 1
            continue
        seen_in_batch.add(iri)
        keep_for_batch.append(i)
    if intra_batch_dups:
        print(
            f"[ingest] {intra_batch_dups} duplicate file(s) within input "
            "(same sha256) — keeping first occurrence only"
        )

    # Dup check against active rows already in the DB.
    candidate_iris = [iris[i] for i in keep_for_batch]
    async with session_scope() as session:
        result = await session.execute(
            select(Document.document_identifier).where(
                Document.document_identifier.in_(candidate_iris),
                Document.status == "ACTIVE",
            )
        )
        existing = {row[0] for row in result.all()}

    keep = [i for i in keep_for_batch if iris[i] not in existing]
    summary.docs_skipped_existing = len(existing)

    if not keep:
        print(
            f"[ingest] all {len(docs)} doc(s) already loaded "
            f"(matched by sha256); nothing to do"
        )
        return summary

    fresh = [docs[i] for i in keep]
    fresh_hashes = [hashes[i] for i in keep]
    fresh_iris = [iris[i] for i in keep]

    print(
        f"[ingest] {len(fresh)} new doc(s); "
        f"skipped {summary.docs_skipped_existing} existing"
    )

    if dry_run:
        print("[ingest] DRY RUN: would ingest the above; exiting.")
        return summary

    router = LLMRouter()
    cost_before = router.total_cost_usd
    print(
        f"[ingest] summarizing (threshold={summarization_threshold} tok, "
        f"concurrency={concurrency}) ..."
    )
    summarized = await summarize_long_documents_async(
        fresh,
        router,
        threshold_tokens=summarization_threshold,
        concurrency=concurrency,
        model_name="gpt-4o-mini",
    )
    summary.summarization_cost_usd = router.total_cost_usd - cost_before
    print(
        f"[ingest] summarization done: ${summary.summarization_cost_usd:.4f}"
    )

    # Chunk every summarized doc.
    chunks_by_doc: list[list[Any]] = []
    for sdoc in summarized:
        cs = list(
            chunk_documents([sdoc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        chunks_by_doc.append(cs)
        summary.per_doc_chunks.append(len(cs))

    all_chunk_texts: list[str] = []
    for cs in chunks_by_doc:
        all_chunk_texts.extend(c.text for c in cs)

    if not all_chunk_texts:
        print("[ingest] WARNING: zero chunks produced; bailing")
        return summary

    print(
        f"[ingest] {len(all_chunk_texts)} chunk(s) across "
        f"{len(fresh)} doc(s); embedding ..."
    )

    embedder = Embedder()
    chunk_vectors = await embedder.embed(all_chunk_texts)
    doc_summary_texts = [sd.text for sd in summarized]
    doc_vectors = await embedder.embed(doc_summary_texts)
    summary.embedding_cost_usd = embedder.total_cost_usd
    print(f"[ingest] embedding done: ${summary.embedding_cost_usd:.4f}")

    # Insert documents.
    doc_payloads = []
    for i, (orig, summ, dhash, diri) in enumerate(
        zip(fresh, summarized, fresh_hashes, fresh_iris, strict=True)
    ):
        doc_payloads.append({
            "document_identifier": diri,
            "title": orig.path.stem,
            "file_name": orig.path.name,
            "file_path": str(orig.path.resolve()),
            "file_type": orig.path.suffix.lstrip("."),
            "document_hash": dhash,
            "text_summary": summ.text,
            "embedding": doc_vectors[i] if i < len(doc_vectors) else None,
            "status": "ACTIVE",
            "version": 1,
            "extra_metadata": {
                "original_text_bytes": len(orig.text.encode("utf-8")),
                "summary_text_bytes": len(summ.text.encode("utf-8")),
            },
        })

    async with session_scope() as session:
        await session.execute(pg_insert(Document).values(doc_payloads))
        result = await session.execute(
            select(Document.id, Document.document_identifier).where(
                Document.document_identifier.in_(fresh_iris)
            )
        )
        iri_to_doc_id = {iri: did for did, iri in result.all()}

    summary.docs_inserted = len(doc_payloads)
    print(f"[ingest] inserted {summary.docs_inserted} document row(s)")

    # Build + insert chunks.
    chunk_payloads = []
    chunk_iris_in_order: list[str] = []
    global_idx = 0
    for di, cs in enumerate(chunks_by_doc):
        doc_id = iri_to_doc_id[fresh_iris[di]]
        for c in cs:
            ciri = _chunk_iri(fresh_hashes[di], c.index)
            chunk_iris_in_order.append(ciri)
            chunk_payloads.append({
                "document_id": doc_id,
                "chunk_identifier": ciri,
                "chunk_index": c.index,
                "text": c.text,
                "token_count": c.token_count,
                "embedding": (
                    chunk_vectors[global_idx]
                    if global_idx < len(chunk_vectors)
                    else None
                ),
                "status": "ACTIVE",
                "extra_metadata": {},
            })
            global_idx += 1

    CHUNK_BATCH = 200
    async with session_scope() as session:
        for i in range(0, len(chunk_payloads), CHUNK_BATCH):
            await session.execute(
                pg_insert(Chunk).values(chunk_payloads[i : i + CHUNK_BATCH])
            )
        result = await session.execute(
            select(Chunk.id, Chunk.chunk_identifier).where(
                Chunk.chunk_identifier.in_(chunk_iris_in_order)
            )
        )
        iri_to_chunk_id = {iri: cid for cid, iri in result.all()}

    summary.chunks_inserted = len(chunk_payloads)
    print(f"[ingest] inserted {summary.chunks_inserted} chunk row(s)")

    # Build + insert chunk -> viao:chunkOf -> document edges.
    async with session_scope() as session:
        gv = await current_version(session)

    edge_payloads = []
    for di, cs in enumerate(chunks_by_doc):
        doc_id = iri_to_doc_id[fresh_iris[di]]
        for c in cs:
            ciri = _chunk_iri(fresh_hashes[di], c.index)
            chunk_id = iri_to_chunk_id[ciri]
            edge_payloads.append({
                "source_node_type": "chunk",
                "source_node_id": chunk_id,
                "target_node_type": "document",
                "target_node_id": doc_id,
                "predicate_iri": VIAO_CHUNK_OF,
                "predicate_label": "viao:chunkOf",
                "relationship_type": "chunkOf",
                "relationship_source": "DOCUMENT_EXTRACTION",
                "is_authoritative": True,
                "source_document_id": doc_id,
                "source_chunk_id": chunk_id,
                "source_artifact_id": None,
                "graph_version": gv,
                "extra_metadata": {},
            })

    EDGE_BATCH = 500
    async with session_scope() as session:
        for i in range(0, len(edge_payloads), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(
                    edge_payloads[i : i + EDGE_BATCH]
                )
            )

    summary.edges_inserted = len(edge_payloads)
    print(f"[ingest] inserted {summary.edges_inserted} chunk-of edge(s)")

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = (
        summary.summarization_cost_usd + summary.embedding_cost_usd
    )
    summary.wall_seconds = time.time() - t0

    print(
        f"[ingest] DONE: docs={summary.docs_inserted}, "
        f"chunks={summary.chunks_inserted}, "
        f"edges={summary.edges_inserted}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )

    return summary

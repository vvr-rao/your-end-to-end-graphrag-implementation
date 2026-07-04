"""Milestone B driver: ingest a folder of documents into the graphrag schema.

Per-doc pipeline:
  1. Read file via document_io.load_document (PDF + TXT today)
  2. Compute sha256 of the ORIGINAL text (the citation anchor)
  3. Dup check: skip if an ACTIVE row with this hash already exists
  4. Summarize oversize docs via Phase 1 `summarize_long_documents_async`
     (warm disk cache at ~/.cache/your-end-to-end-graphrag-implementation/doc_summaries/)
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
from backend.app.db.models.artifacts import IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.chunking import chunk_documents
from backend.app.services.document_io import load_documents
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.pipeline_llm import summarize_long_documents_async
from backend.app.services.predicates import (
    VIAO_CHUNK_OF,
    VIAO_DERIVED_FROM_DOCUMENT,
    VIAO_HAS_INTELLIGENCE_ARTIFACT,
)
from backend.app.services.prompts import PROMPTS

_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"

# Embedding model input cap (OpenAI: 8192 hard; 8000 leaves headroom).
_EMBED_INPUT_CAP_TOKENS = 8000
# Per gpt-4o-mini call we never feed more than ~100K tokens (matches
# Phase 1's `max_doc_input_tokens` in `summarize_long_documents_async`).
# Above that we split + summarize each half + recurse.
_LLM_INPUT_SPLIT_THRESHOLD = 100_000


async def _compress_summary_for_embed(
    text: str, router: LLMRouter, *, target_tokens: int = _EMBED_INPUT_CAP_TOKENS
) -> str:
    """If a doc summary exceeds the embedding model's input cap, run
    extra gpt-4o-mini pass(es) to compress it BEFORE the embedding API
    call. Always preserves the original text -- the caller stores the
    UNCOMPRESSED summary in documents.text_summary; this output is used
    only to compute documents.embedding.

    Recursive contract:
      - Returns the input unchanged if it's already <= target_tokens.
      - For inputs <= 100K tokens: one gpt-4o-mini compression call
        (max_tokens=4096 by task config, well under target_tokens).
      - For inputs > 100K tokens: split in half on a paragraph
        boundary, compress each half independently, recurse on the
        concatenation. Bounded by an iteration cap so a pathological
        case can't loop forever.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
    except (ImportError, KeyError):
        return text  # let the embedder do hard truncation

    def _tok_count(s: str) -> int:
        return len(enc.encode(s, disallowed_special=()))

    async def _compress_once(s: str) -> str:
        """One gpt-4o-mini pass via the existing document_summarize task."""
        system, user = PROMPTS["document_summarize"](s)
        try:
            result = await router.chat(
                "document_summarize", system=system, user=user
            )
        except Exception as exc:
            print(f"[ingest] compression LLM call failed: {exc}")
            return s
        return result.text.strip() or s

    def _split_on_paragraph(s: str) -> tuple[str, str]:
        """Split near the middle on a blank-line boundary if one exists."""
        mid = len(s) // 2
        # Search outwards for a `\n\n` near the midpoint.
        right = s.find("\n\n", mid)
        left = s.rfind("\n\n", 0, mid)
        if right >= 0 and (left < 0 or right - mid <= mid - left):
            return s[:right], s[right + 2 :]
        if left >= 0:
            return s[:left], s[left + 2 :]
        return s[:mid], s[mid:]

    n = _tok_count(text)
    if n <= target_tokens:
        return text

    print(
        f"[ingest] doc summary {n:,} tokens > {target_tokens} cap; "
        f"compressing for embed call (full text still kept in DB)"
    )

    # Iteration cap so we never loop forever on a pathological input.
    current = text
    for iteration in range(4):
        n = _tok_count(current)
        if n <= target_tokens:
            return current

        if n <= _LLM_INPUT_SPLIT_THRESHOLD:
            compressed = await _compress_once(current)
            n2 = _tok_count(compressed)
            if n2 >= n:
                # Model didn't actually compress; bail and let embedder truncate.
                print(
                    f"[ingest] compression no-op (out={n2:,} >= in={n:,}); "
                    "embedder will hard-truncate"
                )
                return text
            print(f"[ingest] compress iter {iteration + 1}: {n:,} -> {n2:,} tokens")
            current = compressed
            continue

        # Above split threshold: divide-and-conquer
        left, right = _split_on_paragraph(current)
        comp_l = await _compress_once(left) if _tok_count(left) > target_tokens else left
        comp_r = await _compress_once(right) if _tok_count(right) > target_tokens else right
        current = comp_l + "\n\n" + comp_r
        print(
            f"[ingest] split-and-compress iter {iteration + 1}: "
            f"{n:,} -> {_tok_count(current):,} tokens"
        )

    if _tok_count(current) > target_tokens:
        print(
            f"[ingest] reached compression iteration cap with "
            f"{_tok_count(current):,} tokens; embedder will hard-truncate"
        )
    return current


@dataclass
class IngestSummary:
    docs_found: int = 0
    docs_skipped_existing: int = 0
    docs_inserted: int = 0
    chunks_inserted: int = 0
    edges_inserted: int = 0
    tables_inserted: int = 0
    table_extract_cost_usd: float = 0.0
    summarization_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    per_doc_chunks: list[int] = field(default_factory=list)


def _document_iri(doc_hash: str) -> str:
    return f"{_VIAO_NS}#Document_{doc_hash[:16]}"


def _clean_text(s: str) -> str:
    """Strip NUL (0x00) bytes. Postgres text/jsonb columns reject them
    ('invalid byte sequence for encoding UTF8: 0x00'), and raw original text
    (esp. pypdf-extracted PDF text) routinely contains them -- unlike the
    LLM-generated summaries, which are always clean. Applied to every chunk
    text before embedding + insert so both summary (for under-threshold docs
    chunked from the original) and full-text chunks are safe."""
    return s.replace("\x00", "") if s else s


def _chunk_iri(doc_hash: str, idx: int) -> str:
    return f"{_VIAO_NS}#Chunk_{doc_hash[:16]}_{idx:04d}"


def _fulltext_chunk_iri(doc_hash: str, idx: int) -> str:
    # Distinct namespace from summary chunks so both sets can coexist for one doc.
    return f"{_VIAO_NS}#Chunk_{doc_hash[:16]}_ft_{idx:04d}"


async def ingest_documents_folder(
    folder: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    summarization_threshold: int = 2000,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    concurrency: int = 4,
    extract_tables: bool = False,
    table_vision: bool = True,
    full_text_chunks: bool = False,
) -> IngestSummary:
    """Ingest every supported file in `folder`. See module docstring.

    `full_text_chunks` (default False): in ADDITION to the summary chunks
    (kind='summary', what gets embedded + used by entity/artifact extraction),
    also chunk + embed the VERBATIM original text as kind='fulltext' chunks.
    Retrieval prefers fulltext chunks when present (better recall + exact
    citations); entity/artifact extraction still runs over summary chunks only.
    Default off → byte-identical output to before.
    """
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
        all_chunk_texts.extend(_clean_text(c.text) for c in cs)

    if not all_chunk_texts:
        print("[ingest] WARNING: zero chunks produced; bailing")
        return summary

    print(
        f"[ingest] {len(all_chunk_texts)} chunk(s) across "
        f"{len(fresh)} doc(s); embedding ..."
    )

    embedder = Embedder()
    chunk_vectors = await embedder.embed(all_chunk_texts)

    # Doc-level embedding inputs: if any summary exceeds the 8K cap,
    # recursively compress it for the embedding call ONLY. The full
    # summary still gets stored in documents.text_summary below.
    cost_before_compress = router.total_cost_usd
    doc_embed_inputs: list[str] = []
    for sd in summarized:
        doc_embed_inputs.append(
            await _compress_summary_for_embed(sd.text, router)
        )
    compress_cost = router.total_cost_usd - cost_before_compress
    if compress_cost > 0:
        print(
            f"[ingest] oversize-summary compression cost: "
            f"${compress_cost:.4f}"
        )
        summary.summarization_cost_usd += compress_cost

    doc_vectors = await embedder.embed(doc_embed_inputs)
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
                "text": _clean_text(c.text),
                "kind": "summary",
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

    # Opt-in: additionally store verbatim full-text chunks (kind='fulltext').
    # Additive — leaves the summary-chunk path above byte-identical.
    if full_text_chunks:
        await _ingest_fulltext_chunks_for_docs(
            fresh, fresh_hashes, fresh_iris, iri_to_doc_id, summary,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

    # Phase 2a opt-in: ingest StructuredTable artifacts for any PDF docs.
    # Uses the disk cache populated by `prune-expand --tables` when
    # available; falls back to inline extraction (paid LLM if vision is
    # on) for cache misses. No-op when `extract_tables=False`.
    if extract_tables:
        await _ingest_tables_for_docs(
            fresh, fresh_hashes, fresh_iris, iri_to_doc_id,
            summary, table_vision=table_vision,
        )

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = (
        summary.summarization_cost_usd
        + summary.embedding_cost_usd
        + summary.table_extract_cost_usd
    )
    summary.wall_seconds = time.time() - t0

    tables_note = (
        f", tables={summary.tables_inserted}" if extract_tables else ""
    )
    print(
        f"[ingest] DONE: docs={summary.docs_inserted}, "
        f"chunks={summary.chunks_inserted}, "
        f"edges={summary.edges_inserted}{tables_note}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )

    return summary


# ---------- Optional: ingest verbatim full-text chunks (kind='fulltext') ----------


async def _ingest_fulltext_chunks_for_docs(
    fresh_docs: list,
    fresh_hashes: list[str],
    fresh_iris: list[str],
    iri_to_doc_id: dict[str, Any],
    summary: IngestSummary,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Chunk + embed each doc's VERBATIM original text and store the rows as
    kind='fulltext', plus their chunk->chunkOf->document edges. Additive: the
    summary-chunk path is untouched. Entity/artifact extraction ignore these
    rows (they filter kind='summary'); retrieval prefers them.
    """
    # Chunk each original doc (per-doc so chunk indices are 0-based per doc).
    ft_chunks_by_doc: list[list[Any]] = []
    for doc in fresh_docs:
        cs = list(
            chunk_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        ft_chunks_by_doc.append(cs)

    all_texts: list[str] = []
    for cs in ft_chunks_by_doc:
        all_texts.extend(_clean_text(c.text) for c in cs)
    if not all_texts:
        print("[ingest][fulltext] zero full-text chunks produced; skipping")
        return

    print(
        f"[ingest][fulltext] {len(all_texts)} full-text chunk(s) across "
        f"{len(fresh_docs)} doc(s); embedding ..."
    )
    embedder = Embedder()
    vectors = await embedder.embed(all_texts)
    summary.embedding_cost_usd += embedder.total_cost_usd

    # Build + insert chunk rows.
    chunk_payloads: list[dict[str, Any]] = []
    chunk_iris_in_order: list[str] = []
    gidx = 0
    for di, cs in enumerate(ft_chunks_by_doc):
        doc_id = iri_to_doc_id.get(fresh_iris[di])
        if doc_id is None:
            gidx += len(cs)
            continue
        for c in cs:
            ciri = _fulltext_chunk_iri(fresh_hashes[di], c.index)
            chunk_iris_in_order.append(ciri)
            chunk_payloads.append({
                "document_id": doc_id,
                "chunk_identifier": ciri,
                "chunk_index": c.index,
                "text": _clean_text(c.text),
                "kind": "fulltext",
                "token_count": c.token_count,
                "embedding": vectors[gidx] if gidx < len(vectors) else None,
                "status": "ACTIVE",
                "extra_metadata": {},
            })
            gidx += 1

    if not chunk_payloads:
        return

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

    summary.chunks_inserted += len(chunk_payloads)
    print(f"[ingest][fulltext] inserted {len(chunk_payloads)} full-text chunk row(s)")

    # chunk -> viao:chunkOf -> document edges.
    async with session_scope() as session:
        gv = await current_version(session)

    edge_payloads: list[dict[str, Any]] = []
    for di, cs in enumerate(ft_chunks_by_doc):
        doc_id = iri_to_doc_id.get(fresh_iris[di])
        if doc_id is None:
            continue
        for c in cs:
            ciri = _fulltext_chunk_iri(fresh_hashes[di], c.index)
            chunk_id = iri_to_chunk_id.get(ciri)
            if chunk_id is None:
                continue
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
                pg_insert(GraphRelationship).values(edge_payloads[i : i + EDGE_BATCH])
            )
    summary.edges_inserted += len(edge_payloads)
    print(f"[ingest][fulltext] inserted {len(edge_payloads)} full-text chunk-of edge(s)")


# ---------- Phase 2a: ingest StructuredTable artifacts for PDF docs ----------


def _table_artifact_iri(doc_hash: str, table_index: int) -> str:
    return f"{_VIAO_NS}#StructuredTable_{doc_hash[:16]}_{table_index:04d}"


async def _ingest_tables_for_docs(
    fresh_docs: list,
    fresh_hashes: list[str],
    fresh_iris: list[str],
    iri_to_doc_id: dict[str, Any],
    summary: IngestSummary,
    *,
    table_vision: bool,
) -> None:
    """Per-PDF: load (cache or fresh-extract) tables, embed the flat
    text summary, insert one IntelligenceArtifact row per table, then
    write the StructuredTable -> derivedFromDocument edges. Skips
    non-PDF inputs silently. Soft-fails per doc -- one bad PDF won't
    take down the rest."""
    pdf_indices: list[int] = [
        i for i, d in enumerate(fresh_docs)
        if d.path.suffix.lower() == ".pdf"
    ]
    if not pdf_indices:
        return

    from backend.app.services import table_cache, table_extract, table_jsonld

    print(
        f"[ingest][tables] {len(pdf_indices)} PDF(s) eligible "
        f"(vision={'ON' if table_vision else 'OFF'}, isolation=subprocess)"
    )

    # Phase 2a v2: extract tables via subprocess-per-PDF workers so each
    # PDF's memory state is fully reclaimed by the OS when its worker
    # exits. Replaces the previous in-process extractor which OOM'd
    # under cumulative heap fragmentation. Each worker writes its
    # JSON-LD to the shared user cache; we read from cache for the
    # DB-insert phase below.
    pdf_paths = [fresh_docs[i].path for i in pdf_indices]
    manifests = await table_extract.extract_tables_for_paths_subprocess(
        pdf_paths,
        run_cache_dir=None,
        use_vision=table_vision,
        concurrency=1,
    )
    for m in manifests.values():
        if m.get("source") not in ("cache", "skipped", "spawn-failed", "worker-failed"):
            summary.table_extract_cost_usd += float(m.get("cost_usd", 0.0) or 0.0)

    # Read each worker's persisted JSON-LD bundle from the user cache
    # and assemble the DB-side artifact payloads. The on-disk payloads
    # are the source of truth -- the manifests above only carry counts.
    user_cache = table_cache.user_cache_dir()
    artifact_payloads: list[dict[str, Any]] = []
    artifact_iris_in_order: list[str] = []
    embed_texts: list[str] = []
    artifact_to_doc_id: list[tuple[str, Any]] = []  # (artifact_iri, doc_id)

    for i in pdf_indices:
        doc = fresh_docs[i]
        doc_id = iri_to_doc_id.get(fresh_iris[i])
        if doc_id is None:
            continue
        try:
            _, cache_key = table_extract._hash_pdf_streaming(doc.path)
        except Exception as exc:
            print(
                f"[ingest][tables] {doc.path.name}: hash failed ({exc}); "
                f"skipping"
            )
            continue
        hit = table_cache.load(user_cache, cache_key)
        if hit is None or not hit.tables:
            continue
        for t_idx, payload in enumerate(hit.tables):
            errors = table_jsonld.validate_table_jsonld(payload)
            if errors:
                continue
            airi = _table_artifact_iri(fresh_hashes[i], t_idx)
            summary_text = table_jsonld.flat_text_summary(payload, max_cells=40)
            if not summary_text.strip():
                # Empty summary text -- skip to keep embedding meaningful.
                continue
            artifact_payloads.append({
                "artifact_identifier": airi,
                "artifact_type": "StructuredTable",
                "title": (payload.get("caption") or "")[:200] or None,
                "text": summary_text,
                "confidence": None,
                "model_name": payload.get("extractionMethod") or "pdfplumber",
                "prompt_version": "phase2a_table_extract@v1",
                "status": "ACTIVE",
                "graph_version": 0,  # filled in below
                "extra_metadata": payload,
            })
            artifact_iris_in_order.append(airi)
            embed_texts.append(summary_text)
            artifact_to_doc_id.append((airi, doc_id))

    if not artifact_payloads:
        print("[ingest][tables] no extractable tables across the PDF set")
        return

    embedder = Embedder()
    vectors = await embedder.embed(embed_texts)
    summary.embedding_cost_usd += embedder.total_cost_usd

    async with session_scope() as session:
        gv = await current_version(session)
    for p in artifact_payloads:
        p["graph_version"] = gv
    for p, v in zip(artifact_payloads, vectors, strict=False):
        p["embedding"] = v

    ART_BATCH = 200
    async with session_scope() as session:
        for i in range(0, len(artifact_payloads), ART_BATCH):
            await session.execute(
                pg_insert(IntelligenceArtifact).values(
                    artifact_payloads[i : i + ART_BATCH]
                )
            )
        rs = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(artifact_iris_in_order))
        )
        iri_to_artifact_id = {iri: aid for aid, iri in rs.all()}

    summary.tables_inserted = len(artifact_payloads)
    print(
        f"[ingest][tables] inserted {summary.tables_inserted} "
        f"StructuredTable artifact(s) "
        f"(extract cost ${summary.table_extract_cost_usd:.4f})"
    )

    # Edges: Artifact -> derivedFromDocument -> Document AND
    #        Document -> hasIntelligenceArtifact -> Artifact (inverse).
    table_edges: list[dict[str, Any]] = []
    for airi, doc_id in artifact_to_doc_id:
        aid = iri_to_artifact_id.get(airi)
        if aid is None:
            continue
        # Forward edge.
        table_edges.append({
            "source_node_type": "intelligence_artifact",
            "source_node_id": aid,
            "target_node_type": "document",
            "target_node_id": doc_id,
            "predicate_iri": VIAO_DERIVED_FROM_DOCUMENT,
            "predicate_label": "viao:derivedFromDocument",
            "relationship_type": "derivedFromDocument",
            "relationship_source": "DOCUMENT_EXTRACTION",
            "is_authoritative": True,
            "source_document_id": doc_id,
            "source_chunk_id": None,
            "source_artifact_id": aid,
            "graph_version": gv,
            "extra_metadata": {},
        })
        # Inverse edge.
        table_edges.append({
            "source_node_type": "document",
            "source_node_id": doc_id,
            "target_node_type": "intelligence_artifact",
            "target_node_id": aid,
            "predicate_iri": VIAO_HAS_INTELLIGENCE_ARTIFACT,
            "predicate_label": "viao:hasIntelligenceArtifact",
            "relationship_type": "hasIntelligenceArtifact",
            "relationship_source": "DOCUMENT_EXTRACTION",
            "is_authoritative": True,
            "source_document_id": doc_id,
            "source_chunk_id": None,
            "source_artifact_id": aid,
            "graph_version": gv,
            "extra_metadata": {},
        })

    EDGE_BATCH = 500
    async with session_scope() as session:
        for i in range(0, len(table_edges), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(
                    table_edges[i : i + EDGE_BATCH]
                )
            )
    summary.edges_inserted += len(table_edges)
    print(f"[ingest][tables] inserted {len(table_edges)} table<->doc edge(s)")

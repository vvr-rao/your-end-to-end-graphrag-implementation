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

from sqlalchemy import func, select, text as sql_text, update as sql_update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.core.config import get_settings
from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import ArtifactSource, IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.entities import Entity
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.models.ontology import OntologyClass
from backend.app.db.session import session_scope
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import (
    VIAO_ASSERTS_ABOUT,
    VIAO_DERIVED_FROM_CHUNK,
    VIAO_DERIVED_FROM_DOCUMENT,
    VIAO_SUMMARIZES,
)
from backend.app.services.prompts import PROMPTS

_VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"
_DEFAULT_PER_CHUNK_TYPES = ("Claim", "Finding", "Observation", "Event")


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


# Short forms at or below this length (acronyms: "FTX", "SEC", "X") match
# case-sensitively, so "sec" in running text does not link the SEC.
_CASE_SENSITIVE_MAX_LEN = 3


def match_artifact_entities(
    text: str, entities: list[dict[str, Any]]
) -> set[Any]:
    """Entity ids an artifact's text NAMES, by full name OR a stored alias.

    `entities` are the candidates in scope -- the artifact's source chunk's
    entities, or a summary's document's -- each {entity_id, canonical_name,
    aliases}. Matching the full canonical name alone linked 3 of the 59
    artifacts that say "FTX" to `FTX Trading Ltd.`, and none that say
    "Bankman-Fried" to `Sam Bankman-Fried`; the aliases recorded by name
    collapse ("FTX", "Bankman-Fried") close that gap.

    Guards: whole-word matches only ("Apple" is not in "pineapple"); forms of
    3 characters or fewer match case-sensitively; and a form shared by two
    candidates ("Kelce" for Travis and Jason) links neither -- an ambiguous
    short form is skipped rather than guessed.
    """
    if not text:
        return set()
    owners: dict[str, set[Any]] = {}
    original: dict[str, str] = {}
    for e in entities:
        eid = e.get("entity_id")
        if eid is None:
            continue
        forms = [e.get("canonical_name") or "", *(e.get("aliases") or [])]
        for form in forms:
            form = " ".join(str(form).split())
            if len(form) < 2:
                continue
            key = form.lower()
            owners.setdefault(key, set()).add(eid)
            original.setdefault(key, form)
    matched: set[Any] = set()
    for key, ids in owners.items():
        if len(ids) != 1:
            continue
        form = original[key]
        flags = 0 if len(form) <= _CASE_SENSITIVE_MAX_LEN else re.IGNORECASE
        if re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", text, flags):
            matched |= ids
    return matched


def _aliases_of(extra_metadata: Any) -> list[str]:
    if isinstance(extra_metadata, dict):
        return [a for a in (extra_metadata.get("aliases") or []) if isinstance(a, str)]
    return []


def _artifact_iri(artifact_type: str) -> str:
    return f"{_VIAO_NS}#{artifact_type}_{uuid.uuid4().hex[:16]}"


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_str_field(value: Any) -> str | None:
    """Strip + null-out empty strings. Non-strings -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def _normalize_date_field(value: Any) -> str | None:
    """Accept a YYYY-MM-DD string from the LLM and return it; otherwise None.

    Kept permissive: we DO NOT parse to date here. The LLM occasionally
    returns "2024" or "Jan 2024"; store null rather than risk a malformed
    column value. Downstream consumers can re-parse from the raw field
    later if we tighten validation.
    """
    s = _normalize_str_field(value)
    if s is None or not _ISO_DATE_RE.match(s):
        return None
    return s


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


def _unprocessed_chunk_filter(
    types: tuple[str, ...], chunk_kind: str, doc_id: Any = None
) -> list[Any]:
    """WHERE clauses for chunks that still need per-chunk artifacts.

    The ONE definition shared by the selection in `generate_per_chunk_artifacts`
    and the count in `_unprocessed_artifact_chunk_count`. If those drift, the
    streaming loop misjudges progress and either skips real work or re-pays for
    the same chunks.
    """
    already_processed_subq = (
        select(ArtifactSource.chunk_id)
        .join(
            IntelligenceArtifact,
            IntelligenceArtifact.id == ArtifactSource.artifact_id,
        )
        .where(IntelligenceArtifact.artifact_type.in_(types))
    )
    clauses = [
        Chunk.status == "ACTIVE",
        Chunk.kind == chunk_kind,  # 'summary' (default) or 'fulltext' (--from-fulltext)
        Chunk.id.notin_(already_processed_subq),
    ]
    if doc_id is not None:
        clauses.append(Chunk.document_id == doc_id)
    return clauses


async def _scope_doc_id(session: Any, scope_document_iri: str | None) -> Any:
    if scope_document_iri is None:
        return None
    doc_id = (await session.execute(
        select(Document.id).where(
            Document.document_identifier == scope_document_iri
        )
    )).scalar_one_or_none()
    if doc_id is None:
        raise ValueError(f"document not found: {scope_document_iri}")
    return doc_id


async def _unprocessed_artifact_chunk_count(
    *,
    types: tuple[str, ...],
    chunk_kind: str,
    scope_document_iri: str | None = None,
) -> int:
    """How many chunks `generate_per_chunk_artifacts` would still pick up."""
    async with session_scope() as session:
        doc_id = await _scope_doc_id(session, scope_document_iri)
        stmt = select(func.count()).select_from(Chunk).where(
            *_unprocessed_chunk_filter(types, chunk_kind, doc_id)
        )
        return int((await session.execute(stmt)).scalar() or 0)


async def generate_per_chunk_artifacts(
    *,
    scope_document_iri: str | None = None,
    limit: int | None = None,
    types: tuple[str, ...] = _DEFAULT_PER_CHUNK_TYPES,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
    use_entities: bool = True,
    chunk_kind: str = "summary",
    chunk_offset: int = 0,
    bump_graph_version: bool = True,
) -> ArtifactGenSummary:
    """Drive per-chunk Claim+Finding+Observation extraction.

    `use_entities` (default True): look up each chunk's named entities
    via the Chunk -> viao:assertsAbout -> Entity edges produced by
    Milestone C, then feed them into the LLM prompt so artifact text
    names the actual entities ("BYD Company Ltd.") instead of generic
    terms ("the manufacturer"). Falls back to the old generic prompt
    per-chunk if that chunk has zero entities. Pass --no-entities (CLI)
    to skip the lookup entirely.

    Idempotent: skips chunks that already have ANY of the target
    artifact types attached.

    `chunk_offset` / `bump_graph_version` exist for
    `generate_per_chunk_artifacts_streamed`, which calls this once per batch.
    """
    t0 = time.time()
    summary = ArtifactGenSummary()
    summary.by_type = {t: 0 for t in types}

    async with session_scope() as session:
        doc_id = await _scope_doc_id(session, scope_document_iri)
        stmt = (
            select(Chunk.id, Chunk.chunk_identifier, Chunk.text, Chunk.document_id)
            .where(*_unprocessed_chunk_filter(types, chunk_kind, doc_id))
            # A TOTAL order, so `chunk_offset` names the same chunks from one
            # batch to the next (created_at alone ties within a document).
            .order_by(Chunk.created_at, Chunk.id)
        )
        if chunk_offset:
            stmt = stmt.offset(chunk_offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        chunks = result.all()

    if not chunks:
        print("[generate-artifacts] no chunks to process")
        return summary

    # Safety check + entity preload: refuse to run entity-grounded
    # extraction if NO entities exist at all in the corpus -- that
    # almost always means extract-entities wasn't run yet.
    chunks_to_entities: dict[Any, list[dict[str, str]]] = {}
    if use_entities:
        async with session_scope() as session:
            total_entities = await session.execute(
                select(Entity.id).limit(1)
            )
            if total_entities.first() is None:
                raise RuntimeError(
                    "use_entities=True but graphrag.entities is empty. "
                    "Run `extract-entities` first, or pass --no-entities "
                    "to opt out of entity grounding."
                )
            # Bulk-load (chunk_id -> [{entity_id, canonical_name, class_label}])
            chunk_ids = [c[0] for c in chunks]
            r = await session.execute(
                select(
                    GraphRelationship.source_chunk_id,
                    Entity.id,
                    Entity.name,
                    OntologyClass.label,
                    Entity.extra_metadata,
                )
                .join(Entity, Entity.id == GraphRelationship.target_node_id)
                .join(OntologyClass, OntologyClass.id == Entity.class_id)
                .where(
                    GraphRelationship.predicate_iri == VIAO_ASSERTS_ABOUT,
                    GraphRelationship.relationship_source == "DOCUMENT_EXTRACTION",
                    GraphRelationship.source_chunk_id.in_(chunk_ids),
                )
            )
            for cid, eid, name, label, emeta in r.all():
                chunks_to_entities.setdefault(cid, []).append({
                    "entity_id": eid,
                    "canonical_name": name,
                    "short_name": name,
                    "class_label": label or "",
                    "aliases": _aliases_of(emeta),
                })

    chunks_with_ents = sum(1 for cid, _, _, _ in chunks if chunks_to_entities.get(cid))
    print(
        f"[generate-artifacts] {len(chunks)} chunk(s) to process "
        f"(types={','.join(types)}, concurrency={concurrency}, "
        f"entity-grounded={use_entities}; "
        f"{chunks_with_ents}/{len(chunks)} chunks have >=1 entity)"
    )

    router = LLMRouter()
    cost_before = router.total_cost_usd
    sem = asyncio.Semaphore(concurrency)
    # Each task returns (chunk_id, chunk_iri, doc_id, parsed_dict or None)
    results: list[tuple[Any, str, Any, dict[str, Any] | None]] = [None] * len(chunks)  # type: ignore[list-item]
    cost_limit_hit = asyncio.Event()

    # Progress reporting: print a heartbeat every ~5% of work + every
    # 30s elapsed so the user knows the run is alive on slow APIs.
    progress_state = {
        "done": 0,
        "ok": 0,
        "fail": 0,
        "next_pct": 5,
        "last_print": time.time(),
        "started": time.time(),
    }
    progress_lock = asyncio.Lock()
    progress_step_pct = 5

    async def _report_progress() -> None:
        elapsed = time.time() - progress_state["started"]
        done = progress_state["done"]
        pct = 100 * done / len(chunks)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(chunks) - done) / rate if rate > 0 else 0
        cost_so_far = router.total_cost_usd - cost_before
        print(
            f"[generate-artifacts] progress: "
            f"{done:,}/{len(chunks):,} chunk(s) ({pct:.1f}%), "
            f"ok={progress_state['ok']:,} fail={progress_state['fail']:,}, "
            f"cost=${cost_so_far:.4f}, "
            f"rate={rate:.1f}/s, ETA={eta/60:.1f} min"
        )

    async def _one(idx: int, chunk_id: Any, chunk_iri: str, text: str, doc_id: Any) -> None:
        if cost_limit_hit.is_set():
            return
        async with sem:
            if cost_limit_hit.is_set():
                return
            entities = chunks_to_entities.get(chunk_id, []) if use_entities else []
            if entities:
                system, user = PROMPTS["artifact_chunk_extract_with_entities"](
                    text, entities
                )
                task_name = "artifact_chunk_extract_with_entities"
            else:
                system, user = PROMPTS["artifact_chunk_extract"](text)
                task_name = "artifact_chunk_extract"
            try:
                # Parse-retry: re-ask once on an unparseable response (Anthropic
                # has no JSON-grammar mode; Haiku occasionally malforms JSON).
                parsed = None
                for _attempt in range(2):
                    out = await router.chat(task_name, system=system, user=user)
                    parsed = _extract_json(out.text)
                    if isinstance(parsed, dict):
                        break
            except Exception as exc:
                print(f"[generate-artifacts] chunk {chunk_iri} LLM failed: {exc}")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return
            if not isinstance(parsed, dict):
                print(f"[generate-artifacts] chunk {chunk_iri} unparseable response (after retry)")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return
            results[idx] = (chunk_id, chunk_iri, doc_id, parsed)

            async with progress_lock:
                progress_state["done"] += 1
                progress_state["ok"] += 1
                pct = 100 * progress_state["done"] / len(chunks)
                now = time.time()
                if (pct >= progress_state["next_pct"]
                        or now - progress_state["last_print"] >= 30):
                    await _report_progress()
                    progress_state["last_print"] = now
                    while progress_state["next_pct"] <= pct:
                        progress_state["next_pct"] += progress_step_pct

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
                used_entities = bool(chunks_to_entities.get(chunk_id))
                # Event items carry a date-shaped extra_metadata; the other
                # three types carry evidence_status / claim_source / time_scope.
                # The prompt-version label tracks which prompt produced them.
                if artifact_type == "Event":
                    extra: dict[str, Any] = {
                        "event_date": _normalize_date_field(item.get("event_date")),
                        "event_start_date": _normalize_date_field(
                            item.get("event_start_date")),
                        "event_end_date": _normalize_date_field(
                            item.get("event_end_date")),
                        "event_category": _normalize_str_field(
                            item.get("event_category")),
                    }
                    prompt_version = (
                        "artifact_chunk_extract_with_entities@v3"
                        if used_entities
                        else "artifact_chunk_extract@v2"
                    )
                else:
                    # New (2026-06-13): evidence_status / claim_source /
                    # time_scope metadata captured by the updated
                    # artifact_chunk_extract_with_entities prompt. Stored
                    # in extra_metadata JSONB so deep_research can use them
                    # in the CLAIMS + KEY INSIGHTS sections.
                    raw_ev_status = (item.get("evidence_status") or "").strip().lower()
                    if raw_ev_status not in ("backed", "partial", "unbacked"):
                        raw_ev_status = None  # leave null rather than guess
                    extra = {
                        "evidence_status": raw_ev_status,
                        "claim_source": _normalize_str_field(item.get("claim_source")),
                        "time_scope": _normalize_str_field(item.get("time_scope")),
                    }
                    prompt_version = (
                        "artifact_chunk_extract_with_entities@v3"
                        if used_entities
                        else "artifact_chunk_extract@v2"
                    )
                artifact_payloads.append({
                    "artifact_identifier": airi,
                    "artifact_type": artifact_type,
                    "title": None,
                    "text": text,
                    "confidence": conf,
                    "model_name": "gpt-4o-mini",
                    "prompt_version": prompt_version,
                    "status": "ACTIVE",
                    "graph_version": 0,  # filled in below
                    "extra_metadata": extra,
                })
                artifact_to_chunk.append((airi, chunk_id, doc_id, text))
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
    asserts_about_seen: set[tuple[Any, Any]] = set()
    for airi, chunk_id, doc_id, art_text in artifact_to_chunk:
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

        # Artifact -> viao:assertsAbout -> Entity edges, for the source
        # chunk's entities the artifact's text names by full name or alias
        # (match_artifact_entities). Idempotent within this run via
        # asserts_about_seen.
        if use_entities:
            for ent_id in match_artifact_entities(
                    art_text, chunks_to_entities.get(chunk_id, [])):
                key = (aid, ent_id)
                if key in asserts_about_seen:
                    continue
                asserts_about_seen.add(key)
                edge_payloads.append({
                    "source_node_type": "intelligence_artifact",
                    "source_node_id": aid,
                    "target_node_type": "entity",
                    "target_node_id": ent_id,
                    "predicate_iri": VIAO_ASSERTS_ABOUT,
                    "predicate_label": "viao:assertsAbout",
                    "relationship_type": "assertsAbout",
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

    if bump_graph_version:
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


def _merge_artifact_summaries(
    total: ArtifactGenSummary, batch: ArtifactGenSummary
) -> None:
    """Accumulate a batch's summary into the run total, in place."""
    for f in (
        "chunks_scanned", "chunks_skipped_already_processed", "chunks_failed",
        "artifacts_inserted", "edges_inserted", "sources_inserted",
        "docs_summarized", "llm_cost_usd", "embedding_cost_usd",
    ):
        setattr(total, f, getattr(total, f) + getattr(batch, f))
    for k, v in batch.by_type.items():
        total.by_type[k] = total.by_type.get(k, 0) + v
    if len(total.samples) < 5:
        total.samples.extend(batch.samples[: 5 - len(total.samples)])


async def generate_per_chunk_artifacts_streamed(
    *,
    batch_size: int,
    max_cost_usd: float = 5.0,
    types: tuple[str, ...] = _DEFAULT_PER_CHUNK_TYPES,
    chunk_kind: str = "summary",
    scope_document_iri: str | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> ArtifactGenSummary:
    """`generate_per_chunk_artifacts` in committed batches, resumable after a kill.

    The single-shot path makes every LLM call first and writes once at the end,
    so a kill or a `--max-cost-usd` trip discards the whole run, and memory
    grows with the corpus. Here each batch commits; because selection already
    skips chunks that carry any target artifact type, the NEXT call resumes
    where this one stopped. The graph is the progress marker -- no checkpoint.

    Unlike entity extraction, batching costs nothing in quality: each chunk's
    artifacts are generated independently, with no cross-chunk consolidation.

    A chunk the model returns nothing for (or whose call failed) stays
    "unprocessed". Those are stepped past with an offset rather than
    re-selected, so they are attempted once per run and never starve the
    chunks behind them. Unlike `extract_entities_streamed`, the offset moves by
    the exact number of such chunks in every batch, not only in a batch that
    made no progress at all -- so a PARTIALLY stalled batch does not re-pay for
    its leftovers either.
    """
    t0 = time.time()
    total = ArtifactGenSummary()
    total.by_type = {t: 0 for t in types}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    remaining_chunks = limit
    batch_no = 0
    offset = 0
    stalled = 0
    while True:
        before = await _unprocessed_artifact_chunk_count(
            types=types, chunk_kind=chunk_kind,
            scope_document_iri=scope_document_iri,
        )
        if before <= offset:
            break

        budget = max_cost_usd - (total.llm_cost_usd + total.embedding_cost_usd)
        if budget <= 0:
            print(f"[generate-artifacts] cost cap ${max_cost_usd:.2f} reached "
                  f"after {batch_no} batch(es); {before - offset} chunk(s) left. "
                  f"Re-run to continue -- completed batches are committed.")
            break

        this_batch = batch_size
        if remaining_chunks is not None:
            if remaining_chunks <= 0:
                break
            this_batch = min(this_batch, remaining_chunks)

        batch_no += 1
        print(f"[generate-artifacts] --- batch {batch_no}: up to {this_batch} "
              f"of {before - offset} remaining chunk(s), budget ${budget:.2f} ---")

        batch = await generate_per_chunk_artifacts(
            limit=this_batch,
            chunk_offset=offset,
            max_cost_usd=budget,
            types=types,
            chunk_kind=chunk_kind,
            scope_document_iri=scope_document_iri,
            # Once for the whole run, after the loop -- not per batch.
            bump_graph_version=False,
            **kwargs,
        )
        _merge_artifact_summaries(total, batch)
        attempted = batch.chunks_scanned + batch.chunks_failed
        if remaining_chunks is not None:
            remaining_chunks -= attempted
        if attempted == 0:
            # Nothing selected, or the cost cap tripped before any call
            # finished: stop rather than spin on the same offset.
            break

        after = await _unprocessed_artifact_chunk_count(
            types=types, chunk_kind=chunk_kind,
            scope_document_iri=scope_document_iri,
        )
        left_behind = max(0, attempted - max(0, before - after))
        if left_behind:
            offset += left_behind
            stalled += left_behind
            print(f"[generate-artifacts] batch {batch_no}: {left_behind} "
                  f"chunk(s) produced no artifacts or failed; stepping past "
                  f"them (offset={offset}).")

    if stalled:
        print(f"[generate-artifacts] {stalled} chunk(s) produced no artifacts "
              f"or failed; attempted once and skipped. A re-run retries them.")

    async with session_scope() as session:
        total.new_graph_version = await bump_version(session)
    total.total_cost_usd = total.llm_cost_usd + total.embedding_cost_usd
    total.wall_seconds = time.time() - t0
    print(
        f"[generate-artifacts] STREAMED DONE: {batch_no} batch(es), "
        f"chunks={total.chunks_scanned} ok / {total.chunks_failed} failed, "
        f"artifacts={total.artifacts_inserted} "
        f"({', '.join(f'{t}={n}' for t, n in total.by_type.items())}), "
        f"edges={total.edges_inserted}, "
        f"cost=${total.total_cost_usd:.4f}, "
        f"wall={total.wall_seconds:.1f}s, "
        f"graph_version -> {total.new_graph_version}"
    )
    return total


async def _auto_summary_rollup(max_cost_usd: float) -> float:
    """Automatically roll up Summary artifacts (consolidate similar per-document
    summaries across the corpus). Config-driven layers (summary_rollup_rounds,
    default 2). Returns the rollup's total cost. No-op when rounds <= 0 or there
    is nothing new to cluster (generate_rollups is idempotent)."""
    sum_cfg = get_settings().app_config.get("summarization", {})
    rounds = int(sum_cfg.get("summary_rollup_rounds", 2))
    if rounds <= 0:
        return 0.0
    eval_rounds = int(sum_cfg.get("rollup_eval_rounds", 2))
    from backend.app.services.db_artifact_rollup import generate_rollups
    print(f"[generate-artifacts] auto Summary rollup ({rounds} layer(s), "
          f"loss-loop eval_rounds={eval_rounds}) ...")
    roll = await generate_rollups(
        types=("Summary",), layers=rounds, max_cost_usd=max_cost_usd,
        eval_rounds=eval_rounds,
    )
    return roll.total_cost_usd


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

    progress_state = {
        "done": 0,
        "next_pct": 5,
        "last_print": time.time(),
        "started": time.time(),
    }
    progress_lock = asyncio.Lock()

    async def _one(idx: int, doc_id: Any, diri: str, title: str, text_summary: str | None) -> None:
        if cost_limit_hit.is_set():
            return
        body = text_summary or ""
        if not body.strip():
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
            # Summary artifact = the document summary VERBATIM (no LLM re-summary
            # -- the artifact would be redundant with documents.text_summary).
            # Consolidation across documents happens in the auto 2-round rollup
            # below, not here.
            doc_results[idx] = (doc_id, diri, title, body.strip())

            async with progress_lock:
                progress_state["done"] += 1
                pct = 100 * progress_state["done"] / len(docs)
                now = time.time()
                if (pct >= progress_state["next_pct"]
                        or now - progress_state["last_print"] >= 30):
                    elapsed = now - progress_state["started"]
                    rate = progress_state["done"] / elapsed if elapsed > 0 else 0
                    eta = (len(docs) - progress_state["done"]) / rate if rate > 0 else 0
                    cost_so_far = router.total_cost_usd - cost_before
                    print(
                        f"[generate-artifacts] Summary progress: "
                        f"{progress_state['done']:,}/{len(docs):,} doc(s) "
                        f"({pct:.1f}%), cost=${cost_so_far:.4f}, "
                        f"ETA={eta/60:.1f} min"
                    )
                    progress_state["last_print"] = now
                    while progress_state["next_pct"] <= pct:
                        progress_state["next_pct"] += 5

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
            "model_name": None,
            "prompt_version": "verbatim_text_summary@v1",
            "status": "ACTIVE",
            "graph_version": 0,
            "extra_metadata": {"source_document_iri": diri},
        })
        art_to_doc.append((airi, doc_id))
        embed_texts.append(text)
        summary.by_type["Summary"] += 1

    if not art_payloads:
        # No NEW Summaries this run, but still consolidate any existing ones.
        print("[generate-artifacts] no new Summary artifacts; running auto rollup only")
        summary.total_cost_usd = summary.llm_cost_usd + await _auto_summary_rollup(max_cost_usd)
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

    # ArtifactSource for every chunk of each doc + summarizes edge + entity edges.
    iri_to_text = {p["artifact_identifier"]: (p.get("text") or "") for p in art_payloads}
    source_payloads = []
    edge_payloads = []
    n_entity_edges = 0
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
            # The doc's entities (linked to its chunks via chunk->assertsAbout->entity).
            er = await session.execute(
                sql_text("""
                SELECT DISTINCT e.id, e.name, e.extra_metadata
                  FROM graphrag.entities e
                  JOIN graphrag.graph_relationships gr ON gr.target_node_id = e.id
                   AND gr.predicate_label = 'viao:assertsAbout'
                   AND gr.target_node_type = 'entity'
                  JOIN graphrag.chunks ch ON ch.id = gr.source_node_id
                 WHERE ch.document_id = :doc
                """),
                {"doc": doc_id},
            )
            doc_entities = er.all()
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
        # Summary -> assertsAbout -> entity, for entities whose name appears in the
        # summary text. Makes Summaries (and their rollups, via inheritance) reachable
        # through the entity graph -> they now surface in deep_research, not just
        # artifact_only. Mirrors the per-chunk entity linker.
        seen_ent: set[Any] = set()
        for ent_id in match_artifact_entities(
                iri_to_text.get(airi, ""),
                [{"entity_id": eid, "canonical_name": nm,
                  "aliases": _aliases_of(em)} for eid, nm, em in doc_entities]):
            if ent_id not in seen_ent:
                seen_ent.add(ent_id)
                edge_payloads.append({
                    "source_node_type": "intelligence_artifact",
                    "source_node_id": aid,
                    "target_node_type": "entity",
                    "target_node_id": ent_id,
                    "predicate_iri": VIAO_ASSERTS_ABOUT,
                    "predicate_label": "viao:assertsAbout",
                    "relationship_type": "assertsAbout",
                    "relationship_source": "LLM_INFERENCE",
                    "is_authoritative": True,
                    "source_chunk_id": None,
                    "source_document_id": doc_id,
                    "source_artifact_id": aid,
                    "graph_version": gv,
                    "extra_metadata": {},
                })
                n_entity_edges += 1

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
        f"docs={summary.docs_summarized} (verbatim, no re-summary), "
        f"sources={summary.sources_inserted}, summarizes+entity edges="
        f"{summary.edges_inserted} (of which assertsAbout->entity={n_entity_edges}), "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )

    # Auto rollup over Summary artifacts (consolidate similar per-document
    # summaries across the corpus). Config-driven layers (default 2).
    summary.total_cost_usd += await _auto_summary_rollup(max_cost_usd)

    return summary


async def relink_artifact_entities(*, dry_run: bool = False) -> dict[str, int]:
    """Add the artifact -> viao:assertsAbout -> entity edges the alias-aware
    matcher finds on an EXISTING build. No LLM calls, no cost.

    Scope mirrors generation: a per-chunk artifact is matched against its
    source chunk's entities; a Summary against its document's entities.
    Additive only -- existing edges are kept, and rollups (which inherit their
    children's edges) are not touched.
    """
    async with session_scope() as session:
        ent_rows = (await session.execute(sql_text("""
            SELECT g.source_chunk_id, c.document_id, e.id, e.name, e.extra_metadata
              FROM graphrag.graph_relationships g
              JOIN graphrag.entities e ON e.id = g.target_node_id
              JOIN graphrag.chunks c ON c.id = g.source_chunk_id
             WHERE g.predicate_label = 'viao:assertsAbout'
               AND g.source_node_type = 'chunk' AND g.target_node_type = 'entity'
        """))).all()
        chunk_arts = (await session.execute(sql_text("""
            SELECT a.id, a.text, g.target_node_id, c.document_id
              FROM graphrag.intelligence_artifacts a
              JOIN graphrag.graph_relationships g
                ON g.source_node_id = a.id AND g.predicate_label = 'viao:derivedFromChunk'
              JOIN graphrag.chunks c ON c.id = g.target_node_id
        """))).all()
        doc_arts = (await session.execute(sql_text("""
            SELECT a.id, a.text, g.target_node_id
              FROM graphrag.intelligence_artifacts a
              JOIN graphrag.graph_relationships g
                ON g.source_node_id = a.id AND g.predicate_label = 'viao:summarizes'
        """))).all()
        have = {(a, e) for a, e in (await session.execute(sql_text("""
            SELECT source_node_id, target_node_id FROM graphrag.graph_relationships
             WHERE source_node_type = 'intelligence_artifact'
               AND predicate_label = 'viao:assertsAbout' AND target_node_type = 'entity'
        """))).all()}
        gv = await current_version(session)

    by_chunk: dict[Any, dict[Any, dict[str, Any]]] = {}
    by_doc: dict[Any, dict[Any, dict[str, Any]]] = {}
    for chunk_id, doc_id, eid, name, emeta in ent_rows:
        ent = {"entity_id": eid, "canonical_name": name, "aliases": _aliases_of(emeta)}
        by_chunk.setdefault(chunk_id, {})[eid] = ent
        by_doc.setdefault(doc_id, {})[eid] = ent

    new_edges: list[dict[str, Any]] = []
    def _edge(aid: Any, eid: Any, chunk_id: Any, doc_id: Any) -> None:
        if (aid, eid) in have:
            return
        have.add((aid, eid))
        new_edges.append({
            "source_node_type": "intelligence_artifact", "source_node_id": aid,
            "target_node_type": "entity", "target_node_id": eid,
            "predicate_iri": VIAO_ASSERTS_ABOUT,
            "predicate_label": "viao:assertsAbout",
            "relationship_type": "assertsAbout",
            "relationship_source": "LLM_INFERENCE", "is_authoritative": True,
            "source_chunk_id": chunk_id, "source_document_id": doc_id,
            "source_artifact_id": aid, "graph_version": gv, "extra_metadata": {},
        })
    for aid, text, chunk_id, doc_id in chunk_arts:
        for eid in match_artifact_entities(text or "", list(by_chunk.get(chunk_id, {}).values())):
            _edge(aid, eid, chunk_id, doc_id)
    for aid, text, doc_id in doc_arts:
        for eid in match_artifact_entities(text or "", list(by_doc.get(doc_id, {}).values())):
            _edge(aid, eid, None, doc_id)

    if new_edges and not dry_run:
        async with session_scope() as session:
            for i in range(0, len(new_edges), 500):
                await session.execute(
                    pg_insert(GraphRelationship).values(new_edges[i: i + 500]))
    out = {"artifacts_checked": len(chunk_arts) + len(doc_arts),
           "edges_added": len(new_edges), "dry_run": int(dry_run)}
    print(f"[relink-artifact-entities] checked {out['artifacts_checked']} artifact(s); "
          f"{'would add' if dry_run else 'added'} {len(new_edges)} artifact->entity edge(s)")
    return out


async def regenerate_stale_artifacts(
    *,
    dry_run: bool = False,
    types: tuple[str, ...] = _DEFAULT_PER_CHUNK_TYPES,
    concurrency: int = 4,
    max_cost_usd: float = 10.0,
    retire: bool = True,
) -> dict[str, Any]:
    """Regenerate artifacts for documents whose OLD versions left STALE artifacts.

    A soft-delete or update marks an artifact STALE when its ENTIRE evidence base
    was in the affected document (mixed-source artifacts stay ACTIVE). Those STALE
    rows are excluded from retrieval but otherwise linger. This command:

      1. finds STALE artifacts and traces each to its origin document (via its
         source chunks' document_id);
      2. finds the ACTIVE successor document -- the one that SUPERSEDED the origin
         (i.e. the doc was UPDATED, not merely deleted);
      3. regenerates artifacts SCOPED to each successor, reusing the normal
         per-chunk + summary generators (idempotent -- already-covered chunks are
         skipped); and
      4. retires the STALE rows (status -> DELETED) so they stop cluttering.

    STALE artifacts from a plain delete (no successor version) have nothing to
    regenerate from -- they are simply retired.

    With dry_run=True, reports what it WOULD do and writes nothing.
    """
    # 1 + 2: trace stale artifacts -> origin docs -> ACTIVE successors.
    async with session_scope() as session:
        stale_rows = await session.execute(
            select(IntelligenceArtifact.id).where(IntelligenceArtifact.status == "STALE")
        )
        stale_ids = [r[0] for r in stale_rows.all()]
        if not stale_ids:
            print("[regenerate-stale] no STALE artifacts -- nothing to do.")
            return {"stale": 0, "successors": 0, "regenerated_docs": [], "retired": 0}

        origin_rows = await session.execute(
            select(Chunk.document_id)
            .join(ArtifactSource, ArtifactSource.chunk_id == Chunk.id)
            .where(ArtifactSource.artifact_id.in_(stale_ids))
            .distinct()
        )
        origin_doc_ids = [r[0] for r in origin_rows.all() if r[0] is not None]

        successor_iris: list[str] = []
        if origin_doc_ids:
            succ_rows = await session.execute(
                select(Document.document_identifier).where(
                    Document.supersedes_document_id.in_(origin_doc_ids),
                    Document.status == "ACTIVE",
                )
            )
            successor_iris = [r[0] for r in succ_rows.all()]

    print(
        f"[regenerate-stale] {len(stale_ids)} STALE artifact(s); "
        f"{len(successor_iris)} updated document(s) to regenerate from."
    )
    if dry_run:
        print("[regenerate-stale] dry-run: no writes.")
        return {
            "stale": len(stale_ids),
            "successors": len(successor_iris),
            "regenerated_docs": successor_iris,
            "retired": 0,
        }

    # 3: regenerate artifacts scoped to each ACTIVE successor (reuse tested paths).
    for iri in successor_iris:
        await generate_per_chunk_artifacts(
            scope_document_iri=iri, types=types,
            concurrency=concurrency, max_cost_usd=max_cost_usd,
        )
        await generate_document_summaries(
            scope_document_iri=iri, concurrency=concurrency, max_cost_usd=max_cost_usd,
        )

    # 4: retire the STALE tombstones (already excluded from retrieval; this clears
    # them so they don't accumulate). Status flip only -- reversible, no FK games.
    retired = 0
    if retire:
        async with session_scope() as session:
            await session.execute(
                sql_update(IntelligenceArtifact)
                .where(IntelligenceArtifact.id.in_(stale_ids))
                .values(status="DELETED")
            )
        retired = len(stale_ids)
        print(f"[regenerate-stale] retired {retired} STALE artifact(s) -> DELETED.")

    return {
        "stale": len(stale_ids),
        "successors": len(successor_iris),
        "regenerated_docs": successor_iris,
        "retired": retired,
    }

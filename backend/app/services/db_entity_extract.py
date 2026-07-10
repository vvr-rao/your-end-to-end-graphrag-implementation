"""Milestone C: extract named entities per chunk.

Per chunk:
  1. Find top-K candidate ontology classes via vector search on the
     chunk embedding.
  2. LLM call (`entity_extract` task, gpt-4o-mini, JSON mode): given
     the chunk text + candidate class list, return entities with
     {canonical_name, short_name, class_iri, confidence}. Validator
     rejects any class_iri not in the candidate list.
  3. For each surviving entity:
     - Normalize name (lowercase + strip punctuation).
     - pg_trgm fuzzy-match against existing rows with the same class.
       similarity >= 0.85 -> reuse the existing entity_id.
       no match -> INSERT a new row + embed (name + class label).
  4. Edges to write (DOCUMENT_EXTRACTION provenance):
     - Chunk -> viao:assertsAbout -> Entity     (always)
     - Entity -> rdf:type -> OntologyClass      (once per entity, on first mint)
  5. Bump graph_version.

Idempotent: skips chunks that already have any viao:assertsAbout edge
from DOCUMENT_EXTRACTION.
Generic: no corpus-specific assumptions; works on any ingested corpus.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select, text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.entities import Entity
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.models.ontology import OntologyClass
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import (
    RDF_TYPE,
    VIAO_ASSERTS_ABOUT,
)
from backend.app.services.prompts import PROMPTS

_ENTITIES_NS = "https://veerla-ramrao.ai/ontology/entities"


@dataclass
class EntityExtractSummary:
    chunks_scanned: int = 0
    chunks_skipped_already: int = 0
    chunks_failed: int = 0
    entities_minted: int = 0
    entities_reused: int = 0
    chunk_entity_edges: int = 0
    type_edges: int = 0
    tables_scanned: int = 0
    table_entity_edges: int = 0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _normalize_name(name: str) -> str:
    """Lowercase + strip non-alphanumerics (except internal spaces)."""
    s = re.sub(r"[^\w\s]", "", name, flags=re.UNICODE).strip().lower()
    return re.sub(r"\s+", " ", s)


# Corporate / legal-entity suffixes commonly appended to organization
# canonical names. Stripping them yields the "short form" the entity is
# usually referred to in tables and shorthand mentions ("BYD" rather
# than "BYD Company Ltd."). Matched as trailing tokens against the
# already-normalized name (post `_normalize_name`), so each entry is
# lowercase, no punctuation, no leading space.
_CORPORATE_SUFFIX_TOKENS: tuple[str, ...] = (
    # Compound suffixes first so they match before their single-word components.
    "co ltd", "company ltd", "company limited", "holdings limited",
    "holdings inc", "holdings group", "holdings ltd", "group plc",
    "group inc", "group ltd", "incorporated", "corporation",
    # Then single tokens.
    "inc", "ltd", "limited", "corp", "company", "co",
    "holdings", "holding", "group", "plc",
    "ag", "gmbh", "sa", "nv", "spa", "llc", "llp", "lp",
    "se", "asa", "ab", "oy", "kk", "kabushiki",
    "pty", "bv", "kg",
)


def _short_form_variants(normalized_name: str) -> list[str]:
    """Return the set of normalized variants an entity might appear as.

    Always includes the input `normalized_name` itself. Repeatedly strips
    trailing corporate-suffix tokens (e.g. "byd company ltd" -> "byd
    company" -> "byd") and adds each intermediate form to the set.

    Stops shrinking once the remaining string is too short (< 3 chars or
    < 1 token) to be a safe table-match candidate.

    Examples:
      "byd company ltd"            -> ["byd company ltd", "byd company", "byd"]
      "tesla inc"                  -> ["tesla inc", "tesla"]
      "saudi aramco"               -> ["saudi aramco"]
      "ford motor company"         -> ["ford motor company", "ford motor"]
      "general motors company"     -> ["general motors company", "general motors"]
      "mercedesbenz"               -> ["mercedesbenz"]      (no suffix)
    """
    base = normalized_name.strip()
    if not base:
        return []
    variants: list[str] = [base]
    seen: set[str] = {base}
    while True:
        stripped: str | None = None
        for suf in _CORPORATE_SUFFIX_TOKENS:
            tail = " " + suf
            if base.endswith(tail):
                cand = base[: -len(tail)].strip()
                if len(cand) >= 3 and " " in (" " + cand):
                    stripped = cand
                    break
        if stripped is None or stripped in seen:
            break
        variants.append(stripped)
        seen.add(stripped)
        base = stripped
    return variants


# ---------------------------------------------------------------------------
# Phase 2a v2 -- table-to-entity linking.
#
# After the per-chunk entity-mining pass finishes, we walk every ACTIVE
# StructuredTable artifact, scan its caption + row labels + cell values
# for strings that match an entity by normalized name, and emit
# `Table -> viao:assertsAbout -> Entity` edges. This makes tables
# first-class graph citizens for the entity-anchored BFS that Phase 2's
# retrieval modes rely on. No LLM calls; pure DB + Python.
# ---------------------------------------------------------------------------

# Candidate-string filter: drop strings that can't possibly be entity
# names (pure numbers, short tokens, generic table-section words). Keeps
# everything else for the normalized-name match.
_NUMERIC_CANDIDATE_RE = re.compile(
    r"^[\s\-\+\$€£¥₹%()0-9,.–—a-z]+$",
    re.IGNORECASE,
)
_DATE_LIKE_RE = re.compile(
    r"^(?:Q[1-4]\s+)?(?:FY|fy|cy|CY|H[12]\s+)?(?:19|20)\d{2}\b",
)

# Generic financial-table noise that should NEVER match an entity even if
# someone unfortunately named their entity that. Conservative -- prefer
# false-negatives (miss-link a row labeled "Total") over false-positives.
_GENERIC_TABLE_TOKENS = frozenset(s.lower() for s in (
    "total", "subtotal", "grand total", "other", "others", "all",
    "balance", "ending balance", "opening balance", "beginning balance",
    "average", "weighted average", "sum", "n/a", "na", "nil", "none",
    "amount", "amounts", "value", "values", "rate", "share", "shares",
    "year", "years", "fy", "cy", "quarter", "month", "day",
    "current", "previous", "prior", "next", "last", "first", "second",
    "third", "fourth", "fifth", "annual", "interim", "ttm",
    "increase", "decrease", "change", "variance",
    "yes", "no", "true", "false", "applicable", "not applicable",
    "above", "below", "see notes", "see note", "note", "notes",
))


def _filter_candidate(s: str | None) -> bool:
    """Return True if `s` could plausibly be an entity-name reference.

    Drops: empty / very short strings, pure numbers, currency / percent
    values, dates / year prefixes, common table-section noise."""
    if not isinstance(s, str):
        return False
    cleaned = s.strip()
    if len(cleaned) < 3:
        return False
    lower = cleaned.lower()
    if lower in _GENERIC_TABLE_TOKENS:
        return False
    if _DATE_LIKE_RE.match(cleaned):
        return False
    if _NUMERIC_CANDIDATE_RE.match(cleaned):
        # Re-check: the regex is permissive (allows letters); demand the
        # string contains at least 3 letter characters to count as text.
        n_letters = sum(1 for c in cleaned if c.isalpha())
        if n_letters < 3:
            return False
    return True


def _collect_table_candidates(payload: Any) -> list[str]:
    """Pull every plausibly-entity-name string out of a StructuredTable
    JSON-LD payload. Order-preserving, deduplicated by normalized form."""
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: Any) -> None:
        if not _filter_candidate(raw):
            return
        norm = _normalize_name(raw)
        if not norm or norm in seen:
            return
        seen.add(norm)
        out.append(raw.strip())

    # Caption
    _push(payload.get("caption"))
    # Row labels
    rows = payload.get("rows")
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            _push(r.get("rowLabel"))
            cells = r.get("cells")
            if isinstance(cells, list):
                for c in cells:
                    if isinstance(c, dict):
                        _push(c.get("cellValue"))
    return out


def _entity_iri(canonical_name: str, class_iri: str) -> str:
    """Stable IRI: kebab-slug of the name + 16-char hash of (name, class)."""
    digest = hashlib.sha256(
        (canonical_name + "|" + class_iri).encode("utf-8")
    ).hexdigest()[:16]
    slug = re.sub(r"[^\w]+", "-", canonical_name.lower(), flags=re.UNICODE).strip("-")
    slug = slug[:50] if slug else "entity"
    return f"{_ENTITIES_NS}#{slug}-{digest}"


async def extract_entities(
    *,
    scope_document_iri: str | None = None,
    limit: int | None = None,
    candidate_classes_per_chunk: int = 50,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
    chunk_kind: str = "summary",
) -> EntityExtractSummary:
    """Drive entity extraction over chunks that haven't been processed.

    `chunk_kind` ('summary' default | 'fulltext'): which chunk set to mine.
    Summary is cheap but only captures what survived summarization; 'fulltext'
    mines the verbatim chunks (far more complete, e.g. every clinical study),
    at ~18x the LLM calls + more DB rows. Requires --full-text-chunks at ingest.
    """
    t0 = time.time()
    summary = EntityExtractSummary()

    # Select chunks not yet processed.
    async with session_scope() as session:
        already_subq = select(GraphRelationship.source_chunk_id).where(
            GraphRelationship.predicate_iri == VIAO_ASSERTS_ABOUT,
            GraphRelationship.relationship_source == "DOCUMENT_EXTRACTION",
            GraphRelationship.source_chunk_id.isnot(None),
        )
        stmt = (
            select(Chunk.id, Chunk.chunk_identifier, Chunk.text, Chunk.embedding, Chunk.document_id)
            .where(
                Chunk.status == "ACTIVE",
                Chunk.kind == chunk_kind,  # 'summary' (default) or 'fulltext' (--from-fulltext)
                Chunk.id.notin_(already_subq),
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
        print("[extract-entities] no chunks to process")
        return summary

    print(
        f"[extract-entities] {len(chunks)} chunk(s) to process "
        f"(top-{candidate_classes_per_chunk} candidate classes per chunk, "
        f"concurrency={concurrency})"
    )

    router = LLMRouter()
    cost_before = router.total_cost_usd
    sem = asyncio.Semaphore(concurrency)
    # Each task returns (chunk_id, chunk_iri, doc_id, list[entity_dict] or None)
    results: list[tuple[Any, str, Any, list[dict[str, Any]] | None]] = (
        [None] * len(chunks)  # type: ignore[list-item]
    )
    cost_limit_hit = asyncio.Event()

    # Progress reporting -- mirrors generate-artifacts pattern.
    progress_state = {
        "done": 0, "ok": 0, "fail": 0, "next_pct": 5,
        "last_print": time.time(), "started": time.time(),
    }
    progress_lock = asyncio.Lock()

    async def _report_progress() -> None:
        elapsed = time.time() - progress_state["started"]
        done = progress_state["done"]
        pct = 100 * done / len(chunks)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(chunks) - done) / rate if rate > 0 else 0
        cost = router.total_cost_usd - cost_before
        print(
            f"[extract-entities] progress: "
            f"{done:,}/{len(chunks):,} chunk(s) ({pct:.1f}%), "
            f"ok={progress_state['ok']:,} fail={progress_state['fail']:,}, "
            f"cost=${cost:.4f}, rate={rate:.1f}/s, ETA={eta/60:.1f} min"
        )

    async def _candidate_classes(chunk_embedding: list[float]) -> list[dict[str, str]]:
        """Top-K class IRIs nearest the chunk's embedding."""
        async with session_scope() as session:
            r = await session.execute(
                select(
                    OntologyClass.iri,
                    OntologyClass.label,
                    OntologyClass.description,
                )
                .where(OntologyClass.embedding.isnot(None))
                .order_by(OntologyClass.embedding.l2_distance(chunk_embedding))
                .limit(candidate_classes_per_chunk)
            )
            return [
                {"iri": iri, "label": label or "", "description": descr or ""}
                for iri, label, descr in r.all()
            ]

    async def _one(idx: int, chunk_id: Any, chunk_iri: str, txt: str,
                   chunk_emb: list[float], doc_id: Any) -> None:
        if cost_limit_hit.is_set():
            return
        async with sem:
            if cost_limit_hit.is_set():
                return
            try:
                candidates = await _candidate_classes(chunk_emb)
                if not candidates:
                    summary.chunks_failed += 1
                    async with progress_lock:
                        progress_state["done"] += 1
                        progress_state["fail"] += 1
                    return
                cand_iris = {c["iri"] for c in candidates}

                system, user = PROMPTS["entity_extract"](txt, candidates)
                # Parse-retry: Anthropic has no JSON-grammar mode, so Haiku
                # occasionally emits an unparseable response. Re-ask once
                # (non-deterministic -> a retry usually parses) before failing.
                parsed = None
                for _attempt in range(2):
                    out = await router.chat("entity_extract", system=system, user=user)
                    parsed = _extract_json(out.text)
                    if isinstance(parsed, dict):
                        break
            except Exception as exc:
                print(f"[extract-entities] chunk {chunk_iri} call failed: {exc}")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return

            if not isinstance(parsed, dict):
                print(f"[extract-entities] chunk {chunk_iri} unparseable response (after retry)")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return

            raw_entities = parsed.get("entities") or []
            kept: list[dict[str, Any]] = []
            for e in raw_entities:
                if not isinstance(e, dict):
                    continue
                name = (e.get("canonical_name") or "").strip()
                short = (e.get("short_name") or name).strip()
                class_iri = (e.get("class_iri") or "").strip()
                if not name or not class_iri or class_iri not in cand_iris:
                    continue
                try:
                    conf = float(e.get("confidence")) if e.get("confidence") is not None else None
                except (TypeError, ValueError):
                    conf = None
                kept.append({
                    "canonical_name": name,
                    "short_name": short,
                    "class_iri": class_iri,
                    "confidence": conf,
                })
            results[idx] = (chunk_id, chunk_iri, doc_id, kept)

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
                        progress_state["next_pct"] += 5

            if router.total_cost_usd - cost_before > max_cost_usd:
                if not cost_limit_hit.is_set():
                    cost_limit_hit.set()
                    print(
                        f"[extract-entities] HALT: cost ceiling "
                        f"${max_cost_usd:.2f} reached"
                    )

    await asyncio.gather(*[
        _one(i, cid, ciri, txt, emb, did)
        for i, (cid, ciri, txt, emb, did) in enumerate(chunks)
    ])
    summary.llm_cost_usd = router.total_cost_usd - cost_before
    summary.chunks_scanned = sum(1 for r in results if r is not None)
    print(
        f"[extract-entities] LLM done: ${summary.llm_cost_usd:.4f}, "
        f"{summary.chunks_scanned} success / {summary.chunks_failed} failed"
    )

    # Build the class_iri -> class_id map for everything we saw.
    all_class_iris: set[str] = set()
    for tup in results:
        if tup is None:
            continue
        for e in tup[3] or []:
            all_class_iris.add(e["class_iri"])

    async with session_scope() as session:
        r = await session.execute(
            select(OntologyClass.id, OntologyClass.iri, OntologyClass.label)
            .where(OntologyClass.iri.in_(all_class_iris))
        )
        class_id_by_iri: dict[str, Any] = {}
        class_label_by_iri: dict[str, str] = {}
        for cid, ciri, clabel in r.all():
            class_id_by_iri[ciri] = cid
            class_label_by_iri[ciri] = clabel or ""

    # Dedup + mint. EXACT in-memory dedup against a one-shot preload of existing
    # entities; the pg_trgm fuzzy DB query is only a FALLBACK for exact-misses,
    # and only when the table already had entities. A fresh run needs no
    # cross-run dedup, so it does ZERO fuzzy queries. This replaces the old
    # per-entity fuzzy query -- thousands of sequential pooler round-trips on one
    # long-held connection that hung on --from-fulltext.
    seen_in_this_run: dict[tuple[str, Any], Any] = {}  # (normalized_name, class_id) -> entity_id
    fresh_to_embed: list[str] = []              # embed text, index-parallel to entity_mints
    entity_mints: list[dict[str, Any]] = []     # payloads waiting for INSERT (parallel to fresh_to_embed)
    type_edge_keys: set[tuple[Any, Any]] = set()  # (entity_id, class_id) for type edges
    minted_entity_class_pairs: list[tuple[Any, Any]] = []  # for type edges
    chunk_entity_pairs: list[tuple[Any, tuple[str, Any], Any]] = []  # (chunk_id, key, doc_id)
    samples_buf: list[dict[str, Any]] = []

    # Preload existing (normalized_name, class_id) -> id for O(1) exact match.
    # Cheap: strings + ids (~250 bytes/entity), NOT embeddings.
    existing_exact: dict[tuple[str, Any], Any] = {}
    async with session_scope() as session:
        r = await session.execute(
            select(Entity.id, Entity.normalized_name, Entity.class_id)
        )
        for eid, nname, cid in r.all():
            existing_exact[(nname, cid)] = eid
    had_existing = bool(existing_exact)

    async def _fuzzy_existing(normalized: str, class_id: Any) -> Any | None:
        """pg_trgm fuzzy match against the DB in a SHORT-lived session (no
        long-held connection). Only called for exact-misses when the table
        already had entities."""
        async with session_scope() as session:
            r = await session.execute(
                sql_text("""
                    SELECT id FROM graphrag.entities
                     WHERE class_id = :cls
                       AND similarity(normalized_name, :nrm) >= 0.85
                     ORDER BY similarity(normalized_name, :nrm) DESC
                     LIMIT 1
                """),
                {"cls": class_id, "nrm": normalized},
            )
            return r.scalar_one_or_none()

    for tup in results:
        if tup is None:
            continue
        chunk_id, chunk_iri, doc_id, kept = tup
        if not kept:
            continue
        for e in kept:
            class_id = class_id_by_iri.get(e["class_iri"])
            if class_id is None:
                continue
            normalized = _normalize_name(e["canonical_name"])
            if not normalized:
                continue
            key = (normalized, class_id)
            if key in seen_in_this_run:
                chunk_entity_pairs.append((chunk_id, key, doc_id))
                continue
            # Exact match against preloaded existing entities.
            hit = existing_exact.get(key)
            if hit is not None:
                seen_in_this_run[key] = hit
                summary.entities_reused += 1
                chunk_entity_pairs.append((chunk_id, key, doc_id))
                continue
            # Fuzzy fallback -- only when the table already had entities.
            if had_existing:
                match = await _fuzzy_existing(normalized, class_id)
                if match is not None:
                    seen_in_this_run[key] = match
                    existing_exact[key] = match
                    summary.entities_reused += 1
                    chunk_entity_pairs.append((chunk_id, key, doc_id))
                    continue
            # Mint new.
            eiri = _entity_iri(e["canonical_name"], e["class_iri"])
            entity_mints.append({
                "entity_identifier": eiri,
                "name": e["canonical_name"],
                "normalized_name": normalized,
                "class_id": class_id,
                "iri": eiri,
                "status": "ACTIVE",
                "extra_metadata": {
                    "first_seen_in_chunk": chunk_iri,
                    "first_confidence": e["confidence"],
                },
            })
            fresh_to_embed.append(
                f"{e['canonical_name']} -- {class_label_by_iri.get(e['class_iri'], '')}"
            )
            seen_in_this_run[key] = None  # backfilled with the real id after INSERT
            summary.entities_minted += 1
            if len(samples_buf) < 10:
                samples_buf.append({
                    "name": e["canonical_name"],
                    "class": class_label_by_iri.get(e["class_iri"], ""),
                    "chunk": chunk_iri,
                })
            chunk_entity_pairs.append((chunk_id, key, doc_id))

    # Stream embed -> insert -> discard per batch: never hold all entity vectors
    # in RAM (bounds peak memory), and retry transient pooler drops so one
    # dropped connection doesn't lose the whole run.
    embedder = Embedder()
    iri_to_id: dict[str, Any] = {}
    _EBATCH = 200
    for i in range(0, len(entity_mints), _EBATCH):
        batch = entity_mints[i : i + _EBATCH]
        texts = fresh_to_embed[i : i + _EBATCH]
        vecs = await embedder.embed(texts) if texts else []
        for p, v in zip(batch, vecs, strict=False):
            p["embedding"] = v
        for _attempt in range(4):
            try:
                async with session_scope() as session:
                    await session.execute(pg_insert(Entity).values(batch))
                    r = await session.execute(
                        select(Entity.id, Entity.entity_identifier).where(
                            Entity.entity_identifier.in_(
                                [p["entity_identifier"] for p in batch]
                            )
                        )
                    )
                    for eid, iri in r.all():
                        iri_to_id[iri] = eid
                break
            except Exception as _exc:
                if _attempt == 3:
                    raise
                await asyncio.sleep(2 ** _attempt)
        for p in batch:
            p.pop("embedding", None)  # free the vector once persisted
    summary.embedding_cost_usd = embedder.total_cost_usd
    print(
        f"[extract-entities] embedded + inserted {len(entity_mints)} new "
        f"entity(ies): ${summary.embedding_cost_usd:.4f}"
    )

    # Backfill seen_in_this_run with the real IDs + collect type-edge pairs.
    for p in entity_mints:
        key = (p["normalized_name"], p["class_id"])
        eid = iri_to_id.get(p["entity_identifier"])
        if eid is None:
            continue
        seen_in_this_run[key] = eid
        minted_entity_class_pairs.append((eid, p["class_id"]))

    # Build edges.
    async with session_scope() as session:
        gv = await current_version(session)

    edge_payloads: list[dict[str, Any]] = []
    chunk_entity_seen: set[tuple[Any, Any]] = set()
    for chunk_id, key, doc_id in chunk_entity_pairs:
        entity_id = seen_in_this_run.get(key)
        if entity_id is None:
            continue
        sig = (chunk_id, entity_id)
        if sig in chunk_entity_seen:
            continue
        chunk_entity_seen.add(sig)
        edge_payloads.append({
            "source_node_type": "chunk",
            "source_node_id": chunk_id,
            "target_node_type": "entity",
            "target_node_id": entity_id,
            "predicate_iri": VIAO_ASSERTS_ABOUT,
            "predicate_label": "viao:assertsAbout",
            "relationship_type": "assertsAbout",
            "relationship_source": "DOCUMENT_EXTRACTION",
            "is_authoritative": True,
            "source_chunk_id": chunk_id,
            "source_document_id": doc_id,
            "source_artifact_id": None,
            "graph_version": gv,
            "extra_metadata": {},
        })
    summary.chunk_entity_edges = len(edge_payloads)

    # Type edges: one per minted entity.
    type_edge_payloads = []
    for entity_id, class_id in minted_entity_class_pairs:
        if (entity_id, class_id) in type_edge_keys:
            continue
        type_edge_keys.add((entity_id, class_id))
        type_edge_payloads.append({
            "source_node_type": "entity",
            "source_node_id": entity_id,
            "target_node_type": "ontology_class",
            "target_node_id": class_id,
            "predicate_iri": RDF_TYPE,
            "predicate_label": "rdf:type",
            "relationship_type": "instanceOf",
            "relationship_source": "DOCUMENT_EXTRACTION",
            "is_authoritative": True,
            "source_chunk_id": None,
            "source_document_id": None,
            "source_artifact_id": None,
            "graph_version": gv,
            "extra_metadata": {},
        })
    summary.type_edges = len(type_edge_payloads)

    EDGE_BATCH = 500
    async with session_scope() as session:
        for i in range(0, len(edge_payloads), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(edge_payloads[i : i + EDGE_BATCH])
            )
        for i in range(0, len(type_edge_payloads), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(type_edge_payloads[i : i + EDGE_BATCH])
            )

    # Phase 2a v2: link StructuredTable artifacts to entities by name. Runs
    # AFTER the per-chunk entity-mining pass so all just-minted entities
    # are visible. Idempotent (ON CONFLICT DO NOTHING) -- safe to re-run
    # on existing corpora to backfill linkage.
    await _link_tables_to_entities(summary)

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    summary.samples = samples_buf

    print(
        f"[extract-entities] DONE: "
        f"entities (minted={summary.entities_minted}, "
        f"reused={summary.entities_reused}), "
        f"chunk_entity_edges={summary.chunk_entity_edges}, "
        f"type_edges={summary.type_edges}, "
        f"tables_scanned={summary.tables_scanned}, "
        f"table_entity_edges={summary.table_entity_edges}, "
        f"cost=${summary.total_cost_usd:.4f}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )
    return summary


async def _link_tables_to_entities(summary: EntityExtractSummary) -> None:
    """Phase 2a v2: write `Table -> viao:assertsAbout -> Entity` edges
    for every ACTIVE StructuredTable whose JSON-LD payload mentions a
    known entity by name.

    Strategy:
      1. Pre-load the full entity name -> id dict (1 query).
      2. Pre-load all ACTIVE StructuredTable rows + their JSONB payloads.
      3. For each table, walk caption + rowLabels + cellValues; normalize
         each candidate and look up in the dict.
      4. Batch-insert the edges with ON CONFLICT DO NOTHING.

    Pure DB + Python. No LLM calls. Safe to run as the final step of
    `extract_entities` -- if no tables exist, returns immediately."""
    async with session_scope() as session:
        # Build the normalized-name lookup dict in TWO PASSES, preferring
        # canonical names over derived short-form variants whenever a
        # collision occurs. Each entity contributes (a) its full
        # normalized name (the canonical form) and (b) short-form
        # variants derived by stripping trailing corporate suffixes
        # (Inc, Ltd, Corp, Co, Holdings, PLC, AG, GmbH, ...). This lets
        # a table cell saying "BYD" link to the entity whose
        # canonical_name is "BYD Company Ltd." while still allowing
        # "BYD" to remain matchable for an entity literally named "BYD".
        #
        # Pass 1: register every entity's full canonical normalized name.
        #         On collision (two entities sharing the same canonical
        #         name), the first writer wins. Rare; usually means the
        #         entity-extraction pass already collapsed them.
        # Pass 2: register short-form variants ONLY when the key isn't
        #         already taken by a canonical name in pass 1. On
        #         variant-vs-variant collision (two entities deriving
        #         the same short form), drop the variant entirely so
        #         neither matches on it -- each entity is still
        #         reachable via its full canonical name from pass 1.
        ent_rows = await session.execute(
            select(Entity.id, Entity.normalized_name).where(
                Entity.status == "ACTIVE"
            )
        )
        all_rows = [
            (eid, norm) for eid, norm in ent_rows.all()
            if isinstance(norm, str) and norm
        ]
        ent_by_norm: dict[str, Any] = {}
        canonical_keys: set[str] = set()
        ambiguous_variants: set[str] = set()

        # Pass 1: canonical names. Don't overwrite (first wins).
        for entity_id, norm in all_rows:
            if norm not in ent_by_norm:
                ent_by_norm[norm] = entity_id
            canonical_keys.add(norm)

        # Pass 2: short-form variants. Skip any key already claimed by
        # a canonical name. Drop variant-vs-variant collisions.
        for entity_id, norm in all_rows:
            variants = _short_form_variants(norm)
            for v in variants:
                if v == norm:
                    continue  # canonical, already handled in pass 1
                if v in canonical_keys:
                    continue  # never override a canonical name
                if v in ambiguous_variants:
                    continue
                existing = ent_by_norm.get(v)
                if existing is None:
                    ent_by_norm[v] = entity_id
                elif existing != entity_id:
                    # Two distinct entities derive the same short form.
                    # Drop it from the lookup; each entity is still
                    # reachable via its full canonical name.
                    del ent_by_norm[v]
                    ambiguous_variants.add(v)

        if not ent_by_norm:
            print(
                "[link-tables] no entities in DB; skipping table->entity "
                "linkage pass"
            )
            return
        n_short_keys = len(ent_by_norm) - len(canonical_keys)
        print(
            f"[link-tables] entity lookup: {len(canonical_keys)} canonical "
            f"name(s) + {n_short_keys} short-form variant(s); "
            f"{len(ambiguous_variants)} ambiguous variant(s) dropped"
        )

        # Pull every ACTIVE StructuredTable artifact + its JSON-LD payload.
        table_rows = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.extra_metadata,
            ).where(
                IntelligenceArtifact.artifact_type == "StructuredTable",
                IntelligenceArtifact.status == "ACTIVE",
            )
        )
        all_tables = table_rows.all()

        gv = await current_version(session)

    if not all_tables:
        print("[link-tables] no StructuredTable artifacts found; nothing to link")
        return

    summary.tables_scanned = len(all_tables)
    print(
        f"[link-tables] scanning {summary.tables_scanned} table(s) for "
        f"name matches against {len(ent_by_norm)} entity name(s)..."
    )

    edge_payloads: list[dict[str, Any]] = []
    seen_pairs: set[tuple[Any, Any]] = set()
    for table_id, payload in all_tables:
        candidates = _collect_table_candidates(payload)
        if not candidates:
            continue
        for cand in candidates:
            norm = _normalize_name(cand)
            entity_id = ent_by_norm.get(norm)
            if entity_id is None:
                continue
            pair = (table_id, entity_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edge_payloads.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": table_id,
                "target_node_type": "entity",
                "target_node_id": entity_id,
                "predicate_iri": VIAO_ASSERTS_ABOUT,
                "predicate_label": "viao:assertsAbout",
                "relationship_type": "assertsAbout",
                "relationship_source": "DOCUMENT_EXTRACTION",
                "is_authoritative": True,
                "source_chunk_id": None,
                "source_document_id": None,
                "source_artifact_id": table_id,
                "graph_version": gv,
                "extra_metadata": {},
            })

    if not edge_payloads:
        print("[link-tables] no name matches found; 0 edges written")
        return

    # Idempotent insert: skip rows that already exist for the same
    # (source, target, predicate) tuple. The current schema doesn't have
    # a unique index on those columns, so we de-dupe by reading existing
    # pairs first instead of relying on ON CONFLICT.
    async with session_scope() as session:
        existing_rows = await session.execute(
            select(
                GraphRelationship.source_node_id,
                GraphRelationship.target_node_id,
            ).where(
                GraphRelationship.predicate_iri == VIAO_ASSERTS_ABOUT,
                GraphRelationship.source_node_type == "intelligence_artifact",
                GraphRelationship.target_node_type == "entity",
            )
        )
        already_have: set[tuple[Any, Any]] = {
            (s, t) for s, t in existing_rows.all()
        }

    fresh = [
        p for p in edge_payloads
        if (p["source_node_id"], p["target_node_id"]) not in already_have
    ]

    if not fresh:
        print(
            f"[link-tables] {len(edge_payloads)} candidate edge(s); "
            "all already present, 0 new"
        )
        return

    EDGE_BATCH = 500
    async with session_scope() as session:
        for i in range(0, len(fresh), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(fresh[i : i + EDGE_BATCH])
            )

    summary.table_entity_edges = len(fresh)
    print(
        f"[link-tables] inserted {len(fresh)} table->entity edge(s) "
        f"({len(edge_payloads) - len(fresh)} duplicate(s) skipped)"
    )

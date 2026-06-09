"""LLM-using pipeline: prune, expand, prune+expand, build.

Stage 1 (Groq, cheap)    — chunk_classification: doc chunk -> relevant top-level IRIs.
Stage 2 (OpenAI, focused) — class_proposal: doc chunk + sliced sub-ontology
                            -> {MATCHES FOUND, MATCH NOT FOUND}.
Stage 3 (OpenAI)         — match_dedup: collapse duplicate proposals across chunks.
Stage 4 (deterministic)  — prune_and_extend_loaded_ontology: pure Python over dicts.

The four CLI entry points share most plumbing; they differ only in WHICH
deterministic stage they invoke at the end (prune-only, expand-only, both).
`build_async` chains `run_merge` + `prune_and_expand_async`.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.helpers.ontology_pruning import (
    _collect_orphan_classes,
    add_new_classes_from_match_not_found,
    add_new_instances_from_match_results,
    add_new_relations_from_match_results,
    apply_concept_grouping,
    collect_full_class_hierarchy,
    collect_related_class_iris,
    expand_with_relationship_partners,
    extract_detected_iris,
    extract_json_from_output,
    infer_geographic_placement,
    infer_stem_relations,
    merge_llm_jsons_recursive,
    prune_classes_dict,
    prune_data_properties_dict,
    prune_instances_dict,
    prune_object_properties_dict,
)
from backend.app.services import (
    document_io,
    folder_io,
    ontology_export,
    versioning,
)
from backend.app.services.chunking import TextChunk, chunk_documents
from backend.app.services.document_io import LoadedDocument
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS
from backend.app.services.suggestions import (
    load_suggested_classes,
    merge_suggestions_into_results,
)

# ---------- Stage 1: chunk classification ----------


# Generic top-types that owlready2 loads into classes_dict alongside the
# user's real domain classes. They get declared as the superclass of every
# domain root (VIAO InformationSource, geography GeographicEntity, time
# DayOfWeek, ...) which is correct in OWL but masks the domain roots from
# `_top_level_branches`: the function treats any class whose super is in
# the dict as "not a root." Treating these IRIs as outside-the-ontology
# for the containment check lets the real domain roots surface to Stage 1.
_GENERIC_TOP_TYPES: frozenset[str] = frozenset({
    "http://www.w3.org/2002/07/owl#Thing",
})


def _top_level_branches(loaded_ontology: dict[str, Any], max_branches: int = 256) -> list[dict[str, Any]]:
    """Return a small summary of top-level classes (those with no named
    superclass inside the ontology) for the Stage-1 classifier.

    Cap at `max_branches` so the Groq prompt stays well under the model's
    context. If the ontology has more than that many top-level classes,
    we fall back to a representative sample (alphabetical by label).

    Generic top-types (`owl:Thing`, ...) are treated as outside-the-ontology
    when checking superclass containment — otherwise every domain root that
    declares `owl:Thing` as its super gets misclassified as non-root and the
    Stage-1 branch set collapses to whichever ontologies happen NOT to do
    that. `owl:Thing` itself is also excluded from the returned roots.
    """
    classes = loaded_ontology.get("classes_dict", {})
    all_iris = set(classes.keys())
    roots: list[dict[str, Any]] = []
    for iri, record in classes.items():
        if iri in _GENERIC_TOP_TYPES:
            continue
        supers = record.get("superclasses") or []
        is_root = True
        for s in supers:
            super_iri = (
                s.get("iri") if isinstance(s, dict) else (s if isinstance(s, str) else None)
            )
            if super_iri and super_iri in all_iris and super_iri not in _GENERIC_TOP_TYPES:
                is_root = False
                break
        if is_root:
            label = _first_label(record) or iri
            roots.append({"iri": iri, "label": label})
    roots.sort(key=lambda r: r["label"])
    return roots[:max_branches]


def _first_label(record: dict[str, Any]) -> str | None:
    labels = record.get("labels") or []
    if labels:
        first = labels[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    name = record.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


# Matches both `try again in 7.5s` and `try again in 200ms` (the latter common on
# Groq Dev-tier TPM bursts). Returns seconds in both cases.
_RETRY_AFTER_RE = re.compile(
    r"try again in ([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\b",
    re.IGNORECASE,
)


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort: pull `Please try again in Xs` (or `Xms`) out of a
    Groq/OpenAI 429 message and return the wait in SECONDS. None if not
    a rate-limit error or if no hint found."""
    msg = str(exc)
    if "rate_limit" not in msg.lower() and "429" not in msg:
        return None
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "s").lower()
    if unit == "ms":
        return val / 1000.0
    return val


async def _classify_chunk(
    router: LLMRouter,
    branches: list[dict[str, Any]],
    chunk: TextChunk,
    max_retries: int = 4,
) -> list[str]:
    """Stage 1: return relevant top-level IRIs for one chunk.

    Retries on Groq's TPM rate-limit (429) up to `max_retries` times,
    sleeping for the hint Groq embeds in the error message
    (`Please try again in Xs`) plus a small buffer, or 5s if no hint.
    Free-tier llama-3.3-70b is capped at 12k TPM, which a doc-classification
    chunk + 16 top-level branches blows through trivially at concurrency=8.
    """
    system, user = PROMPTS["chunk_classification"](branches, chunk.text)
    attempt = 0
    while True:
        try:
            result = await router.chat("chunk_classification", system=system, user=user)
            break
        except Exception as exc:
            wait = _parse_retry_after_seconds(exc)
            if wait is None or attempt >= max_retries:
                print(f"[stage1] chunk #{chunk.index} ({chunk.source_name}) failed: {exc}")
                return []
            attempt += 1
            # +0.5s buffer so the bucket has time to refill before the retry hits.
            sleep_s = wait + 0.5
            print(
                f"[stage1] chunk #{chunk.index} ({chunk.source_name}) rate-limited; "
                f"retry {attempt}/{max_retries} after {sleep_s:.1f}s"
            )
            await asyncio.sleep(sleep_s)
    data = extract_json_from_output(result.text) or {}
    iris = data.get("relevant_iris") or []
    return [i for i in iris if isinstance(i, str) and i]


# ---------- Stage 2: focused class matching + proposal ----------


def _slice_ontology(
    loaded_ontology: dict[str, Any],
    detected_iris: list[str],
    max_hops: int,
) -> dict[str, Any]:
    """Return a sub-dict of `classes_dict` covering the detected IRIs and
    their N-hop neighborhood. Strips heavy fields (raw_axiom_triples) so
    the Stage-2 prompt stays compact.

    Per-class field selection:
      - If the class has a `compact_description` (produced by the
        `summarize-descriptions` step), ship that INSTEAD of the verbose
        `descriptions` + `comments` fields. The compact form is ~15
        words vs the original 60+; saves ~50% per-class slice
        metadata.
      - If no compact_description exists, fall back to the original
        descriptions + comments. So the pipeline still works on
        un-summarized merges -- the compact form is an optional
        optimization the user opts into per merge folder.
    """
    classes = loaded_ontology.get("classes_dict", {})
    if not detected_iris:
        return {}
    # collect_related_class_iris builds its own graph internally; no need to
    # call build_class_graph separately.
    relevant = collect_related_class_iris(classes, list(detected_iris), max_hops=max_hops)
    out: dict[str, Any] = {}
    base_fields = ("name", "iri", "labels", "superclasses")
    for iri in relevant:
        rec = classes.get(iri)
        if rec is None:
            continue
        entry = {k: rec.get(k) for k in base_fields if k in rec}
        compact = rec.get("compact_description")
        if isinstance(compact, str) and compact.strip():
            entry["compact_description"] = compact.strip()
        else:
            # Backward-compatible path for un-summarized merges.
            for k in ("comments", "descriptions"):
                if k in rec:
                    entry[k] = rec.get(k)
        out[iri] = entry
    return out


async def _propose_for_chunk(
    router: LLMRouter,
    loaded_ontology: dict[str, Any],
    detected_iris: list[str],
    chunk: TextChunk,
    max_hops: int,
    suggested_new_classes: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Stage 2 for one chunk."""
    ontology_slice = _slice_ontology(loaded_ontology, detected_iris, max_hops)
    if not ontology_slice:
        # Nothing to match against — still try to surface MATCH NOT FOUND
        # proposals against a trivially-empty slice. The prompt still works.
        ontology_slice = {}
    hint = {"user_suggested": suggested_new_classes} if suggested_new_classes else None
    system, user = PROMPTS["class_proposal"](ontology_slice, chunk.text, suggested_new_classes=hint)
    try:
        result = await router.chat("class_proposal", system=system, user=user)
    except Exception as exc:
        print(f"[stage2] chunk #{chunk.index} ({chunk.source_name}) failed: {exc}")
        return None
    return extract_json_from_output(result.text)


# ---------- Stage 3: dedup ----------


async def _dedup(router: LLMRouter, merged_results: dict[str, Any]) -> dict[str, Any]:
    """Stage 3: collapse duplicate MATCH NOT FOUND across chunks."""
    if not merged_results.get("MATCH NOT FOUND"):
        # Nothing to dedup; skip the LLM call.
        return merged_results
    system, user = PROMPTS["match_dedup"](merged_results)
    try:
        result = await router.chat("match_dedup", system=system, user=user)
    except Exception as exc:
        print(f"[stage3] dedup failed: {exc} — returning merged results unchanged")
        return merged_results
    cleaned = extract_json_from_output(result.text)
    if not cleaned:
        return merged_results
    return cleaned


# ---------- Layer G: top-level concept grouping (one LLM call) ----------


_CONCEPT_GROUPING_BATCH_SIZE = 150


async def _propose_concept_grouping(
    router: LLMRouter,
    orphan_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask gpt-4.1 to propose 5-15 high-level concept classes that group
    the orphan-class list, plus an assignment per orphan.

    BATCHING: each ASSIGNMENT entry in the response is ~30-50 tokens of
    JSON. With max_tokens=8192 on the LLM, the response cap is ~150-200
    orphans before responses get truncated mid-JSON. So we chunk the
    orphans into batches of `_CONCEPT_GROUPING_BATCH_SIZE` each, fire
    them sequentially (different concept proposals per batch are merged
    case-insensitively by label), then return a single merged result.

    Returns the parsed + merged JSON or an empty shape on any failure --
    this pass is purely additive; it must never break the pipeline."""
    empty = {"TOP_LEVEL_CONCEPTS": [], "ASSIGNMENTS": []}
    if not orphan_classes:
        return empty

    # Split into batches.
    batches = [
        orphan_classes[i : i + _CONCEPT_GROUPING_BATCH_SIZE]
        for i in range(0, len(orphan_classes), _CONCEPT_GROUPING_BATCH_SIZE)
    ]
    if len(batches) > 1:
        print(
            f"[stage4-G] concept_grouping: chunking {len(orphan_classes)} "
            f"orphans into {len(batches)} batches of <= {_CONCEPT_GROUPING_BATCH_SIZE}"
        )

    merged_concepts: dict[str, dict[str, Any]] = {}  # lower-case LABEL -> entry
    merged_assignments: list[dict[str, Any]] = []

    for i, batch in enumerate(batches):
        system, user = PROMPTS["concept_grouping"](batch)
        try:
            result = await router.chat("concept_grouping", system=system, user=user)
        except Exception as exc:
            print(f"[stage4-G] batch {i+1}/{len(batches)} LLM call failed: {exc} — skipping batch")
            continue
        parsed = extract_json_from_output(result.text)
        if not isinstance(parsed, dict):
            print(f"[stage4-G] batch {i+1}/{len(batches)} response was not parseable JSON — skipping batch")
            continue

        # Merge concepts (case-insensitive dedup by LABEL).
        for entry in parsed.get("TOP_LEVEL_CONCEPTS") or []:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("LABEL") or "").strip()
            if not label:
                continue
            key = label.lower()
            if key not in merged_concepts:
                merged_concepts[key] = entry

        # Append assignments verbatim.
        for entry in parsed.get("ASSIGNMENTS") or []:
            if isinstance(entry, dict):
                merged_assignments.append(entry)

    return {
        "TOP_LEVEL_CONCEPTS": list(merged_concepts.values()),
        "ASSIGNMENTS": merged_assignments,
    }


# ---------- One-time class-metadata compression ----------


_COMPACT_DESCRIPTION_BATCH_SIZE = 20


def _has_useful_text(rec: dict[str, Any]) -> bool:
    """A class is worth summarizing if it has at least one non-trivial
    description or comment string. Empty or single-character text isn't
    worth a round trip."""
    for field in ("descriptions", "comments"):
        for v in rec.get(field) or []:
            if isinstance(v, str) and len(v.strip()) > 3:
                return True
    return False


async def summarize_class_descriptions_async(
    classes_dict: dict[str, Any],
    router: LLMRouter,
    max_cost_usd: float = 5.0,
    batch_size: int = _COMPACT_DESCRIPTION_BATCH_SIZE,
    concurrency: int = 8,
) -> dict[str, Any]:
    """One-time class-metadata compression. Iterate `classes_dict`,
    batch each group of N classes that have non-trivial descriptions or
    comments, send to gpt-4o-mini for a short rewrite, write the result
    back as `compact_description` on each class record.

    The pipeline never re-fires for a class that already has a non-empty
    `compact_description` field (so re-running this is a no-op cost-wise).

    Skips classes whose source text is empty or trivial -- they don't
    need a compact_description.

    Returns a summary dict {classes_total, classes_summarized,
    classes_skipped, llm_calls, cost_usd}.
    """
    candidates = [
        (iri, rec) for iri, rec in classes_dict.items()
        if not (rec.get("compact_description") or "").strip()
        and _has_useful_text(rec)
    ]
    print(
        f"[compact-desc] {len(candidates)} class(es) to summarize "
        f"(out of {len(classes_dict)} total; already-summarized + "
        f"trivial classes skipped)"
    )
    if not candidates:
        return {
            "classes_total": len(classes_dict),
            "classes_summarized": 0,
            "classes_skipped": len(classes_dict),
            "llm_calls": 0,
            "cost_usd": 0.0,
        }

    # Project to lightweight batch records: just the fields the prompt
    # needs. Keeps each batch under gpt-4o-mini's input limit comfortably.
    def _projection(iri: str, rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "iri": iri,
            "name": rec.get("name") or "",
            "labels": rec.get("labels") or [],
            "descriptions": rec.get("descriptions") or [],
            "comments": rec.get("comments") or [],
        }

    batches = [
        [_projection(iri, rec) for iri, rec in candidates[i : i + batch_size]]
        for i in range(0, len(candidates), batch_size)
    ]
    print(f"[compact-desc] processing {len(batches)} batch(es) of <= {batch_size}")

    sem = asyncio.Semaphore(concurrency)
    cost_before = router.total_cost_usd

    async def _one_batch(batch_idx: int, batch: list[dict[str, Any]]) -> None:
        async with sem:
            # Cost-cap check INSIDE the semaphore so an over-budget batch
            # doesn't fire before we notice.
            if router.total_cost_usd - cost_before > max_cost_usd:
                print(f"[compact-desc] batch {batch_idx+1}: cost cap hit, skipping remaining")
                return
            system, user = PROMPTS["compact_description"](batch)
            try:
                result = await router.chat("compact_description", system=system, user=user)
            except Exception as exc:
                print(f"[compact-desc] batch {batch_idx+1} LLM call failed: {exc}")
                return
            parsed = extract_json_from_output(result.text)
            if not isinstance(parsed, dict):
                print(f"[compact-desc] batch {batch_idx+1} response not parseable JSON; skipping")
                return
            for entry in parsed.get("results") or []:
                if not isinstance(entry, dict):
                    continue
                iri = entry.get("iri")
                cd = entry.get("compact_description")
                if isinstance(iri, str) and iri in classes_dict and isinstance(cd, str) and cd.strip():
                    classes_dict[iri]["compact_description"] = cd.strip()

    await asyncio.gather(*[_one_batch(i, b) for i, b in enumerate(batches)])

    summarized = sum(
        1 for _, rec in candidates
        if isinstance(rec.get("compact_description"), str)
        and rec["compact_description"].strip()
    )
    return {
        "classes_total": len(classes_dict),
        "classes_summarized": summarized,
        "classes_skipped": len(classes_dict) - summarized,
        "llm_calls": len(batches),
        "cost_usd": round(router.total_cost_usd - cost_before, 6),
    }


# ---------- Pre-pipeline document summarization (optional) ----------

# Bump this string ONLY when the document_summarize prompt changes
# meaningfully. The prompt version is mixed into each cache key so old
# cached summaries are silently invalidated.
_DOC_SUMMARY_PROMPT_VERSION = "v1"


def _doc_summary_cache_dir() -> Path:
    """Return the on-disk cache root, creating it if missing."""
    root = Path.home() / ".cache" / "your-personal-knowledge-graph-creator" / "doc_summaries"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _doc_summary_cache_key(text: str, model: str) -> str:
    """SHA-256 over (doc text + model name + prompt version). Changing
    any of these invalidates the cache for that doc."""
    import hashlib

    h = hashlib.sha256()
    h.update(_DOC_SUMMARY_PROMPT_VERSION.encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _doc_summary_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.txt"


def _doc_summary_cache_load(path: Path) -> str | None:
    """Return the cached summary if the file exists and is non-empty,
    else None."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    text = text.strip()
    return text if text else None


def _doc_summary_cache_save(path: Path, text: str) -> None:
    """Atomic write: write to a per-process temp file then rename. POSIX
    rename is atomic within a single filesystem, so concurrent workers
    racing on the same cache key produce a single final file -- no
    partial writes."""
    import os

    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"[summarize-docs] WARN: cache write failed: {exc}")
        # Clean up the temp file if rename didn't happen.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _split_text_into_sub_chunks(
    text: str, max_tokens: int, encoder
) -> list[str]:
    """Split `text` into sub-chunks each within `max_tokens` (no overlap).
    Used only for hierarchical summarization of oversize documents."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    sub_chunks: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        piece = tokens[start : start + max_tokens]
        sub_chunks.append(encoder.decode(piece))
    return sub_chunks


async def _summarize_oversize_doc_async(
    *,
    doc: LoadedDocument,
    router: LLMRouter,
    sub_chunk_tokens: int,
    encoder,
    sem: asyncio.Semaphore,
) -> str | None:
    """Hierarchical summarization for a single oversize document.

    Splits the doc text into sub-chunks of `sub_chunk_tokens` tokens,
    fires gpt-4o-mini against each sub-chunk via the existing
    document_summarize prompt (in parallel, gated by the shared
    semaphore), then concatenates the per-sub-chunk summaries with
    blank-line separators.

    Returns the combined summary text, or None if EVERY sub-chunk
    failed (caller should keep the doc as empty / unchanged).
    """
    sub_texts = _split_text_into_sub_chunks(doc.text, sub_chunk_tokens, encoder)
    n = len(sub_texts)
    print(
        f"[summarize-docs] {doc.name}: splitting into {n} sub-chunk(s) "
        f"of <={sub_chunk_tokens:,} tokens for hierarchical summarization"
    )

    results: list[str | None] = [None] * n

    async def _one_sub(i: int, sub_text: str) -> None:
        async with sem:
            system, user = PROMPTS["document_summarize"](sub_text)
            try:
                result = await router.chat("document_summarize", system=system, user=user)
            except Exception as exc:
                print(
                    f"[summarize-docs] {doc.name}: sub-chunk {i+1}/{n} "
                    f"LLM failed: {exc} -- omitting from combined summary"
                )
                return
            piece = (result.text or "").strip()
            if not piece:
                print(
                    f"[summarize-docs] {doc.name}: sub-chunk {i+1}/{n} "
                    f"empty response -- omitting from combined summary"
                )
                return
            results[i] = piece

    await asyncio.gather(*[_one_sub(i, t) for i, t in enumerate(sub_texts)])

    parts = [r for r in results if r]
    if not parts:
        return None
    combined = "\n\n".join(parts)
    succeeded = len(parts)
    if succeeded < n:
        print(
            f"[summarize-docs] {doc.name}: {succeeded}/{n} sub-chunks "
            f"succeeded; using partial combined summary"
        )
    return combined


async def summarize_long_documents_async(
    documents: list[LoadedDocument],
    router: LLMRouter,
    threshold_tokens: int = 2000,
    encoding_name: str = "o200k_base",
    concurrency: int = 4,
    model_name: str = "gpt-4o-mini",
    use_cache: bool = True,
    max_doc_input_tokens: int = 100_000,
    oversize_doc_sub_chunk_tokens: int = 80_000,
) -> list[LoadedDocument]:
    """Optional pre-pipeline pass that rewrites long source documents
    into denser entity-preserving summaries before chunking.

    For each document:
      - Count tokens via tiktoken.
      - If token count <= threshold: passed through unchanged.
      - If threshold < token count <= max_doc_input_tokens: ONE LLM call
        to gpt-4o-mini with the document_summarize prompt.
      - If token count > max_doc_input_tokens: HIERARCHICAL
        summarization -- the doc is split into sub-chunks of
        `oversize_doc_sub_chunk_tokens` tokens each, every sub-chunk is
        summarized independently via gpt-4o-mini, and the per-sub-chunk
        summaries are concatenated (blank-line separated) into the
        final combined summary. The combined summary is stored as the
        doc's new text and cached under the original doc's hash key.

    `use_cache` (default True) reads/writes the summary at
    ~/.cache/your-personal-knowledge-graph-creator/doc_summaries/. Cache
    key hashes (doc text + model + prompt version), so editing a doc or
    changing the model invalidates that entry automatically. The cache
    works the same for hierarchical summaries -- the COMBINED summary
    is stored under the original doc's hash.

    Returns a NEW list of LoadedDocument (the input list is not mutated).

    Failure modes are purely additive:
      - Single-call doc, LLM raises / returns empty -> original text kept.
      - Hierarchical doc, SOME sub-chunks fail -> partial combined summary used.
      - Hierarchical doc, ALL sub-chunks fail -> empty text (chunker emits zero chunks).

    The pipeline must NEVER fail because of this step.

    Triggered by `chunking.summarization_threshold_tokens` in
    config.yaml. Set the config value to 0 to disable.
    """
    import tiktoken  # local import: only when this pass actually runs

    if not documents or threshold_tokens <= 0:
        return list(documents)

    enc = tiktoken.get_encoding(encoding_name)
    cache_dir = _doc_summary_cache_dir() if use_cache else None

    # Identify which documents need summarization (over threshold) and
    # which need the hierarchical path (also over max_doc_input_tokens).
    out: list[LoadedDocument] = list(documents)
    plan: list[tuple[int, int, bool]] = []  # (idx, tokens, needs_hierarchical)
    for i, doc in enumerate(documents):
        tok = len(enc.encode(doc.text))
        if tok > threshold_tokens:
            needs_hierarchical = tok > max_doc_input_tokens
            plan.append((i, tok, needs_hierarchical))

    if not plan:
        print(
            f"[summarize-docs] all {len(documents)} doc(s) under threshold; "
            f"skipping summarization"
        )
        return out

    # First pass: cache lookups (fast, no LLM, no concurrency needed).
    cache_hits = 0
    needs_llm: list[tuple[int, int, bool]] = []

    if cache_dir is not None:
        for idx, original_tokens, needs_hier in plan:
            doc = documents[idx]
            key = _doc_summary_cache_key(doc.text, model_name)
            cached = _doc_summary_cache_load(_doc_summary_cache_path(cache_dir, key))
            if cached:
                out[idx] = LoadedDocument(path=doc.path, text=cached)
                cache_hits += 1
            else:
                needs_llm.append((idx, original_tokens, needs_hier))
    else:
        needs_llm = list(plan)

    oversize_needs_llm = sum(1 for _, _, h in needs_llm if h)
    print(
        f"[summarize-docs] {len(plan)}/{len(documents)} doc(s) over "
        f"threshold ({threshold_tokens} tokens): "
        f"{cache_hits} cache hit(s), {len(needs_llm)} cache miss(es) "
        f"({oversize_needs_llm} require hierarchical summarization) "
        f"via gpt-4o-mini at concurrency={concurrency}"
    )

    if not needs_llm:
        print("[summarize-docs] DONE: all over-threshold docs served from cache, $0.0000")
        return out

    sem = asyncio.Semaphore(concurrency)
    cost_before = router.total_cost_usd

    async def _one(idx: int, original_tokens: int, needs_hier: bool) -> None:
        doc = documents[idx]
        if needs_hier:
            # Hierarchical path: sub-chunk + summarize each + concatenate.
            # NOTE: gating happens inside _summarize_oversize_doc_async on
            # each sub-chunk, NOT at the outer level -- otherwise this
            # single doc would monopolize the semaphore.
            combined = await _summarize_oversize_doc_async(
                doc=doc,
                router=router,
                sub_chunk_tokens=oversize_doc_sub_chunk_tokens,
                encoder=enc,
                sem=sem,
            )
            if combined is None:
                print(
                    f"[summarize-docs] {doc.name}: all sub-chunks failed -- "
                    f"keeping doc as empty text"
                )
                out[idx] = LoadedDocument(path=doc.path, text="")
                return
            new_tokens = len(enc.encode(combined))
            print(
                f"[summarize-docs] {doc.name}: hierarchical "
                f"{original_tokens:,} -> {new_tokens:,} tokens "
                f"({100 * new_tokens / original_tokens:.1f}%)"
            )
            out[idx] = LoadedDocument(path=doc.path, text=combined)
            if cache_dir is not None:
                key = _doc_summary_cache_key(doc.text, model_name)
                _doc_summary_cache_save(_doc_summary_cache_path(cache_dir, key), combined)
            return

        # Standard single-call path.
        async with sem:
            system, user = PROMPTS["document_summarize"](doc.text)
            try:
                result = await router.chat("document_summarize", system=system, user=user)
            except Exception as exc:
                print(
                    f"[summarize-docs] doc {idx+1}/{len(documents)} "
                    f"({doc.name}): LLM failed: {exc} -- keeping original"
                )
                return
            text = (result.text or "").strip()
            if not text:
                print(
                    f"[summarize-docs] doc {idx+1}/{len(documents)} "
                    f"({doc.name}): empty response -- keeping original"
                )
                return
            new_tokens = len(enc.encode(text))
            print(
                f"[summarize-docs] doc {idx+1}/{len(documents)} "
                f"({doc.name}): {original_tokens} -> {new_tokens} tokens "
                f"({100 * new_tokens / original_tokens:.0f}%)"
            )
            out[idx] = LoadedDocument(path=doc.path, text=text)
            if cache_dir is not None:
                key = _doc_summary_cache_key(doc.text, model_name)
                _doc_summary_cache_save(_doc_summary_cache_path(cache_dir, key), text)

    await asyncio.gather(*[_one(i, tok, h) for i, tok, h in needs_llm])

    cost_delta = router.total_cost_usd - cost_before
    print(
        f"[summarize-docs] DONE: {cache_hits} cached, {len(needs_llm)} summarized, "
        f"{len(documents) - len(plan)} doc(s) unchanged, "
        f"cost ${cost_delta:.4f}"
    )
    return out


# ---------- Stage 4: deterministic prune / extend ----------


def _apply_prune(
    loaded_ontology: dict[str, Any],
    detected_iris: list[str],
    protected_iri_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, Any], set[str]]:
    """Pure Python: build the keep-set as
        detected ∪ full IS-A hierarchy of detected ∪ relationship partners
        ∪ every class IRI whose IRI starts with a protected prefix
    then drop everything else.

    Keep-set construction in order:
      1. Start with the detected (LLM-matched) class IRIs.
      2. Expand to the FULL ancestor + descendant transitive closure via
         subClassOf (collect_full_class_hierarchy). Not N-hop -- the entire
         IS-A neighborhood is preserved so every kept class's place in the
         taxonomy is unambiguous.
      3. Add the other-endpoint classes of every object/data property whose
         domain or range touches the keep-set so far
         (expand_with_relationship_partners). This keeps relationships
         intact end-to-end instead of leaving them with `range=[]` when the
         range class was outside the original hierarchy.
      4. Union in every class IRI whose IRI starts with one of
         `protected_iri_prefixes`. This forces whole-ontology preservation
         for user-curated ontologies (e.g. VIAO) that the user wants to
         maintain regardless of document-driven detection. Property
         survival for protected classes is automatic: any property
         whose domain or range touches a protected class clears the
         `new_domain or new_range` filter in prune_*_properties_dict.

    The `max_hops` argument that used to drive an undirected N-hop BFS here
    has been removed -- it still drives Stage 2's slice-of-the-ontology
    sent to the LLM, but Stage 4 prune now uses the unbounded IS-A closure
    described above.
    """
    classes = loaded_ontology.get("classes_dict", {})
    if not detected_iris and not protected_iri_prefixes:
        return loaded_ontology, set()
    obj_props = loaded_ontology.get("object_properties_dict", {})
    data_props = loaded_ontology.get("data_properties_dict", {})

    keep_iris = collect_full_class_hierarchy(classes, list(detected_iris))
    keep_iris = expand_with_relationship_partners(keep_iris, obj_props, data_props)

    if protected_iri_prefixes:
        protected_class_iris = {
            iri for iri in classes
            if any(iri.startswith(p) for p in protected_iri_prefixes)
        }
        keep_iris = set(keep_iris) | protected_class_iris

    pruned: dict[str, Any] = {
        "classes_dict": prune_classes_dict(classes, keep_iris),
        "object_properties_dict": prune_object_properties_dict(obj_props, keep_iris),
        "data_properties_dict": prune_data_properties_dict(data_props, keep_iris),
        "instances_dict": prune_instances_dict(loaded_ontology.get("instances_dict", {}), keep_iris),
    }
    return pruned, keep_iris


def _apply_expand(
    loaded_ontology: dict[str, Any],
    match_results: dict[str, Any],
    base_iri: str,
    default_parent_iri: str | None,
) -> tuple[dict[str, Any], list[str], list[str], list[dict], list[str]]:
    """Add proposed classes from MATCH NOT FOUND, then named individuals
    from MATCH NOT FOUND INSTANCES, then object-property relations from
    MATCH NOT FOUND RELATIONS.

    Order matters: classes go in first so TYPE_LABEL on instances and
    DOMAIN/RANGE on relations can resolve against classes proposed in
    the same LLM run. Instances go in before relations so relation
    endpoints can resolve to a just-minted instance.

    Returns:
      (extended_ontology, created_class_iris, created_property_iris,
       skipped_relations, created_instance_iris)
    """
    extended, created_classes = add_new_classes_from_match_not_found(
        loaded_ontology=loaded_ontology,
        match_results=match_results,
        new_class_base_iri=base_iri,
        default_parent_iri=default_parent_iri,
    )

    # Mint instances next so relations can resolve endpoints that are
    # named individuals (e.g. "iran_war_2025") rather than classes.
    extended, created_instances = add_new_instances_from_match_results(
        loaded_ontology=extended,
        match_results=match_results,
        new_instance_base_iri=base_iri,
        default_type_iri=default_parent_iri,
    )

    extended, created_props, skipped, auto_minted = add_new_relations_from_match_results(
        loaded_ontology=extended,
        match_results=match_results,
        new_property_base_iri=base_iri,
        default_parent_iri=default_parent_iri,
        new_class_base_iri=base_iri,
    )
    # Auto-minted classes from unresolved relation endpoints count as
    # created_classes from the caller's perspective.
    created_classes = list(created_classes) + list(auto_minted)

    # Layer E: deterministic stem-based relation enrichment. Catches
    # `helium has_market helium_market`-style relations that the LLM didn't
    # propose across chunks. Modifies obj_props_dict in place.
    _, stem_props = infer_stem_relations(
        classes_dict=extended.get("classes_dict", {}),
        obj_props_dict=extended.setdefault("object_properties_dict", {}),
        new_property_base_iri=base_iri,
    )
    created_props = list(created_props) + list(stem_props)

    # Layer F: geographic-entity inference. Re-homes classes that the LLM
    # left at owl:Thing in the default namespace but are clearly geographic
    # entities (named landforms detected by keyword OR class is reached via
    # a located_in/part_of-style predicate to an existing geography class).
    # Mutates all four dicts in place.
    geo_audit = infer_geographic_placement(
        classes_dict=extended.get("classes_dict", {}),
        obj_props_dict=extended.setdefault("object_properties_dict", {}),
        data_props_dict=extended.setdefault("data_properties_dict", {}),
        instances_dict=extended.setdefault("instances_dict", {}),
    )
    if geo_audit:
        print(f"[stage4] geographic-inference re-homed {len(geo_audit)} class(es)")
    return extended, created_classes, list(created_props), skipped, list(created_instances)


# ---------- LLM stage orchestration shared by prune / expand / both ----------


async def _run_llm_stages(
    *,
    loaded_ontology: dict[str, Any],
    documents_dir: Path,
    router: LLMRouter,
    max_hops: int,
    max_cost_usd: float | None,
    dry_run: bool,
    app_cfg: dict[str, Any],
    audit_path: Path,
    suggested_new_classes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stages 1-3. Returns the merged + deduplicated match-results dict.

    If the projected cost exceeds max_cost_usd, raises RuntimeError before
    any expensive calls.
    """
    branches = _top_level_branches(loaded_ontology)
    print(f"[llm] top-level branches surfaced: {len(branches)}")

    chunking_cfg = app_cfg.get("chunking", {}) or {}
    chunk_size = int(chunking_cfg.get("chunk_size", 800))
    chunk_overlap = int(chunking_cfg.get("chunk_overlap", 120))
    encoding = chunking_cfg.get("encoding", "o200k_base")

    docs = list(document_io.load_documents(documents_dir))
    print(f"[llm] loaded {len(docs)} document(s)")
    if not docs:
        raise RuntimeError(f"No PDF/TXT documents found in {documents_dir}")

    # Optional pre-pipeline: documents above the configured token threshold
    # get rewritten via gpt-4o-mini into a denser entity-preserving summary
    # before chunking. Cuts Stage 2 chunk count + cost dramatically on
    # corpora dominated by a few large documents. Set to 0 to disable.
    expansion_cfg_for_concur = app_cfg.get("expansion", {}) or {}
    threshold_tokens = int(chunking_cfg.get("summarization_threshold_tokens", 0))
    use_cache = bool(chunking_cfg.get("use_summary_cache", True))
    max_doc_input_tokens = int(chunking_cfg.get("max_doc_input_tokens", 100_000))
    oversize_sub_chunk = int(chunking_cfg.get("oversize_doc_sub_chunk_tokens", 80_000))
    if threshold_tokens > 0:
        docs = await summarize_long_documents_async(
            documents=docs,
            router=router,
            threshold_tokens=threshold_tokens,
            encoding_name=encoding,
            concurrency=int(expansion_cfg_for_concur.get("max_concurrent_llm_calls", 4)),
            use_cache=use_cache,
            max_doc_input_tokens=max_doc_input_tokens,
            oversize_doc_sub_chunk_tokens=oversize_sub_chunk,
        )

    chunks = list(
        chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name=encoding)
    )
    print(f"[llm] produced {len(chunks)} chunk(s)")
    if not chunks:
        raise RuntimeError("No chunks produced from documents (empty after chunking)")

    if dry_run:
        print("[llm] --dry-run: stopping before any LLM calls")
        return {"MATCHES FOUND": [], "MATCH NOT FOUND": [], "MATCH NOT FOUND RELATIONS": []}

    expansion_cfg = app_cfg.get("expansion", {}) or {}
    concurrency = int(expansion_cfg.get("max_concurrent_llm_calls", 8))
    sem = asyncio.Semaphore(concurrency)

    async def _classify_one(chunk: TextChunk) -> list[str]:
        async with sem:
            return await _classify_chunk(router, branches, chunk)

    print(f"[stage1] classifying {len(chunks)} chunk(s) (Groq, concurrency={concurrency})")
    stage1_results = await asyncio.gather(*[_classify_one(c) for c in chunks])

    async def _propose_one(idx: int, chunk: TextChunk, iris: list[str]) -> dict[str, Any] | None:
        async with sem:
            res = await _propose_for_chunk(
                router,
                loaded_ontology,
                iris,
                chunk,
                max_hops,
                suggested_new_classes=suggested_new_classes,
            )
            if res:
                _append_audit(audit_path, idx, "class_proposal", chunk.source_name, res)
            return res

    # Skip Stage 2 for chunks where Stage 1 returned no relevant branches --
    # the LLM would have no ontology context to anchor against, so the call
    # produces mostly junk MATCH NOT FOUND proposals (paid at gpt-4.1 rates).
    skipped_empty = sum(1 for iris in stage1_results if not iris)
    if skipped_empty:
        print(
            f"[stage2] skipping {skipped_empty}/{len(chunks)} chunks "
            f"where Stage 1 returned no relevant IRIs"
        )

    print(f"[stage2] proposing matches+new for {len(chunks) - skipped_empty} chunk(s) (OpenAI)")
    stage2_results: list[dict[str, Any] | None] = await asyncio.gather(
        *[
            _propose_one(i, c, iris)
            for i, (c, iris) in enumerate(zip(chunks, stage1_results, strict=False))
            if iris  # only fire Stage 2 if Stage 1 found relevant branches
        ]
    )
    valid = [r for r in stage2_results if r]
    print(
        f"[stage2] {len(valid)}/{len(chunks) - skipped_empty} chunks produced "
        f"a usable JSON response (Stage 1 surfaced no branches for "
        f"{skipped_empty} other chunks; those were skipped)"
    )

    if max_cost_usd is not None and router.total_cost_usd > max_cost_usd:
        raise RuntimeError(
            f"Projected cost exceeded cap: ${router.total_cost_usd:.4f} > ${max_cost_usd:.4f}. "
            "Re-run with --max-cost-usd N to lift the cap."
        )

    merged = merge_llm_jsons_recursive(valid) if valid else {"MATCHES FOUND": [], "MATCH NOT FOUND": [], "MATCH NOT FOUND RELATIONS": []}
    print(
        f"[stage3] merging+dedup: {len(merged.get('MATCHES FOUND', []))} matches, "
        f"{len(merged.get('MATCH NOT FOUND', []))} new class proposals, "
        f"{len(merged.get('MATCH NOT FOUND RELATIONS', []))} new relation proposals"
    )
    deduped = await _dedup(router, merged)
    _append_audit(audit_path, -1, "match_dedup", None, deduped)
    return deduped


def _append_audit(path: Path, chunk_idx: int, task: str, source: str | None, payload: dict[str, Any]) -> None:
    rec = {"chunk_idx": chunk_idx, "task": task, "source": source, "payload": payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------- Per-subcommand entry points ----------


async def prune_only_async(
    *,
    input_folder: Path,
    documents_dir: Path,
    output_root: Path,
    max_hops: int | None,
    max_cost_usd: float | None,
    dry_run: bool,
    use_owl: bool = False,
    suggested_new_classes: Path | None = None,
) -> Path:
    return await _run(
        "prune",
        input_folder,
        documents_dir,
        output_root,
        max_hops,
        max_cost_usd,
        dry_run,
        suggestions_path=suggested_new_classes,
    )


async def expand_only_async(
    *,
    input_folder: Path,
    documents_dir: Path,
    output_root: Path,
    max_hops: int | None,
    max_cost_usd: float | None,
    dry_run: bool,
    use_owl: bool = False,
    suggested_new_classes: Path | None = None,
) -> Path:
    return await _run(
        "expand",
        input_folder,
        documents_dir,
        output_root,
        max_hops,
        max_cost_usd,
        dry_run,
        suggestions_path=suggested_new_classes,
    )


async def prune_and_expand_async(
    *,
    input_folder: Path,
    documents_dir: Path,
    output_root: Path,
    max_hops: int | None,
    max_cost_usd: float | None,
    dry_run: bool,
    use_owl: bool = False,
    suggested_new_classes: Path | None = None,
) -> Path:
    return await _run(
        "prune-expand",
        input_folder,
        documents_dir,
        output_root,
        max_hops,
        max_cost_usd,
        dry_run,
        suggestions_path=suggested_new_classes,
    )


async def build_async(
    *,
    input_ontologies: list[Path],
    documents_dir: Path,
    output_root: Path,
    max_hops: int | None,
    max_cost_usd: float | None,
    dry_run: bool,
    suggested_new_classes: Path | None = None,
) -> Path:
    # build = merge + prune-expand chained. Merge first (sync), then drive the
    # async LLM pipeline against the just-written version folder.
    from backend.app.services.pipeline import run_merge

    merged_dir = run_merge(input_ontologies=input_ontologies, output_root=output_root)
    return await _run(
        "build",
        merged_dir,
        documents_dir,
        output_root,
        max_hops,
        max_cost_usd,
        dry_run,
        suggestions_path=suggested_new_classes,
    )


async def _run(
    operation: str,
    input_folder: Path,
    documents_dir: Path,
    output_root: Path,
    max_hops: int | None,
    max_cost_usd: float | None,
    dry_run: bool,
    suggestions_path: Path | None = None,
) -> Path:
    settings = get_settings()
    app_cfg = settings.app_config
    expansion_cfg = app_cfg.get("expansion", {}) or {}
    effective_hops = max_hops if max_hops is not None else int(expansion_cfg.get("prune_max_hops", 1))
    effective_cost_cap = (
        max_cost_usd if max_cost_usd is not None else float(expansion_cfg.get("max_cost_usd", 25.0))
    )

    output_root.mkdir(parents=True, exist_ok=True)
    version_dir = versioning.new_version_dir(output_root, operation)
    audit_path = versioning.ensure_audit_log(version_dir)

    print(f"[{operation}] loading prior version: {input_folder}")
    loaded = folder_io.load_version_folder(input_folder)
    counts_before = folder_io.count_entities(loaded)

    suggested = load_suggested_classes(suggestions_path)
    if suggested:
        print(f"[{operation}] loaded {len(suggested)} user-suggested class(es) from {suggestions_path}")

    router = LLMRouter(settings)
    deduped = await _run_llm_stages(
        loaded_ontology=loaded,
        documents_dir=documents_dir,
        router=router,
        max_hops=effective_hops,
        max_cost_usd=effective_cost_cap,
        dry_run=dry_run,
        app_cfg=app_cfg,
        audit_path=audit_path,
        suggested_new_classes=suggested or None,
    )

    # Inject user-suggested classes that the LLM didn't already propose. These
    # are ADDITIONAL classes the user wants in the ontology regardless of
    # whether the document corpus surfaced them.
    if suggested and operation in ("expand", "prune-expand", "build"):
        before = len(deduped.get("MATCH NOT FOUND", []))
        deduped = merge_suggestions_into_results(deduped, suggested)
        added = len(deduped["MATCH NOT FOUND"]) - before
        print(f"[{operation}] injected {added} user-suggested class(es) into MATCH NOT FOUND")

    # Stage 4: deterministic prune / extend depending on subcommand.
    detected = extract_detected_iris(deduped)
    print(f"[stage4] detected {len(detected)} IRIs from MATCHES FOUND")

    out_ontology = loaded
    created: list[str] = []

    if operation in ("prune", "prune-expand", "build"):
        ontology_cfg = app_cfg.get("ontology", {}) or {}
        protected_prefixes = tuple(ontology_cfg.get("protected_iri_prefixes") or [])
        out_ontology, keep = _apply_prune(out_ontology, detected, protected_prefixes)
        n_protected = sum(
            1 for iri in out_ontology.get("classes_dict", {})
            if any(iri.startswith(p) for p in protected_prefixes)
        ) if protected_prefixes else 0
        protected_note = (
            f" ({n_protected} forced by {len(protected_prefixes)} protected prefix(es))"
            if protected_prefixes else ""
        )
        print(
            f"[stage4] pruned to {len(keep)} kept classes "
            f"(full IS-A hierarchy of detected + relationship partners){protected_note}"
        )

    if operation in ("expand", "prune-expand", "build"):
        ontology_cfg = app_cfg.get("ontology", {}) or {}
        base_iri = ontology_cfg.get("default_base_iri") or "http://your-personal-ontologist.local/ontology/"
        parent_iri = ontology_cfg.get("default_parent_iri")
        out_ontology, created, created_props, skipped_rels, created_instances = _apply_expand(
            out_ontology, deduped, base_iri, parent_iri
        )
        print(
            f"[stage4] created {len(created)} new classes from MATCH NOT FOUND, "
            f"{len(created_instances)} new instances from MATCH NOT FOUND INSTANCES, "
            f"{len(created_props)} new relations from MATCH NOT FOUND RELATIONS, "
            f"{len(skipped_rels)} relation(s) skipped (unresolved endpoints)"
        )
        if skipped_rels:
            for s in skipped_rels:
                print(f"[stage4]   skipped relation: {s.get('reason')} -> {s.get('relation')}")

        # Layer G: top-level concept grouping (one LLM call). Collects the
        # remaining orphan classes (still parented at owl:Thing after the
        # geography pass) and asks the LLM to propose a small set of high-
        # level concept classes to group them. Purely additive; failures
        # are logged and ignored.
        orphan_classes = _collect_orphan_classes(
            out_ontology.get("classes_dict", {}),
            base_iri,
        )
        if orphan_classes:
            print(f"[stage4-G] concept_grouping: {len(orphan_classes)} orphan class(es) to group")
            cg_result = await _propose_concept_grouping(router, orphan_classes)
            _append_audit(audit_path, -1, "concept_grouping", None, cg_result)
            concept_iris, cg_audit = apply_concept_grouping(
                classes_dict=out_ontology.setdefault("classes_dict", {}),
                default_base_iri=base_iri,
                llm_result=cg_result,
                default_parent_iri=parent_iri,
            )
            if cg_audit or concept_iris:
                print(
                    f"[stage4-G] concept-grouping re-homed {len(cg_audit)} class(es) "
                    f"under {len(concept_iris)} new concept class(es)"
                )
                created.extend(concept_iris)

    counts_after = folder_io.count_entities(out_ontology)
    print(f"[{operation}] entity counts: before={counts_before}, after={counts_after}")

    folder_io.write_merged_json(version_dir, out_ontology)
    ontology_export.write_owl(out_ontology, version_dir / folder_io.MERGED_OWL)
    versioning.write_manifest(
        version_dir,
        operation=operation,
        parent_version=input_folder,
        input_documents=sorted(documents_dir.rglob("*")) if documents_dir.exists() else [],
        model_ids={
            task: f"{spec['provider']}/{spec['model']}"
            for task, spec in router._tasks.items()
        },
        extra={
            "llm_total_cost_usd": round(router.total_cost_usd, 6),
            "max_hops_used": effective_hops,
        },
    )
    versioning.write_stats(
        version_dir,
        {
            "before": counts_before,
            "after": counts_after,
            "created_classes": created,
            "llm_total_cost_usd": round(router.total_cost_usd, 6),
        },
    )
    return version_dir

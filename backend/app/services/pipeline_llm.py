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
    add_new_classes_from_match_not_found,
    add_new_relations_from_match_results,
    collect_full_class_hierarchy,
    collect_related_class_iris,
    expand_with_relationship_partners,
    extract_detected_iris,
    extract_json_from_output,
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


_RETRY_AFTER_RE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort: pull `Please try again in Xs` out of a Groq/OpenAI 429
    message. Returns None if not a rate-limit error or if no hint found."""
    msg = str(exc)
    if "rate_limit" not in msg.lower() and "429" not in msg:
        return None
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


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
    their N-hop neighborhood. Strips heavy fields (raw_axiom_triples) so the
    Stage-2 prompt stays compact."""
    classes = loaded_ontology.get("classes_dict", {})
    if not detected_iris:
        return {}
    # collect_related_class_iris builds its own graph internally; no need to
    # call build_class_graph separately.
    relevant = collect_related_class_iris(classes, list(detected_iris), max_hops=max_hops)
    out: dict[str, Any] = {}
    keep_fields = ("name", "iri", "labels", "comments", "descriptions", "superclasses")
    for iri in relevant:
        rec = classes.get(iri)
        if rec is None:
            continue
        out[iri] = {k: rec.get(k) for k in keep_fields if k in rec}
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
) -> tuple[dict[str, Any], list[str], list[str], list[dict]]:
    """Add proposed classes from MATCH NOT FOUND, then add proposed
    object-property relations from MATCH NOT FOUND RELATIONS.

    Order matters: classes go in first so the relation-injection step can
    resolve a DOMAIN/RANGE label that refers to a class proposed in the
    same LLM run.

    Returns:
      (extended_ontology, created_class_iris, created_property_iris, skipped_relations)
    """
    extended, created_classes = add_new_classes_from_match_not_found(
        loaded_ontology=loaded_ontology,
        match_results=match_results,
        new_class_base_iri=base_iri,
        default_parent_iri=default_parent_iri,
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
    return extended, created_classes, list(created_props), skipped


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

    print(f"[stage2] proposing matches+new for {len(chunks)} chunk(s) (OpenAI)")
    stage2_results: list[dict[str, Any] | None] = await asyncio.gather(
        *[_propose_one(i, c, iris) for i, (c, iris) in enumerate(zip(chunks, stage1_results, strict=False))]
    )
    valid = [r for r in stage2_results if r]
    print(f"[stage2] {len(valid)}/{len(chunks)} chunks produced a usable JSON response")

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
        out_ontology, created, created_props, skipped_rels = _apply_expand(
            out_ontology, deduped, base_iri, parent_iri
        )
        print(
            f"[stage4] created {len(created)} new classes from MATCH NOT FOUND, "
            f"{len(created_props)} new relations from MATCH NOT FOUND RELATIONS, "
            f"{len(skipped_rels)} relation(s) skipped (unresolved endpoints)"
        )
        if skipped_rels:
            for s in skipped_rels:
                print(f"[stage4]   skipped relation: {s.get('reason')} -> {s.get('relation')}")

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

"""Centralized LLM prompts.

Every prompt is a function returning `(system, user)`. Functions are
domain-agnostic (no biomedical keywords baked in) and JSON-mode-friendly.

Three prompts in Phase 1:
    chunk_classification — Stage 1: Groq-cheap "which top-level branches
        is this text relevant to?" — returns a short IRI list.
    class_identification_and_expansion — Stage 2: OpenAI focused matching
        of a doc chunk against a sliced ontology, plus MATCH NOT FOUND
        proposals for new classes/relationships.
    match_dedup — Stage 3: collapse duplicate proposals across chunks.

Ported from reference/kg_populationv5.ipynb cells 33 (class_id+expansion)
and 39 (dedup); chunk_classification is new (sized for Groq's 70B).
"""

from __future__ import annotations

import json
from typing import Any


def chunk_classification(
    top_level_branches: list[dict[str, Any]],
    text_chunk: str,
) -> tuple[str, str]:
    """Stage 1: given the top-level ontology branches (a few hundred classes
    at most), return the IRIs of branches relevant to this text chunk."""
    system = (
        "You are an expert ontology curator. Given a short list of ontology "
        "top-level branches (each one is a top-level class with a label and "
        "optional description) and a passage of text, return the IRIs of every "
        "branch that could plausibly contain content related to the text.\n\n"
        "Favor recall over precision: include any branch that the text touches "
        "even tangentially. Examples of branches to KEEP even when only "
        "incidentally mentioned:\n"
        "  - Geographic branches (countries, regions, continents) whenever the "
        "text names a place.\n"
        "  - Temporal branches (year, duration, recurring interval, day of week, "
        "time zone) whenever the text mentions a year, deadline, period, or "
        "rate of change.\n"
        "  - Regulatory/policy/economic branches whenever the text frames its "
        "topic in those terms.\n"
        "Only drop a branch if it is clearly unrelated to anything in the text. "
        "If the text is genuinely irrelevant to the entire ontology, return an "
        "empty list.\n\n"
        "Output strict JSON in the shape:\n"
        '{"relevant_iris": ["<iri>", "<iri>", ...]}\n'
        "No prose. No comments. Only the JSON object."
    )
    branches_repr = json.dumps(top_level_branches, ensure_ascii=False)
    user = (
        f"BRANCHES:\n{branches_repr}\n\n"
        f"TEXT TO CLASSIFY:\n{text_chunk}\n\n"
        "Return JSON: {\"relevant_iris\": [...]}"
    )
    return system, user


def class_identification_and_expansion(
    ontology_slice: dict[str, Any],
    text_chunk: str,
    suggested_new_classes: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Stage 2: given a sliced sub-ontology (only the classes relevant to a
    chunk, per Stage-1 narrowing) plus the chunk's text, match concepts to
    existing classes and propose new ones for unmatched concepts."""
    suggested_block = ""
    if suggested_new_classes:
        suggested_block = (
            "\n\nSUGGESTED NEW CLASSES (consider but don't blindly accept):\n"
            + json.dumps(suggested_new_classes, ensure_ascii=False)
        )

    system = (
        "You are an expert ontology curator. You will be given a slice of an "
        "ontology (DATA_CLASSES, a JSON dict keyed by class IRI) and a passage "
        "of TEXT. Your job:\n"
        "  1. Identify every passage substring that maps to an existing class "
        "in DATA_CLASSES. The IRI you return MUST be an exact key of the "
        "DATA_CLASSES dict.\n"
        "  2. For every distinct concept in the TEXT that has NO matching "
        "existing class, propose a new class with a clear LABEL and "
        "DESCRIPTION.\n"
        "  3. Identify RELATIONS between classes that the TEXT asserts or "
        "implies. Both endpoints may be existing classes (use their LABEL or "
        "the local name from their IRI) OR classes you just proposed in step "
        "2 (use the same LABEL you gave them). Skip relations whose endpoints "
        "you can't identify -- do not invent classes here.\n\n"
        "Review thoroughly -- do not stop after a few matches. When in doubt, "
        "INCLUDE the match. Specifically:\n"
        "  - Geographic mentions (countries, regions, continents, cities) MUST "
        "be matched to existing geographic classes if any are present in "
        "DATA_CLASSES. 'China' -> matches Country/Asia; 'ASEAN' -> matches "
        "Region/SoutheastAsia; etc.\n"
        "  - Temporal mentions (years like '2030', durations like 'next "
        "decade', recurring intervals like 'annual', day-of-week references) "
        "MUST be matched to existing temporal classes if any are present.\n"
        "  - Regulatory, economic, and policy concepts should be matched even "
        "if the text discusses them indirectly.\n"
        "Only emit a MATCH NOT FOUND proposal when NO existing class in "
        "DATA_CLASSES plausibly fits the concept.\n\n"
        "For relations, however, stay precision-biased: prefer concrete, "
        "asserted relations over speculative ones, and skip a relation if "
        "either endpoint cannot be cleanly identified as an existing or "
        "newly-proposed class.\n\n"
        "Output strict JSON in the shape:\n"
        "{\n"
        '  "MATCHES FOUND": ['
        '    {"IRI": "<exact iri from DATA_CLASSES>", "TEXT_SNIPPET": "<excerpt from TEXT>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND": ['
        '    {"LABEL": "<short label>", "DESCRIPTION": "<one or two sentences>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND RELATIONS": ['
        '    {"LABEL": "<verb-phrase relation name, e.g. treats>",'
        ' "DESCRIPTION": "<one-sentence statement of the relation>",'
        ' "DOMAIN": "<label or IRI of the source class>",'
        ' "RANGE": "<label or IRI of the target class>"}, ...'
        "  ]\n"
        "}\n"
        "No prose. No comments. Only the JSON object."
        + suggested_block
    )
    ontology_repr = json.dumps(ontology_slice, ensure_ascii=False, default=str)
    user = (
        f"DATA_CLASSES:\n{ontology_repr}\n\n"
        f"TEXT TO EVALUATE:\n{text_chunk}\n\n"
        'Return JSON: {"MATCHES FOUND": [...], "MATCH NOT FOUND": [...], '
        '"MATCH NOT FOUND RELATIONS": [...]}'
    )
    return system, user


def match_dedup(merged_match_results: dict[str, Any]) -> tuple[str, str]:
    """Stage 3: after merging Stage-2 outputs from many chunks, collapse
    duplicate `MATCH NOT FOUND` proposals (same concept with slightly
    different labels), dedupe `MATCH NOT FOUND RELATIONS`, and drop
    proposals that turn out to overlap with existing `MATCHES FOUND`."""
    system = (
        "You are deduplicating proposed ontology classes and relations. "
        "You will receive a JSON object with three keys:\n"
        "  - \"MATCHES FOUND\": matches against the existing ontology -- "
        "DO NOT modify these.\n"
        "  - \"MATCH NOT FOUND\": proposed new classes that may contain "
        "duplicates and overlaps with MATCHES FOUND.\n"
        "  - \"MATCH NOT FOUND RELATIONS\": proposed new object-property "
        "relations (LABEL + DOMAIN + RANGE) that may contain duplicates.\n\n"
        "Your tasks:\n"
        "  1. From MATCH NOT FOUND, remove any entry whose concept already "
        "exists in MATCHES FOUND (same idea, even if labeled differently).\n"
        "  2. From MATCH NOT FOUND, collapse entries that propose the same "
        "concept under different labels into a single entry (pick the "
        "clearest label; merge descriptions concisely).\n"
        "  3. From MATCH NOT FOUND RELATIONS, collapse entries that propose "
        "the same relation (same LABEL + same DOMAIN + same RANGE, or "
        "trivially-paraphrased verb labels) into a single entry. Different "
        "DOMAIN/RANGE pairs are NOT duplicates even with the same LABEL.\n"
        "  4. Do NOT add new entries to any list.\n"
        "  5. Do NOT modify any MATCHES FOUND entries.\n\n"
        "Output strict JSON with the same three keys, no prose, no comments."
    )
    user = (
        "INPUT:\n"
        + json.dumps(merged_match_results, ensure_ascii=False, default=str)
        + "\n\nReturn the deduplicated JSON object."
    )
    return system, user


# Public registry so callers can look up a prompt builder by task name.
PROMPTS = {
    "chunk_classification": chunk_classification,
    "class_proposal": class_identification_and_expansion,
    "match_dedup": match_dedup,
}

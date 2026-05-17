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
        "optional description) and a passage of text, return ONLY the IRIs "
        "of branches that are clearly relevant to the text.\n\n"
        "Be selective. Aim for the smallest set of branches that fully covers "
        "the text. If the text is irrelevant to the ontology, return an empty "
        "list.\n\n"
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
        "DESCRIPTION.\n\n"
        "Review thoroughly — do not stop after a few matches. Prefer accurate "
        "matches over speculative ones.\n\n"
        "Output strict JSON in the shape:\n"
        "{\n"
        '  "MATCHES FOUND": ['
        '    {"IRI": "<exact iri from DATA_CLASSES>", "TEXT_SNIPPET": "<excerpt from TEXT>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND": ['
        '    {"LABEL": "<short label>", "DESCRIPTION": "<one or two sentences>"}, ...'
        "  ]\n"
        "}\n"
        "No prose. No comments. Only the JSON object."
        + suggested_block
    )
    ontology_repr = json.dumps(ontology_slice, ensure_ascii=False, default=str)
    user = (
        f"DATA_CLASSES:\n{ontology_repr}\n\n"
        f"TEXT TO EVALUATE:\n{text_chunk}\n\n"
        'Return JSON: {"MATCHES FOUND": [...], "MATCH NOT FOUND": [...]}'
    )
    return system, user


def match_dedup(merged_match_results: dict[str, Any]) -> tuple[str, str]:
    """Stage 3: after merging Stage-2 outputs from many chunks, collapse
    duplicate `MATCH NOT FOUND` proposals (same concept with slightly
    different labels) and drop proposals that turn out to overlap with
    existing `MATCHES FOUND` entries."""
    system = (
        "You are deduplicating proposed ontology classes. You will receive a "
        "JSON object with two keys: \"MATCHES FOUND\" (matches against the "
        "existing ontology — DO NOT modify these) and \"MATCH NOT FOUND\" "
        "(proposed new classes that may contain duplicates and overlaps with "
        "MATCHES FOUND).\n\n"
        "Your tasks:\n"
        "  1. From MATCH NOT FOUND, remove any entry whose concept already "
        "exists in MATCHES FOUND (same idea, even if labeled differently).\n"
        "  2. From MATCH NOT FOUND, collapse entries that propose the same "
        "concept under different labels into a single entry (pick the "
        "clearest label; merge descriptions concisely).\n"
        "  3. Do NOT add new entries to MATCH NOT FOUND.\n"
        "  4. Do NOT modify any MATCHES FOUND entries.\n\n"
        "Output strict JSON with the same two keys, no prose, no comments."
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

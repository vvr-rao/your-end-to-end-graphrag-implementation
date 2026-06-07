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
        "existing class, propose a new class with a clear LABEL, DESCRIPTION, "
        "and PARENT_LABEL (the most specific class in DATA_CLASSES that this "
        "new class should be a kind of -- e.g. 'Country' for 'Washington', "
        "'Manufacturer' for 'BMW', 'Chemical Element' for 'helium'). If no "
        "suitable parent exists in DATA_CLASSES, you MUST propose the parent "
        "as ANOTHER MATCH NOT FOUND entry in the same response (chain rule) "
        "and use its LABEL as the PARENT_LABEL. Only use PARENT_LABEL='NONE' "
        "if the concept genuinely has no natural parent.\n"
        "  3. Identify RELATIONS between classes the TEXT asserts or implies. "
        "Both endpoints may be existing classes (use their LABEL or IRI) or "
        "classes you just proposed in step 2 (use the same LABEL).\n"
        "  4. INSTANCES (named individuals). For specific time points / "
        "time periods (e.g. 'Jan 2004', 'January 2004', 'Jan04', '2030', "
        "'Q3 2025', 'next decade') and other PROPER-NOUN named individuals "
        "the text references (specific events, specific documents, named "
        "incidents that are one-of-a-kind), emit a MATCH NOT FOUND INSTANCES "
        "entry instead of a MATCH NOT FOUND class entry. Each instance "
        "entry must carry:\n"
        "       - LABEL: the surface form as found in the text\n"
        "       - CANONICAL_FORM: a single canonical form so equivalent "
        "surface forms ('Jan 2004' / 'Jan04' / 'January 2004') collapse to "
        "ONE instance. Pick the most readable canonical form (e.g. "
        "'January 2004').\n"
        "       - TYPE_LABEL: the most specific class in DATA_CLASSES (or a "
        "MATCH NOT FOUND label) that this individual is an instance of. "
        "For time points prefer time classes if available (Year, Month, "
        "DurationDescription, TemporalEntity, etc.). For named events, "
        "prefer 'Event' / 'Crisis' / 'Conflict' / etc.\n"
        "       - DESCRIPTION: a brief sentence of context.\n"
        "     Rule of thumb: classes are KINDS OF things ('Year', 'Crisis', "
        "'Country'); instances are SPECIFIC things ('January 2004', 'Iran "
        "war 2025', 'Kingdom of Saudi Arabia'). Do NOT create a new class "
        "for each year mentioned in the text -- create instances of Year.\n"
        "\n"
        "Review thoroughly -- do not stop after a few matches. When in doubt, "
        "INCLUDE the match. Specifically:\n"
        "  - Any proper-noun PLACE -- including political-administrative "
        "places (country, region, province, state, county, city, town, "
        "district, neighborhood) AND natural / physical-geography features "
        "(continent, ocean, sea, gulf, bay, strait, channel, isthmus, "
        "peninsula, island, archipelago, mountain, mountain range, valley, "
        "river, lake, desert, plateau, coast, basin) -- MUST be matched to "
        "an existing geographic class in DATA_CLASSES when one is present. "
        "Use the most specific matching class as PARENT_LABEL; fall back "
        "to the most general one (e.g. GeographicEntity / Place) when no "
        "specific kind-of class exists. Examples of the resolution pattern "
        "(your DATA_CLASSES will differ): a country name -> a Country "
        "class; a continent name -> a Continent class; a body of water -> "
        "a Sea / Ocean / Lake / River class; a landform -> an Island / "
        "Mountain / Peninsula / Strait class. NEVER root a place under "
        "owl:Thing when a geographic class is available, even if the chunk "
        "discusses the place in a non-geographic context (e.g. a strait "
        "named in a shipping story is still a Strait).\n"
        "  - Temporal mentions (years like '2030', durations like 'next "
        "decade', recurring intervals like 'annual', day-of-week references) "
        "MUST be matched to existing temporal classes if any are present.\n"
        "  - Regulatory, economic, and policy concepts should be matched even "
        "if the text discusses them indirectly.\n"
        "Only emit a MATCH NOT FOUND proposal when NO existing class in "
        "DATA_CLASSES plausibly fits the concept.\n\n"
        "SELF-CONSISTENCY (critical, do NOT skip):\n"
        "  - Every DOMAIN and RANGE label you use in MATCH NOT FOUND "
        "RELATIONS MUST appear either (a) as an exact label/IRI in "
        "DATA_CLASSES, OR (b) as a LABEL in your own MATCH NOT FOUND list. "
        "If a relation references a concept that is not in DATA_CLASSES, "
        "you MUST propose that concept as a MATCH NOT FOUND class first.\n"
        "  - When the chunk introduces multiple concepts whose labels share "
        "a common stem or topic (e.g. 'helium', 'helium market', 'helium "
        "supply', 'helium price'; or 'BMW', 'Honda', 'Hyundai' as siblings "
        "of 'CarManufacturer'), propose has-X / part-of / sibling-of "
        "relations between them using the chunk text as evidence. Stay "
        "grounded -- don't invent relations the text doesn't support.\n"
        "  - When the text mentions a concept under multiple surface forms "
        "(e.g. 'Washington', 'Washington DC', 'Washington D.C.'), emit ONE "
        "MATCH NOT FOUND entry using the most canonical form. If two surface "
        "forms refer to different specific entities (e.g. 'Washington' the "
        "state vs 'Washington' the city), distinguish them with a qualifier "
        "in the label.\n\n"
        "For relations, stay precision-biased on endpoints: skip a relation "
        "only if either endpoint cannot be identified as existing OR proposed "
        "(which means you should have proposed it -- see self-consistency).\n\n"
        "Output strict JSON in the shape:\n"
        "{\n"
        '  "MATCHES FOUND": ['
        '    {"IRI": "<exact iri from DATA_CLASSES>", "TEXT_SNIPPET": "<excerpt from TEXT>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND": ['
        '    {"LABEL": "<short canonical label>", "DESCRIPTION": "<one or two sentences>",'
        ' "PARENT_LABEL": "<label or IRI of most specific parent in DATA_CLASSES'
        " or another MATCH NOT FOUND entry, or 'NONE'>\"}, ..."
        "  ],\n"
        '  "MATCH NOT FOUND RELATIONS": ['
        '    {"LABEL": "<verb-phrase relation name, e.g. treats>",'
        ' "DESCRIPTION": "<one-sentence statement of the relation>",'
        ' "DOMAIN": "<label or IRI of the source class (must exist or be proposed)>",'
        ' "RANGE": "<label or IRI of the target class (must exist or be proposed)>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND INSTANCES": ['
        '    {"LABEL": "<surface form from text>",'
        ' "CANONICAL_FORM": "<single canonical form>",'
        ' "TYPE_LABEL": "<label of class this is an instance of, from DATA_CLASSES or MATCH NOT FOUND>",'
        ' "DESCRIPTION": "<one-sentence context>"}, ...'
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
        '"MATCH NOT FOUND RELATIONS": [...], "MATCH NOT FOUND INSTANCES": [...]}'
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
        "  - \"MATCH NOT FOUND\": proposed new classes (each carries LABEL, "
        "DESCRIPTION, and PARENT_LABEL) that may contain duplicates and "
        "overlaps with MATCHES FOUND.\n"
        "  - \"MATCH NOT FOUND RELATIONS\": proposed new object-property "
        "relations (LABEL + DOMAIN + RANGE) that may contain duplicates.\n"
        "  - \"MATCH NOT FOUND INSTANCES\" (may be missing): proposed "
        "named individuals (each carries LABEL, CANONICAL_FORM, TYPE_LABEL, "
        "DESCRIPTION).\n\n"
        "Your tasks:\n"
        "  1. From MATCH NOT FOUND, remove any entry whose concept already "
        "exists in MATCHES FOUND (same idea, even if labeled differently).\n"
        "  2. From MATCH NOT FOUND, collapse entries that propose the same "
        "concept under different labels into a single entry. Pick the "
        "clearest CANONICAL short label; merge DESCRIPTIONs concisely; "
        "preserve the most specific PARENT_LABEL.\n"
        "     Variant patterns to collapse aggressively:\n"
        "       - Proper-noun variants: 'Washington' / 'Washington DC' / "
        "'Washington, D.C.' -> ONE entry (they refer to the same entity "
        "unless the input genuinely distinguishes the state from the city).\n"
        "       - Country aliases: 'United States' / 'USA' / 'US' -> one.\n"
        "       - Acronym + expansion: 'ASEAN' / 'Association of Southeast "
        "Asian Nations' -> one entry; keep the acronym as the label.\n"
        "       - Plural/singular and tense variants of relation labels.\n"
        "  3. From MATCH NOT FOUND RELATIONS, collapse entries that propose "
        "the same relation (same LABEL + same DOMAIN + same RANGE, or "
        "trivially-paraphrased verb labels) into a single entry. Different "
        "DOMAIN/RANGE pairs are NOT duplicates even with the same LABEL.\n"
        "  4. COMMON-PARENT INFERENCE (narrowly scoped exception to "
        "'do not add entries'): if you see THREE OR MORE proposed classes "
        "in MATCH NOT FOUND that obviously share a common parent (e.g. "
        "BMW + Honda + Hyundai are all car manufacturers; or "
        "Germany + France + Italy are all countries; or "
        "helium + neon + argon are all noble gases), AND that parent does "
        "not already exist in MATCHES FOUND, you MAY:\n"
        "       (a) add ONE new MATCH NOT FOUND entry for the parent "
        "(LABEL='CarManufacturer' or similar canonical name, DESCRIPTION "
        "explaining what kind of thing it is, PARENT_LABEL of its own); AND\n"
        "       (b) update each child's PARENT_LABEL to the new parent's "
        "LABEL.\n"
        "     Use this sparingly -- only when the grouping is obvious from "
        "the labels themselves. Do NOT invent parents for groups of 2 or for "
        "weakly-related concepts.\n"
        "  5. From MATCH NOT FOUND INSTANCES, collapse entries that share "
        "the same CANONICAL_FORM (case-insensitive after stripping "
        "punctuation) into ONE entry. Merge descriptions; pick the most "
        "specific TYPE_LABEL. Example collapses: 'Jan 2004' / 'Jan04' / "
        "'January 2004' all share CANONICAL_FORM='January 2004' -> one "
        "instance.\n"
        "  6. Do NOT modify any MATCHES FOUND entries.\n"
        "  7. Do NOT add new entries for any reason other than rule 4.\n\n"
        "Output strict JSON with all four keys present (use [] if empty), "
        "no prose, no comments."
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

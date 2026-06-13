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
        "     CRITICAL: Do NOT use a geography class (GeographicEntity, "
        "Place, Country, Region, Continent, City, or any other class in "
        "the geography ontology) as PARENT_LABEL for a concept that is "
        "NOT itself a geographic place or landform. For example, things "
        "like 'EV bus', 'Subsidy', 'Trade flow', 'Ministry of Finance', "
        "'Supply shock', 'Subsidy', 'Hydrogen' are NOT geographic. If "
        "you can't find a non-geographic parent that fits, use "
        "PARENT_LABEL='NONE' -- a later step will assign a proper "
        "top-level concept.\n"
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
        "ENTITY-TYPE RULES (critical -- get these right):\n"
        "  1. NAMED INDIVIDUAL PEOPLE (politicians, executives, public "
        "figures, researchers, named officials -- e.g. Donald Trump, "
        "Vladimir Putin, Xi Jinping, Tim Cook): emit as MATCH NOT FOUND "
        "INSTANCES.\n"
        "       - TYPE_LABEL = the foaf:Person class IRI if it appears "
        "in DATA_CLASSES (look for 'http://xmlns.com/foaf/0.1/Person' "
        "or any class labeled 'Person' / 'Agent'); else use 'Person' as "
        "the label and let the deterministic pass anchor it.\n"
        "       - LABEL = the person's full name as written in the "
        "text. CANONICAL_FORM = the same name in CamelCase "
        "(DonaldTrump, VladimirPutin, XiJinping).\n"
        "       - If the text mentions the person's ROLE (President, "
        "Prime Minister, CEO, Minister of X, Secretary of Y), emit a "
        "SEPARATE MATCH NOT FOUND RELATION with LABEL='hasRole', "
        "DOMAIN=<person canonical form>, RANGE=<role label>.\n"
        "       - NEVER create a CLASS for a specific individual person. "
        "NEVER bake the role into the instance label "
        "('President Donald Trump' is WRONG; emit 'Donald Trump' as the "
        "instance + a hasRole relation to 'President').\n"
        "  2. COMPANIES + GOVERNMENT BODIES + NGOS + AGENCIES + "
        "ASSOCIATIONS. Any entity whose name has a corporate suffix or "
        "agency pattern (Inc, Ltd, Corp, Corporation, Industries, "
        "Petrochemicals, Petroleum, Company, Co., Co. Ltd., Group, plc, "
        "GmbH, S.A., Holdings, Bank, University, Ministry of X, "
        "Department of X, Bureau of X, NGO, Council, Federation, "
        "Association, OPEC-style acronym) is an ORGANIZATION.\n"
        "       - PARENT_LABEL = the foaf:Organization / "
        "org:Organization class from DATA_CLASSES if present, else "
        "'Organization'. NEVER under Material, Helium, "
        "ChemicalProcessSystem, ProcessUnit, Infrastructure, or any "
        "non-org class.\n"
        "       - Examples: 'Samsung Electronics' -> Organization. "
        "'Air Products and Chemicals' -> Organization. 'Saudi Aramco "
        "Total Refinery & Petrochemicals Co.' -> Organization. 'Sinopec "
        "Maoming Company' -> Organization.\n"
        "  3. PHYSICAL FACILITIES (refinery complexes, plants, ports, "
        "terminals) named after a company or location -- e.g. 'Jamnagar "
        "Refinery Complex', 'GS-Caltex Yeosu Refinery', 'Hormuz Oil "
        "Terminal'. These are INFRASTRUCTURE, not process units.\n"
        "       - PARENT_LABEL = 'Infrastructure'.\n"
        "       - Emit a MATCH NOT FOUND RELATION linking the facility "
        "to its owner: LABEL='operatedBy' (or 'owner'), DOMAIN=<facility>, "
        "RANGE=<organization>.\n"
        "  4. EVENTS named after a place or entity. Patterns: '<X> "
        "crisis', '<X> closure', '<X> war', '<X> disruption', '<X> "
        "shortage', '<X> conflict', '<X> incident', '<X> shutdown', "
        "'<X> blockage', '<X> attack', '<X> embargo', '<X> escalation', "
        "'<X> summit'.\n"
        "       - PARENT_LABEL = 'Event' (or 'Crisis' / 'Conflict' "
        "subclass if present in DATA_CLASSES). EVEN IF the entity "
        "contains a geographic name: 'Strait of Hormuz crisis' -> "
        "Event, NOT Strait. 'Russia-Ukraine war' -> Event, NOT Russia.\n"
        "  5. ROLE TYPES (kinds of positions): 'President', "
        "'PrimeMinister', 'CEO', 'Chairman', 'Secretary of <X>', "
        "'Minister of <Y>', 'Director', 'Founder'.\n"
        "       - PARENT_LABEL = the org:Role class IRI if present in "
        "DATA_CLASSES, else 'Role'. Roles are KINDS OF positions; "
        "specific people HOLD them via hasRole.\n"
        "       - NEVER conflate a role type with a person. 'President' "
        "is a role; 'Donald Trump' is the person who holds it. Two "
        "separate entities.\n"
        "  6. GEOGRAPHIC FEATURES: a class is a geographic place only "
        "when its label IS a place name with no event / process / role "
        "modifier. 'Strait of Hormuz' = yes. 'Strait of Hormuz crisis' "
        "= no (Event). 'Suez Canal' = yes. 'Suez Canal blockage' = no "
        "(Event).\n\n"
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


def concept_grouping(orphan_classes: list[dict[str, Any]]) -> tuple[str, str]:
    """Layer G: ONE call per prune-expand. Given a list of orphan classes
    (no parent except owl:Thing, in the synthesized namespace), propose
    a small set of high-level concept classes that group them and assign
    each orphan to one concept.

    Each entry in `orphan_classes` is expected to be `{LABEL, DESCRIPTION}`.
    The prompt deliberately gives ILLUSTRATIVE concept examples but does
    not bake in a fixed taxonomy -- the LLM picks whatever buckets fit
    the actual orphan set.
    """
    system = (
        "You are an ontology curator. You will be given a JSON list of "
        "orphan classes (each has LABEL and DESCRIPTION). These classes "
        "currently have no parent and need to be organised under a small "
        "set of high-level concept classes.\n\n"
        "Your task:\n"
        "  1. Propose between 5 and 15 high-level concept classes that "
        "would together cover the orphan set well. The concept labels "
        "should be canonical CamelCase nouns describing a KIND of thing. "
        "Illustrative bucket examples (use as inspiration, NOT as a "
        "fixed list):\n"
        "       - Material               (substances, chemicals, "
        "compounds: helium, fertilizer, oil, urea, polymer)\n"
        "       - NaturalResource        (extracted natural inputs: "
        "crude oil, natural gas, ore, timber)\n"
        "       - TechnologyConcept      (technical artifacts/processes: "
        "semiconductor, chip, wafer, AI, battery)\n"
        "       - Industry               (kinds of economic activity: "
        "Agriculture, Manufacturing, Mining, Healthcare, Energy, "
        "Transportation, Finance, Aerospace, Tourism, Construction)\n"
        "       - Infrastructure         (built things: pipeline, port, "
        "refinery, grid, factory, station, terminal)\n"
        "       - SupplyChainConcept     (logistics + trade flows: "
        "supply chain, shipping route, freight corridor)\n"
        "       - EconomicConcept        (economic phenomena: price, "
        "market, subsidy, inflation, demand, recession)\n"
        "       - PolicyConcept          (rules / frameworks: regulation, "
        "law, treaty, sanction, tariff, policy initiative)\n"
        "       - Organization           (companies, agencies, "
        "governments, NGOs, ASSOCIATIONS only -- NOT activity domains: "
        "FAO, World Bank, BMW, Iran government, OPEC)\n"
        "       - Person / Role          (named individuals + role types: "
        "farmer, policymaker, consumer)\n"
        "       - Event                  (named happenings: Iran war, "
        "Hormuz crisis, COVID-19 pandemic)\n"
        "       - Process                (named ongoing activities: "
        "manufacturing, refining, photosynthesis, decarbonization)\n"
        "       - DomainConcept          (CATCH-ALL fallback: use this "
        "for genuinely broad/abstract concepts that don't cleanly fit "
        "any of the above. Better than forcing a bad placement.)\n"
        "     Pick the actual buckets that best fit the corpus content -- "
        "don't force one of the examples if the data wants different "
        "ones.\n"
        "  2. Assign each orphan to EXACTLY ONE concept. Use the orphan's "
        "LABEL verbatim (case-insensitive match is fine for resolution).\n"
        "  3. Skip an orphan only if it genuinely fits none of your "
        "proposed concepts.\n"
        "  4. Avoid extremely narrow concepts (one or two orphans each) "
        "and avoid extremely broad concepts (everything goes to 'Thing'). "
        "Aim for buckets of 5-100 orphans each.\n\n"
        "CRITICAL placement rules:\n"
        "  - Industry vs Organization: an INDUSTRY is a kind of economic "
        "activity ('Agriculture', 'Mining', 'Healthcare'). An "
        "ORGANIZATION is a specific entity that operates within an "
        "industry ('Cargill', 'WHO', 'BMW'). Do NOT place 'Agriculture' "
        "under Organization -- Agriculture is an Industry.\n"
        "  - If you find yourself proposing 'OrganizationConcept' as a "
        "concept that covers many activity domains, you probably want "
        "'Industry' or 'DomainConcept' instead.\n"
        "  - Use 'DomainConcept' as the fallback rather than forcing a "
        "concept into a bucket that doesn't fit. The downstream system "
        "treats DomainConcept as a legitimate top-level grouping.\n\n"
        "Output strict JSON in the shape:\n"
        "{\n"
        '  "TOP_LEVEL_CONCEPTS": ['
        '    {"LABEL": "<CamelCase concept name>",'
        ' "DESCRIPTION": "<one-sentence definition>"}, ...'
        "  ],\n"
        '  "ASSIGNMENTS": ['
        '    {"CLASS_LABEL": "<exact orphan LABEL>",'
        ' "CONCEPT_LABEL": "<a LABEL from TOP_LEVEL_CONCEPTS>"}, ...'
        "  ]\n"
        "}\n"
        "No prose. No comments. Only the JSON object."
    )
    user = (
        "ORPHAN_CLASSES:\n"
        + json.dumps(orphan_classes, ensure_ascii=False, default=str)
        + '\n\nReturn JSON: {"TOP_LEVEL_CONCEPTS": [...], "ASSIGNMENTS": [...]}'
    )
    return system, user


def compact_description(class_batch: list[dict[str, Any]]) -> tuple[str, str]:
    """One-time class-metadata compression. Given a small batch of class
    records (each {iri, name, labels, comments, descriptions}), return a
    short `compact_description` per class -- the SAME semantic content as
    the original descriptions+comments but typically 70-85% smaller.

    The output is stored on each class record once (in merged.json) and
    reused by every subsequent Stage 2 call's `_slice_ontology`, which
    ships `compact_description` instead of the raw fields. Cuts per-class
    slice footprint roughly in half.

    The model used is gpt-4o-mini (configured in models.yaml) -- 16x
    cheaper than gpt-4.1 in/out and plenty strong for "rewrite this in
    <= 15 words.\""""
    system = (
        "You are an ontology curator. For each class in the input batch, "
        "produce a SHORT description (at most 15 words, one sentence, no "
        "trailing period needed) that captures what kind of thing the "
        "class is. Use the existing `descriptions` + `comments` as your "
        "primary input; the `labels` and `name` are also useful. "
        "Examples:\n"
        "  - 'A material stream representing the flow between two unit "
        "operations in a chemical process system. Used to capture "
        "intermediate products that pass through reactors, separators, "
        "or heat exchangers.' -> 'Material stream between two unit "
        "operations in a chemical process.'\n"
        "  - 'A country located in the Middle East geopolitical region.' "
        "-> 'A Middle Eastern country.'\n"
        "Constraints:\n"
        "  - <= 15 words per class.\n"
        "  - Preserve the semantic kind-of relationship if obvious from "
        "the source text (e.g. 'A country', 'A material stream', 'A "
        "year').\n"
        "  - Drop boilerplate like 'Country/country-like geography class. "
        "Document-parsed mentions of this geography can be...' since it "
        "adds no semantic info beyond the type.\n"
        "  - If the source text is empty or trivial, emit a one-word "
        "compact_description derived from the LABEL (e.g. 'Helium' -> "
        "'Helium').\n\n"
        "Output strict JSON in the shape:\n"
        '{"results": [{"iri": "<exact input iri>", "compact_description": "<short string>"}, ...]}'
        "\n"
        "Include exactly one results entry per input class, in the same "
        "order. No prose, no comments."
    )
    user = (
        "CLASSES:\n"
        + json.dumps(class_batch, ensure_ascii=False, default=str)
        + '\n\nReturn JSON: {"results": [{"iri": "...", "compact_description": "..."}, ...]}'
    )
    return system, user


def classification_audit(items: list[dict[str, Any]]) -> tuple[str, str]:
    """Layer H: post-Stage-4 misclassification audit. Given a batch of
    newly-created classes (each with name, current_parent_label,
    description), decide for each: KEEP the parent, RE_HOME under a
    better parent from the allowed bucket list, or CONVERT_TO_INSTANCE
    when the entity is a specific individual that should be an instance
    rather than a class.

    Each input item: {LABEL, CURRENT_PARENT, DESCRIPTION}.
    Output one decision per item.
    """
    system = (
        "You are an ontology curator reviewing a batch of recently-minted "
        "classes for misclassification. The minting pipeline made best-effort "
        "guesses; some are wrong. Your job is to spot and fix the bad ones.\n\n"
        "For each item you will see:\n"
        "  LABEL          - the class label (a name extracted from documents)\n"
        "  CURRENT_PARENT - the parent class IRI / label assigned during minting\n"
        "  DESCRIPTION    - a short description of what the class represents\n\n"
        "Decide ONE of three actions per item:\n"
        "  KEEP                  - current parent is semantically correct\n"
        "  RE_HOME               - propose a better parent (NEW_PARENT)\n"
        "  CONVERT_TO_INSTANCE   - this entity is a specific INDIVIDUAL, not "
        "a class. Move it to an instance of NEW_PARENT (which IS a class).\n\n"
        "Allowed top-level parent buckets (use the exact label):\n"
        "  Person          - human individuals (only valid as parent for "
        "CONVERT_TO_INSTANCE actions on people-shaped LABELS)\n"
        "  Organization    - companies, agencies, NGOs, governments, "
        "associations. Includes anything with corporate suffix (Inc, Ltd, "
        "Corp, Industries, Petrochemicals, Co., Group, plc, GmbH).\n"
        "  Role            - position types (President, CEO, Minister of X)\n"
        "  Event           - named happenings (X crisis, X war, X closure, "
        "X disruption, X attack, X conflict, X blockage, X embargo, X summit)\n"
        "  Process         - ongoing activities (refining, distillation, "
        "decarbonization)\n"
        "  Material        - substances (helium, urea, oil, steel)\n"
        "  NaturalResource - extracted inputs (crude oil, natural gas, ore)\n"
        "  Infrastructure  - physical facilities (refinery complexes, ports, "
        "terminals, plants, pipelines)\n"
        "  Industry        - kinds of economic activity (Agriculture, "
        "Manufacturing, Mining, Healthcare, Energy)\n"
        "  TechnologyConcept - artifacts / processes (semiconductor, AI, "
        "battery, MRI)\n"
        "  EconomicConcept - market phenomena (price, demand, subsidy, "
        "tariff, inflation)\n"
        "  PolicyConcept   - rules / frameworks (regulation, treaty, ban, "
        "law)\n"
        "  SupplyChainConcept - logistics + trade flows (route, freight "
        "corridor)\n"
        "  GeographicFeature - places named only by location, no event/role "
        "modifier (Strait of Hormuz YES; Strait of Hormuz crisis NO)\n"
        "  DomainConcept   - catch-all for genuinely broad/abstract concepts "
        "that don't fit any other bucket\n\n"
        "CRITICAL PATTERNS (always apply these):\n"
        "  - LABEL looks like a named person (FirstName LastName, "
        "Title + Name) AND CURRENT_PARENT is Person / PersonRole / "
        "PersonOrRole / Role -> CONVERT_TO_INSTANCE, NEW_PARENT=Person.\n"
        "  - LABEL has a corporate suffix (Inc, Ltd, Corp, Corporation, "
        "Industries, Petrochemicals, Co., Group, plc) AND CURRENT_PARENT is "
        "NOT Organization or a sub-Organization -> RE_HOME under Organization.\n"
        "  - LABEL contains an event keyword (crisis, closure, war, "
        "disruption, shortage, conflict, incident, shutdown, blockage, "
        "attack, sanction, embargo, summit, escalation) AND CURRENT_PARENT "
        "is NOT Event or a sub-Event -> RE_HOME under Event.\n"
        "  - LABEL is a refinery / petrochemical complex / oil terminal / "
        "plant (e.g. 'Jamnagar Refinery Complex') AND CURRENT_PARENT is a "
        "process unit / material -> RE_HOME under Infrastructure.\n"
        "  - LABEL is a role TYPE ('President', 'PrimeMinister', 'CEO', "
        "'Minister of X') AND CURRENT_PARENT is Person / not Role -> "
        "RE_HOME under Role.\n\n"
        "Output strict JSON ONLY:\n"
        "{\n"
        '  "DECISIONS": [\n'
        '    {\n'
        '      "LABEL": "<exact LABEL from input>",\n'
        '      "ACTION": "KEEP" | "RE_HOME" | "CONVERT_TO_INSTANCE",\n'
        '      "NEW_PARENT": "<label of new parent if RE_HOME or '
        'CONVERT_TO_INSTANCE; otherwise omit or set null>",\n'
        '      "REASON": "<one short phrase explaining the decision>"\n'
        '    }, ...\n'
        '  ]\n'
        "}\n"
        "No prose. Only the JSON object."
    )
    user = (
        "ITEMS TO REVIEW:\n"
        + json.dumps(items, ensure_ascii=False, default=str)
        + '\n\nReturn JSON: {"DECISIONS": [...]}'
    )
    return system, user


def document_summarize(text: str) -> tuple[str, str]:
    """Pre-pipeline document compression. Given a long source document,
    rewrite it as a denser summary that preserves the ENTITIES,
    RELATIONSHIPS, conceptual buckets, and proper-noun content the
    downstream ontology pipeline cares about. Throws away repetitive
    prose, anecdotes, and editorial commentary.

    Returns plain prose, NOT JSON, because the downstream chunker
    consumes plain text and splits on paragraph boundaries."""
    system = (
        "You are a research assistant compressing a long document into a "
        "much shorter summary that preserves the information needed for "
        "ontology population. Your output will be analyzed by another "
        "system to extract concepts, entities, and relationships.\n\n"
        "The summary MUST preserve:\n"
        "  - Every named entity: countries, regions, cities, places, "
        "geographic features (straits, islands, oceans, mountains, "
        "rivers), organizations, companies, brands, products, people, "
        "studies, reports, regulations, laws, treaties, dates, time "
        "periods, monetary amounts, measurements.\n"
        "  - Every conceptual category mentioned: industries, sectors, "
        "technologies, materials, processes, methods, frameworks.\n"
        "  - All relationships between entities or concepts (X causes Y, "
        "X is part of Y, X impacts Y, X depends on Y, X exports Y, X is "
        "produced in Y, etc.).\n"
        "  - Numerical specifics tied to a named thing (percentages, "
        "dates, quantities tied to an entity).\n\n"
        "What you can drop:\n"
        "  - Repetitive prose / rephrasings of the same fact.\n"
        "  - Anecdotes and illustrative examples that don't introduce "
        "any new entity or relationship.\n"
        "  - Editorial commentary, opinions, transitions, hedges.\n"
        "  - Filler ('in conclusion', 'in summary', 'as we have seen').\n\n"
        "Output the summary as flowing prose paragraphs. Do NOT use "
        "bullet points or markdown headings -- the downstream chunker "
        "is paragraph-first and treats bullets as one giant paragraph. "
        "Use blank lines between paragraphs.\n\n"
        "Target length: 20-50% of the input. Err on the side of "
        "preserving entities + relationships even if that means a "
        "longer summary. Do NOT add anything not in the source text."
    )
    user = (
        "DOCUMENT TO SUMMARIZE:\n\n"
        + text
        + "\n\nReturn ONLY the summary as flowing prose paragraphs. No "
        "preamble, no commentary, no markdown."
    )
    return system, user


def artifact_chunk_extract(text: str) -> tuple[str, str]:
    """Phase 2 Milestone E: extract Claims + Findings + Observations
    from one chunk in a single LLM call (3x cheaper than 3 calls).

    Returns JSON: {
      claims:       [{text, confidence}],
      findings:     [{text, confidence}],
      observations: [{text, confidence}]
    }

    The prompt is corpus-agnostic — it defines each type abstractly
    so the same instructions work on any domain (legal, financial,
    scientific, web search, etc.).
    """
    system = (
        "You extract structured intelligence artifacts from text chunks. "
        "You return ONE JSON object with three keys: claims, findings, "
        "observations. Each is a list of `{text, confidence}` items.\n\n"
        "DEFINITIONS:\n"
        "  - Claim: a factual assertion the text MAKES (e.g. \"X "
        "owns 30% of Y\"). Specific and verifiable.\n"
        "  - Finding: an analytical conclusion or insight (e.g. \"the "
        "trend suggests Z is accelerating\"). Goes beyond raw facts.\n"
        "  - Observation: a raw factual statement directly visible in "
        "the text (e.g. \"price rose 5% in March\"). The most concrete "
        "of the three.\n\n"
        "GUIDELINES:\n"
        "  - 0 to 8 of each type per chunk; only include items the "
        "text actually supports.\n"
        "  - `text` should be the artifact as a standalone sentence, "
        "NOT a quote.\n"
        "  - `confidence` is a float in [0,1] reflecting how directly "
        "the chunk supports the artifact.\n"
        "  - Skip claims / findings / observations that are too vague, "
        "uncertain, or generic to be useful.\n"
        "  - Return ONLY the JSON object — no preamble, no markdown."
    )
    user = (
        "TEXT CHUNK:\n```\n"
        + text
        + "\n```\n\n"
        "Return the JSON now."
    )
    return system, user


def entity_extract(
    chunk_text: str,
    candidate_classes: list[dict[str, str]],
) -> tuple[str, str]:
    """Phase 2 Milestone C: extract named entities from a chunk.

    `candidate_classes` is a list of {iri, label, description} dicts
    -- the top-K classes the chunk's vector matched against
    ontology_classes. The LLM MUST pick a class_iri from this list
    for each entity (caller validates + drops mismatches).

    Returns JSON:
      {entities: [{canonical_name, short_name, class_iri, confidence}]}

    Corpus-agnostic: works on any domain (legal, financial, science,
    web search). No keyword baked in.
    """
    candidates_block_lines: list[str] = []
    for c in candidate_classes:
        label = (c.get("label") or "(unlabelled)").strip()
        descr = (c.get("description") or "").strip()
        descr_short = descr[:100] + ("..." if len(descr) > 100 else "")
        candidates_block_lines.append(
            f"  - {c['iri']} ─ {label}"
            + (f" ─ {descr_short}" if descr_short else "")
        )
    candidates_block = "\n".join(candidates_block_lines)

    system = (
        "You extract NAMED ENTITIES from text chunks. For each entity you find:\n"
        "  - canonical_name: the full proper form (e.g. \"BYD Company Ltd.\", "
        "\"United Kingdom\", \"Donald Trump\")\n"
        "  - short_name: how it was referred to in this chunk (e.g. \"BYD\", \"UK\")\n"
        "  - class_iri: pick ONE IRI from the CANDIDATE CLASSES list below. "
        "Do not invent IRIs. If no candidate is a sensible fit, SKIP that entity.\n"
        "  - confidence: a float in [0,1] reflecting how clearly the entity "
        "is present + how clearly it instantiates that class.\n\n"
        "RULES:\n"
        "  - Only PROPER NOUN entities: organizations, people, places, "
        "products, named events, programs. Skip generic terms like "
        "\"the manufacturer\", \"the report\", \"the country\".\n"
        "  - 0 to 15 entities per chunk; quality over quantity.\n"
        "  - Years (e.g. \"2024\", \"Q1 2024\", \"January 2024\") are handled "
        "by a separate temporal pass -- DO NOT include them here.\n"
        "  - canonical_name should be the entity's most complete proper name "
        "as commonly written. If only an abbreviation is in the chunk, expand it "
        "if the expansion is unambiguous; otherwise use the abbreviation.\n"
        "  - Return ONLY JSON, no preamble, no markdown."
    )
    user = (
        "CANDIDATE CLASSES (pick class_iri from this list ONLY):\n"
        + candidates_block
        + "\n\nTEXT CHUNK:\n```\n"
        + chunk_text
        + "\n```\n\n"
        "Return JSON: {\"entities\": [{\"canonical_name\": ..., "
        "\"short_name\": ..., \"class_iri\": ..., \"confidence\": ...}]}"
    )
    return system, user


def artifact_chunk_extract_with_entities(
    chunk_text: str,
    entities: list[dict[str, str]],
) -> tuple[str, str]:
    """Phase 2 Milestone E (revised): entity-grounded Claim/Finding/
    Observation extraction.

    `entities` is the list of entities present in this chunk (from
    extract-entities). Each item: {canonical_name, short_name,
    class_label}.

    Adds an ENTITY NAMING REQUIREMENT to the base prompt: forces the
    LLM to use canonical entity names instead of generic terms.
    If `entities` is empty, falls back to the original generic prompt.
    """
    if not entities:
        return artifact_chunk_extract(chunk_text)

    entity_lines = []
    for e in entities:
        name = e.get("canonical_name") or e.get("short_name") or ""
        cls = e.get("class_label") or ""
        if name:
            entity_lines.append(f"  - {name} ({cls})" if cls else f"  - {name}")
    entities_block = "\n".join(entity_lines)

    system = (
        "You extract structured intelligence artifacts from text chunks. "
        "You return ONE JSON object with three keys: claims, findings, "
        "observations. Each is a list of `{text, confidence}` items.\n\n"
        "DEFINITIONS:\n"
        "  - Claim: a factual assertion the text MAKES (e.g. \"X "
        "owns 30% of Y\"). Specific and verifiable.\n"
        "  - Finding: an analytical conclusion or insight (e.g. \"the "
        "trend suggests Z is accelerating\"). Goes beyond raw facts.\n"
        "  - Observation: a raw factual statement directly visible in "
        "the text (e.g. \"price rose 5% in March\").\n\n"
        "ENTITY NAMING REQUIREMENT (READ CAREFULLY):\n"
        "  - You will be given a list of canonical ENTITIES present in this "
        "chunk. Whenever your Claim / Finding / Observation refers to one of "
        "these entities, you MUST use its EXACT canonical name as listed.\n"
        "  - NEVER substitute generic terms when the chunk's subject is in "
        "the entity list. Replace \"the company\" with the company's name, "
        "\"the country\" with the country's name, \"the report\" with the "
        "report's title, etc.\n"
        "  - If multiple entities are involved in one assertion, name all of "
        "them.\n"
        "  - If an assertion is about something NOT in the entity list "
        "(an abstract concept, a generic group), generic phrasing is fine.\n\n"
        "GUIDELINES:\n"
        "  - 0 to 8 of each type per chunk; only include items the "
        "text actually supports.\n"
        "  - `text` should be the artifact as a standalone sentence.\n"
        "  - `confidence` is a float in [0,1] reflecting how directly "
        "the chunk supports the artifact.\n"
        "  - Skip items that are too vague, uncertain, or generic to be "
        "useful.\n"
        "  - Return ONLY the JSON object -- no preamble, no markdown."
    )
    user = (
        "ENTITIES IN THIS CHUNK (use the canonical names below in your output):\n"
        + entities_block
        + "\n\nTEXT CHUNK:\n```\n"
        + chunk_text
        + "\n```\n\n"
        "Return the JSON now."
    )
    return system, user


def question_parse(question: str) -> tuple[str, str]:
    """Phase 2 Milestone F step 3: parse a user question into structured
    constraints the retrieval pipeline can use as graph seeds.

    Returns JSON:
      {
        "entities": ["BYD", "Vietnam"],   # proper-noun mentions
        "classes":  ["regulation", "manufacturer"],   # category words
        "time_terms": ["2024", "Q1 2024", "since 2020"],
        "intent": "comparison" | "enumeration" | "summary" | "factoid" | "research"
      }
    """
    system = (
        "You parse user questions into structured constraints. Return ONE JSON "
        "object with four keys: entities, classes, time_terms, intent.\n\n"
        "  - entities: proper-noun mentions (people, organizations, places, "
        "products). Use the form that appears in the question.\n"
        "  - classes: category words / common nouns naming a kind of thing "
        "(e.g. 'regulation', 'manufacturer', 'country').\n"
        "  - time_terms: any time / date expressions in the question.\n"
        "  - intent: one of 'comparison', 'enumeration', 'summary', "
        "'factoid', 'research'. Pick the closest fit.\n\n"
        "Return ONLY the JSON; no preamble, no markdown."
    )
    user = f"QUESTION: {question}\n\nReturn the JSON now."
    return system, user


def concept_expansion(
    question: str, matched_class_iris_labels: list[tuple[str, str]]
) -> tuple[str, str]:
    """Phase 2 Milestone F step 5: given the question + ontology classes
    that vector-matched, propose 5-15 additional related class IRIs that
    might be involved. Helps graph BFS find evidence the seed didn't.

    Returns JSON: {"related_classes": ["<iri>", ...]}
    """
    listing = "\n".join(
        f"  - {iri} ({label})" for iri, label in matched_class_iris_labels[:20]
    )
    system = (
        "You expand a set of ontology class candidates with 5 to 15 related "
        "classes the question might also involve. You will pick from the "
        "candidate list ONLY -- never invent IRIs. Return JSON.\n\n"
        "GUIDELINES:\n"
        "  - Pick classes that are conceptually related (siblings, "
        "supertypes, neighbors).\n"
        "  - Don't repeat the IRIs you were given; ADD related ones.\n"
        "  - If you can't find good additions, return an empty list.\n"
        "  - Return ONLY {\"related_classes\": [\"<iri>\", ...]}."
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"MATCHED CLASSES (already in seed set):\n{listing}\n\n"
        "Return JSON with 0-15 additional class IRIs from the matched list."
    )
    return system, user


def query_decompose(question: str) -> tuple[str, str]:
    """Phase 2 Milestone F step 9a: decompose a complex/comparative
    question into 1-5 atomic sub-questions for multi-probe vector
    rerank.

    Returns JSON: {"sub_questions": ["<q1>", "<q2>", ...]}

    Empty list -> caller falls back to using just the original query.
    """
    system = (
        "You break a user question into 1-5 atomic sub-questions. Each "
        "sub-question must be answerable independently with focused "
        "evidence. Use this when the question has multiple comparison "
        "sides, multiple subjects, or distinct angles. For simple "
        "factoid questions, return ONE element (the question itself).\n\n"
        "EXAMPLES:\n"
        "  Q: 'How do manufacturing prospects in Vietnam compare to the rest of Asia?'\n"
        "    A: ['Vietnam manufacturing prospects',\n"
        "        'Manufacturing prospects in Asia excluding Vietnam',\n"
        "        'Comparison axes across Asian manufacturing']\n"
        "  Q: 'What is BYD's annual production capacity?'\n"
        "    A: ['BYD annual production capacity']\n\n"
        "Return ONLY {\"sub_questions\": [\"...\", ...]}."
    )
    user = f"QUESTION: {question}\n\nReturn the JSON now."
    return system, user


def chunk_relevance_filter(
    question: str, chunks_with_ids: list[tuple[str, str]]
) -> tuple[str, str]:
    """Phase 2 Milestone F step 11 for deep_research / insights: filter
    a batch of retrieved chunks down to the relevant portions before
    stuffing into the expensive synthesis prompt. Map-reduce-style.

    `chunks_with_ids` is a list of (chunk_iri, chunk_text). Returns
    JSON: {"chunks": [{"iri": ..., "relevance": "yes"|"partial"|"no",
                       "extract": "..."}, ...]}
    For 'no' the extract is "".
    For 'partial' the extract is a 1-3 sentence relevant snippet.
    For 'yes' the extract is the chunk verbatim.
    """
    listing = "\n\n".join(
        f"---CHUNK {iri}---\n{text[:1200]}" for iri, text in chunks_with_ids
    )
    system = (
        "You filter retrieved chunks for relevance to a question. For each "
        "chunk:\n"
        "  - relevance='yes' if the chunk DIRECTLY addresses the question; "
        "the extract is the chunk verbatim.\n"
        "  - relevance='partial' if PART of the chunk is relevant; the "
        "extract is a 1-3 sentence excerpt of just the relevant part.\n"
        "  - relevance='no' if the chunk doesn't address the question; "
        "the extract is an empty string.\n\n"
        "Return ONLY JSON: {\"chunks\": [{\"iri\": ..., \"relevance\": ..., "
        "\"extract\": ...}]}"
    )
    user = f"QUESTION: {question}\n\n{listing}\n\nReturn the JSON now."
    return system, user


def _format_evidence_block(evidence_items: list[dict]) -> str:
    """Compact textual rendering of evidence items for stuffing into
    answer-synthesis prompts. Each item: {iri, kind, text}."""
    lines = []
    for it in evidence_items:
        kind = it.get("kind", "evidence")
        iri = it.get("iri", "")
        text = (it.get("text") or "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "..."
        lines.append(f"  [{kind} {iri}] {text}")
    return "\n".join(lines)


_FACTS_FIRST_RULE = (
    "FACTS-FIRST RULE (mandatory):\n"
    "  - When the evidence contains specific named entities, numbers, "
    "dates, percentages, or proper-noun examples, you MUST include "
    "them in the answer. Do not paraphrase 'frameworks' when "
    "'FAO Hand-in-Hand Initiative and Zero Hunger 2030' are in the "
    "evidence. Do not paraphrase 'a major producer' when '16.1 million "
    "metric tons' is in the evidence.\n"
    "  - STRUCTURE: lead with the specific facts (with their citations), "
    "THEN provide any synthesis / analysis / interpretation. If you have "
    "an opinion or conclusion, state it AFTER the facts it rests on, "
    "and flag it as your own judgement ('this suggests...', 'on balance...', "
    "etc.).\n"
    "  - Cite every fact by its IRI in brackets, e.g. [viao:Chunk_abc...] "
    "or [viao:Claim_...]. Multiple citations on one fact are fine.\n"
    "  - If a specific name or figure is NOT in the evidence, do not "
    "invent one; say what's known and what's missing."
)


def answer_simple_qa(
    question: str, evidence: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone F simple_qa mode: short factoid answer."""
    system = (
        "You answer questions briefly and accurately using ONLY the evidence "
        "below.\n\n"
        + _FACTS_FIRST_RULE
        + "\n\nLENGTH: 1-3 sentences. If the evidence doesn't answer the "
        "question, say so explicitly. Return ONLY the answer text."
    )
    user = (
        f"QUESTION: {question}\n\nEVIDENCE:\n"
        + _format_evidence_block(evidence)
        + "\n\nAnswer now."
    )
    return system, user


def answer_summarize(
    question: str, evidence: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone F summarize mode: 2-4 paragraph thematic summary."""
    system = (
        "You synthesize a thematic summary of what the corpus says about the "
        "question, using ONLY the evidence below.\n\n"
        + _FACTS_FIRST_RULE
        + "\n\nLENGTH: 2-4 paragraphs. Open with the most specific facts "
        "(names, numbers, dates) and group them by theme. Save any synthesis "
        "for the closing paragraph. If the evidence is thin, say what's "
        "missing. Return ONLY the summary -- no markdown headers."
    )
    user = (
        f"QUESTION: {question}\n\nEVIDENCE:\n"
        + _format_evidence_block(evidence)
        + "\n\nWrite the summary now."
    )
    return system, user


def answer_deep_research(
    question: str, evidence: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone F deep_research mode: long-form synthesis."""
    system = (
        "You produce a thorough research-style synthesis using ONLY the "
        "evidence below.\n\n"
        + _FACTS_FIRST_RULE
        + "\n\nSTRUCTURE:\n"
        "  1. Brief framing of the question (one paragraph).\n"
        "  2. EVIDENCE-FIRST BODY: enumerate the specific facts grouped "
        "by sub-topic. Every claim cited by IRI. Use named entities, "
        "numbers, and dates verbatim from the evidence -- do not summarize "
        "them away.\n"
        "  3. ANALYSIS (clearly separated from the facts): compare angles, "
        "surface tensions, note where the evidence supports multiple "
        "interpretations.\n"
        "  4. SHORT CONCLUSION (one paragraph). Your own judgement, flagged "
        "as such.\n\n"
        "400-800 words. Return ONLY the answer text -- no markdown headers."
    )
    user = (
        f"QUESTION: {question}\n\nEVIDENCE:\n"
        + _format_evidence_block(evidence)
        + "\n\nWrite the synthesis now."
    )
    return system, user


def answer_insights(
    question: str, evidence: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone F insights mode: pattern surfacing."""
    system = (
        "You surface non-obvious INSIGHTS across the evidence below.\n\n"
        + _FACTS_FIRST_RULE
        + "\n\nSTRUCTURE: Group findings into 2-5 themes. For each theme:\n"
        "  - First: 2-3 specific facts grounding the theme (names, "
        "numbers, dates verbatim from evidence, each cited).\n"
        "  - Then: the insight in 1-2 sentences, clearly framed as your "
        "synthesis ('this pattern suggests...', 'taken together...').\n"
        "Insights must go beyond what any single claim says -- look for "
        "cross-cutting patterns, contradictions, or emerging trends. "
        "Return plain text, no markdown."
    )
    user = (
        f"QUESTION: {question}\n\nEVIDENCE:\n"
        + _format_evidence_block(evidence)
        + "\n\nProduce the insights now."
    )
    return system, user


def answer_knowledge_gaps(
    question: str,
    sub_questions: list[str],
    found_for: list[str],
) -> tuple[str, str]:
    """Phase 2 Milestone F knowledge_gaps mode: report what's missing."""
    found_block = "\n".join(f"  - {q}" for q in found_for) or "  (none)"
    missing = [q for q in sub_questions if q not in found_for]
    missing_block = "\n".join(f"  - {q}" for q in missing) or "  (none)"
    system = (
        "You report knowledge gaps about a question. You're given (a) the "
        "sub-questions implied by the user's question, (b) which sub-questions "
        "the corpus DOES have evidence for, and (c) which it does NOT. Write "
        "2-3 short paragraphs describing the gaps and what kinds of "
        "additional documents would close them. Return plain text only."
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"SUB-QUESTIONS COVERED BY CORPUS:\n{found_block}\n\n"
        f"SUB-QUESTIONS NOT COVERED:\n{missing_block}\n\n"
        "Write the gap report now."
    )
    return system, user


def answer_exhaustive_group_caption(
    question: str, document_title: str, snippets: list[str]
) -> tuple[str, str]:
    """Phase 2 Milestone F exhaustive_search: 1-sentence caption per
    matched document. Run in parallel; no global synthesis.
    """
    snip = "\n".join(f"  - {s[:300]}" for s in snippets[:5])
    system = (
        "You write a one-sentence caption describing what a document says "
        "about a user query. The caption is for a list of matches, so be "
        "specific to THIS document. <= 25 words. No preamble. Plain text."
    )
    user = (
        f"USER QUERY: {question}\n\n"
        f"DOCUMENT TITLE: {document_title}\n\n"
        f"MATCHING SNIPPETS:\n{snip}\n\n"
        "Write the one-sentence caption now."
    )
    return system, user


def follow_up_resolution(
    new_question: str, prior_turns: list[tuple[str, str, str]]
) -> tuple[str, str]:
    """Phase 2 Milestone G: rewrite a conversational follow-up into a
    self-contained question.

    `prior_turns` is a list of (user_question, resolved_question, answer)
    triples in chronological order (oldest first). The ANSWER from each
    prior turn is what makes follow-up resolution work for references
    like 'what frameworks' or 'which of those' -- the rewriter pulls
    named entities from prior answers into the rewritten question.
    """
    if not prior_turns:
        # caller should skip the LLM call in this case
        return ("", new_question)

    def _trim(s: str, n: int) -> str:
        s = (s or "").strip().replace("\n", " ")
        return (s[:n] + "...") if len(s) > n else s

    hist_blocks = []
    for i, (ask, res, ans) in enumerate(prior_turns):
        hist_blocks.append(
            f"  turn {i+1}:\n"
            f"    asked:      {_trim(ask, 250)}\n"
            f"    resolved:   {_trim(res, 250)}\n"
            f"    answer:     {_trim(ans, 600)}"
        )
    hist = "\n".join(hist_blocks)

    system = (
        "You rewrite a follow-up user question into a SELF-CONTAINED, "
        "standalone question that fully captures what the user actually "
        "wants, using the prior CONVERSATION HISTORY.\n\n"
        "RULES:\n"
        "  - Replace pronouns ('it', 'they', 'those', 'the same') with "
        "the specific referents from prior questions OR answers.\n"
        "  - When the user asks for elaboration on something the prior "
        "answer mentioned (e.g. 'what frameworks?', 'which of those?', "
        "'tell me more about that'), the rewritten question MUST name "
        "the specific entities, frameworks, programs, or concepts the "
        "prior answer cited. Pull names directly from the prior answer.\n"
        "  - If the new question already stands alone, return it unchanged.\n"
        "  - Do NOT answer the question; only rewrite it.\n"
        "  - Return ONLY the rewritten question -- no preamble, no quotes.\n\n"
        "EXAMPLES:\n"
        "  Prior answer: 'Countries are implementing frameworks like the "
        "FAO Hand-in-Hand Initiative, Zero Hunger 2030, and the Hunger-Free "
        "World Index.'\n"
        "  Follow-up: 'What frameworks?'\n"
        "  Rewrite:   'What are the FAO Hand-in-Hand Initiative, Zero "
        "Hunger 2030, and the Hunger-Free World Index, and what do they "
        "do to reduce hunger?'\n\n"
        "  Prior answer: 'BYD produced 3 million EVs in 2024.'\n"
        "  Follow-up: 'And how does that compare to Honda?'\n"
        "  Rewrite:   'How does Honda's 2024 EV production compare to "
        "BYD's 3 million EVs?'"
    )
    user = (
        f"CONVERSATION HISTORY:\n{hist}\n\n"
        f"NEW USER QUESTION: {new_question}\n\n"
        "Rewrite the question to be self-contained."
    )
    return system, user


def answer_conversation_turn(
    resolved_query: str,
    current_evidence: list[dict],
    prior_turns: list[tuple[str, str]],
    base_mode: str = "simple_qa",
) -> tuple[str, str]:
    """Phase 2 Milestone G: synthesize a conversation-turn answer with
    BOTH this turn's retrieved evidence and prior conversation context
    in scope.

    `prior_turns` items: (user_question, answer). Most recent last.
    `base_mode` selects the depth/length style:
       simple_qa | summarize | knowledge_gaps -> short
       deep_research | insights              -> long

    Falls back to plain answer_simple_qa-style if prior_turns is empty.
    """
    if not prior_turns:
        if base_mode == "deep_research":
            return answer_deep_research(resolved_query, current_evidence)
        if base_mode == "insights":
            return answer_insights(resolved_query, current_evidence)
        if base_mode == "summarize":
            return answer_summarize(resolved_query, current_evidence)
        if base_mode == "knowledge_gaps":
            # caller computes the sub_questions / found_for lists; fallback simple
            return answer_simple_qa(resolved_query, current_evidence)
        return answer_simple_qa(resolved_query, current_evidence)

    def _trim(s: str, n: int) -> str:
        s = (s or "").strip().replace("\n", " ")
        return (s[:n] + "...") if len(s) > n else s

    hist = "\n".join(
        f"  Q{i+1}: {_trim(q, 250)}\n  A{i+1}: {_trim(a, 800)}"
        for i, (q, a) in enumerate(prior_turns)
    )
    is_long = base_mode in ("deep_research", "insights")
    length = "400-800 words" if is_long else "1-3 paragraphs"
    style = (
        "thorough research-style synthesis"
        if is_long else "concise, grounded answer"
    )

    system = (
        f"You produce a {style} for a multi-turn conversation. "
        "You will see the CONVERSATION HISTORY (prior questions and your "
        "earlier answers) plus the CURRENT QUESTION and the EVIDENCE "
        "retrieved for it.\n\n"
        + _FACTS_FIRST_RULE
        + "\n\nCONVERSATION RULES:\n"
        "  - Treat the prior answers as already-established context. "
        "Do NOT restate them; build on them.\n"
        "  - If the user is asking for elaboration on specific things "
        "the prior answer mentioned by name (frameworks, companies, "
        "policies, numbers), name those SAME things in your new answer "
        "and pull more detail from the CURRENT EVIDENCE about them.\n"
        "  - Every NEW claim must come from the CURRENT EVIDENCE, cited "
        "by its IRI in brackets like [viao:Chunk_abc...].\n"
        "  - If you reference something from a prior answer, you may say "
        "'as discussed' or 'building on the prior answer' without "
        "re-citing it.\n"
        "  - If the CURRENT EVIDENCE doesn't address the question, say so "
        "explicitly -- do NOT fabricate.\n\n"
        "STRUCTURE: lead with the SPECIFIC FACTS the user is asking about "
        "(names, numbers, dates, with citations), THEN any synthesis or "
        "interpretation last.\n"
        f"LENGTH: {length}. Return ONLY the answer text -- no markdown headers."
    )
    user = (
        f"CONVERSATION HISTORY:\n{hist}\n\n"
        f"CURRENT QUESTION: {resolved_query}\n\n"
        f"CURRENT EVIDENCE:\n"
        + _format_evidence_block(current_evidence)
        + "\n\nAnswer now."
    )
    return system, user


def _format_judge_evidence(evidence: list[dict], cap: int = 15) -> str:
    """Compact evidence block for judge prompts."""
    lines = []
    for ev in evidence[:cap]:
        kind = ev.get("kind", "evidence")
        iri = ev.get("iri", "")
        text = (ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"  [{kind} {iri}] {text}")
    return "\n".join(lines)


def judge_comprehensiveness(
    question: str, evidence: list[dict], answer: str
) -> tuple[str, str]:
    """Eval metric 1: did the answer actually address what was asked?

    Returns JSON {score: 0.0-1.0, justification: str}.
    """
    system = (
        "You judge whether an answer comprehensively addresses a user "
        "question. Score 0.0-1.0:\n"
        "  1.0 = answer fully addresses every part of the question.\n"
        "  0.7-0.9 = mostly answers; minor gaps acceptable.\n"
        "  0.4-0.6 = partial; misses important parts of the question.\n"
        "  0.0-0.3 = answer ignores or sidesteps the question.\n\n"
        "Do NOT judge factual accuracy here -- only whether the question's "
        "scope is covered. Return ONLY JSON: "
        "{\"score\": <float>, \"justification\": \"<one sentence>\"}"
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return the JSON now."
    )
    return system, user


def judge_no_hallucination(
    question: str, evidence: list[dict], answer: str
) -> tuple[str, str]:
    """Eval metric 2: every claim in the answer must be grounded in
    the retrieved evidence. Unsupported claims dock the score.

    Returns JSON {score: 0.0-1.0, justification: str}.
    """
    system = (
        "You check whether every claim in an answer is grounded in the "
        "provided evidence. Score 0.0-1.0:\n"
        "  1.0 = every claim is directly supported by at least one piece "
        "of evidence.\n"
        "  0.7-0.9 = most claims supported; ONE minor claim is unsupported "
        "but plausible.\n"
        "  0.4-0.6 = SEVERAL claims unsupported or unstated in evidence.\n"
        "  0.0-0.3 = answer fabricates a substantive claim. THE MOST "
        "DANGEROUS FAILURE.\n\n"
        "An answer that correctly says 'evidence does not cover this' "
        "scores 1.0.\n\n"
        "Return ONLY JSON: {\"score\": <float>, \"justification\": "
        "\"<one sentence, name the unsupported claim if any>\"}"
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"EVIDENCE:\n{_format_judge_evidence(evidence)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return the JSON now."
    )
    return system, user


def judge_gap_detection(
    question: str,
    evidence: list[dict],
    answer: str,
    expected_gap: bool,
) -> tuple[str, str]:
    """Eval metric 3: does the LLM correctly say when the corpus has
    no answer? Behavior flips based on `expected_gap`.

    Returns JSON {score: 0.0-1.0, justification: str}.
    """
    if expected_gap:
        rule = (
            "The corpus has NO information on this question. The answer "
            "SHOULD explicitly acknowledge this (e.g. 'the corpus does "
            "not cover...', 'no evidence available', etc.). Score:\n"
            "  1.0 = answer explicitly says corpus lacks this info.\n"
            "  0.7-0.9 = answer hedges adequately but doesn't say so "
            "outright.\n"
            "  0.4-0.6 = answer answers anyway with shaky grounding.\n"
            "  0.0-0.3 = answer confidently fabricates."
        )
    else:
        rule = (
            "The corpus DOES have evidence. The answer should USE it, "
            "not refuse. Score:\n"
            "  1.0 = answer engages with the evidence.\n"
            "  0.7-0.9 = answer engages with mild hedging.\n"
            "  0.4-0.6 = answer hedges excessively despite ample evidence.\n"
            "  0.0-0.3 = answer refuses despite clear coverage."
        )
    system = (
        "You judge whether an answer's gap-handling behavior is correct.\n\n"
        + rule
        + "\n\nReturn ONLY JSON: {\"score\": <float>, \"justification\": "
        "\"<one sentence>\"}"
    )
    user = (
        f"QUESTION: {question}\n"
        f"EXPECTED_GAP: {expected_gap}\n\n"
        f"EVIDENCE (first 5):\n{_format_judge_evidence(evidence, cap=5)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return the JSON now."
    )
    return system, user


def judge_consistency(
    question: str, answers: list[str]
) -> tuple[str, str]:
    """Eval metric 4: are N answers to the same question semantically
    equivalent? ONE call across all N answers.

    Returns JSON {score: 0.0-1.0, justification: str}.
    """
    if len(answers) < 2:
        # Caller should skip the call in this case.
        return ("", "")
    body = "\n\n".join(
        f"--- Answer {i+1} ---\n{a}" for i, a in enumerate(answers)
    )
    system = (
        "You judge whether multiple answers to the same question are "
        "semantically equivalent. Same facts, same framing, same level of "
        "specificity. Score 0.0-1.0:\n"
        "  1.0 = answers are equivalent (same key facts; phrasing may differ).\n"
        "  0.7-0.9 = answers agree on the main facts but differ in detail/"
        "emphasis.\n"
        "  0.4-0.6 = answers cover overlapping but distinct facts.\n"
        "  0.0-0.3 = answers contradict each other or share no content.\n\n"
        "Return ONLY JSON: {\"score\": <float>, \"justification\": "
        "\"<one sentence, note key differences>\"}"
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"{body}\n\n"
        "Return the JSON now."
    )
    return system, user


def insight_gen(
    class_label: str, claims_findings: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone E (Insight subtype): synthesize 1-3 Insights
    across the Claims+Findings attached to a single ontology class.

    `claims_findings` items: {type, text, confidence}
    Returns JSON: {"insights": [{"text", "confidence"}]}
    """
    listing = "\n".join(
        f"  [{c['type']}, c={c.get('confidence','?')}] {c['text']}"
        for c in claims_findings[:25]
    )
    system = (
        "You synthesize INSIGHTS across multiple Claims and Findings about "
        "a topic. Look for cross-cutting patterns, contradictions, emerging "
        "trends. Each insight must go BEYOND restating any single claim and "
        "ground itself in 2 or more of them. 1-3 insights, 1-2 sentences "
        "each. Return ONLY JSON: {\"insights\": [{\"text\": ..., "
        "\"confidence\": ...}]}"
    )
    user = (
        f"TOPIC (ontology class): {class_label}\n\n"
        f"CLAIMS AND FINDINGS:\n{listing}\n\n"
        "Return the insights JSON now."
    )
    return system, user


def recommendation_gen(
    theme_label: str, insights: list[dict]
) -> tuple[str, str]:
    """Phase 2 Milestone E (Recommendation subtype): propose actionable
    recommendations grounded in a theme of Insights.

    `insights` items: {text, confidence}
    Returns JSON: {"recommendations": [{"text", "confidence"}]}
    """
    listing = "\n".join(
        f"  [Insight] {i['text']}" for i in insights[:15]
    )
    system = (
        "You propose ACTIONABLE recommendations given a coherent theme of "
        "insights. Each recommendation must be specific, name the entities "
        "involved, and be grounded in the insights provided. 1-3 "
        "recommendations, 1-3 sentences each. Return ONLY JSON: "
        "{\"recommendations\": [{\"text\": ..., \"confidence\": ...}]}"
    )
    user = (
        f"THEME: {theme_label}\n\n"
        f"INSIGHTS:\n{listing}\n\n"
        "Return the recommendations JSON now."
    )
    return system, user


def artifact_document_summary(chunks_text: str) -> tuple[str, str]:
    """Phase 2 Milestone E: per-document Summary artifact.

    Given the concatenated chunks of a document (already summarized
    upstream if oversize), produce a single Summary artifact.
    Corpus-agnostic prose, no domain-specific framing.
    """
    system = (
        "You write neutral, third-person document summaries. Capture "
        "the document's main points in 150-250 words. No opinions, no "
        "editorial framing, no bullet lists — flowing prose only. "
        "Return ONLY the summary text."
    )
    user = (
        "DOCUMENT CONTENT:\n```\n"
        + chunks_text
        + "\n```\n\n"
        "Write the summary now."
    )
    return system, user


# Public registry so callers can look up a prompt builder by task name.
PROMPTS = {
    "chunk_classification": chunk_classification,
    "class_proposal": class_identification_and_expansion,
    "match_dedup": match_dedup,
    "concept_grouping": concept_grouping,
    "compact_description": compact_description,
    "document_summarize": document_summarize,
    "classification_audit": classification_audit,
    "artifact_chunk_extract": artifact_chunk_extract,
    "artifact_chunk_extract_with_entities": artifact_chunk_extract_with_entities,
    "artifact_document_summary": artifact_document_summary,
    "entity_extract": entity_extract,
    # Milestone F
    "question_parse": question_parse,
    "concept_expansion": concept_expansion,
    "query_decompose": query_decompose,
    "chunk_relevance_filter": chunk_relevance_filter,
    "answer_simple_qa": answer_simple_qa,
    "answer_summarize": answer_summarize,
    "answer_deep_research": answer_deep_research,
    "answer_insights": answer_insights,
    "answer_knowledge_gaps": answer_knowledge_gaps,
    "answer_exhaustive_group_caption": answer_exhaustive_group_caption,
    # Milestone G
    "follow_up_resolution": follow_up_resolution,
    "answer_conversation_turn": answer_conversation_turn,
    # Insight + Recommendation generation
    "insight_gen": insight_gen,
    "recommendation_gen": recommendation_gen,
    # Eval framework (LLM-as-judge)
    "judge_comprehensiveness": judge_comprehensiveness,
    "judge_no_hallucination": judge_no_hallucination,
    "judge_gap_detection": judge_gap_detection,
    "judge_consistency": judge_consistency,
}

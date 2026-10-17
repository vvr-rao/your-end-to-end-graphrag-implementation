"""DeepSeek-V4.1-Flash-specific prompts.

Mixed mode (user decision 2026-10-14): Qwen3.8-27B does knowledge creation,
DeepSeek-V4.1-Flash does relationship extraction -- 27B proposed few and
wildly varying relationships (65 / 5 / 12 on identical chunks) whatever the
prompt, while DeepSeek-V4.1-Flash found 135 in its own 3-doc screen.

Kept SEPARATE from prompts.py and prompts_qwen.py: edits here change only
what DeepSeek is sent. Selected with `prompt_set: deepseek` on the task.
Started as a clone of the Qwen relationship prompt: the chunks it reads are
still Qwen's markdown summaries, so the markdown rules carry over.
"""
from __future__ import annotations

from backend.app.services.prompts import GENERIC_PREDICATE_IRIS

# Relationship extraction (DeepSeek-only; cloned from the Qwen version 2026-10-14). Qwen's chunks are its own MARKDOWN
# summaries (18/18 chunks vs 0/8 for OpenAI's plain-text ones, 2026-10-14):
# assertions sit in bullet labels ("**Ozempic:** Approved for ...") and table
# rows, which a model reads as formatting, and a quote that drops the bold
# label no longer names both entities.
DEEPSEEK_RELATIONSHIP_RULES = (
    "\n\nTHE PASSAGE IS A MARKDOWN SUMMARY -- read its structure as text:\n"
    "  - A bullet is a sentence. '**Tesla:** Supplies batteries to Panasonic "
    "plants' or '**Ozempic:** Approved for type 2 diabetes' each state a "
    "relationship between the bold label and the rest of the bullet. Quote "
    "the WHOLE bullet INCLUDING its bold label, because the label is where "
    "the subject is named.\n"
    "  - A table row is a sentence. In a row '| Acme Corp | Berlin | Battery "
    "cells |' under headers '| Company | Headquarters | Main product |', Acme "
    "Corp is headquartered in Berlin and makes battery cells. Quote the "
    "row.\n"
    "  - A heading names the subject of the bullets beneath it; extend the "
    "quote back to the heading when a bullet does not name the subject.\n"
    "  - Markdown symbols (**, *, |, #) may be copied or left out -- the "
    "checker ignores them -- but every WORD must be copied exactly, in "
    "order, with no paraphrase.\n"
    "  - Expect MANY relationships in these summaries: they are dense lists "
    "of who owns, makes, supplies, regulates, causes, treats or competes with "
    "what. Work through every bullet and row.\n"
    "  - Write the JSON on ONE line with no indentation."
)


def deepseek_relationship_extract(
    chunk_text: str,
    entities: list[dict[str, str]],
    candidate_predicates: list[dict[str, str]],
    *,
    max_relationships: int = 10,
    already_found: list[str] | None = None,
) -> tuple[str, str]:
    """DeepSeek-V4.1-Flash variant (see DEEPSEEK_RELATIONSHIP_RULES). Phase 2 Milestone C, second pass: relationships between entities ALREADY
    extracted from this chunk.

    Why a second call rather than one combined request: the predicate menu can
    only be narrowed to the entities' ACTUAL classes once those entities exist.
    In the combined version predicates were derived from the chunk's top-50
    candidate CLASSES, so the model was shown a median of 12 predicates while
    holding 4-5 entities, most of which no offered predicate fitted -- measured
    669 of 1,140 proposals rejected on domain/range alone.

    `entities` are {canonical_name, class_label} for this chunk only.
    `candidate_predicates` are {iri, label, domain_label, range_label} whose
    domain AND range both match classes those entities actually have.

    Returns JSON:
      {relationships: [{subject, predicate_iri, object, evidence, confidence}]}

    `evidence` is a VERBATIM span from the chunk. The caller checks it really
    occurs in the text and drops the relationship otherwise -- narrowing the
    menu weakens the type check (offer and check now share a basis), so this
    replaces the protection that removes.
    """
    ent_lines = "\n".join(
        f"  - {e['canonical_name']}  [{e.get('class_label') or '?'}]"
        for e in entities
    )

    def _fmt(p: dict[str, str]) -> str:
        return (f"  - {p['iri']} \u2500 {p.get('label') or '?'}"
                f"  [{p.get('domain_label') or '?'} -> "
                f"{p.get('range_label') or '?'}]")

    # Split the menu. Catch-all predicates fit almost any pair of entities, so
    # offered flat they crowd out the specific ones: measured on the news
    # corpus, org:hasMember + org:linkedTo + foaf:topic took 37 of 65 edges
    # (57%) while every corpus-specific predicate the ontology minted for that
    # run -- competesWith, filesLawsuitAgainst, integrates, backs -- shared 12.
    # Listing them separately, under a rule that they are a fallback, makes the
    # model rule out a precise predicate before reaching for a vague one.
    specific = [p for p in candidate_predicates
                if p["iri"] not in GENERIC_PREDICATE_IRIS]
    generic = [p for p in candidate_predicates
               if p["iri"] in GENERIC_PREDICATE_IRIS]
    pred_lines = "\n".join(_fmt(p) for p in specific) or "  (none)"
    if generic:
        pred_lines += ("\n\nLAST-RESORT PREDICATES (vague -- use ONLY if no "
                       "predicate above fits):\n"
                       + "\n".join(_fmt(p) for p in generic))
    # already_found: the SECOND pass. The first pass's accepted relationships
    # are shown and only the ones it missed are asked for -- a single pass
    # finds a different ~70% of a chunk's relationships on each run.
    _gap = (
        "A FIRST PASS has already extracted the relationships listed under "
        "ALREADY FOUND. Your job now is to find what it MISSED: relationships "
        "the passage states that are NOT already listed (the same pair with "
        "subject and object swapped counts as listed). Return ONLY new "
        "relationships -- an empty list is the correct answer when nothing was "
        "missed. Every rule below still applies.\n\n"
    ) if already_found is not None else ""
    system = (
        _gap +
        "You identify RELATIONSHIPS that a passage explicitly asserts between "
        "entities that have ALREADY been extracted from it. Return ONE JSON "
        "object and nothing else -- no prose, no markdown fences.\n\n"
        "For each relationship:\n"
        "  - subject / object: copy a canonical_name VERBATIM from the ENTITIES "
        "list. Never introduce a name that is not on that list.\n"
        "  - predicate_iri: copy ONE IRI VERBATIM from the PREDICATES list. Do "
        "not invent, abbreviate or reformat it.\n"
        "  - evidence: the VERBATIM sentence or clause from the passage that "
        "states this relationship. It must appear in the passage character for "
        "character. Do not paraphrase. It MUST NAME BOTH the subject and the "
        "object -- a sentence mentioning only one of them does not state a "
        "relationship between them.\n"
        "  - WHEN THE STATING SENTENCE USES A PRONOUN OR ROLE for one of them "
        "(\"he\", \"the company\", \"her father\", \"the Gangs of New York "
        "director\"), extend the quote BACKWARDS to begin at the words that "
        "NAME that entity, and copy everything in between unchanged. Example: "
        "for \"Alameda Research is a hedge fund. Bankman-Fried founded it in "
        "2017.\" quote exactly that, not just \"Bankman-Fried founded it in "
        "2017.\" The quote must stay ONE continuous span of at most 3 "
        "sentences -- never skip text, never join separate passages. If the "
        "name is further back than that, do not emit the relationship.\n"
        "  - relation: ONLY when predicate_iri is "
        "https://veerla-ramrao.ai/ontology/graphrag#relatedTo -- a short verb "
        "phrase (2-6 words) taken from the quote, read subject -> object, "
        "saying HOW they relate: \"was born in\", \"alleges fraud "
        "against\", \"signed for\". Omit it for every other predicate.\n"
        "  - confidence: float in [0,1].\n\n"
        "RULES THAT MATTER:\n"
        "  - graphrag#relatedTo is the LAST resort. Use it ONLY when the "
        "passage explicitly states a relationship between the two AND no "
        "other listed predicate expresses it. It is never for two entities "
        "that merely appear together, sit in the same list, or share a "
        "topic -- the quote must still state how they relate.\n"
        "  - Assert ONLY what the passage states. Do NOT use world knowledge. "
        "Two entities appearing near each other is NOT a relationship.\n"
        "  - If you cannot quote a span that states it, do not emit it.\n"
        "  - READ YOUR OWN QUOTE BEFORE EMITTING. It must assert THIS claim in "
        "THIS direction. 'Lawsuits filed against A by B' means B sued A, NOT "
        "that A sued B. If the quote states the reverse, swap subject and "
        "object or drop the relationship.\n"
        "  - Prefer the MOST SPECIFIC predicate that fits. Reach for a "
        "last-resort predicate only when no specific one applies.\n"
        "  - Each predicate shows [Domain -> Range]: the subject must be of the "
        "domain type and the object of the range type. Respect direction -- "
        "'A causes B' and 'B causes A' are different claims.\n"
        "  - RETURNING AN EMPTY LIST IS THE CORRECT ANSWER when the passage "
        "asserts none of these relationships. Do not force a match merely "
        "because a predicate is offered.\n"
        f"  - At most {max_relationships} relationships.\n\n"
        "BEFORE YOU FINISH -- completeness check:\n"
        "  Every rule above tells you when NOT to emit a relationship, and "
        "they all still hold. But leaving out a relationship the passage "
        "plainly states is just as much an error as inventing one, and it is "
        "the more common failure: a passage naming eight entities usually "
        "asserts more than two or three links between them.\n"
        "  So before you return, walk the ENTITIES list and ask of each one: "
        "does the passage say anything about how this entity stands to "
        "another entity on the list -- who employs or leads it, who makes or "
        "owns it, what it treats or causes or competes with, where it sits? "
        "If a quotable sentence names both and states the link, and a "
        "predicate fits, emit it. You are not being asked to guess; you are "
        "being asked not to stop early.\n"
        "  A person named alongside their employer, a product named alongside "
        "its maker, and a drug named alongside what it treats are the three "
        "most frequently missed."
        + DEEPSEEK_RELATIONSHIP_RULES
    )
    user = (
        "ENTITIES (subject/object must come from here):\n" + ent_lines
        + "\n\nPREDICATES (predicate_iri must come from here):\n" + pred_lines
        + ("\n\nALREADY FOUND (do NOT repeat these):\n"
           + ("\n".join(f"  - {f}" for f in already_found) or "  (none)")
           if already_found is not None else "")
        + "\n\nPASSAGE:\n```\n" + chunk_text + "\n```\n\n"
        'Return JSON: {"relationships": [{"subject": ..., "predicate_iri": ..., '
        '"object": ..., "evidence": ..., "confidence": ..., '
        '"relation": ... (graphrag#relatedTo only)}]}'
    )
    return system, user


def deepseek_class_synonym_judge(pairs: list[dict[str, str]]) -> tuple[str, str]:
    """Confirm or reject candidate duplicate CLASS pairs (embedding-proposed).

    Similarity alone cannot tell synonyms from siblings or opposites: on the
    2026-10-14 preset run BeforeBreakfast~AfterBreakfast scored 0.930 and
    CleftLip~CleftPalate 0.890, above Hypertension~HighBloodPressure (0.908)
    and Hypoglycemia~LowBloodSugar (0.873). This call makes that judgement.

    `pairs`: [{"id", "a_label", "a_desc", "b_label", "b_desc"}].
    Returns JSON {"decisions": [{"id", "same", "keep", "reason"}]}.
    """
    system = (
        "You review pairs of ontology CLASS names proposed as possible "
        "duplicates. For each pair decide whether A and B name the SAME "
        "concept, so that one class can replace the other everywhere without "
        "losing meaning. Return ONE JSON object and nothing else.\n\n"
        "SAME (merge): synonyms, abbreviations, spelling or plural variants, "
        "lay vs technical terms for one thing. E.g. Hypertension / "
        "HighBloodPressure, BMI / BodyMassIndex, Child / Children, "
        "Fatigue / Tiredness, CloudStorage / CloudStorageService, "
        "CentralBank / NationalCentralBank when used interchangeably.\n"
        "DIFFERENT (keep both): opposites (BeforeBreakfast / AfterBreakfast, "
        "ShortTerm / LongTerm, Import / Export), siblings (CleftLip / "
        "CleftPalate, Anorexia / Bulimia, Equity / Debt), a broader vs a "
        "narrower class (Drug / Antidepressant, Disease / HeartDisease), a "
        "condition vs its consequence or a part vs a whole, and different "
        "groups (ObesePerson / OverweightPerson).\n"
        "When unsure, answer DIFFERENT: a wrong merge corrupts the ontology; "
        "a missed merge only leaves a duplicate.\n\n"
        "For each pair return: id (as given), same (true/false), keep ('A' "
        "or 'B' -- the clearer, more standard name; only when same), and "
        "reason (at most 12 words). Write the JSON on ONE line."
    )
    lines = "\n".join(
        f'{p["id"]}. A="{p["a_label"]}" ({p.get("a_desc", "")[:160]}) | '
        f'B="{p["b_label"]}" ({p.get("b_desc", "")[:160]})'
        for p in pairs
    )
    user = (
        f"PAIRS:\n{lines}\n\n"
        'Return JSON: {"decisions": [{"id": ..., "same": ..., "keep": ..., "reason": ...}]}'
    )
    return system, user


# --------------------------------------------------------------------------- #
# Relationship rescue (DeepSeek-only; cloned from the shared relationship_repair
# 2026-10-17 so the deepinfra mode needs no OpenAI model). Same contract: the
# claims already passed the quote checks; only the predicate is re-chosen.
# --------------------------------------------------------------------------- #

def deepseek_relationship_repair(
    chunk_text: str,
    rejects: list[dict[str, str]],
    candidate_predicates: list[dict[str, str]],
) -> tuple[str, str]:
    """Re-home claims the strict domain/range check rejected, from a WIDER
    predicate menu, or drop them. Returns JSON {"relationships": [...]}."""
    reject_lines = "\n".join(
        f'  [{i}] {r["subject"]} --{r["predicate_label"]}--> {r["object"]}\n'
        f'      quote: "{r["evidence"]}"'
        for i, r in enumerate(rejects)
    )
    pred_lines = "\n".join(
        f"  - {p['iri']} ─ {p.get('label') or '?'}"
        f"  [{p.get('domain_label') or '?'} -> {p.get('range_label') or '?'}]"
        for p in candidate_predicates
    )
    system = (
        "Each claim below was extracted from the passage with a quote that "
        "really occurs in it and really names both entities -- but the "
        "predicate chosen was a poor fit, because the model that produced it "
        "had only a handful of predicates to choose from. You are given a "
        "WIDER list. Re-express each claim with the predicate that actually "
        "fits, or drop it.\n\n"
        "Return ONE JSON object and nothing else -- no prose, no fences.\n\n"
        "RULES THAT MATTER:\n"
        "  - subject / object: copy VERBATIM from the claim. You may SWAP them "
        "if the quote states the relationship the other way round; do not "
        "rename them.\n"
        "  - predicate_iri: copy ONE IRI VERBATIM from the PREDICATES list.\n"
        "  - evidence: keep the SAME quote. Do not substitute another.\n"
        "  - Choose the predicate the QUOTE supports, reading it literally. "
        "'X acquired Y' is an acquisition, not a sub-organization. 'Y received "
        "funding from X' means Y is funded by X.\n"
        "  - The passage may be a markdown summary: a bullet or table row is a "
        "sentence, and its bold label is usually the subject.\n"
        "  - DROPPING IS CORRECT when no predicate in the list expresses what "
        "the quote says. Emit fewer relationships than you were given rather "
        "than forcing a fit -- a wrong edge is worse than a missing one.\n"
        "  - Do not invent claims. Every output must correspond to an input "
        "claim, with at most its predicate changed and its ends swapped.\n"
        "  - Write the JSON on ONE line with no indentation."
    )
    user = (
        "PREDICATES (predicate_iri must come from here):\n" + pred_lines
        + "\n\nPASSAGE:\n```\n" + chunk_text + "\n```\n\n"
        + "CLAIMS TO RE-EXPRESS OR DROP:\n" + reject_lines
        + '\n\nReturn JSON: {"relationships": [{"subject": ..., '
          '"predicate_iri": ..., "object": ..., "evidence": ...}]}'
    )
    return system, user


# --------------------------------------------------------------------------- #
# Answer-quality judges for evaluate-queries (DeepSeek-only; cloned from the
# shared judge_* prompts 2026-10-17 so the deepinfra mode is OpenAI-free).
# Each returns JSON {"score": 0.0-1.0, "justification": str}.
# --------------------------------------------------------------------------- #

def _judge_evidence(evidence: list[dict], cap: int = 25) -> str:
    lines = []
    for ev in evidence[:cap]:
        kind = ev.get("kind", "?")
        iri = ev.get("iri", "?")
        text = (ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"  [{kind} {iri}] {text}")
    return "\n".join(lines)


_JUDGE_TAIL = (
    "Return ONLY one JSON object on one line: "
    "{\"score\": <float>, \"justification\": \"<one sentence%s>\"}"
)


def deepseek_judge_comprehensiveness(question: str, evidence: list[dict], answer: str) -> tuple[str, str]:
    system = (
        "You judge whether an answer comprehensively addresses a user "
        "question. Score 0.0-1.0:\n"
        "  1.0 = answer fully addresses every part of the question.\n"
        "  0.7-0.9 = mostly answers; minor gaps acceptable.\n"
        "  0.4-0.6 = partial; misses important parts of the question.\n"
        "  0.0-0.3 = answer ignores or sidesteps the question.\n\n"
        "Do NOT judge factual accuracy here -- only whether the question's "
        "scope is covered. " + _JUDGE_TAIL % ""
    )
    user = f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nReturn the JSON now."
    return system, user


def deepseek_judge_no_hallucination(question: str, evidence: list[dict], answer: str) -> tuple[str, str]:
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
        "scores 1.0.\n\n" + _JUDGE_TAIL % ", name the unsupported claim if any"
    )
    user = (
        f"QUESTION: {question}\n\nEVIDENCE:\n{_judge_evidence(evidence)}\n\n"
        f"ANSWER:\n{answer}\n\nReturn the JSON now."
    )
    return system, user


def deepseek_judge_gap_detection(question: str, evidence: list[dict], answer: str,
                                 expected_gap: bool) -> tuple[str, str]:
    if expected_gap:
        rule = (
            "The corpus has NO information on this question. The answer "
            "SHOULD explicitly acknowledge this (e.g. 'the corpus does "
            "not cover...', 'no evidence available', etc.). Score:\n"
            "  1.0 = answer explicitly says corpus lacks this info.\n"
            "  0.7-0.9 = answer hedges adequately but doesn't say so outright.\n"
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
    system = ("You judge whether an answer's gap-handling behavior is correct.\n\n"
              + rule + "\n\n" + _JUDGE_TAIL % "")
    user = (
        f"QUESTION: {question}\nEXPECTED_GAP: {expected_gap}\n\n"
        f"EVIDENCE (first 5):\n{_judge_evidence(evidence, cap=5)}\n\n"
        f"ANSWER:\n{answer}\n\nReturn the JSON now."
    )
    return system, user


def deepseek_judge_consistency(question: str, answers: list[str]) -> tuple[str, str]:
    if len(answers) < 2:
        return ("", "")
    body = "\n\n".join(f"--- Answer {i+1} ---\n{a}" for i, a in enumerate(answers))
    system = (
        "You judge whether multiple answers to the same question are "
        "semantically equivalent. Same facts, same framing, same level of "
        "specificity. Score 0.0-1.0:\n"
        "  1.0 = answers are equivalent (same key facts; phrasing may differ).\n"
        "  0.7-0.9 = answers agree on the main facts but differ in detail/emphasis.\n"
        "  0.4-0.6 = answers cover overlapping but distinct facts.\n"
        "  0.0-0.3 = answers contradict each other or share no content.\n\n"
        + _JUDGE_TAIL % ", note key differences"
    )
    user = f"QUESTION: {question}\n\n{body}\n\nReturn the JSON now."
    return system, user


DEEPSEEK_JUDGES = {
    "judge_comprehensiveness": deepseek_judge_comprehensiveness,
    "judge_no_hallucination": deepseek_judge_no_hallucination,
    "judge_gap_detection": deepseek_judge_gap_detection,
    "judge_consistency": deepseek_judge_consistency,
}

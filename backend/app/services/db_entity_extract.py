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
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.entities import Entity
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.models.ontology import OntologyClass
from backend.app.db.session import session_scope
from backend.app.helpers.ontology_pruning import split_disjunction_label
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
# Sibling import: pipeline_llm does NOT import this module, so no cycle. Reused
# rather than re-implemented so the ontology-build guard and the extraction-time
# menu filter can never disagree about what an entity-shaped label looks like.
from backend.app.services.pipeline_llm import (
    _looks_like_entity_not_class,
    _split_camel,
)
from backend.app.services.predicates import (
    RDF_TYPE,
    VIAO_ASSERTS_ABOUT,
)
from backend.app.services.prompts import PROMPTS

_ENTITIES_NS = "https://veerla-ramrao.ai/ontology/entities"

# The sentinel the extractor returns when no candidate class denotes the KIND
# of an entity it found. Before this existed the prompt said "SKIP that
# entity", which is indistinguishable from "there is no entity here" -- so the
# model always picked a same-topic sibling instead. That is how a furniture
# retailer became a ContainerShippingCarrier and a canal became a region.
ABSTAIN_SENTINEL = "NONE_OF_THESE"

# The verdicts `entity_validate` may return. An unrecognised string is treated
# as "correct" -- fail open per item, matching the pass-level policy below.
_VALID_VERDICTS = frozenset({
    "correct", "wrong_class", "not_an_entity", "no_class_fits", "not_in_text",
})
# Verdicts that remove the entity from the kept set.
_DROPPING_VERDICTS = {
    "not_an_entity": "reviewer_removed",
    "not_in_text": "reviewer_removed",
    "no_class_fits": "abstained",
}


def _filter_entities(
    raw_entities: Any,
    cand_iris: set[str],
    ent_drops: dict[str, int],
    abstained: list[dict[str, Any]],
    abstain_cap: int = 200,
) -> list[dict[str, Any]]:
    """Validate one `entity_extract` response into the kept-entity list.

    Pure (no DB, no I/O) so the drop accounting is unit-testable. Every
    rejection increments a named counter rather than vanishing -- the previous
    version used a single bare `continue`, which is why nobody could see how
    often the model went off-menu.
    """
    kept: list[dict[str, Any]] = []
    if not isinstance(raw_entities, list):
        return kept
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        name = (e.get("canonical_name") or "").strip()
        short = (e.get("short_name") or name).strip()
        class_iri = (e.get("class_iri") or "").strip()
        if not name:
            ent_drops["no_name"] += 1
            continue
        if class_iri == ABSTAIN_SENTINEL:
            # The model found a real entity and honestly reported that no
            # candidate class denotes its kind. `entities.class_id` is NOT
            # NULL, so this cannot be persisted without a migration -- record
            # it instead. A rising count is the signal that the ontology is
            # missing a branch, which is actionable in a way a wrong type
            # never was.
            ent_drops["abstained"] += 1
            if len(abstained) < abstain_cap:
                abstained.append({
                    "canonical_name": name,
                    "proposed_type": (e.get("proposed_type") or "").strip(),
                })
            continue
        if not class_iri or class_iri not in cand_iris:
            ent_drops["off_menu_iri"] += 1
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
    return kept


def _sanitise_verdicts(
    parsed: dict[str, Any],
    kept: list[dict[str, Any]],
    cand_iris: set[str],
    class_meta: dict[str, dict[str, str]],
    allowlist: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Clean one `entity_validate` response before it is acted on.

    The reviewer is not trusted either. It can name an IRI that was never on
    the menu, suggest a class that is itself instance-shaped (which would just
    swap one bad type for another), index an entity that does not exist, or
    invent a verdict. Each of those is neutralised here rather than deeper in
    the loop, so the caller only ever sees well-formed input.
    """
    out_verdicts: list[dict[str, Any]] = []
    seen_idx: set[int] = set()
    for v in (parsed.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        try:
            idx = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(kept) or idx in seen_idx:
            continue
        seen_idx.add(idx)
        verdict = str(v.get("verdict") or "").strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "correct"
        better = (v.get("better_class_iri") or "").strip() or None
        if verdict == "wrong_class":
            meta = class_meta.get(better or "", {})
            unfit, _r = _is_menu_unfit_class(
                meta.get("label", ""), meta.get("description", ""), allowlist
            )
            # A suggestion that is off-menu or itself unusable is no
            # suggestion at all: downgrade rather than drop, so a bad
            # reviewer costs precision but never costs the entity.
            if not better or better not in cand_iris or unfit:
                verdict, better = "correct", None
        else:
            better = None
        out_verdicts.append({
            "index": idx,
            "verdict": verdict,
            "better_class_iri": better,
            "reason": str(v.get("reason") or "").strip()[:300],
        })

    out_missing: list[dict[str, Any]] = []
    for m in (parsed.get("missing_entities") or []):
        if not isinstance(m, dict):
            continue
        name = (m.get("canonical_name") or "").strip()
        if not name:
            continue
        iri = (m.get("class_iri") or "").strip()
        if iri and iri != ABSTAIN_SENTINEL and iri not in cand_iris:
            iri = ""
        out_missing.append({
            "canonical_name": name,
            "short_name": (m.get("short_name") or name).strip(),
            "class_iri": iri,
            "reason": str(m.get("reason") or "").strip()[:300],
        })

    actionable = (
        any(v["verdict"] != "correct" for v in out_verdicts) or bool(out_missing)
    )
    return {
        "verdicts": out_verdicts,
        "missing_entities": out_missing,
        "note": str(parsed.get("note") or "").strip()[:400],
        "_actionable": actionable,
    }


def _is_generated_class(extra_metadata: Any) -> bool:
    """Was this class minted by the expansion pipeline, or does it come from a
    source ontology?

    prune-expand stamps `annotations.generated = [true]` and
    `review_status = ["proposed"]` on everything it creates
    (`create_new_class_entry`). Source-ontology classes carry neither. On the
    live DB this is 550 of 2,244 classes -- and it is the same
    newly-created distinction the Layer-H audit already gates on, so the two
    passes agree about whose vocabulary is open to question.
    """
    if not isinstance(extra_metadata, dict):
        return False
    ann = extra_metadata.get("annotations")
    if not isinstance(ann, dict):
        return False
    gen = ann.get("generated")
    if gen is True or (isinstance(gen, list) and True in gen):
        return True
    rev = ann.get("review_status")
    return isinstance(rev, list) and "proposed" in rev


def _is_menu_unfit_class(
    label: str,
    description: str = "",
    allowlist: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Should this class be withheld from the extractor's candidate menu?

    Not a claim the class is wrong -- a claim that OFFERING it invites a wrong
    answer. prune-expand mints individuals as classes (`ChollaUnit4`,
    `ASU2019-01`) and disjunctions from unresolved relation endpoints
    (`SegmentOperatingExpense or SegmentAsset`). Neither is a KIND, so no
    entity can legitimately instantiate them -- yet both rank high in the
    vector search precisely because they are lexically close to the entities
    they were derived from.

    Deliberately uses only the STRONG tier plus disjunctions. There is no LLM
    adjudication on this path, so a wrong exclusion here silently removes a
    legitimate typing target -- the same asymmetric loss the Stage-2 filter
    documents. Borderline shapes stay in the menu and are caught by the
    validator instead.

    Callers must restrict this to LLM-GENERATED classes -- see
    `_is_generated_class`. The heuristics were built to judge fresh Stage-2
    proposals, and turning them loose on a whole merged ontology mis-fires on
    established vocabulary: measured against the live DB it withheld
    `promissory note` and `floating rate note` (the "Note" document tail),
    `bank holding company` and `clearing corporation` (the corporate-suffix
    rule), and `loan or credit account` (a real FIBO label that happens to
    contain "or"). None of those came from an LLM, so none of them should be
    second-guessed by a rule written for LLM output.
    """
    if not isinstance(label, str) or not label.strip():
        return False, ""
    cleaned = label.strip()
    if allowlist and cleaned.lower() in allowlist:
        return False, ""
    if split_disjunction_label(cleaned) or split_disjunction_label(_split_camel(cleaned)):
        return True, "disjunction"
    unfit, reason = _looks_like_entity_not_class(
        cleaned,
        description=description if isinstance(description, str) else "",
        allowlist=allowlist,
    )
    return (True, reason) if unfit else (False, "")


# Warn (not raise) above this share of failed chunks: partial results are real
# and worth keeping, but silent partial loss is how a corpus ends up with gaps
# nobody notices until retrieval is thin.
_FAILURE_WARN_PCT = 10.0


class EntityExtractionFailedError(RuntimeError):
    """Every chunk failed -- the step accomplished nothing.

    Raised so a total provider outage cannot masquerade as "this corpus has no
    entities". Extraction is idempotent, so the fix is to re-run.
    """


@dataclass
class EntityExtractSummary:
    chunks_scanned: int = 0
    chunks_skipped_already: int = 0
    chunks_failed: int = 0
    entities_minted: int = 0
    entities_reused: int = 0
    chunk_entity_edges: int = 0
    entity_relationship_edges: int = 0
    type_edges: int = 0
    tables_scanned: int = 0
    table_entity_edges: int = 0
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    # Why these exist: the entity path used to drop candidates silently at the
    # `class_iri not in cand_iris` guard, which is how the loss stayed
    # invisible while ChollaUnit4 accumulated 19 mistyped entities. The
    # relationship path already had `rel_drops`; this is the entity mirror.
    entity_drops: dict[str, int] = field(default_factory=dict)
    entity_validation: dict[str, int] = field(default_factory=dict)
    # Names the model declined to type because no candidate class denoted
    # their KIND. A rising count here is the honest signal that the ontology
    # is missing a branch -- act on it with a prune-expand run.
    abstained_samples: list[dict[str, Any]] = field(default_factory=list)
    menu_classes_withheld: int = 0


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


# Name tokens too generic to prove an entity is named in a quote. Without
# this, "Ray-Ban Meta smart glasses" would count as named by any sentence
# containing the word "smart".
_WEAK_NAME_TOKENS: frozenset[str] = frozenset({
    "smart", "group", "company", "technologies", "technology", "systems",
    "solutions", "services", "holdings", "international", "global", "digital",
    "media", "news", "online", "network", "networks", "labs", "team", "board",
    "app", "apps", "platform", "glasses", "music", "games", "gaming", "studio",
    "studios", "ventures", "capital", "partners", "fund", "the", "and", "for",
})


def _evidence_names(evidence: str, ent: dict[str, Any]) -> bool:
    """Is `ent` actually NAMED in this evidence span?

    The occurs-in-the-chunk check cannot tell "states a relationship between
    A and B" from "is a real sentence that happens to mention A". On the news
    corpus that gap produced `Anthropic --hasMember--> Mustafa Suleyman`
    quoting a sentence about Anthropic founders that never says "Suleyman".

    Deliberately LENIENT about form, strict about presence: a surname alone
    ("Silverman" for "Sara Silverman"), a possessive ("OpenAI's"), or an
    adjectival form ("Israeli" for "Israel") all count, because dropping a
    true edge over inflection is the worse error. What must be there is some
    distinctive token of the name -- generic ones are excluded, so a quote
    saying "smart" does not name "Ray-Ban Meta smart glasses".
    """
    hay = _normalize_name(evidence)
    if not hay:
        return False
    for form in (ent.get("canonical_name") or "", ent.get("short_name") or ""):
        norm = _normalize_name(form)
        if not norm:
            continue
        if norm in hay:
            return True
        for variant in _short_form_variants(norm):
            if variant and variant in hay:
                return True
        # Fall back to the name's distinctive tokens, so an inflected or
        # partial mention still counts.
        for tok in norm.split():
            if len(tok) >= 4 and tok not in _WEAK_NAME_TOKENS and tok in hay:
                return True
    return False


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


# --------------------------------------------------------------------------- #
# Entity -> entity relationships (Milestone C, second half)
# --------------------------------------------------------------------------- #
# Candidate predicates are derived from the ontology's own DOMAIN and RANGE
# rather than by vector search. Given the classes actually present in a chunk,
# an object property is offerable only when its declared domain and range both
# fall within those classes (or their ancestors) -- so every predicate handed
# to the LLM is type-valid for these entities BY CONSTRUCTION, and the model
# picks from a short, already-correct list instead of being asked to guess and
# be checked afterwards.
#
# prune-expand records domain/range for every property it mints (746/746 on the
# pharma run) and `db_ontology_import` stores the whole record in
# `extra_metadata`, so this needs no schema change and no embedding column.
#
# Ancestors, not descendants: a property declared on `Organization` applies to
# a `PublicCompany` entity, so the entity's classes are walked UP the
# subClassOf chain (source=child -> target=parent, as the importer writes it).

_MAX_CANDIDATE_PREDICATES = 40
# Rescue menu is wider (either-end match), so it needs a higher ceiling; the
# measured either-end pool is a median of 52 per chunk.
_MAX_RESCUE_PREDICATES = 60
# Supporting chunk ids kept per edge (the COUNT is always exact).
_MAX_SUPPORTING_CHUNKS = 20

_ANCESTOR_SQL = sql_text("""
WITH RECURSIVE up(origin, id) AS (
    SELECT oc.iri, oc.id FROM graphrag.ontology_classes oc
     WHERE oc.iri = ANY(CAST(:iris AS text[]))
  UNION
    SELECT up.origin, gr.target_node_id
      FROM up JOIN graphrag.graph_relationships gr
        ON gr.source_node_id = up.id
       AND gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND gr.predicate_label  = 'rdfs:subClassOf'
)
SELECT DISTINCT up.origin, oc.iri
  FROM up JOIN graphrag.ontology_classes oc ON oc.id = up.id
""")

_CANDIDATE_PREDICATE_SQL = sql_text("""
SELECT op.iri, op.label,
       (SELECT d->>'iri' FROM jsonb_array_elements(op.extra_metadata->'domain') d
         WHERE d->>'iri' = ANY(CAST(:iris AS text[])) LIMIT 1) AS dom_iri,
       (SELECT r->>'iri' FROM jsonb_array_elements(op.extra_metadata->'range') r
         WHERE r->>'iri' = ANY(CAST(:iris AS text[])) LIMIT 1) AS rng_iri
  FROM graphrag.ontology_object_properties op
 WHERE jsonb_typeof(op.extra_metadata->'domain') = 'array'
   AND jsonb_typeof(op.extra_metadata->'range')  = 'array'
   AND EXISTS (SELECT 1 FROM jsonb_array_elements(op.extra_metadata->'domain') d
                WHERE d->>'iri' = ANY(CAST(:iris AS text[])))
   AND EXISTS (SELECT 1 FROM jsonb_array_elements(op.extra_metadata->'range') r
                WHERE r->>'iri' = ANY(CAST(:iris AS text[])))
 ORDER BY op.label NULLS LAST
 LIMIT :limit
""")


# The RESCUE menu. The strict menu requires BOTH ends to match a declared
# domain and range, which on the news corpus left a median of 8 predicates out
# of 620 -- so a real relationship with no fitting predicate got forced onto a
# wrong one and then rejected (114 domain_range drops in one run). Widening to
# "either end matches" takes the median to 52 WITHOUT touching the ontology.
# Only rescued claims use it, so first-pass precision is unchanged.
_WIDE_PREDICATE_SQL = sql_text("""
SELECT op.iri, op.label,
       (SELECT d->>'iri' FROM jsonb_array_elements(op.extra_metadata->'domain') d LIMIT 1),
       (SELECT r->>'iri' FROM jsonb_array_elements(op.extra_metadata->'range')  r LIMIT 1)
  FROM graphrag.ontology_object_properties op
 WHERE jsonb_typeof(op.extra_metadata->'domain') = 'array'
   AND jsonb_typeof(op.extra_metadata->'range')  = 'array'
   AND (EXISTS (SELECT 1 FROM jsonb_array_elements(op.extra_metadata->'domain') d
                 WHERE d->>'iri' = ANY(CAST(:iris AS text[])))
     OR EXISTS (SELECT 1 FROM jsonb_array_elements(op.extra_metadata->'range') r
                 WHERE r->>'iri' = ANY(CAST(:iris AS text[]))))
 ORDER BY op.label NULLS LAST
 LIMIT :limit
""")


async def _wide_predicates(
    session: AsyncSession, class_iris: set[str], ancestors: dict[str, set[str]]
) -> list[dict[str, str]]:
    """Predicates where EITHER end matches, for the rescue pass only."""
    if not class_iris:
        return []
    expanded = {a for anc in ancestors.values() for a in anc} | set(class_iris)
    r = await session.execute(
        _WIDE_PREDICATE_SQL,
        {"iris": list(expanded), "limit": _MAX_RESCUE_PREDICATES},
    )
    return [{"iri": iri,
             "label": label or iri.rsplit("#", 1)[-1],
             "domain_label": (dom or "?").rsplit("#", 1)[-1],
             "range_label": (rng or "?").rsplit("#", 1)[-1]}
            for iri, label, dom, rng in r.all()]


async def _rescue_relationships(
    router: Any,
    chunk_text: str,
    rejects: list[dict[str, Any]],
    wide_preds: list[dict[str, str]],
    known_iris: set[str],
    ent_by_norm: dict[str, dict[str, Any]],
    rel_repairs: dict[str, int],
    chunk_iri: str,
) -> list[dict[str, Any]]:
    """Re-home claims the strict type check rejected, using the WIDE menu.

    These are not hallucinations -- each already cleared "the quote occurs in
    the chunk" and "the quote names both ends". What failed is that the
    ontology offered no predicate whose declared domain and range fit this
    particular pair, so the model reached for the closest of its 8 options.
    Given a wider menu it can usually name the relationship properly:
    `Google --hasSubOrganization--> DeepMind` (from "Google ACQUIRED DeepMind")
    becomes `acquires`, which existed in the ontology all along.

    Rescued edges are marked `type_check: "relaxed"` in extra_metadata and
    still face the verification pass, so a wrong rescue is caught there.
    """
    if not rejects or not wide_preds:
        return []
    try:
        s_sys, s_user = PROMPTS["relationship_repair"](
            chunk_text, rejects, wide_preds
        )
        out = await router.chat("relationship_repair", system=s_sys, user=s_user)
        parsed = _extract_json(out.text)
    except Exception as exc:
        print(f"[extract-entities] relationship rescue failed on "
              f"{chunk_iri}: {exc}")
        return []
    if not isinstance(parsed, dict):
        return []

    hay = " ".join(chunk_text.split()).lower()
    rescued: list[dict[str, Any]] = []
    for rel in (parsed.get("relationships") or []):
        if not isinstance(rel, dict):
            continue
        s_ent = ent_by_norm.get(_normalize_name(rel.get("subject") or ""))
        o_ent = ent_by_norm.get(_normalize_name(rel.get("object") or ""))
        pred = (rel.get("predicate_iri") or "").strip()
        if s_ent is None or o_ent is None or pred not in known_iris:
            continue
        if s_ent["canonical_name"] == o_ent["canonical_name"]:
            continue
        ev = " ".join((rel.get("evidence") or "").split())
        # Same evidence bar as the strict path -- widening WHICH predicate may
        # be used never widens what counts as proof.
        if len(ev) < 12 or ev.lower() not in hay:
            continue
        if not (_evidence_names(ev, s_ent) and _evidence_names(ev, o_ent)):
            continue
        rescued.append({
            "subject": s_ent["canonical_name"],
            "object": o_ent["canonical_name"],
            "predicate_iri": pred,
            "confidence": None,
            "evidence": ev[:400],
            "type_check": "relaxed",
        })
    rel_repairs["rescued"] += len(rescued)
    return rescued


async def _verify_relationships(
    router: Any,
    chunk_text: str,
    rels: list[dict[str, Any]],
    pred_list: list[dict[str, str]],
    rel_drops: dict[str, int],
    rel_repairs: dict[str, int],
    pred_constraints: dict[str, tuple[str, str]],
    pred_ancestors: dict[str, set[str]],
    ent_by_norm: dict[str, dict[str, Any]],
    chunk_iri: str,
) -> list[dict[str, Any]]:
    """Drop relationships whose own evidence does not support them.

    Structural checks (quote occurs, both ends named, domain/range) verify
    FORM. This verifies MEANING, which is where the news-corpus errors lived:
    roughly half of a hand-checked sample was wrong despite passing every
    structural gate.

    Three verdicts:
      supported   -> keep
      reversed    -> re-emit with subject/object swapped, IF the swap still
                     satisfies the predicate's domain/range; else drop
      unsupported -> drop

    Fails OPEN on an LLM or parse error: an unreachable verifier must not
    silently empty the graph. A missing verdict is treated as unsupported,
    though, so a model that skips a claim does not smuggle it through.
    """
    label_of = {p["iri"]: (p.get("label") or p["iri"]) for p in pred_list}
    claims = [{
        "subject": r["subject"],
        "object": r["object"],
        "predicate_label": label_of.get(r["predicate_iri"], r["predicate_iri"]),
        "evidence": r["evidence"],
    } for r in rels]

    try:
        v_sys, v_user = PROMPTS["relationship_verify"](chunk_text, claims)
        v_out = await router.chat(
            "relationship_verify", system=v_sys, user=v_user
        )
        parsed = _extract_json(v_out.text)
    except Exception as exc:
        print(f"[extract-entities] relationship verify failed on "
              f"{chunk_iri}: {exc} -- keeping claims unverified")
        return rels

    if not isinstance(parsed, dict):
        return rels

    verdicts: dict[int, str] = {}
    for v in (parsed.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        verdicts[i] = str(v.get("verdict") or "").strip().lower()

    out: list[dict[str, Any]] = []
    for i, rel in enumerate(rels):
        verdict = verdicts.get(i, "unsupported")
        if verdict == "supported":
            out.append(rel)
            continue
        if verdict == "reversed":
            # The assertion is real; only the direction was mis-read. Keep it
            # ONLY if the flipped pair still type-checks -- a swap that
            # violates domain/range means the model is confused about more
            # than direction.
            s_ent = ent_by_norm.get(_normalize_name(rel["object"]))
            o_ent = ent_by_norm.get(_normalize_name(rel["subject"]))
            dom_rng = pred_constraints.get(rel["predicate_iri"])
            if s_ent and o_ent and dom_rng:
                dom_iri, rng_iri = dom_rng
                s_cls, o_cls = s_ent["class_iri"], o_ent["class_iri"]
                if (dom_iri in pred_ancestors.get(s_cls, {s_cls})
                        and rng_iri in pred_ancestors.get(o_cls, {o_cls})):
                    out.append({**rel,
                                "subject": s_ent["canonical_name"],
                                "object": o_ent["canonical_name"]})
                    rel_repairs["direction_swapped"] += 1
                    continue
            rel_drops["reversed"] += 1
            continue
        rel_drops["unsupported"] += 1
    return out


async def _candidate_predicates(
    session: AsyncSession, class_iris: set[str]
) -> tuple[list[dict[str, str]], dict[str, tuple[str, str]], dict[str, set[str]]]:
    """Object properties whose domain AND range both fall in `class_iris`
    (expanded to include ancestors).

    Returns (rendered_for_prompt, constraints, ancestors) where `constraints`
    maps predicate IRI -> (domain_iri, range_iri) and `ancestors` maps each
    input class IRI -> itself plus every superclass.

    `ancestors` exists so the write path can re-check the model's answer with
    the SAME rule used to offer the predicate. Originally the offer expanded
    ancestors while the check demanded exact class equality, so a predicate
    declared on `Organization` was offered for a `PublicCompany` entity and
    then always rejected -- 635 of 1,109 drops on the first real run.
    """
    if not class_iris:
        return [], {}, {}
    r = await session.execute(_ANCESTOR_SQL, {"iris": list(class_iris)})
    ancestors: dict[str, set[str]] = {c: {c} for c in class_iris}
    for origin, anc in r.all():
        ancestors.setdefault(origin, {origin}).add(anc)
    expanded = {a for anc in ancestors.values() for a in anc} | set(class_iris)

    r = await session.execute(
        _CANDIDATE_PREDICATE_SQL,
        {"iris": list(expanded), "limit": _MAX_CANDIDATE_PREDICATES},
    )
    labels: dict[str, str] = {}
    out: list[dict[str, str]] = []
    constraints: dict[str, tuple[str, str]] = {}
    for iri, label, dom_iri, rng_iri in r.all():
        if not dom_iri or not rng_iri:
            continue
        constraints[iri] = (dom_iri, rng_iri)
        out.append({
            "iri": iri,
            "label": label or iri.rsplit("#", 1)[-1],
            "domain_label": labels.get(dom_iri) or dom_iri.rsplit("#", 1)[-1],
            "range_label": labels.get(rng_iri) or rng_iri.rsplit("#", 1)[-1],
        })
    return out, constraints, ancestors


async def report_candidate_distances(
    *,
    chunk_kind: str = "summary",
    sample: int = 300,
) -> None:
    """Print the chunk->class embedding distance distribution by rank.

    Exists because `max_candidate_l2` must be set from measurement, not from a
    guess: too tight and chunks lose every candidate, too loose and it does
    nothing. Reports both the raw per-rank spread AND the distances of the
    assignments that were actually written, which is the more useful number --
    a ceiling below the p99 of real assignments would start discarding work
    the pipeline currently does correctly.
    """
    async with session_scope() as session:
        rows = (await session.execute(
            sql_text("""
                WITH s AS (
                  SELECT id, embedding FROM graphrag.chunks
                   WHERE status='ACTIVE' AND kind=:kind AND embedding IS NOT NULL
                   ORDER BY random() LIMIT :n
                ), d AS (
                  SELECT s.id,
                         row_number() OVER (
                           PARTITION BY s.id ORDER BY oc.embedding <-> s.embedding
                         ) AS rnk,
                         (oc.embedding <-> s.embedding) AS l2
                    FROM s JOIN graphrag.ontology_classes oc
                      ON oc.embedding IS NOT NULL
                )
                SELECT rnk,
                       percentile_cont(0.05) WITHIN GROUP (ORDER BY l2),
                       percentile_cont(0.50) WITHIN GROUP (ORDER BY l2),
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY l2)
                  FROM d WHERE rnk IN (1,5,10,25,50)
                 GROUP BY rnk ORDER BY rnk
            """),
            {"kind": chunk_kind, "n": sample},
        )).all()
        if not rows:
            print(f"[candidate-distances] no {chunk_kind} chunks with embeddings")
            return
        print(f"[candidate-distances] chunk->class L2 by rank "
              f"(sample={sample}, kind={chunk_kind}):")
        print(f"    {'rank':>5}  {'p05':>7}  {'p50':>7}  {'p95':>7}")
        for rnk, p05, p50, p95 in rows:
            print(f"    {rnk:>5}  {p05:>7.4f}  {p50:>7.4f}  {p95:>7.4f}")

        assigned = (await session.execute(
            sql_text("""
                SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY dist),
                       percentile_cont(0.90) WITHIN GROUP (ORDER BY dist),
                       percentile_cont(0.99) WITHIN GROUP (ORDER BY dist),
                       count(*)
                  FROM (
                    SELECT (oc.embedding <-> c.embedding) AS dist
                      FROM graphrag.graph_relationships gr
                      JOIN graphrag.chunks c ON c.id = gr.source_chunk_id
                      JOIN graphrag.entities e ON e.id = gr.target_node_id
                      JOIN graphrag.ontology_classes oc ON oc.id = e.class_id
                     WHERE gr.predicate_iri = :pred
                       AND gr.source_node_type = 'chunk'
                       AND c.embedding IS NOT NULL
                       AND oc.embedding IS NOT NULL
                  ) t
            """),
            {"pred": VIAO_ASSERTS_ABOUT},
        )).first()
    if assigned and assigned[3]:
        print(
            f"\n[candidate-distances] distances of assignments ACTUALLY written "
            f"(n={assigned[3]:,}): p50={assigned[0]:.4f} p90={assigned[1]:.4f} "
            f"p99={assigned[2]:.4f}"
        )
        print(
            f"[candidate-distances] a ceiling near p99 ({assigned[2]:.2f}) keeps "
            f"today's correct assignments while cutting the long tail. Set it "
            f"as extraction.max_candidate_l2 or --max-candidate-l2."
        )
    else:
        print("\n[candidate-distances] no existing assignments to calibrate against")


async def extract_entities(
    *,
    scope_document_iri: str | None = None,
    limit: int | None = None,
    candidate_classes_per_chunk: int = 50,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
    chunk_kind: str = "summary",
    extract_relationships: bool = True,
    verify_relationships: bool = True,
    rescue_relationships: bool = False,
    entity_identity: str = "name",
    validate_entities: bool = False,
    validation_rounds: int = 2,
    filter_candidate_menu: bool = True,
    max_candidate_l2: float | None = None,
    menu_filter_allowlist: frozenset[str] | None = None,
    menu_ancestor_closure: bool = True,
    pinned_class_labels: tuple[str, ...] = (),
) -> EntityExtractSummary:
    """Drive entity extraction over chunks that haven't been processed.

    `chunk_kind` ('summary' default | 'fulltext'): which chunk set to mine.
    Summary is cheap but only captures what survived summarization; 'fulltext'
    mines the verbatim chunks (far more complete, e.g. every clinical study),
    at ~18x the LLM calls + more DB rows. Requires --full-text-chunks at ingest.

    `validate_entities` (default OFF) turns on the review loop: a stronger
    model audits each chunk's entities + class assignments and, when it finds
    something actionable, extraction re-runs for that chunk with the critique
    appended. `validation_rounds` caps the validate->re-extract cycles.

    `filter_candidate_menu` withholds instance-shaped and disjunction-shaped
    class labels from the top-K menu. `max_candidate_l2` (None = no ceiling)
    caps how far a candidate class may sit from the chunk in embedding space;
    without it every chunk gets exactly K classes however unrelated, so the
    model always has a same-topic sibling to force an entity into.

    `menu_ancestor_closure` adds the superclass chain of every menu entry, and
    `pinned_class_labels` force-includes universal classes (person /
    organization / place). Together they give the model a general answer to
    fall back on -- without them a domain corpus fills all K slots with
    hyper-specific classes and named people get abstained rather than typed.
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
    # `relationship_extract` is a NEW models.yaml task. A deployment running an
    # older config would otherwise raise KeyError once per chunk -- caught, but
    # 442 identical tracebacks. Check once and degrade to entities-only.
    if extract_relationships:
        try:
            router.task_spec("relationship_extract")
        except KeyError:
            print(
                "[extract-entities] models.yaml has no 'relationship_extract' "
                "task -- skipping entity->entity relationships. Add it (see "
                "config/models.example.yaml) or pass --no-relationships to "
                "silence this."
            )
            extract_relationships = False
    # Same degradation for the verifier: an older config should lose the
    # extra precision, not the whole relationship pass.
    if extract_relationships and verify_relationships:
        try:
            router.task_spec("relationship_verify")
        except KeyError:
            print(
                "[extract-entities] models.yaml has no 'relationship_verify' "
                "task -- relationships will be written WITHOUT the "
                "evidence-support check. Add it (see "
                "config/models.example.yaml) for higher precision."
            )
            verify_relationships = False
    # Same degradation for the entity reviewer: an older models.yaml should
    # lose the review pass, not raise once per chunk.
    if validate_entities:
        try:
            router.task_spec("entity_validate")
        except KeyError:
            print(
                "[extract-entities] models.yaml has no 'entity_validate' task "
                "-- entities will be written WITHOUT the class-assignment "
                "review pass. Add it (see config/models.example.yaml) to "
                "enable --validate-entities."
            )
            validate_entities = False
    if validate_entities and validation_rounds < 1:
        validate_entities = False

    # Candidate-menu quality filter, computed ONCE per run: one query over
    # labels + descriptions, no vectors. Withholding a class here is cheaper
    # and safer than trying to repair the answer afterwards -- the extractor
    # cannot pick what it was never shown.
    menu_excluded: set[str] = set()
    if filter_candidate_menu:
        _reasons: dict[str, int] = {}
        async with session_scope() as session:
            r = await session.execute(
                select(
                    OntologyClass.iri,
                    OntologyClass.label,
                    OntologyClass.description,
                    OntologyClass.extra_metadata,
                )
            )
            for iri, label, descr, meta in r.all():
                # Only LLM-minted classes are candidates for withholding. A
                # source ontology's vocabulary is not ours to second-guess.
                if not _is_generated_class(meta):
                    continue
                unfit, reason = _is_menu_unfit_class(
                    label or "", descr or "", menu_filter_allowlist
                )
                if unfit:
                    menu_excluded.add(iri)
                    _reasons[reason] = _reasons.get(reason, 0) + 1
        summary.menu_classes_withheld = len(menu_excluded)
        if menu_excluded:
            _detail = ", ".join(f"{k}={v}" for k, v in sorted(_reasons.items()))
            print(
                f"[extract-entities] candidate-menu filter: "
                f"{len(menu_excluded)} class(es) withheld ({_detail})"
            )

    # Resolve the pinned universal classes once. Matched on LABEL (case- and
    # space-insensitive) rather than IRI so the same config works whichever
    # upper ontology a corpus merged -- foaf, org, schema.org.
    pinned_iris: set[str] = set()
    if pinned_class_labels:
        wanted = {
            re.sub(r"[^a-z0-9]", "", lbl.lower())
            for lbl in pinned_class_labels if isinstance(lbl, str) and lbl.strip()
        }
        async with session_scope() as session:
            rows = await session.execute(
                select(OntologyClass.iri, OntologyClass.label)
                .where(OntologyClass.embedding.isnot(None))
            )
            for iri, label in rows.all():
                if re.sub(r"[^a-z0-9]", "", (label or "").lower()) in wanted:
                    pinned_iris.add(iri)
        print(
            f"[extract-entities] menu backstop: ancestor-closure="
            f"{'on' if menu_ancestor_closure else 'off'}, "
            f"{len(pinned_iris)} pinned class(es) resolved from "
            f"{len(wanted)} configured label(s)"
        )

    sem = asyncio.Semaphore(concurrency)
    # (chunk_id, chunk_iri, doc_id, list[entity_dict], list[relationship_dict])
    results: list[
        tuple[Any, str, Any, list[dict[str, Any]] | None, list[dict[str, Any]]]
    ] = ([None] * len(chunks))  # type: ignore[list-item]
    # Why a drop tally: the entity path's `class_iri not in cand_iris` guard
    # discards silently, which is how losses stayed invisible. Count instead.
    rel_drops: dict[str, int] = {
        "unresolved": 0, "bad_predicate": 0, "domain_range": 0,
        "no_evidence": 0, "one_sided_evidence": 0,
        "unsupported": 0, "reversed": 0,
    }
    # The entity mirror of `rel_drops`. Line-for-line, the old code did a bare
    # `continue` for every one of these -- so the corpus could lose entities
    # steadily with nothing in the output to show for it.
    ent_drops: dict[str, int] = {
        "off_menu_iri": 0, "no_name": 0, "abstained": 0,
        "reviewer_removed": 0, "no_candidates": 0,
    }
    val_stats: dict[str, int] = {
        "clean": 0, "revised": 0, "reclassified": 0, "removed": 0,
        "added": 0, "empty_reextract_rejected": 0, "validator_failed": 0,
    }
    abstained: list[dict[str, Any]] = []
    _abstain_sample_cap = 200
    # Verification pass repairs as well as rejects: a claim whose quote states
    # the relationship backwards is re-emitted with the ends swapped rather
    # than discarded, since the model found a real assertion and only mis-read
    # its direction.
    rel_repairs = {"direction_swapped": 0, "rescued": 0}
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
        """The candidate menu: vector top-K, plus a general-class backstop.

        Guards beyond the plain ANN query:

        * `menu_excluded` drops instance-shaped / disjunction labels. We
          over-fetch 3x first so filtering rarely shrinks the menu below K.
        * `max_candidate_l2` caps the distance. Without a ceiling this is a
          pure `ORDER BY ... LIMIT K`, so a chunk ALWAYS receives exactly K
          classes no matter how unrelated they are.
        * ANCESTOR CLOSURE + PINS -- see below.

        Why the backstop exists. A pure top-K menu is dominated by whatever is
        lexically closest to the chunk, which on a domain corpus means K
        hyper-specific classes and no general ones. Measured on a shipping
        article, `foaf:Person` ranked 191st while the top 10 were
        ShippingBoomBustCycle, ContainerCarrier, FreightRate...; every named
        person in the passage was dropped because nothing on the menu denoted
        "a human being". On a utility 10-K the menu offered `AESO` and `MISO`
        but not `Organization`, and 276 of ~340 mentions were abstained.

        Two additions, in order of principle:

        1. Ancestor closure -- for every class on the menu, its superclass
           chain is added too. Fully domain-neutral: if `ContainerCarrier` is
           offered then `Organization` is offered, and if `Israel` is offered
           then `Country` and `GeographicEntity` are. This is what lets the
           model answer "what KIND of thing is this?" with the right level of
           generality instead of the only level it was shown.
        2. Configured pins -- a short list of universal classes (person,
           organization, place) that entity extraction needs in EVERY domain.
           Needed on top of (1) because a chunk whose top-K contains no
           person-ish class at all has no person ancestor to close over.
        """
        over_fetch = (
            candidate_classes_per_chunk * 3
            if menu_excluded
            else candidate_classes_per_chunk
        )
        async with session_scope() as session:
            stmt = select(
                OntologyClass.iri,
                OntologyClass.label,
                OntologyClass.description,
            ).where(OntologyClass.embedding.isnot(None))
            if max_candidate_l2 is not None:
                stmt = stmt.where(
                    OntologyClass.embedding.l2_distance(chunk_embedding)
                    <= max_candidate_l2
                )
            r = await session.execute(
                stmt.order_by(OntologyClass.embedding.l2_distance(chunk_embedding))
                .limit(over_fetch)
            )
            out = [
                {"iri": iri, "label": label or "", "description": descr or ""}
                for iri, label, descr in r.all()
                if iri not in menu_excluded
            ][:candidate_classes_per_chunk]

            have = {c["iri"] for c in out}
            extra_iris: set[str] = set()
            if menu_ancestor_closure and out:
                anc = await session.execute(
                    _ANCESTOR_SQL, {"iris": [c["iri"] for c in out]}
                )
                extra_iris |= {
                    a for _origin, a in anc.all()
                    if a not in have and a not in menu_excluded
                }
            extra_iris |= {i for i in pinned_iris if i not in have}
            if not extra_iris:
                return out
            rows = await session.execute(
                select(
                    OntologyClass.iri, OntologyClass.label, OntologyClass.description
                ).where(OntologyClass.iri.in_(sorted(extra_iris)))
            )
            for iri, label, descr in rows.all():
                out.append({
                    "iri": iri, "label": label or "", "description": descr or ""
                })
            return out

    async def _validate_pass(
        txt: str,
        candidates: list[dict[str, str]],
        kept: list[dict[str, Any]],
        cand_iris: set[str],
        class_meta: dict[str, dict[str, str]],
        chunk_iri: str,
    ) -> dict[str, Any] | None:
        """One `entity_validate` call, sanitised. None => fail open.

        Applies the reviewer's DROP verdicts here (they need no second
        extraction) and leaves `wrong_class` / `missing_entities` to the
        re-extraction, which is the only pass that can reconsider what to
        pull out of the passage in the first place.
        """
        entities_for_review = [
            {
                "index": i,
                "canonical_name": e["canonical_name"],
                "short_name": e["short_name"],
                "class_iri": e["class_iri"],
                "class_label": class_meta.get(e["class_iri"], {}).get("label", ""),
                "class_description":
                    class_meta.get(e["class_iri"], {}).get("description", ""),
            }
            for i, e in enumerate(kept)
        ]
        try:
            v_sys, v_user = PROMPTS["entity_validate"](
                txt, candidates, entities_for_review
            )
            out = await router.chat("entity_validate", system=v_sys, user=v_user)
            parsed = _extract_json(out.text)
        except Exception as exc:
            print(f"[extract-entities] chunk {chunk_iri} validator failed: {exc}")
            val_stats["validator_failed"] += 1
            return None
        if not isinstance(parsed, dict):
            val_stats["validator_failed"] += 1
            return None

        feedback = _sanitise_verdicts(
            parsed, kept, cand_iris, class_meta, menu_filter_allowlist
        )
        # Count what the reviewer actually asked for, so the run summary can
        # show whether the pass is earning its cost.
        for v in feedback["verdicts"]:
            if v["verdict"] == "wrong_class":
                val_stats["reclassified"] += 1
            elif v["verdict"] in _DROPPING_VERDICTS:
                val_stats["removed"] += 1
        val_stats["added"] += len(feedback["missing_entities"])

        # Apply the drop verdicts immediately: removing a hallucinated or
        # generic entity needs no re-extraction, and dropping it now keeps it
        # out of the PRIOR ATTEMPT block so the next round is not re-primed
        # with the thing we just rejected.
        drops = {
            v["index"]: _DROPPING_VERDICTS[v["verdict"]]
            for v in feedback["verdicts"]
            if v["verdict"] in _DROPPING_VERDICTS
        }
        if drops:
            n_before = len(kept)
            for i, bucket in drops.items():
                ent_drops[bucket] += 1
                if bucket == "abstained" and len(abstained) < _abstain_sample_cap:
                    abstained.append({
                        "canonical_name": kept[i]["canonical_name"],
                        "proposed_type": "(reviewer: no class fits)",
                    })
            kept[:] = [e for i, e in enumerate(kept) if i not in drops]
            # Verdict indices refer to the PRE-drop ordering, so rebuild them
            # against the surviving list before they are rendered into the
            # feedback block -- otherwise the re-extraction prompt points its
            # critique at the wrong entities.
            remap = {
                old: new
                for new, old in enumerate(
                    i for i in range(n_before) if i not in drops
                )
            }
            feedback["verdicts"] = [
                {**v, "index": remap[v["index"]]}
                for v in feedback["verdicts"]
                if v["index"] in remap
            ]
        # Recompute: the drops are already applied, so they no longer justify
        # paying for a re-extraction. Only an unresolved reclassification or a
        # missed entity does -- both need the extractor to look again.
        feedback["_actionable"] = (
            any(v["verdict"] == "wrong_class" for v in feedback["verdicts"])
            or bool(feedback["missing_entities"])
        )
        feedback["prior"] = [
            {
                "canonical_name": e["canonical_name"],
                "class_iri": e["class_iri"],
                "class_label": class_meta.get(e["class_iri"], {}).get("label", ""),
            }
            for e in kept
        ]
        return feedback

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
                    # Not a failure. With an L2 ceiling set, a chunk whose
                    # subject matter the ontology simply does not cover
                    # legitimately has no candidates -- and routing that into
                    # `chunks_failed` would pollute the failure-rate warning
                    # and could trip EntityExtractionFailedError, turning a
                    # too-tight ceiling into a hard abort mid-run.
                    ent_drops["no_candidates"] += 1
                    results[idx] = (chunk_id, chunk_iri, doc_id, [], [])
                    summary.chunks_scanned += 1
                    async with progress_lock:
                        progress_state["done"] += 1
                        progress_state["ok"] += 1
                    return
                cand_iris = {c["iri"] for c in candidates}
                class_meta = {
                    c["iri"]: {"label": c["label"], "description": c["description"]}
                    for c in candidates
                }

                async def _extract_pass(
                    feedback: dict[str, Any] | None,
                ) -> list[dict[str, Any]] | None:
                    """One entity_extract call + parse-retry + filtering.

                    Returns None when the response never parsed, so the caller
                    can keep the previous round's result rather than treating
                    an unparseable retry as "no entities here"."""
                    system, user = PROMPTS["entity_extract"](
                        txt, candidates, feedback=feedback
                    )
                    # Parse-retry: Anthropic has no JSON-grammar mode, so Haiku
                    # occasionally emits an unparseable response. Re-ask once
                    # (non-deterministic -> a retry usually parses).
                    parsed_local = None
                    for _attempt in range(2):
                        out = await router.chat(
                            "entity_extract", system=system, user=user
                        )
                        parsed_local = _extract_json(out.text)
                        if isinstance(parsed_local, dict):
                            break
                    if not isinstance(parsed_local, dict):
                        return None
                    return _filter_entities(
                        parsed_local.get("entities"),
                        cand_iris,
                        ent_drops,
                        abstained,
                        _abstain_sample_cap,
                    )

                kept = await _extract_pass(None)
            except Exception as exc:
                print(f"[extract-entities] chunk {chunk_iri} call failed: {exc}")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return

            if kept is None:
                print(f"[extract-entities] chunk {chunk_iri} unparseable response (after retry)")
                summary.chunks_failed += 1
                async with progress_lock:
                    progress_state["done"] += 1
                    progress_state["fail"] += 1
                return

            # ---- Review loop: validate -> critique -> re-extract ----
            #
            # The reviewer FAILS OPEN. On any exception or unparseable
            # response we keep the pass-1 entities. This is deliberately the
            # opposite of `relationship_verify`, where a missing verdict means
            # "unsupported": there the default discards one claim, here it
            # would discard an entire chunk's entities. An unreachable or
            # misbehaving reviewer must never empty the graph.
            if validate_entities and kept:
                for _round in range(validation_rounds):
                    feedback = await _validate_pass(
                        txt, candidates, kept, cand_iris, class_meta, chunk_iri,
                    )
                    if feedback is None:
                        break
                    if not feedback["_actionable"]:
                        val_stats["clean"] += 1
                        break
                    try:
                        revised = await _extract_pass(feedback)
                    except Exception as exc:
                        print(
                            f"[extract-entities] chunk {chunk_iri} re-extract "
                            f"failed: {exc}"
                        )
                        break
                    if revised is None:
                        break
                    if not revised:
                        # A re-extraction that empties a chunk which had
                        # entities is a regression, not a correction. Keep
                        # pass-1 rather than letting one bad round delete the
                        # chunk's contribution to the graph.
                        val_stats["empty_reextract_rejected"] += 1
                        break
                    val_stats["revised"] += 1
                    kept = revised

            # ---- Second pass: relationships ----
            #
            # Runs only now, because the predicate menu can only be narrowed
            # to the entities' ACTUAL classes once those entities exist. In
            # the single-call version predicates came from the chunk's top-50
            # candidate CLASSES, so the model saw a median of 12 predicates
            # while holding 4-5 entities -- 669 of 1,140 proposals were
            # rejected on domain/range alone.
            #
            # Skipped entirely when no predicate fits those classes, so a
            # chunk with nothing assertable costs no second call.
            rels: list[dict[str, Any]] = []
            if extract_relationships and len(kept) >= 2:
                ent_class_iris = {e["class_iri"] for e in kept}
                async with session_scope() as session:
                    pred_list, pred_constraints, pred_ancestors = (
                        await _candidate_predicates(session, ent_class_iris)
                    )
                if pred_list:
                    _lbl = {c["iri"]: c["label"] for c in candidates}
                    ents_for_prompt = [
                        {"canonical_name": e["canonical_name"],
                         "class_label": _lbl.get(e["class_iri"], "")}
                        for e in kept
                    ]
                    try:
                        r_sys, r_user = PROMPTS["relationship_extract"](
                            txt, ents_for_prompt, pred_list
                        )
                        r_out = await router.chat(
                            "relationship_extract", system=r_sys, user=r_user
                        )
                        r_parsed = _extract_json(r_out.text)
                    except Exception as exc:
                        print(f"[extract-entities] relationship call failed "
                              f"on {chunk_iri}: {exc}")
                        r_parsed = None

                    if isinstance(r_parsed, dict):
                        _by_norm = {
                            _normalize_name(e["canonical_name"]): e for e in kept
                        }
                        for e in kept:
                            _by_norm.setdefault(
                                _normalize_name(e["short_name"]), e
                            )
                        # Evidence is matched against the chunk with whitespace
                        # collapsed, so a quote that differs only in wrapping
                        # still counts.
                        hay = " ".join(txt.split()).lower()
                        # Claims with sound evidence but an ill-fitting
                        # predicate; re-homed by the rescue pass below.
                        rejects: list[dict[str, Any]] = []
                        _pred_label = {p["iri"]: (p.get("label") or p["iri"])
                                       for p in pred_list}
                        for rel in (r_parsed.get("relationships") or []):
                            if not isinstance(rel, dict):
                                continue
                            s_ent = _by_norm.get(
                                _normalize_name(rel.get("subject") or "")
                            )
                            o_ent = _by_norm.get(
                                _normalize_name(rel.get("object") or "")
                            )
                            pred = (rel.get("predicate_iri") or "").strip()
                            if s_ent is None or o_ent is None:
                                rel_drops["unresolved"] += 1
                                continue
                            if s_ent["canonical_name"] == o_ent["canonical_name"]:
                                continue
                            if pred not in pred_constraints:
                                rel_drops["bad_predicate"] += 1
                                continue
                            dom_iri, rng_iri = pred_constraints[pred]
                            s_cls, o_cls = s_ent["class_iri"], o_ent["class_iri"]
                            _type_ok = (
                                dom_iri in pred_ancestors.get(s_cls, {s_cls})
                                and rng_iri in pred_ancestors.get(
                                    o_cls, {o_cls}))
                            # Narrowing the menu makes the type check weaker
                            # (offer and check now share a basis), so the
                            # model must QUOTE the text that asserts this.
                            # An unquotable claim is world knowledge, not
                            # something this passage said.
                            ev = " ".join((rel.get("evidence") or "").split())
                            if len(ev) < 12 or ev.lower() not in hay:
                                rel_drops["no_evidence"] += 1
                                continue
                            # ...and it must NAME BOTH ends. A quote that
                            # mentions only one is about something else: the
                            # news run asserted `Anthropic hasMember Mustafa
                            # Suleyman` on a real sentence that never says
                            # "Suleyman", and `Datadog linkedTo Valor VC` on
                            # one that never says "Valor".
                            if not (_evidence_names(ev, s_ent)
                                    and _evidence_names(ev, o_ent)):
                                rel_drops["one_sided_evidence"] += 1
                                continue
                            # The type check runs LAST now, so a claim whose
                            # evidence is sound but whose predicate does not
                            # type-check is held for the rescue pass rather
                            # than thrown away. It has already proved it is
                            # not a hallucination.
                            if not _type_ok:
                                rel_drops["domain_range"] += 1
                                rejects.append({
                                    "subject": s_ent["canonical_name"],
                                    "object": o_ent["canonical_name"],
                                    "predicate_label": _pred_label.get(
                                        pred, pred),
                                    "evidence": ev[:400],
                                })
                                continue
                            try:
                                rconf = (float(rel["confidence"])
                                         if rel.get("confidence") is not None
                                         else None)
                            except (TypeError, ValueError):
                                rconf = None
                            rels.append({
                                "subject": s_ent["canonical_name"],
                                "object": o_ent["canonical_name"],
                                "predicate_iri": pred,
                                "confidence": rconf,
                                "evidence": ev[:400],
                            })

                        # ---- Rescue: re-home the type-check rejects --------
                        #
                        # Answers "are the dropped edges added back?" -- yes,
                        # when the drop was a vocabulary problem rather than a
                        # truth problem. Each reject already proved its quote
                        # is real and names both ends; only the predicate did
                        # not fit. The wide menu usually holds the right one.
                        if rejects and rescue_relationships:
                            async with session_scope() as session:
                                wide = await _wide_predicates(
                                    session, ent_class_iris, pred_ancestors)
                            _known = {p["iri"] for p in wide}
                            rels.extend(await _rescue_relationships(
                                router, txt, rejects, wide, _known,
                                _by_norm, rel_repairs, chunk_iri,
                            ))

                        # ---- Third pass: does the quote SUPPORT the claim? --
                        #
                        # Everything above verifies form, not meaning: the
                        # quote is real, both ends are named, the types line
                        # up. None of that catches `OpenAI
                        # --filesLawsuitAgainst--> Sara Silverman` quoting
                        # "lawsuits filed against OpenAI BY Sara Silverman" --
                        # a quote that passes every structural check while
                        # asserting the opposite. One batched call per chunk
                        # judges all surviving claims at once.
                        if rels and verify_relationships:
                            rels = await _verify_relationships(
                                router, txt, rels, pred_list,
                                rel_drops, rel_repairs, pred_constraints,
                                pred_ancestors, _by_norm, chunk_iri,
                            )

            results[idx] = (chunk_id, chunk_iri, doc_id, kept, rels)

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

    # Fail loudly when nothing worked. Previously this returned normally after
    # 0 successes, printing "DONE: entities (minted=0)" and exiting 0 -- making
    # a total outage (expired key, exhausted credit, provider down)
    # indistinguishable from "this corpus genuinely had no entities". Observed
    # for real: 0 success / 31 failed, exit 0, empty entities table, no error.
    # Only the NEXT stage's precondition guard revealed it.
    attempted = summary.chunks_scanned + summary.chunks_failed
    if attempted and summary.chunks_scanned == 0:
        raise EntityExtractionFailedError(
            f"all {summary.chunks_failed} chunk(s) failed entity extraction; "
            f"refusing to report success with an empty entities table. "
            f"Check provider credentials/credit and re-run -- the step is "
            f"idempotent, so nothing is lost."
        )
    # A partial outage still writes real rows, so it must not abort -- but it
    # must be impossible to miss in a long detached run's log.
    if summary.chunks_failed and attempted:
        pct = 100.0 * summary.chunks_failed / attempted
        if pct >= _FAILURE_WARN_PCT:
            print(
                f"[extract-entities] WARNING: {summary.chunks_failed}/{attempted} "
                f"chunk(s) failed ({pct:.0f}%) -- entities from those chunks are "
                f"MISSING. Re-run after fixing the cause; extraction is idempotent."
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
    # normalized_name -> every class_id the extractor assigned it anywhere, so
    # collapsing to one node still records all of its types.
    extra_type_classes: dict[str, set[Any]] = {}

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

    # ---- Name-level identity ------------------------------------------------
    #
    # Identity used to be (normalized_name, class_id), so the SAME entity typed
    # differently in two chunks became two nodes. Measured on the 30-doc news
    # corpus: 901 rows for 624 distinct names -- "google" x18, "microsoft" x16,
    # "meta" x12, each carrying a different hyper-specific class
    # (SearchEngine, OpenAICompetitor, SearchMonopoly, ...).
    #
    # That is fatal for multi-hop: an edge hung off Google-as-SearchEngine is
    # unreachable when traversal arrives at Google-as-OpenAICompetitor.
    #
    # Fix: one node per name. The PRIMARY class is the one the extractor chose
    # most often for that name (ties -> first seen, which is stable because
    # `results` is index-ordered). Every other observed class is still recorded
    # as an additional rdf:type edge below, so nothing is lost -- an entity
    # genuinely may hold several types.
    if entity_identity == "name":
        _class_votes: dict[str, dict[str, int]] = {}
        for tup in results:
            if tup is None:
                continue
            for e in (tup[3] or []):
                nrm = _normalize_name(e["canonical_name"])
                if nrm and e["class_iri"] in class_id_by_iri:
                    _class_votes.setdefault(nrm, {})
                    _class_votes[nrm][e["class_iri"]] = (
                        _class_votes[nrm].get(e["class_iri"], 0) + 1)
        primary_class_by_name = {
            nrm: max(votes.items(), key=lambda kv: kv[1])[0]
            for nrm, votes in _class_votes.items()
        }
        _merged = sum(1 for v in _class_votes.values() if len(v) > 1)
        if _merged:
            print(f"[extract-entities] name-level identity: {_merged} name(s) "
                  f"seen with more than one class collapsed to a single node "
                  f"(extra classes kept as additional rdf:type edges)")
    else:
        primary_class_by_name = {}

    for tup in results:
        if tup is None:
            continue
        chunk_id, chunk_iri, doc_id, kept, _rels = tup
        if not kept:
            continue
        for e in kept:
            normalized = _normalize_name(e["canonical_name"])
            if not normalized:
                continue
            # Every mention of a name resolves to the SAME node, whatever class
            # this particular chunk chose for it.
            observed_class_iri = e["class_iri"]
            primary_iri = primary_class_by_name.get(normalized, e["class_iri"])
            class_id = class_id_by_iri.get(primary_iri)
            if class_id is None:
                continue
            observed_class_id = class_id_by_iri.get(observed_class_iri)
            if observed_class_id is not None:
                extra_type_classes.setdefault(normalized, set()).add(
                    observed_class_id)
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
        # Collapsing to one node must not lose the other types the extractor
        # saw for this name -- emit an rdf:type edge for each.
        for extra_cid in extra_type_classes.get(p["normalized_name"], ()):
            if extra_cid != p["class_id"]:
                minted_entity_class_pairs.append((eid, extra_cid))

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

    # Entity -> entity edges (Milestone C, second half).
    #
    # Resolved only now, because entity ids do not exist until the mint/reuse
    # pass above has run. A name is resolved the same way the entity itself
    # was keyed -- (normalized_name, class_id) -- so a relationship can only
    # point at an entity this run actually persisted.
    #
    # `graph_relationships` has NO unique constraint, so re-running would
    # duplicate every edge. Dedupe explicitly, both within this run and
    # against what the DB already holds.
    rel_payloads: list[dict[str, Any]] = []
    if extract_relationships:
        _pred_label: dict[str, str] = {}
        rel_by_sig: dict[tuple[Any, str, Any], dict[str, Any]] = {}
        for tup in results:
            if tup is None:
                continue
            chunk_id, chunk_iri, doc_id, kept, rels = tup
            if not rels or not kept:
                continue
            _cls_of = {e["canonical_name"]: e["class_iri"] for e in kept}

            def _resolve(name: str, _cls_of=_cls_of) -> Any:
                # Look the entity up under the class it was STORED with, not
                # the one this chunk happened to assign. Under name-level
                # identity those differ: the node carries the majority class,
                # so keying on the per-chunk class misses and the edge is lost
                # as "unresolved" -- measured 49 spurious drops, edges falling
                # 30 -> 10, when identity was collapsed without fixing this.
                nrm = _normalize_name(name)
                if not nrm:
                    return None
                primary_iri = primary_class_by_name.get(nrm) or _cls_of.get(name)
                cid = class_id_by_iri.get(primary_iri) if primary_iri else None
                if cid is None:
                    return None
                return seen_in_this_run.get((nrm, cid))

            for rel in rels:
                sid = _resolve(rel["subject"])
                oid = _resolve(rel["object"])
                if sid is None or oid is None or sid == oid:
                    rel_drops["unresolved"] += 1
                    continue
                sig = (sid, rel["predicate_iri"], oid)
                if sig in rel_by_sig:
                    # Same triple asserted by another chunk. ONE edge, but
                    # record the extra chunk: how many independent passages
                    # support an edge separates a corroborated claim from a
                    # one-off mention, and keeping only the first chunk id
                    # threw that away.
                    rel_by_sig[sig]["_chunks"].append(chunk_id)
                    continue
                rel_by_sig[sig] = ({
                    "source_node_type": "entity",
                    "source_node_id": sid,
                    "target_node_type": "entity",
                    "target_node_id": oid,
                    "predicate_iri": rel["predicate_iri"],
                    "predicate_label": _pred_label.get(
                        rel["predicate_iri"],
                        rel["predicate_iri"].rsplit("#", 1)[-1],
                    ),
                    "relationship_type": rel["predicate_iri"].rsplit("#", 1)[-1],
                    "relationship_source": "DOCUMENT_EXTRACTION",
                    "is_authoritative": False,
                    # Traceability: which chunk asserted it.
                    "source_chunk_id": chunk_id,
                    "source_document_id": doc_id,
                    "source_artifact_id": None,
                    "graph_version": gv,
                    "extra_metadata": {
                        **({"confidence": rel["confidence"]}
                           if rel.get("confidence") is not None else {}),
                        # The verbatim span that asserted it -- makes an edge
                        # auditable without re-reading the whole chunk.
                        **({"evidence": rel["evidence"]}
                           if rel.get("evidence") else {}),
                        # Marks an edge the strict domain/range check rejected
                        # and the rescue pass re-homed, so a later audit can
                        # tell the two populations apart.
                        **({"type_check": rel["type_check"]}
                           if rel.get("type_check") else {}),
                    },
                })
                rel_by_sig[sig]["_chunks"] = [chunk_id]

        # Fold the supporting-chunk list into extra_metadata. Capped so a
        # heavily-repeated triple cannot grow the JSONB without bound; the
        # count stays exact either way.
        for payload in rel_by_sig.values():
            chunks = payload.pop("_chunks", [])
            payload["extra_metadata"] = {
                **payload["extra_metadata"],
                "support_count": len(chunks),
                "supporting_chunks": [str(c) for c in chunks[:_MAX_SUPPORTING_CHUNKS]],
            }
        rel_payloads = list(rel_by_sig.values())

        # Drop anything the DB already has (idempotent re-runs).
        if rel_payloads:
            async with session_scope() as session:
                existing = await session.execute(
                    select(
                        GraphRelationship.id,
                        GraphRelationship.source_node_id,
                        GraphRelationship.predicate_iri,
                        GraphRelationship.target_node_id,
                    ).where(
                        GraphRelationship.source_node_type == "entity",
                        GraphRelationship.target_node_type == "entity",
                        GraphRelationship.source_node_id.in_(
                            [p["source_node_id"] for p in rel_payloads]
                        ),
                    )
                )
                have = {(s, p, t): i for i, s, p, t in existing.all()}

            # An edge the DB already holds is NOT simply dropped: this run may
            # have found new passages supporting it, and losing those would
            # make support_count depend on how the corpus happened to be
            # batched. Merge the new chunk ids into the stored list instead.
            merged = 0
            to_update: list[tuple[Any, dict[str, Any]]] = []
            fresh: list[dict[str, Any]] = []
            for pay in rel_payloads:
                key = (pay["source_node_id"], pay["predicate_iri"],
                       pay["target_node_id"])
                row_id = have.get(key)
                if row_id is None:
                    fresh.append(pay)
                else:
                    to_update.append((row_id, pay["extra_metadata"]))
            if to_update:
                async with session_scope() as session:
                    for row_id, meta in to_update:
                        cur = await session.execute(
                            select(GraphRelationship.extra_metadata).where(
                                GraphRelationship.id == row_id
                            )
                        )
                        old_meta = cur.scalar_one_or_none() or {}
                        old_chunks = list(old_meta.get("supporting_chunks") or [])
                        new_chunks = [
                            c for c in (meta.get("supporting_chunks") or [])
                            if c not in old_chunks
                        ]
                        if not new_chunks:
                            continue
                        await session.execute(
                            sql_text("""
                            UPDATE graphrag.graph_relationships
                               SET extra_metadata = :meta
                             WHERE id = :id
                            """),
                            {
                                "id": row_id,
                                "meta": json.dumps({
                                    **old_meta,
                                    "support_count": (
                                        int(old_meta.get("support_count") or
                                            len(old_chunks))
                                        + int(meta.get("support_count") or 0)
                                    ),
                                    "supporting_chunks": (
                                        old_chunks + new_chunks
                                    )[:_MAX_SUPPORTING_CHUNKS],
                                }),
                            },
                        )
                        merged += 1
            if merged:
                print(
                    f"[extract-entities] merged new supporting chunk(s) into "
                    f"{merged} existing entity->entity edge(s)"
                )
            rel_payloads = fresh
    summary.entity_relationship_edges = len(rel_payloads)

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
        for i in range(0, len(rel_payloads), EDGE_BATCH):
            await session.execute(
                pg_insert(GraphRelationship).values(rel_payloads[i : i + EDGE_BATCH])
            )

    # Entity-side accounting. Previously every one of these was a silent
    # `continue`, so a corpus could lose entities steadily with nothing in the
    # output to show for it.
    summary.entity_drops = dict(ent_drops)
    summary.abstained_samples = abstained
    _ent_dropped = sum(ent_drops.values())
    if _ent_dropped:
        print(
            f"[extract-entities] entity drops: {_ent_dropped} "
            f"(off_menu_iri={ent_drops['off_menu_iri']}, "
            f"no_name={ent_drops['no_name']}, "
            f"abstained={ent_drops['abstained']}, "
            f"reviewer_removed={ent_drops['reviewer_removed']}, "
            f"no_candidates={ent_drops['no_candidates']})"
        )
    if ent_drops["abstained"]:
        _names = ", ".join(
            repr(a["canonical_name"]) for a in abstained[:5]
        )
        print(
            f"[extract-entities] {ent_drops['abstained']} entity mention(s) had "
            f"no fitting class in the ontology and were NOT minted "
            f"(e.g. {_names}). A high count here means the ontology is missing "
            f"a branch -- consider a prune-expand run over this corpus."
        )
    if validate_entities:
        summary.entity_validation = dict(val_stats)
        print(
            f"[extract-entities] entity review ({validation_rounds} round(s) max): "
            f"{val_stats['clean']} chunk(s) clean, "
            f"{val_stats['revised']} re-extracted "
            f"(reclassified={val_stats['reclassified']}, "
            f"removed={val_stats['removed']}, added={val_stats['added']}); "
            f"{val_stats['empty_reextract_rejected']} empty re-extract(s) "
            f"rejected; {val_stats['validator_failed']} validator call(s) "
            f"failed (kept the unreviewed result)"
        )

    if extract_relationships:
        _dropped = sum(rel_drops.values())
        print(
            f"[extract-entities] entity->entity relationships: "
            f"{len(rel_payloads)} written"
            + (f", {_dropped} dropped "
               f"(unresolved={rel_drops['unresolved']}, "
               f"bad_predicate={rel_drops['bad_predicate']}, "
               f"domain_range={rel_drops['domain_range']}, "
               f"no_evidence={rel_drops['no_evidence']}, "
               f"one_sided_evidence={rel_drops['one_sided_evidence']}, "
               f"unsupported={rel_drops['unsupported']}, "
               f"reversed={rel_drops['reversed']})" if _dropped else "")
            + (f", {rel_repairs['rescued']} rescued by re-homing to a better "
               f"predicate" if rel_repairs["rescued"] else "")
            + (f", {rel_repairs['direction_swapped']} direction(s) repaired"
               if rel_repairs["direction_swapped"] else "")
        )
        if not rel_payloads and summary.chunks_scanned:
            print(
                "[extract-entities] NOTE: no relationships written. Either the "
                "chunks assert none, or the ontology declares no object "
                "property whose domain AND range both match the classes in "
                "this corpus -- check with: SELECT count(*) FROM "
                "graphrag.ontology_object_properties;"
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

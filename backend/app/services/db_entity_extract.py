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
import collections
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Mapping
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
    GRAPHRAG_RELATED_TO,
    RDF_TYPE,
    VIAO_ASSERTS_ABOUT,
)
from backend.app.services.prompts import ORPHAN_REASONS, PROMPTS

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


def _confidence_of(e: dict[str, Any]) -> float | None:
    try:
        return float(e["confidence"]) if e.get("confidence") is not None else None
    except (TypeError, ValueError):
        return None


def _normalise_type_label(text: str) -> str:
    """Fold a type name to a comparison key: lowercase, alphanumerics only.

    So the model's "coffee maker" reaches the class `CoffeeMaker`, and
    "Credit Scoring Model" reaches `CreditScoringModel`. Deliberately NOT
    fuzzy -- this resolves an abstention into a real class, and a near-miss
    here recreates exactly the wrong-class problem the abstain path exists to
    prevent.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _resolve_proposed_type(
    proposed: str, type_index: Mapping[str, str] | None
) -> str | None:
    """An abstention recovered when the model's own `proposed_type` names a
    class that exists but never reached this chunk's menu.

    MEASURED on 20 news chunks: of 29 distinct `proposed_type` values the
    model asked for, 15 (51%) ALREADY EXISTED in the ontology -- CoffeeMaker,
    VacuumCleaner, MeshRouter, VideoGame, KitchenAppliance among them, and the
    most frequently requested ones at that. They simply did not rank into the
    chunk's top-K candidates, so the model could not pick what it was never
    shown and abstained, correctly, on its own terms.

    That is a menu-coverage failure, not an ontology gap, and no amount of
    re-running prune-expand fixes it -- the class is already there.
    """
    if not proposed or not type_index:
        return None
    return type_index.get(_normalise_type_label(proposed))


def _build_menu_index(
    cand_iris: set[str], cand_labels: Mapping[str, str] | None
) -> dict[str, str]:
    """Lowercase key -> IRI, for recovering a class named by label.

    Keys are each candidate's IRI local-name and, when the caller supplies
    them, its label. A key claimed by two DIFFERENT classes is dropped rather
    than resolved arbitrarily: `foaf#Person` and `org#Person` share a local
    name, and picking between them by set-iteration order would make
    extraction non-deterministic across runs.
    """
    idx: dict[str, str | None] = {}

    def offer(key: str, iri: str) -> None:
        key = key.strip().lower()
        if not key:
            return
        if key in idx and idx[key] != iri:
            idx[key] = None          # ambiguous -- never resolve it
        else:
            idx.setdefault(key, iri)

    for iri in cand_iris:
        offer(iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1], iri)
    for label, iri in (cand_labels or {}).items():
        if iri in cand_iris:
            offer(label, iri)
    return {k: v for k, v in idx.items() if v is not None}


def _resolve_menu_iri(raw: str, menu_index: dict[str, str]) -> str | None:
    """Recover an off-menu `class_iri` that actually names a menu class.

    MEASURED on the pharma corpus: of 11 off-menu values, 10 were the class
    LABEL emitted into the IRI field -- "brandnamedrug", "lipaseinhibitor",
    "semaglutide" -- each of them a class that was on the menu. The model
    chose correctly and serialised wrongly, and the entity was dropped for it
    (Ozempic, Wegovy and Orlistat among them). Only 1 of 11 was a genuine
    invention.

    This is why the rate differed so sharply between corpora: that ontology's
    labels are lowercase single tokens that look like IRI fragments, while a
    CamelCase taxonomy ("ContainerShip") does not invite the same slip.
    """
    hit = menu_index.get(raw.strip().lower())
    if hit:
        return hit
    tail = raw.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return menu_index.get(tail.strip().lower())


def _predicate_menu_index(preds: list[dict[str, str]]) -> dict[str, str]:
    """`_build_menu_index` for a predicate menu, so a mis-serialised
    `predicate_iri` resolves to the predicate that was actually offered.

    MEASURED on multihop-rag-subset: the merged ontology stores predicate IRIs
    lowercased (`merged#employedby`) under camelCase labels (`employedBy`).
    gpt-4.1-mini "corrected" the IRI to match the label in 127 of 590
    proposals -- `merged#collaboratesWith`, even `foaf/0.1/employedBy` -- and
    every one was dropped as bad_predicate. Resolution only ever lands on a
    menu predicate, and a key two menu predicates share is not resolved.
    """
    return _build_menu_index(
        {p["iri"] for p in preds},
        {p["label"]: p["iri"] for p in preds if p.get("label")},
    )


def _filter_entities(
    raw_entities: Any,
    cand_iris: set[str],
    ent_drops: dict[str, int],
    abstained: list[dict[str, Any]],
    abstain_cap: int = 200,
    cand_labels: Mapping[str, str] | None = None,
    type_index: Mapping[str, str] | None = None,
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
    menu_index = _build_menu_index(cand_iris, cand_labels)
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
            # Before recording a loss: the model names the type it wanted, and
            # half the time that class EXISTS -- it just never reached this
            # chunk's top-K menu. Recover those rather than discarding a real
            # entity over a ranking miss.
            proposed = (e.get("proposed_type") or "").strip()
            recovered_type = _resolve_proposed_type(proposed, type_index)
            if recovered_type is not None:
                ent_drops["recovered_proposed_type"] = (
                    ent_drops.get("recovered_proposed_type", 0) + 1
                )
                kept.append({
                    "canonical_name": name,
                    "short_name": short,
                    "class_iri": recovered_type,
                    "confidence": _confidence_of(e),
                })
                continue
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
        if class_iri and class_iri not in cand_iris:
            # Before counting this a loss, see whether the value NAMES a class
            # that is on the menu -- the model routinely writes the label into
            # the IRI field. Recovered separately from `off_menu_iri` so the
            # slip stays visible instead of being papered over.
            recovered = _resolve_menu_iri(class_iri, menu_index)
            if recovered is not None:
                class_iri = recovered
                ent_drops["recovered_label_iri"] = (
                    ent_drops.get("recovered_label_iri", 0) + 1
                )
        if not class_iri or class_iri not in cand_iris:
            ent_drops["off_menu_iri"] += 1
            continue
        conf = _confidence_of(e)
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
    # Edges given a vector by the post-insert embedding pass (0008).
    relationship_embeddings: int = 0
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
    concept_extraction: dict[str, int] = field(default_factory=dict)
    # Names the model declined to type because no candidate class denoted
    # their KIND. A rising count here is the honest signal that the ontology
    # is missing a branch -- act on it with a prune-expand run.
    abstained_samples: list[dict[str, Any]] = field(default_factory=list)
    menu_classes_withheld: int = 0
    # Name collapse audit: every spelling merged into another entity's name
    # ({"from": "Alameda Research", "to": "Alameda Research LLC"}), and how many
    # relationship endpoints were rewritten to follow. Before the endpoints
    # were rewritten, a relationship naming a merged spelling could not find
    # its entity and was dropped as "unresolved" -- 90 of 668 claims on
    # multihop-rag-subset, concentrated on FTX, Alameda Research, Epic Games.
    renamed_entities: list[dict[str, str]] = field(default_factory=list)
    relationship_endpoints_renamed: int = 0
    # Per-step relationship accounting (pass1 / gap / orphan_check): claims
    # kept, duplicates discarded, orphans flagged and why, and how many of each
    # step's claims the verifier kept. See `_relationships_for_chunk`.
    relationship_pass_stats: dict[str, int] = field(default_factory=dict)
    orphan_flags: list[dict[str, str]] = field(default_factory=list)


def _normalize_name(name: str) -> str:
    """Lowercase + strip non-alphanumerics (except internal spaces)."""
    s = re.sub(r"[^\w\s]", "", name, flags=re.UNICODE).strip().lower()
    return re.sub(r"\s+", " ", s)


# Legal-form suffixes stripped when deciding whether two names denote the SAME
# entity. Deliberately a COMPARISON key only -- the stored name keeps its
# suffix, because "A.P. Moller-Maersk A/S" is the right thing to display.
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:inc|corp|corporation|co|ltd|limited|llc|llp|lp|nv|n\s*v|"
    r"gmbh|ag|sa|s\s*a|as|a\s*s|spa|plc|pte|bhd|pty|oyj|ab|asa|se|kk|"
    r"holdings?|group|company)\b",
    re.IGNORECASE,
)


def _dedup_key(name: str) -> str:
    """Comparison key for "is this the same entity?".

    `_normalize_name` alone leaves legal forms attached, so `Hapag-Lloyd` and
    `Hapag-Lloyd AG` normalise to different strings and the pg_trgm gate
    (>= 0.85) scores them 0.79 -- just under. Measured on a 3-document news
    corpus that produced 7 near-duplicate node pairs: CMA CGM / CMA CGM SA,
    Hapag-Lloyd / Hapag-Lloyd AG, Kovatchev / Stephane Kovatchev.

    Duplicate nodes are not cosmetic: each carries its own edges, so a
    multi-hop traversal that arrives at one cannot see what is hung off the
    other -- the same failure that motivated name-level identity below.
    """
    base = _normalize_name(name)
    stripped = _LEGAL_SUFFIX_RE.sub(" ", base)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    # Never collapse a name to nothing: "Group" and "Co" are real short names.
    return stripped or base


def _person_surname_merges(
    names_by_class: dict[str, set[str]],
    person_class_iris: set[str],
) -> dict[str, str]:
    """Merge `Amosi` into `Guy Amosi` -- but never `Utah` into `Utah Utes`.

    Scoped to PERSON-typed entities on purpose. The safe signal is that a
    person referred to by surname is a strict token SUFFIX of their full name
    (`Amosi` of `Guy Amosi`, `McMillan` of `Tetairoa McMillan`, `Ben Zvi
    Klein` of `Or Ben Zvi Klein`), whereas the short forms that must NOT merge
    are prefixes -- `Utah` of `Utah Utes`, `Arizona` of `Arizona Wildcats`.

    Even so this stays Person-only, because the suffix rule alone would merge
    `Times` into `New York Times`, and those are different publications.
    """
    out: dict[str, str] = {}
    for cls, names in names_by_class.items():
        if cls not in person_class_iris:
            continue
        ordered = sorted(names, key=lambda n: (-len(n.split()), n))
        for short in names:
            st = short.split()
            for full in ordered:
                ft = full.split()
                if len(ft) > len(st) and ft[-len(st):] == st:
                    out[short] = full
                    break
    return out


def _word_order_merges(names_by_class: dict[str, set[str]]) -> dict[str, str]:
    """`Park Naimi` and `Naimi Park` are one thing written two ways.

    Uses exact token-MULTISET equality within a class, which is why it cannot
    confuse `Arizona State University` with `University of Arizona`: those
    differ by a token ("state" vs "of"), so their multisets differ.
    """
    out: dict[str, str] = {}
    for names in names_by_class.values():
        buckets: dict[tuple[str, ...], list[str]] = {}
        for n in names:
            buckets.setdefault(tuple(sorted(n.split())), []).append(n)
        for variants in buckets.values():
            if len(variants) < 2:
                continue
            target = sorted(variants)[0]
            for v in variants:
                if v != target:
                    out[v] = target
    return out


def _resolve_canonical_forms(
    results: list[Any],
    person_class_iris: set[str] | None = None,
) -> tuple[dict[str, str], int]:
    """Map every extracted name to ONE canonical form for the run.

    Two sources, both grounded in what the model actually saw:

    1. The extractor's own `canonical_name` / `short_name` pairing. When it
       reports canonical "A.P. Moller-Maersk A/S" with short "Maersk" it is
       asserting, from that chunk, that the two denote one entity. That pair
       was previously discarded for identity purposes.
    2. The legal-suffix-stripped key, which merges `CMA CGM` with
       `CMA CGM SA`.

    Returns (normalized_variant -> canonical_normalized, n_merged). The
    surviving form is the LONGEST variant seen, matching the extract prompt's
    instruction that canonical_name be "the entity's most complete proper
    name as commonly written".
    """
    groups: dict[str, set[str]] = {}
    longest: dict[str, str] = {}

    def _note(key: str, norm: str) -> None:
        if not norm:
            return
        groups.setdefault(key, set()).add(norm)
        if len(norm) > len(longest.get(key, "")):
            longest[key] = norm

    for tup in results or []:
        if tup is None:
            continue
        for e in (tup[3] or []):
            canon = _normalize_name(e.get("canonical_name") or "")
            short = _normalize_name(e.get("short_name") or "")
            if not canon:
                continue
            key = _dedup_key(canon)
            _note(key, canon)
            # The model's own short form joins the canonical form's group,
            # even when its stripped key differs ("maersk" vs "a p moller
            # maersk a s").
            #
            # EXCEPT when the short form is a token PREFIX of the canonical
            # and the entity is not a person. "Utah" reported as the short
            # form of "Utah Utes", or "Arizona" of "Arizona Wildcats", would
            # otherwise collapse a state into a sports team. A person's short
            # form is a suffix (their surname), so people are unaffected --
            # and a genuine abbreviation ("ASU", "Maersk") is not a prefix
            # either, so those still merge.
            if short and short != canon:
                ct, st_ = canon.split(), short.split()
                is_prefix = len(st_) < len(ct) and ct[:len(st_)] == st_
                if not is_prefix or (e.get("class_iri") or "") in (
                    person_class_iris or set()
                ):
                    _note(key, short)
                    groups.setdefault(_dedup_key(short), set()).add(short)

    alias: dict[str, str] = {}
    for key, variants in groups.items():
        target = longest.get(key) or next(iter(variants))
        for v in variants:
            if v != target:
                alias[v] = target

    # Two narrower rules that need the entity's CLASS, so they run separately.
    names_by_class: dict[str, set[str]] = {}
    for tup in results or []:
        if tup is None:
            continue
        for e in (tup[3] or []):
            nrm = _normalize_name(e.get("canonical_name") or "")
            cls = e.get("class_iri") or ""
            if nrm and cls:
                names_by_class.setdefault(cls, set()).add(nrm)
    for extra in (
        _person_surname_merges(names_by_class, person_class_iris or set()),
        _word_order_merges(names_by_class),
    ):
        for k, v in extra.items():
            if k != v and k not in alias:
                alias[k] = v

    # Collapse chains (a -> b -> c becomes a -> c) so every variant lands on
    # one final spelling rather than an intermediate one.
    for k in list(alias):
        seen = {k}
        tgt = alias[k]
        while tgt in alias and tgt not in seen:
            seen.add(tgt)
            tgt = alias[tgt]
        alias[k] = tgt
    alias = {k: v for k, v in alias.items() if k != v}
    return alias, len(alias)


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


# Longest evidence quote accepted, and stored. The relationship prompt lets a
# quote reach back up to 3 sentences to include the name a pronoun refers to
# ("Alameda is a hedge fund. Bankman-Fried founded it."); the ceiling makes
# that cap real in code, since a long quote can otherwise bridge unrelated
# sentences that happen to name both ends. The old 400-char storage cut would
# also have dropped the naming sentence from citations.
_MAX_EVIDENCE_CHARS = 700


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
    for form in (ent.get("canonical_name") or "", ent.get("short_name") or "",
                 *(ent.get("aliases") or ())):
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

# Predicates shown per chunk, AFTER ranking by fit (see `_rank_predicates`).
# The pool used to be cut at 40 in ALPHABETICAL order: on multihop-rag-subset
# 84 of 89 chunks had a larger pool (median 102), the menu typically ended at
# "h", and only 11 of 516 proposals used a predicate past it -- memberOf,
# ownedBy, partOf, worksFor were effectively never offered.
_MAX_CANDIDATE_PREDICATES = 60
# Upper bound on the unranked pool fetched from the DB before ranking.
_CANDIDATE_POOL_LIMIT = 2000
# Relationships the model may return per chunk: one per entity in the chunk,
# within these bounds. A flat 10 could link at most 20 of a chunk's entities,
# while summary chunks hold a median of 19 and up to 80 (full team lineups,
# product round-ups) -- so rosters were cut off by construction. Half the
# entity count still left a connected roster short; the ceiling exists only to
# keep one response inside relationship_extract's max_tokens.
_MIN_RELATIONSHIPS_PER_CHUNK = 10
# Orphans handed to one `relationship_orphan_check` call. See the batching
# comment at the call site for the measurement that set this.
_DEFAULT_ORPHAN_BATCH = 8
_MAX_RELATIONSHIPS_PER_CHUNK = 100
# Longest free-text `relation` phrase accepted on a graphrag#relatedTo edge.
_MAX_RELATION_WORDS = 8
# Rescue menu is wider (either-end match), so it needs a higher ceiling; the
# measured either-end pool is a median of 52 per chunk.
_MAX_RESCUE_PREDICATES = 60
# Supporting chunk ids kept per edge (the COUNT is always exact).
_MAX_SUPPORTING_CHUNKS = 20

_DESCENDANT_SQL = sql_text("""
WITH RECURSIVE down(origin, id) AS (
    SELECT oc.iri, oc.id FROM graphrag.ontology_classes oc
     WHERE oc.iri = ANY(CAST(:iris AS text[]))
  UNION
    SELECT down.origin,
           CASE WHEN gr.predicate_label = 'rdfs:subClassOf' THEN gr.source_node_id
                WHEN gr.source_node_id = down.id THEN gr.target_node_id
                ELSE gr.source_node_id END
      FROM down JOIN graphrag.graph_relationships gr
        ON gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND (
             (gr.predicate_label = 'rdfs:subClassOf'
              AND gr.target_node_id = down.id)
          OR (gr.predicate_label = 'owl:equivalentClass'
              AND (gr.source_node_id = down.id OR gr.target_node_id = down.id))
       )
)
SELECT DISTINCT down.origin, oc.iri
  FROM down JOIN graphrag.ontology_classes oc ON oc.id = down.id
""")


_CONCEPT_CLOSURE_SQL = sql_text("""
WITH RECURSIVE down(id) AS (
    SELECT oc.id FROM graphrag.ontology_classes oc
     WHERE lower(oc.label) = ANY(CAST(:labels AS text[]))
  UNION
    SELECT gr.source_node_id
      FROM down
      JOIN graphrag.graph_relationships gr
        ON gr.target_node_id = down.id
       AND gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND gr.predicate_label = 'rdfs:subClassOf'
)
SELECT oc.iri FROM down JOIN graphrag.ontology_classes oc ON oc.id = down.id
""")


# Both closures cross owl:equivalentClass in EITHER direction, in addition to
# walking rdfs:subClassOf. An equivalent class IS the same class, so it is an
# ancestor and a descendant at once. Without this, `org:Organization` and
# `foaf:Organization` -- declared equivalent in the W3C ORG ontology -- are
# unrelated siblings under `foaf:Agent`, and every predicate declared on one
# rejects entities typed with the other: `Caroline Ellison --controls-->
# Alameda Research LLC` failed the type check on a news build although
# `controls` is declared Person -> Organization. UNION (not UNION ALL) stops the
# symmetric edge from looping.
_ANCESTOR_SQL = sql_text("""
WITH RECURSIVE up(origin, id) AS (
    SELECT oc.iri, oc.id FROM graphrag.ontology_classes oc
     WHERE oc.iri = ANY(CAST(:iris AS text[]))
  UNION
    SELECT up.origin,
           CASE WHEN gr.source_node_id = up.id THEN gr.target_node_id
                ELSE gr.source_node_id END
      FROM up JOIN graphrag.graph_relationships gr
        ON gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND (
             (gr.predicate_label = 'rdfs:subClassOf'
              AND gr.source_node_id = up.id)
          OR (gr.predicate_label = 'owl:equivalentClass'
              AND (gr.source_node_id = up.id OR gr.target_node_id = up.id))
       )
)
SELECT DISTINCT up.origin, oc.iri
  FROM up JOIN graphrag.ontology_classes oc ON oc.id = up.id
""")

# Returns EVERY in-scope declared domain and range, not the first one: 173 of
# 434 predicates on the news build declare several (`created` has domain
# [Organization, Agent]), and checking only the first rejected
# `Sam Bankman-Fried --created--> FTT token`.
_CANDIDATE_PREDICATE_SQL = sql_text("""
SELECT op.iri, op.label,
       ARRAY(SELECT DISTINCT d->>'iri'
               FROM jsonb_array_elements(op.extra_metadata->'domain') d
              WHERE d->>'iri' = ANY(CAST(:iris AS text[]))) AS dom_iris,
       ARRAY(SELECT DISTINCT r->>'iri'
               FROM jsonb_array_elements(op.extra_metadata->'range') r
              WHERE r->>'iri' = ANY(CAST(:iris AS text[]))) AS rng_iris
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
    wide_index = _predicate_menu_index(wide_preds)
    rescued: list[dict[str, Any]] = []
    for rel in (parsed.get("relationships") or []):
        if not isinstance(rel, dict):
            continue
        # No type hint available here (the wide menu's constraints are not
        # threaded in), so this resolves on name alone -- exact, else a single
        # unambiguous word-boundary prefix candidate. It never guesses.
        s_ent = _resolve_entity_ref(rel.get("subject") or "", ent_by_norm)
        o_ent = _resolve_entity_ref(rel.get("object") or "", ent_by_norm)
        pred = (rel.get("predicate_iri") or "").strip()
        if pred and pred not in known_iris:
            recovered = _resolve_menu_iri(pred, wide_index)
            if recovered in known_iris:
                pred = recovered
                rel_repairs["predicate_recovered"] += 1
        if s_ent is None or o_ent is None or pred not in known_iris:
            continue
        if s_ent["canonical_name"] == o_ent["canonical_name"]:
            continue
        ev = " ".join((rel.get("evidence") or "").split())
        # Same evidence bar as the strict path -- widening WHICH predicate may
        # be used never widens what counts as proof.
        if len(ev) < 12 or len(ev) > _MAX_EVIDENCE_CHARS or ev.lower() not in hay:
            continue
        if not (_evidence_names(ev, s_ent) and _evidence_names(ev, o_ent)):
            continue
        rescued.append({
            "subject": s_ent["canonical_name"],
            "object": o_ent["canonical_name"],
            "predicate_iri": pred,
            "confidence": None,
            "evidence": ev[:_MAX_EVIDENCE_CHARS],
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
    # The declared domain/range travel with each claim so the auditor can see
    # what the predicate MEANS, not just its camelCase name. Measured: it
    # affirmed "Northern Powergrid --monitors--> Gas and Electricity Markets
    # Authority" against a quote reading "enforced BY the Authority", and
    # "BHE GT&S --hasMember--> FERC" against "rate-regulated BY the Federal
    # Energy...". Knowing hasMember runs <Organization> -> <Agent> makes a
    # regulator in the object slot visibly wrong.
    shape_of = {
        p["iri"]: (p.get("domain_label") or "", p.get("range_label") or "")
        for p in pred_list
    }
    claims = []
    for r in rels:
        dom_lbl, rng_lbl = shape_of.get(r["predicate_iri"], ("", ""))
        generic = r["predicate_iri"] == GRAPHRAG_RELATED_TO
        claims.append({
            "subject": r["subject"],
            "object": r["object"],
            # A relatedTo claim is judged on its phrase ("was born in"); the
            # bare word "relatedTo" would let almost any co-mention pass.
            "predicate_label": (r.get("relation") or "related to") if generic
            else label_of.get(r["predicate_iri"], r["predicate_iri"]),
            "domain_label": "" if generic else dom_lbl,
            "range_label": "" if generic else rng_lbl,
            "evidence": r["evidence"],
        })

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
        if verdict == "reversed" and rel["predicate_iri"] == GRAPHRAG_RELATED_TO:
            # Flipping the ends would leave the phrase backwards ("Failsworth
            # was born in Ratcliffe"), and there is no reliable way to invert
            # free text -- so drop it rather than write a garbled edge.
            rel_drops["reversed"] += 1
            continue
        if verdict == "reversed":
            # The assertion is real; only the direction was mis-read. Keep it
            # ONLY if the flipped pair still type-checks -- a swap that
            # violates domain/range means the model is confused about more
            # than direction.
            s_ent = ent_by_norm.get(_normalize_name(rel["object"]))
            o_ent = ent_by_norm.get(_normalize_name(rel["subject"]))
            menu_pred = next((p for p in pred_list
                              if p["iri"] == rel["predicate_iri"]), None)
            if s_ent and o_ent and menu_pred:
                s_cls, o_cls = s_ent["class_iri"], o_ent["class_iri"]
                if _types_fit(menu_pred, s_cls, o_cls, pred_ancestors):
                    out.append({**rel,
                                "subject": s_ent["canonical_name"],
                                "object": o_ent["canonical_name"]})
                    rel_repairs["direction_swapped"] += 1
                    continue
            rel_drops["reversed"] += 1
            continue
        rel_drops["unsupported"] += 1
    return out


def _resolve_entity_ref(
    raw: str,
    by_norm: dict[str, dict[str, Any]],
    *,
    expect_class: str | None = None,
    ancestors: dict[str, set[str]] | None = None,
) -> dict[str, Any] | None:
    """Match a name the model returned back to an extracted entity.

    The relationship pass is handed each entity's `canonical_name` and asked to
    echo it. It does not, reliably: measured on a utility 10-K it was given
    "MidAmerican Energy Company" and answered "MidAmerican Energy", and the
    exact-dict lookup dropped every claim about it as `unresolved` -- 13 of 150
    drops on that run, 12 of 65 on a pharma run, including relationships that
    were correct.

    Exact (normalised) match first. On a miss, fall back to WORD-BOUNDARY
    prefix candidates in either direction -- but only commit when exactly one
    survives, because guessing here writes a wrong edge. "MidAmerican Energy"
    prefix-matches three entities on that corpus (`...Company`, `...Services`,
    `...wind facilities repowering`), so name alone is not enough.

    `expect_class` is the predicate's declared domain (for a subject) or range
    (for an object). Using it to filter candidates is what makes the ambiguous
    case resolvable without guessing: `hasMember` declares domain Organization,
    which picks `MidAmerican Energy Company` over the `Repowering` one. When
    the filter leaves more than one candidate, the claim is still dropped.
    """
    n = _normalize_name(raw or "")
    if not n:
        return None
    hit = by_norm.get(n)
    if hit is not None:
        return hit
    seen_ids: set[int] = set()
    cands: list[dict[str, Any]] = []
    for key, ent in by_norm.items():
        if key == n or not key:
            continue
        # Word-boundary containment only: "Craig" must not match "Craigslist".
        if key.startswith(n + " ") or n.startswith(key + " "):
            if id(ent) not in seen_ids:
                seen_ids.add(id(ent))
                cands.append(ent)
    if len(cands) > 1 and expect_class and ancestors is not None:
        typed = [
            c for c in cands
            if expect_class in ancestors.get(c["class_iri"], {c["class_iri"]})
        ]
        if typed:
            cands = typed
    return cands[0] if len(cands) == 1 else None


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

    # DESCENDANTS TOO, and this is the difference between a usable menu and a
    # starved one. Walking only UPWARD means a generically-typed entity can
    # never use a specifically-declared predicate: an entity typed
    # `Organization` fails every predicate whose domain is `pipelineoperator`,
    # even though pipeline operators ARE organizations. Measured on the
    # finance build, that left a MEDIAN OF 8 PREDICATES OUT OF 563 per chunk
    # -- so the model, holding 8-15 entities, forced real relationships onto
    # whatever it had been shown. One chunk put 10 proposals through
    # `hasMember`, every one junk.
    #
    # This is also what OWL actually means. `rdfs:domain` is an INFERENCE rule,
    # not a constraint: asserting `X hasPipeline Y` entails that X is a
    # pipeline operator, it does not require X to have been typed one first.
    # The old reading treated it as a precondition.
    #
    # It does NOT open the gate to unrelated pairs. The two classes must still
    # share an IS-A line, so `PolicyConcept` still fails a predicate whose
    # range is `Agent` -- which is what correctly rejected
    # `PacifiCorp --hasMember--> renewable resource`. And every claim still has
    # to quote text that names both ends and survive the direction auditor.
    # Descendants are attributed PER CLASS, exactly as ancestors are, so the
    # per-claim check keeps requiring domain/range to sit on the same IS-A line
    # as that entity's own class. A shared pool would let any entity satisfy
    # any other entity's subtree, which is a different and much looser rule.
    up_only = {c: set(a) for c, a in ancestors.items()}
    r = await session.execute(_DESCENDANT_SQL, {"iris": list(class_iris)})
    for origin, desc in r.all():
        ancestors.setdefault(origin, {origin}).add(desc)
    expanded = {a for anc in ancestors.values() for a in anc} | set(class_iris)

    r = await session.execute(
        _CANDIDATE_PREDICATE_SQL,
        {"iris": list(expanded), "limit": _CANDIDATE_POOL_LIMIT},
    )
    pool = [(iri, label, list(doms or []), list(rngs or []))
            for iri, label, doms, rngs in r.all()]
    # Depth of each declared domain/range (its ancestor count) -- a deeper
    # class is a more specific predicate. One query per chunk.
    typ_iris = {t for _, _, d, g in pool for t in (*d, *g)}
    depth: dict[str, int] = {}
    if typ_iris:
        r = await session.execute(_ANCESTOR_SQL, {"iris": list(typ_iris)})
        for origin, _anc in r.all():
            depth[origin] = depth.get(origin, 0) + 1
    ranked = _rank_predicates(pool, set(class_iris), up_only, ancestors, depth)

    out: list[dict[str, Any]] = []
    constraints: dict[str, tuple[str, str]] = {}
    for iri, label, doms, rngs, dom_iri, rng_iri in (
            ranked[:_MAX_CANDIDATE_PREDICATES]):
        constraints[iri] = (dom_iri, rng_iri)
        out.append({
            "iri": iri,
            "label": label or iri.rsplit("#", 1)[-1],
            "domain_label": dom_iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1],
            "range_label": rng_iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1],
            # Every in-scope declared type, for `_types_fit`. The labels above
            # show the best-fitting pair only.
            "domain_iris": doms,
            "range_iris": rngs,
        })
    # The reserved fallback is always on the menu (outside the cap), so a
    # stated relationship the ontology has no predicate for is recorded with
    # its phrase rather than discarded. The prompt lists it last-resort.
    constraints[GRAPHRAG_RELATED_TO] = ("", "")
    out.append({
        "iri": GRAPHRAG_RELATED_TO,
        "label": "relatedTo",
        "domain_label": "Thing",
        "range_label": "Thing",
        "domain_iris": [],
        "range_iris": [],
    })
    return out, constraints, ancestors


def _rank_predicates(
    pool: list[tuple[str, str | None, list[str], list[str]]],
    class_iris: set[str],
    up_only: dict[str, set[str]],
    lines: dict[str, set[str]],
    depth: dict[str, int],
) -> list[tuple[str, str | None, list[str], list[str], str, str]]:
    """Keep predicates that fit SOME ordered pair of this chunk's classes, and
    order them by how well they fit.

    A predicate is offered only if one class's IS-A line meets a declared
    domain and one class's line meets a declared range -- the pooled SQL alone
    also admits predicates whose domain fits entity A and range fits only A,
    which no pair can use. Ranking, best first:

      tier 0  both ends are the entity's own class
      tier 1  worst end is an ANCESTOR (`Organization` for a `PublicCompany`)
      tier 2  some end matches only a DESCENDANT (`pipelineoperator` for an
              entity typed `Organization`) -- valid inference, weakest fit

    then deeper (more specific) domain+range first, then label. Returns each
    predicate with the best-fitting (domain, range) pair appended.
    """
    def tier(t: str, cls: str) -> int:
        if t == cls:
            return 0
        return 1 if t in up_only.get(cls, {cls}) else 2

    ranked = []
    for iri, label, doms, rngs in pool:
        best: tuple[int, int, str, str] | None = None
        for s_cls in class_iris:
            s_line = lines.get(s_cls, {s_cls})
            ds = [d for d in doms if d in s_line]
            if not ds:
                continue
            for o_cls in class_iris:
                o_line = lines.get(o_cls, {o_cls})
                for g in (g for g in rngs if g in o_line):
                    for d in ds:
                        key = (max(tier(d, s_cls), tier(g, o_cls)),
                               -(depth.get(d, 1) + depth.get(g, 1)), d, g)
                        if best is None or key[:2] < best[:2]:
                            best = key
        if best is not None:
            ranked.append((best[0], best[1], (label or iri).lower(),
                           (iri, label, doms, rngs, best[2], best[3])))
    ranked.sort(key=lambda x: x[:3])
    return [x[3] for x in ranked]


def _apply_merged_names(
    results: list[Any],
    merged_name: Any,
) -> tuple[dict[str, set[str]], int]:
    """Rename entities to their merged spelling, and the relationship
    endpoints that name them, with ONE function so the two cannot diverge.

    `results` holds per-chunk tuples whose [3] is the kept entities and [4] the
    relationships; both are rewritten in place. `merged_name(own)` returns the
    new spelling or None. Returns ({new_name: {old spellings}}, number of
    relationship endpoints rewritten).
    """
    renamed_from: dict[str, set[str]] = {}
    endpoints = 0
    for tup in results:
        if tup is None:
            continue
        for e in (tup[3] or []):
            own = (e.get("canonical_name") or "").strip()
            new = merged_name(own)
            if new:
                e["canonical_name"] = new
                renamed_from.setdefault(new, set()).add(own)
        # A chunk's relationships name their ends by the spelling THAT chunk
        # used, copied before the merge. Left alone, the write step looks up a
        # name that no longer exists and drops the edge as "unresolved".
        for rel in (tup[4] or []):
            for side in ("subject", "object"):
                new = merged_name((rel.get(side) or "").strip())
                if new:
                    rel[side] = new
                    endpoints += 1
    return renamed_from, endpoints


def _relationship_cap(n_entities: int) -> int:
    """Relationships to ask for in a chunk holding `n_entities` entities."""
    return max(_MIN_RELATIONSHIPS_PER_CHUNK,
               min(_MAX_RELATIONSHIPS_PER_CHUNK, n_entities))


def _clean_relation(raw: Any) -> str | None:
    """The `relation` phrase of a relatedTo claim, or None if unusable.

    Required: an edge saying only "related to" is exactly the co-mention link
    the evidence rules exist to keep out, and it gives retrieval nothing to
    rank on or show.
    """
    phrase = " ".join(str(raw or "").split()).strip(" .;:,")
    if not phrase or phrase.lower() in {"related to", "relatedto", "related"}:
        return None
    if len(phrase.split()) > _MAX_RELATION_WORDS:
        return None
    return phrase


def _types_fit(
    pred: dict[str, Any] | None,
    s_cls: str,
    o_cls: str,
    ancestors: dict[str, set[str]],
) -> bool:
    """Does (s_cls, o_cls) satisfy ANY declared domain and ANY declared range
    of this menu predicate? Uses the same per-class IS-A lines that offered it.
    """
    if not pred:
        return False
    if pred.get("iri") == GRAPHRAG_RELATED_TO:
        return True       # untyped by design; the evidence checks still apply
    doms = pred.get("domain_iris") or []
    rngs = pred.get("range_iris") or []
    s_line = ancestors.get(s_cls, {s_cls})
    o_line = ancestors.get(o_cls, {o_cls})
    return any(d in s_line for d in doms) and any(g in o_line for g in rngs)


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


def _build_relationship_payloads(
    resolved: list[tuple[Any, Any, Any, Any, dict[str, Any]]],
    *,
    gv: int,
    rel_drops: dict[str, int],
) -> list[dict[str, Any]]:
    """Turn checked claims into edge rows: one edge per (subject, predicate,
    object) with every supporting chunk counted, direction contradictions
    reconciled, and relatedTo dropped where a specific edge links the pair.

    `resolved` is [(chunk_id, doc_id, subject_id, object_id, claim)], the ids
    already looked up. No DB access.
    """
    _pred_label: dict[str, str] = {}
    rel_by_sig: dict[tuple[Any, str, Any], dict[str, Any]] = {}
    for chunk_id, doc_id, sid, oid, rel in resolved:
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
                # How a graphrag#relatedTo edge relates its ends, in
                # the passage's words -- what retrieval shows.
                **({"relation": rel["relation"]}
                   if rel.get("relation") else {}),
                # Which relationship step found it: pass1, gap, orphan_check.
                **({"found_by": rel["found_by"]}
                   if rel.get("found_by") else {}),
            },
        })
        rel_by_sig[sig]["_chunks"] = [chunk_id]

    # RECONCILE CONTRADICTIONS ACROSS CHUNKS.
    #
    # Each chunk is judged on its own, so nothing stops two passages
    # asserting the same predicate in opposite directions. Measured on the
    # finance build: `BHE U.S. Transmission --hasSubOrganization--> MATL
    # LLP` and `MATL LLP --hasSubOrganization--> BHE U.S. Transmission`
    # were both written. One of them is necessarily false -- these
    # predicates are asymmetric -- and a graph asserting both is worse
    # than one asserting neither, because BFS will happily traverse the
    # wrong one.
    #
    # The tie-break is corroboration: `_chunks` already records how many
    # independent passages asserted each triple. More passages wins. On a
    # TIE both are dropped: with one passage each there is no ground to
    # prefer either, and inventing a preference is how a confident false
    # edge gets in.
    #
    # DIFFERENT predicates between the same pair are left alone. "A
    # regulates B" and "B is subject to A" are not contradictory, and the
    # observed case (`AUC --hasSubOrganization--> AltaLink` alongside
    # `AltaLink --monitors--> AUC`) is two wrong PREDICATES rather than a
    # direction conflict -- a problem for the menu, not for this pass.
    _seen_dirs: dict[tuple[Any, str, Any], tuple[Any, str, Any]] = {}
    _kill: set[tuple[Any, str, Any]] = set()
    for sig in rel_by_sig:
        sid, pred, oid = sig
        mirror = (oid, pred, sid)
        # relatedTo is not one relation but many ("was born in" one way,
        # "is the birthplace of" the other), so a mirror is no conflict.
        if pred == GRAPHRAG_RELATED_TO:
            continue
        if mirror in rel_by_sig:
            pair = (min(str(sid), str(oid)), pred, max(str(sid), str(oid)))
            if pair in _seen_dirs:
                continue
            _seen_dirs[pair] = sig
            n_here = len(rel_by_sig[sig].get("_chunks") or [])
            n_there = len(rel_by_sig[mirror].get("_chunks") or [])
            if n_here > n_there:
                _kill.add(mirror)
            elif n_there > n_here:
                _kill.add(sig)
            else:
                _kill.add(sig)
                _kill.add(mirror)
    for sig in _kill:
        rel_by_sig.pop(sig, None)
        rel_drops["contradictory_direction"] += 1

    # A pair the corpus links with a SPECIFIC predicate anywhere does not
    # also need the fallback: same BFS reach, and the specific edge says
    # more. Keeps relatedTo to what the ontology genuinely cannot express.
    _specific_pairs = {
        frozenset((str(sid), str(oid)))
        for sid, pred, oid in rel_by_sig if pred != GRAPHRAG_RELATED_TO
    }
    for sig in [g for g in rel_by_sig if g[1] == GRAPHRAG_RELATED_TO]:
        if frozenset((str(sig[0]), str(sig[2]))) in _specific_pairs:
            rel_by_sig.pop(sig)
            rel_drops["generic_superseded"] += 1

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
    return list(rel_by_sig.values())


async def _relationships_for_chunk(
    router: Any,
    txt: str,
    kept: list[dict[str, Any]],
    class_label_of: dict[str, str],
    chunk_iri: str,
    *,
    rel_drops: dict[str, int],
    rel_repairs: dict[str, int],
    rescue: bool = False,
    verify: bool = True,
    gap_pass: bool = False,
    orphan_check: bool = False,
    orphan_batch_size: int = _DEFAULT_ORPHAN_BATCH,
    pass_stats: dict[str, int] | None = None,
    orphan_flags: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract, check and verify the relationships ONE chunk states among its
    entities.

    Steps, each claim tagged with the step that found it (`found_by`):
      pass1         relationship_extract over the chunk
      gap           (gap_pass) the same call shown pass 1's accepted claims and
                    asked only for what it missed -- one pass finds a different
                    ~70% of a chunk's relationships on each run
      orphan_check  (orphan_check) entities still without a relationship are
                    listed; the model returns a stated relationship for each or
                    flags it with a reason (prompts.ORPHAN_REASONS)
      verify        relationship_verify over ALL of the above
    Every claim from every step passes the same gates. Duplicates of an
    earlier claim (either direction) are discarded and counted.

    Each entity is {canonical_name, short_name?, aliases?, class_iri}.
    Returns claims naming their ends by `canonical_name`.
    """
    stats = pass_stats if pass_stats is not None else {}
    rels: list[dict[str, Any]] = []
    if len(kept) < 2:
        return rels
    ent_class_iris = {e["class_iri"] for e in kept}
    async with session_scope() as session:
        pred_list, pred_constraints, pred_ancestors = (
            await _candidate_predicates(session, ent_class_iris)
        )
    if not pred_list:
        return rels
    ents_for_prompt = [
        {"canonical_name": e["canonical_name"],
         "class_label": class_label_of.get(e["class_iri"], "")}
        for e in kept
    ]
    cap = _relationship_cap(len(kept))
    try:
        r_sys, r_user = PROMPTS["relationship_extract"](
            txt, ents_for_prompt, pred_list, max_relationships=cap,
        )
        r_out = await router.chat(
            "relationship_extract", system=r_sys, user=r_user
        )
        r_parsed = _extract_json(r_out.text)
    except Exception as exc:
        print(f"[extract-entities] relationship call failed "
              f"on {chunk_iri}: {exc}")
        r_parsed = None
    if not isinstance(r_parsed, dict):
        return rels

    _by_norm = {_normalize_name(e["canonical_name"]): e for e in kept}
    for e in kept:
        for form in (e.get("short_name"), *(e.get("aliases") or ())):
            if form:
                _by_norm.setdefault(_normalize_name(form), e)
    hay = " ".join(txt.split()).lower()
    rejects: list[dict[str, Any]] = []
    _pred_label = {p["iri"]: (p.get("label") or p["iri"]) for p in pred_list}
    _pred_index = _predicate_menu_index(pred_list)
    _pred_by_iri = {p["iri"]: p for p in pred_list}

    def _bump(key: str, n: int = 1) -> None:
        stats[key] = stats.get(key, 0) + n

    def _gate(raw: Any, found_by: str) -> list[dict[str, Any]]:
        """The evidence and type gates, identical for every step."""
        out: list[dict[str, Any]] = []
        for rel in (raw or []):
            if not isinstance(rel, dict):
                continue
            # Predicate FIRST: its declared domain/range is what disambiguates
            # an inexact entity reference.
            pred = (rel.get("predicate_iri") or "").strip()
            if pred not in pred_constraints:
                recovered = _resolve_menu_iri(pred, _pred_index)
                if recovered not in pred_constraints:
                    rel_drops["bad_predicate"] += 1
                    continue
                pred = recovered
                rel_repairs["predicate_recovered"] += 1
            dom_iri, rng_iri = pred_constraints[pred]
            relation = None
            if pred == GRAPHRAG_RELATED_TO:
                relation = _clean_relation(rel.get("relation"))
                if relation is None:
                    rel_drops["no_relation_phrase"] += 1
                    continue
            s_ent = _resolve_entity_ref(
                rel.get("subject") or "", _by_norm,
                expect_class=dom_iri, ancestors=pred_ancestors,
            )
            o_ent = _resolve_entity_ref(
                rel.get("object") or "", _by_norm,
                expect_class=rng_iri, ancestors=pred_ancestors,
            )
            if s_ent is None or o_ent is None:
                rel_drops["unresolved"] += 1
                continue
            if s_ent["canonical_name"] == o_ent["canonical_name"]:
                rel_drops["self_loop"] += 1
                continue
            s_cls, o_cls = s_ent["class_iri"], o_ent["class_iri"]
            _type_ok = _types_fit(
                _pred_by_iri.get(pred), s_cls, o_cls, pred_ancestors)
            # The model must QUOTE the text that asserts the claim...
            ev = " ".join((rel.get("evidence") or "").split())
            if len(ev) < 12 or ev.lower() not in hay:
                rel_drops["no_evidence"] += 1
                continue
            if len(ev) > _MAX_EVIDENCE_CHARS:
                rel_drops["overlong_evidence"] += 1
                continue
            # ...and the quote must NAME BOTH ends.
            if not (_evidence_names(ev, s_ent) and _evidence_names(ev, o_ent)):
                rel_drops["one_sided_evidence"] += 1
                continue
            # Types fit the other way round: swap, and let the verifier
            # confirm the direction from the quote.
            if not _type_ok and _types_fit(
                    _pred_by_iri.get(pred), o_cls, s_cls, pred_ancestors):
                s_ent, o_ent = o_ent, s_ent
                rel_repairs["type_swapped"] += 1
                _type_ok = True
            if not _type_ok:
                rel_drops["domain_range"] += 1
                rejects.append({
                    "subject": s_ent["canonical_name"],
                    "object": o_ent["canonical_name"],
                    "predicate_label": _pred_label.get(pred, pred),
                    "evidence": ev[:_MAX_EVIDENCE_CHARS],
                })
                continue
            try:
                rconf = (float(rel["confidence"])
                         if rel.get("confidence") is not None else None)
            except (TypeError, ValueError):
                rconf = None
            out.append({
                "subject": s_ent["canonical_name"],
                "object": o_ent["canonical_name"],
                "predicate_iri": pred,
                "confidence": rconf,
                "evidence": ev[:_MAX_EVIDENCE_CHARS],
                "found_by": found_by,
                **({"relation": relation} if relation else {}),
            })
        return out

    def _add(new: list[dict[str, Any]], tag: str) -> None:
        have = set()
        for c in rels:
            have.add((c["subject"], c["predicate_iri"], c["object"]))
            have.add((c["object"], c["predicate_iri"], c["subject"]))
        for c in new:
            key = (c["subject"], c["predicate_iri"], c["object"])
            if key in have:
                _bump(f"{tag}_duplicates")
                continue
            have.add(key)
            have.add((c["object"], c["predicate_iri"], c["subject"]))
            rels.append(c)
            _bump(f"{tag}_kept")

    def _show(c: dict[str, Any]) -> str:
        label = (c.get("relation") if c["predicate_iri"] == GRAPHRAG_RELATED_TO
                 else _pred_label.get(c["predicate_iri"], c["predicate_iri"]))
        return f"{c['subject']} --{label}--> {c['object']}"

    _add(_gate(r_parsed.get("relationships"), "pass1"), "pass1")

    if gap_pass:
        try:
            g_sys, g_user = PROMPTS["relationship_extract"](
                txt, ents_for_prompt, pred_list, max_relationships=cap,
                already_found=[_show(c) for c in rels],
            )
            g_out = await router.chat(
                "relationship_extract", system=g_sys, user=g_user
            )
            g_parsed = _extract_json(g_out.text)
        except Exception as exc:
            print(f"[extract-entities] gap relationship pass failed "
                  f"on {chunk_iri}: {exc}")
            g_parsed = None
        _bump("gap_calls")
        if isinstance(g_parsed, dict):
            _add(_gate(g_parsed.get("relationships"), "gap"), "gap")

    if orphan_check:
        linked = {c["subject"] for c in rels} | {c["object"] for c in rels}
        orphans = [e["canonical_name"] for e in kept
                   if e["canonical_name"] not in linked]
        if orphans:
            _bump("orphans_flagged", len(orphans))
            # BATCHED. One call covering every orphan in the chunk loses
            # recall badly as the list grows: measured on the
            # websearch-geo-time corpus, the same chunk, model and prompt
            # rescued Beijing when asked about it alone (conf 0.95) and
            # labelled it `only_listed` when asked about 31 orphans at once,
            # proposing just 2 rescues for the whole batch. Labelling is the
            # cheap path once attention is divided. Smaller batches trade
            # calls for recall -- cost on this step scales with
            # ceil(len(orphans)/batch), everything else is unchanged.
            for _i in range(0, len(orphans), orphan_batch_size):
                _batch = orphans[_i : _i + orphan_batch_size]
                try:
                    o_sys, o_user = PROMPTS["relationship_orphan_check"](
                        txt, ents_for_prompt, pred_list,
                        orphans=_batch, found=[_show(c) for c in rels],
                        max_relationships=max(_MIN_RELATIONSHIPS_PER_CHUNK,
                                              len(_batch)),
                    )
                    o_out = await router.chat(
                        "relationship_orphan_check", system=o_sys, user=o_user
                    )
                    o_parsed = _extract_json(o_out.text)
                except Exception as exc:
                    print(f"[extract-entities] orphan check failed "
                          f"on {chunk_iri}: {exc}")
                    o_parsed = None
                _bump("orphan_calls")
                if isinstance(o_parsed, dict):
                    _add(_gate(o_parsed.get("relationships"), "orphan_check"),
                         "orphan")
                    valid = set(ORPHAN_REASONS)
                    _verdicted: set[str] = set()
                    for f in (o_parsed.get("no_relationship") or []):
                        if not isinstance(f, dict):
                            continue
                        _name = str(f.get("entity") or "")
                        _verdicted.add(_name)
                        reason = str(f.get("reason") or "").strip().lower()
                        reason = reason if reason in valid else "unspecified"
                        _bump(f"orphan_reason_{reason}")
                        if orphan_flags is not None and len(orphan_flags) < 300:
                            orphan_flags.append({
                                "entity": _name,
                                "reason": reason, "chunk": chunk_iri})
                    # An orphan that came back with neither a relationship nor
                    # a reason was silently unaccounted before: the reasons
                    # summed to 863 against 1066 still-unlinked entities, and
                    # a 31-orphan call returned 2 + 28 = 30 verdicts for 31.
                    _rel_names = ({c["subject"] for c in rels}
                                  | {c["object"] for c in rels})
                    for _o in _batch:
                        if _o not in _verdicted and _o not in _rel_names:
                            _bump("orphan_reason_no_verdict")
            linked = {c["subject"] for c in rels} | {c["object"] for c in rels}
            _bump("orphans_still_unlinked",
                  sum(1 for o in orphans if o not in linked))

    if rejects and rescue:
        async with session_scope() as session:
            wide = await _wide_predicates(session, ent_class_iris, pred_ancestors)
        _known = {p["iri"] for p in wide}
        rescued = await _rescue_relationships(
            router, txt, rejects, wide, _known, _by_norm, rel_repairs, chunk_iri,
        )
        for c in rescued:
            c.setdefault("found_by", "rescue")
        rels.extend(rescued)

    if rels and verify:
        rels = await _verify_relationships(
            router, txt, rels, pred_list,
            rel_drops, rel_repairs, pred_constraints,
            pred_ancestors, _by_norm, chunk_iri,
        )
    for c in rels:
        _bump(f"verified_{c.get('found_by') or 'pass1'}")
    return rels


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
    relationship_gap_pass: bool = True,
    relationship_orphan_check: bool = True,
    orphan_batch_size: int = _DEFAULT_ORPHAN_BATCH,
    entity_identity: str = "name",
    validate_entities: bool = False,
    validation_rounds: int = 2,
    concept_pass: bool = False,
    concept_class_roots: tuple[str, ...] = (),
    recovery_pool: int = 400,
    filter_candidate_menu: bool = True,
    max_candidate_l2: float | None = None,
    menu_filter_allowlist: frozenset[str] | None = None,
    menu_ancestor_closure: bool = True,
    pinned_class_labels: tuple[str, ...] = (),
    embed_relationships: bool = True,
    link_tables: bool = True,
    bump_graph_version: bool = True,
    chunk_offset: int = 0,
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
            # `Chunk.id` breaks created_at ties so the order is TOTAL. The
            # streaming driver pages with `chunk_offset`, and an unstable sort
            # would make it skip or repeat chunks between batches.
            .order_by(Chunk.created_at, Chunk.id)
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
        # `chunk_offset` lets the streaming driver step PAST chunks that yield
        # no entities. Such a chunk never gets a viao:assertsAbout edge, so it
        # stays "unprocessed" forever; without an offset it would sit at the
        # head of every batch and starve the chunks behind it.
        if chunk_offset:
            stmt = stmt.offset(chunk_offset)
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
    # And for the orphan check: without its task the step is skipped, not fatal.
    if extract_relationships and relationship_orphan_check:
        try:
            router.task_spec("relationship_orphan_check")
        except KeyError:
            print(
                "[extract-entities] models.yaml has no "
                "'relationship_orphan_check' task -- skipping the orphan "
                "check. Add it (see config/models.example.yaml)."
            )
            relationship_orphan_check = False
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
    # Same degradation for the concept pass.
    if concept_pass:
        try:
            router.task_spec("concept_extract")
        except KeyError:
            print(
                "[extract-entities] models.yaml has no 'concept_extract' task "
                "-- running WITHOUT the concept pass. Add it (see "
                "config/models.example.yaml) to enable --concept-pass."
            )
            concept_pass = False
    # Which classes the concept pass is allowed to assign to, resolved ONCE.
    #
    # The prompt tells the model to use only classes denoting an abstract
    # kind, and on a concept-POOR document it ignores that: measured on a
    # WIRED product-deals article it typed "laptop" -> Laptop, "speaker" ->
    # SpeakerDevice, "battery life" -> FitnessDevice -- bare category nouns
    # forced under concrete product classes, which is precisely the noise the
    # old proper-noun-only rule existed to prevent. Restricting the MENU makes
    # those picks impossible instead of merely discouraged; the model's only
    # remaining options are a real concept class or abstaining.
    #
    # Roots come from config and close over subclasses, so a corpus-specific
    # `AudioTechnology` under `TechnologyConcept` is included automatically.
    concept_class_iris: set[str] = set()
    concept_root_iris: set[str] = set()
    if concept_pass:
        _roots = [r.strip().lower() for r in concept_class_roots if r and r.strip()]
        if _roots:
            async with session_scope() as session:
                rows = await session.execute(
                    _CONCEPT_CLOSURE_SQL, {"labels": _roots}
                )
                concept_class_iris = {r[0] for r in rows.all()}
                # The roots themselves, kept apart from their descendants.
                # A concept typed to a ROOT can never take part in a
                # relationship: no object property declares `Process` or
                # `PolicyConcept` as a domain or range, so "energy efficiency
                # programs -> Process" ended the run with 0 edges while
                # "rate change -> RateChange" got one. Rendering the roots
                # LAST makes the specific classes the ones the model reads
                # first.
                rr = await session.execute(
                    select(OntologyClass.iri).where(
                        func.lower(OntologyClass.label).in_(_roots)
                    )
                )
                concept_root_iris = set(rr.scalars().all())
        if not concept_class_iris:
            print(
                "[extract-entities] concept pass ON but no class matched "
                f"extraction.concept_class_roots ({list(concept_class_roots)}) "
                "-- the ontology has no concept branch, so the pass is "
                "disabled rather than left to pick from concrete classes."
            )
            concept_pass = False
        else:
            print(
                f"[extract-entities] concept pass menu: "
                f"{len(concept_class_iris)} class(es) under "
                f"{len(_roots)} configured root(s)"
            )

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

    if recovery_pool:
        print(
            f"[extract-entities] abstention recovery ON: a proposed_type may "
            f"resolve to any class in this chunk's nearest {recovery_pool}"
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
        "self_loop": 0, "contradictory_direction": 0,
        "no_evidence": 0, "overlong_evidence": 0, "one_sided_evidence": 0,
        "unsupported": 0, "reversed": 0,
        "no_relation_phrase": 0, "generic_superseded": 0,
    }
    # The entity mirror of `rel_drops`. Line-for-line, the old code did a bare
    # `continue` for every one of these -- so the corpus could lose entities
    # steadily with nothing in the output to show for it.
    ent_drops: dict[str, int] = {
        "off_menu_iri": 0, "no_name": 0, "abstained": 0,
        "reviewer_removed": 0, "no_candidates": 0,
        "recovered_label_iri": 0, "recovered_proposed_type": 0,
    }
    val_stats: dict[str, int] = {
        "clean": 0, "revised": 0, "reclassified": 0, "removed": 0,
        "added": 0, "empty_reextract_rejected": 0, "validator_failed": 0,
    }
    concept_stats: dict[str, int] = {
        "chunks_run": 0, "added": 0, "duplicate_of_entity": 0, "failed": 0,
        "no_concept_classes": 0,
    }
    abstained: list[dict[str, Any]] = []
    _abstain_sample_cap = 200
    # Verification pass repairs as well as rejects: a claim whose quote states
    # the relationship backwards is re-emitted with the ends swapped rather
    # than discarded, since the model found a real assertion and only mis-read
    # its direction.
    rel_pass_stats: dict[str, int] = {}
    orphan_flags: list[dict[str, str]] = []
    rel_repairs = {"direction_swapped": 0, "rescued": 0,
                   "predicate_recovered": 0, "type_swapped": 0}
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

    async def _candidate_classes(
        chunk_embedding: list[float],
    ) -> tuple[list[dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
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
        # One query serves two purposes. The first `candidate_classes_per_chunk`
        # survivors become the MENU; the whole ranked pool becomes the recovery
        # index, so an abstention whose `proposed_type` names a class can only
        # reach a class that was at least in this chunk's semantic
        # neighbourhood. Measured, that bound costs ~nothing: 24 recoveries
        # against 25 for an index over the entire ontology, while excluding
        # ~40% of classes -- which is what keeps a homonym (a tennis "Court"
        # resolving to a legal one) out of the graph.
        over_fetch = max(
            candidate_classes_per_chunk * 3
            if menu_excluded
            else candidate_classes_per_chunk,
            recovery_pool,
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
            _pool = [
                {"iri": iri, "label": label or "", "description": descr or ""}
                for iri, label, descr in r.all()
                if iri not in menu_excluded
            ]
            out = _pool[:candidate_classes_per_chunk]
            # Ambiguous keys resolve to nothing rather than an arbitrary
            # winner -- same rule as the menu index, same reason.
            def _recovery(menu: list[dict[str, str]]) -> tuple[
                dict[str, str], dict[str, dict[str, str]]
            ]:
                """Build the abstention-recovery index. Called once the MENU is
                final, because the menu is always recoverable whatever its
                vector rank.

                Pinned and ancestor-closure classes reach the menu without
                ranking well. `Organization` -- pinned, and offered on every
                chunk -- sits at MEDIAN RANK 687 of 1524 on the finance build,
                inside the top-400 pool for only 3 of 27 chunks. So an
                abstention proposing "Organization" found nothing to resolve
                against and the entity was dropped, even though that exact
                class was on the menu the model had been shown. Anything the
                extractor was offered is by definition an acceptable answer.

                Ambiguous keys still resolve to nothing.
                """
                ri: dict[str, str | None] = {}
                rm: dict[str, dict[str, str]] = {}
                if not recovery_pool:
                    return {}, {}
                for c in list(_pool[:recovery_pool]) + list(menu):
                    key = _normalise_type_label(c["label"])
                    if not key:
                        continue
                    if key in ri and ri[key] != c["iri"]:
                        ri[key] = None
                    else:
                        ri.setdefault(key, c["iri"])
                    rm[c["iri"]] = {
                        "label": c["label"], "description": c["description"]
                    }
                return {k: v for k, v in ri.items() if v is not None}, rm

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
                _idx, _meta = _recovery(out)
                return out, _idx, _meta
            rows = await session.execute(
                select(
                    OntologyClass.iri, OntologyClass.label, OntologyClass.description
                ).where(OntologyClass.iri.in_(sorted(extra_iris)))
            )
            for iri, label, descr in rows.all():
                out.append({
                    "iri": iri, "label": label or "", "description": descr or ""
                })
            _idx, _meta = _recovery(out)
            return out, _idx, _meta

    async def _validate_pass(
        txt: str,
        candidates: list[dict[str, str]],
        kept: list[dict[str, Any]],
        cand_iris: set[str],
        class_meta: dict[str, dict[str, str]],
        chunk_iri: str,
        review_meta: dict[str, dict[str, str]] | None = None,
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
        # The reviewer's own rules define `no_class_fits` as "nothing IN THE
        # MENU denotes its kind", so a class that is not on the menu cannot be
        # endorsed -- and a recovered class is off-menu by definition. Measured
        # before this fix: the reviewer returned `no_class_fits` on 68% of
        # recovered entities while returning `correct` on 96% of everything
        # else, rejecting assignments that were plainly right --
        # "Netherlands -> Country", "Dyson V12 Detect Slim -> VacuumCleaner".
        # It was obeying its instructions; the menu was wrong.
        review_meta = review_meta or {}
        review_menu = list(candidates)
        _have = {c["iri"] for c in candidates}
        for _iri in {e["class_iri"] for e in kept} - _have:
            _m = review_meta.get(_iri)
            if _m:
                review_menu.append({
                    "iri": _iri,
                    "label": _m["label"],
                    "description": _m["description"],
                })
        try:
            v_sys, v_user = PROMPTS["entity_validate"](
                txt, review_menu, entities_for_review
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

        # Sanitise against the WIDENED set, or a verdict naming the recovered
        # class is discarded as off-menu -- the same mismatch one level down.
        _review_iris = cand_iris | {c["iri"] for c in review_menu}
        _review_meta_all = {
            **class_meta,
            **{c["iri"]: {"label": c["label"], "description": c["description"]}
               for c in review_menu},
        }
        feedback = _sanitise_verdicts(
            parsed, kept, _review_iris, _review_meta_all, menu_filter_allowlist
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
                candidates, recovery_index, recovery_meta = (
                    await _candidate_classes(chunk_emb)
                )
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
                menu_labels = {c["label"]: c["iri"] for c in candidates}

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
                        cand_labels=menu_labels,
                        type_index=recovery_index,
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
                        review_meta=recovery_meta,
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

            # ---- Concept pass ----
            #
            # A separate call asking ONLY for the concepts the passage
            # develops. `entity_extract` now permits concepts too, but one
            # call holding both jobs spends its attention on the named
            # entities: measured on a shipping chunk, widening that prompt
            # alone moved concept yield 0 -> 1 while the passage developed at
            # least five. This pass has one job.
            #
            # Fails SOFT, like the reviewer: a concept-pass error must not
            # cost the chunk its named entities, which are already in `kept`.
            # Menu narrowed to the concept branch. A chunk whose candidates
            # contain no concept class costs NO call -- on a concept-free
            # corpus the pass is free, not merely harmless.
            # Specific classes first, broad roots last -- see the note where
            # `concept_root_iris` is resolved. Order only; nothing is withheld,
            # because a genuinely generic concept ("inflation") may have no
            # narrower class and must still be typeable.
            c_menu = (
                sorted(
                    (c for c in candidates if c["iri"] in concept_class_iris),
                    key=lambda c: c["iri"] in concept_root_iris,
                )
                if concept_pass else []
            )
            if concept_pass and not c_menu:
                concept_stats["no_concept_classes"] += 1
            if c_menu:
                concept_stats["chunks_run"] += 1
                try:
                    c_sys, c_user = PROMPTS["concept_extract"](txt, c_menu)
                    c_out = await router.chat(
                        "concept_extract", system=c_sys, user=c_user
                    )
                    c_parsed = _extract_json(c_out.text)
                    c_iris = {c["iri"] for c in c_menu}
                    c_kept = (
                        _filter_entities(
                            c_parsed.get("entities"), c_iris, ent_drops,
                            abstained, _abstain_sample_cap,
                            cand_labels={c["label"]: c["iri"] for c in c_menu},
                            type_index=recovery_index,
                        )
                        if isinstance(c_parsed, dict)
                        else []
                    )
                    # entity_extract sees the concept classes too (they are
                    # in the full menu), so it may already have found what
                    # this pass returns. Dedup on the same normalised key the
                    # DB upsert uses, so a duplicate is dropped here rather
                    # than counted twice in the stats.
                    seen = {_dedup_key(e["canonical_name"]) for e in kept}
                    for c in c_kept:
                        key = _dedup_key(c["canonical_name"])
                        if key in seen:
                            concept_stats["duplicate_of_entity"] += 1
                            continue
                        seen.add(key)
                        kept.append(c)
                        concept_stats["added"] += 1
                except Exception as exc:
                    concept_stats["failed"] += 1
                    print(
                        f"[extract-entities] chunk {chunk_iri} concept pass "
                        f"failed: {exc}"
                    )

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
                rels = await _relationships_for_chunk(
                    router, txt, kept,
                    {c["iri"]: c["label"] for c in candidates}, chunk_iri,
                    rel_drops=rel_drops, rel_repairs=rel_repairs,
                    rescue=rescue_relationships, verify=verify_relationships,
                    gap_pass=relationship_gap_pass,
                    orphan_check=relationship_orphan_check,
                    orphan_batch_size=orphan_batch_size,
                    pass_stats=rel_pass_stats, orphan_flags=orphan_flags,
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

    # ---- Collapse near-duplicate names BEFORE identity is decided ----------
    #
    # Applied here rather than at mint time so that the class vote, the entity
    # rows and every edge all agree on one spelling. Rewrites canonical_name
    # in place on the extracted dicts; short_name is left alone because it is
    # the form the chunk actually used and the table linker wants it.
    # Person-descendant classes, for the surname rule. Resolved via the same
    # rdfs:subClassOf walk the predicate menu uses, so a corpus that types
    # people as `Executive` or `Journalist` is covered without configuration.
    _person_iris: set[str] = set()
    async with session_scope() as session:
        _roots = (await session.execute(
            select(OntologyClass.iri).where(
                func.lower(func.replace(OntologyClass.label, " ", "")).in_(
                    ["person", "agent", "foafperson"])
            )
        )).scalars().all()
        if _roots:
            _all = (await session.execute(select(OntologyClass.iri))).scalars().all()
            anc = await session.execute(_ANCESTOR_SQL, {"iris": list(_all)})
            _rootset = set(_roots)
            for origin, a in anc.all():
                if a in _rootset:
                    _person_iris.add(origin)
            _person_iris |= _rootset

    _alias_map, _n_alias = _resolve_canonical_forms(results, _person_iris)
    _renamed_from: dict[str, set[str]] = {}
    if _alias_map:
        # Display form for every normalised key: the longest RAW spelling seen
        # anywhere in the run. Built over ALL entities, not just merge targets
        # -- the old version registered only targets, so a target that no
        # entity spelled out kept an empty string and the rewrite fell through
        # to the NORMALISED key. That is how "Ozempic (semaglutide)" was stored
        # as `ozempic semaglutide` and a bylined author as
        # `phuoc anh anne nguyen pharmd ms bcps`: lowercased, punctuation
        # stripped, and then unfindable, since entity seeding matches on
        # similarity >= 0.4 and "wegovy" scores 0.163 against
        # "wegovy semaglutide injection and oral pill".
        _raw_of: dict[str, str] = {}
        for tup in results:
            if tup is None:
                continue
            for e in (tup[3] or []):
                raw = (e.get("canonical_name") or "").strip()
                if not raw:
                    continue
                nrm = _normalize_name(raw)
                if len(raw) > len(_raw_of.get(nrm, "")):
                    _raw_of[nrm] = raw
        # A target may be a spelling no entity used as its canonical_name (it
        # can come from a short_name or from the word-order/surname merges).
        # Fall back to the best raw form among the variants that alias TO it.
        _variants_of: dict[str, list[str]] = {}
        for variant, target in _alias_map.items():
            _variants_of.setdefault(target, []).append(variant)

        def _display_for(target: str, own: str) -> str:
            best = _raw_of.get(target, "")
            for v in _variants_of.get(target, ()):
                cand = _raw_of.get(v, "")
                if len(cand) > len(best):
                    best = cand
            # Never store a normalised key as a display name: keeping the
            # entity's OWN spelling is wrong-but-readable, which beats
            # wrong-and-lowercased.
            return best or own

        def _merged_name(own: str) -> str | None:
            """The merged display name for `own`, or None if it is unchanged.
            Entities and relationship endpoints both go through here, so the
            two can never disagree about what a name became."""
            nrm = _normalize_name(own)
            tgt = _alias_map.get(nrm)
            if not tgt or tgt == nrm:
                return None
            new = _display_for(tgt, own)
            return new if new != own else None

        _renamed_from, summary.relationship_endpoints_renamed = (
            _apply_merged_names(results, _merged_name))
        summary.renamed_entities = [
            {"from": old, "to": new}
            for new, olds in sorted(_renamed_from.items()) for old in sorted(olds)
        ][:500]
        print(
            f"[extract-entities] name collapse: {_n_alias} variant spelling(s) "
            f"merged into their fullest form (legal suffixes + the model's own "
            f"short_name pairing); {len(summary.renamed_entities)} rename(s) "
            f"applied, {summary.relationship_endpoints_renamed} relationship "
            f"endpoint(s) rewritten to follow. e.g. "
            + "; ".join(f"{r['from']} -> {r['to']}"
                        for r in summary.renamed_entities[:8])
        )

    # Every surface form each final name was written as: the spellings merged
    # into it plus the chunks' own short forms ("FTX" for "FTX Trading Ltd.").
    # Stored on the entity so later passes that start from STORED entities can
    # still recognise the name in text.
    _aliases_by_norm: dict[str, set[str]] = {}
    for new, olds in _renamed_from.items():
        _aliases_by_norm.setdefault(_normalize_name(new), set()).update(olds)
    for tup in results:
        if tup is None:
            continue
        for e in (tup[3] or []):
            short = (e.get("short_name") or "").strip()
            name = (e.get("canonical_name") or "").strip()
            if short and short != name:
                _aliases_by_norm.setdefault(_normalize_name(name), set()).add(short)

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
                    **({"aliases": sorted(
                        _aliases_by_norm[normalized] - {e["canonical_name"]})[:25]}
                       if _aliases_by_norm.get(normalized) else {}),
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
    _conflicts = 0          # mints whose identifier the table already held
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
                    # ON CONFLICT DO NOTHING, not a bare INSERT. Two entities
                    # can arrive with the same `entity_identifier` --
                    # `_entity_iri` hashes (canonical_name, class_iri) while
                    # the reuse check keys on (normalized_name, class_id), and
                    # the name-collapse step renames entities AFTER that check
                    # -- so a rename onto a name the DB already holds collided.
                    # A bare INSERT then rolled back the WHOLE batch, losing
                    # every LLM call in the run: measured 2026-09-20, two
                    # consecutive runs spent $1.94 and wrote zero rows, each
                    # exiting 0. That made the project's own "smoke-test with
                    # --limit N, then scale up" workflow impossible, because
                    # the scale-up run is exactly the colliding case.
                    #
                    # The SELECT below maps EVERY identifier in the batch back
                    # to an id, so a row that lost the conflict resolves to the
                    # entity already in the table -- which is the correct
                    # outcome: it is the same entity.
                    _ins = await session.execute(
                        pg_insert(Entity).values(batch).on_conflict_do_nothing(
                            index_elements=["entity_identifier"]
                        )
                    )
                    _conflicts += max(0, len(batch) - (_ins.rowcount or 0))
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
        f"[extract-entities] embedded + inserted "
        f"{len(entity_mints) - _conflicts} new "
        f"entity(ies): ${summary.embedding_cost_usd:.4f}"
    )
    if _conflicts:
        summary.entities_reused += _conflicts
        print(
            f"[extract-entities] {_conflicts} mint(s) already existed by "
            f"identifier and were reused rather than inserted (usually a "
            f"name collapse landing on a name a previous run stored). Before "
            f"0009 this raised a unique violation that rolled back the whole "
            f"batch and lost the run's LLM work."
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
        resolved_claims: list[tuple[Any, Any, Any, Any, dict[str, Any]]] = []
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
                resolved_claims.append((chunk_id, doc_id, sid, oid, rel))

        rel_payloads = _build_relationship_payloads(
            resolved_claims, gv=gv, rel_drops=rel_drops)

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

    # Vectorize the new entity->entity edges so retrieval can match a
    # question against what an edge SAYS, not just who it touches. Embeddings
    # only (no chat model), pennies per 100k edges, and idempotent -- edges
    # that already carry a vector are skipped. `embed-relationships` does the
    # same thing standalone for a graph built before 0008.
    if embed_relationships and rel_payloads:
        from backend.app.services.db_relationship_embed import (
            embed_relationships as _embed_rels,
        )
        try:
            _rel_emb = await _embed_rels(only_missing=True)
            summary.relationship_embeddings = _rel_emb.embedded
            print(
                f"[extract-entities] embedded {_rel_emb.embedded} "
                f"relationship(s) (${_rel_emb.cost_usd:.4f})"
            )
        except Exception as exc:                      # pragma: no cover
            # Never fail a completed extraction over the vector pass: the
            # edges are already written and `embed-relationships` can
            # backfill. Retrieval falls back to the broad walk meanwhile.
            print(
                f"[extract-entities] WARNING relationship embedding failed "
                f"({exc}); run `embed-relationships` to backfill"
            )

    # Entity-side accounting. Previously every one of these was a silent
    # `continue`, so a corpus could lose entities steadily with nothing in the
    # output to show for it.
    summary.entity_drops = dict(ent_drops)
    summary.abstained_samples = abstained
    # `recovered_label_iri` counts entities SAVED, not lost -- excluded from
    # the drop total so a recovery is never reported as a loss.
    _recovery_keys = ("recovered_label_iri", "recovered_proposed_type")
    _recovered = ent_drops.get("recovered_label_iri", 0)
    _recovered_type = ent_drops.get("recovered_proposed_type", 0)
    _ent_dropped = sum(
        v for k, v in ent_drops.items() if k not in _recovery_keys
    )
    if _recovered_type:
        print(
            f"[extract-entities] recovered {_recovered_type} abstention(s) "
            f"whose proposed_type named a class that EXISTS but had not "
            f"reached that chunk's candidate menu. A high number here means "
            f"the menu is too narrow, NOT that the ontology is missing a "
            f"branch -- a prune-expand run would not change it."
        )
    if _recovered:
        print(
            f"[extract-entities] recovered {_recovered} entity assignment(s) "
            f"where the model wrote the class LABEL into the class_iri field "
            f"instead of the IRI. These would previously have been dropped."
        )
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

    if concept_pass:
        summary.concept_extraction = dict(concept_stats)
        print(
            f"[extract-entities] concept pass: {concept_stats['added']} "
            f"concept(s) added over {concept_stats['chunks_run']} chunk(s) "
            f"({concept_stats['duplicate_of_entity']} already found by the "
            f"entity pass, {concept_stats['no_concept_classes']} chunk(s) had "
            f"no concept class on the menu and cost no call, "
            f"{concept_stats['failed']} call(s) failed)"
        )

    if extract_relationships:
        _dropped = sum(rel_drops.values())
        print(
            f"[extract-entities] entity->entity relationships: "
            f"{len(rel_payloads)} written"
            + (f", {_dropped} dropped "
               f"(unresolved={rel_drops['unresolved']}, "
               f"self_loop={rel_drops['self_loop']}, "
               f"contradictory_direction={rel_drops['contradictory_direction']}, "
               f"bad_predicate={rel_drops['bad_predicate']}, "
               f"domain_range={rel_drops['domain_range']}, "
               f"no_evidence={rel_drops['no_evidence']}, "
               f"overlong_evidence={rel_drops['overlong_evidence']}, "
               f"one_sided_evidence={rel_drops['one_sided_evidence']}, "
               f"unsupported={rel_drops['unsupported']}, "
               f"reversed={rel_drops['reversed']}, "
               f"no_relation_phrase={rel_drops['no_relation_phrase']}, "
               f"generic_superseded={rel_drops['generic_superseded']})"
               if _dropped else "")
            + (f", {rel_repairs['rescued']} rescued by re-homing to a better "
               f"predicate" if rel_repairs["rescued"] else "")
            + (f", {rel_repairs['direction_swapped']} direction(s) repaired"
               if rel_repairs["direction_swapped"] else "")
            + (f", {rel_repairs['type_swapped']} claim(s) swapped to fit the "
               f"predicate's types" if rel_repairs["type_swapped"] else "")
            + (f", {rel_repairs['predicate_recovered']} predicate IRI(s) "
               f"recovered from label/case" if rel_repairs["predicate_recovered"]
               else "")
        )
        _generic = [p for p in rel_payloads
                    if p["predicate_iri"] == GRAPHRAG_RELATED_TO]
        if _generic:
            _phrases = collections.Counter(
                (p["extra_metadata"].get("relation") or "").lower()
                for p in _generic)
            print(
                f"[extract-entities] relatedTo (no ontology predicate fitted): "
                f"{len(_generic)} of {len(rel_payloads)} edges. Most common "
                f"relations -- candidates for new predicates: "
                + ", ".join(f'"{k}" x{v}' for k, v in _phrases.most_common(12))
            )
        summary.relationship_pass_stats = dict(rel_pass_stats)
        summary.orphan_flags = orphan_flags
        if rel_pass_stats:
            _st = rel_pass_stats
            _written_by = collections.Counter(
                (p["extra_metadata"].get("found_by") or "pass1")
                for p in rel_payloads)
            _reasons = {k[len("orphan_reason_"):]: v for k, v in _st.items()
                        if k.startswith("orphan_reason_")}
            print(
                f"[extract-entities] relationship steps: "
                f"pass1 kept {_st.get('pass1_kept', 0)}; "
                f"gap pass ({_st.get('gap_calls', 0)} calls) added "
                f"{_st.get('gap_kept', 0)} (+{_st.get('gap_duplicates', 0)} "
                f"duplicates); orphan check ({_st.get('orphan_calls', 0)} calls) "
                f"flagged {_st.get('orphans_flagged', 0)} entities, added "
                f"{_st.get('orphan_kept', 0)} (+{_st.get('orphan_duplicates', 0)} "
                f"duplicates), {_st.get('orphans_still_unlinked', 0)} still "
                f"unlinked -- reasons {_reasons}; verifier kept "
                f"pass1={_st.get('verified_pass1', 0)} "
                f"gap={_st.get('verified_gap', 0)} "
                f"orphan_check={_st.get('verified_orphan_check', 0)}; "
                f"edges written by step {dict(_written_by)}"
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
    # `link_tables` / `bump_graph_version` exist for the batch-streaming driver
    # (`extract_entities_streamed`), which calls this function once per batch and
    # wants both to happen ONCE for the whole run rather than per batch: table
    # linkage needs every entity visible, and bumping the version per batch would
    # turn one logical ingestion run into N versions, breaking the
    # time-bounded-query semantics graph_version exists for.
    if link_tables:
        await _link_tables_to_entities(summary)

    if bump_graph_version:
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


# ---------------------------------------------------------------------------
# Batch-streamed, resumable driver
# ---------------------------------------------------------------------------

async def _unprocessed_chunk_count(
    *, chunk_kind: str, scope_document_iri: str | None = None
) -> int:
    """How many chunks `extract_entities` would still pick up.

    Mirrors the selection predicate at the top of `extract_entities` exactly.
    If the two ever drift, the streaming loop's progress guard stops firing and
    a zero-entity batch becomes an infinite paid loop -- so they are asserted
    against each other in test_entity_streaming.py.
    """
    async with session_scope() as session:
        already_subq = select(GraphRelationship.source_chunk_id).where(
            GraphRelationship.predicate_iri == VIAO_ASSERTS_ABOUT,
            GraphRelationship.relationship_source == "DOCUMENT_EXTRACTION",
            GraphRelationship.source_chunk_id.isnot(None),
        )
        stmt = select(func.count()).select_from(Chunk).where(
            Chunk.status == "ACTIVE",
            Chunk.kind == chunk_kind,
            Chunk.id.notin_(already_subq),
        )
        if scope_document_iri is not None:
            doc_id = (await session.execute(
                select(Document.id).where(
                    Document.document_identifier == scope_document_iri
                )
            )).scalar_one_or_none()
            if doc_id is None:
                raise ValueError(f"document not found: {scope_document_iri}")
            stmt = stmt.where(Chunk.document_id == doc_id)
        return int((await session.execute(stmt)).scalar() or 0)


def _merge_summaries(
    total: EntityExtractSummary, batch: EntityExtractSummary
) -> None:
    """Accumulate a batch's summary into the run total, in place."""
    for f in (
        "chunks_scanned", "chunks_skipped_already", "chunks_failed",
        "entities_minted", "entities_reused", "chunk_entity_edges",
        "entity_relationship_edges", "relationship_embeddings", "type_edges",
        "tables_scanned", "table_entity_edges", "menu_classes_withheld",
        "relationship_endpoints_renamed",
    ):
        setattr(total, f, getattr(total, f) + getattr(batch, f))
    for f in ("llm_cost_usd", "embedding_cost_usd", "total_cost_usd"):
        setattr(total, f, getattr(total, f) + getattr(batch, f))
    for f in ("entity_drops", "entity_validation", "concept_extraction",
              "relationship_pass_stats"):
        dst, src = getattr(total, f), getattr(batch, f) or {}
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v
    # Bounded: samples exist to be eyeballed, not enumerated. Without a cap a
    # 10M-token run would accumulate every sample from every batch in memory,
    # which is the very thing streaming is here to avoid.
    for f, cap in (("samples", 8), ("abstained_samples", 200),
                   ("renamed_entities", 50), ("orphan_flags", 50)):
        dst, src = getattr(total, f), getattr(batch, f) or []
        if len(dst) < cap:
            dst.extend(src[: cap - len(dst)])
    total.new_graph_version = max(total.new_graph_version,
                                  batch.new_graph_version)


async def extract_entities_streamed(
    *,
    batch_size: int,
    max_cost_usd: float = 5.0,
    chunk_kind: str = "summary",
    scope_document_iri: str | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> EntityExtractSummary:
    """`extract_entities` in committed batches, resumable after a kill.

    WHY: the single-shot path does every LLM call first and writes once at the
    end, so a kill, a crash or a `--max-cost-usd` trip at 90% discards the whole
    run's spend. Measured 2026-09-26: a 282-chunk run costs $8.75 and takes 17
    minutes with nothing durable until the final second.

    Each batch commits, and `extract_entities` already selects only chunks with
    no `viao:assertsAbout` edge -- so the NEXT call naturally resumes where this
    one stopped, whether it stopped cleanly or not. No checkpoint file, no
    resume token: the graph itself is the progress marker.

    The cost of batching is that `_resolve_canonical_forms` (variant-spelling
    collapse) and the plurality class vote see one batch rather than the whole
    run, so a spelling that appears in batch 1 and batch 5 may not merge.
    `existing_exact` still reuses entities already in the DB, so identity is
    preserved for exact normalised matches; it is near-duplicate merging that
    degrades. Chunks are ordered by `created_at`, i.e. document by document, so
    variants of one name usually land in the same batch. Prefer the largest
    batch you can afford to lose.
    """
    t0 = time.time()
    total = EntityExtractSummary()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    remaining_chunks = limit
    batch_no = 0
    # Chunks that yield NO entities never get a viao:assertsAbout edge, so they
    # stay "unprocessed" for ever and sit at the head of every batch. Stepping
    # past them with an offset is what lets the loop terminate AND still reach
    # the chunks behind them -- a plain "no progress -> stop" guard would let a
    # front-loaded cluster of them starve everything newer.
    offset = 0
    stalled = 0
    while True:
        before = await _unprocessed_chunk_count(
            chunk_kind=chunk_kind, scope_document_iri=scope_document_iri
        )
        if before <= offset:
            if stalled:
                print(f"[extract-entities] {stalled} chunk(s) yielded no "
                      f"entities and cannot be marked done; they were skipped "
                      f"after being attempted once. This is a corpus/ontology-"
                      f"fit signal -- see the abstention counts above.")
            break

        spent = total.llm_cost_usd + total.embedding_cost_usd
        budget = max_cost_usd - spent
        if budget <= 0:
            print(f"[extract-entities] cost cap ${max_cost_usd:.2f} reached "
                  f"after {batch_no} batch(es); {before} chunk(s) left. "
                  f"Re-run to continue -- completed batches are committed.")
            break

        this_batch = batch_size
        if remaining_chunks is not None:
            if remaining_chunks <= 0:
                break
            this_batch = min(this_batch, remaining_chunks)

        batch_no += 1
        print(f"[extract-entities] --- batch {batch_no}: up to {this_batch} "
              f"of {before} remaining chunk(s), budget ${budget:.2f} ---")

        batch = await extract_entities(
            limit=this_batch,
            chunk_offset=offset,
            max_cost_usd=budget,
            chunk_kind=chunk_kind,
            scope_document_iri=scope_document_iri,
            # Once for the whole run, after the loop -- not per batch.
            link_tables=False,
            bump_graph_version=False,
            **kwargs,
        )
        _merge_summaries(total, batch)
        if remaining_chunks is not None:
            remaining_chunks -= batch.chunks_scanned

        after = await _unprocessed_chunk_count(
            chunk_kind=chunk_kind, scope_document_iri=scope_document_iri
        )
        if after >= before:
            # No progress: every chunk in this batch yielded no entities, so
            # none can be marked done. Step past them rather than re-selecting
            # (and re-paying for) the same chunks on the next pass. The loop
            # terminates because `offset` only ever grows and `before` is finite.
            attempted = batch.chunks_scanned or this_batch
            offset += attempted
            stalled += attempted
            print(f"[extract-entities] batch {batch_no}: {attempted} chunk(s) "
                  f"yielded no entities; skipping past them (offset={offset}) "
                  f"so the chunks behind them are still reached.")

    # Deferred to here so table linkage sees every entity from every batch, and
    # one logical run produces exactly one graph_version.
    await _link_tables_to_entities(total)
    async with session_scope() as session:
        total.new_graph_version = await bump_version(session)

    total.total_cost_usd = total.llm_cost_usd + total.embedding_cost_usd
    total.wall_seconds = time.time() - t0
    print(
        f"[extract-entities] STREAMED DONE: {batch_no} batch(es), "
        f"chunks={total.chunks_scanned}, "
        f"entities (minted={total.entities_minted}, "
        f"reused={total.entities_reused}), "
        f"chunk_entity_edges={total.chunk_entity_edges}, "
        f"entity_relationship_edges={total.entity_relationship_edges}, "
        f"cost=${total.total_cost_usd:.4f}, "
        f"wall={total.wall_seconds:.1f}s, "
        f"graph_version -> {total.new_graph_version}"
    )
    return total

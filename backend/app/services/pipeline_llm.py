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
    ontology_io,
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
    if not iris and result.text:
        # Salvage path: Groq sometimes truncates mid-string when the
        # relevant_iris list is very long for an exhaustive prompt
        # ("favor recall"). Even if the trailing `]` is missing, we can
        # still extract every fully-quoted IRI before the truncation.
        # Heuristic: find every `"http(s)://..."` token in the raw text.
        salvaged = re.findall(r'"(https?://[^"]+)"', result.text)
        if salvaged:
            print(
                f"[stage1] chunk #{chunk.index} ({chunk.source_name}): "
                f"JSON parse failed; salvaged {len(salvaged)} IRI(s) from "
                f"truncated response"
            )
            iris = salvaged
    cleaned = [i for i in iris if isinstance(i, str) and i]
    # Defense-in-depth: the prompt threatens the model with job loss if
    # it emits > 50 IRIs, but Groq occasionally over-emits anyway. Cap
    # server-side so the downstream slice + the implicit response-budget
    # guarantee both hold even when the model ignores the instruction.
    _STAGE1_HARD_CAP = 50
    if len(cleaned) > _STAGE1_HARD_CAP:
        cleaned = cleaned[:_STAGE1_HARD_CAP]
    return cleaned


# ---------- Stage 2: focused class matching + proposal ----------


# Anchor IRIs that should ALWAYS be available to Stage 2 as candidates,
# regardless of which top-level branches Stage 1 surfaces for a given
# chunk. People + organizations + roles + posts are foundational enough
# that the LLM should never have to invent parents like "BusinessEntity"
# (observed failure mode in pre-fix prune-expand runs). Surfacing them
# via Stage 1 is unreliable because the label "Agent" is too generic for
# the Groq classifier to pick on a domain chunk.
_UNIVERSAL_ANCHOR_IRIS: tuple[str, ...] = (
    "http://xmlns.com/foaf/0.1/Agent",
    "http://xmlns.com/foaf/0.1/Person",
    "http://xmlns.com/foaf/0.1/Organization",
    "http://xmlns.com/foaf/0.1/Group",
    "http://www.w3.org/ns/org#Organization",
    "http://www.w3.org/ns/org#FormalOrganization",
    "http://www.w3.org/ns/org#OrganizationalUnit",
    "http://www.w3.org/ns/org#Role",
    "http://www.w3.org/ns/org#Post",
    "http://www.w3.org/ns/org#Membership",
    "http://www.w3.org/ns/org#Site",
)


# Canonical labels → existing IRIs. When the LLM emits free-text labels
# like "Person" / "Organization" / "Role" for PARENT_LABEL or TYPE_LABEL
# (instead of the full IRI), this map routes them to the canonical
# FOAF/ORG class IRIs in the loaded ontology, preventing orphan-class
# duplication like merged#person and merged#organization.
_CANONICAL_CLASS_LABEL_TO_IRI: dict[str, str] = {
    # FOAF
    "person": "http://xmlns.com/foaf/0.1/Person",
    "organization": "http://xmlns.com/foaf/0.1/Organization",
    "agent": "http://xmlns.com/foaf/0.1/Agent",
    "group": "http://xmlns.com/foaf/0.1/Group",
    "foaf:person": "http://xmlns.com/foaf/0.1/Person",
    "foaf:organization": "http://xmlns.com/foaf/0.1/Organization",
    "foaf:agent": "http://xmlns.com/foaf/0.1/Agent",
    "foaf:group": "http://xmlns.com/foaf/0.1/Group",
    # ORG
    "role": "http://www.w3.org/ns/org#Role",
    "post": "http://www.w3.org/ns/org#Post",
    "membership": "http://www.w3.org/ns/org#Membership",
    "formalorganization": "http://www.w3.org/ns/org#FormalOrganization",
    "formal organization": "http://www.w3.org/ns/org#FormalOrganization",
    "organizationalunit": "http://www.w3.org/ns/org#OrganizationalUnit",
    "organizational unit": "http://www.w3.org/ns/org#OrganizationalUnit",
    "site": "http://www.w3.org/ns/org#Site",
    "org:role": "http://www.w3.org/ns/org#Role",
    "org:post": "http://www.w3.org/ns/org#Post",
    "org:membership": "http://www.w3.org/ns/org#Membership",
    "org:organization": "http://www.w3.org/ns/org#Organization",
    "org:formalorganization": "http://www.w3.org/ns/org#FormalOrganization",
}


# Canonical predicate labels → existing IRIs in the FOAF/ORG ontologies.
# Used to route LLM-emitted MATCH NOT FOUND RELATIONS LABELs to existing
# object_properties instead of minting merged#holds when org:holds
# already exists.
_CANONICAL_PREDICATE_LABEL_TO_IRI: dict[str, str] = {
    # FOAF
    "knows": "http://xmlns.com/foaf/0.1/knows",
    "foaf:knows": "http://xmlns.com/foaf/0.1/knows",
    "member": "http://xmlns.com/foaf/0.1/member",
    "foaf:member": "http://xmlns.com/foaf/0.1/member",
    "topic_interest": "http://xmlns.com/foaf/0.1/topic_interest",
    "topic": "http://xmlns.com/foaf/0.1/topic",
    "interest": "http://xmlns.com/foaf/0.1/interest",
    # ORG
    "holds": "http://www.w3.org/ns/org#holds",
    "heldby": "http://www.w3.org/ns/org#heldBy",
    "held by": "http://www.w3.org/ns/org#heldBy",
    "role": "http://www.w3.org/ns/org#role",
    "haspost": "http://www.w3.org/ns/org#hasPost",
    "has post": "http://www.w3.org/ns/org#hasPost",
    "postin": "http://www.w3.org/ns/org#postIn",
    "post in": "http://www.w3.org/ns/org#postIn",
    "hasmember": "http://www.w3.org/ns/org#hasMember",
    "has member": "http://www.w3.org/ns/org#hasMember",
    "memberof": "http://www.w3.org/ns/org#memberOf",
    "member of": "http://www.w3.org/ns/org#memberOf",
    "hasmembership": "http://www.w3.org/ns/org#hasMembership",
    "has membership": "http://www.w3.org/ns/org#hasMembership",
    "memberduring": "http://www.w3.org/ns/org#memberDuring",
    "member during": "http://www.w3.org/ns/org#memberDuring",
    "organization": "http://www.w3.org/ns/org#organization",
    "hassuborganization": "http://www.w3.org/ns/org#hasSubOrganization",
    "has suborganization": "http://www.w3.org/ns/org#hasSubOrganization",
    "has sub organization": "http://www.w3.org/ns/org#hasSubOrganization",
    "suborganizationof": "http://www.w3.org/ns/org#subOrganizationOf",
    "sub organization of": "http://www.w3.org/ns/org#subOrganizationOf",
    "originalorganization": "http://www.w3.org/ns/org#originalOrganization",
    "original organization": "http://www.w3.org/ns/org#originalOrganization",
    "resultingorganization": "http://www.w3.org/ns/org#resultingOrganization",
    "resulting organization": "http://www.w3.org/ns/org#resultingOrganization",
    "hasprimarysite": "http://www.w3.org/ns/org#hasPrimarySite",
    "has primary site": "http://www.w3.org/ns/org#hasPrimarySite",
    "hassite": "http://www.w3.org/ns/org#hasSite",
    "has site": "http://www.w3.org/ns/org#hasSite",
    "siteof": "http://www.w3.org/ns/org#siteOf",
    "site of": "http://www.w3.org/ns/org#siteOf",
    "changedby": "http://www.w3.org/ns/org#changedBy",
    "resultedfrom": "http://www.w3.org/ns/org#resultedFrom",
    "resultedin": "http://www.w3.org/ns/org#resultedIn",
    "transitivelyhassubord": "http://www.w3.org/ns/org#transitiveSubOrganization",
}


def _coerce_canonical_labels(
    stage2_result: dict[str, Any],
    loaded_ontology: dict[str, Any],
) -> dict[str, Any]:
    """Walk a Stage-2 result (merged + deduped or raw) and coerce
    free-text canonical labels to their existing IRIs.

    - MATCH NOT FOUND classes whose PARENT_LABEL is "Person" /
      "Organization" / "Role" / "Post" / "Membership" (case-insensitive,
      with FOAF/ORG prefix variants) → PARENT_LABEL becomes the
      foaf:Person / foaf:Organization / org:Role / org:Post / org:Membership
      IRI from `loaded_ontology['classes_dict']`.
    - MATCH NOT FOUND INSTANCES with the same TYPE_LABEL stop-list →
      TYPE_LABEL becomes the canonical IRI.
    - MATCH NOT FOUND RELATIONS whose LABEL matches a canonical FOAF/ORG
      predicate (case-insensitive, with prefix variants) → LABEL is left
      alone (predicate-level coercion happens later in
      add_new_relations_from_match_results via the existing-predicate
      lookup), but DOMAIN / RANGE are coerced if they hit the class
      stop-list.

    Coercion only happens when the canonical IRI actually exists in
    loaded_ontology['classes_dict'] (or object_properties_dict for
    predicates). If the merge doesn't carry FOAF or ORG, the labels are
    left as-is so downstream auto-mint still works.

    Mutates the input and returns it for convenience.
    """
    classes_dict = loaded_ontology.get("classes_dict") or {}
    obj_props_dict = loaded_ontology.get("object_properties_dict") or {}

    def _coerce_class(label: Any) -> Any:
        if not isinstance(label, str):
            return label
        norm = label.strip().lower()
        target = _CANONICAL_CLASS_LABEL_TO_IRI.get(norm)
        if target and target in classes_dict:
            return target
        return label

    def _coerce_predicate(label: Any) -> Any:
        if not isinstance(label, str):
            return label
        norm = label.strip().lower()
        target = _CANONICAL_PREDICATE_LABEL_TO_IRI.get(norm)
        if target and target in obj_props_dict:
            return target
        return label

    coerced_classes = 0
    coerced_instances = 0
    coerced_relations = 0

    for cls in stage2_result.get("MATCH NOT FOUND") or []:
        if not isinstance(cls, dict):
            continue
        before = cls.get("PARENT_LABEL")
        after = _coerce_class(before)
        if after != before:
            cls["PARENT_LABEL"] = after
            coerced_classes += 1

    for inst in stage2_result.get("MATCH NOT FOUND INSTANCES") or []:
        if not isinstance(inst, dict):
            continue
        before = inst.get("TYPE_LABEL")
        after = _coerce_class(before)
        if after != before:
            inst["TYPE_LABEL"] = after
            coerced_instances += 1

    for rel in stage2_result.get("MATCH NOT FOUND RELATIONS") or []:
        if not isinstance(rel, dict):
            continue
        # Coerce class endpoints (DOMAIN/RANGE)
        for endpoint in ("DOMAIN", "RANGE"):
            before = rel.get(endpoint)
            after = _coerce_class(before)
            if after != before:
                rel[endpoint] = after
        # Coerce predicate LABEL if it matches a canonical FOAF/ORG name.
        before_label = rel.get("LABEL")
        after_label = _coerce_predicate(before_label)
        if after_label != before_label:
            rel["LABEL"] = after_label
            coerced_relations += 1

    if coerced_classes or coerced_instances or coerced_relations:
        print(
            f"[stage3-coerce] canonical-label coercion: "
            f"{coerced_classes} class parents, "
            f"{coerced_instances} instance types, "
            f"{coerced_relations} predicate labels routed to existing IRIs"
        )
    return stage2_result


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

    Always-include anchors: the universal FOAF + ORG seed IRIs are
    unioned into `detected_iris` before slicing so Stage 2's
    DATA_CLASSES reliably contains foaf:Person / foaf:Organization /
    org:Organization / org:Role / org:Post / org:Membership for every
    chunk -- the LLM uses them as parents for any people/organization/
    role/post proposals.
    """
    classes = loaded_ontology.get("classes_dict", {})
    if not detected_iris:
        return {}
    # Always include the universal anchors that are present in this merge.
    anchor_iris = [iri for iri in _UNIVERSAL_ANCHOR_IRIS if iri in classes]
    seeds = list(detected_iris) + anchor_iris
    # collect_related_class_iris builds its own graph internally; no need to
    # call build_class_graph separately.
    relevant = collect_related_class_iris(classes, seeds, max_hops=max_hops)
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


# ---------- Stage 2 post-filter: drop entity-shaped class proposals ----------
#
# Even with the prompt's "HARD RULE" telling the LLM to emit specific named
# entities as INSTANCES rather than CLASSES, occasional leaks happen at
# gpt-4.1 — proper-noun company names ("BYD Company Ltd."), country names
# ("Myanmar"), and document titles ("Sovereign Risk Tracker") show up
# under MATCH NOT FOUND. The post-filter catches those by label signature
# and demotes them to MATCH NOT FOUND INSTANCES so downstream entity
# extraction can still surface them.

_CORPORATE_SUFFIX_RE = re.compile(
    r"\b(?:Inc|Corp|Corporation|Company|Co|Ltd|Limited|LLC|"
    r"N\.?V\.?|GmbH|AG|S\.?A\.?|S\.?p\.?A\.?|plc|PLC|"
    r"Holdings|Group|Industries|Partners|Bhd|Pty)\b\.?",
    re.IGNORECASE,
)

# Document/report title hint words. The check fires on either a
# whitespace-separated label or a CamelCase label (we split into words
# before testing); so `FertilizerMarketDashboard` and `Fertilizer Market
# Dashboard` both match.
_DOCUMENT_TITLE_TAIL_WORDS = (
    "Report", "Factbook", "Dashboard", "Tracker", "Index", "Bulletin",
    "Briefing", "Forecast", "Outlook", "Whitepaper", "Yearbook", "Atlas",
    "Monitor", "Hub", "Tool", "Database", "Calendar", "Alert", "Portal",
    "Platform", "Survey", "Directory", "Source", "App", "Service",
    "Review", "Reviews", "Newsletter", "Brief", "Memo", "Note", "Page",
    "Site", "Paper",
)
_DOCUMENT_TITLE_TAIL_RE = re.compile(
    # `\s` (NOT `(?:^|\s)`) means the tail word must follow another word.
    # That keeps generic category names like "Forecast" / "Dashboard" /
    # "Alert" as legitimate classes while still catching multi-word /
    # CamelCase named documents like "FoodSecurityTracker".
    r"\s(?:" + "|".join(_DOCUMENT_TITLE_TAIL_WORDS) + r")\s*$",
    re.IGNORECASE,
)

# Years with optional descriptor (e.g. "2025 Factbook", "Q1 2024 Outlook")
_YEAR_PREFIX_RE = re.compile(r"^(?:Q[1-4]\s+)?(?:19|20)\d{2}\b")

# CamelCase splitter. Splits "FertilizerMarketDashboard" -> ["Fertilizer",
# "Market", "Dashboard"]; preserves acronym runs like "WEO" -> ["WEO"],
# and "EUTrade" -> ["EU", "Trade"]; respects existing whitespace.
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"   # acronym before a CamelCase word: "EUT" in "EUTrade"
    r"|[A-Z][a-z]+"            # standard CamelCase token: "Fertilizer"
    r"|[A-Z]+"                 # trailing acronym: "WEO"
    r"|[a-z]+"                 # lowercase token (rare in class labels)
    r"|\d+"                    # digit run: "2025"
)


def _split_camel(label: str) -> str:
    """Return the label with CamelCase boundaries replaced by spaces.

    Already-spaced labels pass through unchanged. Used to canonicalize a
    label before applying tail-word / prefix regexes that key on whole
    words. Examples:
        FertilizerMarketDashboard -> "Fertilizer Market Dashboard"
        EarlyWarningHub           -> "Early Warning Hub"
        WEO                       -> "WEO"
        2025 Factbook             -> "2025 Factbook"
    """
    if " " in label:
        return label
    tokens = _CAMEL_SPLIT_RE.findall(label)
    return " ".join(tokens) if tokens else label


def _looks_like_entity_not_class(
    label: str,
    *,
    known_places: frozenset[str] | None = None,
    extra_corporate_suffix_re: re.Pattern[str] | None = None,
    extra_tail_word_re: re.Pattern[str] | None = None,
) -> tuple[bool, str]:
    """Return (is_entity, reason). When `is_entity` is True, the caller
    should demote the MATCH NOT FOUND class proposal to MATCH NOT FOUND
    INSTANCES instead of letting it land in the ontology as a class.

    `known_places` is the set of lowercased place labels that should NOT
    appear as new classes (because the same name already exists as a
    class in the loaded ontology, OR because they're a configured extra).
    Empty set / None -> the place check is skipped.

    `extra_corporate_suffix_re` / `extra_tail_word_re` are optional user-
    extensions sourced from `config/config.yaml`; same semantics as the
    built-ins."""
    if not isinstance(label, str):
        return False, ""
    cleaned = label.strip()
    if not cleaned:
        return False, ""
    # Heuristic 1: corporate suffix anywhere in the label (built-in or
    # user-extended).
    if _CORPORATE_SUFFIX_RE.search(cleaned):
        return True, "corporate-suffix"
    if extra_corporate_suffix_re is not None and extra_corporate_suffix_re.search(cleaned):
        return True, "corporate-suffix"
    # Heuristic 2: known proper-noun place (case-insensitive exact match).
    # The source set is built from the loaded ontology at runtime.
    if known_places and cleaned.lower() in known_places:
        return True, "known-place"
    # Heuristic 3: document/report title ending. Works on both spaced
    # and CamelCase labels (we split CamelCase first).
    split = _split_camel(cleaned)
    if _DOCUMENT_TITLE_TAIL_RE.search(split):
        return True, "document-title"
    if extra_tail_word_re is not None and extra_tail_word_re.search(split):
        return True, "document-title"
    # Heuristic 4: year prefix (e.g. "2025 Factbook" already caught by
    # heuristic 3; this catches "2025 Strategy Conference" with no tail
    # word). Multi-word labels only -- a bare "2025" gets caught by the
    # temporal-instance path elsewhere.
    if " " in cleaned and _YEAR_PREFIX_RE.match(cleaned):
        return True, "year-prefix"
    return False, ""


# Class-label substrings that mark a "place-kind" class. Used to walk
# the loaded ontology's class hierarchy and collect the labels of every
# class whose ancestry includes one of these -- those labels become the
# dynamic `known_places` set passed to the heuristic.
_PLACE_KIND_LABELS = frozenset(
    s.lower() for s in (
        "Country", "Continent", "Region", "City", "State", "Province",
        "Territory", "GeopoliticalEntity", "PoliticalRegion",
        "AdministrativeRegion", "GeographicRegion", "GeographicEntity",
        "AdministrativeArea",
    )
)


def _build_known_places_from_ontology(
    loaded_ontology: dict[str, Any] | None,
    *,
    extra_labels: list[str] | None = None,
) -> frozenset[str]:
    """Walk the loaded ontology's class hierarchy and return the set of
    labels (lowercased) of every class whose superclass-chain includes
    a place-kind class (Country / Continent / Region / City / etc.).

    Adapts to any merge: a corpus that imports `geography_ontology.owl`
    yields ~250 country / region labels; one that doesn't yields an
    empty set. `extra_labels` from `config/config.yaml` are unioned on
    top. Returns lowercased + stripped labels for case-insensitive
    exact-match comparison.

    Safe on `None` / missing keys -- returns an empty frozenset."""
    out: set[str] = set()
    classes = (loaded_ontology or {}).get("classes_dict") or {}
    if not isinstance(classes, dict):
        classes = {}

    # First, identify the IRIs of the place-kind ROOT classes by label match.
    place_root_iris: set[str] = set()
    for iri, c in classes.items():
        if not isinstance(c, dict):
            continue
        labels = c.get("labels") or []
        if not isinstance(labels, list):
            continue
        for lbl in labels:
            if not isinstance(lbl, str):
                continue
            normalized = re.sub(r"\s+", "", lbl).lower()
            if normalized in _PLACE_KIND_LABELS:
                place_root_iris.add(iri)
                break

    if not place_root_iris:
        # No geography ontology imported -- just return the extras.
        if extra_labels:
            for lbl in extra_labels:
                if isinstance(lbl, str) and lbl.strip():
                    out.add(lbl.strip().lower())
        return frozenset(out)

    # Walk: for each class, follow its superclass chain. If any ancestor
    # is in place_root_iris, every label of THIS class joins the
    # known-places set. Bounded by class count; ontologies have <10k
    # classes so the O(N * avg-chain-depth) walk is fast enough.
    def _superclass_iris(cls_entry: dict[str, Any]) -> list[str]:
        raw = cls_entry.get("superclasses") or cls_entry.get("superclass_iris") or []
        if not isinstance(raw, list):
            return []
        iris: list[str] = []
        for s in raw:
            if isinstance(s, str):
                iris.append(s)
            elif isinstance(s, dict):
                v = s.get("iri") or s.get("name")
                if isinstance(v, str):
                    iris.append(v)
        return iris

    # Cache: iri -> True if its ancestry hits a place root, False otherwise.
    is_place_class: dict[str, bool] = {iri: True for iri in place_root_iris}

    def _walks_to_place(iri: str, depth: int = 0) -> bool:
        if iri in is_place_class:
            return is_place_class[iri]
        if depth > 50:  # cycle guard
            is_place_class[iri] = False
            return False
        entry = classes.get(iri)
        if not isinstance(entry, dict):
            is_place_class[iri] = False
            return False
        result = any(
            _walks_to_place(parent_iri, depth + 1)
            for parent_iri in _superclass_iris(entry)
        )
        is_place_class[iri] = result
        return result

    for iri, c in classes.items():
        if not isinstance(c, dict):
            continue
        if not _walks_to_place(iri):
            continue
        labels = c.get("labels") or []
        if not isinstance(labels, list):
            continue
        for lbl in labels:
            if isinstance(lbl, str) and lbl.strip():
                out.add(lbl.strip().lower())

    # Extras from config last (union -- never override).
    if extra_labels:
        for lbl in extra_labels:
            if isinstance(lbl, str) and lbl.strip():
                out.add(lbl.strip().lower())

    return frozenset(out)


def _compile_extra_word_regex(words: list[str] | None) -> re.Pattern[str] | None:
    """Compile a `\\s(word1|word2|...)\\s*$` tail-word regex from a
    user-provided list. Returns None when the list is empty. Same
    leading-`\\s` policy as the built-in regex (so bare category names
    matching one of the extras still survive)."""
    if not words:
        return None
    safe = [re.escape(w) for w in words if isinstance(w, str) and w.strip()]
    if not safe:
        return None
    return re.compile(
        r"\s(?:" + "|".join(safe) + r")\s*$",
        re.IGNORECASE,
    )


def _compile_extra_suffix_regex(suffixes: list[str] | None) -> re.Pattern[str] | None:
    """Compile a `\\b(s1|s2|...)\\b\\.?` corporate-suffix regex from a
    user-provided list. Returns None when the list is empty."""
    if not suffixes:
        return None
    safe = [re.escape(s) for s in suffixes if isinstance(s, str) and s.strip()]
    if not safe:
        return None
    return re.compile(
        r"\b(?:" + "|".join(safe) + r")\b\.?",
        re.IGNORECASE,
    )


def _filter_entity_shaped_classes(
    stage2_result: dict[str, Any] | None,
    *,
    known_places: frozenset[str] | None = None,
    extra_corporate_suffix_re: re.Pattern[str] | None = None,
    extra_tail_word_re: re.Pattern[str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Strip entity-shaped entries from MATCH NOT FOUND in a single
    stage-2 chunk result, promoting each to MATCH NOT FOUND INSTANCES.

    `known_places` is the dynamic place-class label set built from the
    loaded ontology (see `_build_known_places_from_ontology`); pass
    `None`/empty to skip the place check entirely. `extra_*_re` come
    from `config/config.yaml` user extensions.

    Returns the updated result + a list of demotion records (for audit).
    Safe on `None` / missing keys."""
    demotions: list[dict[str, Any]] = []
    if not isinstance(stage2_result, dict):
        return stage2_result, demotions
    classes = stage2_result.get("MATCH NOT FOUND")
    if not isinstance(classes, list):
        return stage2_result, demotions
    kept_classes: list[Any] = []
    promoted: list[dict[str, Any]] = []
    for entry in classes:
        if not isinstance(entry, dict):
            kept_classes.append(entry)
            continue
        label = entry.get("LABEL", "")
        is_entity, reason = _looks_like_entity_not_class(
            label,
            known_places=known_places,
            extra_corporate_suffix_re=extra_corporate_suffix_re,
            extra_tail_word_re=extra_tail_word_re,
        )
        if not is_entity:
            kept_classes.append(entry)
            continue
        parent = entry.get("PARENT_LABEL") or "NONE"
        promoted.append(
            {
                "LABEL": label,
                "CANONICAL_FORM": label,
                "TYPE_LABEL": parent,
                "DESCRIPTION": entry.get("DESCRIPTION", ""),
            }
        )
        demotions.append({"label": label, "parent": parent, "reason": reason})
    if not promoted:
        return stage2_result, demotions
    stage2_result["MATCH NOT FOUND"] = kept_classes
    existing_instances = stage2_result.get("MATCH NOT FOUND INSTANCES")
    if not isinstance(existing_instances, list):
        existing_instances = []
    existing_instances.extend(promoted)
    stage2_result["MATCH NOT FOUND INSTANCES"] = existing_instances
    return stage2_result, demotions


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


# ---------- Layer H: post-Stage-4 misclassification audit ----------

# Batch size for the classification_audit prompt. Items are ~80-120 tokens
# of JSON each. With max_tokens=4096 on the LLM, ~50 per batch keeps the
# response well within the cap.
_CLASSIFICATION_AUDIT_BATCH_SIZE = 50

# Corporate-suffix patterns. A LABEL containing any of these (case-insensitive,
# word-boundary respecting) signals the entity is an organization.
_CORPORATE_SUFFIXES: tuple[str, ...] = (
    " inc", " inc.", ", inc", " ltd", " ltd.", ", ltd", " corp", " corp.",
    " corporation", " industries", " petrochemicals", " petroleum",
    " company", " co.", " co ltd", " group", " plc", " gmbh", " s.a.",
    " holdings", " bank ", " bank of ", " university", " federation",
    " association", " council ",
)

# Event-keyword patterns. A LABEL containing any of these is almost
# certainly an event named after a place/entity.
_EVENT_KEYWORDS: tuple[str, ...] = (
    "crisis", "closure", "war", "disruption", "shortage", "conflict",
    "incident", "shutdown", "blockage", "attack", "sanctions", "embargo",
    "dispute", "treaty", "escalation", "mobilization", "summit",
)

# Role-keyword patterns. A LABEL that matches one of these exactly (after
# normalisation) is a role type.
_ROLE_LABELS: frozenset[str] = frozenset({
    "president", "primeminister", "prime minister", "chancellor",
    "ceo", "cfo", "cto", "coo", "chairman", "chairwoman", "chairperson",
    "founder", "director", "minister", "secretary",
})


def _label_of(rec: dict[str, Any]) -> str:
    """First non-empty label string for a class record."""
    for lab in (rec.get("labels") or []):
        if isinstance(lab, str) and lab.strip():
            return lab.strip()
        if isinstance(lab, dict):
            v = lab.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    name = rec.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def _is_newly_created(rec: dict[str, Any]) -> bool:
    """A class minted from MATCH NOT FOUND in this run is tagged in
    semantic_role.reasons. Existing classes from source ontologies have
    no such marker."""
    sr = rec.get("semantic_role")
    if not isinstance(sr, dict):
        return False
    reasons = sr.get("reasons") or []
    return any(isinstance(r, str) and "MATCH NOT FOUND" in r for r in reasons)


def _has_corporate_suffix(label: str) -> bool:
    lower = " " + label.lower() + " "
    return any(s in lower for s in _CORPORATE_SUFFIXES)


def _has_event_keyword(label: str) -> bool:
    lower = label.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _EVENT_KEYWORDS)


def _looks_like_person_name(label: str) -> bool:
    """Two- or three-word capitalised names with no corporate / event /
    role tokens. Heuristic; the LLM gives the final verdict."""
    if _has_corporate_suffix(label) or _has_event_keyword(label):
        return False
    parts = label.split()
    if len(parts) < 2 or len(parts) > 4:
        return False
    # All parts start with an uppercase letter, no digits.
    if not all(p[:1].isupper() and not any(c.isdigit() for c in p) for p in parts):
        return False
    # Filter out obvious non-person patterns.
    blocked = {"of", "the", "and", "in", "on", "for", "to", "at"}
    if any(p.lower() in blocked for p in parts):
        return False
    return True


def _first_parent_label(rec: dict[str, Any], classes_dict: dict[str, Any]) -> str:
    """Return the first superclass's local-name or label (whichever is
    informative). owl:Thing returns 'owl:Thing'."""
    for sup in (rec.get("superclasses") or []):
        if not isinstance(sup, dict):
            continue
        iri = sup.get("iri")
        if not iri:
            continue
        if iri.endswith("#Thing") or iri.endswith("/Thing"):
            return "owl:Thing"
        parent_rec = classes_dict.get(iri)
        if parent_rec:
            lab = _label_of(parent_rec)
            if lab:
                return lab
        # Fall back to local name.
        return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return "owl:Thing"


def _is_suspicious(
    iri: str,
    rec: dict[str, Any],
    classes_dict: dict[str, Any],
) -> bool:
    """Quick deterministic filter to decide whether a newly-created class
    is worth a (paid) LLM audit. Returns True if any of:
      - parent is owl:Thing
      - LABEL has corporate suffix but parent isn't Organization-shaped
      - LABEL has event keyword but parent isn't Event-shaped
      - LABEL looks like a person name (regardless of parent -- people
        should be instances, not classes)
      - LABEL matches a role label but parent isn't Role-shaped
    """
    label = _label_of(rec)
    if not label:
        return False
    parent = _first_parent_label(rec, classes_dict).lower()

    if parent == "owl:thing":
        return True

    if _has_corporate_suffix(label) and "organization" not in parent and "organisation" not in parent:
        return True
    if _has_event_keyword(label) and "event" not in parent and "crisis" not in parent and "conflict" not in parent:
        return True
    if _looks_like_person_name(label):
        return True
    if label.lower().replace(" ", "") in _ROLE_LABELS and "role" not in parent:
        return True
    return False


def _classification_audit_cache_key(items: list[dict[str, Any]], model: str) -> str:
    """SHA-256 over the canonical JSON of the audit batch. Same items in
    the same order produce the same key (deterministic cache hits)."""
    import hashlib
    canonical = json.dumps(items, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256()
    h.update(b"audit-v1|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(canonical.encode("utf-8"))
    return h.hexdigest()


def _build_audit_items(
    classes_dict: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Find every NEWLY-CREATED class whose current placement is
    suspicious. Returns (iri, item_dict_for_llm) tuples preserving
    classes_dict order so audit batches are deterministic."""
    items: list[tuple[str, dict[str, Any]]] = []
    for iri, rec in classes_dict.items():
        if not _is_newly_created(rec):
            continue
        if not _is_suspicious(iri, rec, classes_dict):
            continue
        label = _label_of(rec)
        parent_label = _first_parent_label(rec, classes_dict)
        descr = ""
        for field in ("descriptions", "compact_description", "comments"):
            v = rec.get(field)
            if isinstance(v, str) and v.strip():
                descr = v.strip()
                break
            if isinstance(v, list):
                for vv in v:
                    if isinstance(vv, str) and vv.strip():
                        descr = vv.strip()
                        break
                if descr:
                    break
        items.append((iri, {
            "LABEL": label,
            "CURRENT_PARENT": parent_label,
            "DESCRIPTION": descr[:300],
        }))
    return items


def _find_or_synthesize_parent_iri(
    new_parent_label: str,
    classes_dict: dict[str, Any],
    default_base_iri: str,
) -> str:
    """Look up `new_parent_label` in classes_dict by label match
    (case-insensitive). If not found, synthesize a new class at
    `default_base_iri + slug(label)` and create a minimal record so the
    re-homed children have a real anchor."""
    target = new_parent_label.strip().lower()
    for iri, rec in classes_dict.items():
        for lab in (rec.get("labels") or []):
            if isinstance(lab, str) and lab.strip().lower() == target:
                return iri
            if isinstance(lab, dict):
                v = lab.get("value")
                if isinstance(v, str) and v.strip().lower() == target:
                    return iri
        name = rec.get("name")
        if isinstance(name, str) and name.strip().lower() == target:
            return iri
    # Synthesize a new bucket class.
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", new_parent_label.strip()).strip("_")
    if not slug:
        slug = "auto_bucket"
    new_iri = default_base_iri.rstrip("/").rstrip("#") + "/" + slug
    classes_dict[new_iri] = {
        "iri": new_iri,
        "name": slug,
        "labels": [new_parent_label.strip()],
        "descriptions": [
            f"Auto-created top-level bucket via Layer H classification audit "
            f"({new_parent_label.strip()})."
        ],
        "superclasses": [{
            "iri": "http://www.w3.org/2002/07/owl#Thing",
            "name": "Thing",
        }],
        "semantic_role": {
            "role": "proposed_concept_class",
            "confidence": "rule_based",
            "reasons": ["Created from MATCH NOT FOUND (Layer H audit)"],
            "scores": {},
        },
        "auto_created_via": ["classification_audit"],
    }
    return new_iri


# Parent labels for which a person-shape LABEL is unambiguously an
# individual person, not a class. Layer H overrides the LLM's verdict
# for these cases (force-convert to instance of Person).
_PERSON_PARENT_LABELS: frozenset[str] = frozenset({
    "person", "personrole", "personorrole", "role", "agent",
    "foaf:person", "foaf:agent", "org:role",
})


def _force_convert_person_shape(
    label: str, current_parent_label: str
) -> bool:
    """True when an entity whose LABEL looks like a person name AND
    is currently parented under a Person/PersonRole/Role-shaped class
    SHOULD be converted to an instance, regardless of LLM verdict.

    Catches Layer H's blind spots -- the gpt-4o-mini audit sometimes
    KEEPs entities like "Elon Musk" parented under foaf:Person, even
    though they are obviously individuals, not subclasses."""
    if not _looks_like_person_name(label):
        return False
    p = (current_parent_label or "").strip().lower()
    return p in _PERSON_PARENT_LABELS


def _apply_audit_decision(
    iri: str,
    rec: dict[str, Any],
    decision: dict[str, Any],
    classes_dict: dict[str, Any],
    instances_dict: dict[str, Any],
    default_base_iri: str,
    current_parent_label: str = "",
) -> str:
    """Apply ONE decision to the class. Returns the action that was
    actually applied: 'kept', 'rehomed', 'converted', or 'noop' (if
    validation failed).

    Pre-check: if the entity has a person-name shape AND is parented
    under Person/PersonRole/Role, force CONVERT_TO_INSTANCE under
    Person. Overrides any LLM verdict (the LLM sometimes KEEPs these
    despite the prompt's explicit instruction). `current_parent_label`
    is the label of `rec`'s current parent (caller passes it in to
    avoid re-resolving)."""
    label = _label_of(rec)
    if _force_convert_person_shape(label, current_parent_label):
        decision = dict(decision)
        decision["ACTION"] = "CONVERT_TO_INSTANCE"
        decision["NEW_PARENT"] = "Person"

    action = (decision.get("ACTION") or "").strip().upper()
    new_parent_label = (decision.get("NEW_PARENT") or "").strip()

    if action == "KEEP" or action == "":
        return "kept"

    if action == "RE_HOME":
        if not new_parent_label:
            return "noop"
        new_parent_iri = _find_or_synthesize_parent_iri(
            new_parent_label, classes_dict, default_base_iri
        )
        rec["superclasses"] = [{
            "iri": new_parent_iri,
            "name": classes_dict[new_parent_iri].get("name") or _label_of(classes_dict[new_parent_iri]),
        }]
        # Tag for audit.
        rec.setdefault("auto_created_via", []).append("classification_audit")
        return "rehomed"

    if action == "CONVERT_TO_INSTANCE":
        if not new_parent_label:
            return "noop"
        type_iri = _find_or_synthesize_parent_iri(
            new_parent_label, classes_dict, default_base_iri
        )
        label = _label_of(rec)
        # Build the instance record. The instance keeps the class's IRI
        # so that any existing relations pointing at this entity still
        # resolve (they'll just point at an instance instead of a class).
        instances_dict[iri] = {
            "iri": iri,
            "name": rec.get("name") or label,
            "labels": rec.get("labels") or [label],
            "descriptions": rec.get("descriptions") or [],
            "comments": rec.get("comments") or [],
            "types": [{
                "iri": type_iri,
                "name": classes_dict[type_iri].get("name") or _label_of(classes_dict[type_iri]),
            }],
            "direct_types": [{
                "iri": type_iri,
                "name": classes_dict[type_iri].get("name") or _label_of(classes_dict[type_iri]),
            }],
            "semantic_role": rec.get("semantic_role"),
            "auto_created_via": ["classification_audit"],
        }
        # Drop the class entry.
        del classes_dict[iri]
        return "converted"

    return "noop"


async def run_classification_audit_async(
    classes_dict: dict[str, Any],
    instances_dict: dict[str, Any],
    router: LLMRouter,
    default_base_iri: str = "https://veerla-ramrao.ai/ontology/merged#",
    concurrency: int = 8,
    use_cache: bool = True,
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Layer H: run the LLM-based misclassification audit over the
    newly-created classes in `classes_dict`. Mutates classes_dict and
    instances_dict in place. Returns a summary dict.

    Caches each batch by SHA-256 of its items at
    ~/.cache/your-end-to-end-graphrag-implementation/doc_summaries/
    (the same cache directory as document summaries) under a key
    prefixed with 'audit-v1'. Re-running against the same set of
    suspicious classes costs nothing for LLM calls.
    """
    items = _build_audit_items(classes_dict)
    if not items:
        print("[stage4-H] classification_audit: no suspicious classes -- skipping")
        return {"suspicious": 0, "decisions": 0, "kept": 0, "rehomed": 0,
                "converted": 0, "noop": 0, "llm_calls": 0,
                "cost_usd": 0.0, "batches": 0}

    print(
        f"[stage4-H] classification_audit: {len(items)} suspicious "
        f"class(es) to review (out of {len(classes_dict)} total)"
    )

    # Slice into batches.
    batches: list[list[tuple[str, dict[str, Any]]]] = [
        items[i : i + _CLASSIFICATION_AUDIT_BATCH_SIZE]
        for i in range(0, len(items), _CLASSIFICATION_AUDIT_BATCH_SIZE)
    ]
    print(f"[stage4-H] processing {len(batches)} batch(es) of <= {_CLASSIFICATION_AUDIT_BATCH_SIZE}")

    cache_dir = _doc_summary_cache_dir() if use_cache else None
    sem = asyncio.Semaphore(concurrency)
    cost_before = router.total_cost_usd

    # Each batch is processed independently; results are merged afterwards.
    batch_decisions: list[list[dict[str, Any]]] = [[] for _ in batches]
    llm_calls = 0
    llm_call_lock = asyncio.Lock()

    async def _one_batch(batch_idx: int, batch: list[tuple[str, dict[str, Any]]]) -> None:
        nonlocal llm_calls
        # Cache lookup
        batch_items = [item for _, item in batch]
        cache_key = _classification_audit_cache_key(batch_items, model_name)
        if cache_dir is not None:
            cached = _doc_summary_cache_load(_doc_summary_cache_path(cache_dir, cache_key))
            if cached:
                try:
                    parsed = json.loads(cached)
                    if isinstance(parsed, dict) and isinstance(parsed.get("DECISIONS"), list):
                        batch_decisions[batch_idx] = parsed["DECISIONS"]
                        return
                except json.JSONDecodeError:
                    pass  # fall through to fresh LLM call
        # LLM call
        async with sem:
            system, user = PROMPTS["classification_audit"](batch_items)
            try:
                result = await router.chat("classification_audit", system=system, user=user)
            except Exception as exc:
                print(f"[stage4-H] batch {batch_idx+1}/{len(batches)} LLM call failed: {exc} — skipping")
                return
            async with llm_call_lock:
                llm_calls += 1
            parsed = extract_json_from_output(result.text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("DECISIONS"), list):
                print(f"[stage4-H] batch {batch_idx+1}/{len(batches)} response unparseable — skipping")
                return
            decisions = parsed["DECISIONS"]
            batch_decisions[batch_idx] = decisions
            # Cache write
            if cache_dir is not None:
                _doc_summary_cache_save(
                    _doc_summary_cache_path(cache_dir, cache_key),
                    json.dumps({"DECISIONS": decisions}, ensure_ascii=False),
                )

    await asyncio.gather(*[_one_batch(i, batch) for i, batch in enumerate(batches)])

    # Apply decisions. Match by exact LABEL (case-insensitive).
    counters = {"kept": 0, "rehomed": 0, "converted": 0, "noop": 0}
    total_decisions = 0
    for batch_idx, batch in enumerate(batches):
        decisions_by_label: dict[str, dict[str, Any]] = {}
        for d in batch_decisions[batch_idx]:
            if not isinstance(d, dict):
                continue
            lab = (d.get("LABEL") or "").strip().lower()
            if lab:
                decisions_by_label[lab] = d
        for iri, item in batch:
            lab = (item.get("LABEL") or "").strip().lower()
            if not lab or lab not in decisions_by_label:
                counters["noop"] += 1
                continue
            rec = classes_dict.get(iri)
            if rec is None:  # already converted/deleted by an earlier decision pointing at same iri
                continue
            outcome = _apply_audit_decision(
                iri, rec, decisions_by_label[lab],
                classes_dict, instances_dict, default_base_iri,
                current_parent_label=item.get("CURRENT_PARENT", ""),
            )
            counters[outcome] = counters.get(outcome, 0) + 1
            total_decisions += 1

    cost_delta = router.total_cost_usd - cost_before
    print(
        f"[stage4-H] DONE: {total_decisions} decisions applied "
        f"(kept={counters['kept']}, rehomed={counters['rehomed']}, "
        f"converted={counters['converted']}, noop={counters['noop']}), "
        f"{llm_calls} LLM call(s), cost ${cost_delta:.4f}"
    )

    return {
        "suspicious": len(items),
        "decisions": total_decisions,
        "kept": counters["kept"],
        "rehomed": counters["rehomed"],
        "converted": counters["converted"],
        "noop": counters["noop"],
        "llm_calls": llm_calls,
        "cost_usd": round(cost_delta, 6),
        "batches": len(batches),
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

# Bump this string ONLY when the document_summarize prompt OR the
# summarization strategy changes meaningfully. The version is mixed into each
# cache key so old cached summaries are silently invalidated.
# v4 (2026-07-04): segment-wise summarization for docs >4k tokens (was single
#   over-compressed summary >100k only) -> fuller, proportional summaries.
_DOC_SUMMARY_PROMPT_VERSION = "v4"


def _doc_summary_cache_dir() -> Path:
    """Return the on-disk cache root, creating it if missing."""
    root = Path.home() / ".cache" / "your-end-to-end-graphrag-implementation" / "doc_summaries"
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
    max_doc_input_tokens: int = 4_000,
    oversize_doc_sub_chunk_tokens: int = 4_000,
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
    ~/.cache/your-end-to-end-graphrag-implementation/doc_summaries/. Cache
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
    # Show the ACTUAL configured model (from models.yaml document_summarize),
    # not a hardcoded label -- otherwise a gpt-4.1 run misreports as gpt-4o-mini.
    try:
        _summ_model = router.task_spec("document_summarize").get("model", model_name)
    except Exception:
        _summ_model = model_name
    print(
        f"[summarize-docs] {len(plan)}/{len(documents)} doc(s) over "
        f"threshold ({threshold_tokens} tokens): "
        f"{cache_hits} cache hit(s), {len(needs_llm)} cache miss(es) "
        f"({oversize_needs_llm} require segment-wise summarization) "
        f"via {_summ_model} at concurrency={concurrency}"
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


# ---------- Streaming load+summarize+chunk (memory-bounded) ----------


async def stream_summarize_and_chunk_async(
    *,
    documents_dir: Path,
    router: LLMRouter,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    encoding_name: str = "o200k_base",
    threshold_tokens: int = 2000,
    max_doc_input_tokens: int = 4_000,
    oversize_doc_sub_chunk_tokens: int = 4_000,
    use_cache: bool = True,
    concurrency: int = 4,
    batch_size: int = 16,
) -> list[TextChunk]:
    """Walk `documents_dir` in fixed-size batches, summarize each batch
    via `summarize_long_documents_async`, chunk the (possibly
    summarized) text, and return the accumulated chunks.

    Memory bound: at any moment, only `batch_size` raw doc texts are
    held in memory. Once a batch is chunked, its LoadedDocuments are
    released. The returned chunk list is the only persistent state --
    typically ~10 MB even for a 22M-token corpus because summarization
    compresses 95% of the volume away.

    Replaces the load-everything-at-once pattern that OOM'd the
    pipeline on the 348-doc / 22M-token corpus:

        docs = list(document_io.load_documents(...))   # ALL IN MEMORY
        docs = await summarize_long_documents_async(...)
        chunks = list(chunk_documents(docs, ...))

    Set `batch_size=0` to fall back to that legacy "load all at once"
    behaviour (useful for tests or environments with plenty of RAM).
    """
    paths = list(ontology_io.iter_documents(documents_dir))
    if not paths:
        raise RuntimeError(f"No PDF/TXT documents found in {documents_dir}")

    if batch_size <= 0:
        # Legacy path: load everything, summarize, chunk in one shot.
        docs = [document_io.load_document(p) for p in paths]
        # Filter out empty docs (read failures, scanned PDFs).
        docs = [d for d in docs if d.text.strip()]
        if threshold_tokens > 0:
            docs = await summarize_long_documents_async(
                documents=docs,
                router=router,
                threshold_tokens=threshold_tokens,
                encoding_name=encoding_name,
                concurrency=concurrency,
                use_cache=use_cache,
                max_doc_input_tokens=max_doc_input_tokens,
                oversize_doc_sub_chunk_tokens=oversize_doc_sub_chunk_tokens,
            )
        chunks = list(chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        ))
        return chunks

    print(
        f"[stream-loader] streaming {len(paths)} doc(s) in batches of "
        f"{batch_size} (peak in-memory docs <= batch_size)"
    )

    all_chunks: list[TextChunk] = []
    total_batches = (len(paths) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch_paths = paths[start : start + batch_size]

        # Load this batch.
        batch_docs: list[LoadedDocument] = []
        for p in batch_paths:
            try:
                doc = document_io.load_document(p)
            except Exception as exc:
                print(f"[stream-loader] skipping {p.name}: {exc}")
                continue
            if doc.text.strip():
                batch_docs.append(doc)

        if not batch_docs:
            continue

        # Summarize this batch (cache reads + writes go through the
        # existing helper; oversize docs go through hierarchical path).
        if threshold_tokens > 0:
            batch_docs = await summarize_long_documents_async(
                documents=batch_docs,
                router=router,
                threshold_tokens=threshold_tokens,
                encoding_name=encoding_name,
                concurrency=concurrency,
                use_cache=use_cache,
                max_doc_input_tokens=max_doc_input_tokens,
                oversize_doc_sub_chunk_tokens=oversize_doc_sub_chunk_tokens,
            )

        # Chunk this batch and append to the master list. Generator is
        # consumed eagerly here so the LoadedDocument refs can be freed
        # right after the loop.
        for chunk in chunk_documents(
            batch_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        ):
            all_chunks.append(chunk)

        # Drop refs to this batch's docs so the GC can release the text.
        del batch_docs
        print(
            f"[stream-loader] batch {batch_idx + 1}/{total_batches} done; "
            f"chunks so far: {len(all_chunks)}"
        )

    return all_chunks


async def stream_evaluated_summarize_and_chunk_async(
    *,
    documents_dir: Path,
    router: LLMRouter,
    threshold_tokens: int = 2000,
    eval_rounds: int = 3,
    questions_per_chunk: int = 12,
    max_chunk_tokens: int = 12_000,
    overlap_tokens: int = 500,
    summary_chunk_max_tokens: int = 1_200,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "o200k_base",
    use_cache: bool = True,
    concurrency: int = 4,
    batch_size: int = 16,
    max_cost_usd: float | None = None,
) -> list[TextChunk]:
    """Evaluated (near-lossless) counterpart of stream_summarize_and_chunk_async
    used when `summarization.method == 'evaluated'`. Walks `documents_dir` in
    fixed-size batches, runs `evaluated_summarize_documents_async` per batch, and
    flattens each doc's evaluated chunk summaries into TextChunks (one chunk per
    stored summary piece -- NO re-chunk). Writes to the shared `eval_summaries/`
    cache with the SAME key params register-documents uses, so a later
    register-documents run reuses these summaries for free.

    `batch_size <= 0` loads everything at once (tests / high-RAM)."""
    from backend.app.services.evaluated_summarizer import (
        evaluated_result_to_chunks,
        evaluated_summarize_documents_async,
    )

    paths = list(ontology_io.iter_documents(documents_dir))
    if not paths:
        raise RuntimeError(f"No PDF/TXT documents found in {documents_dir}")

    async def _summarize_batch(batch_docs: list[LoadedDocument]) -> list[TextChunk]:
        results = await evaluated_summarize_documents_async(
            batch_docs,
            router,
            threshold_tokens=threshold_tokens,
            eval_rounds=eval_rounds,
            questions_per_chunk=questions_per_chunk,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
            summary_chunk_max_tokens=summary_chunk_max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            concurrency=concurrency,
            use_cache=use_cache,
            max_cost_usd=max_cost_usd if max_cost_usd is not None else 1000.0,
        )
        out: list[TextChunk] = []
        for r in results:
            out.extend(evaluated_result_to_chunks(r))
        return out

    if batch_size <= 0:
        docs = [document_io.load_document(p) for p in paths]
        docs = [d for d in docs if d.text.strip()]
        return await _summarize_batch(docs)

    print(
        f"[stream-loader] evaluated summarization: streaming {len(paths)} doc(s) "
        f"in batches of {batch_size} (eval_rounds={eval_rounds})"
    )
    all_chunks: list[TextChunk] = []
    total_batches = (len(paths) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch_docs: list[LoadedDocument] = []
        for p in paths[start : start + batch_size]:
            try:
                doc = document_io.load_document(p)
            except Exception as exc:
                print(f"[stream-loader] skipping {p.name}: {exc}")
                continue
            if doc.text.strip():
                batch_docs.append(doc)
        if not batch_docs:
            continue
        all_chunks.extend(await _summarize_batch(batch_docs))
        del batch_docs
        print(
            f"[stream-loader] batch {batch_idx + 1}/{total_batches} done; "
            f"chunks so far: {len(all_chunks)}"
        )
    return all_chunks


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
    extra_stage2_results: list[dict[str, Any]] | None = None,
    single_pass_summaries: bool = False,
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

    # Memory-bounded load-summarize-chunk: paths are walked in batches
    # of `streaming_batch_size`, so peak memory stays low even for
    # large corpora (the previous "list(load_documents(...))" pattern
    # OOM'd at 22M-token / 348-doc scale on a 2.7 GiB box).
    expansion_cfg_for_concur = app_cfg.get("expansion", {}) or {}
    threshold_tokens = int(chunking_cfg.get("summarization_threshold_tokens", 0))
    use_cache = bool(chunking_cfg.get("use_summary_cache", True))
    max_doc_input_tokens = int(chunking_cfg.get("max_doc_input_tokens", 4_000))
    oversize_sub_chunk = int(chunking_cfg.get("oversize_doc_sub_chunk_tokens", 4_000))
    streaming_batch_size = int(chunking_cfg.get("streaming_batch_size", 16))
    _concurrency = int(expansion_cfg_for_concur.get("max_concurrent_llm_calls", 4))

    # Method: evaluated (near-lossless, new default) vs single_pass (legacy).
    # `single_pass_summaries=True` (CLI --single-pass-summaries) forces legacy.
    sum_cfg = app_cfg.get("summarization", {}) or {}
    method = "single_pass" if single_pass_summaries else sum_cfg.get("method", "evaluated")

    if method == "evaluated":
        chunks = await stream_evaluated_summarize_and_chunk_async(
            documents_dir=documents_dir,
            router=router,
            threshold_tokens=threshold_tokens,
            eval_rounds=int(sum_cfg.get("eval_rounds", 3)),
            questions_per_chunk=int(sum_cfg.get("questions_per_chunk", 12)),
            max_chunk_tokens=int(sum_cfg.get("max_chunk_tokens", 12_000)),
            overlap_tokens=int(sum_cfg.get("overlap_tokens", 500)),
            summary_chunk_max_tokens=int(sum_cfg.get("summary_chunk_max_tokens", 1_200)),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding,
            use_cache=use_cache,
            concurrency=_concurrency,
            batch_size=streaming_batch_size,
            max_cost_usd=max_cost_usd,
        )
    else:
        chunks = await stream_summarize_and_chunk_async(
            documents_dir=documents_dir,
            router=router,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding,
            threshold_tokens=threshold_tokens,
            max_doc_input_tokens=max_doc_input_tokens,
            oversize_doc_sub_chunk_tokens=oversize_sub_chunk,
            use_cache=use_cache,
            concurrency=_concurrency,
            batch_size=streaming_batch_size,
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

    # Entity-shaped-class filter inputs: dynamic place set sourced from the
    # loaded ontology + optional user extensions from config.yaml.
    entity_filter_cfg = expansion_cfg.get("entity_class_filter", {}) or {}
    known_places = _build_known_places_from_ontology(
        loaded_ontology,
        extra_labels=entity_filter_cfg.get("extra_known_place_labels") or [],
    )
    extra_suffix_re = _compile_extra_suffix_regex(
        entity_filter_cfg.get("extra_corporate_suffixes") or []
    )
    extra_tail_re = _compile_extra_word_regex(
        entity_filter_cfg.get("extra_doc_tail_words") or []
    )
    print(
        f"[stage2-filter] known_places={len(known_places)} (from ontology + config), "
        f"extra_suffixes={'on' if extra_suffix_re else 'off'}, "
        f"extra_tail_words={'on' if extra_tail_re else 'off'}"
    )

    async def _classify_one(chunk: TextChunk) -> list[str]:
        async with sem:
            return await _classify_chunk(router, branches, chunk)

    _s1_spec = router.task_spec("chunk_classification")
    print(
        f"[stage1] classifying {len(chunks)} chunk(s) "
        f"({_s1_spec.get('provider', '?')}:{_s1_spec.get('model', '?')}, "
        f"concurrency={concurrency})"
    )
    stage1_results = await asyncio.gather(*[_classify_one(c) for c in chunks])

    # Stage-2 progress heartbeat: class_proposal fires one LLM call per chunk
    # with no per-chunk output, so under rate limiting the run can look frozen
    # for a long time. Count completions and log every `stage2_log_every`.
    stage2_done = 0
    stage2_total = 0
    stage2_log_every = 1
    stage2_over_budget = False

    async def _propose_one(idx: int, chunk: TextChunk, iris: list[str]) -> dict[str, Any] | None:
        nonlocal stage2_done, stage2_over_budget
        async with sem:
            # Cost short-circuit: once the cap is crossed, stop paying for
            # further Stage-2 calls (queued chunks return None). Bounds the
            # overshoot to ~concurrency in-flight calls instead of all of them.
            if stage2_over_budget:
                return None
            res = await _propose_for_chunk(
                router,
                loaded_ontology,
                iris,
                chunk,
                max_hops,
                suggested_new_classes=suggested_new_classes,
            )
            if res:
                res, demotions = _filter_entity_shaped_classes(
                    res,
                    known_places=known_places,
                    extra_corporate_suffix_re=extra_suffix_re,
                    extra_tail_word_re=extra_tail_re,
                )
                if demotions:
                    _append_audit(
                        audit_path,
                        idx,
                        "entity_shaped_class_demotions",
                        chunk.source_name,
                        {"demotions": demotions},
                    )
                _append_audit(audit_path, idx, "class_proposal", chunk.source_name, res)
            stage2_done += 1
            if stage2_done == stage2_total or stage2_done % stage2_log_every == 0:
                print(f"[stage2]   {stage2_done}/{stage2_total} chunks proposed")
            if max_cost_usd is not None and router.total_cost_usd > max_cost_usd:
                stage2_over_budget = True
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

    _s2_spec = router.task_spec("class_proposal")
    stage2_total = len(chunks) - skipped_empty
    stage2_log_every = max(1, stage2_total // 50)
    # Cost gate BEFORE the expensive Stage-2 fan-out: if summarization +
    # Stage 1 already blew the budget, abort here instead of paying for
    # hundreds of Stage-2 calls first (the old check ran only AFTER Stage 2,
    # so a blown budget still paid for every Stage-2 call before aborting).
    if max_cost_usd is not None and router.total_cost_usd > max_cost_usd:
        raise RuntimeError(
            f"Cost cap reached before Stage 2: ${router.total_cost_usd:.4f} > "
            f"${max_cost_usd:.4f} (summarization + Stage 1). Stage 2 "
            f"({stage2_total} calls) was NOT started, so no further spend. "
            "Re-run with a higher --max-cost-usd to lift the cap."
        )
    print(
        f"[stage2] proposing matches+new for {stage2_total} chunk(s) "
        f"({_s2_spec['provider']}:{_s2_spec['model']})"
    )
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

    # Phase 2a follow-up: optional extra Stage-2-shaped JSON dicts from the
    # table-concept-grouping pass.  These are merged into the chunk results
    # via the same recursive merge so Stage 3 dedup collapses table-derived
    # proposals against prose-derived ones uniformly.
    if extra_stage2_results:
        valid.extend(r for r in extra_stage2_results if r)
        print(
            f"[stage2] merged {len(extra_stage2_results)} extra Stage-2 "
            "result(s) from table mining"
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
    # Canonical-label coercion: replace free-text "Person" / "Organization"
    # / "Role" / "Post" PARENT_LABELs and TYPE_LABELs with the actual FOAF/
    # ORG class IRIs from the merge, and route predicate LABELs like
    # "holds" / "memberOf" / "hasPost" to org#holds etc. Prevents the
    # orphan-class duplication (merged#person, merged#organization, ...)
    # and predicate explosion (merged#holds vs org#holds).
    deduped = _coerce_canonical_labels(deduped, loaded_ontology)
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
    extract_tables: bool = False,
    table_vision: bool = True,
    single_pass_summaries: bool = False,
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
        extract_tables=extract_tables,
        table_vision=table_vision,
        single_pass_summaries=single_pass_summaries,
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
    extract_tables: bool = False,
    table_vision: bool = True,
    single_pass_summaries: bool = False,
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
        extract_tables=extract_tables,
        table_vision=table_vision,
        single_pass_summaries=single_pass_summaries,
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
    *,
    extract_tables: bool = False,
    table_vision: bool = True,
    single_pass_summaries: bool = False,
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

    # Pre-flight: surface unreadable documents BEFORE the first paid stage.
    # Text extraction can yield confident gibberish (subsetted /Type0 fonts with
    # no /ToUnicode CMap), and every downstream stage will happily pay to process
    # it. The check is local CPU only -- no LLM, page-sampled -- so it costs
    # seconds and can save an entire run.
    if documents_dir.exists():
        document_io.preflight_documents(documents_dir)

    # Phase 2a v2 (Option B): table extraction runs in PER-PDF SUBPROCESS
    # workers. Each worker exits before the next starts, so the kernel
    # reclaims all per-PDF memory unconditionally and the parent process
    # never accumulates extraction-loop heap. Empirically: peak worker
    # RSS ~39 MB on the largest PDF in the financial corpus, vs. ~640 MB
    # in-process before the OOM kill.
    if extract_tables and not dry_run:
        from backend.app.services import table_extract  # local import: pdfplumber heavy

        tables_dir = version_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{operation}] table extraction: scanning {documents_dir} for "
            f"PDFs (vision={'ON' if table_vision else 'OFF'}, "
            f"isolation=subprocess)"
        )
        manifests = await table_extract.extract_tables_for_folder_subprocess(
            documents_dir,
            run_cache_dir=tables_dir,
            use_vision=table_vision,
            concurrency=1,
        )
        total = sum(int(m.get("n_tables", 0) or 0) for m in manifests.values())
        cost = sum(float(m.get("cost_usd", 0.0) or 0.0) for m in manifests.values())
        n_cached = sum(1 for m in manifests.values() if m.get("source") == "cache")
        n_failed = sum(
            1 for m in manifests.values()
            if m.get("source") in ("spawn-failed", "worker-failed")
        )
        print(
            f"[{operation}] table extraction done: {total} tables across "
            f"{len(manifests)} PDF(s) ({n_cached} cache-hits, {n_failed} failed), "
            f"cost ${cost:.4f}"
        )
        _append_audit(
            audit_path, -1, "table_extract", None,
            {
                "n_pdfs": len(manifests),
                "n_tables": total,
                "n_cached_pdfs": n_cached,
                "n_failed_pdfs": n_failed,
                "cost_usd": round(cost, 5),
                "vision_enabled": table_vision,
                "isolation": "subprocess",
            },
        )
        # The manifests dict only carries lightweight per-PDF stats (the
        # full JSON-LD payloads stay on disk in tables_dir + user cache).
        # Drop the dict + force a GC pass before the LLM stages begin.
        del manifests
        import gc as _gc
        _gc.collect()
        print(f"[{operation}] table extraction memory released; proceeding to LLM stages")

    # Phase 2a follow-up: anchor-bucket grouping for every extracted table.
    # Runs ONLY when tables were just extracted (i.e. tables_dir exists).
    # Two-layer matching: reuse existing ontology classes first, then collapse
    # cross-table duplicates, then emit MATCH NOT FOUND proposals anchored under
    # the 6 buckets in domain_concepts.owl.
    table_mining_stage2: dict[str, Any] | None = None
    if extract_tables and not dry_run:
        from backend.app.services import table_ontology_mining

        def _audit_table_mining(task: str, payload: dict[str, Any]) -> None:
            _append_audit(audit_path, -1, task, None, payload)

        table_mining_stage2 = await table_ontology_mining.mine_table_concepts_async(
            tables_dir=tables_dir,
            loaded_ontology=loaded,
            router=router,
            cache_dir=tables_dir,
            audit_callback=_audit_table_mining,
        )

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
        extra_stage2_results=[table_mining_stage2] if table_mining_stage2 else None,
        single_pass_summaries=single_pass_summaries,
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
        base_iri = ontology_cfg.get("default_base_iri") or "https://veerla-ramrao.ai/ontology/merged#"
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

        # Layer H: post-Stage-4 misclassification audit. Scans newly-
        # created classes for label patterns that don't match their
        # current parent (corporate suffix not under Organization, event
        # keyword not under Event, person-shape under Person/Role,
        # owl:Thing parent) and asks gpt-4o-mini to KEEP / RE_HOME /
        # CONVERT_TO_INSTANCE each. Purely additive.
        if app_cfg.get("expansion", {}).get("classification_audit_enabled", True):
            audit_summary = await run_classification_audit_async(
                classes_dict=out_ontology.setdefault("classes_dict", {}),
                instances_dict=out_ontology.setdefault("instances_dict", {}),
                router=router,
                default_base_iri=base_iri,
                concurrency=int(app_cfg.get("expansion", {}).get("max_concurrent_llm_calls", 8)),
                use_cache=True,
            )
            _append_audit(audit_path, -1, "classification_audit", None, audit_summary)

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
    # Per-task spend breakdown. manifest/stats already carry the TOTAL, but a
    # total alone can't answer "where did the money go?" after the fact -- and at
    # tens of dollars a run, that is the question worth answering.
    versioning.write_cost_report(version_dir, router.cost_report())
    _print_cost_summary(router)
    return version_dir


def _print_cost_summary(router: LLMRouter) -> None:
    """Print the run's spend, biggest task first, so it is visible without
    digging through the run folder."""
    rep = router.cost_report()
    by_task = rep["by_task"]
    print(f"\n[cost] LLM total: ${rep['total_cost_usd']:.2f} "
          f"across {rep['total_calls']:,} calls "
          f"(prompt-cache hit {rep['prompt_cache']['cache_hit_rate'] * 100:.0f}%)")
    for task, row in list(by_task.items())[:8]:
        share = (row["cost_usd"] / rep["total_cost_usd"] * 100) if rep["total_cost_usd"] else 0.0
        print(f"[cost]   ${row['cost_usd']:>7.2f}  {share:>4.0f}%  {row['calls']:>4} call(s)  "
              f"{task} ({row['model']})")

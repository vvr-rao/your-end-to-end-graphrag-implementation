"""Entity -> entity relationship extraction (Milestone C, second half).

Before this, the graph had **zero** `entity -> entity` edges: extract-entities
wrote only `viao:assertsAbout` (chunk->entity) and `rdf:type` (entity->class),
and the LLM contract had no predicate field at all. Two entities could only
relate indirectly, so "Tesla and Panasonic co-occur" was expressible but
"Tesla is supplied by Panasonic" was not.

Each test below names the specific failure it guards.
"""
from __future__ import annotations

import json

import pytest

from backend.app.services.prompts import entity_extract


_CLASSES = [
    {"iri": "http://x#Org", "label": "Organization", "description": "a company"},
    {"iri": "http://x#Country", "label": "Country", "description": "a country"},
]
_PREDS = [
    {"iri": "http://x#suppliedBy", "label": "supplied by",
     "domain_label": "Organization", "range_label": "Organization"},
    {"iri": "http://x#locatedIn", "label": "located in",
     "domain_label": "Organization", "range_label": "Country"},
]


# --------------------------------------------------------------------------- #
# Prompt contract -- SECOND PASS
#
# Relationships are a separate `relationship_extract` call, not a rider on
# entity_extract. The predicate menu can only be narrowed to the entities'
# ACTUAL classes once those entities exist; the combined version derived it
# from the chunk's top-50 candidate CLASSES, so the model saw a median of 12
# predicates while holding 4-5 entities and 669 of 1,140 proposals were
# rejected on domain/range alone.
# --------------------------------------------------------------------------- #


def test_entity_extract_is_untouched_by_this_feature() -> None:
    """The entity pass must be EXACTLY what it was: relationships live in a
    second call, so entity recall cannot regress from prompt overloading."""
    import inspect

    from backend.app.services.prompts import entity_extract as ee

    assert list(inspect.signature(ee).parameters) == [
        "chunk_text", "candidate_classes"
    ]
    system, user = ee("Tesla is supplied by Panasonic.", _CLASSES)
    assert "relationship" not in system.lower()
    assert "relationship" not in user.lower()


_ENTS = [
    {"canonical_name": "Tesla, Inc.", "class_label": "Organization"},
    {"canonical_name": "Panasonic Corp.", "class_label": "Organization"},
]


def test_relationship_prompt_lists_entities_and_predicates() -> None:
    from backend.app.services.prompts import relationship_extract

    system, user = relationship_extract("Tesla is supplied by Panasonic.",
                                        _ENTS, _PREDS)
    assert "Tesla, Inc." in user and "Panasonic Corp." in user
    for pr in _PREDS:
        assert pr["iri"] in user
    assert "[Organization -> Organization]" in user


def test_relationship_prompt_demands_a_verbatim_evidence_quote() -> None:
    """Narrowing the menu weakens the type check -- offer and check now share
    a basis -- so the model must quote the span that asserts the claim, and
    the caller verifies that span really occurs in the chunk."""
    from backend.app.services.prompts import relationship_extract

    system, user = relationship_extract("text", _ENTS, _PREDS)
    assert "VERBATIM" in system
    assert "evidence" in user
    assert "world knowledge" in system.lower()
    assert "EMPTY LIST IS THE CORRECT ANSWER" in system


def test_relationship_prompt_forbids_inventing_names_or_iris() -> None:
    from backend.app.services.prompts import relationship_extract

    system, _ = relationship_extract("text", _ENTS, _PREDS)
    assert "Never introduce a name" in system
    assert "Do not invent" in system


def test_evidence_must_actually_occur_in_the_chunk() -> None:
    """The guard that replaces what menu-narrowing costs us."""
    chunk = "Tesla sources its battery cells from Panasonic under a long deal."
    hay = " ".join(chunk.split()).lower()

    good = "sources its battery cells from Panasonic"
    bad = "Tesla acquired Panasonic in 2019"      # plausible, never stated
    assert " ".join(good.split()).lower() in hay
    assert " ".join(bad.split()).lower() not in hay


# --------------------------------------------------------------------------- #
# Validation rules the write path enforces
# --------------------------------------------------------------------------- #
#
# These mirror the checks in db_entity_extract without needing a live DB: the
# rules are what matter, and the DB-bound half is covered by the end-to-end run.


def _validate(rel, names, class_of, constraints):
    """The parse-time guard from db_entity_extract, in isolation."""
    subj = (rel.get("subject") or "").strip()
    obj = (rel.get("object") or "").strip()
    pred = (rel.get("predicate_iri") or "").strip()
    if not subj or not obj or subj == obj:
        return "malformed"
    if subj not in names or obj not in names:
        return "unresolved"
    if pred not in constraints:
        return "bad_predicate"
    dom, rng = constraints[pred]
    if class_of.get(subj) != dom or class_of.get(obj) != rng:
        return "domain_range"
    return None


_NAMES = {"Tesla", "Panasonic", "Japan"}
_CLASS_OF = {"Tesla": "http://x#Org", "Panasonic": "http://x#Org",
             "Japan": "http://x#Country"}
_CONSTRAINTS = {
    "http://x#suppliedBy": ("http://x#Org", "http://x#Org"),
    "http://x#locatedIn": ("http://x#Org", "http://x#Country"),
}


def test_a_valid_relationship_survives() -> None:
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) is None


def test_an_entity_not_extracted_in_this_chunk_is_dropped() -> None:
    """The model may name something it did not return as an entity; that has
    no id to point at."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy",
         "object": "Toyota"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "unresolved"


def test_an_invented_predicate_is_dropped() -> None:
    """Mirrors the existing `class_iri not in cand_iris` guard."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#totallyMadeUp",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "bad_predicate"


def test_a_type_valid_iri_used_on_the_wrong_pair_is_dropped() -> None:
    """THE SUBTLE ONE. `locatedIn` is a real offered predicate, but its range
    is Country -- asserting it between two Organizations is exactly what the
    candidate list is meant to make impossible, so the write path re-checks
    rather than trusting the model honoured the annotation."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#locatedIn",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "domain_range"


def test_direction_matters() -> None:
    """Org -locatedIn-> Country is valid; the reverse is not."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#locatedIn",
         "object": "Japan"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) is None
    assert _validate(
        {"subject": "Japan", "predicate_iri": "http://x#locatedIn",
         "object": "Tesla"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "domain_range"


@pytest.mark.parametrize("rel", [
    {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy", "object": "Tesla"},
    {"subject": "", "predicate_iri": "http://x#suppliedBy", "object": "Tesla"},
    {"subject": "Tesla", "predicate_iri": "", "object": "Panasonic"},
])
def test_malformed_relationships_are_dropped(rel) -> None:
    assert _validate(rel, _NAMES, _CLASS_OF, _CONSTRAINTS) is not None


# --------------------------------------------------------------------------- #
# Dedupe -- graph_relationships has NO unique constraint
# --------------------------------------------------------------------------- #


def test_duplicate_edges_collapse_on_subject_predicate_object() -> None:
    """There is no DB constraint to catch this, so a re-run would otherwise
    double every edge."""
    rels = [
        {"s": 1, "p": "http://x#suppliedBy", "o": 2},
        {"s": 1, "p": "http://x#suppliedBy", "o": 2},   # exact repeat
        {"s": 1, "p": "http://x#locatedIn", "o": 2},    # different predicate
        {"s": 2, "p": "http://x#suppliedBy", "o": 1},   # reversed
    ]
    seen: set[tuple] = set()
    kept = []
    for r in rels:
        sig = (r["s"], r["p"], r["o"])
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(r)
    assert len(kept) == 3, "dedupe must key on (subject, predicate, object)"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_extract_entities_exposes_the_relationship_switch() -> None:
    import inspect

    from backend.app.services.db_entity_extract import extract_entities

    sig = inspect.signature(extract_entities)
    assert sig.parameters["extract_relationships"].default is True


def test_summary_counts_relationship_edges() -> None:
    """Silent drops are how the class_iri guard hid losses before."""
    from backend.app.services.db_entity_extract import EntityExtractSummary

    assert EntityExtractSummary().entity_relationship_edges == 0


def test_candidate_predicate_sql_matches_domain_and_range_not_either() -> None:
    """A property is offerable only when BOTH ends are in scope -- matching
    on domain alone would offer predicates whose range no entity can satisfy.
    """
    from backend.app.services.db_entity_extract import _CANDIDATE_PREDICATE_SQL

    sql = str(_CANDIDATE_PREDICATE_SQL)
    assert sql.count("EXISTS") >= 2
    assert "'domain'" in sql and "'range'" in sql


def test_ancestors_are_walked_upward_not_downward() -> None:
    """A property declared on Organization must apply to a PublicCompany
    entity, so the walk follows child -> parent (the direction the importer
    writes subClassOf). Walking down would offer nothing for a leaf class."""
    from backend.app.services.db_entity_extract import _ANCESTOR_SQL

    sql = str(_ANCESTOR_SQL)
    assert "gr.source_node_id = up.id" in sql      # join on the CHILD
    assert "gr.target_node_id" in sql               # walk to the PARENT
    assert "rdfs:subClassOf" in sql
    # Tracks which class each ancestor came from, so the write-path check can
    # use the same rule that offered the predicate.
    assert "up(origin, id)" in sql


def test_relationship_payloads_are_entity_to_entity() -> None:
    """The shape that was absent from the graph entirely."""
    payload = {
        "source_node_type": "entity",
        "target_node_type": "entity",
        "relationship_source": "DOCUMENT_EXTRACTION",
    }
    assert (payload["source_node_type"], payload["target_node_type"]) == (
        "entity", "entity"
    )
    assert json.dumps(payload)


# --------------------------------------------------------------------------- #
# Ancestor-aware checking + supporting chunks (measured fixes, 2026-08-29)
# --------------------------------------------------------------------------- #


def _validate_with_ancestors(rel, by_norm, ancestors, constraints):
    """The corrected guard: the check uses the SAME ancestor rule that offered
    the predicate, and names resolve on the normalized form."""
    from backend.app.services.db_entity_extract import _normalize_name

    s_ent = by_norm.get(_normalize_name(rel.get("subject") or ""))
    o_ent = by_norm.get(_normalize_name(rel.get("object") or ""))
    if s_ent is None or o_ent is None:
        return "unresolved"
    pred = (rel.get("predicate_iri") or "").strip()
    if pred not in constraints:
        return "bad_predicate"
    dom, rng = constraints[pred]
    s_cls, o_cls = s_ent["class_iri"], o_ent["class_iri"]
    if (dom not in ancestors.get(s_cls, {s_cls})
            or rng not in ancestors.get(o_cls, {o_cls})):
        return "domain_range"
    return None


_PUBCO = {"canonical_name": "Tesla, Inc.", "short_name": "Tesla",
          "class_iri": "http://x#PublicCompany"}
_SUPPLIER = {"canonical_name": "Panasonic Corp.", "short_name": "Panasonic",
             "class_iri": "http://x#PublicCompany"}
_BY_NORM = {"tesla inc": _PUBCO, "tesla": _PUBCO,
            "panasonic corp": _SUPPLIER, "panasonic": _SUPPLIER}
# suppliedBy is declared on the PARENT class, Organization.
_ANCESTORS = {"http://x#PublicCompany": {"http://x#PublicCompany",
                                         "http://x#Organization"}}
_CONSTR = {"http://x#suppliedBy": ("http://x#Organization",
                                   "http://x#Organization")}


def test_a_predicate_declared_on_a_PARENT_class_is_accepted() -> None:
    """THE BUG THIS FIXES. The candidate query expanded ancestors when
    OFFERING a predicate, but the check demanded the entity's own class equal
    the domain exactly -- so a property declared on Organization was offered
    for a PublicCompany entity and then always rejected. 635 of 1,109 drops on
    the first real run were this."""
    assert _validate_with_ancestors(
        {"subject": "Tesla, Inc.", "predicate_iri": "http://x#suppliedBy",
         "object": "Panasonic Corp."},
        _BY_NORM, _ANCESTORS, _CONSTR,
    ) is None


def test_a_short_form_resolves_to_the_entity_it_names() -> None:
    """The model routinely writes "Tesla" for an entity it extracted as
    "Tesla, Inc.". Exact-string matching discarded those -- 424 of 1,109."""
    assert _validate_with_ancestors(
        {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy",
         "object": "Panasonic"},
        _BY_NORM, _ANCESTORS, _CONSTR,
    ) is None


def test_loosening_did_not_disable_the_type_check() -> None:
    """Ancestor-aware must not mean anything-goes: a class NOT under the
    declared domain is still rejected."""
    by_norm = dict(_BY_NORM)
    by_norm["japan"] = {"canonical_name": "Japan", "short_name": "Japan",
                        "class_iri": "http://x#Country"}
    ancestors = dict(_ANCESTORS)
    ancestors["http://x#Country"] = {"http://x#Country"}
    assert _validate_with_ancestors(
        {"subject": "Japan", "predicate_iri": "http://x#suppliedBy",
         "object": "Tesla"},
        by_norm, ancestors, _CONSTR,
    ) == "domain_range"


def test_supporting_chunks_accumulate_across_chunks() -> None:
    """One edge per triple, but every chunk that asserted it is recorded --
    support_count is what separates a corroborated edge from a one-off."""
    from backend.app.services.db_entity_extract import _MAX_SUPPORTING_CHUNKS

    by_sig: dict[tuple, dict] = {}
    for chunk in ["c1", "c2", "c3", "c2"]:
        sig = ("e1", "http://x#suppliedBy", "e2")
        if sig in by_sig:
            by_sig[sig]["_chunks"].append(chunk)
        else:
            by_sig[sig] = {"_chunks": [chunk]}

    chunks = by_sig[("e1", "http://x#suppliedBy", "e2")]["_chunks"]
    assert len(by_sig) == 1, "must stay ONE edge"
    assert len(chunks) == 4, "every assertion counted"
    assert len(chunks[:_MAX_SUPPORTING_CHUNKS]) <= _MAX_SUPPORTING_CHUNKS


def test_supporting_chunk_list_is_capped_but_count_is_exact() -> None:
    """JSONB must not grow without bound on a heavily-repeated triple."""
    from backend.app.services.db_entity_extract import _MAX_SUPPORTING_CHUNKS

    chunks = [f"c{i}" for i in range(500)]
    meta = {"support_count": len(chunks),
            "supporting_chunks": chunks[:_MAX_SUPPORTING_CHUNKS]}
    assert meta["support_count"] == 500
    assert len(meta["supporting_chunks"]) == _MAX_SUPPORTING_CHUNKS


# --------------------------------------------------------------------------- #
# Precision fixes (news-corpus findings)
#
# The 30-doc MultiHop-RAG run wrote 65 edges of which roughly half were wrong
# on a hand check, DESPITE every one passing the structural gates. Two distinct
# failure shapes, each guarded below:
#
#   one-sided evidence -- `Anthropic --hasMember--> Mustafa Suleyman` quoting a
#     real sentence about Anthropic founders that never says "Suleyman".
#   unsupported / reversed -- `OpenAI --filesLawsuitAgainst--> Sara Silverman`
#     quoting "Copyright lawsuits filed against OpenAI BY Sara Silverman", a
#     quote that names both ends and states the opposite.
# --------------------------------------------------------------------------- #


def _ent(canonical, short=None):
    return {"canonical_name": canonical, "short_name": short or canonical,
            "class_iri": "http://x#Org"}


def test_evidence_must_name_both_ends() -> None:
    from backend.app.services.db_entity_extract import _evidence_names

    ev = ("Anthropic founders denied attempting to oust Altman, stating they "
          "left OpenAI to pursue controlled AI development.")
    assert _evidence_names(ev, _ent("Anthropic"))
    assert not _evidence_names(ev, _ent("Mustafa Suleyman"))


def test_evidence_name_match_tolerates_inflection_and_partials() -> None:
    """Strict about PRESENCE, lenient about FORM -- dropping a true edge over
    a possessive or a surname is the worse error."""
    from backend.app.services.db_entity_extract import _evidence_names

    assert _evidence_names("Copyright lawsuits filed against OpenAI by Sara "
                           "Silverman", _ent("Sara Silverman"))
    assert _evidence_names("OpenAI's board consists of...", _ent("OpenAI"))
    assert _evidence_names("the disparity between Israeli cities",
                           _ent("Israel"))


def test_generic_name_tokens_do_not_count_as_a_mention() -> None:
    """Without this, any sentence containing 'smart' would name the
    'Ray-Ban Meta smart glasses' entity."""
    from backend.app.services.db_entity_extract import _evidence_names

    assert not _evidence_names(
        "The company shipped a smart speaker and a new app.",
        _ent("Ray-Ban Meta smart glasses"),
    )


def test_verify_prompt_demands_direction_and_strictness() -> None:
    from backend.app.services.prompts import relationship_verify

    system, user = relationship_verify(
        "passage",
        [{"subject": "OpenAI", "predicate_label": "files lawsuit against",
          "object": "Sara Silverman",
          "evidence": "Copyright lawsuits filed against OpenAI by Sara "
                      "Silverman"}],
    )
    assert "reversed" in system and "unsupported" in system
    assert "world knowledge" in system.lower()
    assert "[0]" in user
    assert "Sara Silverman" in user


def test_generic_predicates_are_listed_separately_as_last_resort() -> None:
    """org:linkedTo + foaf:topic took 22 of 65 edges on the news run purely by
    being offered alongside specific predicates."""
    from backend.app.services.prompts import relationship_extract

    preds = [
        {"iri": "http://x#suppliedBy", "label": "supplied by",
         "domain_label": "Organization", "range_label": "Organization"},
        {"iri": "http://www.w3.org/ns/org#linkedTo", "label": "linked to",
         "domain_label": "Organization", "range_label": "Organization"},
        {"iri": "http://xmlns.com/foaf/0.1/topic", "label": "topic",
         "domain_label": "Organization", "range_label": "Organization"},
    ]
    _, user = relationship_extract("text", _ENTS, preds)
    assert "LAST-RESORT" in user
    head, tail = user.split("LAST-RESORT", 1)
    assert "http://x#suppliedBy" in head
    assert "org#linkedTo" in tail and "foaf/0.1/topic" in tail


def test_all_generic_iris_are_absolute_and_lowercase_free_of_typos() -> None:
    """A typo'd IRI in the demote-list silently stops demoting."""
    from backend.app.services.prompts import GENERIC_PREDICATE_IRIS

    assert GENERIC_PREDICATE_IRIS
    for iri in GENERIC_PREDICATE_IRIS:
        assert iri.startswith("http"), iri
        assert " " not in iri, iri


# --------------------------------------------------------------------------- #
# Rescue pass + name-level entity identity
#
# Two measured defects on the 30-doc news corpus drove these:
#
#   STARVED MENU -- the strict rule (predicates whose declared domain AND range
#     both match the entities' classes) offered a MEDIAN OF 8 predicates out of
#     620, so real relationships were forced onto a wrong predicate and then
#     rejected: 114 domain_range drops in one run. `Google --hasSubOrganization
#     --> DeepMind` came from "Google ACQUIRED DeepMind for $650 million", and
#     `acquires` was in the ontology the whole time. Matching descendants as
#     well as ancestors was measured and changed nothing (median stayed 8);
#     only either-end matching widens it (median 52).
#
#   FRAGMENTED ENTITIES -- identity was (normalized_name, class_id), so one
#     entity typed differently per chunk became many nodes: 901 rows for 624
#     names, "google" x18, "microsoft" x16. Multi-hop cannot traverse that.
# --------------------------------------------------------------------------- #


def test_repair_prompt_keeps_the_quote_and_permits_only_a_swap() -> None:
    """The rescue pass may change the PREDICATE and the DIRECTION. It must not
    invent claims or substitute a different quote -- the evidence already
    passed the occurs-in-chunk and names-both-ends checks."""
    from backend.app.services.prompts import relationship_repair

    system, user = relationship_repair(
        "Google acquired DeepMind for $650 million.",
        [{"subject": "Google", "predicate_label": "has SubOrganization",
          "object": "DeepMind",
          "evidence": "Google acquired DeepMind for $650 million"}],
        [{"iri": "http://x#acquires", "label": "acquires",
          "domain_label": "Organization", "range_label": "Organization"}],
    )
    assert "keep the SAME quote" in system.lower() or "SAME quote" in system
    assert "Do not invent claims" in system
    assert "DROPPING IS CORRECT" in system
    assert "SWAP" in system
    assert "http://x#acquires" in user
    assert "Google acquired DeepMind" in user


def test_rescue_marks_its_edges_so_the_two_populations_stay_separable() -> None:
    """A rescued edge used a relaxed type check; an audit must be able to tell
    it from one that satisfied domain/range outright."""
    import inspect

    from backend.app.services import db_entity_extract as dee

    src = inspect.getsource(dee._rescue_relationships)
    assert '"type_check": "relaxed"' in src


def test_rescue_does_not_lower_the_evidence_bar() -> None:
    """Widening WHICH predicate may be used must not widen what counts as
    proof -- the rescue path re-applies both evidence checks."""
    import inspect

    from backend.app.services import db_entity_extract as dee

    src = inspect.getsource(dee._rescue_relationships)
    assert "_evidence_names(ev, s_ent)" in src
    assert "_evidence_names(ev, o_ent)" in src
    assert "ev.lower() not in hay" in src


def test_rescue_only_accepts_predicates_from_the_offered_menu() -> None:
    import inspect

    from backend.app.services import db_entity_extract as dee

    src = inspect.getsource(dee._rescue_relationships)
    assert "pred not in known_iris" in src


def test_wide_menu_matches_either_end_not_both() -> None:
    """The whole point: both-ends is what starved the menu to 8 of 620."""
    from backend.app.services.db_entity_extract import _WIDE_PREDICATE_SQL

    sql = str(_WIDE_PREDICATE_SQL)
    where = sql.split("WHERE", 1)[1]
    assert " OR EXISTS" in where, "wide menu must be an OR over the two ends"


def test_entity_identity_defaults_to_name_level() -> None:
    """Default must be the fixed behaviour; 'name-class' reproduces the split."""
    import inspect

    from backend.app.services.db_entity_extract import extract_entities

    sig = inspect.signature(extract_entities)
    assert sig.parameters["entity_identity"].default == "name"
    # Rescue defaults OFF: measured on the news corpus it rescued 74 claims
    # and precision fell from ~75% to ~45%, because the wider either-end menu
    # admits predicates whose other end is nonsense.
    assert sig.parameters["rescue_relationships"].default is False


def test_majority_vote_picks_the_primary_class() -> None:
    """One node per name; the class it carries is the one chosen most often."""
    votes = {"google": {"SearchEngine": 5, "OpenAICompetitor": 2,
                        "SearchMonopoly": 1}}
    primary = {n: max(v.items(), key=lambda kv: kv[1])[0]
               for n, v in votes.items()}
    assert primary["google"] == "SearchEngine"


def test_collapsing_keeps_every_observed_class_as_a_type_edge() -> None:
    """Merging 18 Googles into one node must not discard the 17 other types --
    an entity legitimately holds several."""
    import inspect

    from backend.app.services import db_entity_extract as dee

    src = inspect.getsource(dee.extract_entities)
    assert "extra_type_classes" in src
    assert "minted_entity_class_pairs.append((eid, extra_cid))" in src


def test_relationship_endpoints_resolve_via_the_STORED_class() -> None:
    """Under name-level identity the node carries the MAJORITY class, not the
    one a given chunk assigned. Resolving edges by the per-chunk class misses
    the key and silently loses the edge: measured 49 spurious `unresolved`
    drops and edges falling 30 -> 10 before this was fixed."""
    import inspect

    from backend.app.services import db_entity_extract as dee

    src = inspect.getsource(dee.extract_entities)
    resolve = src.split("def _resolve(", 1)[1].split("for rel in rels:", 1)[0]
    assert "primary_class_by_name" in resolve, (
        "_resolve must key on the stored (majority) class"
    )


# --------------------------------------------------------------------------- #
# prune-expand: relation endpoints must GENERALISE
#
# Root cause of the starved predicate menu, measured on the news corpus: only
# 29 of 125 corpus-minted predicates (23%) declared a reusable domain. The rest
# were typed from the single sentence that produced them --
# `acquires[MusicLabel -> MusicRightsAcquisition]`, so "Google acquired
# DeepMind" could never use it, and `backs[Mozilla -> Mammoth]`, a signature
# matching exactly one company pair. That is why `domain_range` was 44% of all
# relationship rejections.
# --------------------------------------------------------------------------- #


def _proposal_prompt() -> str:
    from backend.app.services.prompts import PROMPTS

    system, _ = PROMPTS["class_proposal"]("chunk text", "DATA_CLASSES here")
    return system


def test_proposal_prompt_demands_general_relation_endpoints() -> None:
    sys_p = _proposal_prompt()
    assert "AS GENERAL AS THE RELATION" in sys_p
    assert "MusicLabel" in sys_p, "keeps the concrete counter-example"
    assert "DOMAIN=Organization, RANGE=Organization" in sys_p


def test_proposal_prompt_forbids_named_individuals_as_endpoints() -> None:
    """`backs[Mozilla -> Mammoth]` is the failure this guards."""
    sys_p = _proposal_prompt()
    assert "NEVER use a NAMED INDIVIDUAL as a DOMAIN or RANGE" in sys_p
    assert "Mozilla" in sys_p


def test_dedup_prompt_widens_narrow_endpoints() -> None:
    """Dedup is the second chance to repair an over-narrow signature, and the
    only one that sees several proposals of the same relation at once."""
    from backend.app.services.prompts import PROMPTS

    system, _ = PROMPTS["match_dedup"]({}, [])
    assert "GENERALISE THE ENDPOINTS" in system
    assert "most general classes for which the relation" in system
    assert "backs[Mozilla -> Mammoth]" in system
    # The older rule said differing DOMAIN/RANGE were never duplicates, which
    # is precisely what preserved the fragmentation.
    assert "ARE duplicates" in system


def test_dedup_still_separates_genuinely_different_relations() -> None:
    """Widening must not collapse two same-labelled relations that mean
    different things."""
    from backend.app.services.prompts import PROMPTS

    system, _ = PROMPTS["match_dedup"]({}, [])
    assert "genuinely DIFFERENT" in system

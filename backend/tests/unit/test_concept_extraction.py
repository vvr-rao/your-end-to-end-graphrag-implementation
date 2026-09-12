"""Concepts as entities: the widened `entity_extract` and the concept pass.

Why this exists. Measured on a 40-document news build, 995 of 1,008 entities
had proper-noun names and 248 of 328 prune-expand classes held nothing at all.
`EconomicConcept`, `Metric`, `FinancialObservation` and `Process` held ZERO --
not because they were missing from the candidate menu (`Industry` ranked 4th
and `EconomicConcept` 8th on a shipping chunk, and both also arrive via
ancestor closure: `TechnologyConcept` has 75 descendant classes) but because
`entity_extract` said "Only PROPER NOUN entities ... Skip generic terms". The
instances of a concept class are common nouns, so those classes were
unreachable by construction.

Two changes are pinned here:

  1. `entity_extract` may now return concepts, and `entity_validate` was
     realigned so the reviewer does not delete what the extractor was just
     told to find. Both halves matter: leaving the reviewer saying "Only
     PROPER NOUNS" would have made the loop undo the fix.
  2. `concept_extract`, a second pass with one job. On a shipping chunk the
     widened single call found 1 concept where the passage developed at least
     five; the dedicated pass found them.

The menu narrowing has its own reason. Asked politely to use only classes
denoting an abstract kind, the model does not comply on a concept-POOR
document: on a product-deals article it typed "laptop" -> Laptop, "speaker" ->
SpeakerDevice, "battery life" -> FitnessDevice. Restricting the MENU to the
configured concept roots (closed over subclasses) cut that from 28 junk
concepts to 1, while the shipping article kept 10 real ones.
"""

from __future__ import annotations

import pytest
import yaml

from backend.app.services.db_entity_extract import (
    ABSTAIN_SENTINEL,
    _dedup_key,
    _filter_entities,
)
from backend.app.services.prompts import (
    PROMPTS,
    concept_extract,
    entity_extract,
    entity_validate,
)

_CANDIDATES = [
    {"iri": "http://ex#EconomicConcept", "label": "EconomicConcept",
     "description": "An idea or mechanism in economics."},
    {"iri": "http://ex#Organization", "label": "Organization",
     "description": "A group of people working together."},
]
_CAND_IRIS = {c["iri"] for c in _CANDIDATES}


def _drops() -> dict[str, int]:
    return {"off_menu_iri": 0, "no_name": 0, "abstained": 0,
            "reviewer_removed": 0, "no_candidates": 0}


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #


def test_concept_extract_is_registered() -> None:
    assert PROMPTS["concept_extract"] is concept_extract


def test_concept_pass_sees_the_same_menu_renderer() -> None:
    """The concept prompt must render its menu the way every other pass does,
    or it proposes IRIs the caller then discards as off-menu."""
    _c_sys, c_user = concept_extract("text", _CANDIDATES)
    for cand in _CANDIDATES:
        assert cand["iri"] in c_user
        assert cand["label"] in c_user


def test_concept_prompt_states_the_load_bearing_rules() -> None:
    c_sys, _ = concept_extract("text", _CANDIDATES)
    low = c_sys.lower()
    # It must not re-extract the proper nouns another pass already has.
    assert "do not repeat" in low
    # An empty answer has to be framed as correct, or a concept-free passage
    # gets a manufactured list -- the doc_597 failure.
    assert "empty" in low and "correct answer" in low
    # Abstract-kind-only is stated even though the menu also enforces it.
    assert "abstract" in low
    # A cap, so one chunk cannot flood the graph.
    assert "0 to 5" in low


def test_entity_extract_no_longer_forbids_common_nouns() -> None:
    """The single rule that made every concept class unreachable."""
    sys_p, _ = entity_extract("text", _CANDIDATES)
    assert "Only PROPER NOUN entities" not in sys_p
    assert "concept" in sys_p.lower()


def test_reviewer_does_not_delete_what_the_extractor_now_finds() -> None:
    """`entity_validate` ran "Only PROPER NOUNS" and `not_an_entity` meant
    "not a proper-noun named entity". Left alone, the review loop would have
    removed every concept the extractor added."""
    v_sys, _ = entity_validate("text", _CANDIDATES, [
        {"canonical_name": "freight rates", "class_iri": "http://ex#EconomicConcept"},
    ])
    assert "Only PROPER NOUNS actually" not in v_sys
    assert "not a proper-noun named entity" not in v_sys
    assert "concept" in v_sys.lower()


def test_entity_extract_still_rejects_empty_referring_phrases() -> None:
    """Admitting concepts must not admit "the manufacturer" -- that is the
    distinction the whole change rests on."""
    sys_p, _ = entity_extract("text", _CANDIDATES)
    assert "the manufacturer" in sys_p
    assert "the report" in sys_p


# --------------------------------------------------------------------------- #
# Caller-side handling
# --------------------------------------------------------------------------- #


def test_concept_output_flows_through_the_same_filter() -> None:
    """`concept_extract` returns the `entity_extract` shape on purpose, so the
    caller reuses `_filter_entities` and the drop accounting stays one set of
    counters."""
    drops = _drops()
    kept = _filter_entities(
        [{"canonical_name": "freight rates", "short_name": "freight rates",
          "class_iri": "http://ex#EconomicConcept", "confidence": 0.9}],
        _CAND_IRIS, drops, [], 200,
    )
    assert [e["canonical_name"] for e in kept] == ["freight rates"]
    assert sum(drops.values()) == 0


def test_concept_off_the_narrowed_menu_is_dropped_and_counted() -> None:
    """The narrowed menu is enforced by validating against ITS iris, not the
    full candidate set -- otherwise "laptop" -> Laptop is admitted after all."""
    drops = _drops()
    narrowed = {"http://ex#EconomicConcept"}
    kept = _filter_entities(
        [{"canonical_name": "laptop", "short_name": "laptop",
          "class_iri": "http://ex#Organization", "confidence": 0.9}],
        narrowed, drops, [], 200,
    )
    assert kept == []
    assert drops["off_menu_iri"] == 1


def test_duplicate_between_passes_uses_the_db_dedup_key() -> None:
    """Both passes see the concept classes, so the same concept can come back
    twice. Dedup must use the key the DB upsert uses, or the two rows merge
    unpredictably later."""
    assert _dedup_key("Freight Rates") == _dedup_key("freight rates")


# --------------------------------------------------------------------------- #
# Config contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", ["config/config.example.yaml"])
def test_concept_pass_ships_off_by_default(path: str) -> None:
    cfg = yaml.safe_load(open(path))["extraction"]
    assert cfg["concept_pass"] is False


@pytest.mark.parametrize("path", ["config/config.example.yaml"])
def test_concept_roots_exclude_the_other_passes_territory(path: str) -> None:
    """Event holds named events ("Super Bowl") and belongs to the entity pass;
    TimePeriod belongs to enrich-time; FinancialTable to the table pass.
    Including them would have the concept pass compete for their rows."""
    roots = yaml.safe_load(open(path))["extraction"]["concept_class_roots"]
    assert "EconomicConcept" in roots
    assert "Event" not in roots
    assert "TimePeriod" not in roots
    assert "FinancialTable" not in roots


@pytest.mark.parametrize("preset", [
    "config/models.example.yaml",
    "config/models.openai.example.yaml",
    "config/models.anthropic.example.yaml",
])
def test_concept_extract_task_exists_in_every_preset(preset: str) -> None:
    """Missing in one preset means that provider mode degrades to no concept
    pass at all -- silently, since the caller warns once and continues."""
    tasks = yaml.safe_load(open(preset))["tasks"]
    assert "concept_extract" in tasks
    spec = tasks["concept_extract"]
    assert spec["provider"] and spec["model"]


@pytest.mark.parametrize("preset", [
    "config/models.example.yaml",
    "config/models.openai.example.yaml",
    "config/models.anthropic.example.yaml",
])
def test_concept_extract_matches_the_entity_extractor(preset: str) -> None:
    """The concept pass reads the same chunk against the same kind of menu as
    `entity_extract`, so the two are kept on one model rather than split
    across tiers.

    This started as "stays on the cheap tier", asserted as equality with
    entity_extract. Both halves were wrong for their own reasons. The equality
    was over-coupled -- it broke the moment entity_extract moved for reasons
    unrelated to concepts. And "cheap tier" assumed gpt-4o-mini was adequate
    here; its measured instability on the entity pass (2/5 on Ramon Fernandez,
    0 clean validation rounds across 3 runs) is a property of the model, not
    of that one prompt.

    What is worth pinning is that the two do not silently DIVERGE, leaving the
    concept pass on a model nobody chose for it.
    """
    tasks = yaml.safe_load(open(preset))["tasks"]
    assert tasks["concept_extract"]["model"] == tasks["entity_extract"]["model"], (
        f"{preset}: concept_extract ({tasks['concept_extract']['model']}) has "
        f"drifted from entity_extract ({tasks['entity_extract']['model']})"
    )


# --------------------------------------------------------------------------- #
# Off-menu IRI recovery
# --------------------------------------------------------------------------- #


_MENU_IRIS = {
    "https://x/merged#brandnamedrug",
    "https://x/merged#lipaseinhibitor",
    "http://xmlns.com/foaf/0.1/Person",
}
_MENU_LABELS = {
    "brandnamedrug": "https://x/merged#brandnamedrug",
    "lipaseinhibitor": "https://x/merged#lipaseinhibitor",
    "Person": "http://xmlns.com/foaf/0.1/Person",
}


def _fe(entities, drops, **kw):
    return _filter_entities(entities, _MENU_IRIS, drops, [], 200, **kw)


def test_label_written_into_the_iri_field_is_recovered() -> None:
    """Measured on the pharma corpus: 10 of 11 off-menu values were the class
    LABEL, not an invented class -- "brandnamedrug", "lipaseinhibitor",
    "semaglutide", each of them on the menu. Ozempic, Wegovy and Orlistat were
    dropped over a serialisation slip."""
    drops = _drops()
    kept = _fe(
        [{"canonical_name": "Ozempic", "class_iri": "brandnamedrug"},
         {"canonical_name": "Orlistat", "class_iri": "lipaseinhibitor"}],
        drops, cand_labels=_MENU_LABELS,
    )
    assert [e["canonical_name"] for e in kept] == ["Ozempic", "Orlistat"]
    assert kept[0]["class_iri"] == "https://x/merged#brandnamedrug"
    assert drops["off_menu_iri"] == 0
    assert drops["recovered_label_iri"] == 2


def test_a_genuinely_invented_class_is_still_dropped() -> None:
    """The recovery must not become a way for hallucinated classes to land.
    `prescriptiondrug` carries the right namespace and is not a class."""
    drops = _drops()
    kept = _fe(
        [{"canonical_name": "topiramate",
          "class_iri": "https://x/merged#prescriptiondrug"}],
        drops, cand_labels=_MENU_LABELS,
    )
    assert kept == []
    assert drops["off_menu_iri"] == 1
    assert drops.get("recovered_label_iri", 0) == 0


def test_recovery_works_from_the_iri_local_name_without_labels() -> None:
    """Callers that pass no label map still recover, via the IRI local-name --
    which is what the model usually echoes."""
    drops = _drops()
    kept = _fe([{"canonical_name": "Ozempic", "class_iri": "brandnamedrug"}], drops)
    assert kept and kept[0]["class_iri"] == "https://x/merged#brandnamedrug"
    assert drops["recovered_label_iri"] == 1


def test_an_ambiguous_local_name_is_never_resolved() -> None:
    """`foaf#Person` and `org#Person` share a local name. Resolving by set
    iteration order would make extraction non-deterministic across runs, so an
    ambiguous key resolves to nothing and the entity is dropped as before."""
    iris = {"http://xmlns.com/foaf/0.1/Person", "http://www.w3.org/ns/org#Person"}
    drops = _drops()
    kept = _filter_entities(
        [{"canonical_name": "Jane", "class_iri": "Person"}], iris, drops, [], 200,
    )
    assert kept == []
    assert drops["off_menu_iri"] == 1


def test_recovery_is_counted_apart_from_drops() -> None:
    """A recovery is an entity SAVED. Folding it into the drop tally would
    report the fix as damage."""
    drops = _drops()
    _fe([{"canonical_name": "Ozempic", "class_iri": "brandnamedrug"}],
        drops, cand_labels=_MENU_LABELS)
    assert "recovered_label_iri" in drops
    assert drops["off_menu_iri"] == 0


def test_abstain_still_wins_over_recovery() -> None:
    """NONE_OF_THESE must never be coerced into a class by the resolver."""
    drops = _drops()
    abst: list = []
    kept = _filter_entities(
        [{"canonical_name": "Xyz", "class_iri": ABSTAIN_SENTINEL,
          "proposed_type": "Enzyme"}],
        _MENU_IRIS, drops, abst, 200, cand_labels=_MENU_LABELS,
    )
    assert kept == []
    assert drops["abstained"] == 1
    assert drops.get("recovered_label_iri", 0) == 0


# --------------------------------------------------------------------------- #
# Abstention recovery via proposed_type
# --------------------------------------------------------------------------- #


_TYPE_INDEX = {
    "coffeemaker": "https://x/merged#CoffeeMaker",
    "vacuumcleaner": "https://x/merged#VacuumCleaner",
}


def test_abstention_recovered_when_the_proposed_class_exists() -> None:
    """Measured on 20 news chunks: of 29 distinct `proposed_type` values, 15
    (51%) named a class ALREADY in the ontology -- CoffeeMaker, VacuumCleaner,
    MeshRouter, VideoGame -- which simply never reached that chunk's top-K
    menu. The model could not pick what it was not shown, so it abstained and
    a real entity was discarded over a ranking miss.
    """
    drops = _drops()
    abst: list = []
    kept = _filter_entities(
        [{"canonical_name": "Keurig K-Express", "class_iri": ABSTAIN_SENTINEL,
          "proposed_type": "coffee maker"}],
        {"https://x/merged#Person"}, drops, abst, 200, type_index=_TYPE_INDEX,
    )
    assert [e["class_iri"] for e in kept] == ["https://x/merged#CoffeeMaker"]
    assert drops["abstained"] == 0
    assert drops["recovered_proposed_type"] == 1
    assert abst == []


def test_a_real_ontology_gap_still_abstains() -> None:
    """The recovery must not mask the signal it is filtering. `Processor` and
    `PoliticalParty` genuinely do not exist, and those abstentions are what a
    prune-expand run should act on."""
    drops = _drops()
    abst: list = []
    kept = _filter_entities(
        [{"canonical_name": "AMD Ryzen 7", "class_iri": ABSTAIN_SENTINEL,
          "proposed_type": "Processor"}],
        {"https://x/merged#Person"}, drops, abst, 200, type_index=_TYPE_INDEX,
    )
    assert kept == []
    assert drops["abstained"] == 1
    assert drops.get("recovered_proposed_type", 0) == 0
    assert abst[0]["proposed_type"] == "Processor"


def test_abstention_with_no_proposed_type_is_untouched() -> None:
    drops = _drops()
    abst: list = []
    kept = _filter_entities(
        [{"canonical_name": "Thing", "class_iri": ABSTAIN_SENTINEL}],
        {"https://x/merged#Person"}, drops, abst, 200, type_index=_TYPE_INDEX,
    )
    assert kept == [] and drops["abstained"] == 1


def test_type_lookup_folds_spacing_and_case_but_is_not_fuzzy() -> None:
    """"coffee maker" must reach CoffeeMaker; "coffee machine" must NOT reach
    anything. A near-miss here recreates the wrong-class problem the abstain
    path exists to prevent."""
    for proposed, expect in (("Coffee Maker", True), ("coffee-maker", True),
                             ("COFFEEMAKER", True), ("coffee machine", False),
                             ("maker", False)):
        drops = _drops()
        kept = _filter_entities(
            [{"canonical_name": "X", "class_iri": ABSTAIN_SENTINEL,
              "proposed_type": proposed}],
            {"https://x/merged#Person"}, drops, [], 200, type_index=_TYPE_INDEX,
        )
        assert bool(kept) is expect, f"{proposed!r} resolved={bool(kept)}"


def test_recovery_is_inert_without_an_index() -> None:
    """Callers that pass no index keep the old behaviour exactly."""
    drops = _drops()
    abst: list = []
    kept = _filter_entities(
        [{"canonical_name": "Keurig", "class_iri": ABSTAIN_SENTINEL,
          "proposed_type": "coffee maker"}],
        {"https://x/merged#Person"}, drops, abst, 200,
    )
    assert kept == [] and drops["abstained"] == 1


def test_concept_menu_ordering_not_prompt_wording_carries_specificity() -> None:
    """A concept filed under a catch-all root can never be linked: no object
    property declares `Process` or `PolicyConcept` as a domain or range.
    Measured on the finance build -- "rate change" -> RateChange carried an
    edge, "energy efficiency programs" -> Process carried none.

    The fix is the MENU ORDER (specific classes first, roots last), applied in
    `_candidate_classes`. A prompt rule was tried and REMOVED: naming the
    catch-alls in order to discourage them primed the model to use them.
    Four-arm A/B over 12 chunks, % of concepts on a specific class:

        original order, no rule    55%
        specific-first, no rule    60%   <- kept
        original order, rule on    52%
        specific-first, rule on    43%   <- worst

    This test pins the removal so the rule does not get helpfully re-added.
    """
    c_sys, _ = concept_extract("text", _CANDIDATES)
    assert "MOST SPECIFIC" not in c_sys
    assert "last resort" not in c_sys.lower()


def test_recovery_index_is_built_after_the_menu_is_final() -> None:
    """Pinned and ancestor-closure classes join the menu WITHOUT ranking well,
    so an index built from the vector pool alone misses them.

    Measured on the finance build: `Organization` -- pinned, offered on every
    chunk -- sits at MEDIAN RANK 687 of 1524 and is inside the top-400 pool for
    only 3 of 27 chunks. An abstention proposing "Organization" therefore found
    nothing to resolve against and the entity was dropped, though that exact
    class was on the menu the model had been shown.

    Pinned to the build ORDER because the bug was ordering: the index used to
    be computed before pins and ancestors were appended to `out`.
    """
    import inspect

    from backend.app.services import db_entity_extract as m

    src = inspect.getsource(m.extract_entities)
    build = src.index("def _recovery(")
    # both return paths must build the index from the completed menu
    assert src.count("_recovery(out)") == 2, "index not built at every return"
    for call in [i for i in range(len(src)) if src.startswith("_recovery(out)", i)]:
        assert call > build, "index built before the menu is final"
    # and the menu must actually be folded in alongside the pool
    body = src[build:build + 2000]
    assert "list(_pool[:recovery_pool]) + list(menu)" in body


def test_abstain_demands_a_proposed_type_imperatively() -> None:
    """`proposed_type` is what abstention recovery resolves against, so an
    abstention without one is unrecoverable by construction.

    Three wordings A/B'd over 27 finance chunks, % of abstentions carrying a
    type, and NET entities (directly kept + recoverable):

        A  current, "put your own short type name ... so the gap can be seen"
             43% typed,  73 typed,   9 recoverable,  NET 424
        B  imperative, "MUST ... REQUIRED ... never be empty or omitted"
            100% typed, 235 typed,  40 recoverable,  NET 443   <- shipped
        C  reason-giving, "that name is used to look for a class you were
           NOT shown ... one left empty is discarded"
              0% typed  <- the model omitted the field entirely

    C is the counter-intuitive one and was verified against raw responses
    rather than assumed: explaining WHY buried the instruction and compliance
    collapsed to zero. Blunt beats motivated here.
    """
    sys_p, _ = entity_extract("text", _CANDIDATES)
    assert "MUST also fill in" in sys_p
    assert "REQUIRED" in sys_p
    assert "never be empty or omitted" in sys_p

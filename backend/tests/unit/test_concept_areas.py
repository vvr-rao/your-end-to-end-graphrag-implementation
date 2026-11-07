"""Concept terms: phrases in a question and entity names resolved to classes."""
from __future__ import annotations

import xml.dom.minidom as minidom
from pathlib import Path

from backend.app.services.concept_areas import classes_in_question, plan_retype
from backend.app.services.db_entity_extract import _normalize_name
from backend.app.services.retrieval_ranking import dead_end_seeds_trigger, documents_for_fraction

OWL = Path("source_ontologies/core_ontologies/domain_concepts.owl")
CLASSES = [
    ("fin", "Financials", ["financial information", "financials"]),
    ("rev", "Revenue", ["revenue", "total revenue"]),
    ("cost", "Cost", ["cost", "costs"]),
    ("mfg", "ManufacturingCosts", ["manufacturing costs"]),
]


def test_a_term_in_a_question_finds_its_class():
    assert classes_in_question("financial information for LG Electronics", CLASSES) == ["fin"]


def test_case_does_not_matter():
    assert classes_in_question("FINANCIALS of Samsung SDI", CLASSES) == ["fin"]


def test_the_longest_term_wins():
    assert classes_in_question("What are manufacturing costs?", CLASSES) == ["mfg"]


def test_several_terms_find_several_classes():
    assert sorted(classes_in_question("revenue and costs", CLASSES)) == ["cost", "rev"]


def test_a_word_containing_a_term_does_not_match():
    assert classes_in_question("Is this costly or revenues-free?", CLASSES) == []


def test_an_entity_named_by_a_term_is_retyped():
    moves = plan_retype([("e1", "revenue", "metric")], CLASSES, _normalize_name)
    assert moves == [("e1", "rev")]


def test_an_entity_already_typed_with_it_is_left_alone():
    assert plan_retype([("e1", "Revenue", "rev")], CLASSES, _normalize_name) == []


def test_a_name_that_only_contains_a_term_is_left_alone():
    assert plan_retype([("e1", "related party revenue", "m")], CLASSES, _normalize_name) == []


def _classes() -> dict[str, dict]:
    out = {}
    for node in minidom.parse(str(OWL)).getElementsByTagName("owl:Class"):
        iri = node.getAttribute("rdf:about").split("#")[-1]
        out[iri] = {
            "parents": [p.getAttribute("rdf:resource").split("#")[-1]
                        for p in node.getElementsByTagName("rdfs:subClassOf")],
            "areas": [p.getAttribute("rdf:resource").split("#")[-1]
                      for p in node.getElementsByTagName("inConceptArea")],
            "terms": [t.firstChild.data for t in node.getElementsByTagName("conceptTerm")],
        }
    return out


def test_every_term_means_one_class():
    seen: dict[str, str] = {}
    for name, cls in _classes().items():
        for term in cls["terms"]:
            assert term == term.lower(), term
            assert term not in seen, (term, name, seen.get(term))
            seen[term] = name


def test_every_class_with_terms_reaches_a_concept_root_or_is_an_area():
    classes = _classes()
    roots = {"Quantity", "Condition", "Process", "Method", "Outcome", "PolicyConcept",
             "Event", "TechnologyConcept", "Infrastructure", "SupplyChainConcept",
             "Intervention", "EconomicConcept", "Material", "Agreement",
             "IntellectualProperty", "Plan", "Classification", "InformationArtifact",
             "Dimension", "DomainConcept"}

    def reaches(name: str, seen: set[str]) -> bool:
        if name in roots:
            return True
        if name in seen or name not in classes:
            return False
        seen.add(name)
        return any(reaches(p, seen) for p in classes[name]["parents"])

    for name, cls in classes.items():
        if cls["terms"]:
            assert reaches(name, set()) or cls["parents"] == ["ConceptArea"], name


def test_the_finance_area_reaches_revenue_and_cost():
    classes = _classes()
    assert "FinancialPerformance" in classes["Revenue"]["areas"]
    assert "FinancialPerformance" in classes["Cost"]["areas"]
    assert classes["FinancialPerformance"]["parents"] == ["ConceptArea"]


def test_a_fraction_scales_with_the_corpus():
    assert documents_for_fraction(0.25, 19) == 5
    assert documents_for_fraction(0.25, 400) == 100


def test_a_fraction_has_a_floor_and_a_ceiling():
    assert documents_for_fraction(0.1, 19, minimum=3) == 3
    assert documents_for_fraction(0.5, 2, minimum=3) == 2
    assert documents_for_fraction(0.5, 0) == 0
    assert documents_for_fraction(0, 19) == 0


def test_any_dead_end_triggers_without_a_share():
    assert dead_end_seeds_trigger(1, 30, 0)
    assert not dead_end_seeds_trigger(0, 30, 0)


def test_a_share_needs_that_many_dead_ends():
    assert not dead_end_seeds_trigger(1, 4, 0.5)
    assert dead_end_seeds_trigger(2, 4, 0.5)
    assert not dead_end_seeds_trigger(0, 0, 0.5)

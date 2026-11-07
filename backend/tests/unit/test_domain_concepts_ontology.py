"""domain_concepts.owl v2: branding, type/topic separation, SKOS navigation, terms vs
synonyms, migration, and query expansion through the topic layer.

Everything here reads the shipped file; no database or LLM."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest
import rdflib
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from backend.app.services.concept_areas import (
    AMBIGUOUS_NAMES,
    classes_in_question,
    expand_topics,
    plan_retype,
)
from backend.app.services.db_entity_extract import _normalize_name

OWL_PATH = Path("source_ontologies/core_ontologies/domain_concepts.owl")
MIGRATION = Path("source_ontologies/core_ontologies/domain_concepts_migration.json")
ONT = "https://veerla-ramrao.ai/ontology/domain-concepts"
NS = ONT + "#"
BRAND = "https://veerla-ramrao.ai/ontology/"
SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
DCT = rdflib.Namespace("http://purl.org/dc/terms/")
DC = rdflib.Namespace(NS)


@lru_cache(maxsize=1)
def graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(OWL_PATH, format="xml")
    return g


def own(x) -> bool:
    return isinstance(x, URIRef) and str(x).startswith(NS)


def name(x) -> str:
    return str(x).rsplit("#", 1)[-1]


def classes() -> set[URIRef]:
    return {c for c in graph().subjects(RDF.type, OWL.Class) if own(c)}


def deprecated() -> set[URIRef]:
    return {c for c in classes() if (c, OWL.deprecated, None) in graph()}


def areas() -> set[URIRef]:
    return {c for c in classes() if (c, RDFS.subClassOf, DC.ConceptArea) in graph()}


def concepts() -> set[URIRef]:
    """Maintained, non-area concept classes."""
    return classes() - areas() - deprecated() - {DC.ConceptArea}


def parents(c) -> set[URIRef]:
    return set(graph().objects(c, RDFS.subClassOf))


def ancestors(c) -> set[URIRef]:
    out: set[URIRef] = set()
    todo = list(parents(c))
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += parents(p)
    return out


def literals(c, pred) -> list[str]:
    return [str(o) for o in graph().objects(c, pred) if isinstance(o, Literal)]


# ---------------------------------------------------------------- parsing + references

def test_parses_and_every_referenced_own_iri_is_declared():
    g = graph()
    declared = {s for s in g.subjects(RDF.type, None) if own(s)}
    referenced = {o for o in g.objects() if own(o)}
    assert not referenced - declared, sorted(map(name, referenced - declared))


def test_no_subclass_or_broader_cycles():
    for c in classes():
        assert c not in ancestors(c), name(c)
    for c in classes():
        seen, todo = set(), list(graph().objects(c, SKOS.broader))
        while todo:
            b = todo.pop()
            assert b != c, f"skos:broader cycle through {name(c)}"
            if b not in seen:
                seen.add(b)
                todo += list(graph().objects(b, SKOS.broader))


def test_owlready2_loads_it():
    """Structural OWL load (no reasoner is installed; HermiT/Pellet need Java)."""
    from owlready2 import World
    w = World()
    onto = w.get_ontology(OWL_PATH.resolve().as_uri()).load()
    names = {c.name for c in onto.classes()}
    assert {"Revenue", "Port", "ConceptArea", "InformationArtifact"} <= names


# ---------------------------------------------------------------- branding

def test_ontology_iri_and_namespace_keep_the_brand():
    g = graph()
    assert (URIRef(ONT), RDF.type, OWL.Ontology) in g
    assert ONT == "https://veerla-ramrao.ai/ontology/domain-concepts"
    assert NS.startswith(BRAND)


def test_every_own_resource_is_branded():
    g = graph()
    kinds = (OWL.Class, OWL.ObjectProperty, OWL.AnnotationProperty, SKOS.ConceptScheme)
    external = {"http://xmlns.com/foaf/0.1/Document", "http://purl.org/dc/terms/isReplacedBy"}
    for kind in kinds:
        for s in g.subjects(RDF.type, kind):
            if str(s) in external:
                continue
            assert str(s).startswith(BRAND), s
    for prop in ("inConceptArea", "conceptTerm"):
        assert (DC[prop], RDF.type, OWL.AnnotationProperty) in g


@pytest.mark.parametrize("path", ["config/config.example.yaml"])
def test_shipped_config_protects_the_brand(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert BRAND in (cfg["ontology"]["protected_iri_prefixes"] or [])


def test_prune_keeps_every_domain_concept_and_its_annotations():
    """A corpus that detects nothing still keeps the whole branded layer, with
    its topic annotations, under the protected prefix."""
    from backend.app.helpers.ontology_parsing import extract_ontology_to_dicts
    from backend.app.services.pipeline_llm import _apply_prune

    loaded = extract_ontology_to_dicts(str(OWL_PATH))
    pruned, _keep = _apply_prune(loaded, [], (BRAND,))
    kept = pruned["classes_dict"]
    assert {str(c) for c in classes()} <= set(kept)
    port = kept[NS + "Port"]["annotations"]
    # In memory rdflib terms; merged.json holds the same values as strings.
    assert set(map(str, port[NS + "inConceptArea"])) == {NS + "Transportation", NS + "Logistics"}
    assert "seaport" in map(str, port[str(SKOS.altLabel)])
    fp = kept[NS + "FinancialPerformance"]["annotations"]
    assert list(map(str, fp[str(SKOS.broader)])) == [NS + "FinancialReporting"]


# ---------------------------------------------------------------- types vs topics

def test_areas_sit_only_under_concept_area():
    for a in areas():
        assert parents(a) == {DC.ConceptArea}, name(a)


def test_no_concept_is_a_subclass_of_an_area():
    for c in concepts() | deprecated():
        assert not ancestors(c) & (areas() | {DC.ConceptArea}), name(c)


def test_membership_points_at_areas_only():
    for c, a in graph().subject_objects(DC.inConceptArea):
        assert a in areas(), (name(c), name(a))
        assert c in concepts(), name(c)


def test_maintenance_is_a_process_in_the_equipment_topic():
    assert DC.Process in ancestors(DC.Maintenance)
    assert DC.Equipment not in ancestors(DC.Maintenance)
    assert DC.Equipment in set(graph().objects(DC.Maintenance, DC.inConceptArea))


def test_ports_and_airports_are_infrastructure_in_transport_and_logistics():
    for c in (DC.Port, DC.Airport):
        assert DC.Infrastructure in ancestors(c)
        assert {DC.Transportation, DC.Logistics} <= set(graph().objects(c, DC.inConceptArea))


def test_new_areas_exist():
    want = {
        "FinancialReporting", "FinancialInstruments", "FinancingAndCapitalManagement",
        "WorkingCapitalManagement", "InvestmentAndPortfolioManagement", "Taxation",
        "DrugDevelopment", "TherapeuticAreasAndDiseases", "DrugModalitiesAndComposition",
        "ClinicalStudyDesignAndPopulation", "Pharmacology", "Pharmacovigilance",
        "PharmaceuticalQualityAndManufacturing", "RegulatoryAffairs",
        "ProductionPlanningAndControl", "MaterialsAndProductStructure", "ManufacturingProcesses",
        "AssetAndMaintenanceManagement", "Transportation", "WarehousingAndDistribution",
        "SupplyChainPlanning", "TradeAndCustoms", "TraceabilityAndColdChain",
        "SupplyChainRiskAndResilience",
    }
    assert want <= {name(a) for a in areas()}


def test_corrections():
    g = graph()
    assert DC.Measure in parents(DC.Metric) and DC.Quantity in parents(DC.Measure)
    assert DC.Measure in ancestors(DC.Profit)          # profit is not forced to be raw
    assert DC.EconomicConcept in parents(DC.Asset) and DC.Measure in parents(DC.Assets)
    assert DC.EconomicConcept in parents(DC.Debt) and DC.Measure in parents(DC.DebtBalance)
    assert DC.Metric in parents(DC.ScrapRate) and DC.Material in parents(DC.ScrapMaterial)
    assert DC.Process in parents(DC.Pricing) and DC.Measure in ancestors(DC.ListPrice)
    assert DC.Classification in parents(DC.TrialPhase)
    assert DC.Outcome not in ancestors(DC.TrialEndpoint)
    assert (DC.TrialEndpoint, SKOS.related, DC.Outcome) in g
    assert (DC.RegulatoryStatus, SKOS.broader, DC.RegulatoryAffairs) in g
    for c in (DC.Partnership, DC.TradeDeal, DC.SupplyContract, DC.PowerPurchaseAgreement):
        assert DC.Agreement in ancestors(c) and DC.Event not in ancestors(c), name(c)
    assert DC.Event in parents(DC.AgreementSigning) and DC.Event in parents(DC.LawEnactment)
    assert DC.PolicyConcept in parents(DC.Law)
    assert DC.IntellectualProperty in parents(DC.Patent)
    assert DC.Plan in parents(DC.ProductRoadmap)
    assert DC.ReportingDocument in parents(DC.AuditReport)
    assert DC.FinancialStatements not in ancestors(DC.AuditReport)
    assert DC.FinancialStatements in parents(DC.StatementOfChangesInEquity)
    foaf_doc = URIRef("http://xmlns.com/foaf/0.1/Document")
    assert foaf_doc in parents(DC.InformationArtifact)  # reused, not duplicated
    assert DC.InformationArtifact in parents(DC.ReportingDocument)
    tc = " ".join(literals(DC.TechnologyConcept, SKOS.definition)
                  + literals(DC.TechnologyConcept, RDFS.comment)).lower()
    assert "technique" not in tc and "method" not in tc   # techniques are Method
    assert "Method" in " ".join(literals(DC.TechnologyConcept, SKOS.scopeNote))


def test_generic_predicates_keep_their_iris_and_describe_kinds():
    for p in ("hasMetric", "hasCondition", "undergoesProcess", "usesMethod",
              "receivesIntervention", "hasOutcome", "regulatedBy"):
        assert (DC[p], RDF.type, OWL.ObjectProperty) in graph(), p
        note = " ".join(literals(DC[p], SKOS.scopeNote))
        assert "DESCRIPTOR" in note and "period" in note and "unit" in note


# ---------------------------------------------------------------- SKOS

def test_one_english_preflabel_and_a_definition_per_maintained_concept():
    for c in concepts() | areas() | {DC.ConceptArea}:
        prefs = [o for o in graph().objects(c, SKOS.prefLabel) if o.language == "en"]
        assert len(prefs) == 1, name(c)
        assert literals(c, SKOS.definition), name(c)
        assert (c, SKOS.inScheme, None) in graph(), name(c)


def test_every_area_is_reachable_from_a_top_concept():
    tops = set(graph().objects(DC.ConceptAreaScheme, SKOS.hasTopConcept))
    assert tops and tops <= areas()
    for a in areas():
        seen, node = set(), a
        chain = [a]
        while chain:
            node = chain.pop()
            if node in tops:
                break
            seen.add(node)
            chain += [b for b in graph().objects(node, SKOS.broader) if b not in seen]
        assert node in tops, name(a)


def test_areas_with_scope_notes_for_llm_placement():
    for a in areas():
        assert literals(a, SKOS.scopeNote), name(a)


# ---------------------------------------------------------------- terms vs synonyms

def _all_names(c) -> set[str]:
    return {_normalize_name(x) for x in literals(c, SKOS.prefLabel) + literals(c, SKOS.altLabel)}


def test_ambiguous_words_are_never_alt_labels():
    for c in classes():
        for alt in literals(c, SKOS.altLabel):
            assert _normalize_name(alt) not in AMBIGUOUS_NAMES, (name(c), alt)


def test_no_two_concepts_share_a_name():
    """altLabels feed alias merging; a shared name would merge different things."""
    seen: dict[str, URIRef] = {}
    for c in concepts():
        for n in _all_names(c):
            assert n not in seen or seen[n] == c, (n, name(c), name(seen[n]))
            seen[n] = c


@pytest.mark.parametrize("a,b", [
    ("GrossProfit", "OperatingProfit"), ("GrossProfit", "NetIncome"),
    ("OperatingProfit", "NetIncome"), ("GrossMargin", "OperatingMargin"),
    ("SingleSourcing", "SoleSourcing"), ("EarningsRelease", "EarningsCall"),
    ("EarningsRelease", "QuarterlyResults"), ("AdverseEvent", "AdverseDrugReaction"),
    ("IncomeStatement", "StatementOfComprehensiveIncome"),
])
def test_distinct_concepts_stay_distinct(a, b):
    assert DC[a] != DC[b] and DC[a] in concepts() and DC[b] in concepts()
    assert not _all_names(DC[a]) & _all_names(DC[b])
    assert (DC[a], OWL.equivalentClass, DC[b]) not in graph()


def test_every_term_is_lowercase_and_means_one_class():
    seen: dict[str, URIRef] = {}
    for c, t in graph().subject_objects(DC.conceptTerm):
        assert str(t) == str(t).lower(), t
        assert str(t) not in seen, (str(t), name(c), name(seen.get(str(t), c)))
        seen[str(t)] = c


# ---------------------------------------------------------------- migration

def test_deprecated_iris_are_mapped_and_replaced():
    mig = json.loads(MIGRATION.read_text(encoding="utf-8"))
    assert mig["ontology"] == ONT and mig["namespace"] == NS
    by_old = {m["old"]: m for m in mig["mappings"]}
    for c in deprecated():
        assert list(graph().objects(c, DCT.isReplacedBy)), name(c)
        assert by_old[str(c)]["kind"] == "deprecated"
        assert not list(graph().objects(c, DC.conceptTerm))
        assert not list(graph().objects(c, DC.inConceptArea))
    for m in mig["mappings"]:
        for iri in [m["old"], *m["new"]]:
            assert URIRef(iri) in classes(), iri
            assert iri.startswith(BRAND)
    assert {name(c) for c in deprecated()} == {"Scrap", "NewLaw"}


def test_v1_iris_all_survive():
    """Every IRI the v1 file declared still exists (kept or deprecated)."""
    v1 = {"Acquisition", "AdverseEvent", "Approval", "Assets", "AuditReport", "BalanceSheet",
          "CashFlow", "ClinicalEvidence", "ConceptArea", "CorporateEvents", "Debt", "Equipment",
          "Financials", "FinancialPerformance", "FinancialStatements", "Logistics", "Maintenance",
          "Margin", "NewLaw", "Partnership", "Patent", "Pricing", "Profit", "RegulatoryStatus",
          "Revenue", "Scrap", "SingleSourcing", "TradeDeal", "TrialEndpoint", "TrialPhase"}
    assert {DC[n] for n in v1} <= classes()


# ---------------------------------------------------------------- query expansion

def _term_classes():
    return [(c, name(c), [str(t) for t in graph().objects(c, DC.conceptTerm)])
            for c in classes() if (c, DC.conceptTerm, None) in graph()]


def _links():
    g = graph()
    return [(c, str(c), [str(b) for b in g.objects(c, SKOS.broader)],
             [str(a) for a in g.objects(c, DC.inConceptArea)]) for c in classes()]


def _reach(question: str) -> set[str]:
    """What a question's concept-area route covers: matched classes -> topic
    layer -> subclasses (the SQL's subClassOf step, done here in memory)."""
    hit = classes_in_question(question, _term_classes())
    topic = expand_topics(hit, _links(), {c: str(c) for c in hit})
    out = set(topic)
    for c in classes():
        if ancestors(c) & set(topic):
            out.add(c)
    return {name(c) for c in out}


def test_financial_performance_reaches_revenue_cost_profit_and_margins():
    got = _reach("How was LG's financial performance last year?")
    assert {"Revenue", "Cost", "Profit", "GrossProfit", "NetIncome", "Margin",
            "GrossMargin", "OperatingMargin"} <= got


def test_transportation_infrastructure_reaches_ports_and_airports():
    got = _reach("Which transportation infrastructure did the strike affect?")
    assert {"Port", "Airport", "Canal"} <= got


def test_logistics_also_reaches_ports():
    assert {"Port", "Airport", "Transportation"} <= _reach("logistics bottlenecks in Europe")


def test_equipment_maintenance_reaches_maintenance_not_as_equipment():
    got = _reach("What equipment maintenance did the plant report?")
    assert "Maintenance" in got
    assert DC.TechnologyConcept not in ancestors(DC.Maintenance)


def test_drug_safety_reaches_both_without_merging_them():
    got = _reach("What does the drug safety data show?")
    assert {"AdverseEvent", "AdverseDrugReaction", "SeriousAdverseEvent", "Toxicity"} <= got
    assert DC.AdverseEvent in parents(DC.AdverseDrugReaction)
    assert DC.AdverseDrugReaction not in ancestors(DC.AdverseEvent)


def test_gross_profit_does_not_route_to_net_income():
    got = _reach("What was Geely's gross profit in 2023?")
    assert "GrossProfit" in got
    assert "NetIncome" not in got and "Profit" not in got


def test_shared_area_membership_creates_no_relationship():
    """Membership is an annotation, not an object property: there is no
    property a pipeline could use to link two entities through an area."""
    g = graph()
    assert (DC.inConceptArea, RDF.type, OWL.ObjectProperty) not in g
    from backend.app.services.predicates import _BUILTIN_ALLOWLIST, GENERIC_CONCEPT_PREDICATES
    assert NS + "inConceptArea" not in GENERIC_CONCEPT_PREDICATES
    assert NS + "inConceptArea" not in _BUILTIN_ALLOWLIST


def test_expansion_is_downward_only():
    hit = [DC.Revenue]
    assert expand_topics(hit, _links(), {DC.Revenue: str(DC.Revenue)}) == [DC.Revenue]


# ---------------------------------------------------------------- retyping

def _name_classes():
    return [(c, name(c), literals(c, SKOS.prefLabel) + literals(c, SKOS.altLabel))
            for c in concepts()]


def test_retype_uses_names_not_routing_terms():
    ents = [("e1", "COGS", "x"), ("e2", "turnover", "x"), ("e3", "side effects", "x"),
            ("e4", "net profit", "x"), ("e5", "margin", "x"), ("e6", "equity", "x")]
    moves = dict(plan_retype(ents, _name_classes(), _normalize_name))
    assert moves["e1"] == DC.CostOfGoodsSold
    assert moves["e4"] == DC.NetIncome
    assert "e2" not in moves          # a routing term (Revenue), not a name
    assert "e3" not in moves          # routing term of AdverseDrugReaction
    assert "e5" not in moves and "e6" not in moves   # ambiguous


def test_retype_never_targets_an_area():
    area_names = [(a, name(a), literals(a, SKOS.prefLabel)) for a in areas()]
    # The caller only passes classes under the concept roots; areas are not.
    assert not any(DC.ConceptArea in parents(c) for c, _, _ in _name_classes())
    assert plan_retype([("e1", "Logistics", "x")], _name_classes(), _normalize_name) == []
    assert area_names  # sanity: areas exist, they are just never offered


# ---------------------------------------------------------------- observation context

def test_a_table_value_keeps_its_source_period_and_concept():
    """Numerical facts come from StructuredTables, not from the generic
    predicates. Each value stays tied to its document, page and caption (the
    caption usually states the unit and scale), its column (usually the period)
    and its row (the concept measured). No structured unit or reporting-entity
    field exists yet: the unit is read from caption/cell text and the entity
    from the document."""
    from backend.app.services.table_jsonld import (
        build_cell_iri,
        build_column_iri,
        build_row_iri,
        empty_payload,
        validate_table_jsonld,
    )
    p = empty_payload("abc123", 0, extraction_method="pdfplumber",
                      caption="Consolidated revenue (USD millions)", page_number=41)
    col = build_column_iri(p["@id"], 0)
    p["columns"] = [{"@id": col, "@type": "viao:TableColumn",
                     "columnLabel": "FY2024", "columnIndex": 0}]
    p["rows"] = [{"@id": build_row_iri(p["@id"], 0), "@type": "viao:TableRow",
                  "rowLabel": "Revenue", "rowIndex": 0, "cells": [
                      {"@id": build_cell_iri(p["@id"], 0, 0), "@type": "viao:TableCell",
                       "inColumn": col, "cellValue": "78,400"}]}]
    assert validate_table_jsonld(p) == []
    cell = p["rows"][0]["cells"][0]
    column = next(c for c in p["columns"] if c["@id"] == cell["inColumn"])
    assert (p["derivedFromDocument"]["@id"], p["pageNumber"], p["caption"],
            column["columnLabel"], p["rows"][0]["rowLabel"], cell["cellValue"]) == (
        p["derivedFromDocument"]["@id"], 41, "Consolidated revenue (USD millions)",
        "FY2024", "Revenue", "78,400")
    assert "abc123" in p["derivedFromDocument"]["@id"]


def test_layer_g_root_menu_offers_types_not_topics_or_deprecated():
    from backend.app.helpers.ontology_parsing import extract_ontology_to_dicts
    from backend.app.helpers.ontology_pruning import concept_root_menu

    menu = {m["LABEL"] for m in concept_root_menu(
        extract_ontology_to_dicts(str(OWL_PATH))["classes_dict"])}
    assert {"Quantity", "Agreement", "IntellectualProperty", "Plan", "Classification",
            "InformationArtifact", "DomainConcept"} <= menu
    assert not menu & {name(a) for a in areas()}
    assert not menu & {"ConceptArea", "Scrap", "NewLaw", "FinancialTable", "Measure"}


def test_head_kinds_map_to_roots_that_exist():
    from backend.app.services.concept_heads import HEAD_KINDS, HEAD_ROOTS
    for _kind, root, _desc in HEAD_KINDS:
        if root is not None:
            assert DC[root] in classes(), root
    assert {"Agreement", "Plan", "InformationArtifact"} <= set(HEAD_ROOTS)

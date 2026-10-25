"""Domain-neutral concept layer: concept heads (Stage 3c), Layer G roots,
generic concept predicates, concept seeding + co-occurrence ranking, and
fiscal-period time detection."""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from types import SimpleNamespace

from backend.app.services import concept_heads as ch
from backend.app.services import retrieval as rt
from backend.app.services import retrieval_sql as rsql

# --------------------------------------------------------------------------- #
# Stage 3c: concept heads
# --------------------------------------------------------------------------- #


def test_head_of_takes_last_word_or_two_when_last_is_weak() -> None:
    assert ch.head_of("SemiconductorRevenue") == "Revenue"
    assert ch.head_of("Related Party Revenue") == "Revenue"
    assert ch.head_of("ObjectiveResponseRate") == "ResponseRate"   # pharma
    assert ch.head_of("WeldDefect") == "Defect"                    # manufacturing
    assert ch.head_of("DefectRate") is None                        # 'Rate' alone is too generic
    assert ch.head_of("Revenue") is None                           # a one-word label IS a head
    assert ch.head_of("HbA1c") is None


def test_candidate_groups_need_two_members_and_skip_the_head_itself() -> None:
    items = [{"LABEL": "SemiconductorRevenue"}, {"LABEL": "RelatedPartyRevenue"},
             {"LABEL": "Revenue"}, {"LABEL": "InsulinTherapy"}]
    assert ch.candidate_groups(items) == {"Revenue": [0, 1]}


def test_apply_heads_reparents_only_broadly_parented_members() -> None:
    payload = {"MATCH NOT FOUND": [
        {"LABEL": "SemiconductorRevenue", "PARENT_LABEL": "Metric"},
        {"LABEL": "RelatedPartyRevenue", "PARENT_LABEL": "EconomicConcept"},
        {"LABEL": "SegmentRevenue", "PARENT_LABEL": "SegmentDisclosure"},   # specific: kept
    ]}
    out, moved = ch.apply_heads(payload, [{
        "head": "Revenue", "root": "Measure", "description": "Income from sales.",
        "members": ["SemiconductorRevenue", "RelatedPartyRevenue", "SegmentRevenue"]}])
    by = {it["LABEL"]: it for it in out["MATCH NOT FOUND"]}
    assert by["Revenue"]["PARENT_LABEL"] == "Measure"
    assert by["SemiconductorRevenue"]["PARENT_LABEL"] == "Revenue"
    assert by["RelatedPartyRevenue"]["PARENT_LABEL"] == "Revenue"
    assert by["SegmentRevenue"]["PARENT_LABEL"] == "SegmentDisclosure"
    assert moved == 2


def test_apply_heads_treats_a_root_iri_parent_as_broad() -> None:
    # Saved Qwen proposals carry the parent as a full IRI.
    payload = {"MATCH NOT FOUND": [
        {"LABEL": "GreenhouseGasLiability",
         "PARENT_LABEL": "https://veerla-ramrao.ai/ontology/domain-concepts#EconomicConcept"},
        {"LABEL": "OtherLiability", "PARENT_LABEL": "https://example.org/fibo/LiabilityKind"},
    ]}
    out, moved = ch.apply_heads(payload, [{
        "head": "Liability", "root": "EconomicConcept", "description": "",
        "members": ["GreenhouseGasLiability", "OtherLiability"]}])
    by = {it["LABEL"]: it for it in out["MATCH NOT FOUND"]}
    assert by["GreenhouseGasLiability"]["PARENT_LABEL"] == "Liability"
    assert by["OtherLiability"]["PARENT_LABEL"].endswith("LiabilityKind")
    assert moved == 1


class _Router:
    def __init__(self, reply: dict | None, spec: dict | None) -> None:
        self.reply, self.spec, self.calls = reply, spec, 0

    def task_spec(self, task):
        if self.spec is None:
            raise KeyError(task)
        return self.spec

    async def chat(self, task, system, user):
        self.calls += 1
        return SimpleNamespace(text=json.dumps(self.reply))


_PAYLOAD = {"MATCH NOT FOUND": [
    {"LABEL": "SemiconductorRevenue", "PARENT_LABEL": "Metric"},
    {"LABEL": "RelatedPartyRevenue", "PARENT_LABEL": "NONE"},
    {"LABEL": "RevenueRecognitionPolicy", "PARENT_LABEL": "NONE"},
    {"LABEL": "BatteryManufacturer", "PARENT_LABEL": "Organization"},
    {"LABEL": "EVManufacturer", "PARENT_LABEL": "Organization"},
]}


def test_generalize_accepts_rejects_and_drops_rejected_members() -> None:
    # Groups go largest-first, ties alphabetical: Manufacturer=0, Revenue=1.
    # RevenueRecognitionPolicy heads 'Policy' (alone), so it is no Revenue member.
    reply = {"decisions": [
        {"id": 0, "accept": False},
        {"id": 1, "accept": True, "root": "measure", "description": "Income.",
         "rejected_members": []},
    ]}
    router = _Router(reply, {"enabled": True})
    out, accepted = asyncio.run(ch.generalize_concept_heads(router, _PAYLOAD, log=lambda *_: None))
    assert [a["head"] for a in accepted] == ["Revenue"]
    assert accepted[0]["root"] == "Measure"               # case-normalised to the root label
    by = {it["LABEL"]: it for it in out["MATCH NOT FOUND"]}
    assert by["SemiconductorRevenue"]["PARENT_LABEL"] == "Revenue"
    assert by["EVManufacturer"]["PARENT_LABEL"] == "Organization"


def test_generalize_rejects_a_root_outside_the_concept_layer() -> None:
    reply = {"decisions": [{"id": 1, "accept": True, "root": "Organization"}]}
    out, accepted = asyncio.run(ch.generalize_concept_heads(
        _Router(reply, {"enabled": True}), _PAYLOAD, log=lambda *_: None))
    assert accepted == [] and out is _PAYLOAD


def test_generalize_is_a_noop_when_disabled_or_unconfigured() -> None:
    for spec in (None, {"enabled": False}):
        router = _Router({}, spec)
        out, accepted = asyncio.run(ch.generalize_concept_heads(router, _PAYLOAD, log=lambda *_: None))
        assert out is _PAYLOAD and accepted == [] and router.calls == 0


def test_generalize_skips_an_unknown_prompt_set() -> None:
    router = _Router({}, {"enabled": True, "prompt_set": "qwen"})
    out, _ = asyncio.run(ch.generalize_concept_heads(router, _PAYLOAD, log=lambda *_: None))
    assert out is _PAYLOAD and router.calls == 0


def test_every_preset_defines_the_concept_head_judge() -> None:
    import yaml
    for f in ("config/models.example.yaml", "config/models.openai.example.yaml",
              "config/models.anthropic.example.yaml", "config/models.deepinfra.example.yaml"):
        spec = yaml.safe_load(open(f))["tasks"]["concept_head_judge"]
        assert spec["enabled"] is True, f
    di = yaml.safe_load(open("config/models.deepinfra.example.yaml"))["tasks"]["concept_head_judge"]
    assert di["provider"] == "deepinfra" and di["prompt_set"] == "deepseek"


# --------------------------------------------------------------------------- #
# Layer G roots
# --------------------------------------------------------------------------- #

DC = "https://veerla-ramrao.ai/ontology/domain-concepts#"


def _cls(iri, label, parent=None, comment=""):
    return {"iri": iri, "name": label, "labels": [label], "comments": [comment] if comment else [],
            "superclasses": [{"iri": parent}] if parent else []}


def test_concept_root_menu_reads_the_domain_concepts_layer() -> None:
    from backend.app.helpers.ontology_pruning import concept_root_menu

    classes = {
        DC + "Quantity": _cls(DC + "Quantity", "Quantity", comment="Measurable amounts."),
        DC + "Metric": _cls(DC + "Metric", "Metric", parent=DC + "Quantity"),      # not a root
        DC + "Condition": _cls(DC + "Condition", "Condition"),
        DC + "DomainConcept": _cls(DC + "DomainConcept", "DomainConcept"),
        DC + "FinancialTable": _cls(DC + "FinancialTable", "FinancialTable"),      # table anchor
        "http://xmlns.com/foaf/0.1/Organization": _cls("http://xmlns.com/foaf/0.1/Organization", "Organization"),
    }
    menu = concept_root_menu(classes)
    labels = [m["LABEL"] for m in menu]
    assert labels[-1] == "DomainConcept"                   # last resort goes last
    assert set(labels) == {"Quantity", "Condition", "DomainConcept", "Organization"}
    assert menu[0]["DESCRIPTION"] == "Measurable amounts."
    assert concept_root_menu({"x#A": _cls("x#A", "A")}) == []


def test_apply_concept_grouping_parents_a_subconcept_at_its_root() -> None:
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    base = "https://veerla-ramrao.ai/ontology/merged#"
    classes = {
        DC + "Condition": _cls(DC + "Condition", "Condition"),
        base + "hypoglycemia": _cls(base + "hypoglycemia", "Hypoglycemia"),
    }
    iris, audit = apply_concept_grouping(classes, base, {
        "NEW_SUBCONCEPTS": [{"LABEL": "MetabolicCondition", "DESCRIPTION": "d", "PARENT_ROOT": "Condition"}],
        "ASSIGNMENTS": [{"CLASS_LABEL": "Hypoglycemia", "CONCEPT_LABEL": "MetabolicCondition"}],
    })
    assert len(iris) == 1
    assert classes[iris[0]]["superclasses"][0]["iri"] == DC + "Condition"
    assert classes[base + "hypoglycemia"]["superclasses"][0]["iri"] == iris[0]


def test_concept_grouping_prompt_lists_the_given_roots() -> None:
    from backend.app.services.prompts import concept_grouping

    sys_, _ = concept_grouping([{"LABEL": "X"}], [{"LABEL": "Quantity", "DESCRIPTION": "amounts"}])
    assert "- Quantity: amounts" in sys_ and "NEW_SUBCONCEPTS" in sys_ and "PARENT_ROOT" in sys_


# --------------------------------------------------------------------------- #
# Generic concept predicates
# --------------------------------------------------------------------------- #


def test_generic_predicate_accepts_any_subject_but_checks_the_range() -> None:
    from backend.app.services.db_entity_extract import _types_fit
    from backend.app.services.predicates import GENERIC_CONCEPT_PREDICATES

    has_metric = next(i for i in GENERIC_CONCEPT_PREDICATES if i.endswith("#hasMetric"))
    pred = {"iri": has_metric, "domain_iris": [], "range_iris": [DC + "Quantity"]}
    lines = {"org": {"org", "Agent"}, "rev": {"rev", DC + "Measure", DC + "Quantity"},
             "dis": {"dis", DC + "Condition"}}
    assert _types_fit(pred, "org", "rev", lines) is True
    assert _types_fit(pred, "org", "dis", lines) is False
    # An ordinary predicate with an empty domain still fits nothing.
    assert _types_fit({"iri": "x#p", "domain_iris": [], "range_iris": [DC + "Quantity"]},
                      "org", "rev", lines) is False


def test_domain_concepts_owl_declares_the_generic_predicates() -> None:
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    from backend.app.services.predicates import GENERIC_CONCEPT_PREDICATES

    g = rdflib.Graph()
    g.parse("source_ontologies/core_ontologies/domain_concepts.owl", format="xml")
    for iri, root in GENERIC_CONCEPT_PREDICATES.items():
        s = rdflib.URIRef(iri)
        assert (s, RDF.type, OWL.ObjectProperty) in g, iri
        assert (s, RDFS.range, rdflib.URIRef(root)) in g, iri
        assert (rdflib.URIRef(root), RDF.type, OWL.Class) in g, root


# --------------------------------------------------------------------------- #
# Retrieval: concept seeds, co-occurrence
# --------------------------------------------------------------------------- #

LG, REV = uuid.uuid4(), uuid.uuid4()


class _Emb:
    async def embed(self, texts):
        return [[0.0] for _ in texts]


def _patch_seeding(monkeypatch, concepts):
    @contextlib.asynccontextmanager
    async def scope():
        yield None

    async def srcs(session):
        return []

    async def by_name(session, term, **kw):
        return {"LG Electronics": [LG]}.get(term, [])

    async def concept_seeds(session, terms, vecs, q, concept_ids, **kw):
        assert concept_ids
        return list(concepts)

    monkeypatch.setattr(rt, "session_scope", scope)
    monkeypatch.setattr(rsql, "document_sources", srcs)
    monkeypatch.setattr(rsql, "match_entities_by_name_or_alias", by_name)
    monkeypatch.setattr(rsql, "concept_seed_entities", concept_seeds)


def test_named_seeds_gain_concept_seeds_when_enabled(monkeypatch) -> None:
    _patch_seeding(monkeypatch, [REV])
    parsed = {"entities": ["LG Electronics"], "classes": ["revenue"]}
    seeds, tier = asyncio.run(rt._cascade_entity_seeds(parsed, [0.0], {}, _Emb(), concept_ids={REV}))
    assert (seeds, tier) == ([LG, REV], "named+concepts")


def test_without_concept_ids_the_named_tier_is_unchanged(monkeypatch) -> None:
    _patch_seeding(monkeypatch, [REV])
    parsed = {"entities": ["LG Electronics"], "classes": ["revenue"]}
    assert asyncio.run(rt._cascade_entity_seeds(parsed, [0.0], {}, _Emb())) == ([LG], "named")


def test_concept_knobs_are_off_unless_configured(monkeypatch) -> None:
    monkeypatch.setattr(rt, "_qa_cfg", lambda key, default: default)
    for k in ("concept_seeds", "concept_terminal", "seed_cooccurrence"):
        assert rt._qa_flag(k) is False
    import yaml
    qa = yaml.safe_load(open("config/config.example.yaml"))["qa"]
    assert qa["concept_seeds"] is True and qa["concept_terminal"] is True
    assert qa["seed_cooccurrence"] is True


def test_cooccurrence_needs_at_least_two_seeds_without_touching_the_db() -> None:
    # session=None: the early return must fire before any query.
    out = asyncio.run(rsql.fetch_chunks_by_seed_cooccurrence(None, [LG], {REV}))
    assert out == []
    assert asyncio.run(rsql.fetch_chunks_by_seed_cooccurrence(None, [], {REV}, [uuid.uuid4()])) == []


def test_bfs_sql_never_expands_terminal_nodes() -> None:
    for sql in (rsql._BFS_SQL.text, rsql._ENTITY_BFS_SQL.text):
        assert "terminal_ids" in sql


# --------------------------------------------------------------------------- #
# Time: fiscal periods
# --------------------------------------------------------------------------- #


def test_fiscal_period_forms_become_quarters_and_years() -> None:
    from backend.app.services.db_temporal_enrich import _extract_dates as ex

    assert ex("1Q 2025 results") == {"Q1_2025"}
    assert ex("1Q25 and 4Q'24") == {"Q1_2025", "Q4_2024"}
    assert ex("Q1'25") == {"Q1_2025"}
    assert ex("FY2024 vs FY23") == {"YEAR_2024", "YEAR_2023"}
    assert ex("For the three-month periods ended March 31, 2025") == {"Q1_2025", "DAY_2025_03_31"}
    assert ex("quarter ended June 30, 2024") == {"Q2_2024", "DAY_2024_06_30"}
    # Unchanged forms.
    assert ex("Q3 2024") == {"Q3_2024"}
    assert ex("in 2023") == {"YEAR_2023"}
    assert ex("10Q 2025 filing") == {"YEAR_2025"}


def test_already_enriched_subquery_excludes_null_chunk_ids() -> None:
    # Artifact time edges have no source chunk; a NULL inside NOT IN hid
    # every chunk from enrich-time on a re-run.
    from backend.app.services.db_temporal_enrich import _already_enriched_chunk_ids

    sql = str(_already_enriched_chunk_ids().compile()).lower()
    assert "source_chunk_id is not null" in sql

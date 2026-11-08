import asyncio
import json
from types import SimpleNamespace

from backend.app.services import endpoint_class_review as ecr
from backend.app.services.prompts import endpoint_class_review as prompt

DC = "https://veerla-ramrao.ai/ontology/domain-concepts#"
M = "https://example.com/merged#"
FOAF_AGENT = "http://xmlns.com/foaf/0.1/Agent"


def _ref(iri):
    return {"kind": "entity", "name": iri.rsplit("#", 1)[-1], "iri": iri}


def _cls(label, parent=None, desc=None, auto=False):
    rec = {"labels": [label], "descriptions": [desc] if desc else [],
           "superclasses": [_ref(parent)] if parent else [], "annotations": {}}
    if auto:
        rec["annotations"]["auto_created_from_relation"] = ["reports"]
    return rec


def _ontology():
    classes = {
        DC + "Quantity": _cls("Quantity", desc="An amount."),
        DC + "Infrastructure": _cls("Infrastructure", desc="Built facilities."),
        FOAF_AGENT: _cls("Agent"),
        M + "CostOfSales": _cls("CostOfSales", FOAF_AGENT, "Auto-created from relation 'reports'", True),
        M + "BestForYou": _cls("BestForYou", FOAF_AGENT, "Auto-created from relation 'offers'", True),
        M + "MineSite": _cls("MineSite", DC + "Infrastructure", "Auto-created from relation 'operates'", True),
        M + "Reviewed": _cls("Reviewed", FOAF_AGENT, "A real definition.", True),
        M + "BestChild": _cls("BestChild", M + "BestForYou", "child"),
    }
    props = {
        M + "reports": {"labels": ["reports"], "descriptions": ["An org reports its cost of sales."],
                        "domain": [_ref(FOAF_AGENT)], "range": [_ref(M + "CostOfSales")]},
        M + "offers": {"labels": ["offers"], "domain": [_ref(FOAF_AGENT)], "range": [_ref(M + "BestForYou")]},
        "http://old#uses": {"labels": ["uses"], "domain": [_ref(FOAF_AGENT)],
                            "range": [_ref(M + "BestForYou"), _ref(DC + "Quantity")]},
    }
    instances = {M + "i1": {"types": [_ref(M + "BestForYou")], "direct_types": [_ref(M + "BestForYou")]}}
    return {"classes_dict": classes, "object_properties_dict": props, "instances_dict": instances}


def test_endpoint_classes_only_takes_placeholder_ones():
    onto = _ontology()
    got = ecr.endpoint_classes(onto["classes_dict"])
    assert set(got) == {M + "CostOfSales", M + "BestForYou", M + "MineSite"}
    assert ecr.endpoint_classes(onto["classes_dict"], only=[M + "MineSite"]) == [M + "MineSite"]


def test_items_carry_current_parent_and_relations():
    onto = _ontology()
    items = ecr.build_items([M + "CostOfSales"], onto["classes_dict"], onto["object_properties_dict"])
    assert items[0]["class"] == "CostOfSales" and items[0]["parent"] == "Agent"
    assert "Agent --reports--> CostOfSales (An org reports its cost of sales.)" == items[0]["relations"]


def test_parse_decisions_ignores_bad_rows():
    raw = [{"id": 0, "verdict": "KEEP", "parent": "Quantity", "definition": " An  amount. "},
           {"id": 1, "verdict": "maybe"}, {"id": 9, "verdict": "drop"}, {"id": "x"}, "junk"]
    assert ecr.parse_decisions(raw, 2) == {0: {"verdict": "keep", "parent": "Quantity", "definition": "An amount."}}


def test_apply_reparents_defines_and_drops():
    onto = _ontology()
    classes = onto["classes_dict"]
    roots = {"quantity": DC + "Quantity", "infrastructure": DC + "Infrastructure"}
    items = ecr.build_items([M + "CostOfSales", M + "BestForYou", M + "MineSite"], classes,
                            onto["object_properties_dict"])
    items[0]["decision"] = {"verdict": "keep", "parent": "Quantity",
                            "definition": "The direct cost of the goods a company sold."}
    items[1]["decision"] = {"verdict": "drop", "parent": "", "definition": ""}
    items[2]["decision"] = {"verdict": "keep", "parent": "CURRENT", "definition": "A place where ore is mined."}
    stats = ecr.apply_decisions(onto, items, roots, {M + "offers"})
    assert classes[M + "CostOfSales"]["superclasses"][0]["iri"] == DC + "Quantity"
    assert classes[M + "CostOfSales"]["descriptions"] == ["The direct cost of the goods a company sold."]
    assert classes[M + "MineSite"]["superclasses"][0]["iri"] == DC + "Infrastructure"
    assert M + "BestForYou" not in classes
    # child and instance move to the dropped class's parent
    assert classes[M + "BestChild"]["superclasses"][0]["iri"] == FOAF_AGENT
    assert onto["instances_dict"][M + "i1"]["types"][0]["iri"] == FOAF_AGENT
    # relation created this run is removed; an older one only loses that end
    props = onto["object_properties_dict"]
    assert M + "offers" not in props
    assert [r["iri"] for r in props["http://old#uses"]["range"]] == [DC + "Quantity"]
    assert stats["kept"] == 2 and stats["dropped"] == 1 and stats["reparented"] == 1
    assert stats["defined"] == 2 and stats["relations_removed"] == 1


def test_unknown_parent_keeps_current_and_is_counted():
    onto = _ontology()
    items = ecr.build_items([M + "CostOfSales"], onto["classes_dict"], onto["object_properties_dict"])
    items[0]["decision"] = {"verdict": "keep", "parent": "Nonsense", "definition": ""}
    stats = ecr.apply_decisions(onto, items, {"quantity": DC + "Quantity"})
    assert onto["classes_dict"][M + "CostOfSales"]["superclasses"][0]["iri"] == FOAF_AGENT
    assert stats["bad_parent"] == 1 and stats["defined"] == 0


class _Router:
    def __init__(self, spec, replies):
        self.spec, self.replies, self.total_cost_usd, self.calls = spec, replies, 0.0, []

    def task_spec(self, task):
        if self.spec is None:
            raise KeyError(task)
        return self.spec

    async def chat(self, task, system, user):
        self.calls.append(user)
        self.total_cost_usd += 0.001
        return SimpleNamespace(text=self.replies.pop(0))


def test_review_skips_when_task_missing_or_disabled():
    assert asyncio.run(ecr.review_endpoint_classes(_ontology(), _Router(None, []))) is None
    assert asyncio.run(ecr.review_endpoint_classes(_ontology(), _Router({"enabled": False}, []))) is None


def test_review_end_to_end_with_retry():
    onto = _ontology()
    first = json.dumps({"decisions": [
        {"id": 0, "verdict": "keep", "parent": "Quantity", "definition": "Cost of goods sold."},
        {"id": 1, "verdict": "drop"}]})  # id 2 missing -> retried alone
    retry = json.dumps({"decisions": [{"id": 0, "verdict": "keep", "parent": "CURRENT",
                                       "definition": "A place where ore is mined."}]})
    router = _Router({"provider": "p", "model": "m", "items_per_call": 4}, [first, retry])
    stats = asyncio.run(ecr.review_endpoint_classes(onto, router, log=lambda *_: None))
    assert stats["items"] == 3 and stats["retried"] == 1 and len(router.calls) == 2
    assert stats["kept"] == 2 and stats["dropped"] == 1 and stats["unanswered"] == 0
    assert "ROOTS:" in router.calls[0] and "- Quantity: An amount." in router.calls[0]
    assert len(stats["decisions"]) == 3


def test_failed_call_leaves_classes_untouched():
    onto = _ontology()
    router = _Router({"items_per_call": 20}, ["not json", "still not json", "no", "no"])
    stats = asyncio.run(ecr.review_endpoint_classes(onto, router, log=lambda *_: None))
    assert stats["unanswered"] == 3 and stats["dropped"] == 0
    assert onto["classes_dict"][M + "CostOfSales"]["superclasses"][0]["iri"] == FOAF_AGENT


def test_prompt_lists_roots_and_classes():
    system, user = prompt([{"id": 0, "class": "CostOfSales", "parent": "Agent", "relations": "x"}],
                          [{"LABEL": "Quantity", "DESCRIPTION": "An amount."}])
    assert "keep" in system and "drop" in system and "CURRENT" in system
    assert "- Quantity: An amount." in user and "0. class: CostOfSales | current parent: Agent" in user

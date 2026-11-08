"""Streamed Stage 2 (classes into the staging store per reply, relations /
instances / matches read back from llm_audit.jsonl) must give the same
Stage-3 result as the in-memory merge. Needs ONTOLOGY_BUILD_TEST_DSN."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from types import SimpleNamespace

import pytest

DSN = os.environ.get("ONTOLOGY_BUILD_TEST_DSN")

REPLIES = [
    {"MATCH NOT FOUND": [
        {"LABEL": "Refinery", "PARENT_LABEL": "x#Facility", "DESCRIPTION": "An oil refinery."},
        {"LABEL": "RefineryClosure", "PARENT_LABEL": "x#Event", "DESCRIPTION": "A refinery shutting down."},
        {"LABEL": "InflationRate", "PARENT_LABEL": "Metric", "DESCRIPTION": "Rate of price rises."}],
     "MATCH NOT FOUND RELATIONS": [{"LABEL": "operates", "DOMAIN": "x#Company", "RANGE": "Refinery"}],
     "MATCH NOT FOUND INSTANCES": [{"LABEL": "Jamnagar", "TYPE_LABEL": "Refinery"}],
     "MATCHES FOUND": [{"IRI": "x#Facility", "TEXT_SNIPPET": "plant"}]},
    {"MATCH NOT FOUND": [
        {"LABEL": "Refineries", "PARENT_LABEL": "x#Facility", "DESCRIPTION": "Oil refineries."},
        {"LABEL": "InflationRate", "PARENT_LABEL": "x#EconomicConcept", "DESCRIPTION": "Inflation."}],
     "MATCH NOT FOUND RELATIONS": [{"LABEL": "operates", "DOMAIN": "x#Company", "RANGE": "Refineries"},
                                   {"LABEL": "measures", "DOMAIN": "InflationRate", "RANGE": "x#Price"}],
     "MATCH NOT FOUND INSTANCES": [{"LABEL": "Jamnagar", "TYPE_LABEL": "Refineries"},
                                   {"LABEL": "Ruwais", "TYPE_LABEL": "Refinery"}],
     "MATCHES FOUND": [{"IRI": "x#Facility", "TEXT_SNIPPET": "site"}, {"IRI": "x#Event", "TEXT_SNIPPET": "x"}]},
]
ONTO = {"classes_dict": {f"x#{n}": {"labels": [n], "name": n}
                         for n in ("Facility", "Event", "EconomicConcept", "Company", "Price")}}
SPEC = {"match_dedup": {"method": "pairwise", "audit": False, "t_lo": 0.85, "embed_concurrency": 2},
        "class_proposal": {"junk_filter": False, "adverse_event_taxonomy": False}}


class _Router:
    def task_spec(self, task):
        return SPEC.get(task, {})

    async def chat(self, task, system, user):
        return SimpleNamespace(text='{"decisions": []}')


class _Emb:
    total_cost_usd = 0.0

    def __init__(self, **_):
        pass

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 1024
            v[sum(map(ord, re.findall(r"[a-z]+", t.lower())[0])) % 200] = 1.0
            v[200 + sum(map(ord, t)) % 800] = 0.3
            out.append(v)
        return out


def _canon(result):
    return {
        "classes": sorted((c["LABEL"], c["PARENT_LABEL"]) for c in result["MATCH NOT FOUND"]),
        "relations": sorted((r["LABEL"], r["DOMAIN"], r["RANGE"]) for r in result["MATCH NOT FOUND RELATIONS"]),
        "instances": sorted((i["LABEL"], i["TYPE_LABEL"]) for i in result["MATCH NOT FOUND INSTANCES"]),
        "matches": sorted(m["IRI"] for m in result["MATCHES FOUND"]),
    }


@pytest.mark.skipif(not DSN, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database")
def test_streamed_stage2_matches_in_memory_merge(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATABASE_URL", DSN)
    from backend.app.db.engine import reset_engine_cache

    reset_engine_cache()
    import backend.app.services.embeddings as emb_mod
    import backend.app.services.pipeline_llm as pl
    from backend.app.helpers.ontology_pruning import merge_llm_jsons_recursive

    monkeypatch.setattr(emb_mod, "Embedder", _Emb)
    router = _Router()

    async def go():
        a_dir, b_dir = tmp_path / "pytest-inmem", tmp_path / "pytest-stream"
        a_dir.mkdir(), b_dir.mkdir()
        merged = merge_llm_jsons_recursive(copy.deepcopy(REPLIES))
        out_a = await pl._pairwise_stage3(router, merged, ONTO, a_dir / "llm_audit.jsonl")

        audit_b = b_dir / "llm_audit.jsonl"
        sink = await pl._Stage2Sink.open(router, ONTO, audit_b)
        assert sink is not None
        for i, reply in enumerate(copy.deepcopy(REPLIES)):
            pl._append_audit(audit_b, i, "class_proposal", "doc", reply)
            await sink.add(i, reply)
        audit_b.open("a").write(json.dumps({"chunk_idx": -1, "task": "table_extract", "payload": {}}) + "\n")
        out_b = await pl._pairwise_stage3(router, None, ONTO, audit_b, store=sink.store)
        return out_a, out_b

    out_a, out_b = asyncio.run(go())
    assert _canon(out_a) == _canon(out_b)
    got = _canon(out_b)
    # Refinery / Refineries fold to one class; references to either follow it.
    assert [c for c in got["classes"] if c[0].lower().startswith("refiner")] == [("Refinery", "x#Facility"),
                                                                                 ("RefineryClosure", "x#Event")]
    assert ("operates", "x#Company", "Refinery") in got["relations"]
    assert ("operates", "x#Company", "Refineries") not in got["relations"]
    # InflationRate was proposed under Metric and x#EconomicConcept: the existing class wins.
    assert ("InflationRate", "x#EconomicConcept") in got["classes"]


JUDGED = [
    {"MATCH NOT FOUND": [
        {"LABEL": "WeightLossDrug", "PARENT_LABEL": "x#Drug", "DESCRIPTION": "A drug for obesity."},
        {"LABEL": "Refinery Unit", "PARENT_LABEL": "x#Facility", "DESCRIPTION": "Part of a refinery."}],
     "MATCH NOT FOUND RELATIONS": [{"LABEL": "sells", "DOMAIN": "x#Company", "RANGE": "WeightLossDrug"}]},
    {"MATCH NOT FOUND": [
        {"LABEL": "WeightLossMedication", "PARENT_LABEL": "x#Drug", "DESCRIPTION": "A medicine for weight loss."},
        {"LABEL": "Glp1Pen", "PARENT_LABEL": "WeightLossMedication", "DESCRIPTION": "An injector pen."},
        {"LABEL": "Refining Plant", "PARENT_LABEL": "x#Facility", "DESCRIPTION": "A refinery."}],
     "MATCH NOT FOUND RELATIONS": [{"LABEL": "treats", "DOMAIN": "WeightLossMedication", "RANGE": "x#Obesity"}],
     "MATCH NOT FOUND INSTANCES": [{"LABEL": "Wegovy", "TYPE_LABEL": "WeightLossMedication"}]},
]


class _SameRouter(_Router):
    """The pair judge calls every pair the same."""
    async def chat(self, task, system, user):
        return SimpleNamespace(text=json.dumps({"decisions": [{"id": i, "same": True} for i in range(50)]}))


class _GroupEmb(_Emb):
    """Weight-loss labels share one vector, refinery labels another."""
    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 1024
            low = t.lower()
            v[0 if "weight" in low else 1 if "refin" in low else 2 + sum(map(ord, t)) % 1000] = 1.0
            out.append(v)
        return out


@pytest.mark.skipif(not DSN, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database")
def test_judged_merge_redirects_every_reference(monkeypatch, tmp_path) -> None:
    """A merge the JUDGE made (labels that do not fold together) must redirect
    relation endpoints, instance types and child parents to the surviving
    class, and a merge onto an existing class must become a match. Before
    2026-11-08 the rebuild read rows without `merged_into`, so only same-
    spelling folds were redirected and prune-expand re-created the merged-away
    class from the relation endpoint (WeightLossMedication in the pharma build)."""
    monkeypatch.setenv("DATABASE_URL", DSN)
    from backend.app.db.engine import reset_engine_cache

    reset_engine_cache()
    import backend.app.services.embeddings as emb_mod
    import backend.app.services.pipeline_llm as pl
    from backend.app.helpers.ontology_pruning import merge_llm_jsons_recursive

    monkeypatch.setattr(emb_mod, "Embedder", _GroupEmb)
    onto = {"classes_dict": {f"x#{n}": {"labels": [n], "name": n}
                             for n in ("Drug", "Company", "Obesity", "Facility", "Refinery")}}
    out = asyncio.run(pl._pairwise_stage3(_SameRouter(), merge_llm_jsons_recursive(copy.deepcopy(JUDGED)),
                                          onto, tmp_path / "llm_audit.jsonl"))
    got = _canon(out)
    labels = [c[0] for c in got["classes"]]
    assert labels.count("WeightLossDrug") + labels.count("WeightLossMedication") == 1
    survivor = "WeightLossDrug" if "WeightLossDrug" in labels else "WeightLossMedication"
    gone = "WeightLossMedication" if survivor == "WeightLossDrug" else "WeightLossDrug"
    flat = json.dumps(got)
    assert f'"{gone}"' not in flat
    assert ("treats", survivor, "x#Obesity") in got["relations"]
    assert ("Wegovy", survivor) in got["instances"]
    assert ("Glp1Pen", survivor) in got["classes"]
    # Refinery Unit / Refining Plant were judged the same as the EXISTING x#Refinery.
    assert not {"Refinery Unit", "Refining Plant"} & set(labels)
    assert "x#Refinery" in got["matches"]

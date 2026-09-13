"""Cascade graph seeding (named entities -> classes -> artifacts), the
entity-relationship-only walk, and publications kept out of seeding."""
from __future__ import annotations

import asyncio
import contextlib
import uuid

from backend.app.services import retrieval as rt
from backend.app.services import retrieval_sql as rsql

E1, E2, E3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.0] for _ in texts]


def _patch(monkeypatch, *, names=None, classes=(), artifacts=(), sources=("TechCrunch", "The Verge")):
    @contextlib.asynccontextmanager
    async def scope():
        yield None

    calls = {"names": [], "classes": 0, "artifacts": 0}

    async def srcs(session):
        return list(sources)

    async def by_name(session, term, **kw):
        calls["names"].append(term)
        return (names or {}).get(term, [])

    async def by_class(session, terms, vecs, q, **kw):
        calls["classes"] += 1
        return list(classes), len(classes)

    async def by_artifact(session, q, **kw):
        calls["artifacts"] += 1
        return list(artifacts)

    monkeypatch.setattr(rt, "session_scope", scope)
    monkeypatch.setattr(rsql, "document_sources", srcs)
    monkeypatch.setattr(rsql, "match_entities_by_name_or_alias", by_name)
    monkeypatch.setattr(rsql, "class_seed_entities", by_class)
    monkeypatch.setattr(rsql, "artifact_seed_entities", by_artifact)
    return calls


def _run(parsed, aliases=None):
    return asyncio.run(rt._cascade_entity_seeds(parsed, [0.0], aliases or {}, FakeEmbedder()))


def test_named_entities_win_and_later_tiers_do_not_run(monkeypatch):
    calls = _patch(monkeypatch, names={"FTX": [E1]}, classes=[E2], artifacts=[E3])
    seeds, tier = _run({"entities": ["FTX"], "classes": ["fraud"]})
    assert (seeds, tier) == ([E1], "named")
    assert calls["classes"] == 0 and calls["artifacts"] == 0


def test_publications_are_not_seeds_so_classes_take_over(monkeypatch):
    # "as reported by TechCrunch and The Verge" -- the measured failure shape.
    calls = _patch(monkeypatch, names={"TechCrunch": [E1]}, classes=[E2])
    seeds, tier = _run({"entities": ["TechCrunch", "The Verge"], "classes": ["crypto exchange"]})
    assert (seeds, tier) == ([E2], "classes")
    assert calls["names"] == []


def test_aliases_of_a_named_term_are_tried(monkeypatch):
    calls = _patch(monkeypatch, names={"Mounjaro": [], "tirzepatide": [E1]})
    seeds, tier = _run({"entities": ["Mounjaro"], "classes": []}, {"Mounjaro": ["tirzepatide"]})
    assert (seeds, tier) == ([E1], "named")
    assert calls["names"] == ["Mounjaro", "tirzepatide"]


def test_artifacts_are_the_last_resort(monkeypatch):
    _patch(monkeypatch, artifacts=[E3])
    assert _run({"entities": [], "classes": ["individual"]}) == ([E3], "artifacts")


def test_nothing_found_reports_none(monkeypatch):
    _patch(monkeypatch)
    assert _run({"entities": [], "classes": []}) == ([], "none")


def test_entity_walk_sql_only_follows_entity_relationships():
    sql = rsql._ENTITY_BFS_SQL.text
    assert "gr.source_node_type = 'entity'" in sql
    assert "gr.target_node_type = 'entity'" in sql
    assert "gr.source_node_type = 'entity'" not in rsql._BFS_SQL.text


def test_defaults_are_the_new_design():
    import yaml
    qa = yaml.safe_load(open("config/config.example.yaml"))["qa"]
    assert qa["graph_seeding"] == "cascade"
    assert qa["graph_traversal"] == "entity_relationships"
    assert qa["global_artifact_search"] is True


def test_code_fallbacks_match_the_example_config():
    # A config.yaml missing these keys must not silently keep the old design.
    import inspect
    src = inspect.getsource(rt.retrieve_and_answer)
    assert '_qa_cfg("graph_seeding", "cascade")' in src
    assert '_qa_cfg("graph_traversal", "entity_relationships")' in src
    assert '_qa_cfg("global_artifact_search", True)' in src

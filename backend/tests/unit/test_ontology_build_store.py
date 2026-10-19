"""Staging store: pure helpers always; the pgvector round trip only when
ONTOLOGY_BUILD_TEST_DSN points at a migrated database (e.g. the bench docker
Postgres), so the default suite stays database-free."""
from __future__ import annotations

import asyncio
import os
import re

import pytest

from backend.app.services.ontology_build_store import BuildStore, embed_text, fold_key


@pytest.mark.parametrize("a,b", [
    ("Refinery", "refineries"), ("Refinery", "Refinery "), ("FoodSecurity", "Food security"),
    ("GLP-1 receptor agonist", "GLP1ReceptorAgonists"), ("Tariffs", "tariff"),
])
def test_fold_key_merges_spelling_variants(a, b) -> None:
    assert fold_key(a) == fold_key(b)


@pytest.mark.parametrize("a,b", [("Glass", "Gla"), ("Gas", "Ga"), ("Refinery", "Refining")])
def test_fold_key_keeps_distinct_words_apart(a, b) -> None:
    assert fold_key(a) != fold_key(b)


def test_embed_text_splits_camel_case_and_trims_description() -> None:
    t = embed_text("RareEarthElement", "A group of seventeen metallic elements " + "x " * 50)
    assert t.startswith("Rare Earth Element: A group of seventeen")
    assert len(t.split()) <= 3 + 20
    assert embed_text("Refinery", None) == "Refinery"


DSN = os.environ.get("ONTOLOGY_BUILD_TEST_DSN")


class _Emb:
    """Deterministic 1024-d vectors: labels sharing a first word are close."""
    total_cost_usd = 0.0

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 1024
            v[hash(re.findall(r"[a-z]+", t.lower())[0]) % 200] = 1.0
            v[200 + hash(t) % 800] = 0.3
            out.append(v)
        return out


@pytest.mark.skipif(not DSN, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database")
def test_store_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DSN)
    from backend.app.db.engine import reset_engine_cache

    reset_engine_cache()

    async def go():
        st = BuildStore("pytest-store")
        await st.clear()
        await st.upsert("class", [{"label": "Refinery"}, {"label": "refineries"}, {"label": "Tariff"}], chunk_ref=1)
        await st.upsert("class", [{"label": "Refinery", "description": "plant"}], chunk_ref=2)
        await st.upsert("existing_class", [{"label": "Refinery Plant", "iri": "x#RefineryPlant"}])
        counts = await st.counts()
        assert counts["class:kept"] == 2 and counts["class:kept:occurrences"] == 3
        assert await st.embed_missing(_Emb()) == 3
        # Concurrent first use must create the run's index once (2026-10-19:
        # neighbours' parallel lookups raced into a pg_class unique violation).
        await asyncio.gather(*[st.ensure_index("existing_class") for _ in range(4)])
        pairs =[p async for batch in st.neighbours("class", "existing_class", k=3, min_cosine=0.5) for p in batch]
        assert any(p.cosine > 0.9 for p in pairs)
        same = [p async for batch in st.neighbours("class", "class", k=3, min_cosine=-1) for p in batch]
        assert all(p.a_id < p.b_id for p in same)
        await st.clear()
        assert await st.counts() == {}

    asyncio.run(go())


def test_scalar_collapses_cross_chunk_lists():
    # merge_llm_jsons_recursive turns differing per-chunk values into lists;
    # the staging columns are text (2026-10-18 smoke-test crash).
    from backend.app.services.ontology_build_store import _scalar

    assert _scalar(["", "A#Metric", "A#EconomicConcept"]) == "A#Metric"
    assert _scalar("A#Metric") == "A#Metric"
    assert _scalar(None) is None and _scalar([]) is None


def test_choose_parent_prefers_existing_then_most_voted() -> None:
    from backend.app.services.ontology_build_store import choose_parent

    row = {"parent": "Metric", "extra": {"parents": {"Metric": 3, "x#EconomicConcept": 1}}}
    assert choose_parent(row, {"x#EconomicConcept"}) == "x#EconomicConcept"   # an existing class wins
    assert choose_parent(row) == "Metric"                                     # else the most proposed
    assert choose_parent({"parent": "B", "extra": {"parents": {"B": 1, "A": 1}}}) == "B"   # ties: first seen
    assert choose_parent({"parent": "C", "extra": {"parents": {"B": 1, "A": 1}}}) == "A"   # then alphabetical
    assert choose_parent({"parent": "Metric", "extra": {}}) == "Metric"        # no votes: stored parent
    assert choose_parent({"parent": None, "extra": {"parents": {"NONE": 2}}}) is None


@pytest.mark.skipif(not DSN, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database")
def test_upsert_counts_parent_votes(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DSN)
    from backend.app.db.engine import reset_engine_cache

    reset_engine_cache()

    async def go():
        st = BuildStore("pytest-votes")
        await st.clear()
        await st.upsert("class", [{"label": "Inflation", "parent": "Metric"}])
        await st.upsert("class", [{"label": "Inflation", "parent": "EconomicConcept"}])
        await st.upsert("class", [{"label": "inflation", "parent": "EconomicConcept"}])
        await st.upsert("class", [{"label": "Inflation"}])                 # no parent: no vote
        rows = [r async for batch in st.iter_rows("class", "kept", 10) for r in batch]
        assert len(rows) == 1 and rows[0]["extra"]["parents"] == {"Metric": 1, "EconomicConcept": 2}
        await st.clear()

    asyncio.run(go())

"""structured.db_load + sql_exec against a real migrated Postgres.

Skipped unless ONTOLOGY_BUILD_TEST_DSN points at a migrated database whose
name ends in ``_test`` (the test TRUNCATEs graph tables). CI runs it on Linux
with a pgvector service container.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

from backend.app.services.structured import cache as tc
from backend.app.services.structured.profiler import profile_folder
from backend.tests.unit.test_structured_profiler import COMPANIES, PRICES, FakeRouter, _make, _write

DSN = os.environ.get("ONTOLOGY_BUILD_TEST_DSN", "")
_SAFE = urlparse(DSN).path.rstrip("/").endswith("_test") if DSN else False
pytestmark = pytest.mark.skipif(
    not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")


@pytest.fixture
async def db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DSN)
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import drop_data_tables

    reset_engine_cache()
    async with session_scope() as s:
        await drop_data_tables(s)
        await s.execute(text("TRUNCATE graphrag.graph_relationships, graphrag.intelligence_artifacts, "
                             "graphrag.entities, graphrag.ontology_classes CASCADE"))
        cls = (await s.execute(text(
            "INSERT INTO graphrag.ontology_classes (iri, namespace, label) VALUES "
            "('https://ex.org/o#CommodityThing', 'https://ex.org/o#', 'CommodityThing') RETURNING id"))).scalar()
        for name in ("Copper", "Acme Corporation"):
            norm = name.lower()
            await s.execute(text(
                "INSERT INTO graphrag.entities (entity_identifier, iri, name, normalized_name, class_id) "
                "VALUES (:i, :i, :n, :nn, :c)"), {"i": f"https://ex.org/e#{norm}", "n": name, "nn": norm, "c": cls})
    yield session_scope
    async with session_scope() as s:
        await drop_data_tables(s)
    await dispose_engine()


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    _write(root, "worldbank/pink-sheet/s1/manifest.json", b"{}")
    _write(root, "worldbank/pink-sheet/s1/normalized/prices.csv", PRICES)
    _write(root, "crosswalk/companies.csv", COMPANIES)
    return root


async def _profile(data, tmp_path):
    return await profile_folder(data, _make(), cache=tc.MetadataCache(user_dir=tmp_path / "c"),
                                log=lambda *_: None)


async def test_load_link_and_rerun_is_noop(db, data, tmp_path):
    from sqlalchemy import text

    from backend.app.services.structured.db_load import load_and_link, registered_tables

    run = await _profile(data, tmp_path)
    s1 = await load_and_link(run, root_name="data", log=lambda *_: None)
    assert sorted(s1.loaded) == ["companies", "prices"] and s1.failed == [] and s1.rows == 4
    async with db() as s:
        n = (await s.execute(text('SELECT count(*), sum(value) FROM graphrag."prices"'))).one()
        assert n[0] == 3 and float(n[1]) == 8662.5
        cik = (await s.execute(text('SELECT cik FROM graphrag."companies"'))).scalar()
        assert cik == "0000320193"                        # leading zeros kept
        reg = await registered_tables(s)
        assert set(reg) == {"companies", "prices"}
        kinds = dict((await s.execute(text(
            "SELECT relationship_type, count(*) FROM graphrag.graph_relationships GROUP BY 1"))).all())
    assert kinds.get("tableEntity", 0) >= 2          # Copper (exact) + Acme Corp (short form/fuzzy)
    assert kinds.get("tableColumn", 0) >= 1          # commodity -> CommodityThing
    s2 = await load_and_link(run, root_name="data", log=lambda *_: None)
    assert s2.loaded == [] and sorted(s2.unchanged) == ["companies", "prices"]
    async with db() as s:
        again = dict((await s.execute(text(
            "SELECT relationship_type, count(*) FROM graphrag.graph_relationships GROUP BY 1"))).all())
    assert again == kinds                             # edges rebuilt, not duplicated


async def test_changed_file_reloads_and_marks_old_node_stale(db, data, tmp_path):
    from sqlalchemy import text

    from backend.app.services.structured.db_load import load_and_link

    await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    _write(data, "worldbank/pink-sheet/s1/normalized/prices.csv", PRICES + b"Copper,2024-02-01,8600\n")
    s2 = await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    assert s2.loaded == ["prices"]
    async with db() as s:
        st = sorted((await s.execute(text(
            "SELECT status FROM graphrag.intelligence_artifacts WHERE title = 'prices'"))).scalars().all())
        n = (await s.execute(text('SELECT count(*) FROM graphrag."prices"'))).scalar()
    assert st == ["ACTIVE", "STALE"] and n == 4


async def test_same_name_from_other_dataset_is_prefixed(db, data, tmp_path):
    from backend.app.services.structured.db_load import load_and_link

    await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    other = tmp_path / "other"
    _write(other, "x/companies.csv", b"company,cik,commodity\nZeta,0000000001,Gold\n")
    run2 = await profile_folder(other, _make(), cache=tc.MetadataCache(user_dir=tmp_path / "c2"),
                                log=lambda *_: None)
    s = await load_and_link(run2, root_name="other", log=lambda *_: None)
    assert s.renamed == [("companies", "other_companies")] and s.loaded == ["other_companies"]


async def test_graph_table_name_is_never_overwritten(db, data, tmp_path):
    from backend.app.services.structured.db_load import load_and_link
    from backend.app.services.structured.metadata import rename_table

    run = await _profile(data, tmp_path)
    run.tables = [rename_table(run.by_name()["prices"], "prices", "entities")]
    run.sources["entities"] = run.sources["prices"]
    s = await load_and_link(run, root_name="data", log=lambda *_: None)
    assert s.loaded == [] and "not a structured-data table" in s.failed[0][1]


async def test_sql_exec_guard_and_cap(db, data, tmp_path):
    from backend.app.services.structured.db_load import load_and_link
    from backend.app.services.structured.sql_exec import run_sql

    await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    tables = {"prices", "companies"}
    r = await run_sql("SELECT commodity, AVG(value) AS v FROM prices GROUP BY commodity ORDER BY commodity",
                      {}, tables, max_rows=50)
    assert r.error is None and r.row_count == 2 and not r.truncated
    r = await run_sql("SELECT * FROM prices WHERE commodity = :c", {"c": "Crude oil"}, tables, max_rows=1)
    assert r.truncated and r.row_count == 1 and r.total_rows == 2
    for bad in ("DELETE FROM prices", "SELECT * FROM graphrag.entities"):
        r = await run_sql(bad, {}, tables, max_rows=5)
        assert r.error and r.rows == []
    r = await run_sql("SELECT nosuch FROM prices", {}, tables, max_rows=5)
    assert r.error and "nosuch" in r.error
    _ = FakeRouter


async def test_extract_entities_hook_reads_cache_only(db, data, tmp_path, monkeypatch):
    """prune-expand profiled the tables; extract-entities must not call the LLM."""
    from sqlalchemy import text

    from backend.app.services.structured.graph_hook import run_structured_load

    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "cache"))
    await profile_folder(data, _make(), cache=tc.MetadataCache(), log=lambda *_: None)   # "prune-expand"
    silent = FakeRouter({})

    async def fake_embed(texts):
        return [[0.01] * 1024 for _ in texts]

    profile, summary = await run_structured_load(data, router=silent, embed=fake_embed)
    assert silent.calls == [] and profile["from_cache"] == 2 and profile["llm_calls"] == 0
    assert sorted(summary.loaded) == ["companies", "prices"]
    async with db() as s:
        n = (await s.execute(text("SELECT count(*) FROM graphrag.intelligence_artifacts "
                                  "WHERE artifact_type = 'TableMetadata' AND embedding IS NOT NULL"))).scalar()
    assert n == 2


async def test_retrieval_finds_table_via_entity_and_runs_stored_sql(db, data, tmp_path):
    from sqlalchemy import text

    from backend.app.services.structured.db_load import load_and_link
    from backend.app.services.structured.retrieval_step import candidate_tables, structured_evidence

    await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    async with db() as s:
        copper = (await s.execute(text(
            "SELECT id FROM graphrag.entities WHERE normalized_name = 'copper'"))).scalar()
    cands = await candidate_tables([copper], [], None)
    assert {c["table"] for c in cands} >= {"prices"}
    assert all(c["via"][0].startswith("entity:") for c in cands)
    assert await candidate_tables([], [], None) == []     # nothing reached, no vector -> none

    router = FakeRouter({"table_query_select": [{"needed": True, "calls": [
        {"table": "prices", "query_id": "by_first", "params": {"v": "Copper"}},
        {"table": "prices", "new_sql": "DELETE FROM prices", "params": {}}]},
        {"needed": False}]})
    ev, plan = await structured_evidence("copper price?", router=router, entity_ids=[copper],
                                         class_ids=[], qvec=None)
    assert len(ev) == 1 and ev[0]["rows"] == [["Copper", "2024-01-01", "8500"]]
    assert plan["retried"] and "rejected" in plan["calls"][1]["error"]
    async with db() as s:                                 # the guard blocked the write
        assert (await s.execute(text('SELECT count(*) FROM graphrag."prices"'))).scalar() == 3

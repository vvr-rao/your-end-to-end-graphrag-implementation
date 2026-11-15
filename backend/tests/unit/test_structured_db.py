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

    profile, summary = await run_structured_load(data, router=silent, embed=fake_embed, refine=False)
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
        {"table": "prices", "new_sql": "DELETE FROM prices", "params": {}}]},   # never run
        {"needed": False}]})
    ev, plan = await structured_evidence("copper price?", router=router, entity_ids=[copper],
                                         class_ids=[], qvec=None)
    assert len(ev) == 1 and ev[0]["rows"] == [["Copper", "2024-01-01", "8500"]]
    assert plan["retried"] and "neither query_id nor query" in plan["calls"][1]["error"]
    async with db() as s:                                 # model-written SQL is never run
        assert (await s.execute(text('SELECT count(*) FROM graphrag."prices"'))).scalar() == 3


async def test_export_edit_import_updates_node_edges_and_cache(db, data, tmp_path):
    from sqlalchemy import text

    from backend.app.services.structured.db_load import load_and_link
    from backend.app.services.structured.review import export_tables, import_markdown

    cache = tc.MetadataCache(user_dir=tmp_path / "c")
    run = await profile_folder(data, _make(), cache=cache, log=lambda *_: None)
    await load_and_link(run, root_name="data", cache=cache, log=lambda *_: None)
    written, missing = await export_tables(tmp_path / "md", ["prices", "nope"])
    assert [p.name for p in written] == ["prices.md"] and missing == ["nope"]
    md = written[0]
    original = md.read_text(encoding="utf-8")
    spec_sel = '"select": [\n    "commodity",\n    "period",\n    "value"\n  ],'
    assert spec_sel in original and "```json" in original
    assert 'SELECT "commodity", "period", "value" FROM "prices" WHERE "commodity" = CAST(:v AS text)' in original

    # Unchanged file: nothing to do.
    (rep,) = await import_markdown([md], cache=cache)
    assert not rep.changed and not rep.applied and rep.errors == []

    edited = original.replace("prices table", "Monthly commodity spot prices", 1).replace(
        spec_sel, '"select": ["period", "value"], "order_by": [{"column": "period", "dir": "asc"}],')
    md.write_text(edited, encoding="utf-8")
    (dry,) = await import_markdown([md], dry_run=True, cache=cache)
    assert dry.errors == [] and dry.changed and not dry.applied
    assert "+" in dry.diff and 'ORDER BY "period" ASC' in dry.diff       # SQL rebuilt from the spec
    async with db() as s:
        before = (await s.execute(text("SELECT text FROM graphrag.intelligence_artifacts "
                                       "WHERE artifact_type = 'TableMetadata' AND title = 'prices'"))).scalar()
    assert "Monthly commodity spot prices" not in before          # dry run changed nothing

    (rep,) = await import_markdown([md], cache=cache)
    assert rep.applied and rep.errors == [] and rep.class_edges >= 1
    async with db() as s:
        row = (await s.execute(text(
            "SELECT text, extra_metadata FROM graphrag.intelligence_artifacts "
            "WHERE artifact_type = 'TableMetadata' AND title = 'prices' AND status = 'ACTIVE'"))).one()
        n_edges = (await s.execute(text(
            "SELECT count(*) FROM graphrag.graph_relationships gr JOIN graphrag.intelligence_artifacts a "
            "ON a.id = gr.source_node_id WHERE a.title = 'prices'"))).scalar()
    assert "Monthly commodity spot prices" in row[0] and row[1]["edited"] is True
    assert n_edges == rep.class_edges + rep.entity_edges        # rebuilt, not duplicated
    entry = cache.get(row[1]["cache_key"], "prices")
    assert entry["edited"] is True and entry.get("profile")      # profile kept from the cache
    assert 'ORDER BY "period" ASC' in next(f for f in entry["filters"] if f["id"] == "by_first")["sql"]

    # A spec that does not compile is refused; nothing applied.
    md.write_text(edited.replace('"select": ["period", "value"]', '"select": ["period", "no_such_col"]'),
                  encoding="utf-8")
    (bad,) = await import_markdown([md], cache=cache)
    assert not bad.applied and any("no_such_col" in e for e in bad.errors)


async def test_load_stores_valid_values_and_retrieval_checks_them(db, data, tmp_path):
    from backend.app.services.structured.db_load import load_and_link, registered_tables
    from backend.app.services.structured.metadata import find_item
    from backend.app.services.structured.retrieval_step import structured_evidence

    await load_and_link(await _profile(data, tmp_path), root_name="data", log=lambda *_: None)
    async with db() as s:
        meta = (await registered_tables(s))["prices"]["meta"]
    dom = meta["value_domains"]
    assert dom["commodity"] == {"values": ["Copper", "Crude oil"]}
    assert dom["value"] == {"min": 80.5, "max": 8500}
    _sec, item = find_item(meta, "by_first")
    assert item["params"][0]["allowed_values"] == ["Copper", "Crude oil"]   # travels with the query

    async def ids():
        from sqlalchemy import text
        async with db() as s:
            return [(await s.execute(text(
                "SELECT id FROM graphrag.entities WHERE normalized_name = 'copper'"))).scalar()]

    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "by_first", "params": {"v": "COPPER"}},
                                   {"table": "prices", "query_id": "by_first", "params": {"v": "Gold"}}]},
        {"needed": False}]})
    ev, plan = await structured_evidence("copper?", router=router, entity_ids=await ids(),
                                         class_ids=[], qvec=None)
    assert len(ev) == 2 and ev[0]["params"] == {"v": "Copper"}
    assert ev[1]["query_id"] == "value_check" and "'Gold' is not one of the values" in ev[1]["text"]
    assert plan["calls"][0]["substituted"] == [{"param": "v", "from": "COPPER", "to": "Copper"}]
    bad = plan["calls"][1]
    assert bad["invalid_values"][0]["allowed_values"] == ["Copper", "Crude oil"] and "sql" not in bad
    assert "Crude oil" in router.calls[1][1]                    # valid values reached the retry

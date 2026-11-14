"""structured.analysis + structured.refine against a real migrated Postgres.

The table is a generic "reported values" table: every report restates the
values of earlier periods (comparative periods), so a value repeats across
the report columns. Nothing here is specific to any dataset.
"""

from __future__ import annotations

import json

import pytest

from backend.app.services.structured import cache as tc
from backend.app.services.structured.profiler import profile_folder
from backend.tests.unit.test_structured_db import _SAFE, DSN
from backend.tests.unit.test_structured_profiler import FakeRouter, _write

pytestmark = pytest.mark.skipif(
    not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")

COLS = ["entity", "item", "period_end", "value", "report_year", "report_date"]
REPORTS = {2023: ("2024-02-15", [2021, 2022, 2023]), 2024: ("2025-02-14", [2022, 2023, 2024])}


def _value(entity: str, item: str, year: int) -> float:
    return {"A": 1000, "B": 5000}[entity] + {"Sales": 100, "Income": 10}[item] * (year - 2020)


def _csv() -> bytes:
    rows = [",".join(COLS)]
    for ry, (rdate, years) in REPORTS.items():
        for e in ("A", "B"):
            for it in ("Sales", "Income"):
                for y in years:
                    rows.append(f"{e},{it},{y}-12-31,{_value(e, it, y)},{ry},{rdate}")
    return ("\n".join(rows) + "\n").encode()


def _profile_reply() -> dict:
    roles = {"entity": "identifier", "item": "dimension", "period_end": "time", "value": "measure",
             "report_year": "time", "report_date": "time"}      # report_year mislabelled as time
    return {
        "description": "values reported per entity", "grain": "one value per entity, item, period, report",
        "columns": [{"name": c, "description": f"the {c}", "role": r, "concept": None}
                    for c, r in roles.items()],
        "metrics": [
            # sums one period's value once per report that restates it
            {"id": "total_by_entity_item", "label": "Total", "description": "", "columns": ["value"],
             "concept": None, "params": [],
             "query": {"select": ["entity", "item", {"agg": "sum", "column": "value", "as": "total"}]}},
            # groups by the REPORT year: combines three periods per row
            {"id": "value_by_report_year", "label": "By report year", "description": "",
             "columns": ["value"], "concept": None,
             "params": [{"name": "entity", "description": "", "example": "A"},
                        {"name": "item", "description": "", "example": "Sales"},
                        {"name": "report_year", "description": "", "example": "2024"}],
             "query": {"select": ["entity", "report_year", {"agg": "sum", "column": "value", "as": "v"}],
                       "where": [{"column": "entity", "op": "=", "param": "entity"},
                                 {"column": "item", "op": "=", "param": "item"},
                                 {"column": "report_year", "op": "=", "param": "report_year"}]}},
        ],
        "filters": [{"id": "by_entity", "label": "By entity", "description": "", "columns": ["entity"],
                     "concept": None, "params": [{"name": "entity", "description": "", "example": "A"}],
                     "query": {"select": ["entity", "item", "period_end", "value"],
                               "where": [{"column": "entity", "op": "=", "param": "entity"}],
                               "order_by": [{"column": "period_end", "dir": "desc"}], "limit": 20}}],
        "queries": [],
    }


REFINE_REPLY = {
    "columns": [{"name": "report_year", "role": "attribute",
                 "description": "year of the report the value was published in (not the period)"}],
    "measures": [
        {"id": "sales", "label": "Sales", "description": "Sales of an entity for a period.",
         "value_column": "value", "item": {"column": "item", "values": ["Sales"]},
         "entity_columns": ["entity"], "period_columns": ["period_end"], "qualifier_columns": [],
         "latest_by": "report_date", "additive_over_time": True},
        {"id": "broken", "value_column": "no_such_column", "entity_columns": ["entity"]},
    ],
    "usage_notes": ["report_year and report_date identify the report, not the period.",
                    "Never sum value across reports."],
    "common_questions": [
        {"question": "Sales of an entity in a year", "query_id": "sales",
         "params": {"entity": "A", "period_end_from": "2023-01-01", "period_end_to": "2023-12-31"}},
        {"question": "unknown", "query_id": "nope", "params": {}},
    ],
}

REPAIR_REPLY = {"items": [{
    "id": "value_by_report_year", "params": [],
    "query": {"select": ["entity", "period_end", "value"],
              "where": [{"column": "entity", "op": "=", "param": "entity"},
                        {"column": "item", "op": "=", "param": "item"}],
              "distinct_on": ["period_end"],
              "order_by": [{"column": "report_date", "dir": "desc"}]}}]}


@pytest.fixture
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", DSN)
    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "cache"))
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import drop_data_tables

    reset_engine_cache()
    async with session_scope() as s:
        await drop_data_tables(s)
        await s.execute(text("TRUNCATE graphrag.graph_relationships, graphrag.intelligence_artifacts, "
                             "graphrag.entities, graphrag.ontology_classes CASCADE"))
    yield session_scope
    async with session_scope() as s:
        await drop_data_tables(s)
    await dispose_engine()


async def _fake_embed(texts):
    return [[0.01] * 1024 for _ in texts]


async def _load(tmp_path):
    from backend.app.services.structured.db_load import load_and_link

    root = tmp_path / "data"
    _write(root, "reports.csv", _csv())
    cache = tc.MetadataCache()
    run = await profile_folder(root, FakeRouter({"table_profile": [_profile_reply()]}), cache=cache,
                               log=lambda *_: None)
    await load_and_link(run, root_name="data", cache=cache, embed=_fake_embed, log=lambda *_: None)
    return cache


async def test_analysis_finds_grain_and_report_columns(db, tmp_path):
    from backend.app.services.structured.analysis import analyze_table, report_columns
    from backend.app.services.structured.db_load import registered_tables

    await _load(tmp_path)
    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
        a = await analyze_table(s, meta)
    assert a["row_count"] == 24 and a["grain_unique"]
    assert set(a["grain"]) == {"entity", "item", "period_end", "report_year"} or \
        set(a["grain"]) == {"entity", "item", "period_end", "report_date"}
    assert set(report_columns(a, "value")) == {"report_year", "report_date"}
    assert "period_end" not in report_columns(a)


async def test_refine_defines_measures_tests_queries_and_caches(db, tmp_path):
    from backend.app.services.structured import query_runner as qrun
    from backend.app.services.structured.db_load import registered_tables
    from backend.app.services.structured.md_io import apply_markdown, to_markdown
    from backend.app.services.structured.refine import REFINE_VERSION, refine_registered

    cache = await _load(tmp_path)
    router = FakeRouter({"table_refine": [REFINE_REPLY, REPAIR_REPLY]})
    reps = await refine_registered(router, cache=cache, embed=_fake_embed, log=lambda *_: None)
    assert [c[0] for c in router.calls] == ["table_refine", "table_refine"]     # define + one repair
    rep = reps[0]
    assert rep["measures"] == ["measure_sales"]
    assert rep["rejected_measures"][0]["measure"] == "broken"
    assert rep["repaired"] == ["value_by_report_year"]                # period mixing -> repaired
    assert [d["id"] for d in rep["dropped"]] == ["total_by_entity_item"]   # double count -> dropped
    assert "aggregates the same reported value more than once" in rep["dropped"][0]["error"]

    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert meta["refined"]["version"] == REFINE_VERSION
    assert next(c for c in meta["columns"] if c["name"] == "report_year")["role"] == "attribute"
    assert [q["query_id"] for q in meta["common_questions"]] == ["measure_sales"]
    assert meta["usage_notes"][1] == "Never sum value across reports."

    # the measure returns ONE value per period (the restated ones collapse)
    qr = await qrun.run("reports", {"entity": "A", "period_end_from": "2021-01-01",
                                    "period_end_to": "2024-12-31"}, query_id="measure_sales")
    assert qr.result.error is None
    vals = [(str(r[1]), float(r[2])) for r in qr.result.rows]
    assert vals == [(f"{y}-12-31", _value("A", "Sales", y)) for y in (2021, 2022, 2023, 2024)]

    # Markdown round-trip keeps the measure definition and usage notes
    back = apply_markdown(meta, to_markdown(meta))
    assert back["measures"][0]["measure"] == meta["measures"][0]["measure"]
    assert back["usage_notes"] == meta["usage_notes"]

    # second run: already refined -> no LLM call; the cache carries it too
    again = FakeRouter({})
    assert await refine_registered(again, cache=cache, embed=_fake_embed, log=lambda *_: None) == []
    assert again.calls == []
    cached = cache.get(meta["cache_key"], "reports")
    assert cached["refined"]["version"] == REFINE_VERSION and cached.get("profile")
    assert json.dumps(cached["measures"][0]["query"])          # stored spec, not SQL text only

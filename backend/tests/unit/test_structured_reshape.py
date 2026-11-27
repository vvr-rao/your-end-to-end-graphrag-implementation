"""Bounded changes retrieval may make to a stored query: lists of values for
an equality filter, and a reshape (aggregate / group / order / limit) of the
stored query's own rows. Generic fixtures only."""

from __future__ import annotations

import pytest

from backend.app.services.structured.reshape import (
    ReshapeError,
    listify,
    output_columns,
    reshape_sql,
)
from backend.app.services.structured.retrieval_step import cap_listings, expand_refs
from backend.app.services.structured.sql_exec import SqlResult
from backend.tests.unit.test_structured_db import _SAFE, DSN

SERIES_SQL = ('SELECT "item", "region", "period", "unit", "value" FROM "t_obs" '
              'WHERE (CAST(:item AS text) IS NULL OR "item" = CAST(:item AS text)) '
              'AND "unit" = CAST(:unit AS text) '
              'AND "period" BETWEEN CAST(:p_from AS text) AND CAST(:p_to AS text) '
              'ORDER BY "item" ASC, "period" ASC')
DECLARED = [{"name": "item", "type": "text", "optional": True}, {"name": "unit", "type": "text"},
            {"name": "p_from", "type": "text"}, {"name": "p_to", "type": "text"}]
TYPES = {"item": "text", "region": "text", "period": "text", "unit": "text", "value": "numeric"}
ROLES = {"item": "dimension", "region": "dimension", "period": "time", "unit": "dimension", "value": "measure"}


# ---------------------------------------------------------------- lists of values

def test_equality_param_given_a_list_becomes_any():
    params = {"item": ["a", "b"], "unit": "kg", "p_from": "1", "p_to": "2"}
    sql, declared, done = listify(SERIES_SQL, params, DECLARED)
    assert done == ["item"]
    assert 'CAST(:item AS text[]) IS NULL OR "item" = ANY(CAST(:item AS text[]))' in sql
    assert next(p for p in declared if p["name"] == "item")["type"] == "text[]"
    assert 'CAST(:unit AS text)' in sql                        # other params untouched


def test_one_value_list_is_just_that_value():
    params = {"item": ["a"], "unit": "kg", "p_from": "1", "p_to": "2"}
    sql, _declared, done = listify(SERIES_SQL, params, DECLARED)
    assert done == [] and sql == SERIES_SQL and params["item"] == "a"


def test_range_bound_cannot_take_a_list():
    with pytest.raises(ReshapeError, match="not an equality filter"):
        listify(SERIES_SQL, {"unit": "kg", "p_from": ["1", "2"], "p_to": "3"}, DECLARED)


def test_pattern_param_cannot_take_a_list():
    sql = "SELECT \"item\" FROM \"t_obs\" WHERE \"item\" ILIKE '%' || CAST(:name AS text) || '%'"
    with pytest.raises(ReshapeError):
        listify(sql, {"name": ["a", "b"]}, [{"name": "name", "type": "text"}])


def test_from_call_may_pool_several_earlier_calls():
    done = {1: [SqlResult(sql="", params={}, columns=["code"], rows=[["A"]])],
            2: [SqlResult(sql="", params={}, columns=["code"], rows=[["B"], ["A"]])]}
    call = {"table": "t", "query_id": "q",
            "params": {"code": {"from_call": [1, 2], "column": "code", "as_list": True}}}
    out, err = expand_refs(call, done)
    assert err is None and out[0]["params"]["code"] == ["A", "B"]


# ---------------------------------------------------------------- reshape

def _kinds():
    return output_columns(SERIES_SQL, TYPES, ROLES)


def test_output_columns_are_classified():
    assert _kinds() == {"item": "categorical", "region": "categorical", "period": "time",
                        "unit": "categorical", "value": "numeric"}


def test_order_and_limit_give_the_row_of_the_maximum():
    sql, how = reshape_sql(SERIES_SQL, {"order_by": [{"column": "value", "dir": "desc"}], "limit": 1},
                           _kinds(), max_rows=50)
    assert sql.startswith("SELECT base.* FROM (SELECT")
    assert sql.endswith('ORDER BY base."value" DESC NULLS LAST LIMIT 1')
    assert "value desc" in how


def test_aggregate_per_group():
    sql, _ = reshape_sql(SERIES_SQL, {"aggregate": [{"fn": "max", "column": "value"},
                                                    {"fn": "avg", "column": "value"}],
                                      "group_by": ["region"],
                                      "order_by": [{"column": "max_value", "dir": "desc"}]},
                         _kinds(), max_rows=50)
    assert 'SELECT base."region", MAX(base."value") AS "max_value", AVG(base."value") AS "avg_value"' in sql
    assert 'GROUP BY base."region" ORDER BY "max_value" DESC' in sql


@pytest.mark.parametrize("spec, msg", [
    ({"aggregate": [{"fn": "max", "column": "item"}]}, "not a numeric column"),
    ({"aggregate": [{"fn": "sum", "column": "value"}]}, "fn must be one of"),
    ({"group_by": ["value"]}, "is numeric"),
    ({"order_by": [{"column": "region"}]}, "is categorical"),
    ({"order_by": [{"column": "nope"}]}, "not a column"),
    ({"limit": 0}, "limit must be"),
    ({"where": "x"}, "not allowed"),
    ({"aggregate": [{"fn": "max", "column": "value"}], "order_by": [{"column": "value"}]}, "not a column"),
])
def test_reshape_refuses_what_it_may_not_do(spec, msg):
    with pytest.raises(ReshapeError, match=msg):
        reshape_sql(SERIES_SQL, spec, _kinds(), max_rows=50)


def test_listing_calls_are_capped_at_two():
    plan: dict = {}
    calls = [{"query_id": f"q{i}", "listing": True} for i in range(3)] + [{"query_id": "x"}]
    kept = cap_listings(calls, plan)
    assert [c["query_id"] for c in kept] == ["q0", "q1", "x"] and len(plan["listing_cut"]) == 1


# ---------------------------------------------------------------- against Postgres

db_only = pytest.mark.skipif(not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")


@pytest.fixture
async def pg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DSN)
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope

    reset_engine_cache()
    async with session_scope() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_obs"))
        await s.execute(text("CREATE TABLE graphrag.t_obs (item text, region text, period text, "
                             "unit text, value numeric)"))
        rows = [("a", "north", "2026M01", "kg", 10), ("a", "north", "2026M02", "kg", 30),
                ("a", "north", "2026M03", "kg", 20), ("b", "south", "2026M01", "kg", 5),
                ("b", "south", "2026M02", "kg", 7), ("c", "south", "2026M01", "kg", 100)]
        for r in rows:
            await s.execute(text("INSERT INTO graphrag.t_obs VALUES (:a, :b, :c, :d, :e)"),
                            dict(zip("abcde", r, strict=True)))
    yield session_scope
    async with session_scope() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_obs"))
    await dispose_engine()


def _by_table():
    from backend.app.services.structured.metadata import CATALOG_VERSION, sql_sha
    meta = {"table_name": "t_obs", "catalog_version": CATALOG_VERSION,
            "columns": [{"name": c, "type": TYPES[c], "role": ROLES[c]} for c in TYPES],
            "measures": [{"id": "series", "label": "series", "sql": SERIES_SQL, "params": DECLARED,
                          "verification": {"status": "passed", "sql_sha": sql_sha(SERIES_SQL)}}]}
    return {"t_obs": {"meta": meta}}


@db_only
async def test_reshaped_and_listed_calls_run_on_the_stored_rows(pg):
    from backend.app.services.structured.retrieval_step import _execute

    base = {"table": "t_obs", "query_id": "series",
            "params": {"item": "a", "unit": "kg", "p_from": "2026M01", "p_to": "2026M12"}}
    # the period of the maximum: order by value desc, limit 1
    rec, res, err = await _execute({**base, "reshape": {"order_by": [{"column": "value", "dir": "desc"}],
                                                        "limit": 1}},
                                   _by_table(), max_rows=50, timeout_ms=5000, all_tables={"t_obs"})
    assert err is None, rec
    assert res.rows[0][2] == "2026M02" and float(res.rows[0][4]) == 30
    # several items in one call, aggregated per item
    call = {**base, "params": {**base["params"], "item": ["a", "b"]},
            "reshape": {"aggregate": [{"fn": "max", "column": "value"}, {"fn": "count"}],
                        "group_by": ["item"]}}
    rec, res, err = await _execute(call, _by_table(), max_rows=50, timeout_ms=5000, all_tables={"t_obs"})
    assert err is None, rec
    assert rec["listed_params"] == ["item"]
    got = {r[0]: (float(r[1]), int(r[2])) for r in res.rows}
    assert got == {"a": (30.0, 3), "b": (7.0, 2)}                     # 'c' not read
    # a reshape that is not allowed never runs
    rec, res, err = await _execute({**base, "reshape": {"group_by": ["value"]}}, _by_table(),
                                   max_rows=50, timeout_ms=5000, all_tables={"t_obs"})
    assert res is None and err.startswith("reshape refused")


def test_selection_prompt_has_the_generic_ranking_rule():
    from backend.app.services.prompts import table_query_select

    system, _user = table_query_select("Which item had the largest value?", [])
    rule = system[system.index("WHICH IS LARGEST / SMALLEST"):]
    assert "prefer a stored ranking query" in rule
    assert "RESHAPE" in rule and "order_by" in rule
    assert "only a total" in rule


# ---------------------------------------------------------------- latest per item (2026-11-27)
# "most recent value" questions: items can have different latest periods, and a
# fixed period window drops an item whose newest value is older (sc80 q024).

def test_latest_keeps_one_row_per_item_at_its_latest_filled_period():
    kinds = output_columns(SERIES_SQL, TYPES, ROLES)
    sql, how = reshape_sql(SERIES_SQL, {"latest": {"per": ["item"], "by": "period"}}, kinds, max_rows=50)
    assert 'DISTINCT ON (base."item")' in sql
    assert 'ORDER BY base."item", base."period" DESC NULLS LAST' in sql
    assert 'WHERE base."value" IS NOT NULL' in sql                  # only rows with a value
    assert SERIES_SQL.rstrip(";") in sql                             # the stored rows, unchanged
    assert "latest period per item" in how


@pytest.mark.parametrize("spec, msg", [
    ({"latest": {"per": ["item"], "by": "unit"}}, "period"),         # not a period column
    ({"latest": {"per": ["value"], "by": "period"}}, "categorical"),
    ({"latest": {"per": ["nope"], "by": "period"}}, "output columns"),
    ({"latest": {"per": ["item"], "by": "period"}, "group_by": ["item"]}, "cannot be combined"),
])
def test_latest_refuses_what_it_may_not_do(spec, msg):
    kinds = output_columns(SERIES_SQL, TYPES, ROLES)
    with pytest.raises(ReshapeError, match=msg):
        reshape_sql(SERIES_SQL, spec, kinds, max_rows=50)


@db_only
async def test_latest_runs_on_the_stored_rows(pg):
    from sqlalchemy import text

    from backend.app.services.structured.retrieval_step import _execute
    async with pg() as s:      # b's newest period has no value: its latest FILLED period is 2026M02
        await s.execute(text("INSERT INTO graphrag.t_obs VALUES ('b', 'south', '2026M03', 'kg', NULL)"))
    call = {"table": "t_obs", "query_id": "series",
            "params": {"item": None, "unit": "kg", "p_from": "2000M01", "p_to": "2099M12"},
            "reshape": {"latest": {"per": ["item"], "by": "period"}}}
    rec, res, err = await _execute(call, _by_table(), max_rows=50, timeout_ms=5000, all_tables={"t_obs"})
    assert err is None, rec
    got = {r[0]: (r[2], float(r[4])) for r in res.rows}
    assert got == {"a": ("2026M03", 20.0), "b": ("2026M02", 7.0), "c": ("2026M01", 100.0)}


def test_selection_prompt_has_the_latest_per_item_rule():
    from backend.app.services.prompts import table_query_select

    system, _user = table_query_select("What is the most recent value for each item?", [])
    assert "MOST RECENT VALUE per item" in system and '"latest"' in system

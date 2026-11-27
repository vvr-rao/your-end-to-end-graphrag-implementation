"""structured.verify: the independent check of column metadata and stored
queries (task table_verify). The SHAPE text needs no database; the rest runs
against a real migrated Postgres. Nothing here is specific to any dataset."""

from __future__ import annotations

import json

import pytest

from backend.app.services.structured.verify import describe_spec, verify_enabled
from backend.tests.unit.test_structured_db import _SAFE
from backend.tests.unit.test_structured_profiler import FakeRouter
from backend.tests.unit.test_structured_refine import (  # noqa: F401  (db is a fixture)
    REFINE_REPLY,
    REPAIR_REPLY,
    _fake_embed,
    _load,
    db,
)

db_only = pytest.mark.skipif(not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")


# ---------------------------------------------------------------- SHAPE, from the spec alone

def test_shape_says_a_grouped_top_n_is_neither_a_total_nor_complete():
    spec = {"select": ["product", {"agg": "sum", "column": "amount_2024", "as": "total"}],
            "group_by": ["product"], "order_by": [{"column": "total", "dir": "desc"}], "limit": 20}
    shape = describe_spec(spec)
    assert shape.startswith("one row per product; computes SUM(amount_2024) as total")
    assert "filters: none (all rows)" in shape
    assert "only the first 20 rows (NOT a complete list or total)" in shape


def test_shape_separates_user_choices_from_fixed_values():
    spec = {"select": [{"agg": "sum", "column": "amount", "as": "total"}],
            "where": [{"column": "region", "op": "=", "param": "region"},
                      {"column": "kind", "op": "=", "value": "net"},
                      {"column": {"column": "day", "fn": "year"}, "op": "between", "param": ["y0", "y1"]},
                      {"column": "note", "op": "is_null"}]}
    shape = describe_spec(spec)
    assert shape.startswith("ONE row over all matching rows; computes SUM(amount) as total")
    assert "region = :region (user's choice)" in shape
    assert 'kind = "net" (FIXED)' in shape
    assert "year(day) between :y0 and :y1 (user's choice)" in shape
    assert "note is empty (fixed)" in shape


def test_shape_of_listings():
    assert describe_spec({"select": ["city"], "distinct": True}).startswith("the distinct values of city")
    s = describe_spec({"select": ["item", "value"], "distinct_on": ["item"],
                       "order_by": [{"column": "period", "dir": "desc"}], "limit": {"param": "n"}})
    assert s.startswith("one row per item (the first by the order below)")
    assert "ordered by period desc" in s and "only the first :n rows" in s
    assert describe_spec(None) == "stored SQL (no spec)"


def test_verification_is_on_unless_disabled_or_missing():
    assert verify_enabled(FakeRouter({}))
    assert not verify_enabled(FakeRouter({}, spec={"provider": "x", "model": "y", "enabled": False}))

    class NoTask:
        def task_spec(self, task):
            raise KeyError(task)
    assert not verify_enabled(NoTask())


# ---------------------------------------------------------------- measured coverage

# replies to the bounded retries for items left without a verdict
NO_VERDICTS = [{"items": []}] * 2

@db_only
async def test_filled_only_where_shows_which_rows_a_column_covers(db):  # noqa: F811
    from sqlalchemy import text

    from backend.app.services.structured.verify import context_columns, filled_only_where
    async with db() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_cov"))
        await s.execute(text("CREATE TABLE graphrag.t_cov (item text, period text, total_y1 numeric, "
                             "amount numeric)"))
        await s.execute(text("INSERT INTO graphrag.t_cov VALUES ('a', NULL, 5, NULL), "
                             "('a', 'P1', NULL, 1), ('a', 'P2', NULL, 2), ('b', NULL, 7, 3)"))
    meta = {"table_name": "t_cov", "columns": [
        {"name": "item", "type": "text", "role": "dimension"},
        {"name": "period", "type": "text", "role": "time"},
        {"name": "total_y1", "type": "numeric", "role": "measure"},
        {"name": "amount", "type": "numeric", "role": "measure"}],
        "column_stats": {"period": {"distinct": 2}, "item": {"distinct": 2}}}
    try:
        assert context_columns(meta) == ["period"]
        async with db() as s:
            got = await filled_only_where(s, meta)
        # total_y1 is filled only where period is empty; amount and item on every period
        assert got == {"total_y1": {"period": ["(empty)"]}}
    finally:
        async with db() as s:
            await s.execute(text("DROP TABLE IF EXISTS graphrag.t_cov"))


# ---------------------------------------------------------------- verdicts applied

COLUMN_REPLY = {"table_description": None, "grain": None, "columns": [
    {"name": "value", "problem": "description says nothing", "role": "measure",
     "description": "Reported amount of the item for the period ending period_end.", "unit": None}]}

ITEMS_REPLY = {"items": [
    {"id": "measure_sales", "verdict": "ok"},
    {"id": "value_by_report_year", "verdict": "fix_text", "reason": "label names the report year; "
     "the query returns one value per period",
     "label": "Value per period (latest report)", "description": "One value per period_end."},
    {"id": "by_entity", "verdict": "fix_query", "reason": "a listing capped at 20 rows is not complete",
     "label": "Number of values of an entity", "description": "How many values an entity has.",
     "query": {"select": ["entity", {"agg": "count", "as": "n_values"}],
               "where": [{"column": "entity", "op": "=", "param": "entity"}]},
     "params": [{"name": "entity", "description": "entity", "example": "A"}]},
]}

ITEMS_REPLY_2 = {"items": [
    {"id": "measure_sales", "verdict": "fix_query", "reason": "wrong",
     "query": {"select": ["value"]}},             # not allowed on a measure: re-asked, then pending
    {"id": "value_by_report_year", "verdict": "fix_query", "reason": "reads the wrong column",
     "query": {"select": ["no_such_column"]}},                         # the fix does not compile
    {"id": "by_entity", "verdict": "drop", "reason": "duplicates another query"},
]}


@db_only
async def test_verification_corrects_columns_then_items_and_records_everything(db, tmp_path):  # noqa: F811
    from backend.app.services.structured.db_load import registered_tables
    from backend.app.services.structured.refine import refine_registered
    from backend.app.services.structured.verify import VERIFY_VERSION

    cache = await _load(tmp_path)
    await refine_registered(FakeRouter({"table_refine": [REFINE_REPLY, REPAIR_REPLY]}), verify=False,
                            cache=cache, embed=_fake_embed, log=lambda *_: None)

    # already refined, not verified -> verification only (no table_refine call)
    router = FakeRouter({"table_verify": [COLUMN_REPLY, ITEMS_REPLY, *NO_VERDICTS]})
    reps = await refine_registered(router, cache=cache, embed=_fake_embed, log=lambda *_: None)
    assert [c[0] for c in router.calls] == ["table_verify", "table_verify"]
    col_prompt, item_prompt = router.calls[0][1], router.calls[1][1]
    assert '"rows_with_value"' in col_prompt and '"sample_rows"' in col_prompt
    # the item check sees the VERIFIED column text, the SQL, the shape and a sample result
    assert "Reported amount of the item" in item_prompt
    assert '"sql": "SELECT' in item_prompt and "only the first 20 rows (NOT a complete list or total)" in item_prompt
    assert '"sample"' in item_prompt and '"rows"' in item_prompt

    v = reps[0]["verified"]
    assert reps[0]["verified_only"] and v["llm_calls"] == 2
    assert [c["column"] for c in v["columns_fixed"]] == ["value"]
    assert v["ok"] == ["measure_sales"]
    assert [t["id"] for t in v["text_fixed"]] == ["value_by_report_year"]
    assert [q["id"] for q in v["query_fixed"]] == ["by_entity"] and v["dropped"] == []

    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert meta["verified"]["version"] == VERIFY_VERSION
    assert next(c for c in meta["columns"] if c["name"] == "value")["description"].startswith("Reported amount")
    vby = next(i for i in meta["metrics"] if i["id"] == "value_by_report_year")
    assert vby["label"] == "Value per period (latest report)" and vby["verified"] == "fix_text"
    fixed = next(i for i in meta["filters"] if i["id"] == "by_entity")
    assert fixed["label"] == "Number of values of an entity" and "COUNT(*)" in fixed["sql"]
    assert cache.get(meta["cache_key"], "reports")["verified"]["version"] == VERIFY_VERSION

    # verified -> no further call; --reverify runs it again
    quiet = FakeRouter({})
    assert await refine_registered(quiet, cache=cache, embed=_fake_embed, log=lambda *_: None) == []
    assert quiet.calls == []
    again = FakeRouter({"table_verify": [{"columns": []}, ITEMS_REPLY_2, *NO_VERDICTS]})
    reps = await refine_registered(again, reverify=True, cache=cache, embed=_fake_embed, log=lambda *_: None)
    v = reps[0]["verified"]
    dropped = {d["id"]: d for d in v["dropped"]}
    assert set(dropped) == {"value_by_report_year", "by_entity"}
    assert v["rejected_verdicts"][0] == {"id": "measure_sales", "verdict": "fix_query",
                                         "allowed": ["drop", "fix_definition", "fix_text", "ok"]}
    assert v["unverified"] == ["measure_sales"]
    assert "no_such_column" in dropped["value_by_report_year"]["fix_error"]
    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert [m["id"] for m in meta["measures"]] == ["measure_sales"] and meta["filters"] == []
    assert meta["measures"][0]["verify_verdict"]["verdict"] == "pending"
    assert meta["verified"]["pending"] == ["measure_sales"]
    assert {q["query_id"] for q in meta["common_questions"]} == {"measure_sales"}   # the others were dropped
    assert any(d["error"].startswith("verification:") for d in meta["dropped"])
    assert json.dumps(meta["verified"])

    # a later run verifies ONLY the pending item (no column step)
    later = FakeRouter({"table_verify": [{"items": [{"id": "measure_sales", "verdict": "ok"}]}]})
    reps = await refine_registered(later, cache=cache, embed=_fake_embed, log=lambda *_: None)
    assert len(later.calls) == 1 and '"measure_sales"' in later.calls[0][1]
    assert reps[0]["verified"]["only_pending"] and reps[0]["verified"]["ok"] == ["measure_sales"]
    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert meta["verified"]["pending"] == []



@db_only
async def test_new_refine_runs_the_verification_in_the_same_pass(db, tmp_path):  # noqa: F811
    from backend.app.services.structured.refine import refine_registered

    cache = await _load(tmp_path)
    router = FakeRouter({"table_refine": [REFINE_REPLY, REPAIR_REPLY],
                         "table_verify": [COLUMN_REPLY, ITEMS_REPLY]})
    reps = await refine_registered(router, cache=cache, embed=_fake_embed, log=lambda *_: None)
    assert [c[0] for c in router.calls] == ["table_refine", "table_refine", "table_verify", "table_verify"]
    assert reps[0]["llm_calls"] == 4 and reps[0]["verified"]["ok"] == ["measure_sales"]


def test_a_lookup_entry_carries_its_partner_tables_columns():
    from backend.app.services.structured.verify import _item_entry

    item = {"id": "lk_a_b", "label": "A with B", "other_table": "b", "params": [],
            "query": {"select": ["k", "b.name"]}}
    e = _item_entry({"table_name": "a"}, "lookups", item, {"sql": "SELECT 1"},
                    {"b": [{"name": "name", "type": "text", "description": "the name"}]})
    assert e["other_table"] == "b" and e["other_table_columns"][0]["name"] == "name"


ITEMS_REPLY_SPLIT = {"items": [
    {"id": "measure_sales", "verdict": "ok", "new_id": "sales_value"},          # measures keep their id
    {"id": "value_by_report_year", "verdict": "ok", "new_id": "value_per_period",
     "reason": "the id names the report year"},
    {"id": "by_entity", "verdict": "split", "reason": "a capped listing; the words promise a count",
     "new_id": "first_values_of_entity", "label": "First 20 values of an entity",
     "description": "Up to 20 values of one entity, newest first.",
     "add": {"id": "count_values_of_entity", "label": "Number of values of an entity",
             "description": "How many values an entity has.",
             "query": {"select": ["entity", {"agg": "count", "as": "n_values"}],
                       "where": [{"column": "entity", "op": "=", "param": "entity"}]},
             "params": [{"name": "entity", "description": "entity", "example": "A"}]}},
]}


@db_only
async def test_verification_renames_misleading_ids_and_splits_a_query(db, tmp_path):  # noqa: F811
    from backend.app.services.structured.db_load import registered_tables
    from backend.app.services.structured.metadata import find_item
    from backend.app.services.structured.refine import refine_registered

    cache = await _load(tmp_path)
    await refine_registered(FakeRouter({"table_refine": [REFINE_REPLY, REPAIR_REPLY]}), verify=False,
                            cache=cache, embed=_fake_embed, log=lambda *_: None)
    router = FakeRouter({"table_verify": [{"columns": []}, ITEMS_REPLY_SPLIT, *NO_VERDICTS]})
    reps = await refine_registered(router, cache=cache, embed=_fake_embed, log=lambda *_: None)
    v = reps[0]["verified"]
    assert {(r["from"], r["to"]) for r in v["renamed"]} == {
        ("value_by_report_year", "value_per_period"), ("by_entity", "first_values_of_entity")}
    assert [a["id"] for a in v["added"]] == ["count_values_of_entity"] and v["add_failed"] == []

    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert find_item(meta, "measure_sales")[1]["id"] == "measure_sales"
    sec, it = find_item(meta, "value_by_report_year")              # old id still resolves
    assert it["id"] == "value_per_period" and it["aliases"] == ["value_by_report_year"]
    kept = find_item(meta, "first_values_of_entity")[1]
    assert kept["label"] == "First 20 values of an entity" and kept["verified"] == "fix_text"
    sec, new = find_item(meta, "count_values_of_entity")
    assert sec == "filters" and new["verified"] == "added" and "COUNT(*)" in new["sql"]


@db_only
async def test_a_split_whose_new_query_fails_keeps_the_original(db, tmp_path):  # noqa: F811
    from backend.app.services.structured.db_load import registered_tables
    from backend.app.services.structured.metadata import find_item
    from backend.app.services.structured.refine import refine_registered

    cache = await _load(tmp_path)
    await refine_registered(FakeRouter({"table_refine": [REFINE_REPLY, REPAIR_REPLY]}), verify=False,
                            cache=cache, embed=_fake_embed, log=lambda *_: None)
    bad = {"items": [{"id": "by_entity", "verdict": "split", "reason": "r", "label": "Listing",
                      "add": {"id": "x", "query": {"select": ["no_such_column"]}}}]}
    reps = await refine_registered(FakeRouter({"table_verify": [{"columns": []}, bad, *NO_VERDICTS]}),
                                   cache=cache, embed=_fake_embed, log=lambda *_: None)
    v = reps[0]["verified"]
    assert v["added"] == [] and v["add_failed"][0]["from"] == "by_entity"
    async with db() as s:
        meta = (await registered_tables(s))["reports"]["meta"]
    assert find_item(meta, "by_entity")[1]["label"] == "Listing" and find_item(meta, "x") is None


async def test_a_fixed_lookup_is_tested_with_the_partner_tables_metadata(monkeypatch):
    """Regression: the verifier's set of partner table NAMES shadowed the
    ``partners`` argument ({table: metadata}); re-testing a lookup after a
    fix_query verdict then failed with "'set' object is not subscriptable".
    No database: the SQL-touching helpers are stubbed."""
    import contextlib

    from backend.app.db import session as session_mod
    from backend.app.services.structured import db_load, refine, verify

    @contextlib.asynccontextmanager
    async def no_session():
        yield None

    async def no_filled(_s, _meta):
        return {}

    async def no_rows(*_a, **_k):
        return []

    async def sample(*_a, **_k):
        return {"sql": "SELECT 1", "columns": ["k"], "rows": []}

    async def registered(_s):
        return {"b": {"meta": {"columns": [{"name": "k", "type": "text"}]}}}

    seen = []

    async def fake_test_item(meta, sec, item, analysis, types, *, partners=None):
        seen.append(partners)
        return {"status": "passed"}

    monkeypatch.setattr(session_mod, "session_scope", no_session)
    monkeypatch.setattr(verify, "filled_only_where", no_filled)
    monkeypatch.setattr(verify, "_sample_result", sample)
    monkeypatch.setattr(refine, "_sample_rows", no_rows)
    monkeypatch.setattr(refine, "test_item", fake_test_item)
    monkeypatch.setattr(db_load, "registered_tables", registered)

    lookup = {"id": "lk_a_b", "label": "A with B", "other_table": "b", "params": [],
              "query": {"select": ["k"]}, "sql": "SELECT 1"}
    meta = {"table_name": "a", "columns": [{"name": "k", "type": "text"}], "lookups": [lookup]}
    partner_meta = {"b": {"table_name": "b", "columns": [{"name": "k", "type": "text"}]}}
    router = FakeRouter({"table_verify": [
        {"columns": []},
        {"items": [{"id": "lk_a_b", "verdict": "fix_query", "reason": "r",
                    "query": {"select": ["k"], "limit": 5}}]}]})
    await verify.verify_table(meta, router, analysis={}, all_types={"a": {"k": "text"}, "b": {"k": "text"}},
                              partners=partner_meta, log=lambda *_: None)
    assert seen and all(p is partner_meta for p in seen)


# ------------------------------------------------- verdicts are given on the FINAL SQL

def _stub_sql_helpers(monkeypatch, verify, refine, session_mod, db_load):
    import contextlib

    @contextlib.asynccontextmanager
    async def no_session():
        yield None

    async def no_filled(_s, _meta):
        return {}

    async def no_rows(*_a, **_k):
        return []

    async def sample(meta, sec, it, types):
        return {"sql": it.get("sql"), "columns": ["v"], "rows": []}

    async def registered(_s):
        return {}

    monkeypatch.setattr(session_mod, "session_scope", no_session)
    monkeypatch.setattr(verify, "filled_only_where", no_filled)
    monkeypatch.setattr(verify, "_sample_result", sample)
    monkeypatch.setattr(refine, "_sample_rows", no_rows)
    monkeypatch.setattr(db_load, "registered_tables", registered)


async def test_a_column_correction_recompiles_before_the_items_are_verified(monkeypatch):
    """Regression (sc80 fertilizer table, 2026-11-26): the column step changed
    year_code from identifier to time; the catalog was recompiled only AFTER
    the item verdicts, so 18 queries' SQL no longer matched the SQL their
    verdict was given on and retrieval refused them as "semantic review
    stale". Now the recompile happens between the two steps. The compiler is
    stubbed: a query's period span uses every column whose role is time,
    which is what made the real SQL change."""
    from backend.app.db import session as session_mod
    from backend.app.services.structured import db_load, refine, verify
    from backend.app.services.structured.metadata import semantic_status, sql_sha

    _stub_sql_helpers(monkeypatch, verify, refine, session_mod, db_load)
    order = []

    def compiled(meta):
        span = [c["name"] for c in meta["columns"] if c.get("role") == "time"]
        return "SELECT SUM(v)" + "".join(f", MIN({c})" for c in span) + " FROM t"

    async def fake_ensure_tested(meta, *, analysis, all_types, partners=None, force=False):
        order.append("recompile")
        for it in meta["metrics"]:
            it["sql"] = compiled(meta)
        return {"passed": 1, "failed": 0, "not_exercised": 0, "retested": 1}

    monkeypatch.setattr(refine, "ensure_tested", fake_ensure_tested)
    meta = {"table_name": "t", "semantic_review": {"required": True},
            "columns": [{"name": "year", "type": "integer", "role": "time"},
                        {"name": "year_code", "type": "integer", "role": "identifier"},
                        {"name": "v", "type": "numeric", "role": "measure"}],
            "metrics": [{"id": "total_by_year", "label": "Total by year", "params": [],
                         "query": {"select": ["v"]}}]}
    meta["metrics"][0]["sql"] = compiled(meta)
    before = meta["metrics"][0]["sql"]

    class Router(FakeRouter):
        async def chat(self, task, **kw):
            order.append("verdicts" if "items" in kw.get("user", "") or len(self.calls) else "columns")
            return await super().chat(task, **kw)

    router = Router({"table_verify": [
        {"columns": [{"name": "year_code", "role": "time", "problem": "a copy of year"}]},
        {"items": [{"id": "total_by_year", "verdict": "ok", "reason": ""}]}]})
    await verify.verify_table(meta, router, analysis={}, all_types={"t": {}}, log=lambda *_: None)

    it = meta["metrics"][0]
    assert order == ["columns", "recompile", "verdicts"]
    assert it["sql"] != before and "MIN(year_code)" in it["sql"]
    assert it["verify_verdict"]["sql_sha"] == sql_sha(it["sql"])
    assert semantic_status(meta, it) == "ok"


async def test_without_a_column_correction_nothing_is_recompiled(monkeypatch):
    from backend.app.db import session as session_mod
    from backend.app.services.structured import db_load, refine, verify

    _stub_sql_helpers(monkeypatch, verify, refine, session_mod, db_load)
    called = []

    async def fake_ensure_tested(*_a, **_k):
        called.append(1)
        return {}

    monkeypatch.setattr(refine, "ensure_tested", fake_ensure_tested)
    meta = {"table_name": "t", "columns": [{"name": "v", "type": "numeric", "role": "measure"}],
            "metrics": [{"id": "m", "label": "M", "params": [], "query": {"select": ["v"]}, "sql": "SELECT v"}]}
    router = FakeRouter({"table_verify": [{"columns": []}, {"items": [{"id": "m", "verdict": "ok"}]}]})
    await verify.verify_table(meta, router, analysis={}, all_types={"t": {}}, log=lambda *_: None)
    assert called == []


async def test_a_query_whose_sql_changed_after_its_verdict_is_verified_again(monkeypatch):
    """Safety net: whatever recompiles a query after its verdict, the query is
    verified again (only the changed ones, no column step) instead of staying
    unavailable to retrieval."""
    from backend.app.services.structured import refine, verify
    from backend.app.services.structured.metadata import sql_sha

    seen = {}

    async def fake_verify_table(meta, router, **kw):
        seen.update(kw)
        for _s, it in verify.pending_items(meta):
            it["verify_verdict"] = {"verdict": "ok", "version": verify.VERIFY_VERSION,
                                    "sql_sha": sql_sha(it["sql"])}
        return {"llm_calls": 1}

    async def fake_ensure_tested(*_a, **_k):
        return {"passed": 2, "failed": 0, "not_exercised": 0, "retested": 0}

    monkeypatch.setattr(verify, "verify_table", fake_verify_table)
    monkeypatch.setattr(refine, "ensure_tested", fake_ensure_tested)
    ok_v = {"verdict": "ok", "version": verify.VERIFY_VERSION}
    meta = {"table_name": "t", "semantic_review": {"required": True}, "metrics": [
        {"id": "same", "sql": "SELECT 1", "verify_verdict": {**ok_v, "sql_sha": sql_sha("SELECT 1")}},
        {"id": "changed", "sql": "SELECT 2", "verify_verdict": {**ok_v, "sql_sha": sql_sha("SELECT old")}}]}
    out = await refine.reverify_changed(meta, None, analysis={}, all_types={}, log=lambda *_: None)
    assert out == ({"passed": 2, "failed": 0, "not_exercised": 0, "retested": 0}, 1)
    assert seen.get("only_pending") is True
    assert meta["metrics"][1]["verify_verdict"]["sql_sha"] == sql_sha("SELECT 2")
    # nothing stale: no call at all
    assert await refine.reverify_changed(meta, None, analysis={}, all_types={}, log=lambda *_: None) is None

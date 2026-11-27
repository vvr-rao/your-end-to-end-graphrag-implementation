"""A total is answered from a verified stored aggregate, never from its parts.

Regression for the q023 miss (2026-11-26): asked for two countries' total
fertilizer imports, the selector picked the row-level measure (one row per
partner) of a FROZEN low-effort catalog although that catalog has a verified
aggregate (sum by reporter); the answer then said the totals were absent.
Now a call marked scope "total" that names a row-level query is refused when
a verified stored aggregate of the same quantity exists, and the retry is
told which. Retrieval still runs only stored queries; the program never adds
rows up. Also: partial results say how many rows are shown of how many, and
the answer rules forbid presenting them as complete. No database: candidate
tables and SQL are monkeypatched. Nothing here is specific to the dataset --
the fixture only pins a real catalog the rule must work on."""

from __future__ import annotations

import copy
import json
import pathlib
import uuid

import pytest

from backend.app.services.structured import retrieval_step as rs
from backend.app.services.structured.sql_exec import SqlResult
from backend.tests.unit.test_structured_profiler import FakeRouter

FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "structured" / "trade_flows_low_effort_catalog.json"
LOW = json.loads(FIXTURE.read_text(encoding="utf-8"))
NODE = uuid.uuid4()
QUESTION = ("Besides Bangladesh and Jordan, which two countries does the report place in the high-risk "
            "zone, and what were their total fertilizer imports in 2024?")
# the selection the low catalog actually received in the miss, now marked as a total
ORIGINAL = {"table": "trade_flows", "query_id": "measure_trade_value", "listing": True, "scope": "total",
            "params": {"reporterdesc": ["India", "Thailand"], "cmddesc": "Fertilizers",
                       "flowdesc": "Import", "refyear": 2024}}
AGGREGATE = {"table": "trade_flows", "query_id": "trade_value_by_reporter", "scope": "total",
             "params": {"year": 2024, "flow": "Import", "commodity": "Fertilizers"}, "retry_of": 1}
AGG_ROWS = [["India", "7709020032.686", "62", "2024", "2024"],
            ["Thailand", "2561055865.113", "56", "2024", "2024"]]


@pytest.fixture
def harness(monkeypatch):
    meta_box = {"meta": LOW}
    ran: list[str] = []

    async def fake_cands(entity_ids, class_ids, qvec, *, max_tables=3, vector_max_distance=None, time_ids=None,
                         entity_scores=None, question=None):
        return [{"id": NODE, "table": "trade_flows", "meta": meta_box["meta"], "score": 2.0, "via": ["entity:1"]}]

    async def fake_run(sql, params, allowed, *, max_rows=50, timeout_ms=5000, declared_params=None):
        ran.append(sql)
        if 'SUM("primaryvalue")' in sql and 'GROUP BY "reporterdesc"' in sql:
            cols = ["reporterdesc", "total_value", "rows_aggregated", "first_refyear", "last_refyear"]
            return SqlResult(sql=sql, params=dict(params or {}), columns=cols, rows=AGG_ROWS, row_count=2)
        if '"primaryvalue" FROM "trade_flows"' in sql or '"primaryvalue"\nFROM' in sql:
            rows = [["India", f"P{i}", str(1000 + i)] for i in range(118)]
            return SqlResult(sql=sql, params=dict(params or {}), columns=["reporterdesc", "partnerdesc",
                             "primaryvalue"], rows=rows, row_count=118)
        return SqlResult(sql=sql, params=dict(params or {}), columns=["v"], rows=[["x"]], row_count=1)

    async def fake_metas(names):
        return {}

    async def fake_inventory():
        return {"trade_flows": {"id": NODE, "iri": "https://x/o#TableMetadata_trade_flows_1",
                                "meta": meta_box["meta"]}}

    monkeypatch.setattr(rs, "candidate_tables", fake_cands)
    monkeypatch.setattr(rs, "table_metas", fake_metas)
    monkeypatch.setattr(rs, "table_inventory", fake_inventory)
    monkeypatch.setattr(rs, "run_sql", fake_run)
    return meta_box, ran


async def _go(router, **kw):
    return await rs.structured_evidence(QUESTION, router=router, entity_ids=[], class_ids=[], qvec=None, **kw)


async def test_q023_a_total_on_the_row_level_measure_is_refused_and_the_aggregate_is_used(harness):
    _meta, ran = harness
    router = FakeRouter({"table_query_select": [{"needed": True, "calls": [ORIGINAL]},
                                                {"needed": True, "calls": [AGGREGATE]}]})
    ev, plan = await _go(router)

    first = plan["calls"][0]
    assert first["query_id"] == "measure_trade_value" and first.get("selection_error")
    assert first["total_alternatives"] == ["trade_value_by_reporter", "trade_value_by_year"]
    assert "returns the PARTS of a total" in first["error"]
    assert not any('"partnerdesc"' in s and "GROUP BY" not in s for s in ran)   # the parts never ran

    retry_prompt = router.calls[1][1]
    assert "total_alternatives" in retry_prompt and "trade_value_by_reporter" in retry_prompt

    rows = [e for e in ev if e["kind"] == "sql_result"]
    assert [e["query_id"] for e in rows] == ["trade_value_by_reporter"]
    assert plan["calls"][1].get("scope") == "total" and plan["calls"][1].get("retry")
    text = rows[0]["text"]
    assert "7709020032.686" in text and "2561055865.113" in text
    assert "Units: total_value = USD (sum of primaryvalue)" in text
    assert "rows_aggregated = dimensionless (count)" in text


async def test_an_unmarked_call_still_runs_the_row_level_query(harness):
    _meta, _ran = harness
    call = {k: v for k, v in ORIGINAL.items() if k != "scope"}
    ev, plan = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [call]}]}),
                         listing_max_rows=2000, listing_char_cap=60000)
    assert not plan["calls"][0].get("selection_error")
    assert [e["query_id"] for e in ev if e["kind"] == "sql_result"] == ["measure_trade_value"]


async def test_a_total_with_no_stored_aggregate_is_reported_as_parts(harness):
    meta_box, _ran = harness
    meta = copy.deepcopy(LOW)
    for sec in ("metrics", "queries"):          # no verified aggregate of the quantity is left
        meta[sec] = [it for it in meta.get(sec) or []
                     if "SUM(" not in (it.get("sql") or "").upper()]
    meta_box["meta"] = meta
    ev, plan = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [ORIGINAL]}]}),
                         listing_max_rows=2000, listing_char_cap=60000)
    assert plan["calls"][0].get("total_without_aggregate") and not plan["calls"][0].get("selection_error")
    text = next(e["text"] for e in ev if e["kind"] == "sql_result")
    assert rs.TOTAL_PARTS_NOTE in text


async def test_a_result_cut_for_the_prompt_says_how_many_rows_are_shown_of_how_many(harness):
    call = {k: v for k, v in ORIGINAL.items() if k != "scope"}
    ev, _ = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [call]}]}),
                      listing_max_rows=2000, listing_char_cap=1500)
    text = next(e["text"] for e in ev if e["kind"] == "sql_result")
    assert "PARTIAL RESULT:" in text and "the query returned 118 rows" in text
    shown = int(text.split("PARTIAL RESULT: ")[1].split(" row(s) shown")[0])
    assert 0 < shown < 118
    assert "is NOT the complete figure" in text and "not missing from the data" in text


def test_partial_note_distinguishes_known_and_unknown_full_counts():
    assert rs.partial_note(5, 5, False, None) == ""
    assert "the query matched 1234 rows in total" in rs.partial_note(50, 50, True, 1234)
    assert "full count unknown" in rs.partial_note(50, 50, True, None)
    assert "the query returned 118 rows" in rs.partial_note(40, 118, False, None)


def test_answer_rules_forbid_partial_totals_and_cover_units():
    from backend.app.services.prompts import _format_evidence_block

    base = {"kind": "sql_result", "iri": "x#SqlResult_t_q_1", "text": "SQL result ... 2 row(s): (a=1)"}
    block = _format_evidence_block([base])
    assert "PARTIAL RESULT shows only some of the rows" in block
    assert "is NOT the complete figure" in block and "rather than that the data has none" in block
    assert "Units of SQL results" not in block                    # no units line -> no units rule
    block = _format_evidence_block([{**base, "text": "Units: v = USD. 1 row(s): (v=1)"}])
    assert "Units of SQL results" in block and "never assume a currency or unit" in block


def test_the_selection_prompt_asks_for_a_total_scope():
    from backend.app.services.prompts import table_query_select

    system, user = table_query_select("q", [])
    assert "TOTALS:" in system and '"scope": "total"' in system
    assert '"scope": null' in user


def test_a_retry_that_groups_by_a_requested_column_keeps_its_values():
    keep = {"reporterdesc|False|False": {"members": ["India", "Thailand"]}}
    grouped = {"id": "by_reporter", "params": [], "query": {
        "select": ["reporterdesc", {"agg": "sum", "column": "v", "as": "total"}], "group_by": ["reporterdesc"]}}
    assert rs.broadens({"params": {}}, grouped, keep, grouping_keeps=True) is None
    # not a program-named total alternative: grouping is not enough (it may compute something else)
    assert "cannot filter reporterdesc" in rs.broadens({"params": {}}, grouped, keep)
    other = {"id": "by_year", "params": [], "query": {
        "select": ["year", {"agg": "sum", "column": "v", "as": "total"}], "group_by": ["year"]}}
    assert "cannot filter reporterdesc" in rs.broadens({"params": {}}, other, keep, grouping_keeps=True)
    rows = {"id": "all_rows", "params": [], "query": {"select": ["reporterdesc", "v"]}}
    assert "cannot filter reporterdesc" in rs.broadens({"params": {}}, rows, keep, grouping_keeps=True)


async def test_a_total_on_a_stored_aggregate_runs_without_the_parts_note(harness):
    # regression (step-4 rerun, 2026-11-26): a total-scope call that names a stored
    # aggregate was labelled "these rows are its parts, not a total"
    _meta, _ran = harness
    call = {k: v for k, v in AGGREGATE.items() if k != "retry_of"}
    ev, plan = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [call]}]}))
    assert not plan["calls"][0].get("selection_error")
    assert not plan["calls"][0].get("total_without_aggregate")
    text = next(e["text"] for e in ev if e["kind"] == "sql_result")
    assert rs.TOTAL_PARTS_NOTE not in text and "7709020032.686" in text


# --- a total whose every row dimension the call pins (2026-11-27, sc80 q017/q023) ---
# Asked for one reporter's exports of one commodity in one year, the selector
# called the row-level measure marked scope "total" with every dimension of its
# rows fixed (the year as a range 2024..2024). Its one row IS the total, but the
# call was refused as "parts" and the answer said the value was absent.

async def test_a_total_with_every_row_dimension_pinned_runs_without_the_parts_note(harness):
    _meta, _ran = harness
    call = {"table": "trade_flows", "query_id": "measure_trade_value", "scope": "total",
            "params": {"reporterdesc": "India", "partnerdesc": "World", "cmddesc": "Fertilizers",
                       "flowdesc": "Import", "refyear": 2024}}
    ev, plan = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [call]}]}),
                         listing_max_rows=2000, listing_char_cap=60000)
    first = plan["calls"][0]
    assert not first.get("selection_error") and not first.get("total_without_aggregate")
    text = next(e["text"] for e in ev if e["kind"] == "sql_result")
    assert rs.TOTAL_PARTS_NOTE not in text


def _desc(grain, params, fixed=()):
    return {"operation": "one value per observation (no aggregation)", "output_grain": grain,
            "parameters": params, "fixed_conditions": list(fixed)}


def test_open_dimensions_counts_a_single_value_range_as_pinned():
    meta = {"analysis": {"dependencies": {"reportercode": ["reporteriso"], "reporteriso": ["reportercode"]}}}
    params = [{"name": "reporteriso", "column": "reporteriso", "op": "="},
              {"name": "cmd", "column": "cmdcode", "op": "="},
              {"name": "y0", "column": "refyear", "op": "between"},
              {"name": "y1", "column": "refyear", "op": "between"}]
    desc = _desc(["reportercode", "cmdcode", "refyear"], params)
    # the code is pinned through its ISO family; the year range is one value
    assert rs.open_dimensions(meta, desc, {"reporteriso": "MYS", "cmd": 8542, "y0": 2024, "y1": 2024}) == set()
    # a real range leaves the year open: the rows are one per year
    assert rs.open_dimensions(meta, desc, {"reporteriso": "MYS", "cmd": 8542, "y0": 2023, "y1": 2024}) == {"refyear"}
    # an unfilled optional filter leaves its dimension open
    assert "cmdcode" in rs.open_dimensions(meta, desc, {"reporteriso": "MYS", "y0": 2024, "y1": 2024})


def test_open_dimensions_a_requested_list_pins_its_dimension():
    params = [{"name": "iso", "column": "reporteriso", "op": "="},
              {"name": "year", "column": "refyear", "op": "="}]
    desc = _desc(["reporteriso", "refyear"], params)
    # one row per listed country is what was asked, not parts of one total
    assert rs.open_dimensions({}, desc, {"iso": ["IND", "THA"], "year": 2024}) == set()
    # an exclusion does not pin a dimension
    desc2 = _desc(["reporteriso", "partnercode"], params, fixed=[{"column": "partnercode", "op": "!=", "value": 0}])
    assert rs.open_dimensions({}, desc2, {"iso": "IND"}) == {"partnercode"}

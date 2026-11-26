"""`--use-structured-data` retrieval step: selection, execution, retry, evidence.

No database: `candidate_tables` and `run_sql` are monkeypatched. The real-DB
path is covered in test_structured_db.py.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from backend.app.services.structured import retrieval_step as rs
from backend.app.services.structured.metadata import CATALOG_VERSION, finalize_item, items, record_test
from backend.app.services.structured.sql_exec import SqlResult
from backend.tests.unit.test_structured_profiler import FakeRouter

META = {
    "table_name": "prices",
    "description": "Monthly commodity prices",
    "source": {"rel_path": "worldbank/pink-sheet/s1/normalized/prices.csv"},
    "columns": [{"name": "commodity", "type": "text"}, {"name": "period", "type": "date"},
                {"name": "value", "type": "numeric"}],
    "metrics": [{"id": "avg_price", "label": "Average price",
                 "query": {"select": [{"agg": "avg", "column": "value", "as": "avg_value"}],
                           "where": [{"column": "commodity", "op": "=", "param": "c"}]},
                 "params": [{"name": "c", "type": "text", "column": "commodity", "example": "Copper"}]}],
    "filters": [],
    "queries": [{"id": "commodity_counts", "label": "Rows per commodity",
                 "query": {"select": ["commodity", {"agg": "count", "as": "n"}]}, "params": []},
                {"id": "all_prices", "label": "Every price",
                 "query": {"select": ["commodity", "period", "value"]}, "params": []}],
    "lookups": [{"id": "company_prices", "other_table": "companies", "params": [],
                 "join": [{"column": "commodity", "other_column": "commodity"}],
                 "query": {"select": ["companies.company", "value"]}}],
}
AVG_SQL = ('SELECT AVG("value") AS "avg_value", COUNT(*) AS "rows_aggregated", '
           'MIN("period") AS "first_period", MAX("period") AS "last_period" FROM "prices" '
           'WHERE "commodity" = CAST(:c AS text)')
COMPANIES = {"table_name": "companies", "columns": [{"name": "company", "type": "text"},
                                                    {"name": "commodity", "type": "text"}]}
NODE = uuid.uuid4()


def _verify_catalog(meta: dict, partners: dict) -> dict:
    """What ingestion does: finalise every item's SQL and record a passed test."""
    types = {n: {c["name"]: c["type"] for c in m["columns"]} for n, m in partners.items()}
    for sec, it in items(meta):
        assert finalize_item(meta, sec, it, types, partners=partners) is None
        record_test(it, "passed")
    meta["catalog_version"] = CATALOG_VERSION
    return meta


_verify_catalog(META, {"prices": META, "companies": COMPANIES})


@pytest.fixture
def patched(monkeypatch):
    calls: list[tuple[str, dict, set]] = []
    results: list[SqlResult] = []

    async def fake_cands(entity_ids, class_ids, qvec, *, max_tables=3, vector_max_distance=None, time_ids=None,
                         entity_scores=None, question=None):
        return [{"id": NODE, "table": "prices", "meta": META, "score": 2.0, "via": ["entity:1"]}]

    async def fake_run(sql, params, allowed, *, max_rows=50, timeout_ms=5000, declared_params=None):
        calls.append((sql, dict(params or {}), set(allowed)))
        r = results.pop(0) if results else SqlResult(sql=sql, params=dict(params or {}),
                                                      columns=["avg_value"], rows=[["8500"]], row_count=1)
        r.sql = sql
        r.params = r.params or dict(params or {})
        return r

    async def fake_metas(names):
        return {"companies": COMPANIES} if "companies" in names else {}

    async def fake_inventory():
        return {"companies": {"id": uuid.uuid4(), "iri": "https://x/o#TableMetadata_companies_1", "meta": COMPANIES},
                "prices": {"id": NODE, "iri": "https://x/o#TableMetadata_prices_1", "meta": META}}

    monkeypatch.setattr(rs, "candidate_tables", fake_cands)
    monkeypatch.setattr(rs, "table_metas", fake_metas)
    monkeypatch.setattr(rs, "table_inventory", fake_inventory)
    monkeypatch.setattr(rs, "run_sql", fake_run)
    return calls, results


async def _go(router, **kw):
    return await rs.structured_evidence("What was the average copper price?", router=router,
                                        entity_ids=[], class_ids=[], qvec=None, **kw)


async def test_catalog_query_is_used_and_becomes_cited_evidence(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "reason": "price", "calls": [
            {"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]}]})
    ev, plan = await _go(router)
    assert [c[0] for c in router.calls] == ["table_query_select"]
    assert calls == [(AVG_SQL, {"c": "Copper"}, {"prices"})]   # built from the spec, not stored text
    assert len(ev) == 1 and ev[0]["kind"] == "sql_result" and ev[0]["node_id"] == NODE
    assert ev[0]["query_id"] == "avg_price" and ev[0]["sql_source"] == "catalog"
    assert "avg_value=8500" in ev[0]["text"] and "Parameters: c=Copper" in ev[0]["text"]
    assert re.search(r"#SqlResult_prices_avg_price_[0-9a-f]{8}$", ev[0]["iri"])
    assert plan["needed"] and not plan["retried"] and plan["calls"][0]["source"] == "catalog"


async def test_not_needed_runs_no_sql(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [{"needed": False, "reason": "no table data"}]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and plan["needed"] is False


async def test_failed_catalog_query_gets_one_retry_with_a_stored_query(patched):
    calls, results = patched
    results.append(SqlResult(sql="", error="column \"x\" does not exist"))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Cu"}}]},
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Cu"}}]},
    ]})
    ev, plan = await _go(router)
    assert len(router.calls) == 2 and "PREVIOUS ATTEMPT FAILED" in router.calls[1][1]
    assert 'column \\"x\\" does not exist' in router.calls[1][1]
    assert plan["retried"] and len(ev) == 1 and ev[0]["sql_source"] == "catalog"
    assert ev[0]["query_id"] == "avg_price" and calls[1] == (AVG_SQL, {"c": "Cu"}, {"prices"})


async def test_retry_may_not_change_a_value_not_shown_invalid(patched):
    calls, results = patched
    results.append(SqlResult(sql="", error="timeout"))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Cu"}}]},
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]}]})
    ev, plan = await _go(router)
    assert len(calls) == 1 and ev == []
    assert any("changes commodity" in (c.get("error") or "") for c in plan["calls"])


async def test_model_written_sql_is_never_run(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "new_sql": "SELECT 1 FROM prices", "params": {}}]},
        {"needed": True, "calls": [{"table": "prices", "query": {"select": ["nope"]}, "params": {}}]},
    ]})
    ev, plan = await _go(router)
    assert ev == [] and calls == []
    assert all("Inline SQL or a query spec is refused" in c["error"] for c in plan["calls"])


async def test_inline_spec_is_refused_even_with_a_valid_query_id(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"},
                                    "sql": "SELECT * FROM prices"}]},
        {"needed": False}]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and "refused (found: sql)" in plan["calls"][0]["error"]


async def test_query_id_resolves_only_within_the_named_table(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "companies", "query_id": "avg_price", "params": {"c": "Copper"}}]},
        {"needed": False}]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and plan["calls"][0]["error"]


async def test_zero_rows_is_a_valid_result_and_not_retried(patched):
    calls, results = patched
    results.append(SqlResult(sql="", columns=["avg_value"]))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]}]})
    _ev, plan = await _go(router)
    assert len(router.calls) == 1 and not plan["retried"] and len(calls) == 1
    assert plan["calls"][0].get("error") is None


async def test_retry_that_broadens_the_question_is_refused(patched):
    calls, results = patched
    results.append(SqlResult(sql="", error="timeout"))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]},
        {"needed": True, "calls": [{"table": "prices", "query_id": "commodity_counts", "params": {}}]}]})
    ev, plan = await _go(router)
    assert len(calls) == 1 and ev == []                    # the broader retry never ran
    assert any("cannot filter commodity" in (c.get("error") or "") for c in plan["calls"])


async def test_explicit_no_match_reaches_the_answer(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [], "no_match": [{"part": "copper margin", "reason": "no stored query"}]}]})
    ev, plan = await _go(router)
    assert calls == [] and plan["no_match"][0]["part"] == "copper margin"
    assert any("copper margin" in e["text"] for e in ev)


async def test_unverified_catalog_is_unavailable_with_the_rebuild_action(patched, monkeypatch):
    import copy

    calls, _ = patched
    stale = copy.deepcopy(META)
    stale["catalog_version"] = None

    async def cands(*a, **k):
        return [{"id": NODE, "table": "prices", "meta": stale, "score": 2.0, "via": ["entity:1"]}]

    monkeypatch.setattr(rs, "candidate_tables", cands)
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]},
        {"needed": False}]})
    _ev, plan = await _go(router)
    assert calls == [] and plan["unavailable"]
    assert any("table-metadata-refine --table prices --relink" in a for a in plan["unavailable_action"])


async def test_unknown_query_id_is_a_failure_not_a_crash(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "nope", "params": {}}]},
        {"needed": False},
    ]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and plan["retried"]
    assert "'nope' is not a stored query" in plan["calls"][0]["error"]


async def test_lookup_may_read_its_partner_table(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "company_prices", "params": {}}]}]})
    ev, _ = await _go(router)
    assert calls[0][2] == {"prices", "companies"} and ev[0]["table"] == "prices"


async def test_truncated_result_says_so_and_respects_char_cap(patched):
    _, results = patched
    rows = [[f"Commodity{i}", str(i)] for i in range(50)]
    results.append(SqlResult(sql="", columns=["commodity", "value"], rows=rows, row_count=50,
                             truncated=True, total_rows=1234))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "x"}}]}]})
    ev, _ = await _go(router, char_cap=600)
    t = ev[0]["text"]
    assert "PARTIAL RESULT:" in t and "the query matched 1234 rows in total" in t
    assert "is NOT the complete figure" in t
    assert len(t) <= 700


async def test_no_candidate_table_makes_no_llm_call(monkeypatch):
    async def none(*a, **k):
        return []

    monkeypatch.setattr(rs, "candidate_tables", none)
    router = FakeRouter({})
    ev, plan = await _go(router)
    assert ev == [] and router.calls == [] and "no table" in plan["reason"]


def test_evidence_block_note_only_with_sql_results():
    from backend.app.services.prompts import _format_evidence_block

    chunk = {"kind": "chunk", "iri": "https://x#Chunk_1", "text": "hello"}
    assert "SqlResult_" not in _format_evidence_block([chunk])
    sql = {"kind": "sql_result", "iri": "https://x#SqlResult_prices_avg_1", "text": "v" * 3000,
           "char_cap": 5000}
    block = _format_evidence_block([sql, chunk], char_cap=600)
    assert "viao:SqlResult_" in block and "[viao:SqlResult_prices_avg_1]" in block
    assert "v" * 3000 in block                  # per-item cap overrides the global one


def test_sql_result_citations_validate():
    from backend.app.services.retrieval import _validate_citations

    ev = [{"kind": "sql_result", "iri": "https://x#SqlResult_prices_avg_price_1"}]
    out, bad = _validate_citations("Copper averaged 8500 [viao:SqlResult_prices_avg_price_1].", ev)
    assert bad == [] and "SqlResult_prices_avg_price_1" in out


def test_flag_is_off_by_default_in_shipped_config():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    qa = yaml.safe_load((root / "config" / "config.example.yaml").read_text(encoding="utf-8"))["qa"]
    assert qa["use_structured_data"] is False


def test_flag_threads_through_every_entry_point():
    import inspect

    from backend.app.services import db_conversation, retrieval

    assert "use_structured_data" in inspect.signature(retrieval.retrieve_and_answer).parameters
    assert "use_structured_data" in inspect.signature(db_conversation.add_turn).parameters
    src = inspect.getsource(retrieval.retrieve_and_answer)
    assert src.count("use_structured_data=use_structured_data") == 2      # both deep_research rounds
    assert "use_structured_data=use_structured_data" in inspect.getsource(db_conversation.add_turn)
    from backend.app.api.conversations import TurnRequest
    from backend.app.api.qa import QAEvidenceItem, QARequest

    assert QARequest(question="q").use_structured_data is None
    assert TurnRequest(question="q").use_structured_data is None
    item = QAEvidenceItem(kind="sql_result", iri="https://x#SqlResult_a_1", rank=1, score=1.0,
                          sql="SELECT 1", rows=[["1"]], query_id="q", truncated=False)
    assert item.rows == [["1"]] and item.sql == "SELECT 1"


def test_compared_values_finds_literals_params_and_in_lists():
    cols = {"facts": {"tag": "text", "fy": "integer", "cik": "text"},
            "prices": {"commodity": "text", "value": "numeric"}}
    sql = ("SELECT f.tag FROM facts f WHERE f.cik = :cik AND tag IN ('Revenues', :t2) "
           "AND fy = 2025 AND unit_x = 'USD'")
    got = rs.compared_values(sql, {"cik": "0000811809", "t2": "ProfitLoss"}, cols, {"facts": {"cik"}})
    ctx_tag = [("cik", "0000811809")]
    assert got == [
        {"table": "facts", "column": "cik", "value": "0000811809", "context": []},
        {"table": "facts", "column": "tag", "value": "Revenues", "context": ctx_tag},
        {"table": "facts", "column": "tag", "value": "ProfitLoss", "context": ctx_tag},
    ]                                                         # fy is integer; unit_x unknown
    assert rs.compared_values("SELECT * FROM prices WHERE commodity = :c", {"c": "Iron ore"}, cols) \
        == [{"table": "prices", "column": "commodity", "value": "Iron ore", "context": []}]


def test_stored_query_literals_are_not_value_checked():
    """A stored query's fixed ``IN (...)`` list is set membership from its tested
    definition: checking each literal under the bound filters refused valid calls
    (an item that exists in only one of the listed units)."""
    cols = {"obs": {"item": "text", "unit": "text", "value": "numeric"}}
    sql = ("SELECT value FROM obs WHERE unit IN ('u1', 'u2', 'u3') "
           "AND (CAST(:item AS text) IS NULL OR item = CAST(:item AS text)) AND unit = CAST(:unit AS text)")
    got = rs.compared_values(sql, {"item": "A", "unit": "u1"}, cols, {"obs": {"item"}}, bound_only=True)
    assert sorted((c["column"], c["value"]) for c in got) == [("item", "A"), ("unit", "u1")]
    every = rs.compared_values(sql, {"item": "A", "unit": "u1"}, cols, {"obs": {"item"}})
    assert {c["value"] for c in every} == {"A", "u1", "u2", "u3"}       # default still lists literals


async def test_missing_value_is_refused_before_the_query_runs_and_retried(patched, monkeypatch):
    calls, _ = patched
    seen = []

    async def fake_missing(checks):
        seen.append(checks)
        return [{"table": "prices", "column": "commodity", "value": "Iron ore",
                 "existing_similar": ["Iron ore, cfr spot"]}] if len(seen) == 1 else []

    monkeypatch.setattr(rs, "missing_values", fake_missing)
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Iron ore"}}]},
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price",
                                    "params": {"c": "Iron ore, cfr spot"}}]},
    ]})
    ev, plan = await _go(router)
    assert seen[0] == [{"table": "prices", "column": "commodity", "value": "Iron ore", "context": []}]
    assert "Iron ore, cfr spot" in router.calls[1][1]         # hint reached the retry prompt
    assert plan["retried"] and plan["calls"][0]["missing_values"][0]["value"] == "Iron ore"
    assert plan["calls"][0]["error"] == rs.MISSING_VALUE_ERROR and "row_count" not in plan["calls"][0]
    assert [c[1] for c in calls] == [{"c": "Iron ore, cfr spot"}]      # only the corrected call ran
    assert len(ev) == 1


async def test_one_absent_value_of_a_list_keeps_the_rest(monkeypatch):
    async def fake_missing(checks):
        return [{"table": "prices", "column": "commodity", "value": "Zink", "existing_similar": ["Zinc"]}]

    monkeypatch.setattr(rs, "missing_values", fake_missing)
    sql = "SELECT * FROM prices WHERE commodity = ANY(CAST(:cs AS text[]))"
    cols = {"prices": {"commodity": "text"}}
    params, missing, subs = await rs.prevalidate_values(sql, {"cs": ["Copper", "Zink"]}, col_types=cols)
    assert params == {"cs": ["Copper"]} and subs == [] and missing[0]["dropped_from"] == "cs"


async def test_identical_retry_result_is_deduplicated(patched):
    _, results = patched
    # 1st: Copper succeeds, Zinc fails -> retry, which repeats Copper
    # (identical SQL + params) and runs Zinc again.
    results.extend([SqlResult(sql="", columns=["avg_value"], rows=[["8500"]], row_count=1),
                    SqlResult(sql="", error="timeout")])
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [
            {"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}},
            {"table": "prices", "query_id": "avg_price", "params": {"c": "Zinc"}}]},
        {"needed": True, "calls": [
            {"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}},
            {"table": "prices", "query_id": "avg_price", "params": {"c": "zinc"}}]},
    ]})
    ev, _ = await _go(router)
    assert [e["params"] for e in ev] == [{"c": "Copper"}, {"c": "zinc"}]
    assert [e["rank"] for e in ev] == [1, 2]


async def test_empty_result_is_an_answer_not_a_retry(patched, monkeypatch):
    _, results = patched
    results.append(SqlResult(sql="", columns=["avg_value"]))       # empty, values all exist

    async def no_missing(checks):
        return []

    async def combos(checks):
        return [{"table": "prices", "where": {"commodity": "Iron ore, cfr spot"},
                 "existing_combinations": [{"unit": "($/dmtu)"}]}]

    monkeypatch.setattr(rs, "missing_values", no_missing)
    monkeypatch.setattr(rs, "combination_hints", combos)
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "x"}}]},
        {"needed": False},
    ]})
    _, plan = await _go(router)
    assert not plan["retried"] and len(router.calls) == 1


def test_carried_sql_evidence_reaches_the_final_round():
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    assert 'if ev.get("kind") in ("sql_result", "table_metadata")],' in src   # SQL results + table cards
    assert "if _carry_sql_evidence:" in src


async def test_unreadable_selection_reply_is_retried_not_read_as_not_needed(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        '{"needed": true, "calls": [{"table": "prices", "query_id": "avg_pr',       # cut off
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Copper"}}]}]})
    ev, plan = await _go(router)
    assert len(router.calls) == 2 and "cut off or was not valid JSON" in router.calls[1][1]
    assert plan["select_error"].startswith("selection reply unreadable") and plan["needed"]
    assert len(ev) == 1 and len(calls) == 1


async def test_listing_call_gets_the_listing_budget_and_compact_rows(patched, monkeypatch):
    seen = {}

    async def fake_run(sql, params, allowed, *, max_rows=50, timeout_ms=5000, declared_params=None):
        seen["max_rows"] = max_rows
        rows = [[f"c{i}", "2024", str(100 + i)] for i in range(120)]
        return SqlResult(sql=sql, params=dict(params or {}), columns=["commodity", "period", "value"],
                         rows=rows, row_count=120)

    monkeypatch.setattr(rs, "run_sql", fake_run)
    router = FakeRouter({"table_query_select": [{"needed": True, "calls": [
        {"table": "prices", "listing": True, "query_id": "all_prices", "params": {}}]}]})
    ev, plan = await _go(router, max_rows=50, listing_max_rows=2000, listing_char_cap=60000)
    assert seen["max_rows"] == 2000 and plan["calls"][0]["listing"]
    e = ev[0]
    assert e["listing"] and e["char_cap"] == 60000 and len(e["rows"]) == 120
    assert "LISTING, 120 row(s)" in e["text"] and "commodity | period | value ;; c0 | 2024 | 100" in e["text"]
    assert "c119 | 2024 | 219" in e["text"]                      # every row fits the listing budget
    ev2, _ = await _go(FakeRouter({"table_query_select": [{"needed": True, "calls": [
        {"table": "prices", "listing": True, "query_id": "all_prices", "params": {}}]}]}),
        listing_char_cap=600)
    assert "PARTIAL RESULT:" in ev2[0]["text"]                     # shown part is cut; data is not
    assert "the query returned 120 rows" in ev2[0]["text"]
    assert len(ev2[0]["rows"]) == 120


async def test_questions_about_the_tables_get_their_metadata_not_sql(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": False, "reason": "about the tables", "describe": ["prices", "companies", "nope"]}]})
    ev, plan = await _go(router)
    assert calls == []                                            # no SQL at all
    assert [e["kind"] for e in ev] == ["table_metadata", "table_metadata"]
    assert ev[0]["iri"] == "https://x/o#TableMetadata_prices_1" and ev[0]["node_id"] == NODE
    assert "Table prices" in ev[0]["text"] and "worldbank/pink-sheet" in ev[0]["text"]
    assert plan["needed"] and plan["describe"] == ["prices", "companies"]   # unknown table dropped
    user = router.calls[0][1]
    assert "ALL TABLES" in user and '"table":"companies"' in user        # every table is offered


async def test_chained_call_takes_its_param_from_an_earlier_result(patched):
    calls, results = patched
    results.append(SqlResult(sql="", columns=["commodity", "n"], rows=[["Copper", "3"], ["Zinc", "2"]],
                             row_count=2))
    router = FakeRouter({"table_query_select": [{"needed": True, "calls": [
        {"table": "prices", "query_id": "commodity_counts", "params": {}},
        {"table": "prices", "query_id": "avg_price",
         "params": {"c": {"from_call": 1, "column": "commodity"}}}]}]})
    ev, plan = await _go(router)
    assert [c[1] for c in calls[1:]] == [{"c": "Copper"}, {"c": "Zinc"}]     # one run per value
    assert len(ev) == 3 and not plan["retried"]
    assert plan["calls"][1]["from_calls"] == {"c": {"from_call": 1, "column": "commodity"}}


async def test_chained_call_without_a_usable_result_fails_into_the_retry(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price",
                                    "params": {"c": {"from_call": 2, "column": "commodity"}}}]},
        {"needed": False}]})
    ev, plan = await _go(router)
    assert calls == [] and ev == [] and plan["retried"]
    assert "no result to take values from" in plan["calls"][0]["error"]
    assert "no result to take values from" in router.calls[1][1]


async def test_value_the_table_does_not_hold_reaches_the_answer_as_evidence(patched, monkeypatch):
    calls, _ = patched

    async def fake_missing(checks):
        return [{"table": "prices", "column": "commodity", "value": "Unobtainium",
                 "existing_similar": ["Uranium"]}]

    monkeypatch.setattr(rs, "missing_values", fake_missing)
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Unobtainium"}}]},
        {"needed": False}]})
    ev, _plan = await _go(router)
    assert calls == []                                               # refused before any SQL ran
    assert len(ev) == 1 and ev[0]["kind"] == "sql_result" and ev[0]["query_id"] == "value_check"
    assert ev[0]["text"].startswith("Value check on table prices")
    assert "'Unobtainium' does not occur in column commodity" in ev[0]["text"]
    assert "Uranium" in ev[0]["text"] and re.search(r"#SqlResult_prices_value_check_[0-9a-f]{8}$", ev[0]["iri"])


def test_value_check_skips_values_a_call_used_successfully_and_lists_refused_listed_values():
    calls = [
        {"table": "prices", "error": "x", "missing_values": [
            {"table": "prices", "column": "commodity", "value": "Iron ore", "existing_similar": ["Iron ore, cfr spot"]},
            {"table": "prices", "column": "commodity", "value": "Zink", "existing_similar": []}]},
        {"table": "prices", "error": "x", "invalid_values": [
            {"param": "u", "value": "($/t)", "column": "unit", "allowed_values": ["($/mt)", "($/kg)"],
             "more_values": 3}]},
        {"table": "prices", "error": "x", "missing_values": [
            {"table": "prices", "column": "commodity", "value": "Copr", "existing_similar": ["Copper"]}]},
        {"table": "prices", "error": None, "row_count": 1, "params": {"c": "Iron ore"}},
        {"table": "prices", "error": None, "row_count": 1, "params": {"c": ["Copper"]}},
    ]
    ev = rs.value_check_evidence(calls, {"prices": {"meta": META, "id": NODE}}, 3)
    assert len(ev) == 1 and ev[0]["rank"] == 3
    text = ev[0]["text"]
    assert "Iron ore'" not in text and "Copr" not in text            # used / corrected later
    assert "'Zink' does not occur in column commodity; no similar value occurs." in text
    assert "'($/t)' is not one of the values of column unit (its values: ($/mt) | ($/kg) (+3 more))." in text
    assert rs.value_check_evidence([{"table": "prices", "error": None, "row_count": 2}], {}, 1) == []


def test_selection_prompt_puts_the_question_last_and_keeps_catalogs_question_independent():
    from backend.app.services.prompts import PROMPTS
    cat = [rs.catalog_for_prompt(META, ranking={"avg_price": (2.0, ["class:Price"])}, stable=True)]
    assert cat == [rs.catalog_for_prompt(META, stable=True)]          # ranking does not change the text
    assert "matched" not in json.dumps(cat)
    _s, u1 = PROMPTS["table_query_select"]("first question?", cat, relevance=[{"table": "prices"}])
    _s, u2 = PROMPTS["table_query_select"]("another one?", cat,
                                           doc_context=[{"id": "c", "text": "doc"}])
    assert u1.rstrip().endswith("QUESTION: first question?")
    head = u1.index("RELEVANCE")
    assert u1[:head] == u2[:head]                                     # shared prefix (cacheable)
    tables = u1[:u1.index("Return JSON")]
    assert '": ' not in tables and "\n " not in tables                # compact JSON

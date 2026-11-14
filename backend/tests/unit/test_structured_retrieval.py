"""`--use-structured-data` retrieval step: selection, execution, retry, evidence.

No database: `candidate_tables` and `run_sql` are monkeypatched. The real-DB
path is covered in test_structured_db.py.
"""

from __future__ import annotations

import re
import uuid

import pytest

from backend.app.services.structured import retrieval_step as rs
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
                 "params": [{"name": "c", "type": "text", "column": "commodity", "example": "Copper"}],
                 "sql": "(compiled at run time)"}],
    "filters": [], "queries": [],
    "lookups": [{"id": "company_prices", "other_table": "companies", "params": [],
                 "join": [{"column": "commodity", "other_column": "commodity"}],
                 "query": {"select": ["companies.company", "value"]}}],
}
AVG_SQL = ('SELECT AVG("value") AS "avg_value" FROM "prices" '
           'WHERE "commodity" = CAST(:c AS text)')
COMPANIES = {"table_name": "companies", "columns": [{"name": "company", "type": "text"},
                                                    {"name": "commodity", "type": "text"}]}
NODE = uuid.uuid4()


@pytest.fixture
def patched(monkeypatch):
    calls: list[tuple[str, dict, set]] = []
    results: list[SqlResult] = []

    async def fake_cands(entity_ids, class_ids, qvec, *, max_tables=3, vector_max_distance=None):
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

    monkeypatch.setattr(rs, "candidate_tables", fake_cands)
    monkeypatch.setattr(rs, "table_metas", fake_metas)
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


async def test_failed_catalog_query_gets_one_retry_with_a_query_spec(patched):
    calls, results = patched
    results.append(SqlResult(sql="", error="column \"x\" does not exist"))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Cu"}}]},
        {"needed": True, "calls": [{"table": "prices", "purpose": "max price",
                                    "query": {"select": [{"agg": "max", "column": "value", "as": "m"}]},
                                    "params": {}}]},
    ]})
    ev, plan = await _go(router)
    assert len(router.calls) == 2 and "PREVIOUS ATTEMPT FAILED" in router.calls[1][1]
    assert 'column \\"x\\" does not exist' in router.calls[1][1]
    assert plan["retried"] and len(ev) == 1 and ev[0]["sql_source"] == "spec"
    assert ev[0]["query_id"] is None
    assert calls[1][0] == 'SELECT MAX("value") AS "m" FROM "prices"' and calls[1][2] == {"prices"}


async def test_model_written_sql_is_never_run(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "new_sql": "SELECT 1 FROM prices", "params": {}}]},
        {"needed": True, "calls": [{"table": "prices", "query": {"select": ["nope"]}, "params": {}}]},
    ]})
    ev, plan = await _go(router)
    assert ev == [] and calls == []
    assert "neither query_id nor query" in plan["calls"][0]["error"]
    assert "does not compile" in plan["calls"][1]["error"]


async def test_empty_result_also_retries_and_only_once(patched):
    _, results = patched
    results.extend([SqlResult(sql="", columns=["avg_value"]), SqlResult(sql="", columns=["avg_value"])])
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Zinc"}}]},
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "zinc"}}]},
    ]})
    ev, plan = await _go(router)
    assert ev == [] and plan["retried"] and len(router.calls) == 2   # no third call


async def test_unknown_query_id_is_a_failure_not_a_crash(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "nope", "params": {}}]},
        {"needed": False},
    ]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and plan["retried"]
    assert "unknown query_id" in plan["calls"][0]["error"]


async def test_lookup_may_read_its_partner_table(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "companies", "query_id": "company_prices", "params": {}}]}]})
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
    assert "capped at 50 of 1234 rows" in t and "more fetched row(s) not shown" in t
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


async def test_missing_value_triggers_retry_with_hints_and_keeps_partial_result(patched, monkeypatch):
    _calls, _ = patched
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
    assert len(ev) == 2                                        # partial + corrected result


async def test_identical_retry_result_is_deduplicated(patched):
    _, results = patched
    # 1st: Copper succeeds (default), Zinc comes back empty -> retry, which
    # repeats Copper (identical SQL + params) and fixes Zinc.
    results.extend([SqlResult(sql="", columns=["avg_value"], rows=[["8500"]], row_count=1),
                    SqlResult(sql="", columns=["avg_value"])])
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


async def test_empty_result_retry_gets_existing_combinations(patched, monkeypatch):
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
    assert plan["retried"] and "($/dmtu)" in router.calls[1][1]
    assert plan["calls"][0]["value_combinations"][0]["where"] == {"commodity": "Iron ore, cfr spot"}


def test_carried_sql_evidence_reaches_the_final_round():
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    assert '_carry_sql_evidence=[ev for ev in r1.evidence if ev.get("kind") == "sql_result"]' in src
    assert "if _carry_sql_evidence:" in src

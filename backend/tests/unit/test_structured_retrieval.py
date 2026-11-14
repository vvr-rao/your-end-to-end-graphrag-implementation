"""`--use-structured-data` retrieval step: selection, execution, retry, evidence.

No database: `candidate_tables` and `run_sql` are monkeypatched. The real-DB
path is covered in test_structured_db.py.
"""

from __future__ import annotations

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
                 "params": [{"name": "c", "type": "text", "column": "commodity"}],
                 "sql": "SELECT AVG(value) AS avg_value FROM prices WHERE commodity = :c"}],
    "filters": [], "queries": [],
    "lookups": [{"id": "company_prices", "other_table": "companies", "params": [],
                 "sql": "SELECT c.company, p.value FROM companies c JOIN prices p ON p.commodity = c.commodity"}],
}
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
        return r

    monkeypatch.setattr(rs, "candidate_tables", fake_cands)
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
    assert calls == [(META["metrics"][0]["sql"], {"c": "Copper"}, {"prices"})]
    assert len(ev) == 1 and ev[0]["kind"] == "sql_result" and ev[0]["node_id"] == NODE
    assert ev[0]["query_id"] == "avg_price" and ev[0]["sql_source"] == "catalog"
    assert "avg_value=8500" in ev[0]["text"] and "Parameters: c=Copper" in ev[0]["text"]
    assert ev[0]["iri"].endswith("#SqlResult_prices_avg_price_1")
    assert plan["needed"] and not plan["retried"] and plan["calls"][0]["source"] == "catalog"


async def test_not_needed_runs_no_sql(patched):
    calls, _ = patched
    router = FakeRouter({"table_query_select": [{"needed": False, "reason": "no table data"}]})
    ev, plan = await _go(router)
    assert ev == [] and calls == [] and plan["needed"] is False


async def test_failed_catalog_query_gets_one_retry_with_new_sql(patched):
    calls, results = patched
    results.append(SqlResult(sql="", error="column \"x\" does not exist"))
    router = FakeRouter({"table_query_select": [
        {"needed": True, "calls": [{"table": "prices", "query_id": "avg_price", "params": {"c": "Cu"}}]},
        {"needed": True, "calls": [{"table": "prices", "new_sql": "SELECT MAX(value) AS m FROM prices",
                                    "params": {}, "purpose": "max price"}]},
    ]})
    ev, plan = await _go(router)
    assert len(router.calls) == 2 and "PREVIOUS ATTEMPT FAILED" in router.calls[1][1]
    assert 'column \\"x\\" does not exist' in router.calls[1][1]
    assert plan["retried"] and len(ev) == 1 and ev[0]["sql_source"] == "generated"
    assert ev[0]["query_id"] is None and calls[1][2] == {"prices", "companies"}


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

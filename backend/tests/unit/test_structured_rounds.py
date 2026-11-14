"""deep_research 2-round retrieval and structured data: tables are selected on
the USER's question in both rounds (the planner's bridge/final rewrites can
narrow it), round 1's SQL results are carried into round 2, and round 2 does
not select again. No LLM, no database."""

from __future__ import annotations

import inspect

import pytest

from backend.app.services import retrieval
from backend.app.services.retrieval import RetrievalResult, table_selection_question

USER_Q = "how have commodity prices changed over the past 3 years"
BRIDGE_Q = "What are the major commodities to track for price changes?"
FINAL_Q = "How have the prices of the major commodities changed?"
LISTING = {"kind": "sql_result", "listing": True, "node_id": "t1",
           "iri": "https://x/ontology/intelligence-artifact#SqlResult_prices_measure_p_1", "text": "rows"}


class _Router:
    total_cost_usd = 0.0


@pytest.fixture
def rounds(monkeypatch):
    calls: list[dict] = []
    original = retrieval.retrieve_and_answer

    async def plan(_router, q):
        assert q == USER_Q
        return {"needs_second_round": True, "round1_question": BRIDGE_Q, "round2_question": FINAL_Q}

    async def dispatch(question, **kw):
        if kw.get("_round_depth", 0) == 0:
            return await original(question, **kw)
        calls.append({"question": question, **kw})
        if len(calls) == 1:     # round 1 (bridge): found a listing
            return RetrievalResult(answer="bridge answer", mode=kw["mode"], resolved_query=question,
                                   evidence=[dict(LISTING)], cost_usd=0.01)
        return RetrievalResult(answer="final answer", mode=kw["mode"], resolved_query=question,
                               evidence=[dict(LISTING)], cost_usd=0.02)

    monkeypatch.setattr(retrieval, "LLMRouter", _Router)
    monkeypatch.setattr(retrieval, "_plan_rounds", plan)
    monkeypatch.setattr(retrieval, "retrieve_and_answer", dispatch)
    return original, calls


async def test_both_rounds_select_tables_on_the_users_question(rounds):
    original, calls = rounds
    res = await original(USER_Q, mode="deep_research", use_structured_data=True, persist=False)
    assert [c["question"].split("\n")[0] for c in calls] == [BRIDGE_Q, FINAL_Q]   # rewrites drive text retrieval
    assert [c["_table_question"] for c in calls] == [USER_Q, USER_Q]             # ... tables use the user's
    assert calls[0]["mode"] == "simple_qa" and calls[1]["mode"] == "deep_research"
    assert calls[0].get("_carry_sql_evidence") is None
    assert [e["iri"] for e in calls[1]["_carry_sql_evidence"]] == [LISTING["iri"]]
    assert all(c["use_structured_data"] is True for c in calls)
    assert res.answer == "final answer" and res.resolved_query == USER_Q
    assert res.parsed["rounds"]["round1_question"] == BRIDGE_Q


def test_round_two_reuses_round_one_results_instead_of_selecting_again():
    assert table_selection_question(FINAL_Q, None, None) == FINAL_Q                 # single round
    assert table_selection_question(BRIDGE_Q, USER_Q, None) == USER_Q               # round 1
    assert table_selection_question(FINAL_Q, USER_Q, [LISTING]) is None             # round 2: reuse
    assert table_selection_question(FINAL_Q, USER_Q, []) == USER_Q                  # round 1 found none


def test_structured_step_uses_the_selection_question():
    src = inspect.getsource(retrieval.retrieve_and_answer)
    assert "table_selection_question(resolved_query, _table_question, _carry_sql_evidence)" in src
    assert "structured_evidence(\n                _tq," in src

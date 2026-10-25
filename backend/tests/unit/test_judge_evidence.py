"""Eval judges see the evidence the synthesis saw. They used to get the first
15 items cut to 400 chars, so the LG revenue figure (char 761 of its chunk)
was scored as fabricated in both correct simple_qa answers."""
from __future__ import annotations

import types

import pytest

from backend.app.services import eval_judge
from backend.app.services.prompts import PROMPTS
from backend.app.services.prompts_deepseek import DEEPSEEK_JUDGES

FIGURE = "Revenue was KRW 2,177,068 million"


def _evidence(n: int = 30) -> list[dict]:
    chunk = "x" * 761 + FIGURE + "y" * 1900
    return [{"kind": "chunk", "iri": f"Chunk_{i}", "text": chunk} for i in range(n)]


@pytest.mark.parametrize("judges", [PROMPTS, DEEPSEEK_JUDGES], ids=["shared", "deepseek"])
def test_no_hallucination_sees_every_item_in_full(judges) -> None:
    _, user = judges["judge_no_hallucination"]("q", _evidence(), "a")
    assert user.count(FIGURE) == 30
    assert "Chunk_29" in user


@pytest.mark.parametrize("judges", [PROMPTS, DEEPSEEK_JUDGES], ids=["shared", "deepseek"])
def test_gap_detection_keeps_five_items_but_not_cut(judges) -> None:
    _, user = judges["judge_gap_detection"]("q", _evidence(), "a", False)
    assert user.count(FIGURE) == 5


def test_judge_evidence_uses_the_synthesis_cap(monkeypatch) -> None:
    cfg = types.SimpleNamespace(app_config={"qa": {"evidence_char_cap": 800}})
    monkeypatch.setattr(eval_judge, "get_settings", lambda: cfg)
    ev = _evidence(2) + [{"kind": "chunk", "iri": "short", "text": "ok"}]
    out = eval_judge._as_synthesized(ev)
    assert [len(e["text"]) for e in out] == [800, 800, 2]
    assert FIGURE[:20] in out[0]["text"]
    assert len(ev[0]["text"]) > 800                    # caller's list untouched

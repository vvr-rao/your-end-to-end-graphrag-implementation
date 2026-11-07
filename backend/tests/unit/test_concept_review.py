"""The concept review step: verdicts, and failing open."""
from __future__ import annotations

import pytest
import yaml

from backend.app.services import concept_review as cr


MENU = [
    {"iri": "x#Process", "label": "Process", "description": "A series of actions."},
    {"iri": "x#Rule", "label": "Rule", "description": "A rule."},
    {"iri": "x#Market", "label": "Market", "description": "A market phenomenon."},
]


def _concept(name: str, iri: str = "x#Market") -> dict:
    return {"canonical_name": name, "short_name": name, "class_iri": iri, "confidence": 0.9}


def _apply(verdicts, concepts, samples=None):
    stats = cr.new_stats()
    return cr.apply_verdicts({"verdicts": verdicts}, concepts, MENU, stats, samples), stats


def test_each_verdict() -> None:
    concepts = [_concept(n) for n in ("a", "b", "c", "d")]
    out, stats = _apply([
        {"id": 0, "verdict": "keep"},
        {"id": 1, "verdict": "move", "better_class_iri": "x#Rule", "reason": "is a rule"},
        {"id": 2, "verdict": "drop", "reason": "vague"},
        {"id": 3, "verdict": "no_class"},
    ], concepts)
    assert out[0] is concepts[0]
    assert out[1]["class_iri"] == "x#Rule" and out[1]["canonical_name"] == "b"
    assert concepts[1]["class_iri"] == "x#Market"  # the input is not mutated
    assert out[2] is None
    assert out[3] == {**concepts[3], cr.DOUBTFUL_KEY: True}  # kept, flagged
    assert cr.DOUBTFUL_KEY not in concepts[3]
    assert (stats["review_kept"], stats["review_moved"], stats["review_dropped"],
            stats["review_flagged"], stats["review_checked"]) == (1, 1, 1, 1, 4)


def test_move_accepts_a_label_in_the_iri_field() -> None:
    out, _ = _apply([{"id": 0, "verdict": "move", "better_class_iri": "rule"}], [_concept("a")])
    assert out[0]["class_iri"] == "x#Rule"


@pytest.mark.parametrize("target", ["x#NotOnTheMenu", "", None])
def test_unusable_move_keeps_the_class(target) -> None:
    c = _concept("a")
    out, stats = _apply([{"id": 0, "verdict": "move", "better_class_iri": target}], [c])
    assert out == [c] and stats["review_bad_move"] == 1 and stats["review_kept"] == 1


def test_move_to_the_same_class_is_a_keep() -> None:
    out, stats = _apply([{"id": 0, "verdict": "move", "better_class_iri": "x#Market"}], [_concept("a")])
    assert out[0]["class_iri"] == "x#Market"
    assert stats["review_moved"] == 0 and stats["review_kept"] == 1 and stats["review_bad_move"] == 0


@pytest.mark.parametrize("verdicts", [
    [],
    [{"id": 9, "verdict": "drop"}],
    [{"id": "zero", "verdict": "drop"}],
    [{"id": 0, "verdict": "delete it"}],
    ["drop"],
])
def test_missing_or_bad_verdict_keeps_the_concept(verdicts) -> None:
    c = _concept("a")
    out, stats = _apply(verdicts, [c])
    assert out == [c] and stats["review_unanswered"] == 1


def test_first_verdict_for_an_id_wins() -> None:
    c = _concept("a")
    out, _ = _apply([{"id": 0, "verdict": "keep"}, {"id": 0, "verdict": "drop"}], [c])
    assert out == [c]


def test_samples_record_changes_only() -> None:
    samples: list[dict] = []
    _apply([{"id": 0, "verdict": "keep"}, {"id": 1, "verdict": "drop", "reason": "vague"}],
           [_concept("a"), _concept("b")], samples)
    assert [s["name"] for s in samples] == ["b"] and samples[0]["reason"] == "vague"


class _Router:
    def __init__(self, text=None, exc=None, spec=None):
        self.text, self.exc, self.spec, self.calls = text, exc, spec, 0

    def task_spec(self, task):
        if self.spec is None:
            raise KeyError(task)
        return self.spec

    async def chat(self, task, system, user):
        self.calls += 1
        assert task == cr.TASK
        if self.exc:
            raise self.exc
        return type("R", (), {"text": self.text})()


def _build(text, menu, items):
    return "s", "u"


@pytest.mark.asyncio
@pytest.mark.parametrize("router", [_Router(exc=RuntimeError("boom")), _Router(text="not json")])
async def test_failed_call_keeps_every_concept(router) -> None:
    stats = cr.new_stats()
    concepts = [_concept("a"), _concept("b")]
    out = await cr.review_concepts(router, _build, "t", concepts, MENU, stats, log=lambda _: None)
    assert out == concepts and stats["review_failed"] == 1


@pytest.mark.asyncio
async def test_no_concepts_costs_no_call() -> None:
    router = _Router(text="{}")
    assert await cr.review_concepts(router, _build, "t", [], MENU, cr.new_stats()) == []
    assert router.calls == 0


@pytest.mark.asyncio
async def test_a_reply_is_applied() -> None:
    router = _Router(text='{"verdicts": [{"id": 0, "verdict": "drop", "reason": "vague"}]}')
    out = await cr.review_concepts(router, _build, "t", [_concept("a")], MENU, cr.new_stats())
    assert out == [None]


def test_resolve_prompt_gates() -> None:
    assert cr.resolve_prompt(_Router()) is None
    assert cr.resolve_prompt(_Router(spec={"enabled": False})) is None
    assert cr.resolve_prompt(_Router(spec={"enabled": True})).__name__ == "concept_review"
    assert cr.resolve_prompt(_Router(spec={"prompt_set": "deepseek"})).__name__ == "deepseek_concept_review"
    assert cr.resolve_prompt(_Router(spec={"prompt_set": "kimi"})).__name__ == "kimi_concept_review"
    assert cr.resolve_prompt(_Router(spec={"prompt_set": "qwen"}), log=lambda _: None) is None


@pytest.mark.parametrize("build", [cr._prompt_for(None), cr._prompt_for("deepseek"), cr._prompt_for("kimi")])
def test_prompts_show_the_passage_class_list_and_concepts(build) -> None:
    items = [{"id": 0, "name": "harbor fees", "class_iri": "x#Market", "class_label": "Market"}]
    system, user = build("The passage text.", MENU, items)
    for part in ("The passage text.", "x#Rule", "harbor fees", "x#Market", "no_class"):
        assert part in user
    for verdict in ("keep", "move", "drop", "no_class"):
        assert verdict in system


def test_presets_define_the_task() -> None:
    for f in ("example", "openai.example", "anthropic.example"):
        spec = yaml.safe_load(open(f"config/models.{f}.yaml", encoding="utf-8"))["tasks"]["concept_review"]
        assert spec["enabled"] is False and "prompt_set" not in spec
    di = yaml.safe_load(open("config/models.deepinfra.example.yaml", encoding="utf-8"))["tasks"]
    assert di["concept_review"]["enabled"] is False
    assert di["concept_review"]["provider"] == "deepinfra"
    assert di["concept_review"]["prompt_set"] == "kimi"
    assert di["concept_review"]["model"] == "moonshotai/Kimi-K3"
    assert di["concept_review"]["model"] != di["concept_extract"]["model"]


def test_doubtful_names_need_most_mentions_of_the_main_class() -> None:
    primary = {"a": "x#One", "b": "x#One", "c": "x#One"}
    mentions = [
        ("a", "x#One", True), ("a", "x#One", True), ("a", "x#One", False),  # 2 of 3
        ("b", "x#One", True), ("b", "x#One", False),  # 1 of 2 is not most
        ("c", "x#Two", True), ("c", "x#One", False),  # flagged under another class
        ("d", "x#One", True),  # no recorded main class: its own class counts
    ]
    assert cr.doubtful_names(mentions, primary) == {"a", "d"}


class _Session:
    def __init__(self):
        self.sql: list[str] = []

    async def execute(self, statement, params=None):
        self.sql.append(str(statement))
        return type("R", (), {"all": lambda _self: []})()


@pytest.mark.asyncio
async def test_both_concept_seed_routes_skip_flagged_concepts(monkeypatch) -> None:
    from uuid import uuid4

    from backend.app.services import retrieval_sql as rsql

    async def closure(*a, **k):
        return {uuid4(): 1}

    monkeypatch.setattr(rsql, "resolve_class_closure", closure)
    flag = f"'{cr.DOUBTFUL_KEY}'"
    session = _Session()
    await rsql.concept_seed_entities(session, ["t"], [[0.0]], [0.0], {uuid4()})
    await rsql.concept_entities_by_name(session, ["t"], [[0.0]], {uuid4()}, max_l2=1.0)
    assert len(session.sql) == 2 and all(flag in q for q in session.sql)


@pytest.mark.asyncio
async def test_flagged_concepts_stay_concepts_for_the_walk() -> None:
    from backend.app.services import retrieval_sql as rsql

    session = _Session()
    await rsql.concept_entity_ids(session, ["Quantity"])
    assert cr.DOUBTFUL_KEY not in session.sql[0]

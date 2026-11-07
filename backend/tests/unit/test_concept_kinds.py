"""Named things vs real concepts among entities typed under the concept roots."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services import concept_kinds
from backend.app.services.concept_kinds import _judge_prompt, classify_concept_kinds, parse_decisions

BATCH = [{"entity_id": "a", "name": "BMW iX3", "class_label": "Vehicle"},
         {"entity_id": "b", "name": "operating margin", "class_label": "Metric"},
         {"entity_id": "c", "name": "hypoglycemia", "class_label": "Condition"}]


class _Router:
    total_cost_usd = 0.0

    def __init__(self, spec=None, reply=None, error=None):
        self.spec, self.reply, self.error = spec, reply, error
        self.calls = []

    def task_spec(self, task):
        if self.spec is None:
            raise KeyError(task)
        return self.spec

    async def chat(self, task, *, system, user):
        self.calls.append(user)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.reply)


@pytest.fixture
def db(monkeypatch):
    """The two DB helpers, with the saved verdicts captured."""
    saved: list[dict[str, str]] = []

    async def _load(roots, *, limit, recheck):
        return BATCH[:limit] if limit else list(BATCH)

    async def _save(verdicts):
        saved.append(verdicts)

    monkeypatch.setattr(concept_kinds, "_load", _load)
    monkeypatch.setattr(concept_kinds, "_save", _save)
    return saved


def test_parse_keeps_only_valid_answers() -> None:
    got = [{"id": 0, "kind": "named"}, {"id": "1", "kind": " Concept "},
           {"id": 2, "kind": "maybe"}, {"id": 9, "kind": "named"},
           {"kind": "named"}, "junk"]
    assert parse_decisions(got, BATCH) == {"a": "named", "b": "concept"}
    assert parse_decisions(None, BATCH) == {}


async def test_verdicts_are_saved_and_counted(db) -> None:
    reply = json.dumps({"decisions": [{"id": 0, "kind": "named"}, {"id": 1, "kind": "concept"},
                                      {"id": 2, "kind": "concept"}]})
    s = await classify_concept_kinds(_Router({"enabled": True}, reply), ["Quantity"], log=lambda _: None)
    assert db == [{"a": "named", "b": "concept", "c": "concept"}]
    assert (s["named"], s["concept"], s["unanswered"], s["named_sample"]) == (1, 2, 0, ["BMW iX3"])


async def test_dry_run_saves_nothing(db) -> None:
    reply = json.dumps({"decisions": [{"id": 0, "kind": "named"}]})
    s = await classify_concept_kinds(_Router({"enabled": True}, reply), ["Quantity"], dry_run=True, log=lambda _: None)
    assert db == [] and s["named"] == 1 and s["unanswered"] == 2


async def test_failed_call_leaves_entities_unclassified(db) -> None:
    s = await classify_concept_kinds(_Router({"enabled": True}, error=RuntimeError("x")), ["Quantity"], log=lambda _: None)
    assert db == [] and s["failed_calls"] == 1 and s["unanswered"] == 3


async def test_batches_follow_the_task_setting(db) -> None:
    r = _Router({"entities_per_call": 2}, '{"decisions": []}')
    await classify_concept_kinds(r, ["Quantity"], log=lambda _: None)
    assert len(r.calls) == 2


@pytest.mark.parametrize("router,roots", [
    (_Router(None), ["Quantity"]),                 # task not in models.yaml
    (_Router({"enabled": False}), ["Quantity"]),   # switched off
    (_Router({"enabled": True}), []),                            # no concept roots configured
    (_Router({"prompt_set": "qwen"}), ["Quantity"]),  # no prompt for this model
])
async def test_skipped_makes_no_call(db, router, roots) -> None:
    s = await classify_concept_kinds(router, roots, log=lambda _: None)
    assert router.calls == [] and s["entities"] == 0


def test_each_model_has_its_own_prompt_and_all_presets_define_the_task() -> None:
    assert _judge_prompt(None).__name__ == "concept_kind_judge"
    assert _judge_prompt("deepseek").__name__ == "deepseek_concept_kind_judge"
    for f in ("config/models.example.yaml", "config/models.openai.example.yaml",
              "config/models.anthropic.example.yaml", "config/models.deepinfra.example.yaml"):
        spec = yaml.safe_load(open(f, encoding="utf-8"))["tasks"]["concept_kind_judge"]
        assert spec["enabled"] is True, f
    di = yaml.safe_load(open("config/models.deepinfra.example.yaml", encoding="utf-8"))["tasks"]["concept_kind_judge"]
    assert di["provider"] == "deepinfra" and di["prompt_set"] == "deepseek"


@pytest.mark.parametrize("name", ["concept_kind_judge", "deepseek_concept_kind_judge"])
def test_prompts_carry_every_item(name) -> None:
    build = _judge_prompt("deepseek" if name.startswith("deepseek") else None)
    _, user = build([{"id": 7, "name": "BMW iX3", "class": "Vehicle"}])
    assert "BMW iX3" in user and "Vehicle" in user and "7" in user

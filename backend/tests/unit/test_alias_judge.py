"""alias_judge: verdict parsing, prompt selection, batching, and the
fail-closed filter in front of term_aliases. No network, no database."""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.services import alias_judge as aj
from backend.app.services.alias_mining import make_pair

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- parsing


def test_parse_verdicts_reads_both_lists():
    assert aj.parse_verdicts('{"same":[0,2],"different":[1]}', 3) == {0: True, 2: True, 1: False}


def test_parse_verdicts_tolerates_fences_and_string_ids():
    raw = 'Here:\n```json\n{"same": ["1"], "different": ["0"]}\n```'
    assert aj.parse_verdicts(raw, 2) == {1: True, 0: False}


def test_parse_verdicts_drops_contradictions_and_out_of_range():
    got = aj.parse_verdicts('{"same":[0,1,9],"different":[1,-1]}', 2)
    assert got == {0: True}          # 1 is in both lists, 9 / -1 are not ids


def test_parse_verdicts_garbage_is_empty():
    assert aj.parse_verdicts("I cannot help with that", 3) == {}
    assert aj.parse_verdicts('{"same": [0', 3) == {}


# ---------------------------------------------------------------- context


def test_context_snippet_anchors_on_the_bracketed_side():
    text = "Refinery list.\n\nGermu\nRefinery (PetroChina)\n[ 334 ]\nOther Refinery (Sinopec)"
    snip = aj.context_snippet(text, "Refinery", "PetroChina")
    assert "Germu Refinery (PetroChina)" in snip
    assert "\n" not in snip


def test_context_snippet_missing_text():
    assert aj.context_snippet(None, "a", "b") is None
    assert aj.context_snippet("nothing relevant", "Aramco", "Saudi Aramco") is None


# ---------------------------------------------------------------- config


class _Router:
    def __init__(self, spec=None, replies=None, fail_calls=()):
        self.spec = spec
        self.replies = list(replies or [])
        self.fail_calls = set(fail_calls)
        self.calls: list[str] = []
        self.models: list[str | None] = []
        self.total_cost_usd = 0.0

    def task_spec(self, task):
        if self.spec is None:
            raise KeyError(task)
        return self.spec

    async def chat(self, task, *, system, user, model=None):
        n = len(self.calls)
        self.models.append(model)
        self.calls.append(user)
        self.total_cost_usd += 0.001
        if n in self.fail_calls:
            raise TimeoutError("boom")
        return SimpleNamespace(text=self.replies[n] if n < len(self.replies) else "{}")


def test_missing_task_is_a_loud_error():
    with pytest.raises(aj.AliasJudgeNotConfiguredError, match="alias_judge"):
        aj.judge_spec(_Router(spec=None))


def test_disabled_task_returns_none():
    assert aj.judge_spec(_Router(spec={"model": "m", "enabled": False})) is None


def test_prompt_selection_per_model_family():
    from backend.app.services.prompts import alias_judge
    from backend.app.services.prompts_qwen import qwen_alias_judge

    assert aj.judge_prompt(None) is alias_judge
    assert aj.judge_prompt("qwen") is qwen_alias_judge
    with pytest.raises(aj.AliasJudgeNotConfiguredError):
        aj.judge_prompt("deepseek")


@pytest.mark.parametrize("prompt_set", [None, "qwen"])
def test_prompts_carry_every_pair_and_the_contract(prompt_set):
    build = aj.judge_prompt(prompt_set)
    system, user = build([
        {"id": "0", "a": "Refinery", "b": "PetroChina", "context": "Germu Refinery (PetroChina)"},
        {"id": "1", "a": "Mounjaro", "b": "tirzepatide", "context": "MOUNJARO (tirzepatide)"},
    ])
    assert 'A="Refinery" | B="PetroChina"' in user and "1. " in user
    assert '"same"' in user and '"different"' in user
    assert "DIFFERENT" in system


# Each LLM mode judges with its own model; the deepinfra one is serverless
# (no deployment id placeholder), because delete-document runs with no GPU.
@pytest.mark.parametrize("path, provider, model, prompt_set", [
    ("config/models.example.yaml", "openai", "gpt-4.1", None),
    ("config/models.openai.example.yaml", "openai", "gpt-4.1", None),
    ("config/models.anthropic.example.yaml", "anthropic", "claude-sonnet-4-6", None),
    ("config/models.deepinfra.example.yaml", "deepinfra", "__DEEPINFRA_FLASH_MODEL__", "qwen"),
])
def test_every_preset_defines_the_judge(path, provider, model, prompt_set):
    spec = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["tasks"]["alias_judge"]
    assert (spec["provider"], spec["model"], spec.get("prompt_set")) == (provider, model, prompt_set)
    assert spec.get("enabled", True) is True
    if provider == "deepinfra":
        # dedicated GPU first, serverless 27B when none is running; one cache
        assert spec["fallback_model"] == spec["verdict_key"] == "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize("state, expect", [
    ("running", ("VR/qwen27b-x", 32, "dedicated")),
    ("deleted", ("Qwen/Qwen3.8-27B", 8, "serverless")),
    ("unset", ("Qwen/Qwen3.8-27B", 8, "serverless")),
    ("unknown", ("Qwen/Qwen3.8-27B", 8, "serverless")),
])
def test_route_prefers_a_running_dedicated_deployment(state, expect):
    spec = {"model": "VR/qwen27b-x", "fallback_model": "Qwen/Qwen3.8-27B",
            "concurrency": 32, "fallback_concurrency": 8}
    assert aj.resolve_route(spec, status=lambda _m: state, log=lambda _m: None) == expect


def test_route_without_fallback_uses_the_task_model():
    assert aj.resolve_route({"model": "gpt-4.1-mini", "concurrency": 8},
                            status=lambda _m: 1 / 0, log=lambda _m: None) == ("gpt-4.1-mini", 8, "")


def test_dedicated_status_placeholder_needs_no_network():
    from backend.app.services.deepinfra_dedicated import dedicated_status

    assert dedicated_status("__DEEPINFRA_FLASH_MODEL__") == "unset"
    assert dedicated_status(None) == "unset"


# ---------------------------------------------------------------- batching


def test_judge_batches_maps_ids_back_and_skips_failed_calls():
    items = [{"a": f"A{i}", "b": f"B{i}", "context": ""} for i in range(5)]
    router = _Router(
        spec={"model": "m", "pairs_per_call": 2, "concurrency": 1},
        replies=['{"same":[0],"different":[1]}', "", '{"same":[0],"different":[]}'],
        fail_calls={1},
    )
    got = asyncio.run(aj.judge_batches(router, router.spec, items, log=lambda _m: None))
    assert got == {0: True, 1: False, 4: True}     # items 2-3 were in the failed call
    assert len(router.calls) == 3


# ---------------------------------------------------------------- DB pass


class _FakeDB:
    """Stands in for session_scope: answers the three SQL statements."""

    def __init__(self, cached=(), chunk_texts=None):
        self.cached = list(cached)
        self.chunk_texts = chunk_texts or {}
        self.stored: list[dict] = []

    @contextlib.asynccontextmanager
    async def scope(self):
        yield self

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM graphrag.term_alias_judgements" in sql:
            return SimpleNamespace(all=lambda: self.cached)
        if "FROM graphrag.chunks" in sql:
            return SimpleNamespace(all=lambda: list(self.chunk_texts.items()))
        if "INSERT INTO graphrag.term_alias_judgements" in sql:
            self.stored.extend(params)
            return SimpleNamespace(all=lambda: [])
        raise AssertionError(sql)


def _pairs():
    return [
        make_pair("Germu Refinery", "PetroChina", "parenthetical", "chunk-1"),
        make_pair("MOUNJARO", "tirzepatide", "parenthetical", "chunk-2"),
        make_pair("Hypertension", "High Blood Pressure", "ontology", "iri-1"),
    ]


def _run(db, router, monkeypatch, **kw):
    import backend.app.db.session as session_mod

    monkeypatch.setattr(session_mod, "session_scope", db.scope)
    return asyncio.run(aj.judge_mined_pairs(_pairs(), router=router, log=lambda _m: None, **kw))


def _spec():
    return {"model": "judge-m", "pairs_per_call": 40, "concurrency": 2}


def test_rejected_pairs_are_dropped_and_ontology_is_exempt(monkeypatch):
    db = _FakeDB(chunk_texts={"chunk-1": "Germu Refinery (PetroChina)", "chunk-2": "MOUNJARO (tirzepatide)"})
    pairs = _pairs()
    # Uncached pairs keep mined order: index 0 = refinery pair, 1 = mounjaro.
    router = _Router(spec=_spec(), replies=['{"same":[1],"different":[0]}'])
    kept, rep = _run(db, router, monkeypatch)
    assert {(p.surface_a, p.surface_b) for p in kept} == {
        (pairs[1].surface_a, pairs[1].surface_b), (pairs[2].surface_a, pairs[2].surface_b)}
    assert (rep.judged_new, rep.rejected, rep.exempt, rep.unjudged) == (2, 1, 1, 0)
    assert {r["same"] for r in db.stored} == {True, False}
    assert all(r["v"] == aj.JUDGE_VERSION and r["m"] == "judge-m" for r in db.stored)
    assert "PetroChina" in router.calls[0] and "Germu Refinery (PetroChina)" in router.calls[0]
    assert router.models == ["judge-m"]


def test_verdicts_are_cached_under_verdict_key_whatever_the_route(monkeypatch):
    db = _FakeDB()
    spec = {**_spec(), "model": "VR/dedicated", "fallback_model": "Qwen/Qwen3.8-27B",
            "verdict_key": "Qwen/Qwen3.8-27B"}
    router = _Router(spec=spec, replies=['{"same":[0,1],"different":[]}'])
    import backend.app.services.deepinfra_dedicated as dd

    monkeypatch.setattr(dd, "dedicated_status", lambda _m, **_k: "deleted")
    _kept, rep = _run(db, router, monkeypatch)
    assert router.models == ["Qwen/Qwen3.8-27B"] and rep.hosting == "serverless"
    assert {r["m"] for r in db.stored} == {"Qwen/Qwen3.8-27B"}


def test_cached_verdicts_cost_nothing(monkeypatch):
    p = _pairs()
    db = _FakeDB(cached=[(p[0].term_a, p[0].term_b, False), (p[1].term_a, p[1].term_b, True)])
    router = _Router(spec=_spec())
    kept, rep = _run(db, router, monkeypatch)
    assert router.calls == [] and rep.cached == 2 and rep.rejected == 1
    assert len(kept) == 2


def test_unjudged_pairs_fail_closed_but_bank_verdicts(monkeypatch):
    db = _FakeDB()
    router = _Router(spec=_spec(), replies=['{"same":[1],"different":[]}'])   # id 0 missing
    with pytest.raises(aj.AliasJudgeIncompleteError, match="1 pair"):
        _run(db, router, monkeypatch)
    assert len(db.stored) == 1                       # the verdict it did get is cached


def test_dry_run_makes_no_calls_and_keeps_unjudged(monkeypatch):
    db = _FakeDB()
    router = _Router(spec=_spec())
    kept, rep = _run(db, router, monkeypatch, dry_run=True)
    assert router.calls == [] and rep.unjudged == 2 and len(kept) == 3


def test_disabled_judge_passes_everything_through(monkeypatch):
    db = _FakeDB()
    router = _Router(spec={"model": "m", "enabled": False})
    kept, rep = _run(db, router, monkeypatch)
    assert len(kept) == 3 and rep.enabled is False and router.calls == []

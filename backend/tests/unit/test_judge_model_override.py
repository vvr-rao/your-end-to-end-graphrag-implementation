"""evaluate-queries --judge-model: any model, provider inferred or explicit."""
from __future__ import annotations

import pytest

from backend.app.services import eval_judge
from backend.app.services.eval_judge import (
    JUDGE_TASKS,
    describe_judge_models,
    infer_provider,
    override_judge_model,
)
from backend.app.services.llm_router import LLMRouter


def _router(spec: dict, providers=("openai", "anthropic", "groq", "deepinfra")) -> LLMRouter:
    r = LLMRouter.__new__(LLMRouter)
    r._tasks = {t: dict(spec) for t in JUDGE_TASKS}
    r._tasks["other_task"] = {"provider": "openai", "model": "gpt-6-luna"}
    r._providers = {p: object() for p in providers}
    return r


TERRA = {"provider": "openai", "model": "gpt-6.1-terra", "reasoning_effort": "none",
         "temperature": 0.0, "timeout": 820, "max_tokens": 16384, "response_format": "json_object"}


@pytest.mark.parametrize("name,expected", [
    ("claude-sonnet-5-5", "anthropic"), ("gpt-4.1", "openai"), ("gpt-6.2-sol", "openai"),
    ("o4-mini", "openai"), ("Qwen/Qwen3.8-27B", "deepinfra"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepinfra"), ("llama-3.3-70b-versatile", None),
])
def test_infer_provider(name, expected):
    assert infer_provider(name) == expected


def test_same_provider_swaps_model_and_drops_tuned_effort():
    r = _router(TERRA)
    override_judge_model(r, "gpt-4.1")
    for t in JUDGE_TASKS:
        assert r._tasks[t]["provider"] == "openai" and r._tasks[t]["model"] == "gpt-4.1"
        assert "reasoning_effort" not in r._tasks[t]          # gpt-4.1 rejects it
        assert r._tasks[t]["max_tokens"] == 16384             # other settings kept
    assert r._tasks["other_task"]["model"] == "gpt-6-luna"    # non-judge untouched


def test_cross_provider_drops_provider_specific_settings():
    r = _router(TERRA)
    override_judge_model(r, "claude-sonnet-5-5")
    spec = r._tasks["judge_consistency"]
    assert (spec["provider"], spec["model"]) == ("anthropic", "claude-sonnet-5-5")
    assert "temperature" not in spec and "reasoning_effort" not in spec
    assert describe_judge_models(r) == "anthropic:claude-sonnet-5-5"


def test_explicit_provider_and_groq_stream():
    r = _router({"provider": "anthropic", "model": "claude-sonnet-5-5", "stream": True,
                 "max_tokens": 64000, "timeout": 820})
    override_judge_model(r, "groq:llama-3.3-70b-versatile")
    spec = r._tasks["judge_gap_detection"]
    assert (spec["provider"], spec["model"]) == ("groq", "llama-3.3-70b-versatile")
    assert "stream" not in spec                                # groq cannot stream with usage


def test_deepinfra_prompt_set_dropped_when_leaving_deepinfra():
    r = _router({"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                 "prompt_set": "deepseek", "max_tokens": 8192, "timeout": 600})
    override_judge_model(r, "gpt-6.1-terra")
    assert "prompt_set" not in r._tasks["judge_no_hallucination"]
    assert eval_judge._judge_prompt(r, "judge_no_hallucination") is eval_judge.PROMPTS["judge_no_hallucination"]


def test_unknown_name_keeps_preset_provider():
    r = _router(TERRA)
    override_judge_model(r, "my-finetune-v2")
    assert r._tasks["judge_comprehensiveness"]["provider"] == "openai"


def test_unconfigured_provider_fails_early():
    r = _router(TERRA, providers=("openai",))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        override_judge_model(r, "claude-haiku-5-5")


def test_effort_tuned_for_preset_model_is_dropped_on_model_change():
    r = _router(TERRA)
    override_judge_model(r, "gpt-6.2-sol")          # same provider, new model: effort dropped
    assert "reasoning_effort" not in r._tasks["judge_consistency"]

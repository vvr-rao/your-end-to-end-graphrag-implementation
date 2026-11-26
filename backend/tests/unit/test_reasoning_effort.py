"""Optional per-task `reasoning_effort` for OpenAI reasoning models.

Absent = nothing is sent (the model's own default). Offline: fake client.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.services.llm_router import LLMRouter, OpenAIProvider


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None,
                                completion_tokens_details=None)
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def _provider() -> tuple[OpenAIProvider, _FakeCompletions]:
    p = OpenAIProvider(api_key="test")
    fake = _FakeCompletions()
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return p, fake


@pytest.mark.asyncio
async def test_effort_is_sent_only_when_set() -> None:
    p, fake = _provider()
    await p.chat(model="gpt-6.2-sol", system="s", user="u", temperature=0.0, max_tokens=10,
                 timeout=10, response_format=None)
    assert "reasoning_effort" not in fake.kwargs
    await p.chat(model="gpt-6.2-sol", system="s", user="u", temperature=0.0, max_tokens=10,
                 timeout=10, response_format=None, reasoning_effort="low")
    assert fake.kwargs["reasoning_effort"] == "low"


def _bare_router(tasks: dict) -> LLMRouter:
    router = LLMRouter.__new__(LLMRouter)
    router.reset_counters()
    router._tasks = tasks
    return router


def test_valid_effort_on_an_openai_reasoning_model_is_accepted() -> None:
    _bare_router({"t": {"provider": "openai", "model": "gpt-6-luna", "reasoning_effort": "none"},
                  "u": {"provider": "openai", "model": "gpt-4.1"}})._validate_reasoning_effort()


@pytest.mark.parametrize("spec, msg", [
    ({"provider": "openai", "model": "gpt-4.1", "reasoning_effort": "low"}, "only OpenAI reasoning"),
    ({"provider": "anthropic", "model": "claude-sonnet-4-6", "reasoning_effort": "low"}, "only OpenAI"),
    ({"provider": "openai", "model": "gpt-6.2-sol", "reasoning_effort": "turbo"}, "not one of"),
])
def test_invalid_effort_is_rejected_at_load(spec: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        _bare_router({"t": spec})._validate_reasoning_effort()

"""LLM router — task -> provider dispatch with mocked providers."""

from __future__ import annotations

import pytest

from backend.app.services.llm_router import ChatResult, LLMRouter


class _StubProvider:
    """Minimal Provider impl that records the last call and returns a fixed result."""

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.last_call: dict | None = None

    async def chat(self, **kwargs):
        self.last_call = kwargs
        return ChatResult(
            text='{"ok": true}',
            model=kwargs["model"],
            provider=self.name,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0001,
        )


@pytest.mark.asyncio
async def test_router_dispatches_chunk_classification_to_groq() -> None:
    router = LLMRouter.__new__(LLMRouter)
    router._settings = None  # type: ignore[assignment]
    router._providers = {"groq": _StubProvider(name="groq"), "openai": _StubProvider(name="openai")}
    router._tasks = {
        "chunk_classification": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.0,
            "max_tokens": 100,
            "timeout": 30,
        }
    }
    router._retry = {"max_attempts": 1, "initial_wait_seconds": 0, "max_wait_seconds": 0}
    router._total_cost_usd = 0.0
    router._cache_read_tokens = 0
    router._cache_write_tokens = 0
    router._input_full_tokens = 0
    router._cost_by_task = {}
    router._calls_by_task = {}

    result = await router.chat("chunk_classification", system="sys", user="usr")
    assert result.provider == "groq"
    assert result.model == "llama-3.3-70b-versatile"
    assert router.total_cost_usd == 0.0001
    assert router._providers["groq"].last_call["model"] == "llama-3.3-70b-versatile"
    assert router._providers["openai"].last_call is None


@pytest.mark.asyncio
async def test_router_dispatches_class_proposal_to_openai() -> None:
    router = LLMRouter.__new__(LLMRouter)
    router._settings = None  # type: ignore[assignment]
    router._providers = {"groq": _StubProvider(name="groq"), "openai": _StubProvider(name="openai")}
    router._tasks = {
        "class_proposal": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.0,
            "max_tokens": 2048,
            "timeout": 60,
            "response_format": "json_object",
        }
    }
    router._retry = {"max_attempts": 1, "initial_wait_seconds": 0, "max_wait_seconds": 0}
    router._total_cost_usd = 0.0
    router._cache_read_tokens = 0
    router._cache_write_tokens = 0
    router._input_full_tokens = 0
    router._cost_by_task = {}
    router._calls_by_task = {}

    result = await router.chat("class_proposal", system="sys", user="usr")
    assert result.provider == "openai"
    assert router._providers["openai"].last_call["response_format"] == "json_object"


@pytest.mark.asyncio
async def test_router_raises_on_unknown_task() -> None:
    router = LLMRouter.__new__(LLMRouter)
    router._settings = None  # type: ignore[assignment]
    router._providers = {}
    router._tasks = {}
    router._retry = {}
    router._total_cost_usd = 0.0
    router._cache_read_tokens = 0
    router._cache_write_tokens = 0
    router._input_full_tokens = 0
    router._cost_by_task = {}
    router._calls_by_task = {}
    with pytest.raises(KeyError):
        await router.chat("nonexistent_task", system="", user="")


@pytest.mark.asyncio
async def test_router_raises_if_provider_not_configured() -> None:
    router = LLMRouter.__new__(LLMRouter)
    router._settings = None  # type: ignore[assignment]
    router._providers = {"openai": _StubProvider(name="openai")}  # groq missing
    router._tasks = {"chunk_classification": {"provider": "groq", "model": "x", "max_tokens": 1, "timeout": 1, "temperature": 0}}
    router._retry = {"max_attempts": 1, "initial_wait_seconds": 0, "max_wait_seconds": 0}
    router._total_cost_usd = 0.0
    router._cache_read_tokens = 0
    router._cache_write_tokens = 0
    router._input_full_tokens = 0
    router._cost_by_task = {}
    router._calls_by_task = {}
    with pytest.raises(RuntimeError, match="not configured"):
        await router.chat("chunk_classification", system="", user="")

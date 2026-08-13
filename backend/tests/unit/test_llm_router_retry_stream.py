"""Retry discipline, prompt pre-flight, and opt-in streaming.

Covers the failure the pharma-docs-3 run hit twice:
  - a deterministic 400 retried 5x inside tenacity, each attempt itself
    retried 3x by the SDK -> ~45 min on one unwinnable call
  - a 1,348,538-token prompt sent to a 1,047,576-token model, whose 400 was
    caught and swallowed so the run continued with un-deduplicated results
"""

from __future__ import annotations

import httpx
import openai
import pytest

from backend.app.services.llm_router import (
    AnthropicProvider,
    ChatResult,
    GroqProvider,
    LLMRouter,
    OpenAIProvider,
    PromptTooLargeError,
    _consume_openai_stream,
)


def _router(tasks: dict, providers: dict) -> LLMRouter:
    r = LLMRouter.__new__(LLMRouter)
    r.reset_counters()
    r._settings = None  # type: ignore[assignment]
    r._providers = providers
    r._tasks = tasks
    r._retry = {"max_attempts": 5, "initial_wait_seconds": 0, "max_wait_seconds": 0}
    return r


class _CountingProvider:
    def __init__(self, exc: Exception | None) -> None:
        self.calls = 0
        self.exc = exc
        self.last_kwargs: dict = {}

    async def chat(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return ChatResult(text="ok", model=kwargs["model"], provider="openai")


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "http://x"))


_SPEC = {
    "t": {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.0,
        "max_tokens": 100,
        "timeout": 30,
    }
}


@pytest.mark.asyncio
async def test_deterministic_400_is_attempted_exactly_once() -> None:
    p = _CountingProvider(
        openai.BadRequestError("context_length_exceeded", response=_resp(400), body=None)
    )
    router = _router(_SPEC, {"openai": p})
    with pytest.raises(openai.BadRequestError):
        await router.chat("t", system="s", user="u")
    assert p.calls == 1, "a 400 cannot succeed on retry and must not be retried"


@pytest.mark.asyncio
async def test_rate_limit_is_retried() -> None:
    p = _CountingProvider(openai.RateLimitError("429", response=_resp(429), body=None))
    router = _router(_SPEC, {"openai": p})
    with pytest.raises(openai.RateLimitError):
        await router.chat("t", system="s", user="u")
    assert p.calls == 5, "429 is transient and should exhaust the retry budget"


@pytest.mark.asyncio
async def test_auth_error_is_not_retried() -> None:
    p = _CountingProvider(
        openai.AuthenticationError("bad key", response=_resp(401), body=None)
    )
    router = _router(_SPEC, {"openai": p})
    with pytest.raises(openai.AuthenticationError):
        await router.chat("t", system="s", user="u")
    assert p.calls == 1


@pytest.mark.asyncio
async def test_oversized_prompt_is_rejected_before_any_request() -> None:
    p = _CountingProvider(None)
    tasks = {"t": dict(_SPEC["t"], model="gpt-4o-mini", max_tokens=1000)}  # 128k window
    router = _router(tasks, {"openai": p})
    with pytest.raises(PromptTooLargeError) as ei:
        await router.chat("t", system="s", user="word " * 200_000)
    assert p.calls == 0, "the request must never go out"
    assert "gpt-4o-mini" in str(ei.value)


@pytest.mark.asyncio
async def test_prompt_that_fits_is_not_blocked() -> None:
    p = _CountingProvider(None)
    router = _router(_SPEC, {"openai": p})
    result = await router.chat("t", system="s", user="a modest prompt")
    assert result.text == "ok"
    assert p.calls == 1


@pytest.mark.asyncio
async def test_stream_flag_is_passed_through_from_task_spec() -> None:
    p = _CountingProvider(None)
    tasks = {"t": dict(_SPEC["t"], stream=True)}
    router = _router(tasks, {"openai": p})
    await router.chat("t", system="s", user="u")
    assert p.last_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_defaults_off() -> None:
    p = _CountingProvider(None)
    router = _router(_SPEC, {"openai": p})
    await router.chat("t", system="s", user="u")
    assert p.last_kwargs["stream"] is False


def test_groq_stream_is_rejected_at_config_load_not_mid_run() -> None:
    r = LLMRouter.__new__(LLMRouter)
    r.reset_counters()
    r._tasks = {"chunk_classification": {"provider": "groq", "stream": True}}
    with pytest.raises(ValueError) as ei:
        r._validate_stream_support()
    # Groq's SDK cannot request usage on a stream, so cost would read $0.
    assert "groq" in str(ei.value)
    assert "chunk_classification" in str(ei.value)


def test_groq_without_stream_is_fine() -> None:
    r = LLMRouter.__new__(LLMRouter)
    r.reset_counters()
    r._tasks = {"chunk_classification": {"provider": "groq"}}
    r._validate_stream_support()  # must not raise


@pytest.mark.asyncio
async def test_groq_provider_refuses_stream_rather_than_dropping_it() -> None:
    provider = GroqProvider.__new__(GroqProvider)
    with pytest.raises(ValueError) as ei:
        await provider.chat(
            model="llama-3.3-70b-versatile", system="s", user="u",
            temperature=0.0, max_tokens=10, timeout=5,
            response_format=None, stream=True,
        )
    assert "silently zero" in str(ei.value)


@pytest.mark.asyncio
async def test_openai_stream_consumer_recovers_text_and_usage() -> None:
    """The usage chunk arrives last and carries no choices.

    Reading usage off the final *content* chunk would leave it None, and
    _estimate_cost would then report $0 -- making --max-cost-usd unenforceable.
    """

    class _Delta:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content=None, usage=None):
            self.choices = [_Choice(content)] if content is not None else []
            self.usage = usage

    class _Usage:
        prompt_tokens = 1200
        completion_tokens = 340

    async def _gen():
        for part in ["Hello", ", ", "world"]:
            yield _Chunk(content=part)
        yield _Chunk(usage=_Usage())  # final usage-only chunk

    text, usage = await _consume_openai_stream(_gen())
    assert text == "Hello, world"
    assert usage is not None and usage.prompt_tokens == 1200


def test_clients_disable_sdk_level_retries() -> None:
    """tenacity must be the only retry authority.

    The SDK default of 2 internal retries nests inside tenacity's 5 for up to
    15 HTTP attempts per logical call -- ~45 min at a 180s timeout, and
    invisible because SDK retries emit no log line.
    """
    assert OpenAIProvider(api_key="k")._client.max_retries == 0
    assert GroqProvider(api_key="k")._client.max_retries == 0
    assert AnthropicProvider(api_key="k")._client.max_retries == 0

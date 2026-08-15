"""Provider-abstracted LLM dispatcher.

Reads task -> provider/model from `config/models.yaml`, dispatches each task
to the right SDK (OpenAI or Groq today), applies retry/backoff via tenacity,
and tracks rough cost. Designed so a future provider is one new class plus
one line in PROVIDERS — no service-code changes elsewhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import tenacity
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from openai import AsyncOpenAI

from backend.app.core.config import Settings, get_settings
from backend.app.services import retry_policy
from backend.app.services.token_budget import count_tokens

log = logging.getLogger("llm_router")


@dataclass
class ChatResult:
    """Single LLM call result. Token counts are best-effort (provider may not
    report them); cost_usd is best-effort given fixed per-model pricing.
    """

    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    # Prompt-cache accounting (best-effort; providers that support caching populate
    # these). cache_read_tokens = input tokens served from cache (cheap);
    # cache_write_tokens = input tokens written to cache (small premium).
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    # Why generation stopped: "stop" (complete), "length" (hit max_tokens), or
    # a provider-specific value. Without this, a reply truncated at max_tokens
    # is indistinguishable from a model that returned prose -- both surface as
    # "not parseable JSON", and they have opposite remedies. Cost a 65-minute
    # run to a dedup batch whose true cause could not be named.
    finish_reason: str | None = None


class Provider(Protocol):
    """Provider contract. Implementations only need to expose async chat."""

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        response_format: str | None,
        cache_prefix: str | None = None,
        stream: bool = False,
    ) -> ChatResult: ...


# Assumed prefill (input-processing) throughput, tokens/second, used only to
# split an observed latency into "reading the prompt" and "writing the reply".
# Real prefill is far faster than decode and varies by model and load; a low
# constant here credits MORE of the elapsed time to decode, which makes the
# under-budget projection larger and the warning more eager. That is the safe
# direction -- a missed warning is silent, a spurious one is merely noise.
_PREFILL_TOKENS_PER_S = 3_000.0

# Minimum reply length before a decode rate is believable. Below this, elapsed
# time is dominated by time-to-first-token (a fixed per-request cost), so the
# implied "tokens per second" says more about latency than throughput.
_MIN_TOKENS_TO_RATE = 128

# Minimum decode span before the rate is stable. `_FIXED_OVERHEAD_S` is an
# estimate; being wrong about it by a second or two barely moves a rate measured
# over tens of seconds, but swings one measured over a few seconds several fold.
_MIN_DECODE_S = 10.0

# Fixed per-request overhead (connection, queueing, scheduling) subtracted
# before the remainder is attributed to decoding.
_FIXED_OVERHEAD_S = 2.0

# Rough per-1K-token pricing as of 2026-05 — used only for the cost gate, not
# billing. Update as needed; the router never refuses to call a model just
# because its price isn't here (defaults to 0 cost in that case).
_PRICING_PER_1K = {
    "gpt-4.1": (0.0025, 0.010),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-5.4": (0.0025, 0.015),        # $2.50 in / $15 out per 1M (cached in $0.25)
    "gpt-5.4-mini": (0.00075, 0.0045),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.00060),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    # Anthropic (per-1K tokens; $5/$25, $3/$15, $1/$5 per-1M respectively)
    "claude-opus-4-8": (0.005, 0.025),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
}


def _estimate_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_mult: float = 0.1,
    cache_write_mult: float = 1.25,
) -> float | None:
    """Rough cost. `prompt_tokens` is the FULL-PRICE (uncached) input; cache_read /
    cache_write are billed at their provider multipliers (Anthropic: 0.1x read /
    1.25x write; OpenAI: 0.5x read / 1.0x write). Callers pass the split + mults."""
    if prompt_tokens is None and completion_tokens is None and not cache_read_tokens and not cache_write_tokens:
        return None
    if model not in _PRICING_PER_1K:
        return 0.0
    in_price, out_price = _PRICING_PER_1K[model]
    return (
        ((prompt_tokens or 0) / 1000.0) * in_price
        + (cache_read_tokens / 1000.0) * in_price * cache_read_mult
        + (cache_write_tokens / 1000.0) * in_price * cache_write_mult
        + ((completion_tokens or 0) / 1000.0) * out_price
    )


# Context window per model, in tokens. Used only for the pre-flight size check
# below -- an unlisted model falls back to _DEFAULT_CONTEXT_WINDOW rather than
# blocking the call, so adding a model never requires touching this table.
#
# The check exists because an oversized prompt is a *deterministic* failure that
# was previously discovered only after the request went out: prune-expand stage 3
# sent 1,348,538 tokens at a 1,047,576-token limit, and the resulting 400 was
# caught and swallowed, so the run continued with un-deduplicated results.
_MODEL_CONTEXT_WINDOW = {
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-5.4": 400_000,
    "gpt-5.4-mini": 400_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "llama-3.3-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


class PromptTooLargeError(ValueError):
    """The rendered prompt cannot fit the model's context window.

    Raised before the HTTP call. Retrying cannot help -- the same prompt yields
    the same rejection -- so this is deliberately not a retryable error.
    """


def context_window_for(model: str) -> int:
    return _MODEL_CONTEXT_WINDOW.get(model, _DEFAULT_CONTEXT_WINDOW)


# OpenAI GPT-5.x and o-series models reject `max_tokens` and non-default
# `temperature`: they require `max_completion_tokens` and omit sampling params.
_OPENAI_COMPLETION_TOKEN_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _openai_uses_completion_tokens(model: str) -> bool:
    return model.startswith(_OPENAI_COMPLETION_TOKEN_PREFIXES)


async def _consume_openai_stream(stream: Any) -> tuple[str, Any, str | None]:
    """Drain an OpenAI chat stream into (text, usage).

    The usage-bearing chunk arrives last and carries no choices, so it must be
    read separately from the content deltas rather than assumed to be on the
    final content chunk. finish_reason arrives on its own earlier chunk, so it
    is latched rather than read from the last one.
    """
    parts: list[str] = []
    usage = None
    finish: str | None = None
    async for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                parts.append(content)
    return "".join(parts), usage, finish


class OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        # max_retries=0: the SDK defaults to 2 internal retries, which nest
        # inside tenacity's 5 for up to 15 HTTP attempts per logical call. At a
        # 180s timeout that is ~45 minutes spent on one call, invisible from
        # outside because SDK retries are silent -- an `attempt=1/5` log line
        # covered three requests. tenacity is the single retry authority.
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        response_format: str | None,
        cache_prefix: str | None = None,
        stream: bool = False,
    ) -> ChatResult:
        # Tier A -- source-first ordering. OpenAI AUTOMATICALLY caches identical
        # leading prefixes (>=1024 tokens); putting the large stable source before
        # the (varying) task instruction lets that source be reused across a
        # window's calls at 0.5x input price. No explicit cache_control needed.
        sys_content = f"{cache_prefix}\n\n===END SOURCE===\n\n{system}" if cache_prefix else system
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user},
            ],
            "timeout": timeout,
        }
        if _openai_uses_completion_tokens(model):
            # gpt-5.x / o-series: no `max_tokens`, no non-default temperature.
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            # Streaming turns the timeout from a whole-request deadline into a
            # between-chunk one, so a long-but-healthy generation cannot time
            # out. Required for the summarizer tasks, which ask for 8192 output
            # tokens: at typical throughput that alone can exceed a 180s budget,
            # making the call unwinnable and sending it straight to retry.
            #
            # include_usage is mandatory, not an optimisation: without it
            # resp.usage is None on a stream, _estimate_cost returns 0.0, and
            # both cost_report() and --max-cost-usd silently report $0.
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            text, usage, finish = await _consume_openai_stream(
                await self._client.chat.completions.create(**kwargs)
            )
        else:
            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            finish = getattr(choice, "finish_reason", None)

        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        uncached = (prompt_tokens or 0) - cached
        return ChatResult(
            finish_reason=finish,
            text=text,
            model=model,
            provider="openai",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=0,
            # OpenAI cached input is billed at 0.5x, no write premium.
            cost_usd=_estimate_cost(
                model, uncached, completion_tokens,
                cache_read_tokens=cached, cache_read_mult=0.5,
                cache_write_tokens=0, cache_write_mult=1.0,
            ),
        )


class GroqProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncGroq(api_key=api_key, max_retries=0)  # see OpenAIProvider

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        response_format: str | None,
        cache_prefix: str | None = None,
        stream: bool = False,
    ) -> ChatResult:
        if stream:
            # The groq SDK (1.2.0) exposes `stream` but has no `stream_options`,
            # so `include_usage` cannot be requested and a streamed response
            # reports no token counts -- cost tracking and --max-cost-usd would
            # silently read $0. Refuse loudly rather than quietly dropping the
            # flag; a silently ignored setting is the failure mode this whole
            # change set exists to remove.
            raise ValueError(
                "stream=true is not supported for provider 'groq': the SDK "
                "cannot request usage on a stream, which would silently zero "
                "out cost tracking. Remove `stream: true` from this task, or "
                "route it to openai/anthropic."
            )
        # Groq has no prompt caching; keep source-first ordering for correctness
        # (harmless -- no cost benefit).
        sys_content = f"{cache_prefix}\n\n===END SOURCE===\n\n{system}" if cache_prefix else system
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return ChatResult(
            finish_reason=getattr(choice, "finish_reason", None),
            text=text,
            model=model,
            provider="groq",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
        )


# Anthropic models that REJECT sampling params (temperature/top_p/top_k) with a
# 400. Current-generation Opus (4.6+), Fable, and Mythos are adaptive-thinking
# only. Sonnet/Haiku still accept temperature, so we send it for those.
_ANTHROPIC_NO_SAMPLING_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable",
    "claude-mythos",
)


def _anthropic_accepts_temperature(model: str) -> bool:
    return not model.startswith(_ANTHROPIC_NO_SAMPLING_PREFIXES)


def _strip_json_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence and trailing ``` if present.

    Anthropic has no `response_format: json_object`; we instruct the model to
    emit bare JSON but defensively strip fences so strict `json.loads` callers
    (e.g. class_proposal) don't choke. No-op when there's no fence.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) ...
        first_nl = s.find("\n")
        s = s[first_nl + 1 :] if first_nl != -1 else s[3:]
        # ... and a closing fence if present.
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0)  # see OpenAIProvider

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        response_format: str | None,
        cache_prefix: str | None = None,
        stream: bool = False,
    ) -> ChatResult:
        # Anthropic takes the system prompt as a top-level arg, not a message.
        sys_prompt = system
        if response_format == "json_object":
            sys_prompt = (
                f"{system}\n\nOutput only a single valid JSON object. "
                "Do not include any prose, explanation, or markdown code fences."
            )
        # Tier B -- explicit prompt caching. Anthropic has NO automatic cache, so
        # place the large stable source as a leading system block with
        # cache_control; the (varying) task instruction is a second, uncached
        # block AFTER the breakpoint, so the cached [source] prefix hits across a
        # window's calls (summarize / question-gen / revise) within the 5-min TTL.
        if cache_prefix:
            kwargs_system: Any = [
                {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": sys_prompt},
            ]
        else:
            kwargs_system = sys_prompt
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": kwargs_system,
            "messages": [{"role": "user", "content": user}],
            "timeout": timeout,
        }
        # Only send temperature where the model accepts it (Sonnet/Haiku).
        # Opus 4.6+/Fable/Mythos 400 on sampling params.
        if _anthropic_accepts_temperature(model):
            kwargs["temperature"] = temperature

        if stream:
            # Unlike OpenAI, Anthropic needs no include_usage equivalent: usage
            # is native to the event stream (message_start carries input_tokens,
            # message_delta carries cumulative output + cache tokens). The SDK's
            # .stream() helper reassembles those into a final Message with the
            # same shape as a non-streamed response, so everything below is
            # unchanged.
            async with self._client.messages.stream(**kwargs) as s:
                resp = await s.get_final_message()
        else:
            resp = await self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        if response_format == "json_object":
            text = _strip_json_fences(text)
        usage = resp.usage
        # Anthropic reports uncached input separately from cache read/write.
        prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        cache_write = (getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        cache_read = (getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        return ChatResult(
            # Anthropic says "max_tokens" where OpenAI says "length"; normalise
            # so callers test one value.
            finish_reason=(
                "length" if getattr(resp, "stop_reason", None) == "max_tokens"
                else getattr(resp, "stop_reason", None)
            ),
            text=text,
            model=model,
            provider="anthropic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            # Anthropic: read 0.1x, write 1.25x (the _estimate_cost defaults).
            cost_usd=_estimate_cost(
                model, prompt_tokens, completion_tokens,
                cache_read_tokens=cache_read, cache_write_tokens=cache_write,
            ),
        )


def _build_providers(settings: Settings) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(api_key=settings.openai_api_key)
    if settings.groq_api_key:
        providers["groq"] = GroqProvider(api_key=settings.groq_api_key)
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)
    return providers


class LLMRouter:
    """Single entry point for all chat calls. Caller supplies a task name
    (one of the keys in config/models.yaml::tasks); the router picks the
    right provider + model and applies retry/backoff."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._providers = _build_providers(self._settings)
        mc = self._settings.models_config
        self._tasks = mc.get("tasks", {})
        self._retry = mc.get("defaults", {}).get("retries", {})
        self.reset_counters()
        self._validate_stream_support()

    def reset_counters(self) -> None:
        """Initialise every per-run accumulator.

        Kept separate from __init__ because tests build bare routers via
        `LLMRouter.__new__(LLMRouter)` to avoid needing real config, and then
        hand-set the fields `chat()` touches. Every new counter used to break
        those tests with an AttributeError from deep in the hot path; now they
        call this instead and new counters cost nothing.
        """
        self._total_cost_usd = 0.0
        # Prompt-cache accounting across all calls (see cache_hit_rate).
        self._cache_read_tokens = 0     # input served from cache
        self._cache_write_tokens = 0    # input written to cache
        self._input_full_tokens = 0     # uncached, full-price input
        # Per-task spend, so a run can report WHERE the money went, not just a
        # total. Only ~$0.64 of a ~$10 prune-expand was previously recoverable
        # after the fact (a couple of tasks self-report into llm_audit.jsonl);
        # everything else vanished with the terminal scrollback.
        self._cost_by_task: dict[str, float] = {}
        self._calls_by_task: dict[str, int] = {}
        # Observed completion length per task. `max_tokens` and `timeout` are
        # both sized against the LARGEST reply a task may produce, and until now
        # that number was assumed rather than known -- which is how ~25 tasks
        # per preset ended up on a timeout that assumed a decode rate nobody had
        # measured. Recording it makes the next sizing pass evidence-based.
        self._out_tokens_by_task: dict[str, list[int]] = {}

    def _validate_stream_support(self) -> None:
        """Fail at config load, not mid-run, on an unsupportable `stream: true`.

        Only groq is affected (its SDK cannot request usage on a stream). This
        runs over the whole task map so switching presets surfaces the problem
        immediately rather than hours into a paid run.
        """
        bad = [
            name
            for name, spec in self._tasks.items()
            if spec.get("stream") and spec.get("provider") == "groq"
        ]
        if bad:
            raise ValueError(
                f"tasks {sorted(bad)} set `stream: true` with provider 'groq', "
                "whose SDK cannot report token usage on a stream -- cost "
                "tracking and --max-cost-usd would silently read $0. Route "
                "these tasks to openai/anthropic or drop `stream`."
            )

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def cost_by_task(self) -> dict[str, float]:
        """Spend per task name, descending. Empty until calls are made."""
        return dict(sorted(self._cost_by_task.items(), key=lambda kv: -kv[1]))

    @property
    def calls_by_task(self) -> dict[str, int]:
        """Successful call count per task name."""
        return dict(self._calls_by_task)

    def cost_report(self) -> dict[str, Any]:
        """Serializable spend summary for persisting alongside a run's outputs."""
        by_task = self.cost_by_task
        return {
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_calls": sum(self._calls_by_task.values()),
            "prompt_cache": {
                "cache_read_tokens": self._cache_read_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "full_price_input_tokens": self._input_full_tokens,
                "total_input_tokens": self.total_input_tokens,
                "cache_hit_rate": round(self.cache_hit_rate, 4),
            },
            "by_task": {
                t: {
                    "cost_usd": round(c, 6),
                    "calls": self._calls_by_task.get(t, 0),
                    "model": self._tasks.get(t, {}).get("model"),
                    "provider": self._tasks.get(t, {}).get("provider"),
                    **self._output_stats(t),
                }
                for t, c in by_task.items()
            },
        }

    def _output_stats(self, task: str) -> dict[str, Any]:
        """Observed vs configured output size for one task.

        `headroom` is configured max_tokens divided by the largest reply seen.
        A large value means the task is budgeted for output it never produces --
        which inflates the decode rate its timeout implicitly demands, for no
        benefit. A value near 1 means replies are brushing the cap and may be
        getting truncated.
        """
        seen = self._out_tokens_by_task.get(task)
        if not seen:
            return {}
        ordered = sorted(seen)
        configured = int(self._tasks.get(task, {}).get("max_tokens") or 0)
        stats: dict[str, Any] = {
            "out_tokens_max": ordered[-1],
            "out_tokens_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "out_tokens_mean": round(sum(ordered) / len(ordered), 1),
            "max_tokens_configured": configured or None,
        }
        if configured and ordered[-1]:
            stats["headroom"] = round(configured / ordered[-1], 1)
        return stats

    @property
    def cache_read_tokens(self) -> int:
        return self._cache_read_tokens

    @property
    def cache_write_tokens(self) -> int:
        return self._cache_write_tokens

    @property
    def total_input_tokens(self) -> int:
        """All input tokens processed = uncached + cache-read + cache-write."""
        return self._input_full_tokens + self._cache_read_tokens + self._cache_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from the prompt cache (0.0-1.0)."""
        tot = self.total_input_tokens
        return (self._cache_read_tokens / tot) if tot else 0.0

    def task_spec(self, task: str) -> dict[str, Any]:
        if task not in self._tasks:
            raise KeyError(f"Unknown task: {task}. Available: {sorted(self._tasks)}")
        return self._tasks[task]

    async def chat(
        self, task: str, *, system: str, user: str, cache_prefix: str | None = None
    ) -> ChatResult:
        spec = self.task_spec(task)
        provider_name = spec["provider"]
        if provider_name not in self._providers:
            raise RuntimeError(
                f"Provider '{provider_name}' for task '{task}' not configured. "
                f"Check that {provider_name.upper()}_API_KEY is set."
            )
        provider = self._providers[provider_name]
        model = spec["model"]
        max_tokens = int(spec.get("max_tokens", 4096))
        timeout = int(spec.get("timeout", 120))
        retryer = self._make_retryer(task, provider_name, model)

        self._preflight_size(
            task, model, system=system, user=user,
            cache_prefix=cache_prefix, max_tokens=max_tokens,
        )

        async for attempt in retryer:
            with attempt:
                started = time.monotonic()
                result = await provider.chat(
                    model=model,
                    system=system,
                    user=user,
                    temperature=float(spec.get("temperature", 0.0)),
                    max_tokens=max_tokens,
                    timeout=timeout,
                    response_format=spec.get("response_format"),
                    cache_prefix=cache_prefix,
                    stream=bool(spec.get("stream", False)),
                )
                elapsed = time.monotonic() - started
                # A call sitting near its deadline is the signal that precedes a
                # timeout-and-retry spiral. Without it, a healthy-but-slow run is
                # indistinguishable from a deadlocked one from the outside --
                # which is exactly how a working run got killed after an hour.
                self._warn_if_near_deadline(
                    task=task, provider_name=provider_name, model=model,
                    elapsed=elapsed, timeout=timeout, max_tokens=max_tokens,
                    result=result,
                )
                if result.cost_usd:
                    self._total_cost_usd += result.cost_usd
                    self._cost_by_task[task] = self._cost_by_task.get(task, 0.0) + result.cost_usd
                self._calls_by_task[task] = self._calls_by_task.get(task, 0) + 1
                if result.completion_tokens:
                    self._out_tokens_by_task.setdefault(task, []).append(
                        result.completion_tokens
                    )
                # Prompt-cache accounting. OpenAI reports prompt_tokens INCLUDING
                # the cached portion; Anthropic reports input_tokens EXCLUDING it.
                read = result.cache_read_tokens or 0
                write = result.cache_write_tokens or 0
                prompt = result.prompt_tokens or 0
                full = (prompt - read) if result.provider == "openai" else prompt
                self._cache_read_tokens += read
                self._cache_write_tokens += write
                self._input_full_tokens += max(0, full)
                return result
        raise RuntimeError(f"LLM call for task '{task}' exhausted retries")

    def _warn_if_near_deadline(
        self,
        *,
        task: str,
        provider_name: str,
        model: str,
        elapsed: float,
        timeout: int,
        max_tokens: int,
        result: ChatResult,
    ) -> None:
        """Warn when a call is at genuine risk of timing out on a later attempt.

        The first version of this compared `elapsed > timeout * 0.5` and nothing
        else. That is output-blind, and it produced BOTH kinds of error:

        * False alarms on input-heavy tasks. `summary_evaluate` reached 59% of
          its budget while emitting only a few hundred tokens -- the time went
          into prefilling a large prompt, not into slow decoding. Nothing was
          wrong, but it looked identical to the pathology that cost 45 minutes.
        * Silence on the case that actually matters. A task that returns a SHORT
          reply quickly can still be one long reply away from its deadline. At
          25% of a 60s budget for 200 tokens, a full 8,192-token reply would
          need ~20 minutes -- the flat rule says nothing.

        So split elapsed into prefill and decode, measure the decode rate that
        was actually achieved, and project what a MAXIMUM-length reply would
        have cost at that rate. Warn when the projection exceeds the timeout --
        i.e. when this task, on this model, right now, cannot reliably produce
        the output it is configured to be allowed to produce.

        Prefill is estimated rather than measured (no provider reports the
        split). The constant is deliberately conservative: overestimating
        prefill throughput attributes more time to decode, which makes the
        projection larger and the warning more eager. Erring toward a spurious
        warning is the right trade -- the failure this guards against is silent.
        """
        if not timeout:
            return

        prompt_tokens = result.prompt_tokens or 0
        completion_tokens = result.completion_tokens or 0

        # No usage reported (some providers/paths omit it) -- fall back to the
        # old flat rule rather than skipping the check entirely.
        if not completion_tokens:
            if elapsed > timeout * 0.5:
                log.warning(
                    "slow call task=%s provider=%s model=%s took %.0fs "
                    "(%.0f%% of its %ds timeout; no usage reported)",
                    task, provider_name, model, elapsed,
                    100.0 * elapsed / timeout, timeout,
                )
            return

        prefill_s = min(prompt_tokens / _PREFILL_TOKENS_PER_S, elapsed * 0.9)
        overhead_s = min(_FIXED_OVERHEAD_S, max(elapsed - prefill_s - 1e-3, 0.0))
        decode_s = max(elapsed - prefill_s - overhead_s, 1e-3)

        # Is this sample big enough to extrapolate from? Two independent ways it
        # can fail, and both were hit in practice:
        #
        #  * Too few tokens. A short reply is mostly TIME-TO-FIRST-TOKEN --
        #    connection, queueing, scheduling -- a FIXED cost, not a per-token
        #    one. Live: table_concept_grouping returned 67 tokens in 7s and was
        #    reported as "10 tok/s, needs 200s" for a task needing 13.7 tok/s
        #    that clears it several times over.
        #  * Too little decode time. `_FIXED_OVERHEAD_S` is an estimate, not a
        #    measurement. When decoding only spans a few seconds, being wrong
        #    about it by a couple of seconds swings the computed rate several
        #    fold; over tens of seconds the same error is a few percent.
        #
        # Skipping both costs nothing. A task whose replies stay short is not at
        # risk of a LENGTH-driven timeout, and the first reply long enough to
        # matter is also long enough to measure -- the check self-activates
        # exactly when output size becomes the thing worth checking.
        if completion_tokens < _MIN_TOKENS_TO_RATE or decode_s < _MIN_DECODE_S:
            return

        decode_rate = completion_tokens / decode_s
        projected = prefill_s + overhead_s + (max_tokens / decode_rate)

        if projected <= timeout:
            return

        log.warning(
            "task=%s provider=%s model=%s is under-budgeted: took %.0fs for "
            "%d in / %d out (%.0f tok/s decode). A full %d-token reply would "
            "need ~%.0fs, over the %ds timeout -- raise timeout to >=%d or "
            "lower max_tokens.",
            task, provider_name, model, elapsed,
            prompt_tokens, completion_tokens, decode_rate,
            max_tokens, projected, timeout, int(projected * 1.2) + 1,
        )

    def _preflight_size(
        self,
        task: str,
        model: str,
        *,
        system: str,
        user: str,
        cache_prefix: str | None,
        max_tokens: int,
    ) -> None:
        """Reject a prompt that cannot fit, before spending a request on it.

        Sizing is the caller's job -- batch the payload -- but when a caller
        gets it wrong this turns a provider 400 (which the dedup stage caught
        and swallowed, silently emitting un-deduplicated results) into a typed,
        non-retryable error naming the task and the overage.
        """
        window = context_window_for(model)
        budget = window - max_tokens
        if budget <= 0:  # max_tokens misconfigured above the window; let the API judge
            return
        prompt_tokens = count_tokens(system) + count_tokens(user)
        if cache_prefix:
            prompt_tokens += count_tokens(cache_prefix)
        if prompt_tokens > budget:
            raise PromptTooLargeError(
                f"task '{task}' prompt is {prompt_tokens:,} tokens but {model} "
                f"allows {budget:,} (context {window:,} minus max_tokens "
                f"{max_tokens:,}). Batch the payload before calling."
            )

    def _make_retryer(
        self, task: str = "", provider: str = "", model: str = ""
    ) -> tenacity.AsyncRetrying:
        max_attempts = int(self._retry.get("max_attempts", 5))

        def _log_retry(retry_state: tenacity.RetryCallState) -> None:
            # Surface otherwise-silent transient failures + backoff — especially
            # provider 429 rate limits (e.g. Anthropic RateLimitError) that made a
            # run look "stuck" with no visibility. `before_sleep` fires only when a
            # retry is scheduled. WARNING level so it shows without explicit logging
            # config (Python's last-resort handler prints WARNING+ to stderr).
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            sleep_s = getattr(retry_state.next_action, "sleep", 0.0)
            log.warning(
                "retry task=%s provider=%s model=%s attempt=%d/%d err=%s wait=%.1fs",
                task, provider, model, retry_state.attempt_number, max_attempts,
                (f"{type(exc).__name__}: {exc}"[:160] if exc else "?"), sleep_s,
            )

        return tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=tenacity.wait_exponential(
                multiplier=float(self._retry.get("initial_wait_seconds", 1)),
                max=float(self._retry.get("max_wait_seconds", 30)),
            ),
            # Retrying a deterministic failure cannot succeed, it only
            # multiplies the time spent discovering that: a 400
            # context_length_exceeded was observed burning all five attempts.
            # PromptTooLargeError is ours and is equally deterministic.
            retry=tenacity.retry_if_exception(
                lambda exc: not isinstance(exc, PromptTooLargeError)
                and retry_policy.is_retryable(exc)
            ),
            reraise=True,
            before_sleep=_log_retry,
        )

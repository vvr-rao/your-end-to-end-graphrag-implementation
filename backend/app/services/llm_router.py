"""Provider-abstracted LLM dispatcher.

Reads task -> provider/model from `config/models.yaml`, dispatches each task
to the right SDK (OpenAI or Groq today), applies retry/backoff via tenacity,
and tracks rough cost. Designed so a future provider is one new class plus
one line in PROVIDERS — no service-code changes elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import tenacity
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from openai import AsyncOpenAI

from backend.app.core.config import Settings, get_settings

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
    ) -> ChatResult: ...


# Rough per-1K-token pricing as of 2026-05 — used only for the cost gate, not
# billing. Update as needed; the router never refuses to call a model just
# because its price isn't here (defaults to 0 cost in that case).
_PRICING_PER_1K = {
    "gpt-4.1": (0.0025, 0.010),
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.00060),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    # Anthropic (per-1K tokens; $5/$25, $3/$15, $1/$5 per-1M respectively)
    "claude-opus-4-8": (0.005, 0.025),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
}


def _estimate_cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None and completion_tokens is None:
        return None
    if model not in _PRICING_PER_1K:
        return 0.0
    in_price, out_price = _PRICING_PER_1K[model]
    return ((prompt_tokens or 0) / 1000.0) * in_price + ((completion_tokens or 0) / 1000.0) * out_price


class OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

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
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
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
            text=text,
            model=model,
            provider="openai",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
        )


class GroqProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncGroq(api_key=api_key)

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
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
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
        self._client = AsyncAnthropic(api_key=api_key)

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
    ) -> ChatResult:
        # Anthropic takes the system prompt as a top-level arg, not a message.
        sys_prompt = system
        if response_format == "json_object":
            sys_prompt = (
                f"{system}\n\nOutput only a single valid JSON object. "
                "Do not include any prose, explanation, or markdown code fences."
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user}],
            "timeout": timeout,
        }
        # Only send temperature where the model accepts it (Sonnet/Haiku).
        # Opus 4.6+/Fable/Mythos 400 on sampling params.
        if _anthropic_accepts_temperature(model):
            kwargs["temperature"] = temperature

        resp = await self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        if response_format == "json_object":
            text = _strip_json_fences(text)
        usage = resp.usage
        prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        return ChatResult(
            text=text,
            model=model,
            provider="anthropic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
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
        self._total_cost_usd = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    def task_spec(self, task: str) -> dict[str, Any]:
        if task not in self._tasks:
            raise KeyError(f"Unknown task: {task}. Available: {sorted(self._tasks)}")
        return self._tasks[task]

    async def chat(self, task: str, *, system: str, user: str) -> ChatResult:
        spec = self.task_spec(task)
        provider_name = spec["provider"]
        if provider_name not in self._providers:
            raise RuntimeError(
                f"Provider '{provider_name}' for task '{task}' not configured. "
                f"Check that {provider_name.upper()}_API_KEY is set."
            )
        provider = self._providers[provider_name]
        retryer = self._make_retryer()

        async for attempt in retryer:
            with attempt:
                result = await provider.chat(
                    model=spec["model"],
                    system=system,
                    user=user,
                    temperature=float(spec.get("temperature", 0.0)),
                    max_tokens=int(spec.get("max_tokens", 4096)),
                    timeout=int(spec.get("timeout", 120)),
                    response_format=spec.get("response_format"),
                )
                if result.cost_usd:
                    self._total_cost_usd += result.cost_usd
                return result
        raise RuntimeError(f"LLM call for task '{task}' exhausted retries")

    def _make_retryer(self) -> tenacity.AsyncRetrying:
        return tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(int(self._retry.get("max_attempts", 5))),
            wait=tenacity.wait_exponential(
                multiplier=float(self._retry.get("initial_wait_seconds", 1)),
                max=float(self._retry.get("max_wait_seconds", 30)),
            ),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
        )

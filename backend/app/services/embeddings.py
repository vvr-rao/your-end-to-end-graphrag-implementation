"""Embeddings service (OpenAI, or DeepInfra's OpenAI-compatible endpoint).

Wraps the async OpenAI client with batching, retry, and cost tracking.
Defaults to text-embedding-3-small @ 1024 dim per the Phase 2 plan
(33% storage saving vs the 1536 default; negligible retrieval-quality
cost per OpenAI's benchmarks). The `embeddings:` entry in models.yaml
selects provider/model/dim; when absent, those defaults apply unchanged.

Usage:
    embedder = Embedder()
    vectors = await embedder.embed(["text one", "text two", ...])
    # each vector is a list[float] of length 1024.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from backend.app.core.config import Settings, get_settings
from backend.app.services import retry_policy
from backend.app.services.token_budget import plan_batches

# OpenAI's embedding endpoint accepts up to 2048 inputs per request,
# but they recommend <= 100 for practical concurrency. 100 keeps the
# request body small enough to retry cheaply on 429.
_DEFAULT_BATCH_SIZE = 100

# OpenAI caps a single embeddings request at 300,000 tokens across ALL
# inputs. The item count alone cannot enforce that -- 100 inputs at the
# 8,000-token per-item cap is 800,000 -- so batches are sized by measured
# tokens and the count above is only a secondary ceiling. 250,000 leaves
# headroom for disagreement between tiktoken and OpenAI's server-side count.
_EMBED_MAX_REQUEST_TOKENS = 250_000

# text-embedding-3-small: $0.020 per 1M input tokens.
_EMBED3_SMALL_USD_PER_M_TOK = 0.020

# USD per 1M input tokens. An unlisted model falls back to the 3-small rate
# (the historical behaviour) rather than reporting $0.
_EMBED_USD_PER_M_TOK = {
    "text-embedding-3-small": _EMBED3_SMALL_USD_PER_M_TOK,
    "Qwen/Qwen3-Embedding-8B": 0.010,
    "Qwen/Qwen3-Embedding-4B": 0.020,
    "Qwen/Qwen3-Embedding-0.6B": 0.010,
}

_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIM = 1024

# Qwen3-Embedding is asymmetric: queries carry a task instruction, documents
# do not. Omitting it costs a few points of retrieval quality per the model
# card. OpenAI embeddings are symmetric, so their query prefix is empty.
_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\n"
    "Query: "
)


def configured_embedding() -> dict:
    """The active `embeddings:` spec from models.yaml, with defaults filled.

    Single source of truth for provider/model/dim -- the Embedder and the
    DB embedding-model guard both read it, so they cannot disagree.
    """
    try:
        tasks = get_settings().models_config.get("tasks", {}) or {}
    except Exception:  # no .env / config (bare unit tests)
        tasks = {}
    spec = tasks.get("embeddings") or {}
    out = {
        "provider": spec.get("provider", _DEFAULT_PROVIDER),
        "model": spec.get("model", _DEFAULT_MODEL),
        "dim": int(spec.get("dim", _DEFAULT_DIM)),
        "batch_size": int(spec.get("batch_size", _DEFAULT_BATCH_SIZE)),
    }
    # Optional override of the query-side instruction (asymmetric models);
    # "" disables it. Absent -> the per-model default in `default_query_prefix`.
    if "query_prefix" in spec:
        out["query_prefix"] = str(spec["query_prefix"] or "")
    return out


def default_query_prefix(model: str) -> str:
    return _QWEN_QUERY_INSTRUCTION if model.startswith("Qwen/Qwen3-Embedding") else ""


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec

# text-embedding-3-small + -3-large + ada-002 all share an 8192-token
# input cap. We truncate at 8000 to leave headroom for any encoder
# disagreement vs OpenAI's server-side count. Chunks are bounded to
# 800 tokens by the chunker, but hierarchically-summarized document
# texts can exceed this. Truncation is a no-op on already-short inputs.
_EMBED_MAX_INPUT_TOKENS = 8000
_EMBED_ENCODER = None  # lazy-init


def _truncate_to_token_limit(texts: list[str]) -> tuple[list[str], int, list[int]]:
    """Truncate each input to <= 8000 tokens for the embedding model.

    Returns (truncated_texts, num_truncated, token_counts). The per-text
    counts are returned rather than discarded because request batching needs
    them immediately afterwards, and re-encoding every input a second time
    would double the cost of the most expensive synchronous step in `embed`.

    No-op if tiktoken is unavailable -- the caller will then surface the
    OpenAI 400. Counts are empty in that case, and batching falls back to the
    item-count ceiling alone.
    """
    global _EMBED_ENCODER
    try:
        import tiktoken
    except ImportError:
        return texts, 0, []
    if _EMBED_ENCODER is None:
        try:
            _EMBED_ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
        except KeyError:
            _EMBED_ENCODER = tiktoken.get_encoding("cl100k_base")

    out: list[str] = []
    counts: list[int] = []
    truncated = 0
    for t in texts:
        toks = _EMBED_ENCODER.encode(t, disallowed_special=())
        if len(toks) <= _EMBED_MAX_INPUT_TOKENS:
            out.append(t)
            counts.append(len(toks))
            continue
        out.append(_EMBED_ENCODER.decode(toks[:_EMBED_MAX_INPUT_TOKENS]))
        counts.append(_EMBED_MAX_INPUT_TOKENS)
        truncated += 1
    return out, truncated, counts


class EmbeddingBatchError(RuntimeError):
    """One or more embedding batches failed.

    `embed` raises rather than returning holes: a caller that wrote `None` into
    an embedding column would persist silently unsearchable rows, which is
    worse than a loud failure. The message names every failed input range so an
    operator can see how much of the call succeeded before it stopped.
    """

    def __init__(self, message: str, failed_ranges: list[tuple[int, int]]) -> None:
        super().__init__(message)
        self.failed_ranges = failed_ranges


@dataclass
class EmbedResult:
    """One embedding-API response (1+ vectors)."""

    vectors: list[list[float]]
    prompt_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Embedder:
    """Embeddings wrapper with batching + cost tracking.

    provider/model/dim default to the models.yaml `embeddings:` entry (see
    `configured_embedding`); explicit arguments win. Stateful: tracks
    cumulative cost across all calls so the caller can attribute spend per
    ingestion run.
    """

    # Defaults are resolved in field factories (not __post_init__) so a test
    # that neutralises __post_init__ still gets a usable model/dim.
    model: str = field(default_factory=lambda: configured_embedding()["model"])
    dim: int = field(default_factory=lambda: configured_embedding()["dim"])
    batch_size: int = field(default_factory=lambda: configured_embedding()["batch_size"])
    concurrency: int = 4
    provider: str = field(default_factory=lambda: configured_embedding()["provider"])
    settings: Settings | None = field(default=None, repr=False)
    # Query-side instruction. None -> models.yaml `embeddings.query_prefix`
    # when that spec is for this model, else the per-model default.
    query_prefix_override: str | None = None
    total_cost_usd: float = field(default=0.0, init=False)
    total_tokens: int = field(default=0, init=False)
    _client: AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = self.settings or get_settings()
        if self.provider == "deepinfra":
            from backend.app.services.llm_router import DEEPINFRA_BASE_URL

            key = getattr(s, "deepinfra_api_key", None)
            if not key:
                raise RuntimeError(
                    "embeddings are routed to provider 'deepinfra' but "
                    "DEEPINFRA_API_KEY is not set"
                )
            self._client = AsyncOpenAI(api_key=key, base_url=DEEPINFRA_BASE_URL)
        elif self.provider == "openai":
            self._client = AsyncOpenAI(api_key=s.openai_api_key)
        else:
            raise ValueError(f"unsupported embeddings provider: {self.provider!r}")

    @property
    def query_prefix(self) -> str:
        if self.query_prefix_override is not None:
            return self.query_prefix_override
        spec = configured_embedding()
        if "query_prefix" in spec and spec["model"] == self.model:
            return spec["query_prefix"]
        return default_query_prefix(self.model)

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed retrieval QUERIES (as opposed to stored passages).

        Identical to `embed` for symmetric models (OpenAI); prepends the
        query instruction for asymmetric ones (Qwen3-Embedding).
        """
        prefix = self.query_prefix
        return await self.embed([prefix + t for t in texts] if prefix else texts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all `texts` in batches. Preserves input order.

        Empty list returns an empty list. Empty strings within the list
        are replaced with a single space (OpenAI rejects '')."""
        if not texts:
            return []
        cleaned = [t if t.strip() else " " for t in texts]
        cleaned, truncated, counts = _truncate_to_token_limit(cleaned)
        if truncated:
            print(
                f"[embed] {truncated}/{len(cleaned)} input(s) exceeded "
                f"the {_EMBED_MAX_INPUT_TOKENS}-token cap; truncated"
            )

        # Size requests by measured tokens, not item count, so a request is
        # never built over the provider's per-request limit in the first place.
        # `plan_batches` returns index ranges rather than sublists: the true
        # start offset is needed below, and deriving it as `idx * batch_size`
        # only works while every batch is the same length.
        if counts:
            ranges = plan_batches(
                counts,
                max_tokens=_EMBED_MAX_REQUEST_TOKENS,
                max_items=self.batch_size,
            )
        else:  # no tiktoken: fall back to the old item-count batching
            ranges = [
                (i, min(i + self.batch_size, len(cleaned)))
                for i in range(0, len(cleaned), self.batch_size)
            ]

        results: list[list[float]] = [None] * len(cleaned)  # type: ignore[list-item]
        sem = asyncio.Semaphore(self.concurrency)
        failures: list[tuple[int, int, Exception]] = []

        async def _one(start: int, end: int) -> None:
            async with sem:
                resp = await self._call_with_retry(cleaned[start:end])
            for i, item in enumerate(resp.vectors):
                results[start + i] = item

        # return_exceptions so one bad batch does not abandon its siblings
        # mid-flight; every batch reaches an outcome and the report below names
        # exactly which ranges failed instead of surfacing one opaque error.
        outcomes = await asyncio.gather(
            *[_one(s, e) for s, e in ranges], return_exceptions=True
        )
        for (start, end), outcome in zip(ranges, outcomes):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome  # CancelledError et al. must propagate
                failures.append((start, end, outcome))

        if failures:
            ok = len(ranges) - len(failures)
            detail = ", ".join(
                f"[{s}:{e}] {retry_policy.describe(exc)}" for s, e, exc in failures
            )
            raise EmbeddingBatchError(
                f"{len(failures)}/{len(ranges)} embedding batch(es) failed "
                f"({ok} succeeded, {len(cleaned)} input(s) total): {detail}",
                failed_ranges=[(s, e) for s, e, _ in failures],
            ) from failures[0][2]
        return results

    async def _call_with_retry(
        self, batch: list[str], max_attempts: int = 4
    ) -> EmbedResult:
        """One embeddings call with exponential-backoff retry on 429/5xx."""
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dim,
                )
                vectors = [d.embedding for d in resp.data]
                if self.provider != "openai":
                    # The L2<->cosine identities in retrieval_sql assume unit
                    # vectors; OpenAI guarantees it, Matryoshka truncation
                    # does not.
                    vectors = [_l2_normalize(v) for v in vectors]
                prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
                rate = _EMBED_USD_PER_M_TOK.get(self.model, _EMBED3_SMALL_USD_PER_M_TOK)
                cost = prompt_tokens / 1_000_000.0 * rate
                self.total_tokens += prompt_tokens
                self.total_cost_usd += cost
                return EmbedResult(
                    vectors=vectors,
                    prompt_tokens=prompt_tokens,
                    cost_usd=cost,
                )
            except Exception as exc:  # OpenAI SDK raises specific subclasses
                last_exc = exc
                # A deterministic 400 (oversized request, malformed input) or a
                # 401 cannot succeed on retry -- retrying only multiplies the
                # time spent discovering that, three more times per batch.
                if not retry_policy.is_retryable(exc):
                    raise
                if attempt == max_attempts - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

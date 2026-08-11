"""OpenAI embeddings service.

Wraps the async OpenAI client with batching, retry, and cost tracking.
Defaults to text-embedding-3-small @ 1024 dim per the Phase 2 plan
(33% storage saving vs the 1536 default; negligible retrieval-quality
cost per OpenAI's benchmarks).

Usage:
    embedder = Embedder()
    vectors = await embedder.embed(["text one", "text two", ...])
    # each vector is a list[float] of length 1024.
"""
from __future__ import annotations

import asyncio
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
    """OpenAI embeddings wrapper with batching + cost tracking.

    Reads OPENAI_API_KEY from app settings (validated at startup).
    Stateful: tracks cumulative cost across all calls so the caller
    can attribute spend per ingestion run.
    """

    model: str = "text-embedding-3-small"
    dim: int = 1024
    batch_size: int = _DEFAULT_BATCH_SIZE
    concurrency: int = 4
    settings: Settings | None = field(default=None, repr=False)
    total_cost_usd: float = field(default=0.0, init=False)
    total_tokens: int = field(default=0, init=False)
    _client: AsyncOpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = self.settings or get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key)

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
                prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
                cost = prompt_tokens / 1_000_000.0 * _EMBED3_SMALL_USD_PER_M_TOK
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

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

# OpenAI's embedding endpoint accepts up to 2048 inputs per request,
# but they recommend <= 100 for practical concurrency. 100 keeps the
# request body small enough to retry cheaply on 429.
_DEFAULT_BATCH_SIZE = 100

# text-embedding-3-small: $0.020 per 1M input tokens.
_EMBED3_SMALL_USD_PER_M_TOK = 0.020

# text-embedding-3-small + -3-large + ada-002 all share an 8192-token
# input cap. We truncate at 8000 to leave headroom for any encoder
# disagreement vs OpenAI's server-side count. Chunks are bounded to
# 800 tokens by the chunker, but hierarchically-summarized document
# texts can exceed this. Truncation is a no-op on already-short inputs.
_EMBED_MAX_INPUT_TOKENS = 8000
_EMBED_ENCODER = None  # lazy-init


def _truncate_to_token_limit(texts: list[str]) -> tuple[list[str], int]:
    """Truncate each input to <= 8000 tokens for the embedding model.
    Returns (truncated_texts, num_truncated). No-op if tiktoken is
    unavailable -- the caller will then surface the OpenAI 400."""
    global _EMBED_ENCODER
    try:
        import tiktoken
    except ImportError:
        return texts, 0
    if _EMBED_ENCODER is None:
        try:
            _EMBED_ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
        except KeyError:
            _EMBED_ENCODER = tiktoken.get_encoding("cl100k_base")

    out: list[str] = []
    truncated = 0
    for t in texts:
        toks = _EMBED_ENCODER.encode(t, disallowed_special=())
        if len(toks) <= _EMBED_MAX_INPUT_TOKENS:
            out.append(t)
            continue
        out.append(_EMBED_ENCODER.decode(toks[:_EMBED_MAX_INPUT_TOKENS]))
        truncated += 1
    return out, truncated


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
        cleaned, truncated = _truncate_to_token_limit(cleaned)
        if truncated:
            print(
                f"[embed] {truncated}/{len(cleaned)} input(s) exceeded "
                f"the {_EMBED_MAX_INPUT_TOKENS}-token cap; truncated"
            )

        batches = [
            cleaned[i : i + self.batch_size]
            for i in range(0, len(cleaned), self.batch_size)
        ]
        results: list[list[float]] = [None] * len(cleaned)  # type: ignore[list-item]
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(batch_idx: int, batch: list[str]) -> None:
            async with sem:
                resp = await self._call_with_retry(batch)
            offset = batch_idx * self.batch_size
            for i, item in enumerate(resp.vectors):
                results[offset + i] = item

        await asyncio.gather(
            *[_one(i, b) for i, b in enumerate(batches)]
        )
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
                if attempt == max_attempts - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

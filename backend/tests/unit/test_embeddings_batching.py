"""Embedding requests are sized by tokens, and results keep input order.

Regression cover for the two failures in the collection-of-pharma-docs-3 run:
  register-documents:  Requested 330224 tokens, max 300000 tokens per request
  generate-artifacts:  Requested 326428 tokens, max 300000 tokens per request
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import openai
import pytest

from backend.app.services.embeddings import (
    _EMBED_MAX_REQUEST_TOKENS,
    EmbeddingBatchError,
    Embedder,
)
from backend.app.services.token_budget import count_tokens


@dataclass
class _FakeUsage:
    prompt_tokens: int


@dataclass
class _FakeDatum:
    embedding: list[float]


class _FakeEmbeddings:
    """Records every request so the test can assert on request shape."""

    def __init__(self, fail_on=None) -> None:
        self.requests: list[list[str]] = []
        self._fail_on = fail_on or (lambda batch: None)

    async def create(self, *, model, input, dimensions):  # noqa: A002
        self.requests.append(list(input))
        exc = self._fail_on(input)
        if exc is not None:
            raise exc
        # Echo each input's leading "<id>|" marker back as the vector's first
        # component, so a misplaced write-back is visible as reordering.
        def _ident(t: str) -> float:
            head = t.split("|", 1)[0]
            return float(head) if head.isdigit() else float(len(t))

        return type(
            "Resp",
            (),
            {
                "data": [_FakeDatum([_ident(t), float(i)]) for i, t in enumerate(input)],
                "usage": _FakeUsage(prompt_tokens=sum(len(t) for t in input)),
            },
        )()


class _FakeClient:
    def __init__(self, fail_on=None) -> None:
        self.embeddings = _FakeEmbeddings(fail_on)


class _FakeSettings:
    openai_api_key = "test-key-not-used"


def _embedder(fail_on=None) -> tuple[Embedder, _FakeClient]:
    client = _FakeClient(fail_on)
    e = Embedder(settings=_FakeSettings())  # type: ignore[arg-type]
    e._client = client  # type: ignore[assignment]
    return e, client


def test_no_request_exceeds_the_provider_token_cap() -> None:
    # 60 inputs of ~8000 tokens each = ~480k tokens. Under the old
    # item-count batching all 60 went in one request and OpenAI 400'd.
    big = "word " * 8000
    e, client = _embedder()
    asyncio.run(e.embed([big] * 60))

    assert len(client.embeddings.requests) > 1, "expected the budget to split"
    for req in client.embeddings.requests:
        total = sum(count_tokens(t) for t in req)
        assert total <= _EMBED_MAX_REQUEST_TOKENS


def test_output_order_matches_input_order_for_variable_length_batches() -> None:
    # Deliberately uneven sizes so batches vary in length. The old code derived
    # the write-back offset as `batch_idx * batch_size`, which silently placed
    # vectors on the wrong rows as soon as batches stopped being uniform.
    # 99 items stays under the 100-item ceiling, so the split below can only be
    # token-driven: 33 big items x ~8000 tokens = ~264k, over the 250k budget.
    # The row id leads each string so it survives the 8000-token truncation.
    texts = [
        f"{i}| " + "word " * (9000 if i % 3 == 0 else 5) for i in range(99)
    ]
    e, client = _embedder()
    vectors = asyncio.run(e.embed(texts))

    assert len(vectors) == len(texts)
    assert all(v is not None for v in vectors), "every row must be filled"
    # the fake echoes each input's leading id back as the vector's first
    # component, so a misplaced write-back shows up as an out-of-order list
    assert [int(v[0]) for v in vectors] == list(range(99))
    assert len(client.embeddings.requests) > 1
    assert min(len(r) for r in client.embeddings.requests) != max(
        len(r) for r in client.embeddings.requests
    ), "batches should vary in length -- that is the case the offset bug hit"


def test_item_count_ceiling_still_applies_to_small_inputs() -> None:
    e, client = _embedder()
    asyncio.run(e.embed(["hi"] * 250))
    assert all(len(r) <= e.batch_size for r in client.embeddings.requests)


def test_permanent_400_is_not_retried() -> None:
    calls = {"n": 0}

    def fail(batch):
        calls["n"] += 1
        return openai.BadRequestError(
            "context_length_exceeded",
            response=_resp(400),
            body=None,
        )

    e, _ = _embedder(fail)
    with pytest.raises(EmbeddingBatchError):
        asyncio.run(e.embed(["a", "b"]))
    assert calls["n"] == 1, "a deterministic 400 must not be retried"


def test_transient_429_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def fail(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            return openai.RateLimitError("slow down", response=_resp(429), body=None)
        return None

    e, _ = _embedder(fail)
    vectors = asyncio.run(e.embed(["a", "b"]))
    assert calls["n"] == 2
    assert len(vectors) == 2


def test_failure_names_the_failed_range_rather_than_returning_holes() -> None:
    # Fail only the batch containing the large input.
    def fail(batch):
        if any(len(t) > 100 for t in batch):
            return openai.BadRequestError("nope", response=_resp(400), body=None)
        return None

    texts = ["short"] * 30 + ["word " * 7000] + ["short"] * 30
    e, _ = _embedder(fail)
    with pytest.raises(EmbeddingBatchError) as ei:
        asyncio.run(e.embed(texts))
    assert ei.value.failed_ranges, "the failing input range must be reported"
    assert "succeeded" in str(ei.value)


def _resp(status: int):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "http://x"))

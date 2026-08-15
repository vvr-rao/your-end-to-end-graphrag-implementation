"""Token-budgeted batching.

Every provider caps how many tokens one request may carry. Sizing batches by
*item count* is a guess that holds only while the mean item stays small: the
embeddings path shipped `batch_size=100` against a per-item cap of 8,000, so a
single request could reach 800,000 tokens against OpenAI's 300,000 limit. It
worked until document summaries averaged 3,302 tokens, then failed every run.

This module sizes batches by measured tokens instead, so a request can never be
built over the limit in the first place.

Usage:
    # when you have the rendered strings
    batches = batch_by_token_budget(items, render=str, max_tokens=250_000)

    # when you already know the token counts (avoids re-encoding)
    for start, end in plan_batches(counts, max_tokens=250_000, max_items=100):
        ...  # items[start:end]
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TypeVar

import tiktoken

T = TypeVar("T")

# cl100k_base covers text-embedding-3-* and the gpt-4 family. The summarizer
# uses o200k_base; callers pass their own name when it matters. Exact parity
# with the provider's server-side count is not required -- callers leave
# headroom below the hard limit -- but the estimate must never run low, which
# is why oversized items are isolated rather than packed optimistically.
DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def get_encoder(encoding_name: str = DEFAULT_ENCODING):
    """Cached tiktoken encoder. Falls back to cl100k_base on an unknown name."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding(DEFAULT_ENCODING)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Token count for one string.

    `disallowed_special=()` mirrors embeddings.py: corpus text legitimately
    contains sequences like '<|endoftext|>' and encoding must not raise on them.
    """
    return len(get_encoder(encoding_name).encode(text, disallowed_special=()))


def count_tokens_each(
    texts: Sequence[str], encoding_name: str = DEFAULT_ENCODING
) -> list[int]:
    """Token count per string, one encoder lookup for the whole sequence."""
    enc = get_encoder(encoding_name)
    return [len(enc.encode(t, disallowed_special=())) for t in texts]


class OversizedItem(ValueError):
    """A single item exceeds the whole per-request budget on its own."""


def plan_batches(
    token_counts: Sequence[int],
    max_tokens: int,
    max_items: int | None = None,
) -> list[tuple[int, int]]:
    """Group consecutive items into (start, end) half-open index ranges whose
    summed token count never exceeds `max_tokens`.

    Returning index ranges rather than copied sublists is deliberate: callers
    that write results back into a preallocated output list need the true start
    offset of each batch. Deriving it as `batch_index * batch_size` is only
    correct for uniform batches, and silently misplaces results once batches
    vary in length.

    An item larger than `max_tokens` on its own is placed in a batch by itself
    -- it cannot be made to fit by regrouping, so the caller (or the provider)
    surfaces it rather than having it silently poison a shared request.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if max_items is not None and max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")

    ranges: list[tuple[int, int]] = []
    start = 0
    running = 0

    for i, n in enumerate(token_counts):
        at_limit = max_items is not None and (i - start) >= max_items
        # Close before adding, so the batch under construction is always valid.
        # `i > start` guards the oversized-single-item case: an item bigger than
        # the whole budget must still go somewhere, alone.
        if i > start and (running + n > max_tokens or at_limit):
            ranges.append((start, i))
            start = i
            running = 0
        running += n

    if start < len(token_counts):
        ranges.append((start, len(token_counts)))
    return ranges


def batch_by_token_budget(
    items: Sequence[T],
    render: Callable[[T], str],
    max_tokens: int,
    max_items: int | None = None,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[list[T]]:
    """Split `items` into batches whose rendered size fits `max_tokens`.

    `render` turns one item into the text that will actually go into the
    request, so the measurement matches what is sent (e.g. `json.dumps` for a
    JSON payload). Input order is preserved across batches.
    """
    if not items:
        return []
    counts = count_tokens_each([render(it) for it in items], encoding_name)
    return [
        list(items[start:end])
        for start, end in plan_batches(counts, max_tokens, max_items)
    ]


def oversized_indices(token_counts: Sequence[int], max_tokens: int) -> list[int]:
    """Indices of items that exceed the per-request budget on their own.

    Callers use this to report the problem rather than discovering it as a
    provider 400 halfway through a paid run.
    """
    return [i for i, n in enumerate(token_counts) if n > max_tokens]

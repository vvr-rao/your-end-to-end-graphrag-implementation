"""Shrink a long text to fit the embedding model's input cap by SUMMARIZING
it, rather than letting the embedder hard-truncate the tail away.

Relocated verbatim from `db_document_ingest._compress_summary_for_embed` when
corpus selection needed the same behaviour. The ingest path re-imports it from
here, so its behaviour is unchanged; the only addition is `log_prefix`, so
console lines say `[select]` when selection drives it and `[ingest]` when
ingestion does.

Why this exists: `embeddings._truncate_to_token_limit` cuts at 8000 tokens and
silently discards everything after it, so the tail of a long document never
reaches its vector and is unreachable by search. Paying for one more cheap
summarization pass keeps the whole document represented.

Three tiers, in order:
  1. One `document_summarize` pass (prose).
  2. Above 100K tokens, split on a paragraph boundary, compress each half,
     recurse on the concatenation.
  3. When prose fails to shrink at all, a dense `embed_digest` pass (keywords
     and names rather than sentences), which compresses reliably.
"""
from __future__ import annotations

import logging

from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS

log = logging.getLogger(__name__)

# Embedding model input cap (OpenAI: 8192 hard; 8000 leaves headroom).
EMBED_INPUT_CAP_TOKENS = 8000
# Per gpt-4o-mini call we never feed more than ~100K tokens (matches
# Phase 1's `max_doc_input_tokens` in `summarize_long_documents_async`).
# Above that we split + summarize each half + recurse.
_LLM_INPUT_SPLIT_THRESHOLD = 100_000


def count_embed_tokens(text: str) -> int | None:
    """Token count under the embedding model's encoder, or None if
    tiktoken is unavailable (in which case truncation is unknowable here
    and the embedder reports it instead)."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
    except (ImportError, KeyError):
        return None
    return len(enc.encode(text, disallowed_special=()))


async def compress_for_embed(
    text: str,
    router: LLMRouter,
    *,
    target_tokens: int = EMBED_INPUT_CAP_TOKENS,
    log_prefix: str = "ingest",
    task: str = "document_summarize",
) -> str:
    """If a text exceeds the embedding model's input cap, run extra
    gpt-4o-mini pass(es) to compress it BEFORE the embedding API call.
    Always preserves the original text -- the caller keeps the
    UNCOMPRESSED text; this output is used only to compute the vector.

    `task` selects the router task these calls run on. It defaults to
    `document_summarize` so the ingest path is unchanged -- but note that
    task maps to gpt-4.1 in the shipped presets, so a caller compressing
    many documents (corpus selection) should pass a cheaper one.

    Recursive contract:
      - Returns the input unchanged if it's already <= target_tokens.
      - For inputs <= 100K tokens: one gpt-4o-mini compression call
        (max_tokens=4096 by task config, well under target_tokens).
      - For inputs > 100K tokens: split in half on a paragraph
        boundary, compress each half independently, recurse on the
        concatenation. Bounded by an iteration cap so a pathological
        case can't loop forever.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
    except (ImportError, KeyError):
        return text  # let the embedder do hard truncation

    def _tok_count(s: str) -> int:
        return len(enc.encode(s, disallowed_special=()))

    async def _compress_once(s: str) -> str:
        """One compression pass via `task` (default: document_summarize)."""
        system, user = PROMPTS["document_summarize"](s)
        try:
            result = await router.chat(task, system=system, user=user)
        except Exception as exc:
            print(f"[{log_prefix}] compression LLM call failed: {exc}")
            return s
        return result.text.strip() or s

    async def _digest_once(s: str) -> str:
        """Fallback tier: a dense retrieval digest rather than prose.

        `document_summarize` writes readable prose and sometimes fails to
        shrink at all -- at which point the embedder used to hard-truncate,
        silently dropping the tail of the document out of its vector. This
        asks for keywords and names instead, which compresses reliably and
        keeps both the brand and generic name of everything (a vector
        holding only "MOUNJARO" is unreachable by a query saying
        "tirzepatide").

        Runs on the SAME `document_summarize` router task -- PROMPTS keys
        and task names are independent lookups -- so no models.yaml preset
        needs a new entry on a live deployment.
        """
        system, user = PROMPTS["embed_digest"](s, target_tokens)
        try:
            result = await router.chat(task, system=system, user=user)
        except Exception as exc:
            print(f"[{log_prefix}] embed-digest LLM call failed: {exc}")
            return s
        return result.text.strip() or s

    def _split_on_paragraph(s: str) -> tuple[str, str]:
        """Split near the middle on a blank-line boundary if one exists."""
        mid = len(s) // 2
        # Search outwards for a `\n\n` near the midpoint.
        right = s.find("\n\n", mid)
        left = s.rfind("\n\n", 0, mid)
        if right >= 0 and (left < 0 or right - mid <= mid - left):
            return s[:right], s[right + 2 :]
        if left >= 0:
            return s[:left], s[left + 2 :]
        return s[:mid], s[mid:]

    n = _tok_count(text)
    if n <= target_tokens:
        return text

    print(
        f"[{log_prefix}] doc summary {n:,} tokens > {target_tokens} cap; "
        f"compressing for embed call (full text still kept)"
    )

    # Iteration cap so we never loop forever on a pathological input.
    current = text
    for iteration in range(4):
        n = _tok_count(current)
        if n <= target_tokens:
            return current

        if n <= _LLM_INPUT_SPLIT_THRESHOLD:
            compressed = await _compress_once(current)
            n2 = _tok_count(compressed)
            if n2 >= n:
                # Prose summarization didn't shrink it. Previously this
                # returned immediately and let the embedder hard-truncate,
                # dropping the tail of the document out of its vector with
                # only an anonymous "[embed] N/M truncated" line to show
                # for it. Try the dense-digest tier before giving up.
                print(
                    f"[{log_prefix}] compression no-op (out={n2:,} >= in={n:,}); "
                    "falling back to dense embed-digest"
                )
                digest = await _digest_once(current)
                n3 = _tok_count(digest)
                if n3 < n:
                    print(
                        f"[{log_prefix}] embed-digest: {n:,} -> {n3:,} tokens"
                    )
                    current = digest
                    continue
                print(
                    f"[{log_prefix}] embed-digest also failed to shrink "
                    f"(out={n3:,} >= in={n:,})"
                )
                return current
            print(f"[{log_prefix}] compress iter {iteration + 1}: {n:,} -> {n2:,} tokens")
            current = compressed
            continue

        # Above split threshold: divide-and-conquer
        left, right = _split_on_paragraph(current)
        comp_l = await _compress_once(left) if _tok_count(left) > target_tokens else left
        comp_r = await _compress_once(right) if _tok_count(right) > target_tokens else right
        current = comp_l + "\n\n" + comp_r
        print(
            f"[{log_prefix}] split-and-compress iter {iteration + 1}: "
            f"{n:,} -> {_tok_count(current):,} tokens"
        )

    # Last chance before the embedder's hard cut: one dense-digest pass.
    if _tok_count(current) > target_tokens:
        digest = await _digest_once(current)
        if _tok_count(digest) < _tok_count(current):
            current = digest

    if _tok_count(current) > target_tokens:
        log.warning(
            "document embedding input still %s tokens after compression "
            "(cap %s); the embedder will hard-truncate and the tail of this "
            "document will not be searchable by its vector",
            f"{_tok_count(current):,}", f"{target_tokens:,}",
        )
    return current

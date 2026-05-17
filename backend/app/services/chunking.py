"""Paragraph-first token-aware chunker (ported from reference notebook cell 33).

Strategy:
  1. Split text on blank lines (double newlines) into paragraphs.
  2. Pack paragraphs into chunks until adding the next would exceed chunk_size
     tokens (measured with tiktoken).
  3. If a single paragraph is itself larger than chunk_size, split it on token
     boundaries (no overlap inside the paragraph; the chunk-to-chunk overlap
     happens at paragraph boundaries below).
  4. Optional overlap: last N tokens of chunk K appear as the first N tokens
     of chunk K+1.

Chunk sizes default to config.example.yaml values (800 tokens, 120 overlap,
o200k_base encoding) but the caller can override.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import tiktoken


@dataclass
class TextChunk:
    """One chunk of a source document."""

    index: int
    text: str
    token_count: int
    source_name: str | None = None


def _get_encoder(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _split_long_paragraph(paragraph: str, chunk_size: int, encoder) -> list[str]:
    tokens = encoder.encode(paragraph)
    out: list[str] = []
    for start in range(0, len(tokens), chunk_size):
        piece_tokens = tokens[start : start + chunk_size]
        out.append(encoder.decode(piece_tokens))
    return out


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "o200k_base",
    source_name: str | None = None,
) -> Iterator[TextChunk]:
    """Yield TextChunk objects for a single document body.

    chunk_overlap is measured in tokens; the last `chunk_overlap` tokens of
    each chunk are prepended to the next chunk's text.
    """
    if not text.strip():
        return
    encoder = _get_encoder(encoding_name)
    paragraphs = _split_paragraphs(text)

    # Normalize: if a paragraph is itself too big, pre-split it on token bounds.
    flat: list[str] = []
    for para in paragraphs:
        para_tokens = encoder.encode(para)
        if len(para_tokens) <= chunk_size:
            flat.append(para)
        else:
            flat.extend(_split_long_paragraph(para, chunk_size, encoder))

    # Pack paragraphs into chunks.
    chunk_idx = 0
    current_paragraphs: list[str] = []
    current_token_count = 0
    last_chunk_tail: list[int] = []  # tokens to overlap into next chunk

    def _emit_chunk() -> TextChunk | None:
        nonlocal chunk_idx, current_paragraphs, current_token_count, last_chunk_tail
        if not current_paragraphs:
            return None
        body = "\n\n".join(current_paragraphs)
        body_tokens = encoder.encode(body)
        # Prepend overlap if any
        if last_chunk_tail:
            body_tokens = last_chunk_tail + body_tokens
            body = encoder.decode(body_tokens)
        # Remember tail for next chunk
        if chunk_overlap and len(body_tokens) > chunk_overlap:
            last_chunk_tail = body_tokens[-chunk_overlap:]
        else:
            last_chunk_tail = []
        result = TextChunk(index=chunk_idx, text=body, token_count=len(body_tokens), source_name=source_name)
        chunk_idx += 1
        current_paragraphs = []
        current_token_count = 0
        return result

    for para in flat:
        para_tokens = len(encoder.encode(para))
        if current_token_count + para_tokens > chunk_size and current_paragraphs:
            chunk = _emit_chunk()
            if chunk:
                yield chunk
        current_paragraphs.append(para)
        current_token_count += para_tokens

    final = _emit_chunk()
    if final:
        yield final


def chunk_documents(
    docs,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "o200k_base",
) -> Iterator[TextChunk]:
    """Chunk every document in `docs` (an iterable of LoadedDocument)."""
    for doc in docs:
        yield from chunk_text(
            doc.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            source_name=doc.name,
        )

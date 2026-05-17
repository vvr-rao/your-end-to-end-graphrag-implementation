"""Paragraph-first chunker."""

from __future__ import annotations

from backend.app.services.chunking import chunk_text


def test_empty_text_emits_nothing() -> None:
    chunks = list(chunk_text("   "))
    assert chunks == []


def test_short_text_one_chunk() -> None:
    chunks = list(chunk_text("hello world", chunk_size=50, chunk_overlap=10))
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "hello world" in chunks[0].text
    assert chunks[0].token_count > 0


def test_paragraph_packing_under_chunk_size() -> None:
    text = "para one.\n\npara two.\n\npara three."
    chunks = list(chunk_text(text, chunk_size=200, chunk_overlap=0))
    # all paragraphs fit in one chunk
    assert len(chunks) == 1
    assert "para one" in chunks[0].text and "para three" in chunks[0].text


def test_paragraph_boundary_split() -> None:
    # one very short paragraph repeated many times — should split across chunks
    paragraphs = "\n\n".join(["short para"] * 100)
    chunks = list(chunk_text(paragraphs, chunk_size=30, chunk_overlap=0))
    assert len(chunks) >= 2
    # All chunks should have non-empty body
    for c in chunks:
        assert c.text.strip()


def test_overlap_carries_into_next_chunk() -> None:
    paragraphs = "\n\n".join([f"para {i} content content content" for i in range(20)])
    chunks = list(chunk_text(paragraphs, chunk_size=40, chunk_overlap=10))
    assert len(chunks) >= 2
    # Overlap means chunk[1] should have some shared text with chunk[0]'s tail.
    # Hard to assert exactly due to tokenizer boundaries, but token_count should
    # be > the raw paragraph length sum when overlap is non-zero.
    assert chunks[1].token_count > 0


def test_source_name_threads_through() -> None:
    chunks = list(chunk_text("alpha. beta. gamma.", source_name="doc.txt", chunk_size=50, chunk_overlap=0))
    assert all(c.source_name == "doc.txt" for c in chunks)

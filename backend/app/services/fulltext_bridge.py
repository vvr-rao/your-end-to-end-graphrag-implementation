"""Full-text bridge: from the summary chunks the graph found to verbatim passages.

The graph reaches SUMMARY chunks (entity edges point at them). When documents
also carry full-text chunks, the bridge swaps verbatim passages of the best
documents into the candidate pool. Two choices, each behind a `qa.` setting
whose default reproduces the original behaviour:

`fulltext_bridge_document_rank` -- which documents are expanded:
  count  documents with the most graph-found summary chunks (original). Long
         documents win: they own more summary chunks.
  best   documents whose best graph-found chunk scored highest, count as
         tie-break. Each graph route's scores are scaled to 0-1 by that route's
         best, so a chunk that tops the walk, the time route, co-occurrence or
         a concept area counts the same.

`fulltext_bridge_passages` -- which passages of a kept document:
  question  the full-text chunks nearest the QUESTION, up to
            `max_fulltext_chunks_per_document` (original). What the walk
            matched in the document is not used.
  anchors   the full-text chunks nearest each graph-found summary chunk of
            that document (its "anchors", best first): the passage the summary
            was most likely written from. Plus a few nearest the question as a
            backstop, scored document score x similarity to the question.
            Each anchor passage is scored anchor score x cosine similarity,
            so the graph vote ranks passages within a document instead of
            giving them all the document's score.

No LLM calls: the summary chunks' embeddings are already stored.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable

DOCUMENT_RANKS = ("count", "best")
PASSAGE_MODES = ("question", "anchors")


def route_scores(
    routes: Iterable[Iterable[tuple[uuid.UUID, float]]],
) -> dict[uuid.UUID, float]:
    """Each route's scores scaled to 0-1 by its best, then the best per chunk.

    A route with no positive score (e.g. all zeros) counts every chunk as 1:
    being found at all is the signal there."""
    out: dict[uuid.UUID, float] = {}
    for route in routes:
        items = list(route)
        if not items:
            continue
        top = max(float(s) for _, s in items)
        for cid, s in items:
            v = float(s) / top if top > 0 else 1.0
            if v > out.get(cid, -1.0):
                out[cid] = v
    return out


def rank_documents(
    chunk_scores: dict[uuid.UUID, float],
    doc_of: dict[uuid.UUID, uuid.UUID],
    *,
    mode: str = "count",
    candidates: Iterable[uuid.UUID] | None = None,
) -> list[tuple[uuid.UUID, float, int]]:
    """`(document, best chunk score, chunks found)` best first.

    `candidates` limits the result to these documents (those with full text);
    a candidate the scored chunks never reach ranks last with (0, 0)."""
    best: dict[uuid.UUID, float] = {}
    count: dict[uuid.UUID, int] = {}
    for cid, sc in chunk_scores.items():
        d = doc_of.get(cid)
        if d is None:
            continue
        best[d] = max(best.get(d, 0.0), sc)
        count[d] = count.get(d, 0) + 1
    docs = set(candidates) if candidates is not None else set(best)
    rows = [(d, best.get(d, 0.0), count.get(d, 0)) for d in docs]
    if mode == "best":
        rows.sort(key=lambda r: (-r[1], -r[2], str(r[0])))
    else:
        rows.sort(key=lambda r: (-r[2], -r[1], str(r[0])))
    return rows


def pick_anchors(
    chunk_scores: dict[uuid.UUID, float],
    doc_of: dict[uuid.UUID, uuid.UUID],
    documents: Iterable[uuid.UUID],
    *,
    per_document: int = 5,
    allowed: set[uuid.UUID] | None = None,
) -> list[tuple[uuid.UUID, float]]:
    """The best-scored chunks of each document, at most `per_document` each.
    `allowed` restricts anchors to these chunk ids (the summary chunks)."""
    keep = set(documents)
    by_doc: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = {}
    for cid, sc in chunk_scores.items():
        d = doc_of.get(cid)
        if d in keep and (allowed is None or cid in allowed):
            by_doc.setdefault(d, []).append((cid, sc))
    out: list[tuple[uuid.UUID, float]] = []
    for d in sorted(by_doc, key=str):
        items = sorted(by_doc[d], key=lambda t: (-t[1], str(t[0])))
        out += items[:max(0, per_document)]
    return out

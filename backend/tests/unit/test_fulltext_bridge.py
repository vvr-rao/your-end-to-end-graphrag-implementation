"""Full-text bridge scoring: which documents are expanded, which chunks anchor."""
from __future__ import annotations

import uuid

from backend.app.services.fulltext_bridge import pick_anchors, rank_documents, route_scores

U = [uuid.UUID(int=i) for i in range(1, 20)]


def test_route_scores_scale_each_route_and_keep_the_best():
    walk = [(U[0], 4.0), (U[1], 2.0)]
    time = [(U[1], 10.0), (U[2], 5.0)]
    s = route_scores([walk, time])
    assert s == {U[0]: 1.0, U[1]: 1.0, U[2]: 0.5}


def test_route_with_no_positive_score_counts_each_chunk_as_found():
    assert route_scores([[(U[0], 0.0), (U[1], 0.0)], []]) == {U[0]: 1.0, U[1]: 1.0}


def test_count_mode_prefers_the_long_document_best_mode_the_focused_one():
    long_doc, short_doc = U[10], U[11]
    # The long document owns 4 weak chunks; the short one 1 strong chunk.
    scores = {U[0]: 0.2, U[1]: 0.2, U[2]: 0.3, U[3]: 0.2, U[4]: 1.0}
    doc_of = {U[0]: long_doc, U[1]: long_doc, U[2]: long_doc, U[3]: long_doc,
              U[4]: short_doc}
    assert [d for d, _, _ in rank_documents(scores, doc_of, mode="count")] == [
        long_doc, short_doc]
    assert [d for d, _, _ in rank_documents(scores, doc_of, mode="best")] == [
        short_doc, long_doc]


def test_best_mode_breaks_ties_by_count():
    a, b = U[10], U[11]
    scores = {U[0]: 1.0, U[1]: 1.0, U[2]: 0.5}
    doc_of = {U[0]: a, U[1]: b, U[2]: b}
    assert [d for d, _, _ in rank_documents(scores, doc_of, mode="best")] == [b, a]


def test_candidates_limit_documents_and_unreached_ones_rank_last():
    a, b, c = U[10], U[11], U[12]
    scores = {U[0]: 1.0, U[1]: 0.5}
    doc_of = {U[0]: a, U[1]: b}
    out = rank_documents(scores, doc_of, mode="best", candidates={b, c})
    assert out == [(b, 0.5, 1), (c, 0.0, 0)]


def test_anchors_are_the_best_summary_chunks_per_kept_document():
    a, b = U[10], U[11]
    scores = {U[0]: 0.9, U[1]: 0.8, U[2]: 0.7, U[3]: 1.0, U[4]: 0.4}
    doc_of = {U[0]: a, U[1]: a, U[2]: a, U[3]: a, U[4]: b}
    summary = {U[0], U[1], U[2], U[4]}  # U[3] is not a summary chunk
    got = pick_anchors(scores, doc_of, [a], per_document=2, allowed=summary)
    assert got == [(U[0], 0.9), (U[1], 0.8)]
    assert pick_anchors(scores, doc_of, [a, b], per_document=1, allowed=summary) == [
        (U[0], 0.9), (U[4], 0.4)][:: 1 if str(a) < str(b) else -1]

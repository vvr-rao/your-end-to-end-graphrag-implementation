"""candidate_tables ranking helpers: lexical terms, IDF, reciprocal-rank fusion."""
from __future__ import annotations

import uuid

from backend.app.services.structured.retrieval_step import _fuse, _idf, _rank_terms


def test_rank_terms_drops_function_words_and_folds_plurals():
    t = _rank_terms("Which three countries imported the most urea by value in 2023?")
    assert {"three", "country", "imported", "urea", "2023"} <= t
    assert not t & {"which", "the", "most", "by", "in", "value"}
    assert _rank_terms("gross class") == {"gross", "class"}   # -ss is not a plural
    assert _rank_terms(None) == set()


def test_idf_rewards_rare_links():
    # a node linked to every table says little; one linked to one table says much
    assert _idf(7, 1) > _idf(7, 3) > _idf(7, 7) > 0


def test_fusion_lets_agreeing_signals_beat_one_strong_signal():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    graph = {a: 9.0, b: 8.0, c: 1.0}          # a and b share many places with the question
    vector = {c: 0.6, a: 0.5, b: 0.4}         # c is the table the question describes
    lexical = {c: 3.0}                        # and the one whose words it uses
    fused = _fuse([graph, vector, lexical])
    assert max(fused, key=fused.get) == c


def test_fusion_ignores_absent_and_zero_signals():
    a, b = uuid.uuid4(), uuid.uuid4()
    assert _fuse([{a: 0.0}, {}]) == {}
    fused = _fuse([{a: 1.0, b: 0.0}])
    assert set(fused) == {a}

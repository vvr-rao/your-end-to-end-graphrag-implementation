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


def test_a_table_alone_holding_a_long_named_value_gets_a_slot():
    from backend.app.services.structured.retrieval_step import reserve_named_value_slots

    t1, t2, t3, t4, t5 = "t1", "t2", "t3", "t4", "t5"
    long_v = frozenset({"energy", "import", "net", "use"})
    short_v = frozenset({"import"})
    shared = frozenset({"fertilizer", "export", "quantity"})
    matched = {t5: {long_v: "Energy imports, net (% of energy use)"}, t4: {short_v: "Import"},
               t1: {shared: "x"}, t2: {shared: "x"}}
    vdf = {long_v: 1, short_v: 1, shared: 2}
    best, reserved = reserve_named_value_slots([t1, t2, t3, t4, t5], 3, matched, vdf)
    assert best == [t1, t2, t5] and reserved == [t5]          # replaces the lowest-ranked table
    # a short value or one several tables hold reserves nothing
    best, reserved = reserve_named_value_slots([t1, t2, t3, t4], 2, matched, vdf)
    assert best == [t1, t2] and reserved == []

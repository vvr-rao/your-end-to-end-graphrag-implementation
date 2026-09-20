"""The text one relationship edge is embedded as.

The shape is load-bearing in two directions: `retrieval._relation_text`
builds the QUERY side to match it, and the measured similarity floor
(`qa.relationship_similarity`) was chosen against this exact formula. A change
here silently shifts every threshold in the config.
"""
from __future__ import annotations

from backend.app.services.db_relationship_embed import build_edge_text


def test_triple_only_evidence_is_excluded():
    # Appending the evidence sentence measurably LOWERED every probe (0.575 ->
    # 0.548 on the geo hop, 0.716 -> 0.668 on "founder founded FTX") and pushed
    # correct matches below the 0.55 threshold. See the docstring for numbers.
    t = build_edge_text(
        "Manhattan federal court", "New York City", None, "locatedin",
        "Manhattan Federal Court is where the trial was held, a long sentence.",
    )
    assert t == "Manhattan federal court locatedin New York City"


def test_relation_phrase_beats_predicate_label():
    # A relatedTo edge's free-text phrase is the passage's own words and
    # matches question vocabulary far better than a camel-cased ontology term.
    assert build_edge_text(
        "A", "B", "was accused of defrauding", "relatedTo", None,
    ) == "A was accused of defrauding B"


def test_falls_back_to_predicate_label():
    assert build_edge_text("A", "B", "", "playsfor", None) == "A playsfor B"
    assert build_edge_text("A", "B", None, None, None) == "A B"


def test_empty_when_nothing_to_say():
    assert build_edge_text(None, None, None, None, None) == ""


def test_matches_the_query_side_shape():
    # retrieval._relation_text must produce the same shape, or the cosine
    # comparison is between differently-built strings.
    from backend.app.services.retrieval import _relation_text

    probe = _relation_text({
        "subject": {"text": "Sam Bankman-Fried", "kind": "entity"},
        "relation": "founded",
        "object": {"text": "FTX", "kind": "entity"},
    })
    assert probe == "Sam Bankman-Fried founded FTX"
    assert probe == build_edge_text("Sam Bankman-Fried", "FTX", "founded", None, None)


def test_open_end_gets_a_placeholder_not_a_question_word():
    from backend.app.services.retrieval import _relation_text

    assert _relation_text({
        "subject": {"text": "", "kind": "unknown"},
        "relation": "was accused of",
        "object": {"text": "fraud", "kind": "class"},
    }) == "someone was accused of fraud"

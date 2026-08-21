"""Evidence attribution tagging in the answer-synthesis prompts.

Pure-unit; no DB, no LLM, no cost.

Motivation: asked to compare two drugs, the synthesis reported one drug's
maximum dose under the other's name. Every claim traced to retrieved
evidence -- the evidence was simply about the wrong drug, which is why
the `no_hallucination` metric passed it. Tagging each evidence item with
the document and entities it concerns is what makes the attribution rule
in the system prompt checkable rather than aspirational.
"""
from __future__ import annotations

from backend.app.services.prompts import (
    _ATTRIBUTION_RULE,
    _format_evidence_block,
    answer_deep_research,
    answer_simple_qa,
)


def _chunk(**over):
    base = {
        "kind": "chunk",
        "iri": "viao:Chunk_abc",
        "text": "The maximum dosage is 4.5 mg once weekly.",
        "document_title": "TRULICITY dulaglutide injection",
        "entities": ["dulaglutide", "TRULICITY"],
    }
    base.update(over)
    return base


def test_document_and_entities_are_tagged():
    out = _format_evidence_block([_chunk()])
    assert "document: TRULICITY dulaglutide injection" in out
    assert "about: dulaglutide, TRULICITY" in out
    assert "4.5 mg" in out


def test_untagged_evidence_still_renders():
    """Artifacts carry no document/entity tags; they must not break or
    emit an empty '()' marker.

    The bracket now holds the citable short form rather than `kind + iri`,
    so the model copies what it is asked to write instead of transcribing
    a URL into it."""
    out = _format_evidence_block(
        [{"kind": "artifact", "iri": "viao:Claim_1", "text": "A claim."}]
    )
    assert out.strip() == "[viao:Claim_1] A claim."


def test_partial_tags():
    doc_only = _format_evidence_block([_chunk(entities=[])])
    assert "document:" in doc_only and "about:" not in doc_only

    ents_only = _format_evidence_block([_chunk(document_title=None)])
    assert "about:" in ents_only and "document:" not in ents_only


def test_entity_list_is_bounded():
    out = _format_evidence_block([_chunk(entities=[f"e{i}" for i in range(20)])])
    assert "e5" in out and "e6" not in out


def test_char_cap_still_applies_to_text_only():
    """The cap bounds the passage, not the tag -- attribution must not be
    the thing that gets truncated away."""
    out = _format_evidence_block([_chunk(text="x" * 5000)], char_cap=100)
    assert "document: TRULICITY dulaglutide injection" in out
    assert "x" * 100 + "..." in out
    assert "x" * 200 not in out


def test_both_answer_modes_carry_the_attribution_rule():
    for fn in (answer_simple_qa, answer_deep_research):
        system, user = fn("Compare dosing of tirzepatide and dulaglutide", [_chunk()])
        assert _ATTRIBUTION_RULE in system, f"{fn.__name__} missing attribution rule"
        # And the evidence the rule refers to is actually tagged.
        assert "about: dulaglutide" in user


def test_attribution_rule_demands_naming_the_gap():
    """The rule has to require an explicit statement of absence -- silence
    is what let a sibling entity's number get substituted."""
    lowered = _ATTRIBUTION_RULE.lower()
    assert "never transfer" in lowered
    assert "no <property> for <entity>" in lowered or "contains no" in lowered


# --------------------------------------------------------------------------
# Alias block: retrieval knows the synonyms, so the synthesis must too
# --------------------------------------------------------------------------


def test_alias_block_lists_corpus_verified_pairs():
    for fn in (answer_simple_qa, answer_deep_research):
        _, user = fn(
            "Compare dosing of tirzepatide and dulaglutide",
            [_chunk()],
            1500,
            {"tirzepatide": ["Mounjaro", "Zepbound"], "dulaglutide": ["Trulicity"]},
        )
        assert "ALTERNATE NAMES" in user, fn.__name__
        assert "tirzepatide = Mounjaro, Zepbound" in user
        assert "dulaglutide = Trulicity" in user


def test_alias_block_absent_when_nothing_was_mined():
    """REGRESSION GUARD: on a database where `mine-aliases` never ran, the
    prompt must be byte-identical to the pre-alias version."""
    for fn in (answer_simple_qa, answer_deep_research):
        base = fn("q", [_chunk()], 1500)
        for empty in (None, {}, {"tirzepatide": []}):
            assert fn("q", [_chunk()], 1500, empty) == base, fn.__name__
        assert "ALTERNATE NAMES" not in base[1]


# --------------------------------------------------------------------------
# Time scope (audit D3): "by 2025" was extracted, then ignored
# --------------------------------------------------------------------------


def test_evidence_carries_its_time_tag():
    """Without a per-item date the model cannot honour a date constraint
    even when instructed -- the information simply is not in the prompt."""
    out = _format_evidence_block([_chunk(times="2026")])
    assert "time: 2026" in out


def test_evidence_renders_a_time_span():
    out = _format_evidence_block([_chunk(times="2011 ... 2024 (8 periods)")])
    assert "time: 2011 ... 2024 (8 periods)" in out


def test_evidence_still_accepts_a_list_of_labels():
    """The span is pre-rendered upstream, but a caller passing raw labels
    must not silently lose them."""
    out = _format_evidence_block([_chunk(times=["2023", "2024"])])
    assert "time: 2023, 2024" in out


def test_untimed_evidence_has_no_time_tag():
    for empty in ("", None, []):
        assert "time:" not in _format_evidence_block([_chunk(times=empty)])


def test_time_scope_tells_the_model_undated_is_not_out_of_window():
    """The denial this guards against: undated evidence being treated as
    outside the window, so a forward-bounded question reported a gap that
    does not exist."""
    for fn in (answer_simple_qa, answer_deep_research):
        _, user = fn("warnings after 2023", [_chunk()], 1500, {}, ["after 2023"])
        low = user.lower()
        assert "unknown date" in low or "unknown" in low, fn.__name__
        assert "do not exclude" in low
        assert "never report that the corpus contains no such material" in low


def test_time_scope_block_states_the_constraint():
    for fn in (answer_simple_qa, answer_deep_research):
        _, user = fn("competitors by 2025", [_chunk(times=["2026"])], 1500, {}, ["2025"])
        assert "TIME SCOPE" in user, fn.__name__
        assert "2025" in user
        # ...and it must demand flagging, not silent inclusion.
        assert "outside" in user.lower()


def test_time_scope_absent_when_the_question_has_no_date():
    """REGRESSION GUARD: an undated question must produce a prompt
    byte-identical to before this feature."""
    for fn in (answer_simple_qa, answer_deep_research):
        base = fn("what is the dose", [_chunk()], 1500, {})
        for empty in (None, [], [""]):
            assert fn("what is the dose", [_chunk()], 1500, {}, empty) == base
        assert "TIME SCOPE" not in base[1]


def test_attribution_rule_points_at_the_alias_block():
    """The rule must tell the model the listed pairs ARE the same entity.
    Without that it stays conservative and refuses to merge them, which
    is the behavior the alias block exists to override."""
    lowered = _ATTRIBUTION_RULE.lower()
    assert "alternate names" in lowered
    # ...while still forbidding unlisted merges.
    assert "unless an evidence item shows" in lowered

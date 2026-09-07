"""A citation that RESOLVES is not necessarily a citation that SUPPORTS.

`_validate_citations` checks only that a cited id exists in the evidence set.
That leaves a worse defect uncovered: an answer citing a real-but-irrelevant
artifact reads as fully sourced and passes every traceability check we have.

Found by `evaluate-queries` on the pharma corpus, where an answer said

    "semaglutide is approved for type 2 diabetes under the brand name Ozempic
     and for weight management under the brand name Wegovy"

and cited a Claim whose entire text is

    "Mounjaro is approved for Type 2 diabetes, while Zepbound is approved for
     weight management in adults with a BMI of 30 or more."

The fact is true, the id resolves, and the citation supports a different
sentence. Same shape as the evidence_char_cap bug: plausible, well-cited,
wrong.
"""

from __future__ import annotations

import pytest

from backend.app.services.retrieval import (
    _named_things,
    _unsupported_citations,
)

_SEMA = (
    "The drug semaglutide is approved for type 2 diabetes under the brand "
    "name Ozempic and for weight management under the brand name Wegovy"
)
_WRONG_CLAIM = {
    "iri": "viao:Claim_wrong",
    "text": "Mounjaro is approved for Type 2 diabetes, while Zepbound is "
            "approved for weight management in adults with a BMI of 30 or more.",
}
_RIGHT_CLAIM = {
    "iri": "viao:Claim_right",
    "text": "Under the brand name Ozempic, it's approved to treat Type 2 "
            "diabetes, while under the brand name Wegovy, it's approved for "
            "weight loss.",
}


def test_the_real_mis_attribution_is_flagged() -> None:
    out = _unsupported_citations(
        f"{_SEMA} [viao:Claim_wrong].", [_WRONG_CLAIM, _RIGHT_CLAIM])
    assert len(out) == 1
    assert out[0]["citation"] == "Claim_wrong"
    assert set(out[0]["names"]) == {"ozempic", "wegovy"}


def test_a_genuinely_supporting_citation_passes() -> None:
    assert _unsupported_citations(
        f"{_SEMA} [viao:Claim_right].", [_WRONG_CLAIM, _RIGHT_CLAIM]) == []


def test_rarity_weighting_would_have_missed_it() -> None:
    """Guards the design choice, not just the behaviour.

    The first implementation ranked terms by rarity across the evidence set.
    That INVERTS on a topically narrow corpus: measured on 56 pharma
    artifacts, `ozempic` and `wegovy` appeared in 20+ (common -- every
    document is about them) while `name` and `brand` appeared in 0-1 (rare
    and meaningless). Rarity picked the generic words and dropped the drug
    names, so the real case scored as supported.

    Proper nouns survive that inversion: the two spans share topic vocabulary
    (diabetes, weight, management) and no NAMES.
    """
    sent_names = _named_things(_SEMA)
    cited_names = _named_things(_WRONG_CLAIM["text"])
    assert sent_names & cited_names == set()
    shared_topic = {"diabetes", "weight", "management"}
    assert shared_topic <= {w.lower() for w in _SEMA.split()} | {
        w.strip(",.").lower() for w in _WRONG_CLAIM["text"].split()}


def test_a_sentence_naming_nothing_is_skipped_not_guessed() -> None:
    """No names means no signal. Guessing would generate noise on exactly the
    generic sentences where the check has least to say."""
    assert _unsupported_citations(
        "The side effects are generally mild and temporary [viao:Claim_wrong].",
        [_WRONG_CLAIM]) == []


def test_unresolvable_ids_are_left_to_the_other_validator() -> None:
    assert _unsupported_citations(
        f"{_SEMA} [viao:Claim_does_not_exist].", [_WRONG_CLAIM]) == []


def test_sentence_initial_capitals_are_not_treated_as_names() -> None:
    assert "the" not in _named_things("The drug is approved.")
    assert "these" not in _named_things("These results were significant.")


@pytest.mark.parametrize("payload", [("", []), ("answer", []), ("", [_WRONG_CLAIM])])
def test_empty_inputs_are_safe(payload) -> None:
    ans, ev = payload
    assert _unsupported_citations(ans, ev) == []

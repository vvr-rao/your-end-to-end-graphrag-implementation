"""Citation validation + the short-form citation token.

Pure-unit; no DB, no LLM, no cost.

Motivation (audit 2026-08-19, D2): one answer emitted four citations that
resolved to nothing -- three invented, one a real document id with its
`_ft_NNNN` suffix dropped. The claims themselves were true and separately
cited, so no reader was misled on fact, but a citation resolving to
nothing breaks the traceability chain the artifact layer exists to
provide. `no_hallucination` scored that answer a clean 1.0, because it
judges whether CLAIMS are grounded and never looks at identifiers.

Root cause was a format mismatch: the evidence block rendered a full
90-character URL while the instructions asked for `[viao:Chunk_...]`, so
every citation was a hand transcription of a 16-hex hash.
"""
from __future__ import annotations

import pytest

from backend.app.services.prompts import _format_evidence_block, citation_token
from backend.app.services.retrieval import _citation_key, _validate_citations

_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"
_REAL = f"{_NS}#Chunk_8d5104fae90b5632_ft_0005"
_REAL_ART = f"{_NS}#Claim_9212ccabafc04929"


def _ev(iri: str = _REAL, **over):
    base = {
        "kind": "chunk",
        "iri": iri,
        "text": "The maximum dosage is 15 mg once weekly.",
        "document_title": "MOUNJARO label",
        "entities": ["tirzepatide"],
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# Prevention: show the model exactly what it is asked to write
# --------------------------------------------------------------------------


def test_citation_token_is_the_short_form():
    assert citation_token(_ev()) == "viao:Chunk_8d5104fae90b5632_ft_0005"
    assert citation_token(_ev(_REAL_ART)) == "viao:Claim_9212ccabafc04929"


def test_evidence_block_renders_the_citable_form_not_the_url():
    """The block and the instruction must agree, or citing becomes
    transcription and transcription introduces errors."""
    out = _format_evidence_block([_ev()])
    assert "[viao:Chunk_8d5104fae90b5632_ft_0005]" in out
    assert _NS not in out, "full URL leaked back into the prompt"


def test_citation_token_survives_a_missing_iri():
    assert citation_token({"kind": "chunk"}) == "viao:unknown"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_valid_citation_is_kept_untouched():
    answer = "The maximum dose is 15 mg [viao:Chunk_8d5104fae90b5632_ft_0005]."
    cleaned, invalid = _validate_citations(answer, [_ev()])
    assert cleaned == answer
    assert invalid == []


def test_fabricated_citation_is_stripped():
    """Three of the audit's four dead ids were pure inventions."""
    answer = (
        "Tirzepatide reaches 15 mg [viao:Chunk_8d5104fae90b5632_ft_0005] "
        "and competes broadly [viao:Chunk_57020efec0c4433c]."
    )
    cleaned, invalid = _validate_citations(answer, [_ev()])
    assert invalid == ["Chunk_57020efec0c4433c"]
    assert "Chunk_57020efec0c4433c" not in cleaned
    assert "[viao:Chunk_8d5104fae90b5632_ft_0005]" in cleaned


def test_truncated_real_id_is_stripped():
    """The audit's fourth case: a real document hash with the `_ft_NNNN`
    suffix dropped. It looks plausible and resolves to nothing, which is
    the worst combination."""
    answer = "Approved in 2026 [viao:Chunk_f8d5b501138d570e]."
    cleaned, invalid = _validate_citations(answer, [_ev()])
    assert invalid == ["Chunk_f8d5b501138d570e"]
    assert "Chunk_f8d5b501138d570e" not in cleaned


def test_full_url_form_also_validates():
    """The model may echo the block's form rather than the short one;
    both reduce to the same fragment."""
    answer = f"Dose is 15 mg [chunk {_REAL}]."
    cleaned, invalid = _validate_citations(answer, [_ev()])
    assert invalid == []
    assert cleaned == answer


def test_artifact_citations_validate_too():
    answer = (
        "A claim [viao:Claim_9212ccabafc04929] and a fake "
        "[viao:Claim_deadbeefdeadbeef]."
    )
    cleaned, invalid = _validate_citations(
        answer, [_ev(_REAL_ART, kind="artifact")]
    )
    assert invalid == ["Claim_deadbeefdeadbeef"]
    assert "[viao:Claim_9212ccabafc04929]" in cleaned


def test_removal_tidies_the_seam():
    """Stripping mid-sentence must not leave a doubled space or a space
    stranded before the full stop."""
    answer = "Dosing rises steadily [viao:Chunk_bogus1234567890a] ."
    cleaned, _ = _validate_citations(answer, [_ev()])
    assert "  " not in cleaned
    assert cleaned.endswith(".")
    assert " ." not in cleaned


@pytest.mark.parametrize(
    "raw,expected",
    [
        (f"{_NS}#Chunk_abc", "Chunk_abc"),
        ("viao:Chunk_abc", "Chunk_abc"),
        ("Chunk_abc", "Chunk_abc"),
    ],
)
def test_citation_key_normalizes_both_forms(raw: str, expected: str):
    assert _citation_key(raw) == expected


# --------------------------------------------------------------------------
# Regression guards
# --------------------------------------------------------------------------


def test_answer_without_citations_is_untouched():
    answer = "The corpus contains no dosing evidence for liraglutide."
    cleaned, invalid = _validate_citations(answer, [_ev()])
    assert cleaned == answer and invalid == []


@pytest.mark.parametrize("evidence", [[], None])
def test_no_evidence_means_no_stripping(evidence):
    """With nothing retrieved there is nothing to validate against --
    stripping every citation would be worse than leaving them."""
    answer = "Something [viao:Chunk_x]."
    cleaned, invalid = _validate_citations(answer, evidence or [])
    assert cleaned == answer and invalid == []

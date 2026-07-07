"""Unit tests for the evaluated summarizer building blocks (no LLM/DB): section
coverage, section-boundary size-cap splitting, token windowing, prompt/task
registration, and the configurable evidence-char cap."""

from __future__ import annotations

from pathlib import Path

from backend.app.services.evaluated_summarizer import (
    EvaluatedDocSummary,
    _split_on_sections,
    _token_windows,
    check_section_coverage,
    evaluated_result_to_chunks,
    get_encoder,
)
from backend.app.services.prompts import (
    PROMPTS,
    REQUIRED_SECTIONS,
    _format_evidence_block,
    answer_deep_research,
)

_ENC = get_encoder()

_SECTIONS = [
    "CLAIMS", "EVIDENCE", "FINDINGS", "OBSERVATIONS", "EVENTS",
    "ENTITIES", "DATES", "REGULATORY / SAFETY / WARNINGS", "LISTS",
]


def _fake_summary(words_per_section: int = 5) -> str:
    return "\n".join(f"{h}\n" + ("word " * words_per_section) for h in _SECTIONS)


def test_section_coverage_detects_present_and_absent() -> None:
    cov = check_section_coverage("CLAIMS\nfoo\nEVIDENCE\nbar\nLISTS\nnone")
    assert cov["CLAIMS"] and cov["EVIDENCE"] and cov["LISTS"]
    assert not cov["DATES"] and not cov["EVENTS"]
    # REGULATORY/SAFETY/WARNINGS matches any of its aliases.
    assert not cov["REGULATORY_SAFETY_WARNINGS"]
    cov2 = check_section_coverage("SAFETY\nboxed warning")
    assert cov2["REGULATORY_SAFETY_WARNINGS"]


def test_all_required_sections_present() -> None:
    cov = check_section_coverage(_fake_summary())
    assert all(cov.values())
    assert len(REQUIRED_SECTIONS) == 9


def test_split_on_sections_respects_cap_and_is_lossless() -> None:
    big = "\n".join(f"{h}\n" + ("word " * 60) for h in _SECTIONS)
    cap = 50
    pieces = _split_on_sections(big, cap, _ENC)
    assert len(pieces) > 1
    # Every piece is within the cap (allow small tokenizer slack).
    assert all(len(_ENC.encode(p)) <= cap + 5 for p in pieces)
    # Lossless: every section header survives somewhere across the pieces.
    joined = "\n".join(pieces)
    for h in _SECTIONS:
        assert h.split(" /")[0] in joined


def test_split_on_sections_small_input_single_piece() -> None:
    s = "CLAIMS\nshort"
    assert _split_on_sections(s, 1000, _ENC) == [s]


def test_split_on_sections_oversized_single_section() -> None:
    # One section far bigger than the cap must be hard-split, not dropped.
    s = "CLAIMS\n" + ("word " * 200)
    pieces = _split_on_sections(s, 40, _ENC)
    assert len(pieces) >= 2
    assert all(len(_ENC.encode(p)) <= 45 for p in pieces)


def test_token_windows_overlap() -> None:
    text = "token " * 1000
    windows = _token_windows(text, 300, 50, _ENC)
    assert len(windows) == 5  # step 250 over ~1000 tokens
    assert _token_windows("tiny", 300, 50, _ENC) == ["tiny"]


def test_summarizer_prompts_registered() -> None:
    for k in ("evaluated_summary_chunk", "summary_question_gen",
              "summary_evaluate", "summary_revise"):
        assert k in PROMPTS
    s, u = PROMPTS["evaluated_summary_chunk"]("hello source")
    assert "CLAIMS" in s and "hello source" in u


def test_evaluated_result_to_chunks() -> None:
    r = EvaluatedDocSummary(
        path=Path("/tmp/doc.txt"),
        chunk_summaries=["CLAIMS a", "EVIDENCE b", "LISTS c"],
        combined="x", summarized=True,
    )
    chunks = evaluated_result_to_chunks(r)
    assert len(chunks) == 3
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(c.source_name == "doc.txt" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)
    assert chunks[0].text == "CLAIMS a"


def test_evidence_char_cap_configurable() -> None:
    ev = [{"kind": "chunk", "iri": "x", "text": "A" * 3000}]
    assert len(_format_evidence_block(ev)) < 700              # legacy default 600
    long = _format_evidence_block(ev, 1500)
    assert 1400 < len(long) < 1700
    _, user = answer_deep_research("q", ev, 1500)
    assert user.count("A") > 1000                            # cap honored downstream

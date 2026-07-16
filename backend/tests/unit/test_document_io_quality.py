"""Extraction-quality guard: garbled PDF text must never pass silently.

Regression: a valid 144-page PDF with subsetted /Type0 fonts and no /ToUnicode
CMap extracted as ~478k tokens of glyph-id gibberish. Every downstream stage
(summarize -> embed -> entities -> artifacts) processed it without complaint.
"""
from __future__ import annotations

from pathlib import Path

from backend.app.services.document_io import (
    check_extraction_quality,
    text_legibility,
)

# Real sample from the garbled PDF (glyph ids, +44-shifted from ASCII).
GARBLED = (
    "M7<97?A5F?K<9H<9FH<9F9;=GHF5BHG<5J9:=@985@@F9DCFHGF9EI=F98HC69:=@986M-97H=CB"
    "CF8C:H<9-97IF=H=9G L7<5B;97HC:8IF=B;H<9DF9798=B;ACBH<G5B8<5J9699BGI6>97H"
    "HC:=@=B;F9EI=F9A9BHG:CFH<9D5GH85MG" * 4
)
HEALTHY = (
    "The Company reported net revenue growth across all segments in the fiscal "
    "year, and the board of directors has approved a dividend which is payable "
    "to shareholders of record as of the date set forth in this annual report. "
    "We believe that our operations are not materially affected by these risks. "
) * 4


def test_healthy_prose_scores_well_above_floor() -> None:
    assert text_legibility(HEALTHY) > 0.15


def test_garbled_glyph_ids_score_near_zero() -> None:
    assert 0.0 <= text_legibility(GARBLED) < 0.05


def test_short_text_is_not_judged() -> None:
    # Too little text to judge -> -1.0 sentinel, and no warning.
    assert text_legibility("Only a few words here.") == -1.0
    assert check_extraction_quality(Path("x.pdf"), "Only a few words here.") is None


def test_healthy_text_produces_no_warning() -> None:
    assert check_extraction_quality(Path("good.pdf"), HEALTHY) is None


def test_garbled_text_produces_a_warning() -> None:
    w = check_extraction_quality(Path("bad.pdf"), GARBLED)
    assert w is not None
    assert "does not look like language" in w
    assert "bad.pdf" in w


def test_whitespace_free_text_is_not_flagged() -> None:
    """A real 10-K extracted with 0 literal spaces (non-breaking spaces) but is
    perfectly readable. Word-break heuristics flagged it; legibility must not."""
    nbsp_text = HEALTHY.replace(" ", "\xa0")
    assert check_extraction_quality(Path("tsla.pdf"), nbsp_text) is None

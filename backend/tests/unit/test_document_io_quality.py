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


def test_load_document_warns_on_garbled_pdf(tmp_path, capsys, monkeypatch) -> None:
    """The guard must fire on load_document() -- the function the streaming
    loaders (prune-expand, register-documents) actually call per path. It was
    originally only in load_documents(), which those paths never touch, so the
    real ingestion routes were unguarded."""
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    p = tmp_path / "garbled.pdf"
    p.write_bytes(b"%PDF-1.6 fake")
    monkeypatch.setattr(document_io, "read_pdf", lambda _p: GARBLED)

    doc = document_io.load_document(p)
    out = capsys.readouterr().out
    assert "WARNING" in out and "does not look like language" in out
    assert doc.text == GARBLED  # still returned; caller decides


def test_load_document_warns_only_once_per_path(tmp_path, capsys, monkeypatch) -> None:
    """Batched/streamed loaders re-load the same doc; don't spam the log."""
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    p = tmp_path / "garbled2.pdf"
    p.write_bytes(b"%PDF-1.6 fake")
    monkeypatch.setattr(document_io, "read_pdf", lambda _p: GARBLED)

    document_io.load_document(p)
    capsys.readouterr()
    document_io.load_document(p)
    assert "WARNING" not in capsys.readouterr().out


def test_load_document_silent_on_healthy_pdf(tmp_path, capsys, monkeypatch) -> None:
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    p = tmp_path / "good.pdf"
    p.write_bytes(b"%PDF-1.6 fake")
    monkeypatch.setattr(document_io, "read_pdf", lambda _p: HEALTHY)

    document_io.load_document(p)
    assert "WARNING" not in capsys.readouterr().out


def test_preflight_flags_garbled_and_is_silent_on_healthy(tmp_path, capsys, monkeypatch) -> None:
    """Pre-flight must catch unreadable docs BEFORE any paid stage. Previously the
    warning came from load_document() inside the summarizer -- i.e. after table
    extraction + mining had already spent money on the noise."""
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    (tmp_path / "good.txt").write_text(HEALTHY)
    (tmp_path / "bad.txt").write_text(GARBLED)

    flagged = document_io.preflight_documents(tmp_path)
    out = capsys.readouterr().out
    assert flagged == ["bad.txt"]
    assert "bad.txt" in out and "processed as NOISE" in out
    assert "good.txt" not in out


def test_preflight_suppresses_the_later_duplicate_warning(tmp_path, capsys, monkeypatch) -> None:
    """Once preflight warns, ingestion must not repeat it for the same file."""
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    p = tmp_path / "bad.txt"
    p.write_text(GARBLED)

    document_io.preflight_documents(tmp_path)
    capsys.readouterr()
    document_io.load_document(p)
    assert "WARNING" not in capsys.readouterr().out


def test_preflight_ignores_unopenable_files(tmp_path, capsys) -> None:
    """A corrupt file is the loader's problem to report; preflight stays quiet."""
    from backend.app.services import document_io

    document_io._WARNED_UNREADABLE.clear()
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    assert document_io.preflight_documents(tmp_path) == []

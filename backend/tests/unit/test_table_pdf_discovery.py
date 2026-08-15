"""Table extraction finds the same PDFs the document walker does.

Document loading walks recursively (ontology_io.iter_documents uses rglob and
lowercases the suffix); folder-level table extraction used glob("*.pdf"). A
nested corpus therefore ingested every document and extracted zero tables,
silently.
"""

from __future__ import annotations

from pathlib import Path

from backend.app.services.ontology_io import iter_documents
from backend.app.services.table_extract import _find_pdfs


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\n")


def test_finds_pdfs_in_subdirectories(tmp_path: Path) -> None:
    _touch(tmp_path / "top.pdf")
    _touch(tmp_path / "2024" / "nested.pdf")
    _touch(tmp_path / "2024" / "q1" / "deep.pdf")
    names = {p.name for p in _find_pdfs(tmp_path)}
    assert names == {"top.pdf", "nested.pdf", "deep.pdf"}


def test_suffix_match_is_case_insensitive(tmp_path: Path) -> None:
    _touch(tmp_path / "shouty.PDF")
    _touch(tmp_path / "mixed.Pdf")
    assert len(_find_pdfs(tmp_path)) == 2


def test_ignores_non_pdfs(tmp_path: Path) -> None:
    _touch(tmp_path / "a.pdf")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "data.json").write_text("{}")
    assert [p.name for p in _find_pdfs(tmp_path)] == ["a.pdf"]


def test_warns_when_a_tables_run_would_be_a_no_op(tmp_path: Path, capsys) -> None:
    (tmp_path / "notes.txt").write_text("no pdfs here")
    assert _find_pdfs(tmp_path) == []
    assert "WARNING" in capsys.readouterr().out


def test_agrees_with_the_document_walker(tmp_path: Path) -> None:
    """The two must not disagree -- that disagreement is the whole bug."""
    _touch(tmp_path / "2024" / "report.pdf")
    _touch(tmp_path / "2025" / "q2" / "filing.pdf")
    (tmp_path / "2024" / "readme.txt").write_text("x")

    walked = {p for p in iter_documents(tmp_path) if p.suffix.lower() == ".pdf"}
    assert set(_find_pdfs(tmp_path)) == walked

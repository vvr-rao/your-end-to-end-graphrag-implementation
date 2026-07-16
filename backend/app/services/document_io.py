"""Load PDF + TXT documents from a folder into plain text strings."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class LoadedDocument:
    """A document loaded into text plus its origin path."""

    path: Path
    text: str

    @property
    def name(self) -> str:
        return self.path.name


# --------------------------------------------------------------------------- #
# Extraction-quality guard.
# --------------------------------------------------------------------------- #
# A PDF can be structurally valid and render perfectly on screen yet yield pure
# gibberish: if it embeds subsetted /Type0 fonts with /Identity-H encoding and no
# /ToUnicode CMap, the glyph->character mapping simply IS NOT IN THE FILE. pypdf
# then emits raw glyph ids. No extractor can fix this (pdfplumber/pdfminer/PyMuPDF
# all hit the same missing data); only OCR or re-sourcing the file recovers it.
#
# This silently cost a real run: a 144-page annual report produced ~478k tokens of
# garbage that was summarized (~42 LLM windows), embedded, mined for entities, and
# stored -- never matching a single query, because the text was never text.
#
# Detection is by ENGLISH-LIKENESS, not structure, because neither signal alone
# works: a doc can carry a Type0 font with no ToUnicode for a symbol/decorative
# face and still extract fine, while whitespace ratios lie outright (a real 10-K
# scored 0% literal spaces -- it uses non-breaking spaces -- yet reads perfectly).
# Measured separation on the corpus is unambiguous: garbled = 0.3%, healthy = 25-31%.
_STOPWORDS = frozenset({
    "the", "and", "of", "to", "in", "for", "a", "is", "that", "on", "with", "as",
    "by", "are", "we", "our", "or", "from", "this", "be", "which", "has", "have",
    "not", "its", "was",
})
_MIN_WORDS_TO_JUDGE = 50
_LEGIBILITY_FLOOR = 0.05


def text_legibility(text: str) -> float:
    """Share of alphabetic tokens that are common English words.

    Returns -1.0 when there is too little text to judge. Healthy prose lands
    ~0.25-0.31; glyph-id garbage lands ~0.00-0.01.
    """
    words = re.findall(r"[A-Za-z]{2,}", text.lower())
    if len(words) < _MIN_WORDS_TO_JUDGE:
        return -1.0
    return sum(1 for w in words if w in _STOPWORDS) / len(words)


def pdf_fonts_lack_tounicode(path: Path, max_pages: int = 8) -> int:
    """Count /Type0 fonts with no /ToUnicode CMap -- the structural cause of
    unrecoverable extraction. Advisory only (used to explain WHY, not to decide)."""
    n = 0
    try:
        reader = PdfReader(str(path))
        for page in reader.pages[:max_pages]:
            fonts = page.get("/Resources", {}).get("/Font", {})
            if hasattr(fonts, "get_object"):
                fonts = fonts.get_object()
            for key in fonts:
                font = fonts[key].get_object()
                if font.get("/Subtype") == "/Type0" and "/ToUnicode" not in font:
                    n += 1
    except Exception:
        return 0
    return n


def check_extraction_quality(path: Path, text: str) -> str | None:
    """Return a human-readable warning if `text` does not look like language.

    Returns None when the text is fine (or too short to judge). Callers should
    surface the message -- the whole point is that this must never fail silently.
    """
    score = text_legibility(text)
    if score < 0 or score >= _LEGIBILITY_FLOOR:
        return None
    msg = (f"{path.name}: extracted text does not look like language "
           f"(legibility {score * 100:.1f}%, expected >{_LEGIBILITY_FLOOR * 100:.0f}%). "
           f"{len(text):,} chars would be summarized/embedded as noise.")
    if path.suffix.lower() == ".pdf":
        bad = pdf_fonts_lack_tounicode(path)
        if bad:
            msg += (f" Cause: {bad} /Type0 font(s) with no /ToUnicode CMap -- the "
                    f"glyph->text mapping is absent from the file, so NO extractor "
                    f"can recover it. Fix: re-source the PDF or OCR it.")
    return msg


def read_text_file(path: Path) -> str:
    """Read a .txt file with permissive encoding handling."""
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    """Extract text from a PDF. Pages are joined with double newlines so the
    paragraph-aware chunker can detect breaks. PDFs without extractable text
    (scanned images) return empty strings — caller should warn or skip."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return LoadedDocument(path=path, text=read_pdf(path))
    if suffix == ".txt":
        return LoadedDocument(path=path, text=read_text_file(path))
    raise ValueError(f"Unsupported document type: {path} (expected .pdf or .txt)")


def load_documents(documents_dir: Path) -> Iterator[LoadedDocument]:
    """Walk documents_dir and yield each .pdf/.txt as a LoadedDocument.

    Documents whose extracted text is not language are still yielded (the caller
    decides), but they are WARNED about loudly -- processing them costs real money
    and produces artifacts that can never match a query.
    """
    from backend.app.services.ontology_io import iter_documents

    unreadable: list[str] = []
    for path in iter_documents(documents_dir):
        try:
            doc = load_document(path)
        except Exception as exc:
            print(f"[document_io] skipping {path}: {exc}")
            continue
        if not doc.text.strip():
            print(f"[document_io] skipping {path}: no extractable text")
            continue
        warning = check_extraction_quality(path, doc.text)
        if warning:
            unreadable.append(path.name)
            print(f"[document_io] WARNING: {warning}")
        yield doc

    if unreadable:
        print(f"\n[document_io] *** {len(unreadable)} document(s) yielded unreadable "
              f"text and will be processed as noise: {', '.join(unreadable[:5])}"
              f"{' ...' if len(unreadable) > 5 else ''}")
        print("[document_io] *** Re-source or OCR them, or remove them from the "
              "corpus, before paying to summarize/embed them.\n")

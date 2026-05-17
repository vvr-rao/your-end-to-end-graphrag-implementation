"""Load PDF + TXT documents from a folder into plain text strings."""

from __future__ import annotations

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
    """Walk documents_dir and yield each .pdf/.txt as a LoadedDocument."""
    from backend.app.services.ontology_io import iter_documents

    for path in iter_documents(documents_dir):
        try:
            doc = load_document(path)
        except Exception as exc:
            print(f"[document_io] skipping {path}: {exc}")
            continue
        if not doc.text.strip():
            print(f"[document_io] skipping {path}: no extractable text")
            continue
        yield doc

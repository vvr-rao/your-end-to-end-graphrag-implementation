"""Shared helpers for the three downloader CLIs in this folder.

Each tool (dailymed_download.py / websearch_download.py /
financial_report_download.py) uses these so their own files stay focused
on their source-specific scraping logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import requests


# A real contact email is required by SEC EDGAR (they block generic UAs)
# and good practice for any polite scraper. Edit this string before
# running the financial_report_download tool in production.
DEFAULT_CONTACT = "your-personal-knowledge-graph-creator (contact@example.com)"


def safe_filename(text: str, max_len: int = 120) -> str:
    """Slug a string into something safe for a filesystem filename.

    Keeps alphanumerics, hyphens, underscores, and dots. Replaces all
    other characters with `_`, collapses runs of whitespace into `_`,
    strips leading/trailing dots and underscores, and truncates at
    `max_len`.
    """
    text = re.sub(r"[^\w\s\-\.]+", "_", text or "", flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text[:max_len] or "untitled"


def slugify_for_dir(text: str, max_len: int = 60) -> str:
    """Slug a string for a directory name: lowercased, hyphenated.

    Used to derive a default `--output` folder name from the user's
    search term (e.g. "Apple Inc." -> "apple-inc").
    """
    text = re.sub(r"[^\w\s-]+", "", text or "", flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] or "untitled"


def polite_session(contact: str = DEFAULT_CONTACT, timeout: int = 30) -> requests.Session:
    """Return a requests.Session with a real User-Agent header.

    The User-Agent identifies the tool + a contact string. SEC EDGAR
    requires this format; other sources just appreciate it.
    """
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": f"Mozilla/5.0 (compatible; {contact})",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    # Stash the default timeout on the session for callers to consult.
    sess.request_timeout = timeout  # type: ignore[attr-defined]
    return sess


def download_stream(
    session: requests.Session,
    url: str,
    dest_path: Path,
    overwrite: bool = False,
    chunk_size: int = 1024 * 64,
    timeout: int = 60,
    expected_content_types: Iterable[str] | None = None,
) -> Path | None:
    """Download `url` to `dest_path` in chunks.

    Returns the destination path on success, None if skipped (file exists
    and overwrite is False) or if the Content-Type doesn't match
    `expected_content_types` (when provided).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        print(f"  skip (exists): {dest_path.name}")
        return dest_path

    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    if expected_content_types:
        ct = resp.headers.get("Content-Type", "").lower()
        if not any(token in ct for token in expected_content_types):
            print(f"  skip (content-type {ct!r}): {url}")
            return None

    with dest_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
    return dest_path


def default_output_dir(repo_root: Path, prefix: str, search_term: str) -> Path:
    """Build the default destination folder under `source_documents/` for
    a given tool prefix and search term."""
    return repo_root / "source_documents" / f"{prefix}_{slugify_for_dir(search_term)}"

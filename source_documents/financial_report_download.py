"""Free financial-report downloader, sourced from SEC EDGAR.

Searches SEC EDGAR for filings matching a search term (typically a
company name or ticker), then for each matching filing inspects the
document index and downloads any documents with a PDF extension.

Reality check: most US public-company 10-Ks ship as HTML / iXBRL only;
PDF attachments are the EXCEPTION (smaller issuers, some 8-Ks, proxy
statements, foreign 20-F filers occasionally). The tool prints a
per-filing line so you know which ones had PDFs vs. didn't.

For filings without any PDF, pass `--allow-html` to download the
primary HTML 10-K body and convert it to PDF via WeasyPrint. The
result is still a `.pdf` file; on conversion failure the raw `.htm`
is written instead so you always get the filing content. The default
keeps the PDF-only contract (no HTML fetch, no conversion).

Usage:
    # PDF-only (may yield zero files for big iXBRL-only issuers):
    uv run python source_documents/financial_report_download.py \\
      --search "Apple Inc" --output source_documents/finreports_apple --max 5

    # Permissive: PDFs when available, else primary HTML 10-K body:
    uv run python source_documents/financial_report_download.py \\
      --search "Apple Inc" --output source_documents/finreports_apple --max 5 \\
      --allow-html

SEC EDGAR requirements:
  - User-Agent must identify the tool + a real contact email; we set
    that automatically (edit the CONTACT constant below before serious
    use).
  - Rate limit: <= 10 req/sec. We sleep 0.15s between calls.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

from _downloader_common import (
    default_output_dir,
    download_stream,
    polite_session,
    safe_filename,
)


CONTACT = "your-personal-knowledge-graph-creator (contact@example.com)"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_FORMS = "10-K,10-Q,20-F,40-F,8-K"
DEFAULT_SLEEP_SECONDS = 0.15


def _search_filings(session, query: str, forms: str, max_results: int) -> list[dict]:
    """Hit EDGAR's full-text search and return up to `max_results` hits.

    Each hit dict has at least: adsh (accession with dashes), cik
    (zero-padded), file_type, file_date, display_names.
    """
    params = {"q": f'"{query}"', "forms": forms}
    print(f"[edgar] search q={query!r} forms={forms} ...")
    resp = session.get(EDGAR_SEARCH, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])
    out: list[dict] = []
    for h in hits[:max_results]:
        src = h.get("_source", {}) or {}
        adsh = src.get("adsh")
        ciks = src.get("ciks") or []
        if not adsh or not ciks:
            continue
        out.append(
            {
                "adsh": adsh,
                "cik": ciks[0],
                "form": src.get("file_type") or src.get("form"),
                "date": src.get("file_date"),
                "names": src.get("display_names", []),
            }
        )
    print(f"[edgar] {len(out)} matching filing(s)")
    return out


def _filing_index_url(cik: str, adsh: str) -> str:
    """Build the JSON index URL for a single filing. CIK is the int form
    (no leading zeros); the accession in the path has no dashes."""
    cik_int = str(int(cik))
    accession_no_dashes = adsh.replace("-", "")
    return f"{EDGAR_ARCHIVE_BASE}/{cik_int}/{accession_no_dashes}/index.json"


def _extract_pdfs(index_json: dict) -> list[dict]:
    """From a filing's index.json, return the document entries that are
    PDFs (by filename extension OR by described type). Each entry is a
    dict with at least `name` and `description`."""
    out: list[dict] = []
    directory = index_json.get("directory", {}) or {}
    items = directory.get("item", []) or []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if name.lower().endswith(".pdf"):
            out.append(item)
            continue
        # SEC sometimes ships PDFs without .pdf suffix; check the
        # description as a last resort.
        desc = (item.get("description") or "").lower()
        if "pdf" in desc:
            out.append(item)
    return out


_XBRL_FRAGMENT_RE = re.compile(r"^r\d+\.htm?$", re.IGNORECASE)


def _primary_html_doc(index_json: dict) -> dict | None:
    """Return the filing's PRIMARY .htm/.html document, or None.

    Heuristic that matches EDGAR's modern conventions: exclude obvious
    metadata (index pages, FilingSummary, MetaLinks), exhibit documents
    (filename contains 'exhibit'), and XBRL report fragments (R1.htm,
    R12.htm, ...). The first remaining .htm/.html document is returned.
    For an Apple 10-K filing that lands on `aapl-20240928.htm`.
    """
    items = (index_json.get("directory") or {}).get("item") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").lower()
        if not (name.endswith(".htm") or name.endswith(".html")):
            continue
        # EDGAR's own directory-listing pages live alongside the filing's
        # documents. They have names like `<accession>-index.html` and
        # `<accession>-index-headers.html` -- skip both.
        if "index" in name:
            continue
        if name in {"filingsummary.xml", "filingsummary.htm", "filingsummary.html"}:
            continue
        if "exhibit" in name:
            continue
        if _XBRL_FRAGMENT_RE.match(name):
            continue
        return item
    return None


def _filing_doc_url(cik: str, adsh: str, doc_name: str) -> str:
    cik_int = str(int(cik))
    accession_no_dashes = adsh.replace("-", "")
    return f"{EDGAR_ARCHIVE_BASE}/{cik_int}/{accession_no_dashes}/{doc_name}"


def _strip_ixbrl_for_render(html_text: str) -> str:
    """Trim iXBRL wrappers and metadata from EDGAR HTML so WeasyPrint
    spends less time on CSS cascade + DOM walk during rendering.

    On an Apple 10-K (1.5 MB) this drops ~1,162 unknown-namespace
    `<ix:*>` wrapper elements + ~69 `display:none` hidden iXBRL
    metadata divs + one XSD schema link. The visible body text stays
    intact (iXBRL wrappers carry XBRL fact metadata, not visible
    content), so the rendered PDF body is the same as before. Estimated
    3-5x faster on iXBRL filings.

    Concretely:
      - Unwrap every <ix:*> element (keeps inner text/markup).
      - Decompose elements styled `display:none` (whitespace-tolerant).
      - Decompose <link href="*.xsd"> / <link rel="schemaRef">.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in list(soup.find_all(lambda t: t.name and t.name.startswith("ix:"))):
        tag.unwrap()
    for tag in list(
        soup.find_all(
            style=lambda v: bool(v) and "display:none" in v.replace(" ", "").lower()
        )
    ):
        tag.decompose()
    for link in list(soup.find_all("link", href=True)):
        href = (link.get("href") or "").lower()
        rel = link.get("rel") or []
        if href.endswith(".xsd") or "schemaref" in [r.lower() for r in rel]:
            link.decompose()
    return str(soup)


def _blocking_url_fetcher(url, *_, **__):  # type: ignore[no-untyped-def]
    """WeasyPrint url_fetcher callback that short-circuits remote
    fetches. iXBRL filings reference 2 logos and an XSD schema per
    10-K -- skipping those round-trips is a free 1-3 s saving per file.

    `data:` URIs are still resolved via WeasyPrint's default fetcher
    (rare in EDGAR filings but possible).
    """
    if isinstance(url, str) and url.startswith("data:"):
        from weasyprint.urls import default_url_fetcher

        return default_url_fetcher(url)
    return {"string": b"", "mime_type": "application/octet-stream"}


def _html_to_pdf_bytes(html_text: str, base_url: str | None = None) -> bytes:
    """Render an HTML string to PDF bytes via WeasyPrint, with iXBRL
    pre-stripping and remote-fetch blocking for speed.

    `base_url` is still passed so any non-iXBRL relative URLs resolve
    correctly when the cleaned HTML still references them; the custom
    fetcher then returns empty for those. Lazy-imported so the unit
    tests for slug/URL helpers don't pay weasyprint's startup cost.
    """
    import weasyprint  # heavy: brings in cairo/pango via cffi

    cleaned = _strip_ixbrl_for_render(html_text)
    return weasyprint.HTML(
        string=cleaned,
        base_url=base_url,
        url_fetcher=_blocking_url_fetcher,
    ).write_pdf()


def _company_label(filing: dict) -> str:
    names = filing.get("names") or []
    if names:
        # display_names look like "Apple Inc.  (AAPL) (CIK 0000320193)"
        # -- just take the first segment.
        return safe_filename(names[0].split("(")[0].strip(), max_len=60) or "issuer"
    return f"CIK{int(filing.get('cik', '0')):010d}"


def search_and_download(
    query: str,
    output: Path,
    max_results: int,
    forms: str,
    overwrite: bool,
    allow_html: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    session = polite_session(contact=CONTACT)
    filings = _search_filings(session, query, forms=forms, max_results=max_results)
    if not filings:
        print(f"[edgar] no filings found for {query!r}")
        return {"filings": 0, "with_pdf": 0, "downloaded_pdfs": 0, "downloaded_htmls": 0}

    pdfs_downloaded = 0
    htmls_downloaded = 0
    filings_with_pdf = 0
    for fidx, filing in enumerate(filings, start=1):
        adsh = filing["adsh"]
        form = filing.get("form") or "?"
        company = _company_label(filing)
        date = filing.get("date") or "?"
        print(f"  [{fidx}/{len(filings)}] {company} {form} {date} ({adsh})")
        idx_url = _filing_index_url(filing["cik"], adsh)
        try:
            r = session.get(idx_url, timeout=30)
            r.raise_for_status()
            idx = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"    -> index fetch failed: {type(e).__name__}: {e}")
            time.sleep(DEFAULT_SLEEP_SECONDS)
            continue

        pdf_items = _extract_pdfs(idx)
        if pdf_items:
            filings_with_pdf += 1
            for j, item in enumerate(pdf_items, start=1):
                name = item["name"]
                url = _filing_doc_url(filing["cik"], adsh, name)
                slug = safe_filename(name, max_len=80)
                dest = output / f"{fidx:03d}_{company}_{form}_{date}_{j:02d}_{slug}"
                if not dest.name.lower().endswith(".pdf"):
                    dest = dest.with_suffix(".pdf")
                print(f"    PDF: {name}")
                saved = download_stream(
                    session, url, dest, overwrite=overwrite,
                    expected_content_types=("pdf", "octet-stream"),
                )
                if saved is not None:
                    pdfs_downloaded += 1
                time.sleep(DEFAULT_SLEEP_SECONDS)
        elif allow_html:
            primary = _primary_html_doc(idx)
            if primary is None:
                print("    -> no PDF AND no primary HTML found; skipping")
            else:
                name = primary["name"]
                url = _filing_doc_url(filing["cik"], adsh, name)
                slug = safe_filename(name, max_len=80)
                base = output / f"{fidx:03d}_{company}_{form}_{date}_HTML_{slug}"
                pdf_dest = base.with_suffix(".pdf")
                htm_dest = base.with_suffix(".htm")
                if pdf_dest.exists() and not overwrite:
                    print(f"    skip (exists): {pdf_dest.name}")
                    htmls_downloaded += 1
                    time.sleep(DEFAULT_SLEEP_SECONDS)
                    continue
                print(f"    HTML (fallback, converting to PDF): {name}")
                # Fetch + convert. download_stream is bypassed here
                # because weasyprint needs the bytes in memory anyway.
                resp = None
                try:
                    resp = session.get(url, timeout=60)
                    resp.raise_for_status()
                    pdf_bytes = _html_to_pdf_bytes(resp.text, base_url=url)
                    pdf_dest.parent.mkdir(parents=True, exist_ok=True)
                    pdf_dest.write_bytes(pdf_bytes)
                    htmls_downloaded += 1
                    print(f"      -> {pdf_dest.name} ({len(pdf_bytes):,} bytes)")
                except Exception as e:  # noqa: BLE001
                    raw = resp.content if resp is not None else b""
                    if raw:
                        htm_dest.parent.mkdir(parents=True, exist_ok=True)
                        htm_dest.write_bytes(raw)
                        htmls_downloaded += 1
                        print(
                            f"      -> weasyprint conversion failed "
                            f"({type(e).__name__}: {e}); saved raw "
                            f"{htm_dest.name} ({len(raw):,} bytes) instead"
                        )
                    else:
                        print(
                            f"      -> fetch failed ({type(e).__name__}: {e}); "
                            "skipped"
                        )
                time.sleep(DEFAULT_SLEEP_SECONDS)
        else:
            print("    -> no PDF in this filing (rerun with --allow-html for HTML); skipping")
            time.sleep(DEFAULT_SLEEP_SECONDS)

    summary = {
        "filings": len(filings),
        "with_pdf": filings_with_pdf,
        "downloaded_pdfs": pdfs_downloaded,
        "downloaded_htmls": htmls_downloaded,
    }
    msg = (
        f"[edgar] done: {pdfs_downloaded} PDF(s) across {filings_with_pdf}/"
        f"{len(filings)} filings"
    )
    if allow_html:
        msg += f" + {htmls_downloaded} primary HTML 10-K(s) (fallback)"
    msg += f" saved to {output}"
    print(msg)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="financial_report_download",
        description="Download financial-report PDFs from SEC EDGAR filings.",
    )
    p.add_argument("-q", "--search", required=True,
                   help="Company name or ticker to search EDGAR for.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Destination folder. Default: source_documents/finreports_<slug>/")
    p.add_argument("-n", "--max", type=int, default=10, dest="max_results",
                   help="Max number of filings to inspect (default: 10).")
    p.add_argument("--forms", default=DEFAULT_FORMS,
                   help=f"Comma-separated EDGAR forms to search (default: {DEFAULT_FORMS}).")
    p.add_argument("--allow-html", action="store_true", dest="allow_html",
                   help="If a filing has no PDF attachment, fetch the primary "
                        "HTML 10-K body and convert it to PDF via WeasyPrint. "
                        "Most US 10-Ks are iXBRL-only and only produce a "
                        "useful file with this flag. On conversion failure "
                        "the raw .htm is written instead.")
    p.add_argument("--overwrite", action="store_true",
                   help="Redownload even if the destination file already exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output or default_output_dir(repo_root, "finreports", args.search)
    print(f"[edgar] search={args.search!r} -> {output}")
    search_and_download(
        query=args.search,
        output=output,
        max_results=args.max_results,
        forms=args.forms,
        overwrite=args.overwrite,
        allow_html=args.allow_html,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

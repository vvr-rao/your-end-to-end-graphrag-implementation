"""Unit tests for the three CLI downloaders under source_documents/.

Exercise only the pure helpers (URL builders, HTML parsers, slug, filing-
index PDF filter) -- no network calls. Each script's argparse `main()`
is wired up by adding the source_documents/ dir to sys.path before
import so the modules can find their shared `_downloader_common`
sibling.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DOCS = REPO_ROOT / "source_documents"
if str(SOURCE_DOCS) not in sys.path:
    sys.path.insert(0, str(SOURCE_DOCS))

dl_common = importlib.import_module("_downloader_common")
dailymed_mod = importlib.import_module("dailymed_download")
websearch_mod = importlib.import_module("websearch_download")
edgar_mod = importlib.import_module("financial_report_download")


# ---------- Shared helpers ----------------------------------------------------


def test_safe_filename_keeps_alnum_strips_punct_truncates() -> None:
    # Notebook-compatible behavior: each non-allowed character is replaced
    # with `_`, then spaces are also `_`-replaced (no collapse), so
    # `, ` -> `__` and ` ` -> `_`. Trailing dots/underscores are stripped.
    assert dl_common.safe_filename("Hello, World! .pdf") == "Hello__World__.pdf"
    assert dl_common.safe_filename("") == "untitled"
    assert dl_common.safe_filename(None) == "untitled"  # type: ignore[arg-type]
    assert dl_common.safe_filename("a" * 200, max_len=30) == "a" * 30
    assert dl_common.safe_filename("Foo/Bar:baz?qux") == "Foo_Bar_baz_qux"


def test_slugify_for_dir_lowercases_and_hyphenates() -> None:
    assert dl_common.slugify_for_dir("Apple Inc.") == "apple-inc"
    assert dl_common.slugify_for_dir("  diabetes  &  insulin ") == "diabetes-insulin"
    assert dl_common.slugify_for_dir("") == "untitled"
    assert dl_common.slugify_for_dir("foo_bar baz") == "foo-bar-baz"


def test_default_output_dir_sits_under_source_documents() -> None:
    out = dl_common.default_output_dir(REPO_ROOT, "dailymed", "Type 2 diabetes")
    assert out == REPO_ROOT / "source_documents" / "dailymed_type-2-diabetes"


# ---------- DailyMed ----------------------------------------------------------


def test_dailymed_parse_search_results_dedup_by_setid() -> None:
    html = """
    <html><body>
      <a href="/dailymed/drugInfo.cfm?setid=abc-123">Drug ABC</a>
      <a href="/dailymed/drugInfo.cfm?setid=abc-123">Duplicate ABC</a>
      <a href="/dailymed/lookup.cfm?setid=def-456">Drug DEF</a>
      <a href="/some/other/page">Irrelevant</a>
    </body></html>
    """
    client = dailymed_mod.DailyMedClient()
    out = client._parse_search_results(html)
    setids = [item["setid"] for item in out]
    assert setids == ["abc-123", "def-456"]
    assert out[0]["pdf_url"].endswith("setId=abc-123")
    assert out[1]["title"] == "Drug DEF"


# ---------- DuckDuckGo SERP --------------------------------------------------


def test_websearch_extract_ddg_results_unwraps_redirect_links() -> None:
    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A//example.com/page-one">First Result</a>
        <a class="result__snippet">Snippet one</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://example.org/direct">Direct Result</a>
        <a class="result__snippet">Snippet two</a>
      </div>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A//example.com/page-one">Dup of first</a>
      </div>
    </body></html>
    """
    out = websearch_mod._extract_ddg_results(html, max_results=10)
    urls = [r["url"] for r in out]
    # Dedup keeps the first occurrence; both unique URLs survive.
    assert urls == ["https://example.com/page-one", "https://example.org/direct"]
    assert out[0]["title"] == "First Result"
    assert out[0]["snippet"] == "Snippet one"


def test_websearch_extract_text_strips_scripts_and_collapses_blanks() -> None:
    html = """
    <html>
      <head><script>alert('x')</script><style>p{color:red}</style></head>
      <body>
        <p>Hello</p>


        <p>World</p>
        <script>noise()</script>
      </body>
    </html>
    """
    text = websearch_mod._extract_text_from_html(html)
    assert "alert" not in text
    assert "noise" not in text
    assert "color:red" not in text
    # Hello and World survive with at most one blank between them.
    assert text.startswith("Hello")
    assert text.endswith("World")
    assert "\n\n\n" not in text  # no triple blanks


# ---------- EDGAR filing index parse -----------------------------------------


def test_edgar_extract_pdfs_filters_by_extension_and_description() -> None:
    idx = {
        "directory": {
            "item": [
                {"name": "aapl-20231230.htm", "description": "10-K filing"},
                {"name": "aapl-20231230.pdf", "description": "Annual Report PDF"},
                {"name": "exhibit99.txt", "description": "Exhibit"},
                {"name": "ar_2023", "description": "Annual report (PDF format)"},
                {"name": "ar_2023.xml", "description": "XBRL"},
                "not-a-dict-should-be-skipped",
            ]
        }
    }
    pdfs = edgar_mod._extract_pdfs(idx)
    names = [p["name"] for p in pdfs]
    assert names == ["aapl-20231230.pdf", "ar_2023"]


def test_edgar_filing_index_url_strips_dashes_and_int_cik() -> None:
    url = edgar_mod._filing_index_url(cik="0000320193", adsh="0000320193-23-000106")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/index.json"


def test_edgar_primary_html_doc_picks_form_body_over_exhibits_and_fragments() -> None:
    idx = {
        "directory": {
            "item": [
                {"name": "0000320193-24-000123-index-headers.html"},
                {"name": "FilingSummary.xml"},
                {"name": "R1.htm"},
                {"name": "R47.htm"},
                {"name": "a10-kexhibit21109282024.htm", "description": "Subsidiaries exhibit"},
                {"name": "aapl-20240928.htm", "description": "10-K"},  # primary
                {"name": "aapl-20240928.xsd"},
            ]
        }
    }
    primary = edgar_mod._primary_html_doc(idx)
    assert primary is not None
    assert primary["name"] == "aapl-20240928.htm"


def test_edgar_primary_html_doc_returns_none_when_nothing_qualifies() -> None:
    idx = {
        "directory": {
            "item": [
                {"name": "R1.htm"},
                {"name": "exhibit99_1.htm"},
                {"name": "FilingSummary.xml"},
            ]
        }
    }
    assert edgar_mod._primary_html_doc(idx) is None


def test_edgar_filing_doc_url_path_shape() -> None:
    url = edgar_mod._filing_doc_url(
        cik="0000320193", adsh="0000320193-23-000106", doc_name="aapl-20231230.htm"
    )
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20231230.htm"


def test_edgar_html_to_pdf_returns_pdf_bytes() -> None:
    """Smoke the HTML->PDF helper on a tiny synthetic doc. Confirms
    weasyprint is installed AND that our wrapper returns valid PDF
    bytes (magic-header + non-trivial size)."""
    html = (
        "<!doctype html><html><head><title>Test</title></head>"
        "<body><h1>Test Heading</h1><p>Hello world.</p>"
        "<p>This is a second paragraph.</p></body></html>"
    )
    pdf = edgar_mod._html_to_pdf_bytes(html)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-"), pdf[:16]
    assert len(pdf) > 500

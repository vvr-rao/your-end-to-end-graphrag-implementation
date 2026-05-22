"""CLI port of dailymed_pi_download.ipynb.

Searches DailyMed for drug labels matching a condition and downloads
each match's Patient-Information PDF.

Usage:
    uv run python source_documents/dailymed_download.py \\
      --search "diabetes" --output source_documents/dm_diabetes --max 10

Source: https://dailymed.nlm.nih.gov/ (US National Library of Medicine).
Free, no API key, polite scraping with a User-Agent + small sleep.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Allow `python source_documents/dailymed_download.py` to find the
# sibling _downloader_common module: Python prepends the script's dir
# to sys.path automatically, so a bare import works.
from _downloader_common import (
    default_output_dir,
    download_stream,
    polite_session,
    safe_filename,
)


BASE_URL = "https://dailymed.nlm.nih.gov"
SEARCH_URL = f"{BASE_URL}/dailymed/search.cfm"
PDF_URL = f"{BASE_URL}/dailymed/downloadpdffile.cfm"

DEFAULT_SLEEP_SECONDS = 0.5


class DailyMedClient:
    """Lifted from dailymed_pi_download.ipynb (cell 1). Same flow:
    paginated search by condition, dedup by setId, stream-download PDFs.
    Only changes from the notebook: uses the shared `safe_filename` +
    `download_stream` helpers; the per-instance `_safe_filename` method
    is gone (use the shared one).
    """

    def __init__(self, timeout: int = 30, sleep_seconds: float = DEFAULT_SLEEP_SECONDS):
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = polite_session(
            contact="DailyMedPIClient (contact@example.com)", timeout=timeout
        )

    def search_by_condition(
        self,
        condition: str,
        labeltype: str = "human",
        max_results: Optional[int] = None,
    ) -> list[dict]:
        page = 1
        results: list[dict] = []
        seen_setids: set[str] = set()

        while True:
            params = {"labeltype": labeltype, "query": condition, "page": page}
            resp = self.session.get(
                SEARCH_URL, params=params, timeout=self.timeout, allow_redirects=True
            )
            resp.raise_for_status()

            final_url = resp.url
            parsed = urlparse(final_url)
            query_params = parse_qs(parsed.query)

            # Case 1: redirect straight to a single-label page.
            if "drugInfo.cfm" in parsed.path or "lookup.cfm" in parsed.path:
                setid = query_params.get("setid", [None])[0]
                title = self._extract_title_from_label_page(resp.text) or f"label_{setid}"
                if setid and setid not in seen_setids:
                    seen_setids.add(setid)
                    results.append(
                        {
                            "title": title,
                            "setid": setid,
                            "label_url": final_url,
                            "pdf_url": self._build_pdf_url(setid),
                        }
                    )
                break

            # Case 2: a paginated search results page.
            page_results = self._parse_search_results(resp.text)
            new_count = 0
            for item in page_results:
                setid = item["setid"]
                if setid and setid not in seen_setids:
                    seen_setids.add(setid)
                    results.append(item)
                    new_count += 1
                    if max_results is not None and len(results) >= max_results:
                        return results

            if new_count == 0:
                break
            if not self._has_next_page(resp.text):
                break

            page += 1
            time.sleep(self.sleep_seconds)

        return results

    def download_pdfs(
        self,
        results: list[dict],
        output_dir: Path,
        overwrite: bool = False,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for idx, item in enumerate(results, start=1):
            setid = item["setid"]
            title = item["title"]
            file_path = output_dir / f"{idx:03d}_{safe_filename(title)}_{setid}.pdf"
            print(f"  [{idx}/{len(results)}] {title}")
            saved = download_stream(
                self.session,
                item["pdf_url"],
                file_path,
                overwrite=overwrite,
                timeout=self.timeout,
                expected_content_types=("pdf",),
            )
            if saved is not None:
                downloaded.append(saved)
            time.sleep(self.sleep_seconds)
        return downloaded

    def search_and_download(
        self,
        condition: str,
        output_dir: Path,
        labeltype: str = "human",
        max_results: Optional[int] = None,
        overwrite: bool = False,
    ) -> list[Path]:
        results = self.search_by_condition(
            condition=condition, labeltype=labeltype, max_results=max_results
        )
        if not results:
            print(f"No DailyMed results for condition: {condition!r}")
            return []
        print(f"Found {len(results)} matching label(s); downloading PDFs...")
        return self.download_pdfs(results, output_dir=output_dir, overwrite=overwrite)

    # --- HTML parsing helpers (unchanged from the notebook except for
    #     using the shared safe_filename) ---

    def _parse_search_results(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "drugInfo.cfm" not in href and "lookup.cfm" not in href:
                continue
            full_url = urljoin(BASE_URL, href)
            qs = parse_qs(urlparse(full_url).query)
            setid = qs.get("setid", [None])[0]
            if not setid:
                continue
            title = " ".join(a.get_text(" ", strip=True).split()) or f"label_{setid}"
            items.append(
                {
                    "title": title,
                    "setid": setid,
                    "label_url": full_url,
                    "pdf_url": self._build_pdf_url(setid),
                }
            )
        # Dedup within page.
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in items:
            if item["setid"] not in seen:
                seen.add(item["setid"])
                deduped.append(item)
        return deduped

    def _extract_title_from_label_page(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.text.strip():
            return " ".join(soup.title.text.split())
        h1 = soup.find(["h1", "h2"])
        if h1 and h1.get_text(strip=True):
            return " ".join(h1.get_text(" ", strip=True).split())
        return None

    def _has_next_page(self, html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        text_candidates = {"next", "next page", ">"}
        for a in soup.find_all("a"):
            label = " ".join(a.get_text(" ", strip=True).lower().split())
            if label in text_candidates:
                return True
            href = a.get("href", "")
            if "page=" in href and "next" in label:
                return True
        return False

    def _build_pdf_url(self, setid: str) -> str:
        return f"{PDF_URL}?setId={setid}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dailymed_download",
        description=(
            "Download Patient-Information PDFs from DailyMed for drug "
            "labels matching a condition. Source: dailymed.nlm.nih.gov"
        ),
    )
    p.add_argument("-q", "--search", required=True, help="Condition or drug name to search.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Destination folder. Default: source_documents/dailymed_<slug>/")
    p.add_argument("-n", "--max", type=int, default=10, dest="max_results",
                   help="Max number of PDFs to download (default: 10).")
    p.add_argument("--labeltype", default="human", help="DailyMed labeltype (default: human).")
    p.add_argument("--overwrite", action="store_true",
                   help="Redownload even if the destination file already exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output or default_output_dir(repo_root, "dailymed", args.search)
    print(f"[dailymed] search={args.search!r} -> {output}")
    client = DailyMedClient()
    files = client.search_and_download(
        condition=args.search,
        output_dir=output,
        labeltype=args.labeltype,
        max_results=args.max_results,
        overwrite=args.overwrite,
    )
    print(f"[dailymed] done: {len(files)} file(s) saved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

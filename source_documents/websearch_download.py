"""Free web search + plain-text extraction CLI.

Hits DuckDuckGo's JS-free HTML endpoint, takes the top-N results, fetches
each result page, extracts visible text with BeautifulSoup, and saves
one .txt per result into a destination folder. Also writes an `_index.json`
manifest listing every result with its URL + saved filename + char count.

Usage:
    uv run python source_documents/websearch_download.py \\
      --search "GraphRAG ontology techniques" \\
      --output source_documents/ws_graphrag --max 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from _downloader_common import (
    default_output_dir,
    polite_session,
    safe_filename,
    slugify_for_dir,
)


DDG_HTML_ENDPOINT = "https://duckduckgo.com/html/"
DEFAULT_SLEEP_SECONDS = 0.4


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo's HTML SERP wraps each outbound link as
    `/l/?uddg=<percent-encoded-real-url>`. Unwrap that back to the real URL.
    Returns the input unchanged if it's already a regular URL.
    """
    if not href:
        return href
    parsed = urlparse(href)
    if parsed.path == "/l/" or parsed.path.endswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return target
    return href


def _extract_ddg_results(html: str, max_results: int) -> list[dict]:
    """Parse a DuckDuckGo HTML SERP page. Returns up to `max_results` dicts
    of `{title, url, snippet}` deduped by URL."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    seen: set[str] = set()
    # Result containers — DDG's html SERP markup is `<div class="result">`
    # with a `<a class="result__a">` and a snippet `<a class="result__snippet">`.
    for div in soup.find_all("div", class_="result"):
        a = div.find("a", class_="result__a", href=True)
        if a is None:
            continue
        url = _unwrap_ddg_redirect(a["href"].strip())
        if not url or url in seen:
            continue
        # Some DDG redirects produce protocol-less URLs ("//example.com/..."); fix those.
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith(("http://", "https://")):
            continue
        title = " ".join(a.get_text(" ", strip=True).split())
        snippet_el = div.find("a", class_="result__snippet") or div.find(class_="result__snippet")
        snippet = " ".join((snippet_el.get_text(" ", strip=True) if snippet_el else "").split())
        out.append({"title": title or url, "url": url, "snippet": snippet})
        seen.add(url)
        if len(out) >= max_results:
            break
    return out


def _extract_text_from_html(html: str) -> str:
    """Return readable text from a result page's HTML. Strips script/style/
    nav/footer and collapses whitespace runs."""
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("script", "style", "noscript", "iframe", "form"):
        for el in soup.find_all(tag_name):
            el.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    # Collapse consecutive blank lines.
    out_lines: list[str] = []
    blank = False
    for ln in lines:
        if ln:
            out_lines.append(ln)
            blank = False
        elif not blank:
            out_lines.append("")
            blank = True
    return "\n".join(out_lines).strip()


def _search(session, query: str, max_results: int) -> list[dict]:
    # DuckDuckGo's /html/ endpoint returns the SERP on GET (POSTing
    # returns the homepage with no results). Pass q as a query param.
    print(f"[websearch] querying DuckDuckGo for {query!r}...")
    resp = session.get(DDG_HTML_ENDPOINT, params={"q": query}, timeout=30)
    resp.raise_for_status()
    results = _extract_ddg_results(resp.text, max_results=max_results)
    print(f"[websearch] {len(results)} result link(s) collected")
    return results


def _fetch_and_save(
    session,
    result: dict,
    idx: int,
    total: int,
    output: Path,
    overwrite: bool,
) -> dict:
    """Fetch one result URL, extract text, save to a numbered .txt file.
    Returns a dict with metadata for the index file (filename/chars/error)."""
    url = result["url"]
    title = result["title"]
    fname = f"{idx:03d}_{safe_filename(title, max_len=80)}.txt"
    dest = output / fname
    record = {"index": idx, "url": url, "title": title, "filename": fname}
    if dest.exists() and not overwrite:
        record["status"] = "skip_exists"
        record["chars"] = dest.stat().st_size
        print(f"  [{idx}/{total}] skip (exists): {fname}")
        return record
    print(f"  [{idx}/{total}] {url}")
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        record["status"] = "fetch_failed"
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"    -> {record['error']}")
        return record
    # Save body even for non-HTML content-types (we still get_text it).
    text = _extract_text_from_html(resp.text)
    header = (
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Fetched: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n"
        f"Snippet: {result.get('snippet','')}\n"
        f"---\n\n"
    )
    dest.write_text(header + text, encoding="utf-8")
    record["status"] = "ok"
    record["chars"] = dest.stat().st_size
    return record


def search_and_download(query: str, output: Path, max_results: int, overwrite: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    session = polite_session(contact="websearch_download (contact@example.com)")
    results = _search(session, query, max_results=max_results)
    records: list[dict] = []
    for idx, item in enumerate(results, start=1):
        records.append(_fetch_and_save(session, item, idx, len(results), output, overwrite))
        time.sleep(DEFAULT_SLEEP_SECONDS)
    index = {
        "search": query,
        "endpoint": DDG_HTML_ENDPOINT,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "results": records,
    }
    (output / "_index.json").write_text(json.dumps(index, indent=2))
    ok = sum(1 for r in records if r.get("status") in ("ok", "skip_exists"))
    print(f"[websearch] done: {ok}/{len(records)} saved to {output}")
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="websearch_download",
        description="DuckDuckGo HTML search + per-result plain-text extraction.",
    )
    p.add_argument("-q", "--search", required=True, help="Web search query.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Destination folder. Default: source_documents/websearch_<slug>/")
    p.add_argument("-n", "--max", type=int, default=10, dest="max_results",
                   help="Max number of result pages to fetch (default: 10).")
    p.add_argument("--overwrite", action="store_true",
                   help="Redownload even if the destination file already exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output or default_output_dir(repo_root, "websearch", args.search)
    search_and_download(args.search, output, args.max_results, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())

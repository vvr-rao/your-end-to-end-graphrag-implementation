#!/usr/bin/env python3
"""Re-download PDFs that an older websearch_download.py mangled into .txt.

Before commit d671290 (2026-07-12), `_fetch_and_save` ran EVERY response
through the HTML text extractor, so a PDF response had its binary bytes
decoded as text and written as `.txt`. Roughly 44% of the content became
U+FFFD replacement characters, so the bytes are unrecoverable from disk --
renaming to `.pdf` does not work, and the files do not even start with the
`%PDF-` magic bytes.

What IS recoverable: each mangled file kept a `URL: <source>` header line.
This script reads those headers, re-fetches the PDFs, and verifies them.

Safety order -- download and VERIFY first, delete only what was replaced:
  1. Write a manifest of name -> URL (the only provenance record) BEFORE
     touching anything.
  2. Download to a staging dir; accept only files that open as PDFs AND
     whose extracted text passes the legibility floor.
  3. Move verified PDFs into the corpus, then delete the mangled original.
  4. A file whose download fails is left exactly as it was, and reported.

Nothing is deleted unless its replacement is on disk and verified.

    uv run python source_documents/salvage_mangled_pdfs.py <folder> [--apply]

Without --apply it is a dry run: it reports what it would do and downloads
nothing.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.app.services.document_io import text_legibility  # noqa: E402
from source_documents._downloader_common import polite_session  # noqa: E402

LEGIBILITY_FLOOR = 0.05
HEADER_SCAN_BYTES = 4000


def find_mangled(folder: Path) -> list[tuple[Path, str]]:
    """Every .txt holding PDF bytes, paired with its recorded source URL.

    Detection reads RAW BYTES: these files are full of replacement
    characters and invalid sequences, so `grep` and naive text reads both
    misbehave on them.
    """
    out: list[tuple[Path, str]] = []
    for p in sorted(folder.rglob("*.txt")):
        head = p.read_bytes()[:HEADER_SCAN_BYTES].decode("utf-8", "replace")
        if "%PDF" not in head:
            continue
        url = ""
        for line in head.splitlines()[:6]:
            if line.startswith("URL: "):
                url = line[5:].strip()
                break
        out.append((p, url))
    return out


def verify_pdf(raw: bytes) -> tuple[bool, str]:
    """Is this a real PDF whose text extracts as language?

    Both halves matter. A PDF that opens but extracts glyph-id garbage is
    exactly the failure this corpus already suffers from, so accepting one
    here would trade a known-bad file for a differently-bad file.
    """
    if raw[:5] != b"%PDF-":
        return False, "not a PDF (no %PDF- magic bytes)"
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        n_pages = len(reader.pages)
        if n_pages == 0:
            return False, "PDF has no pages"
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages[:8])
    except Exception as exc:
        return False, f"unreadable PDF ({type(exc).__name__})"
    leg = text_legibility(text)
    if leg == -1.0:
        return True, f"{n_pages}p, little extractable text (image-based?) -- accepted"
    if leg < LEGIBILITY_FLOOR:
        return False, f"{n_pages}p but text is garbled (legibility {leg * 100:.1f}%)"
    return True, f"{n_pages}p, legibility {leg * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="actually download and delete; default is a dry run")
    ap.add_argument("--timeout", type=int, default=300,
                help="total seconds allowed per file (this environment's "
                     "proxy adds ~10s of overhead to every request)")
    args = ap.parse_args()

    folder = args.folder
    if not folder.is_dir():
        print(f"not a directory: {folder}")
        return 2

    mangled = find_mangled(folder)
    if not mangled:
        print(f"No mangled PDF-in-txt files found in {folder}")
        return 0

    total_mb = sum(p.stat().st_size for p, _ in mangled) / 1048576
    print(f"Found {len(mangled)} mangled PDF-in-txt file(s), {total_mb:,.1f} MB\n")

    # Provenance manifest FIRST: the URL header is the only record of where
    # each file came from, and deleting the file destroys it.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = folder / f"_salvage_manifest_{stamp}.json"
    manifest = [
        {"file": p.name, "url": u, "size_bytes": p.stat().st_size}
        for p, u in mangled
    ]
    if args.apply:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"provenance manifest: {manifest_path.name}\n")

    no_url = [p.name for p, u in mangled if not u]
    if no_url:
        print(f"!! {len(no_url)} file(s) have no recorded URL and CANNOT be "
              f"re-downloaded; they will be left alone:")
        for n in no_url:
            print(f"     {n}")
        print()

    if not args.apply:
        print("DRY RUN -- nothing downloaded, nothing deleted. Re-run with --apply.\n")
        for p, u in mangled:
            print(f"  {p.stat().st_size / 1048576:>6.1f}MB  {p.name[:52]}")
            print(f"           -> {u or '*** NO URL ***'}")
        return 0

    staging = folder / "_salvage_staging"
    staging.mkdir(exist_ok=True)
    session = polite_session()

    replaced, failed, skipped = [], [], []
    for i, (path, url) in enumerate(mangled, 1):
        label = path.name[:54]
        if not url:
            skipped.append((path.name, "no recorded URL"))
            continue
        print(f"[{i}/{len(mangled)}] {label}", flush=True)
        try:
            # Stream with a TOTAL time budget. requests' timeout only bounds
            # the connect and the gap BETWEEN bytes, so a slow trickle can
            # hang forever -- which is exactly what happened on the first
            # attempt. This environment also adds ~10s of proxy overhead per
            # request, so the budget has to be generous, not tight.
            t0 = time.monotonic()
            resp = session.get(url, timeout=(15, 30), stream=True,
                               allow_redirects=True)
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                if chunk:
                    buf.extend(chunk)
                if time.monotonic() - t0 > args.timeout:
                    raise TimeoutError(
                        f"exceeded {args.timeout}s budget after "
                        f"{len(buf) / 1048576:.1f}MB"
                    )
            resp.close()
            raw = bytes(buf)
            print(f"    fetched {len(raw) / 1048576:>6.1f}MB in "
                  f"{time.monotonic() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"    FAILED download: {type(exc).__name__}: {str(exc)[:80]}",
                  flush=True)
            failed.append((path.name, f"download: {type(exc).__name__}"))
            continue

        ok, why = verify_pdf(raw)
        if not ok:
            print(f"    REJECTED: {why}  (original left in place)")
            failed.append((path.name, why))
            continue

        dest = path.with_suffix(".pdf")
        tmp = staging / dest.name
        tmp.write_bytes(raw)
        tmp.replace(dest)
        path.unlink()  # only now, with a verified replacement on disk
        print(f"    OK {why}  -> {dest.name}", flush=True)
        replaced.append(dest.name)

    try:
        staging.rmdir()
    except OSError:
        pass

    print("\n=== SUMMARY ===")
    print(f"  replaced with real PDFs : {len(replaced)}")
    print(f"  failed (original kept)  : {len(failed)}")
    print(f"  skipped (no URL)        : {len(skipped)}")
    for name, why in failed + skipped:
        print(f"     - {name[:56]}: {why}")
    print(f"\n  provenance manifest kept at {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

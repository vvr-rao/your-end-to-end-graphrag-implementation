"""Subprocess entry point: extract tables from ONE PDF, write JSON-LD,
then exit.

Why this exists
---------------
On constrained hosts (e.g. the dev sandbox at ~2.7 GiB total RAM), running
pdfplumber over many PDFs in-process accumulates heap fragmentation that
Python's allocator does not return to the OS even with explicit
`gc.collect()`. The python process steadily grows toward the kernel's
OOM threshold and gets SIGKILL'd partway through the corpus.

Spawning a fresh `python -m backend.app.services.table_extract_worker
--pdf X.pdf` per PDF guarantees that when the worker exits, the OS
reclaims **all** of its memory unconditionally (heap, file descriptors,
PDF document handles, async loop state, the lot). The orchestrating
parent process never sees the per-PDF memory bloat.

CLI contract
------------
    python -m backend.app.services.table_extract_worker \\
        --pdf /abs/path/to/foo.pdf \\
        --run-cache-dir /abs/path/to/run/tables \\
        [--no-vision] [--max-pages N]

Exit codes:
    0  -- success (or "skipped" with a written manifest)
    1  -- CLI / argument failure
    2  -- unexpected exception inside the extractor

Output:
    Writes the JSON-LD bundle to BOTH `run-cache-dir/<key>.jsonld` AND
    the standard user cache at
    `~/.cache/your-personal-knowledge-graph-creator/tables/<key>.jsonld`.
    The cache key is `sha256(EXTRACTOR_VERSION + doc_bytes)` so re-runs
    over an unchanged PDF are free.

    Prints progress lines to stdout (forwarded by the parent so the
    user sees per-page extraction status).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from backend.app.services.llm_router import LLMRouter
from backend.app.services.table_extract import extract_tables_async


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="table_extract_worker",
        description=(
            "Extract tables from a single PDF in an isolated subprocess. "
            "Writes JSON-LD to the configured caches and exits. Used by "
            "register-documents --tables to keep per-PDF memory state "
            "from accumulating in the parent process."
        ),
    )
    p.add_argument("--pdf", required=True, type=Path,
                   help="Absolute path to one PDF file to process.")
    p.add_argument("--run-cache-dir", type=Path, default=None,
                   help="Optional run-folder cache directory.")
    p.add_argument("--no-vision", dest="use_vision", action="store_false",
                   default=True,
                   help="Disable vision-LLM route (pdfplumber-only).")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Override the per-PDF page cap (smoke tests).")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    router = LLMRouter()
    result = await extract_tables_async(
        args.pdf,
        router=router,
        run_cache_dir=args.run_cache_dir,
        use_vision=args.use_vision,
        max_pages=args.max_pages,
    )
    # extract_tables_async already persists to both caches and prints
    # per-PDF progress + DONE lines via its existing logging.
    # We only need to surface the manifest's success/skipped status
    # so the orchestrator can collect it without parsing logs.
    manifest = result.manifest
    print(
        f"[worker] result: n_tables={manifest.get('n_tables', 0)} "
        f"source={manifest.get('source', '?')} "
        f"cost=${manifest.get('cost_usd', 0):.4f}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 1)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("[worker] interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(
            f"[worker] FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

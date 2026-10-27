"""Two-tier disk cache for `viao:StructuredTable` JSON-LD payloads.

Phase 2a stores extracted tables on disk so Phase 2 ingestion doesn't
pay the extraction cost again. Two tiers:

1. **Run-folder cache** -- `output_ontologies/v<TS>-<subcommand>/tables/`.
   Written by prune-expand alongside `merged.json` etc. Travel-with-the-
   run; useful when you want to inspect what a specific run produced or
   share it with another machine.

2. **User cache** -- `~/.cache/your-end-to-end-graphrag-implementation/tables/`.
   Reused across runs + by Phase 2 `register-documents` for docs that
   never went through a prune-expand. Same hash key shape.

Cache key = `sha256(doc_bytes + EXTRACTOR_VERSION)`. Bumping
`EXTRACTOR_VERSION` (when the prompt or routing logic changes)
invalidates the cache automatically.

The on-disk format is ONE JSON file per document:

    {
      "doc_sha": "<sha256 of doc bytes>",
      "doc_path": "<path of doc at extraction time, informational>",
      "extractor_version": "<EXTRACTOR_VERSION>",
      "tables": [<JSON-LD payload>, <JSON-LD payload>, ...],
      "manifest": {
        "n_tables": N,
        "n_pdfplumber": K,
        "n_vision_llm": M,
        "cost_usd": <float>,
        "wall_seconds": <float>
      },
      "failed_pages": [<page number>, ...]
    }

`failed_pages` lists the pages whose vision request failed. Such an entry is
PARTIAL: ingestion uses its tables (and warns), extraction treats it as a
miss and re-runs. The successful whole-page vision answers are kept per page
under `pages/<key>/p<N>.json`, so the re-run sends only the failed pages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.core.config import cache_root, get_settings


# Bump when the extractor's routing logic or vision prompt changes
# materially. Lower bumps (cosmetic) are OK to skip; on the wire what
# matters is "do the JSON-LD payloads we'd produce now differ".
# p2a-2: vision prompt gained a {"no_table": true} escape + placeholder-drop
# backstop. Without this bump, cached p2a-1 payloads (which contain fabricated
# tables invented from single-header-cell crops) would keep being served and the
# fix would never run.
# p2a-5 (2026-10-23): figure-dense pages with no real table go to whole-page
# vision, and their header-only strips are dropped (borderless financial
# statements). Bumped so cached p2a-4 strips are re-extracted.
EXTRACTOR_VERSION = "p2a-5"


@dataclass
class TableCacheEntry:
    """In-memory representation of a single cache hit/miss result."""
    doc_sha: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    failed_pages: list[int] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.failed_pages)


def vision_models_tag() -> str:
    """`provider:model` of the two vision tasks (crop, whole page), part of
    the cache key. Without it, switching the vision model kept serving the old
    model's tables: 2026-10-24, Qwen3-VL-30B -> 235B on DeepInfra would have
    been a no-op for every PDF already extracted. Empty when the models
    config cannot be read (the key then depends on version + bytes only)."""
    try:
        tasks = get_settings().models_config.get("tasks", {}) or {}
    except Exception:
        return ""
    crop = tasks.get("table_extract_vision") or {}
    page = tasks.get("table_extract_page_vision") or crop
    return "|".join(f"{(t.get('provider') or 'openai').lower()}:{t.get('model') or ''}"
                    for t in (crop, page))


def key_prefix() -> bytes:
    """What the cache key hashes ahead of the document bytes."""
    return f"{EXTRACTOR_VERSION}|{vision_models_tag()}|".encode("utf-8")


def doc_cache_key(doc_bytes: bytes) -> str:
    """Hash of EXTRACTOR_VERSION + vision models + doc-bytes; the cache
    filename stem.

    Tying the version and models into the hash means a bump or a model switch
    does not have to invalidate-then-recompute -- the new hash just doesn't
    match the old file, and a fresh extraction lands at the new key."""
    h = hashlib.sha256()
    h.update(key_prefix())
    h.update(doc_bytes)
    return h.hexdigest()


def doc_sha256(doc_bytes: bytes) -> str:
    """sha256 of just the doc bytes -- used as the canonical `doc_sha`
    in the on-disk JSON. Distinct from `doc_cache_key` so future code
    can still find a doc's tables when looking up by original sha."""
    return hashlib.sha256(doc_bytes).hexdigest()


def user_cache_dir() -> Path:
    """Return (and ensure) the user-level cache directory for tables.

    Mirrors `_doc_summary_cache_dir` in pipeline_llm.py so the cache
    lives under the same root as other caches."""
    root = cache_root() / "tables"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.jsonld"


def load(cache_dir: Path | None, key: str) -> TableCacheEntry | None:
    """Look up a cached entry under `cache_dir / <key>.jsonld`.

    Returns None on cache miss, on unreadable file, or on schema
    mismatch (so a corrupt cache file is treated as a miss). The
    `cache_dir` may be the run-folder cache or the user cache;
    callers usually try the user cache first."""
    if cache_dir is None:
        return None
    path = _cache_file(cache_dir, key)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    doc_sha = payload.get("doc_sha")
    tables = payload.get("tables")
    manifest = payload.get("manifest") or {}
    if not isinstance(doc_sha, str) or not isinstance(tables, list):
        return None
    failed = [int(n) for n in (payload.get("failed_pages") or []) if isinstance(n, int)]
    return TableCacheEntry(doc_sha=doc_sha, tables=tables, manifest=manifest,
                           failed_pages=failed)


def save(
    cache_dir: Path,
    key: str,
    *,
    doc_sha: str,
    doc_path: str | None,
    tables: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    failed_pages: list[int] | None = None,
) -> Path:
    """Write the cache entry. Parent dirs are created on demand."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(cache_dir, key)
    payload = {
        "doc_sha": doc_sha,
        "doc_path": doc_path,
        "extractor_version": EXTRACTOR_VERSION,
        "tables": tables,
        "manifest": manifest or {},
        "failed_pages": sorted(set(failed_pages or [])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def two_tier_load(
    run_cache_dir: Path | None,
    user_cache: Path | None,
    key: str,
) -> TableCacheEntry | None:
    """Run-folder first (most recent + matches the prune-expand we're
    inside of), then user cache. Returns the first hit."""
    hit = load(run_cache_dir, key) if run_cache_dir is not None else None
    if hit is not None:
        return hit
    return load(user_cache, key) if user_cache is not None else None


def _page_answer_file(cache_dir: Path, key: str, page_number: int) -> Path:
    return cache_dir / "pages" / key / f"p{page_number}.json"


def load_page_answer(
    cache_dirs: list[Path], key: str, page_number: int,
) -> tuple[str | None, float] | None:
    """A whole-page vision answer saved by an earlier run of the same PDF and
    models (`key` is the document key), or None."""
    for d in cache_dirs:
        path = _page_answer_file(d, key, page_number)
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(body, dict) and "text" in body:
            return body["text"], float(body.get("cost_usd") or 0.0)
    return None


def save_page_answer(
    cache_dirs: list[Path], key: str, page_number: int, text: str | None, cost_usd: float,
) -> None:
    """Best-effort: keep one page's vision answer so a re-run of a partial
    PDF does not pay for (or wait on) it again."""
    for d in cache_dirs:
        path = _page_answer_file(d, key, page_number)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "cost_usd": cost_usd}), encoding="utf-8")
        except OSError:
            pass

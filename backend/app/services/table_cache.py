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
      }
    }
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Bump when the extractor's routing logic or vision prompt changes
# materially. Lower bumps (cosmetic) are OK to skip; on the wire what
# matters is "do the JSON-LD payloads we'd produce now differ".
EXTRACTOR_VERSION = "p2a-1"


@dataclass
class TableCacheEntry:
    """In-memory representation of a single cache hit/miss result."""
    doc_sha: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def doc_cache_key(doc_bytes: bytes) -> str:
    """Hash of doc-bytes + EXTRACTOR_VERSION; the cache filename stem.

    Tying the version into the hash means version bumps don't have to
    invalidate-then-recompute -- the new hash just doesn't match the
    old file, and a fresh extraction lands at the new key."""
    h = hashlib.sha256()
    h.update(EXTRACTOR_VERSION.encode("utf-8"))
    h.update(b"|")
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
    root = Path.home() / ".cache" / "your-end-to-end-graphrag-implementation" / "tables"
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
    return TableCacheEntry(doc_sha=doc_sha, tables=tables, manifest=manifest)


def save(
    cache_dir: Path,
    key: str,
    *,
    doc_sha: str,
    doc_path: str | None,
    tables: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
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

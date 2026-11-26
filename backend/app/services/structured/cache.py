"""Two-tier cache for table metadata files (same idea as the summary/table caches).

* **User cache** ``cache_root()/table_metadata/<key>.json`` survives across runs
  and is what ``extract-entities --structured-data`` reads after
  ``prune-expand --structured-data`` profiled the tables.
* **Run folder** ``<prune-expand run>/structured/<table>.metadata.json`` keeps a
  readable copy next to the ontology the run produced.

The key is ``sha256(PROFILE_VERSION | file sha256)``. The LLM model is
deliberately NOT part of the key: switching LLM mode between prune-expand and
extract-entities must still hit the cache. The model is recorded inside.

Writes are atomic (temp file + ``os.replace``) and always UTF-8, so a crash or a
concurrent writer never leaves half a file, on Linux, macOS or Windows. A file
that cannot be parsed, or has the wrong shape or key, counts as a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from backend.app.core.config import cache_root

# Bump when the profile prompt or metadata shape changes so old entries miss.
PROFILE_VERSION = "tp-3"   # tp-2: queries are JSON specs compiled to SQL; tp-3: coverage inventory
FORMAT = 2


def user_cache_dir() -> Path:
    d = cache_root() / "table_metadata"
    d.mkdir(parents=True, exist_ok=True)
    return d


def table_key(file_sha: str) -> str:
    return hashlib.sha256(f"{PROFILE_VERSION}|{file_sha}".encode()).hexdigest()


def lookups_key(table_keys: list[str]) -> str:
    joined = "|".join(sorted(table_keys))
    return hashlib.sha256(f"{PROFILE_VERSION}|lookups|{joined}".encode()).hexdigest()


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _valid_table_entry(obj, key: str) -> bool:
    return (isinstance(obj, dict) and obj.get("format") == FORMAT and obj.get("cache_key") == key
            and isinstance(obj.get("table_name"), str) and isinstance(obj.get("columns"), list)
            and obj.get("status") == "complete")


def _valid_lookups_entry(obj, key: str) -> bool:
    return (isinstance(obj, dict) and obj.get("format") == FORMAT and obj.get("cache_key") == key
            and isinstance(obj.get("lookups"), list))


class MetadataCache:
    """Read/write table metadata in the run folder and the user cache."""

    def __init__(self, user_dir: Path | None = None, run_dir: Path | None = None) -> None:
        self.user_dir = Path(user_dir) if user_dir is not None else user_cache_dir()
        self.run_dir = Path(run_dir) if run_dir is not None else None

    # ---- per-table metadata
    def _user_path(self, key: str) -> Path:
        return self.user_dir / f"{key}.json"

    def _run_path(self, table_name: str) -> Path | None:
        return None if self.run_dir is None else self.run_dir / f"{table_name}.metadata.json"

    def get(self, key: str, table_name: str | None = None) -> dict | None:
        """Run folder first (when the table name is known), then the user cache.
        An entry edited by ``table-metadata-import`` lives in the user cache and
        always wins over an older run-folder copy."""
        user = read_json(self._user_path(key))
        user = user if _valid_table_entry(user, key) else None
        run = None
        rp = self._run_path(table_name) if table_name else None
        if rp is not None:
            run = read_json(rp)
            run = run if _valid_table_entry(run, key) else None
        if user and user.get("edited") and not (run and run.get("edited")):
            return user
        return run or user

    def put(self, entry: dict) -> None:
        key = entry["cache_key"]
        atomic_write_json(self._user_path(key), entry)
        rp = self._run_path(entry["table_name"])
        if rp is not None:
            atomic_write_json(rp, entry)

    # ---- cross-table lookups
    def get_lookups(self, key: str) -> dict | None:
        for p in ([self.run_dir / "lookups.json"] if self.run_dir else []) + [self.user_dir / f"lookups_{key}.json"]:
            obj = read_json(p)
            if _valid_lookups_entry(obj, key):
                return obj
        return None

    def put_lookups(self, entry: dict) -> None:
        atomic_write_json(self.user_dir / f"lookups_{entry['cache_key']}.json", entry)
        if self.run_dir is not None:
            atomic_write_json(self.run_dir / "lookups.json", entry)

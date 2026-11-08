"""Per-run ``manifest.json``: what was fetched, from where, and how complete it is.

Rules enforced here rather than left to each source:

* request parameters and URLs are redacted before they are stored -- any
  key that looks like a credential is replaced by ``"***"``;
* a run that hit ``--limit`` / ``--max-pages`` (or any other cap) is marked
  ``truncated`` and its status can never be ``complete`` (it becomes
  ``sample``);
* file paths are stored relative to the snapshot folder, with size,
  SHA-256, delimiter and row counts, and normalized files list the native
  files they were derived from.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

_SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "auth", "credential",
                 "signature", "session")
# Keys that contain a hint word but are not secrets.
_NOT_SECRET = frozenset({"page_token", "pagetoken", "next_page_token", "nextpagetoken",
                         "monkey"})


def is_secret_key(name: str) -> bool:
    low = name.lower().replace("-", "_").replace("[", "_").replace("]", "")
    if low.replace("_", "") in _NOT_SECRET or low in _NOT_SECRET:
        return False
    return any(h in low for h in _SECRET_HINTS)


def redact_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (params or {}).items():
        out[k] = "***" if is_secret_key(str(k)) and v not in (None, "") else v
    return out


def redact_url(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Return ``url`` (+ optional extra params) with credential-looking query
    values replaced by ``***`` and any ``user:pass@`` userinfo removed."""
    parts = urlsplit(url)
    query = [(k, "***" if is_secret_key(k) and v else v)
             for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    for k, v in (params or {}).items():
        if v is None:
            continue
        query.append((str(k), "***" if is_secret_key(str(k)) and v != "" else str(v)))
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    qs = "&".join(quote(k, safe=":,[]._-") if v == "" else
                  f"{quote(k, safe=':,[]._-')}={quote(str(v), safe=':,*._-')}"
                  for k, v in query)
    return urlunsplit((parts.scheme, netloc, parts.path, qs, ""))


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class FileEntry:
    path: str
    role: str                       # native | normalized | extracted | dictionary
    format: str                     # json | csv | tsv | txt | zip | parquet | ...
    bytes: int
    sha256: str
    delimiter: str | None = None
    rows: int | None = None         # data rows (header excluded) / JSON records
    derived_from: list[str] = field(default_factory=list)
    source_url: str | None = None   # redacted
    description: str | None = None


@dataclass
class Manifest:
    source: str
    dataset: str
    mode: str                                   # api | bulk | archive
    params: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    source_release: str | None = None           # provider release / snapshot date
    snapshot_kind: str | None = None            # full | delta | sample | query
    status: str = "running"                     # complete | sample | partial | failed
    truncated: bool = False
    truncation_reasons: list[str] = field(default_factory=list)
    pagination: dict[str, Any] = field(default_factory=dict)
    files: list[FileEntry] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    data_dictionaries: list[str] = field(default_factory=list)
    attribution: str | None = None
    terms_urls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests_made: int = 0
    manifest_version: int = MANIFEST_VERSION

    # ----------------------------------------------------------- building

    def add_file(
        self,
        root: Path,
        path: Path,
        *,
        role: str,
        format: str,
        delimiter: str | None = None,
        rows: int | None = None,
        derived_from: list[Path] | None = None,
        source_url: str | None = None,
        description: str | None = None,
        sha256: str | None = None,
    ) -> FileEntry:
        rel = rel_path(root, path)
        self.files = [f for f in self.files if f.path != rel]
        entry = FileEntry(
            path=rel,
            role=role,
            format=format,
            bytes=path.stat().st_size,
            sha256=sha256 or sha256_file(path),
            delimiter=delimiter,
            rows=rows,
            derived_from=[rel_path(root, p) for p in (derived_from or [])],
            source_url=redact_url(source_url) if source_url else None,
            description=description,
        )
        self.files.append(entry)
        return entry

    def truncate(self, reason: str) -> None:
        self.truncated = True
        if reason not in self.truncation_reasons:
            self.truncation_reasons.append(reason)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def finish(self, status: str | None = None) -> None:
        """Set the final status. ``complete`` downgrades to ``sample`` when
        truncated and to ``partial`` when errors were recorded."""
        if status is None:
            status = "failed" if (self.errors and not self.files) else (
                "partial" if self.errors else "complete")
        if status == "complete" and self.errors:
            status = "partial"
        if status == "complete" and self.truncated:
            status = "sample"
        if self.truncated and self.snapshot_kind in (None, "full"):
            self.snapshot_kind = "sample"
        self.status = status
        self.finished_at = utc_now()

    # ------------------------------------------------------------ persist

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = redact_params(self.params)
        return d

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / MANIFEST_NAME
        part = path.with_name(path.name + ".part")
        with part.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(part, path)
        return path


def rel_path(root: Path, path: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def load_manifest(root: Path) -> dict[str, Any] | None:
    path = root / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verified_entry(prior: dict[str, Any] | None, root: Path, path: Path) -> dict | None:
    """Return the prior manifest's entry for ``path`` if the file on disk still
    has the recorded size and SHA-256 (i.e. it was completed and is intact)."""
    if not prior or not path.exists():
        return None
    rel = rel_path(root, path)
    for entry in prior.get("files", []):
        if entry.get("path") == rel:
            if (entry.get("bytes") == path.stat().st_size
                    and entry.get("sha256") == sha256_file(path)):
                return entry
            return None
    return None

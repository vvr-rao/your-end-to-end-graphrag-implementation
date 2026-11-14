"""Find the tables in a structured-data folder and give each a Postgres name.

Two layouts are understood:

* **Downloader snapshots** (``structured_download.py`` / ``supplementary_download.py``
  / ``build_crosswalk.py``): ``<dataset>/<snapshot>/manifest.json`` with the
  tables under ``normalized/``. Only the newest snapshot of each dataset is used
  and only its ``normalized/`` files (``native/`` holds the raw originals).
* **Plain folders**: every supported file, recursively.

Table names follow the file name (``prices.csv`` -> ``prices``). When two files
share a name, or the name is one of the graph's own tables (``entities.csv``),
the source folder is put in front (``fred_observations``, ``gleif_entities``).
Names are computed over the WHOLE folder before any ``--structured-limit`` so
they do not change between a smoke test and a full run.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet"}
ARCHIVE_SUFFIXES = {".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z"}
MAX_IDENT = 63  # Postgres NAMEDATALEN - 1

# Tables the graph itself owns in the ``graphrag`` schema. Loaded data tables
# must never take one of these names.
_STATIC_RESERVED = {
    "alembic_version", "conversations", "conversation_turns", "ontology_classes",
    "ontology_object_properties", "ontology_data_properties", "ontology_instances",
    "documents", "chunks", "graph_relationships", "graph_version_state",
    "intelligence_artifacts", "artifact_sources", "retrieval_runs",
    "retrieval_evidence", "entities", "time_instances", "term_aliases",
    "term_alias_judgements",
}


def reserved_table_names() -> set[str]:
    """Graph table names (static list + whatever the ORM declares today)."""
    names = set(_STATIC_RESERVED)
    try:  # the ORM may gain tables later; pick them up without a code change
        from backend.app.db import models  # noqa: F401
        from backend.app.db.base import Base

        names |= {t.name for t in Base.metadata.tables.values()}
    except Exception:  # pragma: no cover - import problems must not break discovery
        pass
    return names


@dataclass
class TableSource:
    path: Path
    table_name: str
    fmt: str                      # csv | json | jsonl | parquet
    qualifiers: list[str]         # folder components used to disambiguate names
    rel_path: str                 # path relative to the scanned root (posix)
    dataset: str = ""             # "<source>/<dataset>" for snapshot layouts
    snapshot: str = ""
    file_sha: str = ""
    size_bytes: int = 0
    base_name: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {"path": str(self.path), "table_name": self.table_name, "fmt": self.fmt,
                "rel_path": self.rel_path, "dataset": self.dataset, "snapshot": self.snapshot,
                "file_sha": self.file_sha, "size_bytes": self.size_bytes}


@dataclass
class Discovery:
    tables: list[TableSource]
    skipped: list[tuple[str, str]]  # (rel_path, reason)


def sanitize_identifier(text: str) -> str:
    """Lower-case snake identifier usable unquoted in Postgres."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "t"
    if s[0].isdigit():
        s = f"t_{s}"
    return s


def fit_identifier(name: str) -> str:
    """Cap at 63 bytes; long names keep a stable hash suffix so they stay unique."""
    if len(name.encode("utf-8")) <= MAX_IDENT:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: MAX_IDENT - 9].rstrip('_')}_{digest}"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fmt_for(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return "parquet"
    if suf in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suf == ".json":
        return "json"
    return "csv"


def _skip_reason(path: Path) -> str | None:
    name = path.name.lower()
    if path.suffix.lower() in ARCHIVE_SUFFIXES:
        return "archive (extract it first)"
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return None  # silently ignored (manifests, READMEs, ...)
    if ".rejects." in name:
        return "rejected-rows file written by the downloader"
    if name in {"manifest.json", "yegi_manifest.json"}:
        return None
    if name.startswith(".") or name.endswith(".part"):
        return None
    return ""


def discover_tables(folder: Path) -> Discovery:
    """List every table under ``folder`` with a stable, unique Postgres name."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"structured-data folder not found: {folder}")

    manifests = sorted(folder.rglob("manifest.json"))
    snapshot_dirs = {m.parent for m in manifests if (m.parent / "normalized").is_dir()}
    by_dataset: dict[Path, list[Path]] = defaultdict(list)
    for snap in snapshot_dirs:
        by_dataset[snap.parent].append(snap)
    newest = {max(snaps, key=lambda p: p.name) for snaps in by_dataset.values()}

    found: list[TableSource] = []
    skipped: list[tuple[str, str]] = []

    def _add(path: Path, qualifiers: list[str], dataset: str = "", snapshot: str = "") -> None:
        rel = path.relative_to(folder).as_posix()
        reason = _skip_reason(path)
        if reason is None:
            return
        if reason:
            skipped.append((rel, reason))
            return
        found.append(TableSource(
            path=path, table_name="", fmt=_fmt_for(path), qualifiers=qualifiers,
            rel_path=rel, dataset=dataset, snapshot=snapshot,
            size_bytes=path.stat().st_size,
            base_name=sanitize_identifier(path.stem),
        ))

    for snap in sorted(newest):
        ds_parts = list(snap.parent.relative_to(folder).parts)
        norm = snap / "normalized"
        for p in sorted(norm.rglob("*")):
            if p.is_file():
                inner = list(p.parent.relative_to(norm).parts)
                _add(p, [sanitize_identifier(x) for x in ds_parts + inner],
                     dataset="/".join(ds_parts), snapshot=snap.name)

    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if any(s == p or s in p.parents for s in snapshot_dirs):
            continue  # snapshot files are handled above (newest snapshot only)
        if p.parent in snapshot_dirs or p.name == "manifest.json":
            continue
        parts = list(p.parent.relative_to(folder).parts)
        _add(p, [sanitize_identifier(x) for x in parts])

    assign_table_names(found, reserved_table_names())
    for t in found:
        t.file_sha = file_sha256(t.path)
    found.sort(key=lambda t: t.table_name)
    return Discovery(tables=found, skipped=skipped)


def assign_table_names(tables: list[TableSource], reserved: set[str]) -> None:
    """Name = file stem; on a clash or a reserved name, prefix folder parts
    (source first, then deeper folders) until unique; last resort ``_2``..."""
    groups: dict[str, list[TableSource]] = defaultdict(list)
    for t in tables:
        groups[t.base_name].append(t)
    taken: set[str] = set()
    plain = {b for b, g in groups.items() if len(g) == 1 and b not in reserved}
    for b in plain:
        groups[b][0].table_name = fit_identifier(b)
        taken.add(groups[b][0].table_name)
    for base, group in sorted(groups.items()):
        if base in plain:
            continue
        depth = 1
        max_depth = max((len(t.qualifiers) for t in group), default=0)
        while True:
            names = [fit_identifier("_".join([*t.qualifiers[:depth], base])) for t in group]
            clash = (len(set(names)) < len(names) or any(n in reserved or n in taken for n in names))
            if not clash or depth >= max_depth:
                break
            depth += 1
        for t, n in zip(group, names, strict=True):
            cand, i = n, 2
            while cand in taken or cand in reserved:
                cand = fit_identifier(f"{n}_{i}")
                i += 1
            t.table_name = cand
            taken.add(cand)

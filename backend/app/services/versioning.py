"""Version-folder naming + manifest assembly.

Each CLI subcommand writes a fresh folder under output_ontologies/ named:
    v<UTC-timestamp>-<subcommand>
e.g. v20260518-141232Z-merge.

The manifest.json inside captures everything needed to reconstruct lineage:
operation, parent_version (if any), input ontology files + sha256, document
files + sha256, model IDs/versions for any LLM stages, and counts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def new_version_dir(output_root: Path, subcommand: str) -> Path:
    """Compute the next version folder name and create it. UTC timestamp at second
    resolution is unique enough for human-driven runs; the subcommand suffix makes
    the directory legible at-a-glance."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"v{ts}-{subcommand}"
    target = output_root / name
    if target.exists():
        for i in range(1, 100):
            candidate = output_root / f"{name}-{i}"
            if not candidate.exists():
                target = candidate
                break
    target.mkdir(parents=True, exist_ok=False)
    return target


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of_file(path),
    }


def write_manifest(
    version_dir: Path,
    *,
    operation: str,
    parent_version: Path | None = None,
    input_ontologies: list[Path] | None = None,
    input_documents: list[Path] | None = None,
    model_ids: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "operation": operation,
        "created_at": datetime.now(UTC).isoformat(),
        "version_dir": version_dir.name,
        "parent_version": str(parent_version.resolve()) if parent_version else None,
        "input_ontologies": [file_provenance(p) for p in (input_ontologies or [])],
        "input_documents": [file_provenance(p) for p in (input_documents or [])],
        "model_ids": model_ids or {},
    }
    if extra:
        manifest.update(extra)
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def write_stats(version_dir: Path, stats: dict[str, Any]) -> None:
    (version_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def ensure_audit_log(version_dir: Path) -> Path:
    """Create an empty llm_audit.jsonl so downstream tooling can rely on the file
    existing even for LLM-free operations (merge)."""
    log = version_dir / "llm_audit.jsonl"
    log.touch(exist_ok=True)
    return log

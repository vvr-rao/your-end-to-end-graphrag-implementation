"""File discovery + per-file parsing + LRU cache for the Dash visualizer.

Every parse runs in its own owlready2.World() so switching between
dropdown selections never bleeds entities across files (the same
isolation Commit 1 added to import_ontologies for multi-file merges).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from owlready2 import World

from backend.app.helpers.ontology_parsing import extract_ontology_to_dicts

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output_ontologies"
SOURCE_ROOT = REPO_ROOT / "source_ontologies"

# Files larger than this trigger a "too large to load on this machine" message
# instead of an attempted parse. DRON (670MB) is the motivating example -- it
# OOM-kills the owlready2 parser on machines with <4GB RAM.
MAX_PARSE_BYTES = 200 * 1024 * 1024  # 200 MB

OWL_SUFFIXES = (".owl", ".rdf", ".ttl")


@dataclass(frozen=True)
class DiscoveredFile:
    label: str       # short display name (e.g. "skos.rdf", "v...-merge/merged.owl")
    path: Path
    group: str       # "Generated" or "Source Ontologies"
    size_bytes: int


def discover_owl_files() -> list[DiscoveredFile]:
    """Walk output_ontologies/ + source_ontologies/ for ontology files.

    Returns a flat list sorted by group then label. The Dash dropdown
    groups by `DiscoveredFile.group` to separate generated outputs
    from user-supplied inputs.
    """
    out: list[DiscoveredFile] = []

    # Generated outputs: every v*/merged.owl under output_ontologies/.
    if OUTPUT_ROOT.exists():
        for version_dir in sorted(OUTPUT_ROOT.iterdir()):
            if not version_dir.is_dir():
                continue
            merged = version_dir / "merged.owl"
            if merged.exists():
                out.append(DiscoveredFile(
                    label=f"{version_dir.name}/merged.owl",
                    path=merged,
                    group="Generated",
                    size_bytes=merged.stat().st_size,
                ))

    # Source ontologies: any .owl/.rdf/.ttl under source_ontologies/.
    if SOURCE_ROOT.exists():
        for p in sorted(SOURCE_ROOT.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in OWL_SUFFIXES:
                continue
            try:
                rel = p.relative_to(SOURCE_ROOT)
            except ValueError:
                rel = p
            out.append(DiscoveredFile(
                label=str(rel),
                path=p,
                group="Source Ontologies",
                size_bytes=p.stat().st_size,
            ))

    return out


@lru_cache(maxsize=8)
def _load_ontology_cached(path_str: str, mtime: float, size: int) -> dict:
    """Cache-friendly inner load. Key includes mtime+size so editing or
    regenerating the file busts the cache. Each call uses a fresh
    owlready2.World() -- no shared state across files.
    """
    _ = mtime, size  # part of the cache key
    p = Path(path_str)
    world = World()
    return extract_ontology_to_dicts(
        str(p),
        load_imported=False,
        local_only=True,
        local_ontology_dir=str(p.parent),
        world=world,
    )


def load_ontology(path: Path) -> dict | None:
    """Load an ontology file into the canonical dict-of-dicts form.

    Returns None if the file is larger than MAX_PARSE_BYTES (signals
    the UI to show a "too large to load" message instead of hanging).
    Otherwise returns:
        {classes_dict, object_properties_dict,
         data_properties_dict, instances_dict}
    """
    if not path.exists():
        return None
    stat = path.stat()
    if stat.st_size > MAX_PARSE_BYTES:
        return None
    return _load_ontology_cached(str(path), stat.st_mtime, stat.st_size)


def compute_stats(loaded: dict) -> dict[str, int]:
    """Per-type entity counts. Rendered in the sidebar."""
    return {
        "classes": len(loaded.get("classes_dict", {})),
        "object_properties": len(loaded.get("object_properties_dict", {})),
        "data_properties": len(loaded.get("data_properties_dict", {})),
        "instances": len(loaded.get("instances_dict", {})),
    }


def is_too_large(path: Path) -> bool:
    """True if path is large enough that load_ontology() will refuse it."""
    try:
        return path.stat().st_size > MAX_PARSE_BYTES
    except OSError:
        return True


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _norm_path(p: str | os.PathLike) -> Path:
    return Path(p).expanduser().resolve()


def resolve_custom_path(text: str) -> Path | None:
    """Resolve a user-typed path against the repo root. Returns None on
    obvious bad input (empty, missing file, wrong suffix)."""
    if not text:
        return None
    candidate = _norm_path(text)
    if not candidate.exists() or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in OWL_SUFFIXES:
        return None
    return candidate

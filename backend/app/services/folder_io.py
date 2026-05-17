"""Read and write the canonical version-folder layout.

A version folder always contains:
    merged.owl       Protégé-readable RDF/XML
    merged.json      canonical dict-of-dicts (fast re-load path)
    manifest.json    operation, parent_version, inputs, model IDs
    stats.json       class/property/instance counts
    llm_audit.jsonl  per-LLM-call audit (empty for `merge`)
    prompt_outputs/  raw LLM responses, optional (omitted for `merge`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MERGED_JSON = "merged.json"
MERGED_OWL = "merged.owl"
MANIFEST = "manifest.json"
STATS = "stats.json"


def load_version_folder(folder: Path) -> dict[str, Any]:
    """Read the canonical loaded-ontology dict from a version folder.

    Prefers merged.json (fast). Caller can force a re-import from merged.owl by
    passing the path directly to ontology_io.load_ontology_files instead.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a version folder: {folder}")
    json_path = folder / MERGED_JSON
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} missing. Re-run the parent operation, or use --use-owl to reparse {MERGED_OWL}."
        )
    return json.loads(json_path.read_text())


def write_merged_json(version_dir: Path, loaded_ontology: dict[str, Any]) -> Path:
    path = version_dir / MERGED_JSON
    path.write_text(json.dumps(loaded_ontology, indent=2, default=str))
    return path


def read_manifest(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def count_entities(loaded_ontology: dict[str, Any]) -> dict[str, int]:
    return {
        "classes": len(loaded_ontology.get("classes_dict", {})),
        "object_properties": len(loaded_ontology.get("object_properties_dict", {})),
        "data_properties": len(loaded_ontology.get("data_properties_dict", {})),
        "instances": len(loaded_ontology.get("instances_dict", {})),
    }

"""Version-dir naming + manifest writing."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.services import versioning


def test_new_version_dir_has_subcommand_suffix(tmp_path: Path) -> None:
    d = versioning.new_version_dir(tmp_path, "merge")
    assert d.parent == tmp_path
    assert d.is_dir()
    assert d.name.endswith("-merge")
    assert d.name.startswith("v")


def test_new_version_dir_avoids_collision(tmp_path: Path) -> None:
    a = versioning.new_version_dir(tmp_path, "merge")
    b = versioning.new_version_dir(tmp_path, "merge")
    assert a != b
    assert a.is_dir() and b.is_dir()


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "in.owl"
    f.write_text("<rdf:RDF/>")
    d = versioning.new_version_dir(tmp_path, "merge")
    manifest = versioning.write_manifest(
        d,
        operation="merge",
        input_ontologies=[f],
        model_ids={"task_x": "openai/gpt-4.1"},
    )
    on_disk = json.loads((d / "manifest.json").read_text())
    assert on_disk["operation"] == "merge"
    assert on_disk["input_ontologies"][0]["name"] == "in.owl"
    assert on_disk["input_ontologies"][0]["sha256"]
    assert on_disk["model_ids"]["task_x"] == "openai/gpt-4.1"
    assert manifest == on_disk


def test_ensure_audit_log_creates_empty_file(tmp_path: Path) -> None:
    d = versioning.new_version_dir(tmp_path, "merge")
    log = versioning.ensure_audit_log(d)
    assert log.exists()
    assert log.read_text() == ""

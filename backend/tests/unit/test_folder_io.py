"""Read/write a version folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services import folder_io, versioning


def test_write_and_load_merged_json(tmp_path: Path) -> None:
    d = versioning.new_version_dir(tmp_path, "merge")
    loaded = {
        "classes_dict": {"http://x/A": {"iri": "http://x/A", "labels": ["A"]}},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    folder_io.write_merged_json(d, loaded)
    re_read = folder_io.load_version_folder(d)
    assert re_read == loaded


def test_load_version_folder_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        folder_io.load_version_folder(tmp_path / "missing")


def test_load_version_folder_missing_json_raises(tmp_path: Path) -> None:
    d = versioning.new_version_dir(tmp_path, "merge")
    with pytest.raises(FileNotFoundError):
        folder_io.load_version_folder(d)


def test_count_entities() -> None:
    loaded = {
        "classes_dict": {"a": {}, "b": {}, "c": {}},
        "object_properties_dict": {"p": {}},
        "data_properties_dict": {},
        "instances_dict": {"i1": {}, "i2": {}},
    }
    assert folder_io.count_entities(loaded) == {
        "classes": 3,
        "object_properties": 1,
        "data_properties": 0,
        "instances": 2,
    }

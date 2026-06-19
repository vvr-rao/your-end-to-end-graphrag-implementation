"""Two-tier cache for StructuredTable JSON-LD payloads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.services import table_cache


def _tables() -> list[dict]:
    return [{"@id": "viao:t1", "@type": "viao:StructuredTable", "columns": [], "rows": []}]


def test_doc_cache_key_includes_version(tmp_path: Path, monkeypatch) -> None:
    h1 = table_cache.doc_cache_key(b"hello")
    monkeypatch.setattr(table_cache, "EXTRACTOR_VERSION", "different-version")
    h2 = table_cache.doc_cache_key(b"hello")
    assert h1 != h2  # changing the extractor version invalidates the key


def test_doc_sha256_is_version_independent(monkeypatch) -> None:
    h1 = table_cache.doc_sha256(b"hello")
    monkeypatch.setattr(table_cache, "EXTRACTOR_VERSION", "different-version")
    h2 = table_cache.doc_sha256(b"hello")
    assert h1 == h2


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    key = table_cache.doc_cache_key(b"some-doc-bytes")
    sha = table_cache.doc_sha256(b"some-doc-bytes")
    table_cache.save(
        tmp_path, key,
        doc_sha=sha,
        doc_path="/tmp/doc.pdf",
        tables=_tables(),
        manifest={"n_tables": 1, "cost_usd": 0.0},
    )
    entry = table_cache.load(tmp_path, key)
    assert entry is not None
    assert entry.doc_sha == sha
    assert entry.tables == _tables()
    assert entry.manifest["n_tables"] == 1


def test_load_returns_none_on_miss(tmp_path: Path) -> None:
    assert table_cache.load(tmp_path, "no-such-key") is None
    assert table_cache.load(None, "anything") is None


def test_load_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    key = "corrupted"
    (tmp_path / f"{key}.jsonld").write_text("not json {")
    assert table_cache.load(tmp_path, key) is None


def test_load_returns_none_on_shape_mismatch(tmp_path: Path) -> None:
    # File parses as JSON but lacks the required `doc_sha` + `tables` keys.
    key = "broken-shape"
    (tmp_path / f"{key}.jsonld").write_text(json.dumps({"foo": "bar"}))
    assert table_cache.load(tmp_path, key) is None


def test_two_tier_load_prefers_run_cache(tmp_path: Path) -> None:
    run = tmp_path / "run"
    user = tmp_path / "user"
    run.mkdir()
    user.mkdir()
    key = table_cache.doc_cache_key(b"d")
    sha = table_cache.doc_sha256(b"d")

    table_cache.save(run, key, doc_sha=sha, doc_path=None,
                     tables=[{"@id": "from-run"}], manifest={})
    table_cache.save(user, key, doc_sha=sha, doc_path=None,
                     tables=[{"@id": "from-user"}], manifest={})

    hit = table_cache.two_tier_load(run, user, key)
    assert hit is not None
    assert hit.tables == [{"@id": "from-run"}]


def test_two_tier_load_falls_back_to_user(tmp_path: Path) -> None:
    run = tmp_path / "run"
    user = tmp_path / "user"
    run.mkdir()
    user.mkdir()
    key = table_cache.doc_cache_key(b"d")
    sha = table_cache.doc_sha256(b"d")

    # Only the user tier has the entry.
    table_cache.save(user, key, doc_sha=sha, doc_path=None,
                     tables=[{"@id": "from-user"}], manifest={})

    hit = table_cache.two_tier_load(run, user, key)
    assert hit is not None
    assert hit.tables == [{"@id": "from-user"}]


def test_two_tier_load_returns_none_when_both_miss(tmp_path: Path) -> None:
    hit = table_cache.two_tier_load(
        tmp_path / "no-run-dir",
        tmp_path / "no-user-dir",
        "x",
    )
    assert hit is None

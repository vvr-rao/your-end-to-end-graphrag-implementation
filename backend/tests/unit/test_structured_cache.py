"""structured.cache: keys, two tiers, atomic UTF-8 writes, corrupt entries (all OSes)."""

from __future__ import annotations

import json

from backend.app.services.structured import cache as tc


def _entry(key: str, table: str = "prices", **kw) -> dict:
    return {"format": tc.FORMAT, "status": "complete", "cache_key": key, "table_name": table,
            "columns": [{"name": "commodity", "description": "Café — naïve ✓"}], "edited": False, **kw}


def test_key_depends_on_bytes_and_version_not_model(monkeypatch):
    k1 = tc.table_key("a" * 64)
    assert k1 == tc.table_key("a" * 64) and len(k1) == 64
    assert tc.table_key("b" * 64) != k1
    monkeypatch.setattr(tc, "PROFILE_VERSION", "tp-999")
    assert tc.table_key("a" * 64) != k1
    assert tc.lookups_key(["x", "y"]) == tc.lookups_key(["y", "x"])


def test_user_cache_dir_follows_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "my cache ü"))
    d = tc.user_cache_dir()
    assert d == tmp_path / "my cache ü" / "table_metadata" and d.is_dir()


def test_round_trip_utf8_and_no_temp_files(tmp_path):
    c = tc.MetadataCache(user_dir=tmp_path / "user dir", run_dir=tmp_path / "run ü" / "structured")
    e = _entry("k1")
    c.put(e)
    assert c.get("k1", "prices") == e
    assert c.get("k1") == e                     # user tier without a table name
    raw = (tmp_path / "user dir" / "k1.json").read_bytes()
    assert "Café — naïve ✓".encode() in raw and b"\r\n" not in raw
    assert (tmp_path / "run ü" / "structured" / "prices.metadata.json").exists()
    leftovers = [p for p in tmp_path.rglob("*") if ".tmp." in p.name]
    assert leftovers == []


def test_run_tier_preferred_then_user_fallback(tmp_path):
    user, run = tmp_path / "u", tmp_path / "r"
    tc.MetadataCache(user_dir=user, run_dir=run).put(_entry("k1", description="run copy"))
    tc.atomic_write_json(user / "k1.json", _entry("k1", description="user copy"))
    assert tc.MetadataCache(user, run).get("k1", "prices")["description"] == "run copy"
    (run / "prices.metadata.json").unlink()
    assert tc.MetadataCache(user, run).get("k1", "prices")["description"] == "user copy"


def test_edited_user_entry_wins_over_run_copy(tmp_path):
    user, run = tmp_path / "u", tmp_path / "r"
    tc.MetadataCache(user_dir=user, run_dir=run).put(_entry("k1", description="profiled"))
    tc.atomic_write_json(user / "k1.json", _entry("k1", description="human", edited=True))
    assert tc.MetadataCache(user, run).get("k1", "prices")["description"] == "human"


def test_corrupt_wrong_key_or_partial_is_a_miss(tmp_path):
    c = tc.MetadataCache(user_dir=tmp_path)
    (tmp_path / "k1.json").write_bytes(b'{"format": 1, "cache_key": "k1", "tab')   # truncated
    assert c.get("k1") is None
    tc.atomic_write_json(tmp_path / "k2.json", _entry("other"))
    assert c.get("k2") is None
    tc.atomic_write_json(tmp_path / "k3.json", {**_entry("k3"), "status": "partial"})
    assert c.get("k3") is None
    tc.atomic_write_json(tmp_path / "k4.json", ["not", "a", "dict"])
    assert c.get("k4") is None


def test_lookups_two_tier(tmp_path):
    c = tc.MetadataCache(user_dir=tmp_path / "u", run_dir=tmp_path / "r")
    e = {"format": tc.FORMAT, "cache_key": "L", "lookups": [{"id": "a"}]}
    c.put_lookups(e)
    assert c.get_lookups("L") == e
    assert tc.MetadataCache(user_dir=tmp_path / "u").get_lookups("L") == e
    assert c.get_lookups("other") is None


def test_overwrite_replaces_existing_file(tmp_path):
    p = tmp_path / "x.json"
    tc.atomic_write_json(p, {"a": 1})
    tc.atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}

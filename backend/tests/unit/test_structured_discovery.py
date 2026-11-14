"""structured.discovery: layouts, newest snapshot, table naming (no DB, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.structured.discovery import (
    MAX_IDENT,
    discover_tables,
    file_sha256,
    fit_identifier,
    sanitize_identifier,
)


def _w(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)  # bytes: Windows newline translation can't change the sha
    return p


def _snap(root: Path, ds: str, ts: str, files: dict[str, bytes]) -> Path:
    snap = root / ds / ts
    _w(snap / "manifest.json", json.dumps({"source": ds.split("/")[0]}).encode())
    _w(snap / "native" / "raw.zip", b"PK")
    for name, body in files.items():
        _w(snap / "normalized" / name, body)
    return snap


def test_snapshot_layout_uses_newest_normalized_only(tmp_path):
    _snap(tmp_path, "fred/series", "20260101T000000Z", {"observations.csv": b"a\n1\n"})
    _snap(tmp_path, "fred/series", "20260201T000000Z", {"observations.csv": b"a\n2\n"})
    d = discover_tables(tmp_path)
    assert [t.table_name for t in d.tables] == ["observations"]
    t = d.tables[0]
    assert t.snapshot == "20260201T000000Z" and t.dataset == "fred/series"
    assert "native" not in t.rel_path and t.rel_path.endswith("normalized/observations.csv")


def test_name_clash_prefixes_source_folder(tmp_path):
    _snap(tmp_path, "fred/series", "s1", {"observations.csv": b"a\n1\n", "series.csv": b"a\n1\n"})
    _snap(tmp_path, "owid/iea-ev", "s1", {"observations.csv": b"a\n1\n"})
    names = {t.rel_path.split("/")[0] + ":" + Path(t.rel_path).name: t.table_name
             for t in discover_tables(tmp_path).tables}
    assert names == {"fred:observations.csv": "fred_observations",
                     "owid:observations.csv": "owid_observations",
                     "fred:series.csv": "series"}


def test_graph_table_names_are_never_used(tmp_path):
    _snap(tmp_path, "gleif/lei-records", "s1", {"entities.csv": b"lei\nX\n"})
    _w(tmp_path / "documents.csv", b"a\n1\n")
    names = sorted(t.table_name for t in discover_tables(tmp_path).tables)
    assert names == ["documents_2", "gleif_entities"]


def test_flat_folder_formats_and_skips(tmp_path):
    _w(tmp_path / "a" / "Sales Data.CSV", b"x\n1\n")
    _w(tmp_path / "b.jsonl", b'{"x": 1}\n')
    _w(tmp_path / "c.zip", b"PK")
    _w(tmp_path / "x.rejects.csv", b"x\n")
    _w(tmp_path / "README.md", b"hi")
    d = discover_tables(tmp_path)
    assert {(t.table_name, t.fmt) for t in d.tables} == {("sales_data", "csv"), ("b", "jsonl")}
    assert {r for r, _ in d.skipped} == {"c.zip", "x.rejects.csv"}


def test_identifier_rules():
    assert sanitize_identifier("ApplNo") == "applno"
    assert sanitize_identifier("2024 Revenue ($m)") == "t_2024_revenue_m"
    long = "x" * 80
    fitted = fit_identifier(long)
    assert len(fitted) == MAX_IDENT and fitted == fit_identifier(long)


def test_sha_is_bytes_based(tmp_path):
    a = _w(tmp_path / "a.csv", b"x\r\n1\r\n")
    b = _w(tmp_path / "b.csv", b"x\n1\n")
    assert file_sha256(a) != file_sha256(b)
    assert len(file_sha256(a)) == 64


def test_names_stable_under_unicode_and_spaces(tmp_path):
    _w(tmp_path / "dossier données" / "Prix été.csv", b"x\n1\n")
    (t,) = discover_tables(tmp_path).tables
    assert t.table_name == "prix_ete" and t.rel_path == "dossier données/Prix été.csv"

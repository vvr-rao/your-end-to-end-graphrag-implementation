"""structured.profiling: readers, type inference, stats, sample (no DB, no LLM)."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from backend.app.services.structured.discovery import discover_tables
from backend.app.services.structured.profiling import (
    convert_value,
    iter_records,
    profile_markdown,
    profile_table,
)


def _table(tmp_path: Path, name: str, body: bytes):
    (tmp_path / name).write_bytes(body)
    (t,) = discover_tables(tmp_path).tables
    return t


def test_types_and_leading_zeros(tmp_path):
    t = _table(tmp_path, "p.csv",
               b"ApplNo,Year,Price,Day,Flag,Name\n"
               b"000004,2024,1.5,2024-01-31,Y,Alpha\n"
               b"000159,2025,#VALUE!,2024-02-29,N,Beta\n"
               b"012345,,3e2,,,\n")
    p = profile_table(t)
    types = {c.name: c.type for c in p.columns}
    assert types == {"applno": "text", "year": "integer", "price": "numeric",
                     "day": "date", "flag": "boolean", "name": "text"}
    assert p.row_count == 3
    year = p.column("year")
    assert (year.min, year.max, year.nulls) == ("2024", "2025", 1)
    assert convert_value("numeric", "#VALUE!") is None
    assert convert_value("numeric", "3e2") == Decimal("3e2")
    assert convert_value("date", "2024-01-31") == date(2024, 1, 31)
    assert convert_value("text", "000004") == "000004"


@pytest.mark.parametrize("delim", [b"\t", b"~", b"|"])
def test_delimiter_detection(tmp_path, delim):
    body = delim.join([b"a", b"b", b"c"]) + b"\n" + delim.join([b"1", b"x, y", b"3"]) + b"\n"
    t = _table(tmp_path, "d.txt", body)
    assert list(iter_records(t)) == [{"a": "1", "b": "x, y", "c": "3"}]


def test_cp1252_file_reads(tmp_path):
    t = _table(tmp_path, "w.csv", "name,v\nCafé—x,1\n".encode("cp1252"))
    assert list(iter_records(t)) == [{"name": "Café—x", "v": "1"}]


def test_json_flattening_and_largest_list(tmp_path):
    data = {"meta": {"n": 2}, "results": [
        {"id": 1, "org": {"name": "A", "lei": "L1"}, "tags": ["x", "y"]},
        {"id": 2, "org": {"name": "B"}, "extra": True}]}
    t = _table(tmp_path, "r.json", json.dumps(data).encode())
    p = profile_table(t)
    assert [c.name for c in p.columns] == ["id", "org_name", "org_lei", "tags", "extra"]
    assert p.column("org_lei").nulls == 1 and p.column("extra").nulls == 1
    assert p.column("tags").type == "text"


def test_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"k": ["001", "002"], "v": [1.5, 2.0]}), tmp_path / "t.parquet")
    (t,) = discover_tables(tmp_path).tables
    p = profile_table(t)
    assert {c.name: c.type for c in p.columns} == {"k": "text", "v": "numeric"}


def test_sample_is_deterministic_and_bounded(tmp_path):
    body = b"n\n" + b"".join(str(i).encode() + b"\n" for i in range(500))
    t = _table(tmp_path, "n.csv", body)
    a, b = profile_table(t), profile_table(t)
    assert a.sample_rows == b.sample_rows and len(a.sample_rows) == 20
    md = profile_markdown(a)
    assert "TABLE n" in md and "500 rows" in md and "SAMPLE ROWS (20 of 500" in md


def test_duplicate_headers_get_unique_names(tmp_path):
    t = _table(tmp_path, "h.csv", b"Value,value,VALUE\n1,2,3\n")
    assert [c.name for c in profile_table(t).columns] == ["value", "value_2", "value_3"]

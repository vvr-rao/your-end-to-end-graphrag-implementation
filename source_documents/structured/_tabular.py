"""Streaming delimited-table helpers.

Everything is read and written with the stdlib ``csv`` module, row by row,
so values stay **strings** end to end -- application numbers like ``"004567"``
or CIKs like ``"0000320193"`` keep their leading zeros, and nothing is loaded
whole into memory. Provider-native delimiters are always given explicitly
(tab, tilde, pipe); nothing is sniffed.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

DELIMITERS = {"comma": ",", "tab": "\t", "tilde": "~", "pipe": "|"}
DELIMITER_NAMES = {v: k for k, v in DELIMITERS.items()}

# Big provider text fields (e.g. AACT descriptions) exceed the 128 KB default.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def delimiter_name(delim: str) -> str:
    return DELIMITER_NAMES.get(delim, repr(delim))


def _open_text(path: Path, encoding: str):
    return path.open("r", encoding=encoding, newline="")


def detect_encoding(path: Path, candidates: Sequence[str] = ("utf-8-sig", "cp1252")) -> str:
    """Pick the first candidate encoding that decodes the whole file.

    Streams in blocks (no full load). FDA text tables are usually UTF-8 or
    cp1252; ``cp1252`` never fails on arbitrary bytes, so it is the fallback.
    """
    for enc in candidates:
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                while fh.read(1 << 20):
                    pass
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def iter_rows(path: Path, delimiter: str, *, quoting: int = csv.QUOTE_NONE,
              encoding: str | None = None) -> Iterator[list[str]]:
    """Yield rows of a provider-native delimited file as lists of strings.

    ``QUOTE_NONE`` is the default because the FDA / SEC / AACT flat files are
    not quote-escaped: a stray ``"`` inside a text field is data, not quoting.
    """
    enc = encoding or detect_encoding(path)
    with _open_text(path, enc) as fh:
        yield from csv.reader(fh, delimiter=delimiter, quoting=quoting)


def convert_delimited(
    src: Path,
    dst: Path,
    *,
    delimiter: str,
    quoting: int = csv.QUOTE_NONE,
    null_markers: Iterable[str] = (),
    encoding: str | None = None,
    strip_trailing_empty: bool = True,
) -> tuple[int, list[str], str]:
    """Convert a native delimited table to RFC-4180 CSV (UTF-8, ``\\n``).

    Values are copied verbatim (strings), except that cells equal to one of
    ``null_markers`` become empty. Returns ``(data_rows, header, encoding)``.
    A ragged row (wrong field count) raises ValueError naming the line, so a
    wrong delimiter is never silently accepted.
    """
    enc = encoding or detect_encoding(src)
    nulls = set(null_markers)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    rows = 0
    header: list[str] = []
    try:
        with _open_text(src, enc) as fin, part.open("w", encoding="utf-8", newline="") as fout:
            reader = csv.reader(fin, delimiter=delimiter, quoting=quoting)
            writer = csv.writer(fout, lineterminator="\n")
            for lineno, row in enumerate(reader, start=1):
                if strip_trailing_empty and len(row) > 1 and row[-1] == "" and header \
                        and len(row) == len(header) + 1:
                    row = row[:-1]
                if lineno == 1:
                    if strip_trailing_empty and len(row) > 1 and row[-1] == "":
                        row = row[:-1]
                    header = [h.strip() for h in row]
                    writer.writerow(header)
                    continue
                if not row or (len(row) == 1 and row[0] == ""):
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"{src.name} line {lineno}: {len(row)} fields, header has "
                        f"{len(header)} (delimiter {delimiter_name(delimiter)}). "
                        f"Refusing to write a misaligned CSV.")
                if nulls:
                    row = ["" if v in nulls else v for v in row]
                writer.writerow(row)
                rows += 1
        os.replace(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return rows, header, enc


def count_data_rows(path: Path, delimiter: str, *, quoting: int = csv.QUOTE_NONE,
                    has_header: bool = True) -> int:
    n = sum(1 for row in iter_rows(path, delimiter, quoting=quoting) if row and row != [""])
    return max(0, n - (1 if has_header else 0))


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


class CsvTable:
    """Append-only CSV writer with a fixed column list (atomic on close)."""

    def __init__(self, path: Path, columns: Sequence[str]):
        self.path = path
        self.columns = list(columns)
        self.rows = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._part = path.with_name(path.name + ".part")
        self._fh: io.TextIOWrapper | None = self._part.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns,
                                      extrasaction="raise", lineterminator="\n")
        self._writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({k: _cell(v) for k, v in row.items()})
        self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            os.replace(self._part, self.path)

    def abort(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._part.unlink(missing_ok=True)


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        import json
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return v


class TableSet:
    """A group of related CSV tables written together (parent + children).

    Child tables carry the parent key plus an ``ordinal`` so each repeated
    element becomes exactly one row -- independent arrays are never
    cross-joined.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.tables: dict[str, CsvTable] = {}

    def table(self, name: str, columns: Sequence[str]) -> CsvTable:
        t = self.tables.get(name)
        if t is None:
            t = CsvTable(self.folder / f"{name}.csv", columns)
            self.tables[name] = t
        return t

    def write(self, name: str, columns: Sequence[str], row: Mapping[str, Any]) -> None:
        self.table(name, columns).write(row)

    def close(self) -> dict[str, CsvTable]:
        for t in self.tables.values():
            t.close()
        return self.tables

    def abort(self) -> None:
        for t in self.tables.values():
            t.abort()


# ------------------------------------------------------- JSON -> related tables

def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


class JsonFlattener:
    """Flatten nested JSON records into related CSV tables, without
    Cartesian products.

    * scalars and nested objects become columns of the current table
      (object paths joined with ``__``);
    * a list of objects becomes a child table ``<table>__<field>`` with one
      row per element, carrying the parent's key columns plus
      ``<field>_ordinal``;
    * a list of scalars becomes a child table with a single ``value``
      column (same key + ordinal scheme).

    Independent arrays (e.g. conditions and locations of one study) therefore
    land in separate tables -- 2 conditions and 3 locations give 2 + 3 rows,
    never 6. ``max_depth`` bounds recursion; anything deeper is kept as a
    JSON-string column (recorded in :attr:`json_columns`). ``key(record,
    index)`` returns the key columns copied into every row of that record.

    Column sets are discovered in a first pass over the records and rows are
    written in a second pass, so ``records`` must be a callable returning a
    fresh iterator (e.g. re-reading page files from disk).
    """

    def __init__(self, root: str, key: Callable[[dict, int], dict], *, max_depth: int = 4,
                 skip_fields: Iterable[str] = ()):
        self.root = root
        self.key = key
        self.max_depth = max_depth
        self.skip = set(skip_fields)
        self.columns: dict[str, dict[str, None]] = {}
        self.json_columns: set[str] = set()

    def _rows(self, record: dict, table: str, keys: dict, depth: int,
              out: list[tuple[str, dict]]) -> None:
        row: dict[str, Any] = dict(keys)
        children: list[tuple[str, str, list]] = []

        def walk(obj: dict, prefix: str) -> None:
            for k, v in obj.items():
                if not prefix and k in self.skip:
                    continue
                col = f"{prefix}{k}"
                if _is_scalar(v):
                    row[col] = v
                elif isinstance(v, dict):
                    if depth >= self.max_depth:
                        row[col] = v
                        self.json_columns.add(f"{table}.{col}")
                    else:
                        walk(v, f"{col}__")
                elif isinstance(v, list):
                    if depth >= self.max_depth:
                        row[col] = v
                        self.json_columns.add(f"{table}.{col}")
                    else:
                        children.append((col, f"{table}__{col}", v))

        walk(record, "")
        out.append((table, row))
        for col, child, items in children:
            for i, item in enumerate(items, start=1):
                ckeys = {**keys, f"{col}_ordinal": i}
                if isinstance(item, dict):
                    self._rows(item, child, ckeys, depth + 1, out)
                elif isinstance(item, list):
                    out.append((child, {**ckeys, "value": item}))
                    self.json_columns.add(f"{child}.value")
                else:
                    out.append((child, {**ckeys, "value": item}))

    def rows(self, record: dict, index: int = 1) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        self._rows(record, self.root, dict(self.key(record, index)), 0, out)
        return out

    def write(self, records: Callable[[], Iterable[dict]], folder: Path) -> dict[str, CsvTable]:
        for i, rec in enumerate(records(), start=1):
            for table, row in self.rows(rec, i):
                cols = self.columns.setdefault(table, {})
                for c in row:
                    cols.setdefault(c, None)
        ts = TableSet(folder)
        try:
            for table, cols in self.columns.items():
                ts.table(table, list(cols))
            for i, rec in enumerate(records(), start=1):
                for table, row in self.rows(rec, i):
                    ts.write(table, list(self.columns[table]), row)
        except BaseException:
            ts.abort()
            raise
        return ts.close()

"""Read a table file and profile it in one streaming pass (no LLM, no DB).

The profile is what the LLM sees instead of the table: per-column inferred type,
null and distinct counts, min/max, top values, plus a small deterministic sample
of rows. The same readers feed the Postgres loader, and ``convert_value`` turns
the text read from the file into the Python value for the inferred type, so the
loader and the profile can never disagree.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from backend.app.services.structured.discovery import TableSource, sanitize_identifier

SAMPLE_ROWS = 20
DISTINCT_CAP = 10_000
TOP_VALUES = 8
MAX_JSON_BYTES = 200 * 1024 * 1024
TEXT_SAMPLE_CHARS = 120

# Tokens that mean "no value" in numeric/date columns. A column whose other
# values are numbers keeps its numeric type and loads these as NULL; in a text
# column they stay literal text.
SOFT_NULLS = {"null", "none", "nan", "n/a", "na", ".", "-", "--", "#n/a",
              # spreadsheet error cells exported as text
              "#value!", "#div/0!", "#ref!", "#num!", "#name?", "#null!"}
_BOOL_TRUE = {"true", "t", "yes", "y"}
_BOOL_FALSE = {"false", "f", "no", "n"}

_INT_RE = re.compile(r"^[+-]?(0|[1-9]\d*)$", re.ASCII)
_NUM_RE = re.compile(r"^[+-]?((0|[1-9]\d*)(\.\d+)?|\.\d+)([eE][+-]?\d{1,3})?$", re.ASCII)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$", re.ASCII)
_INT64 = 2**63 - 1

TYPE_ORDER = ("boolean", "integer", "numeric", "date", "timestamp")


class ProfileError(RuntimeError):
    pass


# ----------------------------------------------------------------- readers


def _encoding(path: Path) -> str:
    head = path.open("rb").read(1 << 20)
    try:
        head.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError as exc:
        # A multibyte char cut at the 1 MiB boundary is still utf-8.
        if exc.start >= len(head) - 3:
            return "utf-8-sig"
        return "cp1252"


def _delimiter(path: Path, encoding: str) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    with path.open("r", encoding=encoding, newline="", errors="replace") as fh:
        sample = fh.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t|~;").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        counts = {d: first.count(d) for d in ",\t|~;"}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] else ","


def _flatten(obj: dict, prefix: str = "") -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for k, v in obj.items():
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = _scalar_text(v)
    return out


def _scalar_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return str(v)


def _largest_record_list(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            return max(lists, key=len)
        return [data]
    raise ProfileError("JSON file is neither a list of records nor an object")


def iter_records(src: TableSource) -> Iterator[dict[str, str | None]]:
    """Yield each row as {source column name: text or None}."""
    path = src.path
    if src.fmt == "csv":
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        enc = _encoding(path)
        delim = _delimiter(path, enc)
        with path.open("r", encoding=enc, newline="", errors="replace") as fh:
            reader = csv.reader(fh, delimiter=delim)
            header = next(reader, None)
            if header is None:
                return
            header = [h.strip() for h in header]
            for row in reader:
                if not row or all(not c.strip() for c in row):
                    continue
                yield {h: (row[i] if i < len(row) else None) for i, h in enumerate(header)}
    elif src.fmt == "jsonl":
        with path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    yield _flatten(obj) if isinstance(obj, dict) else {"value": _scalar_text(obj)}
    elif src.fmt == "json":
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ProfileError(f"{path.name}: JSON over {MAX_JSON_BYTES >> 20} MB; convert it to "
                               "JSON Lines (.jsonl) so it can be streamed")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for obj in _largest_record_list(data):
            yield _flatten(obj) if isinstance(obj, dict) else {"value": _scalar_text(obj)}
    elif src.fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ProfileError("Parquet needs pyarrow: run `uv sync --extra structured`") from exc
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=10_000):
            for rec in batch.to_pylist():
                yield {k: (json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
                           if isinstance(v, (dict, list)) else _scalar_text(v))
                       for k, v in rec.items()}
    else:  # pragma: no cover
        raise ProfileError(f"unsupported format {src.fmt}")


# --------------------------------------------------------------- typing


def is_null(raw: str | None) -> bool:
    return raw is None or raw.strip() == ""


def _fits(kind: str, s: str) -> bool:
    if kind == "boolean":
        return s.lower() in _BOOL_TRUE or s.lower() in _BOOL_FALSE
    if kind == "integer":
        return bool(_INT_RE.match(s)) and abs(int(s)) <= _INT64
    if kind == "numeric":
        return bool(_NUM_RE.match(s))
    if kind == "date":
        if not _DATE_RE.match(s):
            return False
        try:
            date.fromisoformat(s)
            return True
        except ValueError:
            return False
    if kind == "timestamp":
        if not (_TS_RE.match(s) or _DATE_RE.match(s)):
            return False
        try:
            datetime.fromisoformat(s.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    return True


def convert_value(kind: str, raw: str | None):
    """Text from the file -> Python value for the column's Postgres type."""
    if is_null(raw):
        return None
    s = raw.strip()
    if kind != "text" and s.lower() in SOFT_NULLS:
        return None
    if kind == "boolean":
        return s.lower() in _BOOL_TRUE
    if kind == "integer":
        return int(s)
    if kind == "numeric":
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    if kind == "date":
        return date.fromisoformat(s)
    if kind == "timestamp":
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return raw


PG_TYPES = {"boolean": "boolean", "integer": "bigint", "numeric": "numeric",
            "date": "date", "timestamp": "timestamptz", "text": "text"}


# --------------------------------------------------------------- profile


@dataclass
class ColumnProfile:
    name: str                       # Postgres column name
    source_name: str                # header / key in the file
    type: str = "text"
    non_null: int = 0
    nulls: int = 0
    distinct: int = 0
    distinct_capped: bool = False
    min: str | None = None
    max: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    top_values: list[list] = field(default_factory=list)   # [[value, count], ...]

    @property
    def pg_type(self) -> str:
        return PG_TYPES[self.type]


@dataclass
class TableProfile:
    table_name: str
    rel_path: str
    fmt: str
    file_sha: str
    row_count: int
    columns: list[ColumnProfile]
    sample_rows: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TableProfile:
        cols = [ColumnProfile(**c) for c in d["columns"]]
        return cls(**{**d, "columns": cols})

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


class _ColAcc:
    __slots__ = ("candidates", "capped", "counts", "hi", "lo", "maxlen", "minlen", "non_null", "nulls")

    def __init__(self) -> None:
        self.candidates = list(TYPE_ORDER)
        self.counts: Counter = Counter()
        self.capped = False
        self.lo = self.hi = None
        self.minlen = self.maxlen = None
        self.non_null = 0
        self.nulls = 0

    def add(self, raw: str | None) -> None:
        if is_null(raw):
            self.nulls += 1
            return
        s = raw.strip()
        self.non_null += 1
        n = len(s)
        self.minlen = n if self.minlen is None else min(self.minlen, n)
        self.maxlen = n if self.maxlen is None else max(self.maxlen, n)
        if s in self.counts or len(self.counts) < DISTINCT_CAP:
            self.counts[s] += 1
        else:
            self.capped = True
        if self.candidates and s.lower() not in SOFT_NULLS:
            self.candidates = [k for k in self.candidates if _fits(k, s)]
            if self.candidates:
                kind = self.candidates[0]
                val = convert_value(kind if kind != "boolean" else "text", s)
                if kind in {"integer", "numeric"}:
                    val = Decimal(s)
                self._range(val, kind)

    def _range(self, val, kind) -> None:
        if kind in {"integer", "numeric"}:
            self.lo = val if self.lo is None or (isinstance(self.lo, Decimal) and val < self.lo) else self.lo
            self.hi = val if self.hi is None or (isinstance(self.hi, Decimal) and val > self.hi) else self.hi
        else:
            sval = str(val)
            self.lo = sval if self.lo is None or str(self.lo) > sval else self.lo
            self.hi = sval if self.hi is None or str(self.hi) < sval else self.hi

    def final_type(self) -> str:
        real = sum(c for v, c in self.counts.items() if v.lower() not in SOFT_NULLS)
        if not self.candidates or (real == 0 and not self.capped):
            return "text"
        return self.candidates[0]


def _column_names(sources: list[str]) -> list[str]:
    out, seen = [], set()
    for s in sources:
        base = sanitize_identifier(s)[:63]
        cand, i = base, 2
        while cand in seen:
            cand = f"{base[:59]}_{i}"
            i += 1
        seen.add(cand)
        out.append(cand)
    return out


def _clip(v: str | None) -> str | None:
    if v is None:
        return None
    return v if len(v) <= TEXT_SAMPLE_CHARS else v[:TEXT_SAMPLE_CHARS] + "…"


def profile_table(src: TableSource, *, sample_rows: int = SAMPLE_ROWS, seed: int = 0) -> TableProfile:
    """One pass: types, stats and a reservoir sample (deterministic per seed)."""
    rng = random.Random(seed)
    order: list[str] = []
    accs: dict[str, _ColAcc] = {}
    sample: list[dict] = []
    rows = 0
    for rec in iter_records(src):
        for k in rec:
            if k not in accs:
                order.append(k)
                acc = _ColAcc()
                acc.nulls = rows  # earlier rows lacked this key
                accs[k] = acc
        for k, acc in accs.items():
            acc.add(rec.get(k))
        if len(sample) < sample_rows:
            sample.append(rec)
        else:
            j = rng.randint(0, rows)
            if j < sample_rows:
                sample[j] = rec
        rows += 1
    if not order:
        raise ProfileError(f"{src.rel_path}: no columns found")

    names = _column_names(order)
    cols: list[ColumnProfile] = []
    for name, srcname in zip(names, order, strict=True):
        acc = accs[srcname]
        kind = acc.final_type()
        top = [[v, c] for v, c in acc.counts.most_common(TOP_VALUES)]
        lo = hi = None
        if kind in {"integer", "numeric", "date", "timestamp"}:
            lo = None if acc.lo is None else str(acc.lo)
            hi = None if acc.hi is None else str(acc.hi)
        cols.append(ColumnProfile(
            name=name, source_name=srcname, type=kind, non_null=acc.non_null, nulls=acc.nulls,
            distinct=len(acc.counts), distinct_capped=acc.capped, min=lo, max=hi,
            min_length=acc.minlen, max_length=acc.maxlen,
            top_values=[[_clip(v), c] for v, c in top] if kind in {"text", "boolean"} or len(acc.counts) < 50
            else [],
        ))
    sample_out = [{c.name: _clip(r.get(c.source_name)) for c in cols} for r in sample]
    return TableProfile(table_name=src.table_name, rel_path=src.rel_path, fmt=src.fmt,
                        file_sha=src.file_sha, row_count=rows, columns=cols, sample_rows=sample_out)


def profile_markdown(p: TableProfile, max_sample: int = SAMPLE_ROWS) -> str:
    """Compact text the LLM reads in place of the table."""
    out = io.StringIO()
    out.write(f"TABLE {p.table_name}  (file {p.rel_path}, {p.row_count} rows)\n\nCOLUMNS\n")
    for c in p.columns:
        bits = [f"- {c.name} [{c.pg_type}]"]
        if c.source_name != c.name:
            bits.append(f"header='{c.source_name}'")
        bits.append(f"non_null={c.non_null} nulls={c.nulls}")
        bits.append(f"distinct={c.distinct}{'+' if c.distinct_capped else ''}")
        if c.min is not None:
            bits.append(f"min={c.min} max={c.max}")
        if c.top_values:
            tv = ", ".join(f"{v!r} ({n})" for v, n in c.top_values[:TOP_VALUES])
            bits.append(f"top: {tv}")
        out.write(" ".join(bits) + "\n")
    out.write(f"\nSAMPLE ROWS ({min(len(p.sample_rows), max_sample)} of {p.row_count}, random)\n")
    for r in p.sample_rows[:max_sample]:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out.getvalue()

"""Run one guarded SQL statement against the loaded data tables.

Every statement is first checked by :func:`sql_guard.check_sql` (one SELECT,
data tables only), then executed in a READ ONLY transaction with a
``statement_timeout``. At most ``max_rows`` rows come back -- the statement is
wrapped as ``SELECT * FROM (...) LIMIT max_rows + 1`` so the database still
computes aggregates over the whole table but only a capped result reaches
memory and the prompt. When the cap is hit a ``count(*)`` of the full result
says how many rows there were. The transaction is always rolled back.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.structured.sql_guard import SqlGuardError, check_sql, count_sql, wrap_limit

CELL_CHARS = 200


@dataclass
class SqlResult:
    sql: str
    params: dict[str, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    rows: list[list[str | None]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    total_rows: int | None = None
    error: str | None = None
    seconds: float = 0.0

    def as_markdown(self, max_chars: int = 6000) -> str:
        if self.error:
            return f"(query failed: {self.error})"
        if not self.rows:
            return "(no rows)"
        head = "| " + " | ".join(self.columns) + " |\n|" + "---|" * len(self.columns) + "\n"
        out = head
        shown = 0
        for r in self.rows:
            line = "| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in r) + " |\n"
            if len(out) + len(line) > max_chars:
                break
            out += line
            shown += 1
        note = []
        if shown < len(self.rows):
            note.append(f"{shown} of {len(self.rows)} fetched rows shown (size cap)")
        if self.truncated:
            total = f"{self.total_rows}" if self.total_rows is not None else "more"
            note.append(f"result capped at {self.row_count} of {total} rows -- filter or aggregate for the rest")
        return out + ("\n(" + "; ".join(note) + ")" if note else "")


def _cell(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        s = format(v.normalize(), "f") if v == v.to_integral_value() else str(v)
    elif isinstance(v, (datetime, date)):
        s = v.isoformat()
    else:
        s = str(v)
    return s if len(s) <= CELL_CHARS else s[:CELL_CHARS] + "…"


def _coerce_one(kind: str, v):
    if v is None:
        return None
    k = (kind or "text").lower()
    try:
        if k.endswith("[]"):
            if isinstance(v, str) and v.strip().startswith("["):
                try:   # a JSON-ish list written as text: ["a", "b"] or ['a', 'b']
                    v = json.loads(v.replace("'", '"'))
                except ValueError:
                    pass
            items = v if isinstance(v, list) else [x.strip() for x in str(v).split(",") if x.strip()]
            return [_coerce_one(k[:-2], x) for x in items]
        if k in {"integer", "int", "bigint", "smallint"}:
            return int(v)
        if k in {"numeric", "decimal", "float", "double", "real", "number"}:
            return Decimal(str(v))
        if k == "date":
            return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])
        if k in {"timestamp", "timestamptz", "datetime"}:
            return v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if k in {"boolean", "bool"}:
            return v if isinstance(v, bool) else str(v).strip().lower() in {"true", "t", "yes", "y", "1"}
    except (ValueError, TypeError, InvalidOperation):
        return v
    if isinstance(v, list):
        return [str(x) for x in v]
    return str(v) if not isinstance(v, str) else v


def coerce_params(values: dict[str, Any], declared: list[dict] | None) -> dict[str, Any]:
    """Cast the model's JSON values to the declared param types so asyncpg can
    bind them (it refuses '2024' for an integer column)."""
    kinds = {p.get("name"): p.get("type") for p in declared or []}
    out = {}
    for k, v in (values or {}).items():
        key = str(k).lstrip(":")
        kind = kinds.get(key)
        if kind is None:  # undeclared (new SQL): keep JSON numbers/bools, else text
            out[key] = v if isinstance(v, (int, float, bool, list)) else (None if v is None else str(v))
        else:
            out[key] = _coerce_one(kind, v)
    return out


def _db_error(exc: BaseException) -> str:
    msg = str(getattr(exc, "orig", None) or exc).strip()
    msg = re.sub(r"^<class '[^']+'>:\s*", "", msg)
    return (msg.splitlines()[0] if msg else type(exc).__name__)[:500]


async def run_sql(
    sql: str,
    params: dict[str, Any] | None,
    allowed_tables: set[str],
    *,
    max_rows: int = 50,
    timeout_ms: int = 5000,
    declared_params: list[dict] | None = None,
) -> SqlResult:
    from backend.app.db.engine import get_sessionmaker

    started = time.monotonic()
    res = SqlResult(sql=sql, params=dict(params or {}))
    try:
        checked = check_sql(sql, allowed_tables)
    except SqlGuardError as exc:
        res.error = f"rejected: {exc}"
        return res
    bound = coerce_params(params or {}, declared_params)
    for p in declared_params or []:            # an omitted optional filter = every value
        if p.get("optional") and p.get("name") not in bound:
            bound[p["name"]] = None
    missing = [p for p in checked.params if p not in bound]
    if missing:
        res.error = f"missing parameter value(s): {', '.join(missing)}"
        return res
    bound = {k: v for k, v in bound.items() if k in checked.params}
    res.params = {k: (v if isinstance(v, (int, float, bool, list)) or v is None else str(v))
                  for k, v in bound.items()}
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            # Unqualified names resolve to the data tables (the guard has
            # already limited which tables the statement may name).
            await s.execute(sql_text("SET LOCAL search_path TO graphrag"))
            out = await s.execute(sql_text(wrap_limit(checked.sql, max_rows + 1)), bound)
            res.columns = list(out.keys())
            rows = out.fetchall()
            res.truncated = len(rows) > max_rows
            res.rows = [[_cell(v) for v in r] for r in rows[:max_rows]]
            res.row_count = len(res.rows)
            if res.truncated:
                try:
                    res.total_rows = int((await s.execute(sql_text(count_sql(checked.sql)), bound)).scalar())
                except Exception:
                    res.total_rows = None
        except Exception as exc:
            res.error = _db_error(exc)
        finally:
            await s.rollback()
    res.seconds = round(time.monotonic() - started, 3)
    return res

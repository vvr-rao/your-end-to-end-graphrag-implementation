"""Check SQL before it runs: one read-only SELECT over the loaded data tables.

The data tables share the ``graphrag`` schema with the graph's own tables, so
every statement -- the pre-written ones in the metadata and any the model
writes while answering -- passes through :func:`check_sql` first:

* exactly one statement, and it is a query (SELECT / WITH ... SELECT / UNION);
* no DML/DDL anywhere (INSERT/UPDATE/DELETE/MERGE/CREATE/DROP/ALTER/COPY/SET,
  ``SELECT ... INTO``, ``FOR UPDATE``);
* every table it reads is one of ``allowed_tables`` (CTE names excepted), in the
  ``graphrag`` schema or unqualified -- so ``entities``, ``pg_catalog.*`` and
  ``information_schema.*`` are rejected;
* no server-side functions that touch files, settings or other databases
  (``pg_*``, ``lo_*``, ``dblink``, ``current_setting``, ``set_config``, ...);
* parameters are named ``:placeholders`` (bound by the driver, never pasted in).

Execution adds a READ ONLY transaction and a statement timeout on top
(``sql_exec``); this module is pure Python so it is unit-tested on every OS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

SCHEMA = "graphrag"

_FORBIDDEN_NODES: tuple[type, ...] = tuple(
    t for t in (
        getattr(exp, n, None) for n in (
            "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable",
            "Command", "Copy", "Set", "Transaction", "Commit", "Rollback", "Into", "Lock",
            "TruncateTable", "Grant", "Use", "Pragma", "LoadData", "Kill", "Analyze",
        )
    ) if t is not None
)
_FORBIDDEN_FUNC_PREFIXES = ("pg_", "lo_", "dblink", "txid_", "set_", "current_setting")
_FORBIDDEN_FUNCS = {
    "set_config", "current_setting", "query_to_xml", "table_to_xml", "cursor_to_xml",
    "query_to_xml_and_xmlschema", "database_to_xml", "schema_to_xml", "xpath_exists",
    "version", "inet_server_addr", "inet_client_addr", "nextval", "setval", "currval",
}


class SqlGuardError(ValueError):
    """The statement is not allowed (or not valid SQL)."""


@dataclass
class GuardResult:
    sql: str
    tables: set[str]
    params: list[str] = field(default_factory=list)


def _func_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this or "").lower()
    try:
        return node.sql_name().lower()
    except Exception:  # pragma: no cover
        return type(node).__name__.lower()


def parse_one_select(sql: str) -> exp.Expression:
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise SqlGuardError("empty SQL")
    try:
        trees = [t for t in sqlglot.parse(text, read="postgres") if t is not None]
    except ParseError as exc:
        raise SqlGuardError(f"SQL does not parse: {str(exc).splitlines()[0]}") from exc
    if len(trees) != 1:
        raise SqlGuardError(f"expected exactly one statement, got {len(trees)}")
    root = trees[0]
    if not isinstance(root, exp.Query):
        raise SqlGuardError(f"only SELECT queries are allowed (got {root.key.upper()})")
    return root


def check_sql(
    sql: str,
    allowed_tables: set[str] | frozenset[str],
    *,
    columns: dict[str, set[str]] | None = None,
) -> GuardResult:
    """Raise :class:`SqlGuardError` unless ``sql`` is a safe read of ``allowed_tables``.

    ``columns`` (table -> column names), when given, also checks that every
    column the query names exists in one of the tables it reads (aliases and
    CTE columns are allowed).
    """
    root = parse_one_select(sql)
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlGuardError(f"{node.key.upper()} is not allowed in a read-only query")
        if isinstance(node, exp.Func):
            name = _func_name(node)
            if name in _FORBIDDEN_FUNCS or name.startswith(_FORBIDDEN_FUNC_PREFIXES):
                raise SqlGuardError(f"function {name}() is not allowed")
        if isinstance(node, exp.Parameter) or (isinstance(node, exp.Placeholder) and not node.name):
            raise SqlGuardError("use named :placeholders for parameters (not $1 or ?)")

    cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    tables: set[str] = set()
    for t in root.find_all(exp.Table):
        name = (t.name or "").lower()
        if not name:
            raise SqlGuardError("table functions are not allowed")
        schema = (t.db or "").lower()
        if t.catalog:
            raise SqlGuardError(f"cross-database reference {t.sql()} is not allowed")
        if schema and schema != SCHEMA:
            raise SqlGuardError(f"table {t.sql()} is outside the {SCHEMA} schema")
        if not schema and name in cte_names:
            continue
        if name not in allowed:
            raise SqlGuardError(f"table '{name}' is not one of the loaded data tables")
        tables.add(name)
    if not tables:
        raise SqlGuardError("query reads no data table")

    if columns is not None:
        known: set[str] = set()
        for t in tables:
            known |= {c.lower() for c in columns.get(t, set())}
        aliases = {a.alias.lower() for a in root.find_all(exp.Alias) if a.alias}
        aliases |= {c.lower() for cte in root.find_all(exp.CTE)
                    for c in (cte.args.get("alias").columns if cte.args.get("alias") else [])
                    for c in [c.name]}
        aliases |= {t.alias_or_name.lower() for t in root.find_all(exp.TableAlias)}
        unknown = sorted({c.name.lower() for c in root.find_all(exp.Column)
                          if c.name and c.name != "*"} - known - aliases)
        if unknown:
            raise SqlGuardError(f"unknown column(s): {', '.join(unknown)}")

    params = []
    for p in root.find_all(exp.Placeholder):
        if p.name and p.name not in params:
            params.append(p.name)
    return GuardResult(sql=quote_reserved((sql or "").strip().rstrip(";").strip(), root),
                       tables=tables, params=params)


# Postgres reserved key words (SQL key words appendix, "reserved" and
# "reserved (can be function or type)"). A data column can legitimately be
# called `end` or `user` (SEC company facts has `start`/`end`); sqlglot parses
# such SQL happily, Postgres rejects it unless the name is double-quoted.
PG_RESERVED = frozenset("""
all analyse analyze and any array as asc asymmetric authorization binary both case cast
check collate collation column concurrently constraint create cross current_catalog
current_date current_role current_schema current_time current_timestamp current_user
default deferrable desc distinct do else end except false fetch for foreign freeze from
full grant group having ilike in initially inner intersect into is isnull join lateral
leading left like limit localtime localtimestamp natural not notnull null offset on only
or order outer overlaps placing primary references returning right select session_user
similar some symmetric system_user table tablesample then to trailing true union unique
user using variadic verbose when where window with
""".split())


def quote_reserved(sql: str, root: exp.Expression | None = None) -> str:
    """Double-quote unquoted identifiers that are Postgres reserved words.

    Spliced at the positions the parser reports (like :func:`rename_tables`),
    so placeholders and formatting are untouched. Stored statements written
    before this existed are repaired at run time without re-profiling."""
    text = (sql or "").strip().rstrip(";").strip()
    root = root if root is not None else parse_one_select(text)
    spans = []
    for ident in root.find_all(exp.Identifier):
        name = ident.name or ""
        meta = ident.meta or {}
        if ident.quoted or name.lower() not in PG_RESERVED or "start" not in meta:
            continue
        start, end = meta["start"], meta["end"] + 1
        if text[start:end].lower() == name.lower():
            spans.append((start, end, f'"{name.lower()}"'))
    for start, end, new in sorted(set(spans), reverse=True):
        text = text[:start] + new + text[end:]
    return text


def rename_tables(sql: str, mapping: dict[str, str]) -> str:
    """Rewrite table references in place (used when a cached entry's table got
    a different name). Only the identifiers are spliced, at the positions the
    parser reports, so placeholders, casts and formatting stay untouched.
    Columns qualified by the table name (``drug.name``) follow the rename;
    columns that merely share the name do not."""
    if not mapping:
        return sql
    lower = {k.lower(): v for k, v in mapping.items()}
    root = parse_one_select(sql)
    text = (sql or "").strip().rstrip(";").strip()
    aliases = {t.alias.lower() for t in root.find_all(exp.Table) if t.alias}
    spans: list[tuple[int, int, str]] = []

    def _mark(ident) -> None:
        if not isinstance(ident, exp.Identifier):
            return
        meta = ident.meta or {}
        if "start" not in meta:
            raise SqlGuardError("cannot locate table name in SQL for renaming")
        new = lower.get(ident.name.lower())
        if new is not None:
            spans.append((meta["start"], meta["end"] + 1, new))

    for t in root.find_all(exp.Table):
        _mark(t.this)
    for c in root.find_all(exp.Column):
        tbl = c.args.get("table")
        if isinstance(tbl, exp.Identifier) and tbl.name.lower() not in aliases:
            _mark(tbl)
    for start, end, new in sorted(set(spans), reverse=True):
        quoted = text[start] == '"' if start < len(text) else False
        text = text[:start] + (f'"{new}"' if quoted else new) + text[end:]
    return text


_SQL_TRAILING = re.compile(r";\s*$")


def wrap_limit(sql: str, n: int) -> str:
    """``SELECT * FROM (<sql>) AS _q LIMIT n`` -- caps rows fetched into memory
    and into the prompt; aggregates inside the query still see every row."""
    inner = _SQL_TRAILING.sub("", sql.strip())
    return f"SELECT * FROM (\n{inner}\n) AS _yegi_q LIMIT {int(n)}"


def count_sql(sql: str) -> str:
    inner = _SQL_TRAILING.sub("", sql.strip())
    return f"SELECT count(*) FROM (\n{inner}\n) AS _yegi_q"

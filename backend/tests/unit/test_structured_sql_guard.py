"""structured.sql_guard: only single read-only SELECTs over loaded data tables."""

from __future__ import annotations

import pytest

from backend.app.services.structured.sql_guard import (
    SqlGuardError,
    check_sql,
    count_sql,
    rename_tables,
    wrap_limit,
)

T = {"prices", "facts"}
COLS = {"prices": {"commodity", "value", "period", "unit"}, "facts": {"cik", "val", "fy"}}


@pytest.mark.parametrize("sql", [
    "SELECT commodity, AVG(value) AS avg_value FROM prices WHERE commodity = :c GROUP BY commodity",
    "SELECT p.commodity, MIN(p.value), MAX(p.value) FROM graphrag.prices AS p GROUP BY 1;",
    "WITH m AS (SELECT commodity, value FROM prices) SELECT commodity, COUNT(*) FROM m GROUP BY commodity",
    "SELECT f.cik, p.value FROM facts f JOIN prices p ON p.period = CAST(f.fy AS text) WHERE f.cik = ANY(CAST(:ciks AS text[]))",
    "SELECT commodity FROM prices UNION SELECT cik FROM facts",
    "SELECT value::numeric FROM prices WHERE period BETWEEN :a AND :b ORDER BY value DESC LIMIT 5",
])
def test_allowed(sql):
    r = check_sql(sql, T, columns=COLS)
    assert r.tables <= T and not r.sql.endswith(";")


@pytest.mark.parametrize("sql, msg", [
    ("DELETE FROM prices", "only SELECT"),
    ("UPDATE prices SET value = 0", "only SELECT"),
    ("DROP TABLE prices", "only SELECT"),
    ("SELECT 1 FROM prices; DELETE FROM prices", "exactly one statement"),
    ("SELECT * FROM entities", "not one of the loaded data tables"),
    ("SELECT * FROM graphrag.chunks", "not one of the loaded data tables"),
    ("SELECT * FROM pg_catalog.pg_tables", "outside the graphrag schema"),
    ("SELECT * FROM information_schema.tables", "outside the graphrag schema"),
    ("SELECT pg_read_file('/etc/passwd') FROM prices", "not allowed"),
    ("SELECT pg_sleep(10) FROM prices", "not allowed"),
    ("SELECT current_setting('x') FROM prices", "not allowed"),
    ("SELECT * INTO newt FROM prices", "not allowed"),
    ("SELECT * FROM prices FOR UPDATE", "not allowed"),
    ("SELECT value FROM prices WHERE commodity = $1", "named :placeholders"),
    ("SELECT nosuch FROM prices", "unknown column"),
    ("SELEC value FROM prices", "."),
    ("", "empty SQL"),
])
def test_rejected(sql, msg):
    with pytest.raises(SqlGuardError, match=msg):
        check_sql(sql, T, columns=COLS)


def test_params_in_order_of_appearance():
    r = check_sql("SELECT value FROM prices WHERE commodity = :c AND period >= :p AND unit = :c", T)
    assert r.params == ["c", "p"]


def test_rename_only_touches_table_references():
    sql = ("SELECT prices.value, x.commodity FROM prices JOIN prices AS x ON x.period = prices.period "
           "WHERE commodity = :prices AND value::numeric > 0")
    out = rename_tables(sql, {"prices": "worldbank_prices"})
    assert out == ("SELECT worldbank_prices.value, x.commodity FROM worldbank_prices JOIN worldbank_prices AS x "
                   "ON x.period = worldbank_prices.period WHERE commodity = :prices AND value::numeric > 0")
    check_sql(out, {"worldbank_prices"})


def test_wrappers():
    assert wrap_limit("SELECT 1;", 51).endswith("LIMIT 51")
    assert count_sql("SELECT value FROM prices").startswith("SELECT count(*) FROM (")


def test_reserved_word_columns_are_quoted_without_touching_params():
    from backend.app.services.structured.sql_guard import check_sql, quote_reserved

    sql = ("SELECT cik, end, f.end AS last FROM facts f "
           "WHERE end >= CAST(:end AS date) AND tag = :tag ORDER BY \"end\" DESC")
    res = check_sql(sql, {"facts"})
    assert res.sql == ('SELECT cik, "end", f."end" AS last FROM facts f '
                       'WHERE "end" >= CAST(:end AS date) AND tag = :tag ORDER BY "end" DESC')
    assert sorted(res.params) == ["end", "tag"]
    assert quote_reserved("SELECT a FROM t WHERE a = :x") == "SELECT a FROM t WHERE a = :x"
    assert quote_reserved(res.sql) == res.sql                 # idempotent

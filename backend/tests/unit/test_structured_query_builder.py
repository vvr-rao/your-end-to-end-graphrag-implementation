"""query_builder: JSON query spec -> SQL, deterministic, no database."""

from __future__ import annotations

import pytest

from backend.app.services.structured.query_builder import QuerySpecError, build_sql, merge_params
from backend.app.services.structured.sql_guard import check_sql

COLS = {
    "facts": {"cik": "text", "tag": "text", "unit": "text", "end": "date", "val": "numeric",
              "filed": "date", "fy": "integer", "form": "text"},
    "companies": {"cik": "text", "company": "text", "commodity": "text"},
}


def _b(spec, table="facts"):
    built = build_sql(spec, table, COLS)
    check_sql(built.sql, {"facts", "companies"})      # always passes the guard
    return built


def test_same_spec_same_sql_and_params_come_from_columns():
    spec = {"select": ["end", "val"],
            "where": [{"column": "cik", "op": "=", "param": "cik"},
                      {"column": "tag", "op": "in", "param": "tags"},
                      {"column": "end", "op": "between", "param": ["start", "stop"]},
                      {"column": "form", "op": "=", "value": "20-F"}],
            "order_by": [{"column": "end", "dir": "desc"}], "limit": 5}
    a, b = _b(spec), _b(spec)
    assert a.sql == b.sql == (
        'SELECT "end", "val" FROM "facts" WHERE "cik" = CAST(:cik AS text) '
        'AND "tag" = ANY(CAST(:tags AS text[])) '
        'AND "end" BETWEEN CAST(:start AS date) AND CAST(:stop AS date) '
        'AND "form" = \'20-F\' ORDER BY "end" DESC NULLS LAST LIMIT 5')
    assert a.params == [{"name": "cik", "type": "text", "column": "cik"},
                        {"name": "tags", "type": "text[]", "column": "tag"},
                        {"name": "start", "type": "date", "column": "end"},
                        {"name": "stop", "type": "date", "column": "end"}]


def test_aggregates_group_by_plain_columns_automatically():
    b = _b({"select": ["unit", {"agg": "sum", "column": "val", "as": "total"}, {"agg": "count"}],
            "where": [{"column": "cik", "op": "=", "param": "cik"}],
            "order_by": [{"column": "total", "dir": "desc"}]})
    assert b.sql == ('SELECT "unit", SUM("val") AS "total", COUNT(*) AS "row_count" FROM "facts" '
                     'WHERE "cik" = CAST(:cik AS text) GROUP BY "unit" ORDER BY "total" DESC NULLS LAST')


def test_distinct_on_leads_order_by_and_year_fn():
    b = _b({"select": ["end", "val", "filed"], "distinct_on": ["end"],
            "where": [{"column": {"column": "end", "fn": "year"}, "op": "=", "param": "year"}],
            "order_by": [{"column": "filed", "dir": "desc"}, {"column": "end", "dir": "desc"}]})
    assert b.sql.startswith('SELECT DISTINCT ON ("end") "end", "val", "filed"')
    assert 'CAST(EXTRACT(YEAR FROM "end") AS integer) = CAST(:year AS bigint)' in b.sql
    assert b.sql.endswith('ORDER BY "end" DESC NULLS LAST, "filed" DESC NULLS LAST')
    assert b.params == [{"name": "year", "type": "integer", "column": "year_end"}]


def test_join_qualifies_columns_and_values_are_quoted_literals():
    b = _b({"from": "companies", "join": {"table": "facts", "on": [{"column": "cik", "other_column": "cik"}]},
            "select": ["company", "facts.tag", "facts.val"],
            "where": [{"column": "company", "op": "contains", "param": "name"},
                      {"column": "facts.unit", "op": "=", "value": "O'Brien"}]}, table="facts")
    assert b.sql == ('SELECT "companies"."company", "facts"."tag", "facts"."val" FROM "companies" '
                     'JOIN "facts" ON "companies"."cik" = "facts"."cik" '
                     'WHERE "companies"."company" ILIKE \'%\' || CAST(:name AS text) || \'%\' '
                     'AND "facts"."unit" = \'O\'\'Brien\'')
    assert b.tables == {"companies", "facts"}


@pytest.mark.parametrize("spec, msg", [
    ({"select": ["nope"]}, "unknown column"),
    ({"select": ["val"], "where": [{"column": "val", "op": "like_magic", "param": "x"}]}, "op must be"),
    ({"select": ["val"], "where": [{"column": "val", "op": "=", "param": "x; DROP"}]}, "bad parameter name"),
    ({"select": [{"agg": "median", "column": "val"}]}, "agg must be"),
    ({"select": ["val"], "limit": 10_000}, "limit must be"),
    ({"select": ["val"], "sql": "DELETE FROM facts"}, "unknown spec key"),
    ({"select": ["val"], "join": {"table": "entities", "on": [{"column": "cik", "other_column": "id"}]}},
     "not loaded"),
    ({"select": ["val"], "where": [{"column": "cik", "op": "=", "param": "a", "value": "b"}]},
     "exactly one of param or value"),
    ({"select": []}, "at least one"),
])
def test_bad_specs_are_rejected_with_a_reason(spec, msg):
    with pytest.raises(QuerySpecError, match=msg):
        build_sql(spec, "facts", COLS)


def test_identifiers_cannot_inject():
    with pytest.raises(QuerySpecError):
        build_sql({"select": ['val" FROM pg_shadow --']}, "facts", COLS)
    with pytest.raises(QuerySpecError, match="bad alias"):
        build_sql({"select": [{"agg": "sum", "column": "val", "as": 'x" FROM y'}]}, "facts", COLS)


def test_merge_params_keeps_descriptions_but_not_types_from_the_model():
    b = _b({"select": ["val"], "where": [{"column": "fy", "op": "=", "param": "fy"}]})
    got = merge_params(b, [{"name": "fy", "type": "text", "description": "fiscal year", "example": 2025},
                           {"name": "unused", "description": "x"}])
    assert got == [{"name": "fy", "type": "integer", "column": "fy", "description": "fiscal year",
                    "example": "2025"}]

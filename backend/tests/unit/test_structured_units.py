"""structured.units: units of a SQL result's output columns, resolved for the
statement that actually ran (aliases, aggregates, reshapes) from the table
metadata and -- for a value whose unit varies by row -- from the data. No
database: the unit lookups run against a small in-memory table. Nothing here
is specific to any dataset."""

from __future__ import annotations

import re

from backend.app.services.structured.sql_exec import SqlResult
from backend.app.services.structured.units import (
    DIMENSIONLESS,
    KNOWN,
    MAPPED,
    MIXED,
    NA,
    PER_ROW,
    UNKNOWN,
    result_units,
    static_units,
    units_line,
)

FLOWS = {"table_name": "flows", "columns": [
    {"name": "reporter", "role": "dimension"},
    {"name": "partner_code", "role": "identifier"},
    {"name": "trade_value", "role": "measure", "roles": ["quantity"], "unit": "USD"},
    {"name": "net_weight", "role": "measure", "roles": ["quantity"], "unit": "kg"},
    {"name": "qty", "role": "measure", "roles": ["quantity"], "unit": "depends on qty_unit (kg or items)"},
    {"name": "qty_unit", "role": "dimension", "roles": ["unit", "qualifier"]},
    {"name": "score", "role": "measure", "roles": ["quantity"]},            # no unit stated
    {"name": "year", "role": "time"},
]}

INDICATORS = {"table_name": "indicators", "columns": [
    {"name": "indicator_id", "role": "identifier"},
    {"name": "indicator", "role": "attribute"},
    {"name": "unit", "role": "attribute", "roles": ["unit", "qualifier"]},
    {"name": "economy", "role": "dimension"},
    {"name": "year", "role": "time"},
    {"name": "value", "role": "measure", "roles": ["quantity"], "unit": "varies by indicator (see unit)"},
], "analysis": {"dependencies": {"indicator_id": ["indicator", "unit"], "indicator": ["indicator_id", "unit"]}}}

UNITS_OF = {"GDP": "Billions of dollars", "UNEMP": "Percent", "INFL": "Annual percent change"}
METAS = {"flows": FLOWS, "indicators": INDICATORS}


async def fake_run(sql: str, params: dict, tables: set[str]):
    """Emulates the two lookups units.py issues against the indicators table."""
    assert tables == {"indicators"} and sql.lstrip().upper().startswith("SELECT DISTINCT")
    vals = params.get("vals")
    ids = [i for i in UNITS_OF if vals is None or i in vals]
    if re.search(r'CAST\("indicator_id" AS text\), CAST\("unit"', sql):
        return [[i, UNITS_OF[i]] for i in ids]
    return sorted({(UNITS_OF[i],) for i in ids})


# ---------------------------------------------------------------- one known unit

def test_a_sum_of_usd_values_is_usd_whatever_the_alias():
    for alias in ("total_value_usd", "total_value"):          # the alias need not name the unit
        sql = (f'SELECT "reporter", SUM(CAST("trade_value" AS numeric)) AS "{alias}", '
               f'COUNT(*) AS "rows_aggregated", MIN("year") AS "first_year" FROM "flows" '
               f'WHERE "partner_code" <> 0 GROUP BY "reporter"')
        u = static_units(sql, METAS)
        assert u[alias]["status"] == KNOWN and u[alias]["unit"] == "USD"
        assert u["rows_aggregated"]["status"] == DIMENSIONLESS
        assert u["reporter"]["status"] == NA and u["first_year"]["status"] == NA
        line = units_line({"columns": u})
        assert f"{alias} = USD (sum of trade_value)" in line
        assert "rows_aggregated = dimensionless (count)" in line
        assert "reporter" not in line and "first_year" not in line


def test_a_reshape_over_the_stored_query_keeps_the_unit():
    inner = 'SELECT "reporter", SUM("trade_value") AS "total_value" FROM "flows" GROUP BY "reporter"'
    sql = f'SELECT "reporter", MAX("total_value") AS "max_total_value" FROM ({inner}) AS base GROUP BY "reporter"'
    u = static_units(sql, METAS)
    assert u["max_total_value"]["status"] == KNOWN and u["max_total_value"]["unit"] == "USD"


# ---------------------------------------------------------------- derived results

def test_counts_and_ratios_are_dimensionless_and_differ_from_unknown():
    sql = ('SELECT COUNT(DISTINCT "partner_code") AS "partners", '
           'SUM("trade_value") / SUM("trade_value") AS "share", '
           'SUM("trade_value") / SUM("net_weight") AS "value_per_kg", '
           'SUM("score") AS "total_score", SUM("score") / SUM("trade_value") AS "odd" FROM "flows"')
    u = static_units(sql, METAS)
    assert u["partners"]["status"] == DIMENSIONLESS
    assert u["share"]["status"] == DIMENSIONLESS and "ratio" in u["share"]["what"]
    assert u["value_per_kg"]["status"] == KNOWN and u["value_per_kg"]["unit"] == "USD per kg"
    assert u["total_score"]["status"] == UNKNOWN
    assert u["odd"]["status"] == UNKNOWN
    line = units_line({"columns": u})
    assert "partners = dimensionless (count)" in line
    assert "total_score = unit unknown (the table does not state one)" in line
    assert "total_score = dimensionless" not in line


# ---------------------------------------------------------------- a unit that varies by row

async def test_a_parameterised_indicator_takes_the_selected_indicators_unit():
    sql = ('SELECT "economy", "year", "value" FROM "indicators" '
           'WHERE "indicator_id" = CAST(:ind AS text) AND "economy" = CAST(:eco AS text)')
    u = await result_units(sql, {"ind": "UNEMP", "eco": "A"}, ["economy", "year", "value"],
                           [["A", "2024", "11.3"]], METAS, run=fake_run)
    v = u["columns"]["value"]
    assert v["status"] == KNOWN and v["unit"] == "Percent" and "indicator_id = UNEMP" in v["source"]
    # a literal in the stored SQL (a measure fixed to one indicator) works the same way
    lit = 'SELECT "economy", "value" FROM "indicators" WHERE "indicator_id" = \'INFL\''
    u = await result_units(lit, {}, ["economy", "value"], [["A", "58.5"]], METAS, run=fake_run)
    assert u["columns"]["value"]["unit"] == "Annual percent change"


async def test_units_are_kept_per_row_when_the_rows_differ():
    # the result names the indicator of each row: the unit is looked up per row
    sql = ('SELECT "indicator_id", "economy", "value" FROM "indicators" '
           'WHERE "indicator_id" = ANY(CAST(:ind AS text[])) AND "economy" = CAST(:eco AS text)')
    rows = [["GDP", "A", "512.4"], ["UNEMP", "A", "11.3"]]
    u = await result_units(sql, {"ind": ["GDP", "UNEMP"], "eco": "A"}, ["indicator_id", "economy", "value"],
                           rows, METAS, run=fake_run)
    assert u["columns"]["value"]["status"] == MAPPED
    assert u["row_units"]["value"] == ["Billions of dollars", "Percent"]

    from backend.app.services.structured.retrieval_step import _evidence_text

    res = SqlResult(sql=sql, columns=["indicator_id", "economy", "value"], rows=rows, row_count=2)
    text = _evidence_text({"table": "indicators"}, res, {}, 4000, None, u)
    assert "value = unit differs by row; shown per row as value_unit" in text
    assert "value=512.4; value_unit=Billions of dollars" in text and "value=11.3; value_unit=Percent" in text

    # the result carries the unit column itself: no lookup needed
    sql2 = 'SELECT "economy", "unit", "value" FROM "indicators" WHERE "year" = CAST(:y AS bigint)'
    u2 = await result_units(sql2, {"y": 2024}, ["economy", "unit", "value"], [["A", "Percent", "1"]],
                            METAS, run=None)
    assert u2["columns"]["value"]["status"] == PER_ROW and u2["columns"]["value"]["unit_output"] == "unit"

    # several indicators, but the rows do not say which: never one unit for all
    sql3 = ('SELECT "economy", "value" FROM "indicators" '
            'WHERE "indicator_id" = ANY(CAST(:ind AS text[]))')
    u3 = await result_units(sql3, {"ind": ["GDP", "UNEMP"]}, ["economy", "value"], [["A", "1"], ["A", "2"]],
                            METAS, run=fake_run)
    v3 = u3["columns"]["value"]
    assert v3["status"] == MIXED and v3["units"] == ["Billions of dollars", "Percent"]
    assert "do not state a unit" in units_line(u3)


async def test_a_measure_definition_is_the_fallback_unit_and_lookups_are_optional():
    sql = 'SELECT "economy", "value" FROM "indicators"'
    measure = {"value_column": "value", "unit": "Percent"}
    u = await result_units(sql, {}, ["economy", "value"], [["A", "1"]], METAS, measure=measure, run=None)
    assert u["columns"]["value"]["status"] == KNOWN and u["columns"]["value"]["source"] == "measure definition"


def test_a_quantity_whose_unit_column_is_named_in_its_unit_text_varies():
    u = static_units('SELECT SUM("qty") AS "q" FROM "flows"', METAS)
    assert u["q"]["status"] == "varies" and u["q"]["unit_column"] == "qty_unit"

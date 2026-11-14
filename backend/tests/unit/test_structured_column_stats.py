"""structured.column_stats (measured coverage + metadata card) and the
relative-period anchoring of the retrieval step. No database."""

from __future__ import annotations

from datetime import date

import pytest

from backend.app.services.structured.column_stats import (
    granularity,
    latest_period,
    metadata_card,
    period_ranges,
)
from backend.app.services.structured.retrieval_step import (
    anchor_relative_periods,
    relative_period_question,
)


@pytest.mark.parametrize("value, want", [
    ("2024", "year"), (2024, "year"), ("2024M07", "month"), ("2024-07", "month"),
    ("2024Q3", "quarter"), ("2024-07-15", "day"), (date(2024, 7, 15), "day"), ("abc", None),
])
def test_granularity(value, want):
    assert granularity(value) == want


def test_period_ranges_per_granularity_in_the_values_own_spelling():
    vals = ["1960", "2025", "1999", "1960M01", "2026M10", "2001M12", "junk"]
    assert period_ranges(vals) == {"year": ["1960", "2025"], "month": ["1960M01", "2026M10"]}


META = {
    "table_name": "prices", "row_count": 100, "description": "Commodity prices. More text.",
    "grain": "one price per item and period", "source": {"rel_path": "wb/prices.csv"},
    "columns": [{"name": "item", "type": "text", "role": "dimension", "description": "the item"},
                {"name": "period", "type": "text", "role": "identifier", "description": "YYYY or YYYYMmm"},
                {"name": "value", "type": "numeric", "role": "measure", "description": "price"}],
    "value_domains": {"item": {"values": ["A", "B"]}},
    "column_stats": {"item": {"non_null": 100, "distinct": 2},
                     "period": {"non_null": 100, "distinct": 80,
                                "periods": {"year": ["1960", "2025"], "month": ["1960M01", "2026M10"]}},
                     "value": {"non_null": 99, "distinct": 90, "min": 0.5, "max": 900}},
    "measures": [{"id": "measure_price", "label": "Price", "params": [],
                  "measure": {"period_columns": ["period"], "value_column": "value"},
                  "query": {"select": ["item", "period", "value"],
                            "where": [{"column": "item", "op": "=", "param": "item", "optional": True},
                                      {"column": "period", "op": "between",
                                       "param": ["period_from", "period_to"]}]}}],
}


def test_metadata_card_carries_meaning_and_measured_coverage():
    card = metadata_card(META)
    assert card.startswith("Table prices (source file wb/prices.csv): 100 rows. Commodity prices.")
    assert "- item (text, dimension): the item [2 distinct value(s) in 100 non-empty row(s); values: A | B]" in card
    assert "periods year 1960..2025; month 1960M01..2026M10" in card
    assert "range 0.5..900" in card and "Measures: Price." in card
    assert latest_period(META, "period", "2021") == "2025"
    assert latest_period(META, "period", "2021M03") == "2026M10"


@pytest.mark.parametrize("q, want", [
    ("How have prices changed over the past 3 years?", True),
    ("Compare items for the past 3 years. The current year is 2026.", True),
    ("What is the latest price of copper?", True),
    ("How did prices move between 2021 and 2023?", False),     # names its own window
    ("Prices in the past 3 years up to 2019", False),
    ("What was the price of copper in 2024?", False),          # no relative wording
])
def test_relative_period_questions(q, want):
    assert relative_period_question(q, 2026) is want


def _call(lo, hi):
    return {"table": "prices", "query_id": "measure_price",
            "params": {"item": None, "period_from": lo, "period_to": hi}}


@pytest.mark.parametrize("lo, hi, want", [
    ("2021", "2023", ["2023", "2025"]),            # stale window moved to the latest year
    ("2024", "2026", ["2023", "2025"]),            # past the data's end (no annual 2026)
    ("2021M01", "2023M12", ["2023M11", "2026M10"]),
])
def test_relative_window_ends_at_the_tables_latest_period(lo, hi, want):
    call = _call(lo, hi)
    adj = anchor_relative_periods("compare items for the past 3 years", call, META, today_year=2026)
    assert adj["now"] == want and adj["was"] == [lo, hi]
    assert [call["params"]["period_from"], call["params"]["period_to"]] == want


def test_window_left_alone_when_right_or_explicit():
    call = _call("2023", "2025")
    assert anchor_relative_periods("past 3 years", call, META, today_year=2026) is None
    call = _call("2019", "2021")
    assert anchor_relative_periods("prices from 2019 to 2021", call, META, today_year=2026) is None
    assert call["params"]["period_to"] == "2021"

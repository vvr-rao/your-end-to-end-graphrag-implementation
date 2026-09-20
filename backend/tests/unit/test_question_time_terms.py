"""A question's time terms are parsed with the SAME function ingestion used on
chunk text, so every granularity `enrich-time` minted is reachable.

The query path previously carried its own `\\b(19|20)\\d{2}\\b` regex, so
"Q3 2023" silently degraded to the year and "October 15, 2023" matched a YEAR
row rather than the DAY row that exists.
"""
from __future__ import annotations

import pytest

from backend.app.services.db_temporal_enrich import extract_time_identifiers


@pytest.mark.parametrize(("term", "expected"), [
    ("2023", {"YEAR_2023"}),
    ("Q3 2023", {"Q3_2023"}),
    ("October 2023", {"MONTH_2023_10"}),
    ("2023-10", {"MONTH_2023_10"}),
    ("October 15, 2023", {"DAY_2023_10_15"}),
    ("2023-10-15", {"DAY_2023_10_15"}),
    ("between 2023 and 2024", {"YEAR_2023", "YEAR_2024"}),
])
def test_granularities_the_old_year_regex_lost(term, expected):
    assert extract_time_identifiers(term) == expected


def test_relative_expressions_yield_nothing():
    # No anchor date, so there is nothing honest to resolve them to. Empty is
    # the correct answer, not a failure -- retrieval simply gets no time seed.
    assert extract_time_identifiers("last quarter") == set()
    assert extract_time_identifiers("recently") == set()

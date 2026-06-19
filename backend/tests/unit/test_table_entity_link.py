"""Phase 2a v2: table-to-entity linking helpers.

Covers the pure-Python pieces of `_link_tables_to_entities`:

- `_filter_candidate` -- weeds out strings that can't be entity names
  (pure numbers, dates, generic table-section noise, short tokens).
- `_collect_table_candidates` -- walks a StructuredTable JSON-LD payload
  and pulls out every plausibly-entity-name string from caption, row
  labels, and cell values, normalized + deduplicated.

The async `_link_tables_to_entities` driver is exercised by the
integration-test marker (not run here -- requires the DB).
"""
from __future__ import annotations

import pytest

from backend.app.services.db_entity_extract import (
    _collect_table_candidates,
    _filter_candidate,
    _normalize_name,
)


# ---------- _filter_candidate ----------


@pytest.mark.parametrize(
    "value",
    [
        # Numeric variants
        "78,400",
        "$ 78,400",
        "(120)",
        "12.5%",
        "€1,500",
        "$78,400.00",
        # Date / year prefixes
        "2024",
        "FY2024",
        "Q3 2024",
        "H1 2024",
        # Generic table noise
        "Total",
        "Subtotal",
        "Other",
        "Balance",
        "n/a",
        "N/A",
        "Note",
        "Notes",
        "Yes",
        "Average",
        # Too short
        "",
        "a",
        "X",
        "ll",
        "   ",
        None,
        42,                    # non-string
    ],
)
def test_filter_drops_non_entity_strings(value) -> None:
    assert _filter_candidate(value) is False


@pytest.mark.parametrize(
    "value",
    [
        # Companies
        "BYD Company Ltd.",
        "Toyota Motor Corporation",
        "Apple Inc",
        "OCI N.V.",
        "Taiwan Semiconductor Manufacturing Co.",
        # People
        "Donald Trump",
        "Tim Cook",
        # Places
        "Vietnam",
        "United States",
        "Tokyo",
        # Products / events
        "Strait of Hormuz",
        "Model S",
        # Mixed alphanum but still entity-like
        "Hyundai Mobis Co. (HMC)",
        "BHP Group",
    ],
)
def test_filter_keeps_plausible_entity_strings(value) -> None:
    assert _filter_candidate(value) is True


# ---------- _collect_table_candidates ----------


def _table(caption=None, rows=None):
    payload = {"@type": "viao:StructuredTable", "columns": [], "rows": rows or []}
    if caption is not None:
        payload["caption"] = caption
    return payload


def test_collect_walks_caption_and_row_labels_and_cells() -> None:
    payload = _table(
        caption="Revenue by segment, FY 2024",
        rows=[
            {
                "rowIndex": 0, "rowLabel": "Automotive",
                "cells": [
                    {"cellValue": "Tesla, Inc."},
                    {"cellValue": "78,400"},
                ],
            },
            {
                "rowIndex": 1, "rowLabel": "Energy storage",
                "cells": [
                    {"cellValue": "BYD Company Ltd."},
                    {"cellValue": "12,400"},
                ],
            },
        ],
    )
    cands = _collect_table_candidates(payload)
    norms = {_normalize_name(c) for c in cands}
    # Generic table labels like "Automotive" / "Energy storage" do NOT
    # appear in our stop-list, so they pass through (they could be real
    # entities in some corpora). The DB lookup decides whether they
    # actually match a known entity.
    assert "tesla inc" in norms
    assert "byd company ltd" in norms
    assert "revenue by segment fy 2024" in norms
    # Numeric values must NOT show up.
    assert "78400" not in norms
    assert "12400" not in norms


def test_collect_dedups_by_normalized_name() -> None:
    # Same entity mentioned multiple times -> one entry only.
    payload = _table(
        rows=[
            {"rowIndex": 0, "rowLabel": "BYD Company Ltd.",
             "cells": [{"cellValue": "byd company ltd"}, {"cellValue": "BYD Company Ltd."}]},
            {"rowIndex": 1, "rowLabel": "Toyota",
             "cells": [{"cellValue": "Toyota"}]},
        ],
    )
    cands = _collect_table_candidates(payload)
    norms = sorted({_normalize_name(c) for c in cands})
    assert norms == ["byd company ltd", "toyota"]


def test_collect_safe_on_none_and_garbage() -> None:
    assert _collect_table_candidates(None) == []
    assert _collect_table_candidates([]) == []
    assert _collect_table_candidates({}) == []
    assert _collect_table_candidates({"rows": "not-a-list"}) == []
    # Rows that aren't dicts get skipped.
    payload = {"rows": ["not-a-row", None, {"rowLabel": "BYD", "cells": "x"}]}
    assert _collect_table_candidates(payload) == ["BYD"]


def test_collect_skips_invalid_cell_shapes() -> None:
    payload = _table(rows=[
        {"rowLabel": "Tesla, Inc.", "cells": [None, "not-a-dict", {"cellValue": "BYD"}, {}]},
    ])
    cands = _collect_table_candidates(payload)
    assert cands == ["Tesla, Inc.", "BYD"]


def test_normalize_name_handles_punctuation_and_caps() -> None:
    # _normalize_name is reused by the linker; sanity-check edge cases.
    assert _normalize_name("BYD Company Ltd.") == "byd company ltd"
    assert _normalize_name("  Toyota  Motor  ") == "toyota motor"
    assert _normalize_name("Hyundai Mobis Co. (HMC)") == "hyundai mobis co hmc"

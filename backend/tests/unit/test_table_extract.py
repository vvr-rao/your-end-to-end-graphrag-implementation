"""Table extractor — exercises the in-process helpers (no real PDFs)."""
from __future__ import annotations

import pytest

from backend.app.services import table_extract
from backend.app.services.table_jsonld import (
    VIAO,
    validate_table_jsonld,
)


DOC_SHA = "f" * 64


# ---------- _is_simple_table ----------


def test_simple_grid_passes_complexity_check() -> None:
    grid = [
        ["Segment", "Revenue 2024", "Revenue 2023"],
        ["Automotive", "78,400", "71,500"],
        ["Energy storage", "12,400", "9,800"],
    ]
    assert table_extract._is_simple_table(grid) is True


def test_grid_with_none_cell_is_now_simple_v2() -> None:
    # v2: tables with a FEW empty cells are still routed to the free
    # pdfplumber path -- empty cells are normal in financial tables. The
    # heuristic now coerces None -> "" and accepts.
    grid = [
        ["Segment", "Revenue 2024", "Revenue 2023"],
        ["Automotive", "78,400", "71,500"],
        ["Energy", None, "9,800"],          # empty 2024 cell -- routine
    ]
    assert table_extract._is_simple_table(grid) is True
    # Side-effect: None coerced to "" in-place.
    assert grid[2][1] == ""


def test_grid_with_too_few_non_empty_cells_is_complex() -> None:
    # Nearly empty 3x3 grid -- only 1 non-empty cell. Below the 30% +
    # 4-cell threshold, so route to vision (or drop if vision off).
    grid = [
        ["x", "", ""],
        ["", "", ""],
        ["", "", ""],
    ]
    assert table_extract._is_simple_table(grid) is False


def test_grid_with_sparse_signal_is_complex() -> None:
    # 4x4 grid with only 3 non-empty cells (~19%) is below the 30%
    # threshold; escalate.
    grid = [
        ["x", "", "", ""],
        ["", "", "y", ""],
        ["", "", "", ""],
        ["", "z", "", ""],
    ]
    assert table_extract._is_simple_table(grid) is False


def test_ragged_grid_is_complex() -> None:
    grid = [
        ["a", "b", "c"],
        ["x", "y"],
    ]
    assert table_extract._is_simple_table(grid) is False


def test_single_column_grid_is_complex() -> None:
    grid = [["a"], ["b"], ["c"]]
    assert table_extract._is_simple_table(grid) is False


def test_empty_grid_is_complex() -> None:
    assert table_extract._is_simple_table([]) is False
    assert table_extract._is_simple_table([["a"]]) is False


# ---------- _build_jsonld_from_grid ----------


def test_build_jsonld_from_grid_emits_valid_payload() -> None:
    grid = [
        ["Segment", "Revenue 2024", "Revenue 2023"],
        ["Automotive", "78,400", "71,500"],
        ["Energy storage", "12,400", "9,800"],
    ]
    payload = table_extract._build_jsonld_from_grid(
        grid,
        doc_sha=DOC_SHA, table_index=0,
        caption="Revenue by segment", page_number=12,
    )
    errors = validate_table_jsonld(payload)
    assert errors == []
    # First col is row-label since all data rows are non-numeric in col 0
    assert len(payload["columns"]) == 2
    assert payload["columns"][0]["columnLabel"] == "Revenue 2024"
    assert payload["columns"][1]["columnLabel"] == "Revenue 2023"
    # Row labels picked up from col 0
    assert [r["rowLabel"] for r in payload["rows"]] == ["Automotive", "Energy storage"]
    # Cell values preserved verbatim
    auto_row = payload["rows"][0]
    assert auto_row["cells"][0]["cellValue"] == "78,400"
    assert auto_row["cells"][1]["cellValue"] == "71,500"


def test_build_jsonld_keeps_first_col_when_numeric() -> None:
    # When col 0 is numeric, treat it as a data column rather than a row label.
    grid = [
        ["2022", "2023", "2024"],
        ["100", "120", "140"],
        ["95", "105", "118"],
    ]
    payload = table_extract._build_jsonld_from_grid(
        grid, doc_sha=DOC_SHA, table_index=1,
        caption=None, page_number=1,
    )
    assert len(payload["columns"]) == 3
    assert [r["rowLabel"] for r in payload["rows"]] == [None, None]
    assert payload["rows"][0]["cells"][0]["cellValue"] == "100"


# ---------- _looks_numeric ----------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("1234", True),
        ("1,234.56", True),
        ("$78,400", True),
        ("(120)", True),       # parenthesized negative
        ("12.5%", True),
        ("€1,500", True),
        ("Automotive", False),
        ("North America", False),
        ("", False),
        ("Q4 2024", False),
    ],
)
def test_looks_numeric(s: str, expected: bool) -> None:
    assert table_extract._looks_numeric(s) is expected


# ---------- _vision_body_to_jsonld ----------


def test_vision_body_to_jsonld_round_trips() -> None:
    body = {
        "caption": "Segment results",
        "columns": [
            {"columnIndex": 0, "columnLabel": "Segment"},
            {"columnIndex": 1, "columnLabel": "Revenue"},
        ],
        "rows": [
            {
                "rowIndex": 0,
                "rowLabel": "Automotive",
                "isHeaderRow": False,
                "cells": [
                    {"columnIndex": 0, "cellValue": "Automotive"},
                    {"columnIndex": 1, "cellValue": "78,400"},
                ],
            }
        ],
    }
    payload = table_extract._vision_body_to_jsonld(
        body, doc_sha=DOC_SHA, table_index=2,
        page_number=47, caption_hint=None,
    )
    assert payload is not None
    assert validate_table_jsonld(payload) == []
    assert payload["caption"] == "Segment results"
    assert payload["extractionMethod"] == "vision-llm"


def test_vision_body_drops_cells_with_unknown_column() -> None:
    body = {
        "columns": [{"columnIndex": 0, "columnLabel": "Only column"}],
        "rows": [
            {
                "rowIndex": 0,
                "cells": [
                    {"columnIndex": 0, "cellValue": "v0"},
                    {"columnIndex": 5, "cellValue": "v5"},  # phantom column
                ],
            }
        ],
    }
    payload = table_extract._vision_body_to_jsonld(
        body, doc_sha=DOC_SHA, table_index=3,
        page_number=1, caption_hint=None,
    )
    assert payload is not None
    assert len(payload["rows"][0]["cells"]) == 1
    assert payload["rows"][0]["cells"][0]["cellValue"] == "v0"


def test_vision_body_safe_on_non_dict() -> None:
    assert table_extract._vision_body_to_jsonld(
        "not a dict",  # type: ignore[arg-type]
        doc_sha=DOC_SHA, table_index=0, page_number=1, caption_hint=None,
    ) is None


def test_vision_body_safe_on_missing_keys() -> None:
    payload = table_extract._vision_body_to_jsonld(
        {"caption": "x"}, doc_sha=DOC_SHA, table_index=0,
        page_number=1, caption_hint=None,
    )
    # No columns + no rows → still a valid skeleton with empty arrays.
    assert payload is not None
    assert payload["columns"] == []
    assert payload["rows"] == []


# ---------- bbox filters (headers / footers / thin bands) ----------


def test_bbox_filter_rejects_paperthin_band() -> None:
    page_w, page_h = 612.0, 792.0
    # Paper-thin band (< 2% page height): running text misclassified.
    bbox = (50.0, 100.0, 562.0, 110.0)  # 10 pt = ~1.3% of 792 pt page
    assert table_extract._bbox_filter_reason(bbox, page_w, page_h) is not None


def test_bbox_filter_keeps_small_data_table_near_header() -> None:
    # SEC cover-page table: ~30 pt tall (~4% of page) sitting in the
    # upper area. NOT killed by bbox alone -- content filter decides.
    page_w, page_h = 612.0, 792.0
    bbox = (50.0, 60.0, 562.0, 95.0)
    assert table_extract._bbox_filter_reason(bbox, page_w, page_h) is None


def test_bbox_filter_keeps_centered_data_table() -> None:
    page_w, page_h = 612.0, 792.0
    bbox = (50.0, 200.0, 562.0, 500.0)
    assert table_extract._bbox_filter_reason(bbox, page_w, page_h) is None


def test_bbox_filter_safe_on_zero_page_height() -> None:
    assert table_extract._bbox_filter_reason((0, 0, 1, 1), 612.0, 0.0) is None


def test_content_filter_drops_single_row_header_band() -> None:
    # Running-text header line caught by pdfplumber as a 1x4 "table".
    grid = [["Tesla, Inc.", "", "", ""]]  # 1 substantive cell
    reason = table_extract._content_filter_reason(
        grid, page_number=5, caption_hint=None,
        bbox=(50.0, 40.0, 562.0, 60.0), page_height=792.0,
    )
    assert reason == "header-band"


def test_content_filter_keeps_cover_page_disclosure_table() -> None:
    # The SEC 10-K cover page table has 2+ substantive cells.
    grid = [
        ["Trading Symbol(s)", "Name of each exchange on which registered"],
        ["Common stock: TSLA", "The Nasdaq Global Select Market"],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=1, caption_hint="Securities registered",
        bbox=(50.0, 60.0, 562.0, 95.0), page_height=792.0,
    )
    assert reason is None


def test_content_filter_drops_footer_band() -> None:
    grid = [["12", "", ""]]  # page number footer
    reason = table_extract._content_filter_reason(
        grid, page_number=5, caption_hint=None,
        bbox=(50.0, 750.0, 562.0, 770.0), page_height=792.0,
    )
    assert reason == "footer-band"


# ---------- content filters (TOC / index / bibliography) ----------


def test_content_filter_drops_toc_pattern() -> None:
    # Classic TOC: section name in col 0, page number in col 1.
    grid = [
        ["Item 1. Business", "1"],
        ["Item 1A. Risk Factors", "12"],
        ["Item 2. Properties", "30"],
        ["Item 3. Legal Proceedings", "31"],
        ["Item 4. Mine Safety Disclosures", "32"],
        ["Item 5. Market for Registrant's Common Equity", "33"],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=2, caption_hint=None,
    )
    assert reason == "toc-or-index"


def test_content_filter_drops_index_with_roman_pagenums() -> None:
    grid = [
        ["Introduction", "i"],
        ["Methodology", "ii"],
        ["Findings", "iii"],
        ["Conclusions", "v"],
        ["Appendix", "ix"],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=3, caption_hint=None,
    )
    assert reason == "toc-or-index"


def test_content_filter_drops_when_caption_says_toc() -> None:
    grid = [
        ["Cash and equivalents", "$ 100", "$ 95"],
        ["Receivables", "$ 50", "$ 48"],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=5, caption_hint="Table of Contents",
    )
    assert reason == "caption-toc-index-bib"


def test_content_filter_drops_bibliography() -> None:
    grid = [
        ["Smith, J. et al. 'Global supply chains'.", ""],
        ["Jones, B. et al. (2024). Vol. 12 pp. 45-67.", ""],
        ["Brown, A. https://doi.org/10.1234/abc", ""],
        ["Garcia, M. et al., ISBN 978-1-23456-789-0", ""],
        ["Cited in proceedings of ABC conference 2024", ""],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=300, caption_hint=None,
    )
    assert reason == "bibliography"


def test_content_filter_keeps_real_data_table() -> None:
    grid = [
        ["Segment", "Revenue 2024", "Revenue 2023", "Change"],
        ["Automotive", "78,400", "71,500", "9.6%"],
        ["Energy storage", "12,400", "9,800", "26.5%"],
        ["Services", "8,500", "7,200", "18.1%"],
    ]
    reason = table_extract._content_filter_reason(
        grid, page_number=47, caption_hint="Revenue by segment",
    )
    assert reason is None


def test_content_filter_safe_on_empty_grid() -> None:
    assert table_extract._content_filter_reason(
        [], page_number=1, caption_hint=None,
    ) is None
    # Too small to classify confidently — fall through to caller.
    assert table_extract._content_filter_reason(
        [["a", "b"]], page_number=1, caption_hint=None,
    ) is None

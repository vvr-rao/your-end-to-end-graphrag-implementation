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


# ---------- fragment merging (_merge_row_fragments) ----------
#
# SEC filings rule each row of figures but draw no vertical rules and no
# outer box, so pdfplumber reports one single-row candidate per ruled row.
# Each fragment is ~15 pt tall and dies to the 2% thin-band guard, taking
# the whole table with it. These cover the reassembly.


class _FakeRow:
    def __init__(self, cells):
        self.cells = cells


class _FakeTable:
    """Minimal stand-in for `pdfplumber.table.Table`."""

    def __init__(self, bbox, grid, col_edges=None):
        self.bbox = bbox
        self._grid = grid
        if col_edges is None:
            x0, _, x1 = bbox[0], bbox[1], bbox[2]
            n = max((len(r) for r in grid), default=1)
            step = (x1 - x0) / n
            col_edges = [x0 + i * step for i in range(n + 1)]
        cells = [
            (col_edges[i], bbox[1], col_edges[i + 1], bbox[3])
            for i in range(len(col_edges) - 1)
        ]
        self.rows = [_FakeRow(cells)] if grid else []

    def extract(self):
        return self._grid


def _stacked_fragments():
    """Four consecutive single-row fragments of one AOCI-style table."""
    edges = [36.0, 300.0, 400.0, 500.0]
    return [
        _FakeTable((36.0, 133.0, 500.0, 158.0),
                   [["Costs of services", "17,944", "5,545"]], edges),
        _FakeTable((36.0, 183.0, 500.0, 198.0),
                   [["Depreciation", "1,964", "647"]], edges),
        _FakeTable((36.0, 213.0, 500.0, 228.0),
                   [["Interest expense, net", "246", "82"]], edges),
        _FakeTable((36.0, 243.0, 500.0, 258.0),
                   [["Other income, net", "(82)", "(10)"]], edges),
    ]


def test_merge_reassembles_stacked_single_row_fragments() -> None:
    out = table_extract._merge_row_fragments(_stacked_fragments())
    assert len(out) == 1
    bbox, grid, _bounds, n_frag = out[0]
    assert n_frag == 4
    assert len(grid) == 4
    # Row order follows page order, top to bottom.
    assert [r[0] for r in grid] == [
        "Costs of services", "Depreciation",
        "Interest expense, net", "Other income, net",
    ]
    # The merged bbox spans every fragment, so it clears the thin-band guard
    # that each individual fragment failed.
    assert bbox == (36.0, 133.0, 500.0, 258.0)
    assert table_extract._bbox_filter_reason(bbox, 612.0, 792.0) is None


def test_unmerged_fragments_are_dropped_as_thin() -> None:
    """Establishes the bug the merge exists to fix: a standard 15 pt ruled
    row is 1.89% of a 792 pt page, just under the 2% guard, so every
    single-line fragment dies on its own."""
    frags = _stacked_fragments()
    reasons = [
        table_extract._bbox_filter_reason(f.bbox, 612.0, 792.0) for f in frags
    ]
    # The lead fragment wraps onto two lines (25 pt) and clears the guard;
    # the three single-line rows beneath it do not.
    assert reasons == [None, "thin-band", "thin-band", "thin-band"]
    # And the survivor is a lone single-row grid, which `_is_simple_table`
    # rejects -- so pre-merge this table reached the pdfplumber path with
    # nothing, and was handed to the vision LLM one row at a time.
    assert table_extract._is_simple_table(frags[0].extract()) is False


def test_merge_refuses_misaligned_columns() -> None:
    edges_a = [36.0, 300.0, 400.0, 500.0]
    edges_b = [90.0, 320.0, 420.0, 500.0]   # different column grid
    tables = [
        _FakeTable((36.0, 133.0, 500.0, 148.0), [["a", "1", "2"]], edges_a),
        _FakeTable((90.0, 158.0, 500.0, 173.0), [["b", "3", "4"]], edges_b),
    ]
    assert len(table_extract._merge_row_fragments(tables)) == 2


def test_merge_refuses_distant_fragments() -> None:
    edges = [36.0, 300.0, 400.0, 500.0]
    tables = [
        _FakeTable((36.0, 133.0, 500.0, 148.0), [["a", "1", "2"]], edges),
        # 200 pt below: a different table further down the page.
        _FakeTable((36.0, 348.0, 500.0, 363.0), [["b", "3", "4"]], edges),
    ]
    assert len(table_extract._merge_row_fragments(tables)) == 2


def test_merge_leaves_multi_row_candidates_untouched() -> None:
    """The safety property: any candidate that already extracts >= 2 rows
    passes through unchanged, so fully-ruled corpora cannot regress."""
    grid = [["Segment", "2024", "2023"], ["Auto", "78,400", "71,500"]]
    tables = [_FakeTable((36.0, 100.0, 500.0, 300.0), grid)]
    out = table_extract._merge_row_fragments(tables)
    assert len(out) == 1
    bbox, out_grid, _bounds, n_frag = out[0]
    assert n_frag == 1
    assert out_grid == grid
    assert bbox == (36.0, 100.0, 500.0, 300.0)


def test_merge_passes_lone_single_row_through() -> None:
    edges = [36.0, 300.0, 400.0, 500.0]
    tables = [_FakeTable((36.0, 133.0, 500.0, 148.0), [["a", "1", "2"]], edges)]
    out = table_extract._merge_row_fragments(tables)
    assert len(out) == 1
    assert out[0][3] == 1


def test_merge_handles_empty_input() -> None:
    assert table_extract._merge_row_fragments([]) == []


# ---------- header recovery for merged tables ----------


def test_merged_grid_uses_recovered_header_not_first_data_row() -> None:
    """Fragments are all DATA rows -- the header sits above the ruled
    region. Without the recovered header the first line item would be
    eaten as the column labels."""
    grid = [
        ["Costs of services", "17,944", "5,545"],
        ["Depreciation", "1,964", "647"],
    ]
    payload = table_extract._build_jsonld_from_grid(
        grid, doc_sha=DOC_SHA, table_index=0, caption=None, page_number=1,
        header_row=["(in millions)", "2018", "2017"],
    )
    assert validate_table_jsonld(payload) == []
    assert [c["columnLabel"] for c in payload["columns"]] == ["2018", "2017"]
    # Both line items survive as rows; neither was consumed as a header.
    assert [r["rowLabel"] for r in payload["rows"]] == [
        "Costs of services", "Depreciation",
    ]


def test_bad_header_width_falls_back_to_legacy_heuristic() -> None:
    """A mis-recovered header must degrade to the old behaviour rather
    than misalign every column."""
    grid = [
        ["Segment", "2024", "2023"],
        ["Auto", "78,400", "71,500"],
    ]
    payload = table_extract._build_jsonld_from_grid(
        grid, doc_sha=DOC_SHA, table_index=0, caption=None, page_number=1,
        header_row=["only", "two"],          # width 2 vs grid width 3
    )
    assert validate_table_jsonld(payload) == []
    assert [c["columnLabel"] for c in payload["columns"]] == ["2024", "2023"]
    assert [r["rowLabel"] for r in payload["rows"]] == ["Auto"]


def test_no_header_row_preserves_existing_behaviour() -> None:
    """header_row=None must produce exactly what the extractor produced
    before the merge work landed."""
    grid = [
        ["Segment", "2024", "2023"],
        ["Auto", "78,400", "71,500"],
    ]
    with_default = table_extract._build_jsonld_from_grid(
        grid, doc_sha=DOC_SHA, table_index=0, caption=None, page_number=1,
    )
    with_explicit_none = table_extract._build_jsonld_from_grid(
        grid, doc_sha=DOC_SHA, table_index=0, caption=None, page_number=1,
        header_row=None,
    )
    assert with_default == with_explicit_none
    assert [c["columnLabel"] for c in with_default["columns"]] == ["2024", "2023"]


def test_content_filter_keeps_financial_table_with_plain_numeric_last_col() -> None:
    """Regression: the TOC heuristic used to accept thousands-separated
    figures as page numbers, so a product-sales table whose last column
    read '3,324 / 2,667 / 1,363' was dropped as a table of contents. Only
    surfaced once fragment merging started producing whole tables."""
    grid = [
        ["Total Sales", "$ 44,033", "$ 47,267", "$ 48,047"],
        ["Januvia", "4,004", "4,086", "3,324"],
        ["Remicade", "2,271", "2,076", "2,667"],
        ["Janumet", "1,829", "1,659", "1,363"],
        ["Vytorin", "1,643", "1,747", "1,882"],
        ["Zetia", "2,658", "2,566", "3,253"],
    ]
    assert table_extract._content_filter_reason(
        grid, page_number=40, caption_hint=None,
    ) is None


def test_content_filter_keeps_table_with_long_unseparated_figures() -> None:
    grid = [
        ["Revenue", "48047", "47267"],
        ["Cost of sales", "16121", "16446"],
        ["Gross margin", "31926", "30821"],
        ["Operating expenses", "24193", "23458"],
        ["Operating income", "7733", "7363"],
    ]
    assert table_extract._content_filter_reason(
        grid, page_number=41, caption_hint=None,
    ) is None


# ---------- caption vs header collision ----------


@pytest.mark.parametrize("line,header_like", [
    ("2013 2012 2011", True),
    ("(in millions) 2018 2017(1) 2016(1)", True),
    ("March 31, March 31, April 1,", True),
    ("$ in millions 2013 2012 2011", True),
    ("A reconciliation of the beginning and ending amount of unrecognized "
     "tax benefits is as follows:", False),
    ("Sales of the Company's top pharmaceutical products were as follows:", False),
    ("Our total costs and expenses were as follows:", False),
])
def test_looks_header_like(line: str, header_like: bool) -> None:
    assert table_extract._looks_header_like(line) is header_like


def test_caption_hint_default_takes_last_line() -> None:
    """Non-merged candidates must keep the original behaviour exactly."""
    class _Crop:
        def extract_text(self):
            return "Some prose above\n2013 2012 2011"

    class _Page:
        def crop(self, bbox):
            return _Crop()

    assert table_extract._caption_hint(
        _Page(), (0.0, 100.0, 500.0, 200.0),
    ) == "2013 2012 2011"


def test_caption_hint_skips_header_block_when_asked() -> None:
    """Merged candidates walk back past the whole header block -- a
    spanning header can run several lines."""
    class _Crop:
        def extract_text(self):
            return (
                "Our total costs and expenses were as follows:\n"
                "Fiscal Years Ended Percentage of Revenues\n"
                "March 31, March 31, April 1,\n"
                "(in millions) 2018 2017(1) 2016(1)"
            )

    class _Page:
        def crop(self, bbox):
            return _Crop()

    got = table_extract._caption_hint(
        _Page(), (0.0, 100.0, 500.0, 200.0), skip_header_lines=True,
    )
    assert got == "Fiscal Years Ended Percentage of Revenues"


def test_caption_hint_falls_back_when_every_line_is_header_like() -> None:
    class _Crop:
        def extract_text(self):
            return "2013 2012 2011\n(in millions) 1 2 3"

    class _Page:
        def crop(self, bbox):
            return _Crop()

    got = table_extract._caption_hint(
        _Page(), (0.0, 100.0, 500.0, 200.0), skip_header_lines=True,
    )
    # No prose to find; returns the last line rather than None.
    assert got == "(in millions) 1 2 3"

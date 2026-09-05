"""Table -> ontology concept mining.

Covers the row-label fix: in a financial statement the columns are fiscal
periods and the ROWS carry the line items, so rows have to be proposal
targets rather than mere prompt context.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from backend.app.services import table_ontology_mining as tom

MEASURE = "https://veerla-ramrao.ai/ontology/domain-concepts#Measure"
TIME = "https://veerla-ramrao.ai/ontology/domain-concepts#TimePeriod"


def _table(table_id: str, columns, rows) -> dict[str, Any]:
    return {
        "@id": table_id,
        "caption": "Total costs and expenses",
        "columns": [
            {"columnIndex": i, "columnLabel": lab}
            for i, lab in enumerate(columns)
        ],
        "rows": [
            {"rowIndex": i, "rowLabel": lab, "isHeaderRow": False, "cells": []}
            for i, lab in enumerate(rows)
        ],
    }


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRouter:
    """Records the prompts it is handed and replays a canned response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[tuple[str, str]] = []

    async def chat(self, task: str, *, system: str, user: str):
        self.calls.append((system, user))
        return _FakeResult(json.dumps(self._payload))


# ---------- _extract_table_fields ----------


def test_row_labels_are_no_longer_capped_at_three() -> None:
    """The old cap of 3 was sized for disambiguating context. Once rows
    drive proposals it would silently discard most line items."""
    rows = [f"Line item {i}" for i in range(20)]
    t = _table("t1", ["2018", "2017"], rows)
    _tid, _cap, _cols, row_labels = tom._extract_table_fields(t)
    assert len(row_labels) == 20
    assert row_labels[0] == "Line item 0"


def test_row_labels_are_bounded_by_max_row_labels() -> None:
    rows = [f"Line item {i}" for i in range(200)]
    t = _table("t1", ["2018"], rows)
    *_rest, row_labels = tom._extract_table_fields(t)
    assert len(row_labels) == tom._MAX_ROW_LABELS


def test_header_and_blank_rows_are_skipped() -> None:
    t = _table("t1", ["2018"], ["Revenue", "", "Depreciation"])
    t["rows"][0]["isHeaderRow"] = True
    *_rest, row_labels = tom._extract_table_fields(t)
    assert row_labels == ["Depreciation"]


# ---------- _classify_one_table gate ----------


@pytest.mark.asyncio
async def test_table_with_rows_but_no_columns_is_still_classified() -> None:
    """Vision extraction routinely returns `columns: []` with usable row
    labels. Requiring columns dropped those tables whole -- not even the
    FinancialTable subclass survived."""
    t = _table("t-no-cols", [], ["Costs of services", "Depreciation"])
    router = _FakeRouter({
        "table_class": {"proposed_label": "CostsTable", "definition": "d"},
        "columns": [],
        "rows": [{"row_label": "Costs of services", "parent_iri": MEASURE,
                  "proposed_label": "CostsOfServices", "definition": "d"}],
    })
    out = await tom._classify_one_table(t, router, None)
    assert out is not None
    assert out["rows"][0]["proposed_label"] == "CostsOfServices"


@pytest.mark.asyncio
async def test_table_with_neither_axis_is_skipped() -> None:
    t = _table("t-empty", [], [])
    router = _FakeRouter({})
    assert await tom._classify_one_table(t, router, None) is None
    assert router.calls == []          # no LLM spend on an unusable table


# ---------- mine_table_concepts_async: rows become proposals ----------


@pytest.mark.asyncio
async def test_row_labels_become_class_proposals(tmp_path, monkeypatch) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table(
            "https://x/table/1",
            ["2018", "2017"],
            ["Costs of services", "Depreciation and amortization"],
        )],
        "manifest": {},
    }), encoding="utf-8")

    router = _FakeRouter({
        "table_class": {"proposed_label": "CostsAndExpensesTable",
                        "definition": "Costs and expenses by period."},
        # A fiscal year is correctly a TimePeriod, not a subject concept.
        "columns": [{"column_index": 0, "parent_iri": TIME,
                     "proposed_label": "FY2018", "definition": "Fiscal 2018."}],
        "rows": [
            {"row_label": "Costs of services", "parent_iri": MEASURE,
             "proposed_label": "CostsOfServices",
             "definition": "Cost of delivering services."},
            {"row_label": "Depreciation and amortization",
             "parent_iri": MEASURE,
             "proposed_label": "DepreciationAndAmortization",
             "definition": "Periodic D&A charge."},
        ],
    })

    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    labels = {p["LABEL"] for p in out["MATCH NOT FOUND"]}
    # The line items -- the whole point of the fix -- are now proposed.
    assert "CostsOfServices" in labels
    assert "DepreciationAndAmortization" in labels
    # And the table-level + column proposals still come through.
    assert "CostsAndExpensesTable" in labels
    assert "FY2018" in labels


@pytest.mark.asyncio
async def test_row_proposal_snippet_is_the_verbatim_row_label(
    tmp_path,
) -> None:
    """Stage 3 dedup sees the source text a proposal came from, not the
    CamelCase label we invented for it."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/table/1", ["2018"], ["Costs of services"])],
        "manifest": {},
    }), encoding="utf-8")

    # An ontology that already has this class -> layer-1 reuse path, which
    # is where the snippet is surfaced.
    loaded = {"classes_dict": {
        "https://x/onto#CostsOfServices": {"name": "CostsOfServices",
                                           "labels": ["CostsOfServices"]},
    }}
    router = _FakeRouter({
        "table_class": {"proposed_label": "Zzz", "definition": "d"},
        "columns": [],
        "rows": [{"row_label": "Costs of services", "parent_iri": MEASURE,
                  "proposed_label": "CostsOfServices", "definition": "d"}],
    })
    out = await tom.mine_table_concepts_async(
        tables_dir, loaded, router, cache_dir=None,
    )
    snippets = [m["TEXT_SNIPPET"] for m in out["MATCHES FOUND"]]
    assert "Costs of services" in snippets


@pytest.mark.asyncio
async def test_missing_rows_key_is_tolerated(tmp_path) -> None:
    """Back-compat: a cached response from the previous prompt version has
    no `rows` key and must still yield its table + column proposals."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/table/1", ["Segment"], ["Auto"])],
        "manifest": {},
    }), encoding="utf-8")
    router = _FakeRouter({
        "table_class": {"proposed_label": "SegmentTable", "definition": "d"},
        "columns": [{"column_index": 0, "parent_iri": MEASURE,
                     "proposed_label": "ReportingSegment", "definition": "d"}],
    })
    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    labels = {p["LABEL"] for p in out["MATCH NOT FOUND"]}
    assert labels == {"SegmentTable", "ReportingSegment"}


# ---------- structural-row filter ----------
#
# The prompt asks the model to omit totals and subtotals; a live smoke run
# showed it does not comply (TotalSalesUSDM, BalanceJanuary1 and
# BalanceDecember31 all came through). These lock in the deterministic
# filter that backstops the instruction.


@pytest.mark.parametrize("source,proposed", [
    ("Total Sales", "TotalSalesUSDM"),
    ("Total", "Total"),
    ("Totals", "Totals"),
    ("Subtotal", "Subtotal"),
    ("Sub-total", "SubTotal"),
    ("Grand total revenue", "GrandTotalRevenue"),
    ("Net total", "NetTotal"),
    ("Balance January 1", "BalanceJanuary1"),
    ("Balance December 31", "BalanceDecember31"),
    ("Balance at beginning of fiscal year", "BalanceAtBeginningOfFiscalYear"),
    ("Opening balance", "OpeningBalance"),
    ("Closing balance", "ClosingBalance"),
    ("Ending balance", "EndingBalance"),
])
def test_structural_row_labels_are_detected(source, proposed) -> None:
    assert tom._is_structural_row_label(source, proposed) is True


@pytest.mark.parametrize("source,proposed", [
    ("Costs of services", "CostsOfServices"),
    ("Depreciation and amortization", "DepreciationAndAmortization"),
    ("Januvia", "JanuviaSalesUSDM"),
    ("Interest expense, net", "InterestExpenseNet"),
    ("Settlements", "Settlements"),
    ("Additions related to prior year positions", "AdditionsPriorYearPositions"),
    ("Revenue", "RevenueUSDM"),
])
def test_real_line_items_survive_the_structural_filter(source, proposed) -> None:
    assert tom._is_structural_row_label(source, proposed) is False


def test_known_false_positive_total_prefixed_metrics() -> None:
    """Documents a real limitation rather than hiding it.

    'Total shareholder return' is a genuine metric, not a sum over sibling
    rows, but it is indistinguishable from 'Total sales' by label alone --
    telling them apart needs the cell arithmetic, which the mining stage
    never sees (it works from labels only). The filter errs toward dropping,
    on the reasoning that a spurious 'sum of my siblings' class is worse
    than a missing one. Revisit if a corpus turns out to be rich in
    Total-prefixed metrics."""
    assert tom._is_structural_row_label(
        "Total shareholder return", "TotalShareholderReturn"
    ) is True


@pytest.mark.asyncio
async def test_structural_filter_applies_to_rows_only(tmp_path) -> None:
    """A COLUMN labelled 'Total' is a dimension member, not a summed row,
    so the filter must not touch it."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/t/1", ["Total"], ["Januvia"])],
        "manifest": {},
    }), encoding="utf-8")
    router = _FakeRouter({
        "table_class": {"proposed_label": "SalesTable", "definition": "d"},
        "columns": [{"column_index": 0, "parent_iri": MEASURE,
                     "proposed_label": "TotalColumn", "definition": "d"}],
        "rows": [{"row_label": "Januvia", "parent_iri": MEASURE,
                  "proposed_label": "JanuviaSalesUSDM", "definition": "d"}],
    })
    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    labels = {p["LABEL"] for p in out["MATCH NOT FOUND"]}
    assert "TotalColumn" in labels
    assert "JanuviaSalesUSDM" in labels


@pytest.mark.asyncio
async def test_total_rows_are_dropped_from_proposals(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/t/1", ["2013"],
                          ["Total Sales", "Januvia", "Balance December 31"])],
        "manifest": {},
    }), encoding="utf-8")
    router = _FakeRouter({
        "table_class": {"proposed_label": "ProductSalesTable", "definition": "d"},
        "columns": [],
        "rows": [
            {"row_label": "Total Sales", "parent_iri": MEASURE,
             "proposed_label": "TotalSalesUSDM", "definition": "d"},
            {"row_label": "Januvia", "parent_iri": MEASURE,
             "proposed_label": "JanuviaSalesUSDM", "definition": "d"},
            {"row_label": "Balance December 31", "parent_iri": MEASURE,
             "proposed_label": "BalanceDecember31", "definition": "d"},
        ],
    })
    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    labels = {p["LABEL"] for p in out["MATCH NOT FOUND"]}
    assert "JanuviaSalesUSDM" in labels
    assert "TotalSalesUSDM" not in labels
    assert "BalanceDecember31" not in labels


# ---------- domain rejection (`no_anchor_fits`) ----------
#
# The 6 anchors describe FINANCIAL reporting. Before this, `parent_iri` for
# the table class was hardcoded to FinancialTable and row/column concepts
# were filed under Measure/Dimension whatever the domain -- a drug label
# produced NonFatalStroke and Nausea as siblings of RevenueUSDM. Phase 2
# cannot correct the taxonomy later, so minting nothing is the right answer.


@pytest.mark.asyncio
async def test_non_financial_table_mints_nothing(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table(
            "https://x/t/1", ["Placebo", "OZEMPIC 1mg"],
            ["Nausea", "Vomiting", "NonFatalStroke"],
        )],
        "manifest": {},
    }), encoding="utf-8")
    router = _FakeRouter({"no_anchor_fits": True})
    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    assert out["MATCH NOT FOUND"] == []
    assert out["MATCHES FOUND"] == []


@pytest.mark.asyncio
async def test_rejection_does_not_suppress_other_tables(tmp_path) -> None:
    """A mixed corpus must still mine its financial tables."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "pharma.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/t/pharma", ["Placebo"], ["Nausea"])],
        "manifest": {},
    }), encoding="utf-8")
    (tables_dir / "finance.jsonld").write_text(json.dumps({
        "doc_sha": "b" * 64,
        "tables": [_table("https://x/t/fin", ["2013"], ["Costs of services"])],
        "manifest": {},
    }), encoding="utf-8")

    class _PerTableRouter:
        async def chat(self, task, *, system, user):
            if "Nausea" in user:
                return _FakeResult(json.dumps({"no_anchor_fits": True}))
            return _FakeResult(json.dumps({
                "table_class": {"proposed_label": "CostsAndExpensesTable",
                                "definition": "d"},
                "columns": [],
                "rows": [{"row_label": "Costs of services",
                          "parent_iri": MEASURE,
                          "proposed_label": "CostsOfServices",
                          "definition": "d"}],
            }))

    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, _PerTableRouter(), cache_dir=None,
    )
    labels = {p["LABEL"] for p in out["MATCH NOT FOUND"]}
    assert "CostsOfServices" in labels
    assert "CostsAndExpensesTable" in labels
    # Nothing clinical leaked through.
    assert not any("Nausea" in lb for lb in labels)


@pytest.mark.asyncio
async def test_rejection_flag_must_be_exactly_true(tmp_path) -> None:
    """A truthy-but-not-True value must not silently discard a table."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "doc.jsonld").write_text(json.dumps({
        "doc_sha": "a" * 64,
        "tables": [_table("https://x/t/1", ["2013"], ["Costs of services"])],
        "manifest": {},
    }), encoding="utf-8")
    router = _FakeRouter({
        "no_anchor_fits": "false",          # a string, not a bool
        "table_class": {"proposed_label": "CostsTable", "definition": "d"},
        "columns": [],
        "rows": [{"row_label": "Costs of services", "parent_iri": MEASURE,
                  "proposed_label": "CostsOfServices", "definition": "d"}],
    })
    out = await tom.mine_table_concepts_async(
        tables_dir, {"classes_dict": {}}, router, cache_dir=None,
    )
    assert "CostsOfServices" in {p["LABEL"] for p in out["MATCH NOT FOUND"]}

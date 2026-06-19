"""StructuredTable JSON-LD schema + helpers."""
from __future__ import annotations

import pytest

from backend.app.services.table_jsonld import (
    VIAO,
    build_cell_iri,
    build_column_iri,
    build_document_iri,
    build_row_iri,
    build_table_iri,
    empty_payload,
    flat_text_summary,
    validate_table_jsonld,
)


DOC_SHA = "a" * 64


def _full_payload() -> dict:
    payload = empty_payload(
        DOC_SHA, 0,
        extraction_method="pdfplumber",
        caption="Revenue by segment",
        page_number=47,
    )
    table_iri = build_table_iri(DOC_SHA, 0)
    payload["columns"] = [
        {"@id": build_column_iri(table_iri, 0), "@type": "viao:TableColumn",
         "columnIndex": 0, "columnLabel": "Segment"},
        {"@id": build_column_iri(table_iri, 1), "@type": "viao:TableColumn",
         "columnIndex": 1, "columnLabel": "Revenue 2024 (USD M)"},
    ]
    payload["rows"] = [
        {
            "@id": build_row_iri(table_iri, 0),
            "@type": "viao:TableRow",
            "rowIndex": 0,
            "rowLabel": "Automotive",
            "isHeaderRow": False,
            "cells": [
                {"@id": build_cell_iri(table_iri, 0, 0),
                 "@type": "viao:TableCell",
                 "inColumn": payload["columns"][0]["@id"],
                 "cellValue": "Automotive"},
                {"@id": build_cell_iri(table_iri, 0, 1),
                 "@type": "viao:TableCell",
                 "inColumn": payload["columns"][1]["@id"],
                 "cellValue": "78,400"},
            ],
        }
    ]
    return payload


def test_iri_builders_are_deterministic() -> None:
    table_a = build_table_iri(DOC_SHA, 3)
    table_b = build_table_iri(DOC_SHA, 3)
    assert table_a == table_b
    assert table_a.startswith(f"{VIAO}StructuredTable_")
    assert build_column_iri(table_a, 0).startswith(table_a + "#col_0")
    assert build_row_iri(table_a, 5).startswith(table_a + "#row_5")
    assert build_cell_iri(table_a, 5, 0).startswith(table_a + "#cell_5_0")
    assert build_document_iri(DOC_SHA).startswith(f"{VIAO}Document_")


def test_empty_payload_has_context_and_required_keys() -> None:
    p = empty_payload(DOC_SHA, 0, extraction_method="pdfplumber")
    assert p["@context"]
    assert p["@id"].startswith(f"{VIAO}StructuredTable_")
    assert p["@type"] == "viao:StructuredTable"
    assert p["extractionMethod"] == "pdfplumber"
    assert p["derivedFromDocument"]["@id"].startswith(f"{VIAO}Document_")
    assert p["columns"] == []
    assert p["rows"] == []


def test_validate_accepts_full_payload() -> None:
    errors = validate_table_jsonld(_full_payload())
    assert errors == []


def test_validate_rejects_missing_required_keys() -> None:
    p = _full_payload()
    del p["columns"]
    errors = validate_table_jsonld(p)
    assert errors
    assert any("columns" in e for e in errors)


def test_validate_rejects_wrong_type() -> None:
    p = _full_payload()
    p["@type"] = "viao:Document"
    errors = validate_table_jsonld(p)
    assert errors


def test_validate_rejects_negative_indices() -> None:
    p = _full_payload()
    p["rows"][0]["rowIndex"] = -1
    errors = validate_table_jsonld(p)
    assert errors


def test_validate_rejects_bad_extraction_method() -> None:
    p = _full_payload()
    p["extractionMethod"] = "what-is-this"
    errors = validate_table_jsonld(p)
    assert errors


def test_validate_safe_on_non_dict() -> None:
    assert validate_table_jsonld(None) == ["payload is not a dict"]
    assert validate_table_jsonld([]) == ["payload is not a dict"]


def test_flat_text_summary_captures_caption_and_headers() -> None:
    p = _full_payload()
    summary = flat_text_summary(p)
    assert "Revenue by segment" in summary
    assert "Segment" in summary
    assert "Revenue 2024 (USD M)" in summary
    assert "Automotive" in summary
    assert "78,400" in summary


def test_flat_text_summary_safe_on_empty() -> None:
    assert flat_text_summary({}) == ""
    assert flat_text_summary({"caption": "x"}).strip() == "x"

"""JSON-LD schema for `viao:StructuredTable` payloads (Phase 2a).

A StructuredTable is the JSON-LD serialisation of one table extracted
from a source document. Nested tables are supported (a cell may carry
`hasNestedTable` referencing another StructuredTable by `@id`).

The schema deliberately tracks the VIAO ontology one-for-one so that
exporting these payloads to RDF later is mechanical: every JSON-LD
key corresponds to a real VIAO predicate. The `@context` block at the
top of every payload names the predicate -> short-key mapping.

`build_table_iri` / `build_column_iri` / `build_row_iri` /
`build_cell_iri` produce deterministic IRIs from
(document_sha, table_index, column/row/cell index). Re-running the
extractor on the same document yields the same IRIs -- important for
the cache + Phase 2 upsert path.

`validate_table_jsonld` is fail-soft: it returns a list of validation
errors (empty list = OK). Callers DROP invalid tables and log a soft
failure to `llm_audit.jsonl` rather than raising; the pipeline must
not break because of a malformed table.
"""
from __future__ import annotations

from typing import Any

import jsonschema


VIAO = "https://veerla-ramrao.ai/ontology/intelligence-artifact#"
ENTITIES = "https://veerla-ramrao.ai/ontology/entities#"

# Fixed @context for every StructuredTable payload. Maps the short JSON
# keys we emit to the canonical VIAO predicate IRIs so the payload
# round-trips to RDF.
CONTEXT: dict[str, Any] = {
    "viao": VIAO,
    "schema": "http://schema.org/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "caption": "viao:tableCaption",
    "pageNumber": {"@id": "viao:pageNumber", "@type": "xsd:integer"},
    "extractionMethod": "viao:extractionMethod",
    "derivedFromDocument": {"@id": "viao:derivedFromDocument", "@type": "@id"},
    "columns": "viao:hasColumn",
    "rows": "viao:hasRow",
    "cells": "viao:hasCell",
    "inColumn": {"@id": "viao:inColumn", "@type": "@id"},
    "nestedTable": {"@id": "viao:hasNestedTable", "@type": "@id"},
    "columnLabel": "viao:columnLabel",
    "columnIndex": {"@id": "viao:columnIndex", "@type": "xsd:integer"},
    "rowLabel": "viao:rowLabel",
    "rowIndex": {"@id": "viao:rowIndex", "@type": "xsd:integer"},
    "isHeaderRow": {"@id": "viao:isHeaderRow", "@type": "xsd:boolean"},
    "cellValue": "viao:cellValue",
}


# JSON-schema describing the shape we emit. Strict enough to reject
# malformed payloads from the vision LLM; loose enough to allow optional
# fields (caption, pageNumber, nested tables) to be omitted.
JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["@id", "@type", "columns", "rows"],
    "properties": {
        "@context": {"type": ["object", "string", "array"]},
        "@id": {"type": "string", "pattern": f"^{VIAO}StructuredTable_"},
        "@type": {"const": "viao:StructuredTable"},
        "caption": {"type": ["string", "null"]},
        "pageNumber": {"type": ["integer", "null"], "minimum": 0},
        "extractionMethod": {
            "type": "string",
            "enum": ["pdfplumber", "vision-llm", "manual"],
        },
        "derivedFromDocument": {
            "type": "object",
            "required": ["@id"],
            "properties": {"@id": {"type": "string"}},
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["@id", "columnIndex"],
                "properties": {
                    "@id": {"type": "string"},
                    "@type": {"const": "viao:TableColumn"},
                    "columnLabel": {"type": ["string", "null"]},
                    "columnIndex": {"type": "integer", "minimum": 0},
                },
            },
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["@id", "rowIndex", "cells"],
                "properties": {
                    "@id": {"type": "string"},
                    "@type": {"const": "viao:TableRow"},
                    "rowLabel": {"type": ["string", "null"]},
                    "rowIndex": {"type": "integer", "minimum": 0},
                    "isHeaderRow": {"type": "boolean"},
                    "cells": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["@id", "inColumn"],
                            "properties": {
                                "@id": {"type": "string"},
                                "@type": {"const": "viao:TableCell"},
                                "inColumn": {"type": "string"},
                                "cellValue": {"type": ["string", "null"]},
                                "nestedTable": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
}

_VALIDATOR = jsonschema.Draft7Validator(JSON_SCHEMA)


def build_table_iri(doc_sha: str, table_index: int) -> str:
    return f"{VIAO}StructuredTable_{doc_sha[:12]}_{table_index}"


def build_column_iri(table_iri: str, col_index: int) -> str:
    return f"{table_iri}#col_{col_index}"


def build_row_iri(table_iri: str, row_index: int) -> str:
    return f"{table_iri}#row_{row_index}"


def build_cell_iri(table_iri: str, row_index: int, col_index: int) -> str:
    return f"{table_iri}#cell_{row_index}_{col_index}"


def build_document_iri(doc_sha: str) -> str:
    return f"{VIAO}Document_{doc_sha[:12]}"


def empty_payload(
    doc_sha: str,
    table_index: int,
    *,
    extraction_method: str,
    caption: str | None = None,
    page_number: int | None = None,
) -> dict[str, Any]:
    """Build a skeleton StructuredTable payload (no rows/columns).

    Callers populate `columns` + `rows` before validating + saving."""
    iri = build_table_iri(doc_sha, table_index)
    payload: dict[str, Any] = {
        "@context": CONTEXT,
        "@id": iri,
        "@type": "viao:StructuredTable",
        "extractionMethod": extraction_method,
        "derivedFromDocument": {"@id": build_document_iri(doc_sha)},
        "columns": [],
        "rows": [],
    }
    if caption is not None:
        payload["caption"] = caption
    if page_number is not None:
        payload["pageNumber"] = int(page_number)
    return payload


def validate_table_jsonld(payload: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty list = OK).

    Fail-soft. Use the returned list to skip + log invalid tables."""
    if not isinstance(payload, dict):
        return ["payload is not a dict"]
    return [
        f"{'/'.join(str(p) for p in err.absolute_path)}: {err.message}"
        for err in _VALIDATOR.iter_errors(payload)
    ]


def flat_text_summary(payload: dict[str, Any], max_cells: int = 60) -> str:
    """Render a compact text summary used for `intelligence_artifacts.text`
    + as the embedding input. The full JSON-LD lives in
    `extra_metadata`; this helper produces a chunk-friendly preview that
    captures the caption + column headers + the first few rows."""
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    caption = payload.get("caption")
    if caption:
        parts.append(str(caption).strip())
    cols = payload.get("columns") or []
    if isinstance(cols, list) and cols:
        col_labels = [
            str(c.get("columnLabel") or "").strip()
            for c in cols
            if isinstance(c, dict)
        ]
        parts.append("Columns: " + " | ".join(col_labels))
    rows = payload.get("rows") or []
    cell_budget = max_cells
    if isinstance(rows, list):
        for row in rows:
            if cell_budget <= 0:
                break
            if not isinstance(row, dict):
                continue
            row_label = str(row.get("rowLabel") or "").strip()
            cells = row.get("cells") or []
            values = [
                str(c.get("cellValue") or "").strip()
                for c in cells
                if isinstance(c, dict)
            ]
            cell_budget -= 1 + len(values)
            parts.append(
                (row_label + ": " if row_label else "")
                + " | ".join(v for v in values if v)
            )
    return "\n".join(p for p in parts if p)

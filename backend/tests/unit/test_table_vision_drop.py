"""Vision table extraction must be allowed to say 'no table' -- and be dropped.

Audit (46 vision tables, GlobalEVPolicyExplorer): 17% fabricated outright, mean
53% of numbers absent from their own page. pdfplumber over the same doc: 0%.
Cause: the prompt asserted "this image contains a table" and gave no way to
disagree, so a bad crop forced the model to invent a plausible one.
"""
from __future__ import annotations

from backend.app.services.prompts import PROMPTS
from backend.app.services.table_extract import _looks_fabricated


def test_prompt_offers_a_no_table_escape_hatch() -> None:
    system, user = PROMPTS["table_extract_vision"](page_number=3, caption_hint=None)
    blob = system + user
    assert '"no_table": true' in blob
    # It must be framed as correct, not a failure -- otherwise the model avoids it.
    assert "better than guessing" in system
    assert "NEVER invent" in system
    assert '{"no_table": true}' in user


def test_placeholder_table_is_flagged() -> None:
    """The real fabrication found in the DB: an IEA EV-policy page that produced
    'Category A | EUR 500,000 / Category B | EUR 300,000'."""
    payload = {
        "columns": [
            {"columnIndex": 0, "columnLabel": "Category"},
            {"columnIndex": 1, "columnLabel": "Value"},
        ],
        "rows": [
            {"rowIndex": 0, "rowLabel": "Category A",
             "cells": [{"columnIndex": 1, "cellValue": "500,000"}]},
            {"rowIndex": 1, "rowLabel": "Category B",
             "cells": [{"columnIndex": 1, "cellValue": "300,000"}]},
        ],
    }
    assert _looks_fabricated(payload) is True


def test_real_table_is_not_flagged() -> None:
    """The genuine Tesla cost-of-revenues table must survive."""
    payload = {
        "columns": [
            {"columnIndex": 0, "columnLabel": "Cost of revenues"},
            {"columnIndex": 1, "columnLabel": "2024"},
            {"columnIndex": 2, "columnLabel": "2023"},
        ],
        "rows": [
            {"rowIndex": 0, "rowLabel": "Automotive sales",
             "cells": [{"columnIndex": 1, "cellValue": "61,870"},
                       {"columnIndex": 2, "cellValue": "65,121"}]},
            {"rowIndex": 1, "rowLabel": "Automotive leasing",
             "cells": [{"columnIndex": 1, "cellValue": "1,003"},
                       {"columnIndex": 2, "cellValue": "1,268"}]},
        ],
    }
    assert _looks_fabricated(payload) is False


def test_single_incidental_placeholder_word_is_not_enough() -> None:
    """One 'example' in a real table must not nuke it -- needs 2 independent hits."""
    payload = {
        "columns": [{"columnIndex": 0, "columnLabel": "Example"}],
        "rows": [{"rowIndex": 0, "rowLabel": "Norway",
                  "cells": [{"columnIndex": 0, "cellValue": "25.4"}]}],
    }
    assert _looks_fabricated(payload) is False


def test_empty_payload_is_not_flagged() -> None:
    assert _looks_fabricated({"columns": [], "rows": []}) is False

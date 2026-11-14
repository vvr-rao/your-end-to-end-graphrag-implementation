"""structured.listing_stats: exact per-item change statistics for listings."""

from __future__ import annotations

from backend.app.services.prompts import PROMPTS
from backend.app.services.structured.listing_stats import change_stats, roles_for, stats_text

COLS = ["item", "period", "unit", "value"]
ROWS = [("A", "2023", "u1", 100), ("A", "2024", "u1", 110), ("A", "2025", "u1", 120),
        ("B", "2023", "u2", 50), ("B", "2025", "u2", 25),
        ("C", "2023", "u1", 10), ("C", "2025", "u1", 10.01),
        ("D", "2025", "u1", 7)]


def test_change_stats_per_item_and_overall():
    st = change_stats(COLS, ROWS, "period", "value")
    by = {it["item"]: it for it in st["items"]}
    assert st["group_columns"] == ["item", "unit"] and st["periods"] == ["2023", "2025"]
    assert by["A / u1"]["pct"] == 20.0 and by["A / u1"]["min"] == 100 and by["A / u1"]["max"] == 120
    assert by["B / u2"]["pct"] == -50.0
    assert by["D / u1"]["pct"] is None                       # one period only
    assert (st["rose"], st["fell"], st["unchanged"], st["no_change_computable"]) == (1, 1, 1, 1)
    assert st["top_risers"] == ["A / u1"] and st["top_fallers"] == ["B / u2"]
    text = stats_text(st)
    assert "1 rose, 1 fell, 1 unchanged" in text and "+20.0%" in text and "-50.0%" in text


def test_no_stats_without_two_periods_or_roles():
    assert change_stats(COLS, [r for r in ROWS if r[1] == "2025"], "period", "value") is None
    assert change_stats(COLS, ROWS, "missing", "value") is None
    meta = {"columns": [{"name": "period", "role": "time"}, {"name": "value", "role": "measure"}],
            "measures": [{"id": "measure_v", "measure": {"period_columns": ["period"], "value_column": "value"}}]}
    assert roles_for(meta, "measure_v", COLS) == ("period", "value")
    assert roles_for(meta, None, COLS) == ("period", "value")
    assert roles_for({"columns": []}, None, COLS) == (None, None)


def test_listing_evidence_switches_answer_prompts():
    ev = [{"kind": "sql_result", "listing": True, "iri": "x", "text": "t"}]
    system, _ = PROMPTS["answer_listing_analysis"]("q", ev)
    assert "ALL ITEMS" in system and "CHANGE SUMMARY" in system
    assert "TABLE LISTINGS" in PROMPTS["answer_deep_research"]("q", ev)[0]
    assert "TABLE LISTINGS" not in PROMPTS["answer_deep_research"]("q", [])[0]   # unchanged otherwise


def test_appended_table_is_exact_and_prompt_defers_to_it():
    from backend.app.services.structured.listing_stats import stats_table

    st = change_stats(COLS, ROWS, "period", "value")
    t = stats_table(st, "[viao:SqlResult_x]")
    lines = t.splitlines()
    assert lines[0].startswith("All items (4,") and "[viao:SqlResult_x]" in lines[0]
    assert lines[4] == "| A / u1 | 2023: 100.0 | 2025: 120.0 | +20.0% |"      # largest rise first
    assert lines[-1].startswith("| D / u1 |") and lines[-1].endswith("n/a |")
    ev = [{"kind": "sql_result", "listing": True, "iri": "x", "text": "t", "change_stats": st}]
    assert "Do NOT write a per-item table" in PROMPTS["answer_listing_analysis"]("q", ev)[0]
    ev[0]["change_stats"] = None
    assert "ALL ITEMS" in PROMPTS["answer_listing_analysis"]("q", ev)[0]

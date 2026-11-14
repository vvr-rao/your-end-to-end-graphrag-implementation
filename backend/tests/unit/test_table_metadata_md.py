"""structured.md_io: Markdown export -> import round trip and strict parsing."""

from __future__ import annotations

import pytest

from backend.app.services.structured.md_io import MetadataParseError, apply_markdown, to_markdown


def _meta() -> dict:
    return {
        "format": 1, "status": "complete", "cache_key": "ab12", "table_name": "prices",
        "source": {"rel_path": "worldbank/pink-sheet/s1/normalized/prices.csv"}, "row_count": 3,
        "description": "Monthly commodity prices.\nSecond line | with pipe.", "grain": "commodity x month",
        "model": "openai:gpt-4.1", "edited": False,
        "columns": [
            {"name": "commodity", "type": "text", "role": "dimension", "unit": None,
             "entity_kind": "commodity", "description": "Commodity name", "class_label": "Commodity",
             "class_iri": "https://ex.org/o#Commodity"},
            {"name": "value", "type": "numeric", "role": "measure", "unit": "USD", "entity_kind": None,
             "description": "Price, a|b", "class_label": None, "class_iri": None},
        ],
        "metrics": [{"id": "avg_price", "label": "Average price", "description": "Mean price per commodity",
                     "columns": ["value", "commodity"], "class_label": "Price", "class_iri": "https://ex.org/o#Price",
                     "params": [{"name": "c", "type": "text", "column": "commodity",
                                 "description": "commodity name", "example": "Crude oil, Brent"}],
                     "sql": "SELECT AVG(value) FROM prices\nWHERE commodity = :c"}],
        "filters": [], "queries": [],
        "lookups": [{"id": "to_series", "label": "Series info", "description": "Join to series",
                     "columns": ["commodity"], "class_label": None, "class_iri": None, "params": [],
                     "other_table": "series", "join": [{"column": "commodity", "other_column": "name"}],
                     "sql": "SELECT p.*, s.unit FROM prices p JOIN series s ON s.name = p.commodity"}],
    }


def test_round_trip_is_lossless():
    m = _meta()
    md = to_markdown(m)
    assert "```sql" in md and "| `commodity` | text |" in md
    assert apply_markdown(m, md) == m


def test_edits_are_applied():
    m = _meta()
    md = to_markdown(m).replace("Commodity name", "Name of the commodity").replace(
        "WHERE commodity = :c", "WHERE commodity = :c AND value > 0")
    md += "\n### `n_rows` — Row count\n\n- **Description:** rows\n- **Columns:** \n- **Class:** \n" \
          "- **Parameters:** none\n\n```sql\nSELECT COUNT(*) FROM prices\n```\n"
    out = apply_markdown(m, md)
    assert out["columns"][0]["description"] == "Name of the commodity"
    assert out["metrics"][0]["sql"].endswith("AND value > 0")
    assert [it["id"] for it in out["lookups"]] == ["to_series", "n_rows"]   # appended to last section


def test_deleting_an_item_removes_it():
    m = _meta()
    md = to_markdown(m)
    start = md.index("### `avg_price`")
    end = md.index("## Filters")
    out = apply_markdown(m, md[:start] + md[end:])
    assert out["metrics"] == []


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.replace("| `value` | numeric |", "| `price` | numeric |"), "columns must stay"),
    (lambda s: s.replace("| `value` | numeric |", "| `value` | text |"), "type is fixed"),
    (lambda s: s.replace("# Table `prices`", "# Table `other`"), "not `prices`"),
    (lambda s: s.replace("```sql\nSELECT AVG", "```\nSELECT AVG").replace("WHERE commodity = :c\n```", "WHERE commodity = :c"),
     "unterminated"),
    (lambda s: s.replace("## Filters", "## Filterz"), "unknown section"),
    (lambda s: s.replace("- **Join:** `commodity` = `series.name`", "- **Join:** commodity=name"), "join must look like"),
])
def test_bad_edits_are_reported_with_lines(mutate, msg):
    with pytest.raises(MetadataParseError, match=msg):
        apply_markdown(_meta(), mutate(to_markdown(_meta())))


def test_crlf_markdown_parses():
    m = _meta()
    assert apply_markdown(m, to_markdown(m).replace("\n", "\r\n")) == m

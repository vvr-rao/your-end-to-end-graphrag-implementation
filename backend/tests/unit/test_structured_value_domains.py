"""value_domains: valid parameter values travel with the query and are
checked before any SQL runs (no database here; the DB path is in
test_structured_db.py)."""

from __future__ import annotations

from backend.app.services.structured.metadata import catalog_for_prompt
from backend.app.services.structured.value_domains import check_params, spec_declared

UNITS = ["($/dmtu)", "($/mt)", "($/kg)"]
COMBOS = {"params": ["c", "u"], "rows": [["Copper", "($/mt)"], ["Iron ore, cfr spot", "($/dmtu)"]]}
DECL = [{"name": "c", "column": "commodity", "allowed_values": ["Copper", "Iron ore, cfr spot"]},
        {"name": "u", "column": "unit", "allowed_values": UNITS},
        {"name": "start", "column": "period", "range": ["1960", "2025"]}]


def test_exact_values_pass_unchanged():
    params, bad, subs = check_params({"c": "Copper", "u": "($/mt)", "start": "2023"}, DECL, COMBOS)
    assert bad == [] and subs == [] and params == {"c": "Copper", "u": "($/mt)", "start": "2023"}


def test_case_and_punctuation_variants_are_replaced_deterministically():
    params, bad, subs = check_params({"c": "iron ore cfr spot", "u": "$/DMTU"}, DECL, COMBOS)
    assert bad == [] and params == {"c": "Iron ore, cfr spot", "u": "($/dmtu)"}
    assert {s["to"] for s in subs} == {"Iron ore, cfr spot", "($/dmtu)"}


def test_unknown_value_is_refused_with_the_valid_values():
    _, bad, _ = check_params({"c": "Iron ore", "u": "($/dmt)"}, DECL, COMBOS)
    assert [b["param"] for b in bad] == ["c", "u"]          # ($/dmt) is not ($/dmtu)
    assert bad[1]["allowed_values"] == UNITS


def test_valid_values_that_never_occur_together_are_refused_with_the_closest_rows():
    _, bad, _ = check_params({"c": "Iron ore, cfr spot", "u": "($/mt)"}, DECL, COMBOS)
    assert len(bad) == 1 and "never occur together" in bad[0]["error"]
    assert {"c": "Iron ore, cfr spot", "u": "($/dmtu)"} in bad[0]["valid_combinations"]


def test_list_params_and_integers():
    decl = [{"name": "fys", "column": "fy", "allowed_values": [2023, 2024, 2025]},
            {"name": "tags", "column": "tag", "allowed_values": ["Revenue", "ProfitLoss"]}]
    params, bad, _ = check_params({"fys": ["2024", 2025], "tags": ["revenue", "Profit"]}, decl)
    assert params["fys"] == [2024, 2025] and params["tags"][0] == "Revenue"
    assert bad == [{"param": "tags", "value": "Profit", "column": "tag",
                    "allowed_values": ["Revenue", "ProfitLoss"]}]


def test_params_without_values_are_not_checked():
    params, bad, _ = check_params({"cik": "anything"}, [{"name": "cik", "column": "cik"}])
    assert bad == [] and params == {"cik": "anything"}


def test_model_written_spec_uses_the_table_domains():
    decl = spec_declared([{"name": "u", "type": "text", "column": "unit"}],
                         {"unit": {"values": UNITS}, "period": {"min": "1960", "max": "2025"}})
    assert decl[0]["allowed_values"] == UNITS
    assert check_params({"u": "($/lb)"}, decl)[1][0]["allowed_values"] == UNITS


def test_catalog_shows_values_ranges_and_shared_combinations_once():
    meta = {"table_name": "prices", "columns": [
                {"name": "commodity", "type": "text", "examples": ["Copper"]},
                {"name": "unit", "type": "text"}, {"name": "period", "type": "text", "role": "time"},
                {"name": "value", "type": "numeric", "role": "measure", "examples": []}],
            "value_domains": {"commodity": {"values": ["Copper", "Iron ore, cfr spot"]},
                              "unit": {"values": UNITS}, "period": {"min": "1960", "max": "2025"}},
            "metrics": [], "filters": [], "lookups": [],
            "queries": [{"id": f"q{i}", "params": DECL[:2], "allowed_combinations": COMBOS,
                         "query": {"select": ["value"]}} for i in range(3)]}
    cat = catalog_for_prompt(meta)
    cols = {c["name"]: c for c in cat["columns"]}
    assert cols["commodity"]["values"] == ["Copper", "Iron ore, cfr spot"] and "examples" not in cols["commodity"]
    assert cols["period"]["range"] == ["1960", "2025"]
    assert len(cat["value_combinations"]) == 1                     # stored per query, shown once
    assert cat["value_combinations"][0]["columns"] == ["commodity", "unit"]
    assert all(q["valid_combinations"] == {"params": ["c", "u"], "see": 0} for q in cat["queries"])

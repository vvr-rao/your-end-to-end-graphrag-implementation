"""Catalog semantics (generic fixtures only):

* ordered version fields (direction, nulls, precedence vs tie-break) and
  population predicates compile into the SQL, parameter domains and checks;
* identical SQL under two names: same population -> alias, different -> both
  fail; required parameters are the witness, optional filters tested alone;
* subject identifier vs uniqueness vs dimension columns (measured);
* unsafe parameter names are renamed and dropped templates re-tested;
* coverage ledger categories; entity matches are type- and context-checked.
"""

from __future__ import annotations

import pytest

from backend.app.services.structured.query_builder import build_sql, sanitize_param_names
from backend.app.services.structured.refine import (
    _with_fixed_conditions,
    apply_uniqueness_evidence,
    coverage_ledger,
    normalize_measure,
    observation_key,
    population_where,
    review_duplicate_sql,
)
from backend.tests.unit.test_structured_catalog_policy import (  # noqa: F401  (fixture)
    ANALYSIS,
    META,
    _state,
    db_only,
    loaded,
)

TYPES = {"costs": {c["name"]: c["type"] for c in META["columns"]}}

# ---------------------------------------------------------------- versions


def test_version_order_keeps_direction_nulls_and_basis():
    m, err = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                                "period_columns": ["year"],
                                "version_order": [{"column": "release", "direction": "desc", "nulls": "last",
                                                   "basis": "precedence"},
                                                  {"column": "plan_name", "direction": "asc", "nulls": "first",
                                                   "basis": "tie_break"}],
                                "version_reason": "a later release restates the same figures"}, META, ANALYSIS)
    assert err is None
    assert [(v["column"], v["direction"], v["nulls"], v["basis"]) for v in m["version_order"]] == [
        ("release", "desc", "last", "precedence"), ("plan_name", "asc", "first", "tie_break")]
    assert "release" not in observation_key(m)


def test_legacy_version_column_becomes_a_precedence_entry():
    m, _ = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                              "period_columns": ["year"], "version_column": "release",
                              "version_reason": "restated"}, META, ANALYSIS)
    assert m["version_order"][0] == {"column": "release", "direction": "desc", "nulls": "last",
                                     "basis": "precedence"}


def test_a_tie_break_alone_is_not_a_version_rule():
    m, _ = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                              "period_columns": ["year"], "version_reason": "r",
                              "version_order": [{"column": "release", "basis": "tie_break"}]}, META, ANALYSIS)
    assert not m.get("version_order")


def test_order_by_nulls_compiles():
    sql = build_sql({"select": ["plan", "cost"], "order_by": [{"column": "release", "dir": "desc", "nulls": "last"},
                                                              {"column": "plan", "dir": "asc", "nulls": "first"}]},
                    "costs", TYPES).sql
    assert 'ORDER BY "release" DESC NULLS LAST, "plan" ASC NULLS FIRST' in sql


# ---------------------------------------------------------------- populations


def test_population_include_and_exclude_compile():
    where = population_where({"include": [{"column": "year", "op": ">=", "value": 2023}],
                              "exclude": [{"column": "drug", "op": "=", "value": "D2"},
                                          {"column": "plan", "op": "in", "value": ["P9", "P8"]}]})
    sql = build_sql({"select": ["plan", "cost"], "where": where}, "costs", TYPES).sql
    assert '"year" >= 2023' in sql
    assert "(\"drug\" IS NULL OR \"drug\" NOT IN ('D2'))" in sql        # an empty value is not excluded
    assert "NOT IN ('P9', 'P8')" in sql


def test_identical_sql_same_population_is_an_alias_different_population_fails_both():
    def it(i, pop):
        return {"id": i, "sql": 'SELECT "cost" FROM "costs"', "params": [],
                "measure": {"id": i, "population": pop}, "verification": {"status": "passed"}}

    meta = {"measures": [it("a", None), it("b", None)], "metrics": [], "filters": [], "queries": [],
            "lookups": [], "dropped": []}
    rep = review_duplicate_sql(meta)
    assert rep[0]["decision"] == "aliased" and [m["id"] for m in meta["measures"]] == ["a"]
    assert meta["measures"][0]["aliases"] == ["b"]
    meta = {"measures": [it("a", None), it("b", {"include": [{"column": "drug", "op": "=", "value": "D1"}]})],
            "metrics": [], "filters": [], "queries": [], "lookups": [], "dropped": []}
    rep = review_duplicate_sql(meta)
    assert rep[0]["decision"].startswith("failed")
    assert {m["verification"]["status"] for m in meta["measures"]} == {"failed"}


def test_witness_probe_carries_the_items_fixed_conditions():
    item = ('SELECT DISTINCT "a" FROM "t" WHERE "k" = CAST(:k AS text) AND "u" IS DISTINCT FROM \'X\' '
            'AND "d" BETWEEN CAST(:d_from AS date) AND CAST(:d_to AS date)')
    out = _with_fixed_conditions('SELECT "k" FROM "t" WHERE "k" IS NOT NULL LIMIT 1', item)
    assert "IS DISTINCT FROM 'X'" in out and ":k" not in out and "BETWEEN" not in out


# ---------------------------------------------------------------- parameter names


def test_unsafe_param_names_are_renamed_everywhere():
    item = {"query": {"select": ["plan", "cost"],
                      "where": [{"column": "year", "op": "between", "param": "end"},
                                {"column": "drug", "op": "=", "param": "select"}]},
            "params": [{"name": "end", "column": "year"}, {"name": "select", "column": "drug"}],
            "examples": [{"end": [2023, 2024], "select": "D1"}]}
    ren = sanitize_param_names(item)
    assert ren["end"] == "end_from,end_to" or ren["end"].count(",") == 1
    names = {p["name"] for p in item["params"]}
    assert "end" not in names and "select" not in names
    assert item["param_renames"] == ren


# ---------------------------------------------------------------- uniqueness


def test_redundant_identity_columns_become_dimensions():
    m = {"identity_columns": ["zip", "site", "n"], "subject_columns": ["zip", "site", "n"],
         "uniqueness_columns": []}
    out = apply_uniqueness_evidence(m, {"uniqueness": {"redundant": ["zip"]}})
    assert out["identity_columns"] == ["site", "n"] and out["subject_columns"] == ["site", "n"]
    assert out["dimension_columns"] == ["zip"] and "no observation collides" in out["review"][0]


# ---------------------------------------------------------------- coverage ledger


def _ledger_meta():
    from backend.app.services.structured.metadata import CATALOG_VERSION, record_test

    def q(i, sql, params=(), measure=None, status="passed", select=None):
        it = {"id": i, "sql": sql, "params": [{"type": "text", **p} for p in params],
              "query": {"select": select or []}}
        record_test(it, status)
        if measure:
            it["measure"] = measure
        return it

    return {
        "table_name": "t", "catalog_version": CATALOG_VERSION,
        "columns": [{"name": "k", "type": "text"}, {"name": "ind", "type": "text"}, {"name": "v", "type": "numeric"},
                    {"name": "w", "type": "numeric"}, {"name": "x", "type": "numeric"}],
        "value_domains": {"ind": {"values": ["Alpha", "Beta", "Gamma"]}},
        "inventory": {"candidates": [
            {"candidate_id": "cand_alpha", "label": "Alpha", "indicator": {"column": "ind", "values": ["Alpha"]}},
            {"candidate_id": "cand_beta", "label": "Beta", "indicator": {"column": "ind", "values": ["Beta"]}},
            {"candidate_id": "cand_wx", "label": "W and X", "source_columns": ["w", "x"]},
            {"candidate_id": "cand_keys", "label": "Distinct keys", "source_columns": ["k"]},
            {"candidate_id": "cand_none", "label": "Unanswerable", "source_columns": ["x"]}]},
        "coverage": [{"candidate_id": "cand_none", "candidate": "Unanswerable", "status": "unsupported_by_source",
                      "reason": "no such data"}],
        "measures": [q("measure_alpha", 'SELECT "k", "v" FROM "t" WHERE "ind" IN (\'Alpha\')',
                       measure={"id": "alpha", "item": {"column": "ind", "values": ["Alpha"]}})],
        "metrics": [q("by_ind", 'SELECT "k", "v" FROM "t" WHERE "ind" = :ind',
                      params=[{"name": "ind", "column": "ind", "allowed_values": ["Alpha", "Beta"]}]),
                    q("w_only", 'SELECT SUM("w") FROM "t"')],
        "filters": [q("count_keys", 'SELECT COUNT(DISTINCT "k") FROM "t"',
                      select=[{"agg": "count_distinct", "column": "k"}])],
        "queries": [], "lookups": [], "dropped": []}


def test_ledger_categories_from_the_final_catalog():
    meta = _ledger_meta()
    meta["metrics"][1]["sql"] = 'SELECT SUM("w") FROM "t"'           # x is never read
    rows = {r["capability_id"]: r for r in coverage_ledger(meta)}
    assert rows["cand_alpha"]["category"] == "dedicated_measure"
    assert rows["cand_alpha"]["answered_by"] == ["measure_alpha"]
    assert rows["cand_beta"]["category"] == "generic_query" and rows["cand_beta"]["answered_by"] == ["by_ind"]
    assert rows["cand_wx"]["category"] == "partial" and "missing: x" in rows["cand_wx"]["detail"]
    assert rows["cand_keys"]["category"] == "generic_query" and rows["cand_keys"]["answered_by"] == ["count_keys"]
    assert rows["cand_none"]["category"] == "unsupported" and rows["cand_none"]["stage"] == "refinement"
    assert meta["coverage_ledger"]                                   # stored with stable ids


def test_ledger_unexercised_unfinished_and_verifier_drop():
    meta = _ledger_meta()
    from backend.app.services.structured.metadata import record_test

    record_test(meta["metrics"][0], "not_exercised")
    meta["measures"] = []
    meta["dropped"] = [{"id": "measure_alpha", "section": "measures", "stage": "verification",
                        "reason": "verification: the SQL omits the promised ordering"}]
    meta["coverage"] = [{"candidate_id": "cand_alpha", "candidate": "Alpha", "status": "materialized",
                         "measure": "alpha"},
                        {"candidate_id": "cand_none", "candidate": "Unanswerable",
                         "status": "incomplete_due_to_budget", "reason": "budget"}]
    rows = {r["capability_id"]: r for r in coverage_ledger(meta)}
    assert rows["cand_beta"]["category"] == "unexercised" and rows["cand_beta"]["stage"] == "testing"
    assert rows["cand_alpha"]["category"] == "unexercised"         # its parameter query is not exercised
    meta["metrics"] = meta["metrics"][1:]
    rows = {r["capability_id"]: r for r in coverage_ledger(meta)}
    assert rows["cand_alpha"]["category"] == "verifier_drop" and "ordering" in rows["cand_alpha"]["reason"]
    assert rows["cand_none"]["category"] == "unfinished"


# ---------------------------------------------------------------- DB-checked


@db_only
async def test_population_scopes_sql_domains_and_witness(loaded):  # noqa: F811
    from backend.app.db.session import session_scope
    from backend.app.services.structured.refine import measure_item, test_item
    from backend.app.services.structured.value_domains import attach_value_domains

    meta, analysis, types, partners = await _state()
    m, err = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                                "display_columns": ["plan_name"], "period_columns": ["year"],
                                "population": {"include": [], "exclude": [{"column": "drug", "op": "=",
                                                                          "value": "D2"}]}},
                               meta, analysis)
    assert err is None
    it = measure_item(m, meta)
    r = await test_item(meta, "measures", it, analysis, types, partners=partners)
    assert r["status"] == "passed", r
    assert "'D2'" in it["sql"]
    meta["measures"] = [it]
    async with session_scope() as s:
        await attach_value_domains(s, meta)
    drug = next(p for p in it["params"] if p["column"] == "drug")
    assert drug["allowed_values"] == ["D1"]                             # the population narrows the domain


@db_only
async def test_subject_call_returns_its_key_and_optional_filters_are_tested(loaded):  # noqa: F811
    from backend.app.services.structured.refine import measure_item, test_item

    meta, analysis, types, partners = await _state()
    m, _ = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                              "display_columns": ["plan_name"], "period_columns": ["year"],
                              "subject_columns": ["plan"]}, meta, analysis)
    it = measure_item(m, meta)
    r = await test_item(meta, "measures", it, analysis, types, partners=partners)
    assert r["status"] == "passed", r
    ident = it["verification"]["identity"]
    assert ident["subject"]["ok"] and set(ident["subject"]["key"]) == {"plan"}
    assert it["verification"].get("optional_filters")
    assert all(v == "passed" for v in it["verification"]["optional_filters"].values())


@db_only
async def test_declared_filter_column_outside_the_key_stays_an_optional_filter(loaded):  # noqa: F811
    from backend.app.services.structured.refine import measure_item, test_item

    meta, analysis, types, partners = await _state()
    m, err = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                                "period_columns": ["year"], "filter_columns": ["plan_name", "year", "drug"]},
                               meta, analysis)
    assert err is None
    it = measure_item(m, meta)
    params = [w for w in it["query"]["where"] if isinstance(w.get("param"), str)]
    assert [w["column"] for w in params].count("drug") == 1           # already a key filter: not repeated
    assert {"column": "plan_name", "op": "=", "param": "plan_name", "optional": True} in params
    assert "plan_name" in it["query"]["select"]
    assert not any(w["column"] == "year" and w.get("op") == "=" for w in params)   # the period keeps its range
    r = await test_item(meta, "measures", it, analysis, types, partners=partners)
    assert r["status"] == "passed", r


@db_only
async def test_entity_matches_check_type_and_leave_place_collisions_unresolved(loaded):  # noqa: F811
    from sqlalchemy import text

    from backend.app.db.session import session_scope
    from backend.app.services.db_entity_extract import _normalize_name
    from backend.app.services.structured.db_load import _entity_lookup, match_entities

    async with session_scope() as s:
        await s.execute(text('CREATE TABLE graphrag."orgs_t" (company text, country text)'))
        await s.execute(text("INSERT INTO graphrag.\"orgs_t\" VALUES ('Northland Holdings Ltd', 'Northland'), "
                             "('Acme Widgets Ltd', 'Southland')"))
        ids = {}
        for label in ("Country", "Organization"):
            ids[label] = (await s.execute(text(
                "INSERT INTO graphrag.ontology_classes (iri, label, namespace) VALUES (:i, :l, 'x') RETURNING id"),
                {"i": f"https://example.org/{label}", "l": label})).scalar()
        for name, cls in (("Northland", "Country"), ("Acme Widgets", "Organization")):
            await s.execute(text(
                "INSERT INTO graphrag.entities (entity_identifier, name, normalized_name, class_id, iri) "
                "VALUES (:e, :n, :nn, :c, :i)"),
                {"e": name, "n": name, "nn": _normalize_name(name), "c": ids[cls], "i": f"https://example.org/e/{name}"})
    try:
        meta = {"table_name": "orgs_t", "columns": [
            {"name": "company", "type": "text", "role": "identifier", "entity_kind": "organization"},
            {"name": "country", "type": "text", "role": "dimension", "entity_kind": "country"}]}
        async with session_scope() as s:
            hits, report = await match_entities(s, meta, await _entity_lookup(s))
        rows = {(r["column"], r["raw_value"]): r for r in report}
        bad = rows[("company", "Northland Holdings Ltd")]
        assert bad["decision"] == "unresolved" and "is a place" in bad["reason"]
        assert bad["method"] == "short_form" and bad["target_type"] == "Country"
        good = rows[("company", "Acme Widgets Ltd")]
        assert good["decision"] == "linked" and good["matched_label"] == "Acme Widgets"
        assert rows[("country", "Northland")]["decision"] == "linked"     # a place column may hold places
        assert all(set(r) >= {"raw_value", "matched_label", "entity_id", "target_type", "method"} for r in report)
        assert len(hits) == 2
    finally:
        async with session_scope() as s:
            await s.execute(text('DROP TABLE IF EXISTS graphrag."orgs_t"'))


if __name__ == "__main__":       # pragma: no cover
    pytest.main([__file__])


# ---------------------------------------------------------------- eligibility


def _elig_meta(**item):
    from backend.app.services.structured.metadata import CATALOG_VERSION, record_test

    it = {"id": "q", "sql": 'SELECT "cost" FROM "costs"', "params": [], **item}
    record_test(it, "passed")
    return {"table_name": "costs", "catalog_version": CATALOG_VERSION, "metrics": [it], "measures": [],
            "filters": [], "queries": [], "lookups": [], "verified": {"version": "tv-3", "ok": []}}, it


def test_pending_review_blocks_retrieval_only_when_review_is_required():
    from backend.app.services.structured.metadata import selectable

    meta, it = _elig_meta(verify_verdict={"verdict": "pending"})
    ok, why = selectable(meta, it)
    assert not ok and why.startswith("semantic review pending")
    assert selectable(meta, it, semantic=False)[0]                   # data test alone passes
    meta["semantic_review"] = {"required": False}
    assert selectable(meta, it)[0]


def test_a_verdict_is_stale_once_its_sql_changes_and_legacy_verdicts_come_from_the_report():
    from backend.app.services.structured.metadata import (
        pin_reviewed_sql,
        record_test,
        selectable,
        semantic_status,
    )

    meta, it = _elig_meta()
    assert semantic_status(meta, it) == "unreviewed"
    meta["verified"]["text_fixed"] = [{"id": "q"}]                    # older catalog: verdict in the report
    assert semantic_status(meta, it) == "fix_text" and selectable(meta, it)[0]
    assert pin_reviewed_sql(meta) == 1 and it["verify_verdict"]["sql_sha"]
    it["sql"] = 'SELECT "cost" FROM "costs" WHERE "year" > 2020'
    record_test(it, "passed")
    assert semantic_status(meta, it) == "stale"
    assert "SQL changed" in selectable(meta, it)[1]


def test_rebuild_keeps_review_for_identical_sql_and_marks_changed_sql_stale():
    from backend.app.services.structured.refine import _carry_review

    meta, old = _elig_meta(verify_verdict={"verdict": "fix_text", "reason": "r"}, label="Reviewed label")
    same = {"id": "q", "sql": old["sql"], "label": "generated"}
    _carry_review(meta, old, same)
    assert same["label"] == "Reviewed label" and same["verify_verdict"]["verdict"] == "fix_text"
    changed = {"id": "q", "sql": old["sql"] + " LIMIT 1", "label": "generated"}
    _carry_review(meta, old, changed)
    assert changed["verify_verdict"]["verdict"] == "stale" and changed["label"] == "generated"


def test_a_hold_makes_an_item_unavailable_with_its_reason():
    from backend.app.services.structured.metadata import rebuild_action, selectable

    meta, it = _elig_meta(verify_verdict={"verdict": "ok"}, hold={"reason": "returns the wrong population"})
    ok, why = selectable(meta, it)
    assert not ok and why == "held: returns the wrong population"
    assert "table-metadata-import" in rebuild_action("costs", why)
    assert "--relink" not in rebuild_action("costs", "semantic review pending")


def test_select_tables_by_name_keeps_folder_names_and_rejects_unknown():
    from types import SimpleNamespace

    from backend.app.services.structured.discovery import select_tables

    found = [SimpleNamespace(table_name=n) for n in ("a", "b", "c")]
    assert [t.table_name for t in select_tables(found, None, ["c", "a"])] == ["c", "a"]
    assert [t.table_name for t in select_tables(found, 2, None)] == ["a", "b"]
    with pytest.raises(SystemExit, match="not found: z"):
        select_tables(found, None, ["z"])

"""Verified-catalog policy for structured data.

* retrieval may run only stored queries whose exact SQL passed its data test
  (``metadata.selectable``), with parameters validated against the
  declaration (``value_domains.validate_call_params``);
* measure grain: identity columns vs display labels, one value per
  observation, no collapse without a version rule
  (``refine.normalize_measure`` / ``measure_identity_check``);
* repeated values are evidence, not a verdict;
* lookups: join cardinality on the complete key, no multiplied aggregates;
* test statuses passed / failed / not_exercised; bounded measure repair.

The fixture is generic: a value per (plan, drug, year), where a plan's name
is a label of the plan. Nothing is specific to any dataset.
"""

from __future__ import annotations

import copy

import pytest

from backend.app.services.structured.metadata import (
    CATALOG_VERSION,
    catalog_for_prompt,
    record_test,
    selectable,
    sql_sha,
)
from backend.app.services.structured.refine import normalize_measure, observation_key
from backend.app.services.structured.value_domains import validate_call_params
from backend.app.services.structured.verify import allowed_verdicts
from backend.tests.unit.test_structured_db import _SAFE
from backend.tests.unit.test_structured_profiler import FakeRouter, _write

db_only = pytest.mark.skipif(not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")

# ---------------------------------------------------------------- pure units

DECLARED = [{"name": "plan", "type": "text", "column": "plan", "optional": True},
            {"name": "year_from", "type": "integer", "column": "year"},
            {"name": "year_to", "type": "integer", "column": "year"}]


def test_call_params_refuse_unknown_and_missing_and_wrong_types():
    _p, bad, _s = validate_call_params({"year_from": 2023, "year_to": 2024, "drug": "D1"}, DECLARED)
    assert [b["param"] for b in bad] == ["drug"] and "not a parameter" in bad[0]["error"]
    _p, bad, _s = validate_call_params({"year_from": 2023}, DECLARED)
    assert [b["param"] for b in bad] == ["year_to"] and "required" in bad[0]["error"]
    _p, bad, _s = validate_call_params({"year_from": "last year", "year_to": 2024}, DECLARED)
    assert [b["param"] for b in bad] == ["year_from"] and "integer" in bad[0]["error"]


def test_optional_param_may_be_omitted_and_allowed_values_still_apply():
    params, bad, _s = validate_call_params({"year_from": 2023, "year_to": "2024"}, DECLARED)
    assert bad == [] and "plan" not in params
    restricted = [{**DECLARED[0], "allowed_values": ["P1", "P2"]}, *DECLARED[1:]]
    _p, bad, _s = validate_call_params({"plan": "P9", "year_from": 2023, "year_to": 2024}, restricted)
    assert bad and bad[0]["param"] == "plan"


def _item(**kw):
    it = {"id": "q", "sql": 'SELECT 1 FROM "t"', "params": [{"name": "x", "type": "text"}], **kw}
    record_test(it, "passed")
    return it


def test_only_passed_items_of_a_current_catalog_are_selectable():
    meta = {"table_name": "t", "catalog_version": CATALOG_VERSION}
    assert selectable(meta, _item()) == (True, None)
    assert not selectable({**meta, "catalog_version": "old"}, _item())[0]
    changed = _item()
    changed["sql"] += " WHERE 1 = 1"                  # SQL changed after its test
    assert "changed after" in selectable(meta, changed)[1]
    idle = _item()
    record_test(idle, "not_exercised")
    assert "not_exercised" in selectable(meta, idle)[1]
    untyped = _item(params=[{"name": "x"}])
    assert "parameter" in selectable(meta, untyped)[1]
    assert idle["verification"]["sql_sha"] == sql_sha(idle["sql"])


def test_catalog_offered_to_the_selector_hides_unselectable_items():
    meta = {"table_name": "t", "catalog_version": CATALOG_VERSION, "columns": [],
            "metrics": [_item(id="ok_q", label="ok")], "filters": [], "queries": []}
    bad = _item(id="idle_q", label="idle")
    record_test(bad, "not_exercised")
    meta["queries"] = [bad]
    text = str(catalog_for_prompt(meta, only_selectable=True))
    assert "ok_q" in text and "idle_q" not in text


META = {"table_name": "costs",
        "columns": [{"name": c, "type": t} for c, t in
                    [("plan", "text"), ("plan_name", "text"), ("drug", "text"), ("year", "integer"),
                     ("cost", "numeric"), ("release", "integer")]]}
ANALYSIS = {"dependencies": {"plan": ["plan_name"]}, "near_dependencies": {},
            "repeats": {"cost": [{"column": "year", "with": [], "share_of_groups_repeated": 0.9,
                                  "share_identical": 0.8}]}}


def test_repeated_values_do_not_remove_a_dimension():
    m, err = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
                                "display_columns": ["plan_name"], "period_columns": ["year"]}, META, ANALYSIS)
    assert err is None and m["period_columns"] == ["year"] and m["version_column"] is None
    assert observation_key(m) == ["plan", "drug", "year"]


def test_display_label_must_follow_from_an_identity_column():
    m, _ = normalize_measure({"id": "cost", "value_column": "cost", "identity_columns": ["plan"],
                              "display_columns": ["plan_name", "drug"], "period_columns": ["year"]},
                             META, ANALYSIS)
    assert m["identity_columns"] == ["plan", "drug"] and m["display_columns"] == ["plan_name"]
    assert any("drug" in r for r in m["review"])        # ambiguous -> kept, flagged


def test_legacy_multi_entity_definition_is_split_by_measured_dependencies():
    m, _ = normalize_measure({"id": "cost", "value_column": "cost",
                              "entity_columns": ["plan", "plan_name", "drug"], "period_columns": ["year"]},
                             META, ANALYSIS)
    assert m["identity_columns"] == ["plan", "drug"] and m["display_columns"] == ["plan_name"]


def test_version_rule_needs_a_reason_and_may_not_be_a_dimension():
    base = {"id": "cost", "value_column": "cost", "identity_columns": ["plan", "drug"],
            "period_columns": ["year"]}
    m, _ = normalize_measure({**base, "version_column": "release"}, META, ANALYSIS)
    assert m["version_column"] is None and any("without a reason" in r for r in m["review"])
    m, _ = normalize_measure({**base, "version_column": "year", "version_reason": "r"}, META, ANALYSIS)
    assert m["version_column"] is None and "year" in observation_key(m)
    m, _ = normalize_measure({**base, "version_column": "release",
                              "version_reason": "each release restates the same figures"}, META, ANALYSIS)
    assert m["version_column"] == "release" and "release" not in observation_key(m)


def test_a_measure_may_only_get_ok_fix_text_fix_definition_or_drop():
    assert allowed_verdicts("measures") == {"ok", "fix_text", "fix_definition", "drop"}
    assert "split" in allowed_verdicts("metrics") and "fix_query" in allowed_verdicts("lookups")


# ---------------------------------------------------------------- database

COSTS = (b"plan,plan_name,drug,year,cost,note\n"
         b"P1,Plan One,D1,2023,10,\nP1,Plan One,D2,2023,20,\nP2,Plan Two,D1,2023,30,\nP2,Plan Two,D2,2023,40,\n"
         b"P1,Plan One,D1,2024,11,\nP1,Plan One,D2,2024,21,\nP2,Plan Two,D1,2024,31,\nP2,Plan Two,D2,2024,41,\n")
REGIONS = b"plan,region\nP1,North\nP1,South\nP2,North\n"


def _costs_profile():
    roles = {"plan": "identifier", "plan_name": "attribute", "drug": "dimension", "year": "time",
             "cost": "measure", "note": "text"}
    return {"description": "cost per plan, drug and year", "grain": "one row per plan, drug, year",
            "columns": [{"name": c, "description": c, "role": r, "concept": None} for c, r in roles.items()],
            "metrics": [], "queries": [],
            "filters": [{"id": "by_note", "label": "By note", "description": "", "columns": ["note"],
                         "concept": None, "params": [{"name": "n", "description": "", "example": "x"}],
                         "query": {"select": ["plan", "cost"],
                                   "where": [{"column": "note", "op": "=", "param": "n"}]}}]}


def _regions_profile():
    return {"description": "regions of a plan", "grain": "one row per plan and region",
            "columns": [{"name": "plan", "description": "plan", "role": "identifier", "concept": None},
                        {"name": "region", "description": "region", "role": "dimension", "concept": None}],
            "metrics": [], "filters": [], "queries": []}


@pytest.fixture
async def loaded(monkeypatch, tmp_path):
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope
    from backend.app.services.structured import cache as tc
    from backend.app.services.structured.db_load import drop_data_tables, load_and_link
    from backend.app.services.structured.profiler import profile_folder
    from backend.tests.unit.test_structured_db import DSN

    monkeypatch.setenv("DATABASE_URL", DSN)
    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "cache"))
    reset_engine_cache()
    async with session_scope() as s:
        await drop_data_tables(s)
        await s.execute(text("TRUNCATE graphrag.graph_relationships, graphrag.intelligence_artifacts, "
                             "graphrag.entities, graphrag.ontology_classes CASCADE"))
    root = tmp_path / "d"
    _write(root, "costs.csv", COSTS)
    _write(root, "plan_regions.csv", REGIONS)
    cache = tc.MetadataCache()
    router = FakeRouter({"table_profile": [_costs_profile(), _regions_profile()], "table_lookups": [{"lookups": []}]})
    run = await profile_folder(root, router, cache=cache, concurrency=1, log=lambda *_: None)
    assert run.failed == []
    await load_and_link(run, root_name="d", cache=cache, log=lambda *_: None)
    yield cache
    async with session_scope() as s:
        await drop_data_tables(s)
    await dispose_engine()


async def _state():
    from backend.app.db.session import session_scope
    from backend.app.services.structured.analysis import analyze_table
    from backend.app.services.structured.db_load import registered_tables
    from backend.app.services.structured.metadata import column_types

    async with session_scope() as s:
        reg = await registered_tables(s)
        meta = reg["costs"]["meta"]
        analysis = await analyze_table(s, meta)
    return meta, analysis, {n: column_types(r["meta"]) for n, r in reg.items()}, {n: r["meta"] for n, r in reg.items()}


def _raw_measure(identity, display=()):
    return {"id": "cost", "label": "Cost", "value_column": "cost", "identity_columns": list(identity),
            "display_columns": list(display), "period_columns": ["year"], "qualifier_columns": []}


@db_only
async def test_load_tests_every_stored_query_and_keeps_not_exercised_ones_out(loaded):
    meta, *_ = await _state()
    by_note = next(f for f in meta["filters"] if f["id"] == "by_note")
    assert by_note["verification"]["status"] == "not_exercised"        # no value to try it with
    assert meta["catalog_version"] == CATALOG_VERSION
    assert not selectable(meta, by_note)[0]
    assert "by_note" not in str(catalog_for_prompt(meta, only_selectable=True))


@db_only
async def test_measure_must_return_every_observation_exactly_once(loaded):
    from backend.app.services.structured.refine import measure_item, test_item

    meta, analysis, types, partners = await _state()
    # drug left out of the identity: (plan, year) does not identify one value
    m, _ = normalize_measure(_raw_measure(["plan"], ["plan_name"]), meta, analysis)
    bad = measure_item(m, meta)
    r = await test_item(meta, "measures", bad, analysis, types, partners=partners)
    assert r["status"] == "failed" and "more than once" in r["error"]
    # complete identity: 8 observations, each once; plan_name only displayed
    m, _ = normalize_measure(_raw_measure(["plan", "drug"], ["plan_name"]), meta, analysis)
    good = measure_item(m, meta)
    r = await test_item(meta, "measures", good, analysis, types, partners=partners)
    assert r["status"] == "passed", r
    ident = good["verification"]["identity"]
    assert ident["source"] == ident["returned_keys"] == ident["returned_rows"] == 8
    assert {p["name"] for p in good["params"] if p.get("optional")} >= {"plan", "drug"}   # identity = filters


@db_only
async def test_join_on_a_non_unique_key_is_refused(loaded):
    from backend.app.services.structured.refine import test_item

    meta, analysis, types, partners = await _state()
    lookup = {"id": "cost_by_region", "label": "Cost by region", "description": "", "other_table": "plan_regions",
              "join": [{"column": "plan", "other_column": "plan"}], "cardinality": "many_to_one", "params": [],
              "query": {"select": ["plan_regions.region", {"agg": "sum", "column": "cost", "as": "total"}]}}
    r = await test_item(meta, "lookups", copy.deepcopy(lookup), analysis, types, partners=partners)
    assert r["status"] == "failed" and "declared many_to_one" in r["error"]
    r = await test_item(meta, "lookups", {**copy.deepcopy(lookup), "cardinality": "many_to_many"},
                        analysis, types, partners=partners)
    assert r["status"] == "failed" and "multiplied by the join" in r["error"]


@db_only
async def test_failed_measure_definition_is_repaired_once_and_retested(loaded):
    from backend.app.services.structured.refine import refine_registered

    refine = {"columns": [], "usage_notes": [], "common_questions": [],
              "measures": [_raw_measure(["plan"], ["plan_name"])]}
    repair = {"measures": [_raw_measure(["plan", "drug"], ["plan_name"])]}
    router = FakeRouter({"table_refine": [refine, repair]})
    reps = await refine_registered(router, tables=["costs"], verify=False, cache=loaded, log=lambda *_: None)
    assert len(router.calls) == 2 and "FAILED MEASURES" in router.calls[1][1]
    assert reps[0]["repaired"] == ["measure_cost"] and reps[0]["failed_measures"] == []
    meta, *_ = await _state()
    log = meta["measure_repairs"][0]
    assert log["outcome"] == "repaired" and "more than once" in log["failure"]
    assert log["original"]["identity_columns"] == ["plan"] and log["proposed"]["identity_columns"] == ["plan", "drug"]
    item = meta["measures"][0]
    assert item["verification"]["status"] == "passed"
    # verification is configured but skipped in this run: the repaired measure
    # passed its data test yet waits for a semantic verdict
    ok, why = selectable(meta, item)
    assert not ok and "semantic review pending" in why
    meta["semantic_review"] = {"required": False}
    assert selectable(meta, item)[0]


@db_only
async def test_unresolved_measure_failure_stays_inspectable_and_unselectable(loaded):
    from backend.app.services.structured.refine import refine_registered

    refine = {"columns": [], "usage_notes": [], "common_questions": [],
              "measures": [_raw_measure(["plan"], ["plan_name"])]}
    router = FakeRouter({"table_refine": [refine, {"measures": [None]}]})
    reps = await refine_registered(router, tables=["costs"], verify=False, cache=loaded, log=lambda *_: None)
    assert reps[0]["failed_measures"] == ["measure_cost"]
    meta, *_ = await _state()
    assert meta["measures"] == []
    kept = meta["failed_measures"][0]
    assert kept["id"] == "measure_cost" and "more than once" in kept["failure"] and kept["definition"]
    assert meta["measure_repairs"][0]["outcome"] == "unresolved"

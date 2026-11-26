"""Correctness fixes of the verified catalog (generic fixtures only):

* dependency invalidation: a query records the data version of every table
  it reads; a changed table makes it stale;
* conflicting values that tie at the winning version keep a measure
  unavailable;
* identity validation runs per qualifier group and records its coverage;
* retries are tied to the failed call they replace and keep its list members;
  malformed params are a selection error;
* the final catalog's concepts are mapped to ontology classes once,
  idempotently, with proposals persisted for an ontology update.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.app.services.structured.metadata import (
    CATALOG_VERSION,
    describe_query,
    record_test,
    selectable,
)
from backend.app.services.structured.retrieval_step import _execute, broadens, requested_values
from backend.tests.unit.test_structured_catalog_policy import (  # noqa: F401  (fixture)
    _raw_measure,
    _state,
    db_only,
    loaded,
)
from backend.tests.unit.test_structured_profiler import FakeRouter

# ---------------------------------------------------------------- pure units


def test_query_tested_on_other_data_is_not_selectable():
    it = {"id": "q", "sql": 'SELECT 1 FROM "t"', "params": []}
    record_test(it, "passed")
    it["verification"]["data_versions"] = {"t": "sha-old", "u": "sha-u"}
    meta = {"table_name": "t", "catalog_version": CATALOG_VERSION}
    assert selectable(meta, it, {"t": "sha-old", "u": "sha-u"}) == (True, None)
    ok, why = selectable(meta, it, {"t": "sha-old", "u": "sha-new"})
    assert not ok and "table u changed" in why


def test_retry_keeps_valid_list_members_and_may_replace_wrong_ones():
    item = {"params": [{"name": "items", "column": "drug", "type": "text"}]}
    rec = {"params": {"items": ["D1", "Dx"]}, "missing_values": [{"value": "Dx"}]}
    keep = requested_values(rec, item)
    assert keep == {"drug|False|False": {"members": ["D1"]}}
    assert broadens({"params": {"items": ["D1", "D2"]}}, item, keep) is None
    assert "drops ['D1']" in broadens({"params": {"items": ["D2"]}}, item, keep)


def test_malformed_params_are_a_selection_error_not_empty_params():
    _rec, res, err = asyncio.run(_execute({"table": "t", "query_id": "q", "params": ["D1"]}, {},
                                         max_rows=5, timeout_ms=1000, all_tables=set()))
    assert res is None and "JSON object" in err


def test_capability_says_what_an_omitted_total_parameter_returns():
    meta = {"table_name": "t", "analysis": {"total_rows": {"company": "ALL"}},
            "columns": [{"name": "company", "type": "text"}, {"name": "year", "type": "integer"},
                        {"name": "value", "type": "numeric"}]}
    item = {"id": "v", "label": "Value", "sql": "SELECT 1",
            "params": [{"name": "company", "type": "text", "column": "company", "optional": True}],
            "query": {"select": ["year", {"agg": "sum", "column": "value", "as": "v"}],
                      "where": [{"column": "company", "op": "=", "param": "company", "optional": True}],
                      "group_by": ["year"]}}
    cap = describe_query(meta, "metrics", item)
    p = next(p for p in cap["parameters"] if p["name"] == "company")
    assert p["optional"] and "when_omitted" in p
    assert "year" in str(cap.get("grouping_columns"))


# ---------------------------------------------------------------- database


async def _sql(stmt: str, binds: dict | None = None):
    from sqlalchemy import text

    from backend.app.db.session import session_scope

    async with session_scope() as s:
        await s.execute(text(stmt), binds or {})


@db_only
@pytest.mark.usefixtures("loaded")
async def test_ties_at_the_winning_version_keep_the_measure_unavailable():
    from backend.app.services.structured.refine import measure_item, normalize_measure, test_item

    await _sql('ALTER TABLE graphrag."costs" ADD COLUMN release integer DEFAULT 1')
    await _sql('INSERT INTO graphrag."costs" (plan, plan_name, drug, year, cost, release) '
               "VALUES ('P1', 'Plan One', 'D1', 2023, 99, 1)")
    meta, analysis, types, partners = await _state()
    meta["columns"].append({"name": "release", "type": "integer", "role": "attribute"})
    types["costs"]["release"] = "integer"
    m, _ = normalize_measure(_raw_measure(["plan", "drug"], ["plan_name"]), meta, analysis)
    m.update(version_column="release", version_reason="a later release restates a value")
    item = measure_item(m, meta)
    r = await test_item(meta, "measures", item, analysis, types, partners=partners)
    assert r["status"] == "failed" and "winning precedence (release)" in r["error"]
    assert item["verification"]["identity"]["tie_examples"][0]["values"]
    # a later release resolves it: the restated value wins, nothing ties
    await _sql('UPDATE graphrag."costs" SET release = 2 WHERE cost = 99')
    item = measure_item(m, meta)
    r = await test_item(meta, "measures", item, analysis, types, partners=partners)
    assert r["status"] == "passed", r
    assert item["verification"]["identity"]["coverage"] == "complete"
    assert item["verification"]["data_versions"] == {"costs": meta["source"]["file_sha"]}
    assert item["capability"]["kind"]


@db_only
@pytest.mark.usefixtures("loaded")
async def test_changed_partner_table_retests_the_queries_that_read_it():
    from backend.app.services.structured.refine import ensure_tested

    meta, analysis, types, partners = await _state()
    tested = [it for sec in ("measures", "metrics", "filters", "queries", "lookups")
              for it in meta.get(sec) or [] if it.get("verification")]
    assert tested and all("data_versions" in it["verification"] for it in tested)
    meta["source"]["file_sha"] = "changed"                     # the table was reloaded
    counts = await ensure_tested(meta, analysis=analysis, all_types=types, partners=partners)
    assert counts["retested"] >= len(tested)
    assert all(it["verification"]["data_versions"]["costs"] == "changed" for it in tested)


@db_only
@pytest.mark.usefixtures("loaded")
async def test_final_catalog_concepts_map_once_and_idempotently():
    from backend.app.services.structured.concepts import catalog_concepts, reconcile_concepts
    from backend.app.services.structured.refine import measure_item, normalize_measure, test_item

    await _sql("INSERT INTO graphrag.ontology_classes (iri, namespace, label) VALUES "
               "('https://ex.org/o#Cost', 'https://ex.org/o#', 'Cost')")
    meta, analysis, types, partners = await _state()
    m, _ = normalize_measure(_raw_measure(["plan", "drug"], ["plan_name"]), meta, analysis)
    item = measure_item(m, meta)
    await test_item(meta, "measures", item, analysis, types, partners=partners)
    meta["measures"] = [item]
    concepts, roles = catalog_concepts(meta)
    assert "measure:cost" in concepts and roles[item["id"]]["measures"][0] == "measure:cost"
    open_keys = [k for k in concepts if k != "measure:cost"]
    reply = {"concepts": [{"key": k, "iri": None, "label": f"Thing {k}", "parent_label": "Dimension",
                           "ambiguous": [], "reason": "no class"} for k in open_keys]}
    router = FakeRouter({"table_concept_map": [reply]})
    rep = await reconcile_concepts(meta, router, base_iri="https://ex.org/new#", log=lambda *_: None)
    cmap = meta["concept_map"]
    assert cmap["measure:cost"]["iri"] == "https://ex.org/o#Cost" and cmap["measure:cost"]["status"] == "reused"
    assert rep["llm_calls"] == 1
    assert {p["key"] for p in meta["class_proposals"]} == set(open_keys)
    assert item["concepts"]["measures"] == ["measure:cost"]
    # second run: nothing new is asked, the same mapping and proposals come back
    before = (dict(cmap), list(meta["class_proposals"]))
    rep2 = await reconcile_concepts(meta, FakeRouter({}), base_iri="https://ex.org/new#", log=lambda *_: None)
    assert rep2["llm_calls"] == 0
    assert meta["concept_map"] == before[0] and meta["class_proposals"] == before[1]


@db_only
@pytest.mark.usefixtures("loaded")
async def test_concepts_a_mapping_reply_skips_are_asked_once_more():
    from backend.app.services.structured.concepts import catalog_concepts, reconcile_concepts
    from backend.app.services.structured.refine import measure_item, normalize_measure, test_item

    meta, analysis, types, partners = await _state()
    m, _ = normalize_measure(_raw_measure(["plan", "drug"], ["plan_name"]), meta, analysis)
    item = measure_item(m, meta)
    await test_item(meta, "measures", item, analysis, types, partners=partners)
    meta["measures"] = [item]
    keys = list(catalog_concepts(meta)[0])
    assert len(keys) >= 2
    first = {"concepts": [{"key": keys[0], "iri": None, "label": None, "reason": "identifier"}]}
    second = {"concepts": [{"key": k, "iri": None, "label": f"T{i}", "parent_label": "Dimension"}
                           for i, k in enumerate(keys[1:])]}
    router = FakeRouter({"table_concept_map": [first, second]})
    rep = await reconcile_concepts(meta, router, log=lambda *_: None)
    assert rep["llm_calls"] == 2 and keys[0] not in router.calls[1][1]
    assert all(meta["concept_map"][k]["status"] == "proposed" for k in keys[1:])
    assert meta["concept_map"][keys[0]]["reason"] == "identifier"

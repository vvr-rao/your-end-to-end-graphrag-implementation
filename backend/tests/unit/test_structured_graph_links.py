"""structured.graph_links (time + place ties) and query tags / ranking."""

from __future__ import annotations

from datetime import date

import pytest

from backend.app.services.structured.graph_links import (
    identifiers_for,
    period_of,
    place_groups,
    place_level,
    resolve_place,
    value_place_level,
)
from backend.app.services.structured.metadata import item_tags
from backend.app.services.structured.retrieval_step import rank_items
from backend.tests.unit.test_structured_db import _SAFE, DSN
from backend.tests.unit.test_structured_profiler import FakeRouter, _write


@pytest.mark.parametrize("value, want", [
    ("2024", (2024, None, None)), (2024, (2024, None, None)), ("2024Q3", (2024, 3, None)),
    ("Q3 2024", (2024, 3, None)), ("2024M07", (2024, 3, 7)), ("2024-07", (2024, 3, 7)),
    ("2024-07-15", (2024, 3, 7)), (date(2023, 12, 31), (2023, 4, 12)), ("20240715", (2024, 3, 7)),
    ("abc", None), ("12", None), (12.5, None), (None, None),
])
def test_period_values_are_parsed(value, want):
    assert period_of(value) == want


def test_identifiers_never_finer_than_the_value():
    ps = {(2024, None, None), (2023, 4, 12)}
    assert identifiers_for(ps, "year") == {"YEAR_2024", "YEAR_2023"}
    assert identifiers_for(ps, "month") == {"YEAR_2024", "MONTH_2023_12"}
    assert identifiers_for(ps, "quarter") == {"YEAR_2024", "Q4_2023"}


def test_iso_codes_become_place_names():
    assert resolve_place("AU", 3) == {"name": "Australia", "level": 3, "code": "AU"}
    assert resolve_place("KOR", 3)["name"] == "South Korea"
    sub = resolve_place("CN-ZJ", 2)
    assert sub["level"] == 2 and sub["country"] == "China" and "Zhejiang" in sub["name"]
    assert resolve_place("US-DE", 3)["level"] == 2           # a jurisdiction holding a subdivision code
    assert resolve_place("SANTIAGO", 1) == {"name": "Santiago", "level": 1}
    assert resolve_place("Chile", 3)["code"] == "CL"
    assert resolve_place("AU", 1) == {"name": "AU", "level": 1}  # a 2-letter city stays as written


def test_place_columns_and_address_groups():
    cols = [{"name": "hq_city", "type": "text", "role": "dimension"},
            {"name": "hq_region", "type": "text", "role": "dimension"},
            {"name": "hq_country", "type": "text", "role": "dimension"},
            {"name": "legal_country", "type": "text", "role": "dimension"},
            {"name": "origin", "type": "text", "role": "dimension", "entity_kind": "country"},
            {"name": "value", "type": "numeric", "role": "measure"},
            {"name": "name", "type": "text", "role": "identifier"}]
    assert [place_level(c) for c in cols] == [1, 2, 3, 3, 3, None, None]
    groups = sorted(place_groups(cols))
    assert [("hq_city", 1), ("hq_region", 2), ("hq_country", 3)] in groups
    assert [("legal_country", 3)] in groups


META = {
    "table_name": "facts",
    "columns": [{"name": "cik", "type": "text", "role": "identifier"},
                {"name": "tag", "type": "text", "role": "dimension",
                 "class_iri": "https://ex.org/o#Metric", "class_label": "Metric"},
                {"name": "country", "type": "text", "role": "dimension"},
                {"name": "end", "type": "date", "role": "time"},
                {"name": "val", "type": "numeric", "role": "measure"}],
    "analysis": {"grain": ["cik", "tag", "end"]},
    "measures": [{"id": "measure_revenue", "label": "Revenue", "class_iri": "https://ex.org/o#Revenue",
                  "class_label": "Revenue", "params": [{"name": "cik", "column": "cik"}],
                  "measure": {"period_columns": ["end"]},
                  "query": {"select": ["cik", "end", "val"],
                            "where": [{"column": "cik", "op": "=", "param": "cik"}]}}],
    "metrics": [{"id": "by_country", "label": "By country", "params": [{"name": "country", "column": "country"}],
                 "query": {"select": ["country", {"agg": "count", "as": "n"}],
                           "where": [{"column": "country", "op": "=", "param": "country"}]}}],
}


def test_items_are_tagged_and_ranked_by_what_the_question_reached():
    tags = item_tags(META, "measures", META["measures"][0])
    assert tags == {"classes": [{"iri": "https://ex.org/o#Revenue", "label": "Revenue"}],
                    "entity_columns": ["cik"], "place_columns": [], "time_columns": ["end"]}
    assert item_tags(META, "metrics", META["metrics"][0])["place_columns"] == ["country"]
    r = rank_items(META, {"classes": {"Revenue"}, "columns": {"cik"}, "time": True})
    assert r["measure_revenue"][0] > r["by_country"][0] == 0
    assert r["measure_revenue"][1] == ["class Revenue", "entity in cik", "period in end"]
    r = rank_items(META, {"classes": set(), "columns": {"country"}, "time": False})
    assert r["by_country"][0] > 0 and r["measure_revenue"][0] == 0


# ------------------------------------------------------------------ database

def test_place_columns_are_found_by_their_values():
    """A column whose name, kind and class carry no place word is still a
    place column when its values are places: countries (names or 3-letter
    codes) or places the graph already holds. Two-letter codes are not
    evidence (state / language / unit codes collide with ISO alpha-2)."""
    assert value_place_level(["Japan", "Rep. of Korea", "Viet Nam", "Germany", "Other Asia, nes"], {}) == 3
    assert value_place_level(["JPN", "KOR", "TWN", "DEU"], {}) == 3
    known = {"ontario": 2, "quebec": 2, "alberta": 2}
    assert value_place_level(["Ontario", "Quebec", "Alberta", "Yukon"], known) == 2
    assert value_place_level(["CA", "IN", "DE", "FR"], {}) is None          # two-letter codes: no evidence
    assert value_place_level(["Urea", "Copper", "Crude oil, Brent", "Gold"], {}) is None
    assert value_place_level(["Japan", "Gold", "Urea", "Tin", "Zinc"], {}) is None    # 1 of 5 is not half
    assert value_place_level(["Japan", "Chad"], {}) is None                  # too few values to judge
    assert value_place_level(["2024", "2025", "-", "1.5"], {}) is None


db_only = pytest.mark.skipif(not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated *_test database")

PLACES_CSV = (b"name,hq_city,hq_region,hq_country,year,value\n"
              b"Alpha,SANTIAGO,CL-RM,CL,2022,1\nBeta,Toronto,CA-ON,CA,2023,2\n"
              b"Gamma,Perth,AU-WA,AU,2023,3\n")


@pytest.fixture
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", DSN)
    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "cache"))
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import drop_data_tables

    reset_engine_cache()
    async with session_scope() as s:
        await drop_data_tables(s)
        await s.execute(text("TRUNCATE graphrag.graph_relationships, graphrag.intelligence_artifacts, "
                             "graphrag.entities, graphrag.ontology_classes, graphrag.time_instances CASCADE"))
        ids = {}
        for lab in ("Place", "Country", "AdministrativeArea", "City"):
            ids[lab] = (await s.execute(text(
                "INSERT INTO graphrag.ontology_classes (iri, namespace, label) VALUES (:i, 'https://ex.org/o#', :l) "
                "RETURNING id"), {"i": f"https://ex.org/o#{lab}", "l": lab})).scalar()
        # Chile already exists (from documents); the others are minted
        await s.execute(text(
            "INSERT INTO graphrag.entities (entity_identifier, iri, name, normalized_name, class_id) "
            "VALUES ('https://ex.org/e#chile', 'https://ex.org/e#chile', 'Chile', 'chile', :c)"),
            {"c": ids["Country"]})
    yield session_scope
    async with session_scope() as s:
        await drop_data_tables(s)
    await dispose_engine()


def _profile_reply():
    return {"description": "companies by headquarters", "grain": "one row per company",
            "columns": [{"name": c, "description": c, "role": r, "concept": None} for c, r in
                        [("name", "identifier"), ("hq_city", "dimension"), ("hq_region", "dimension"),
                         ("hq_country", "dimension"), ("year", "time"), ("value", "measure")]],
            "metrics": [], "filters": [{"id": "by_country", "label": "By country", "description": "",
                                        "columns": ["hq_country"], "concept": None,
                                        "params": [{"name": "c", "description": "", "example": "CL"}],
                                        "query": {"select": ["name", "hq_country"],
                                                  "where": [{"column": "hq_country", "op": "=", "param": "c"}]}}],
            "queries": []}


@db_only
async def test_table_links_known_places_only_and_mints_time_instances(db, tmp_path):
    """Graph identity policy: a known place (Chile) gets an edge; an unknown one
    (Canada, Toronto, ...) stays unresolved; no entity is created on load or
    on relink. Time instances keep their own policy (minted)."""
    from sqlalchemy import text

    from backend.app.services.structured import cache as tc
    from backend.app.services.structured.db_load import load_and_link, registered_tables
    from backend.app.services.structured.profiler import profile_folder
    from backend.app.services.structured.refine import refine_registered

    async def embed(texts):
        return [[0.01] * 1024 for _ in texts]

    async def n_entities():
        async with db() as s:
            return (await s.execute(text("SELECT count(*) FROM graphrag.entities"))).scalar()

    before = await n_entities()
    root = tmp_path / "d"
    _write(root, "hq.csv", PLACES_CSV)
    cache = tc.MetadataCache()
    run = await profile_folder(root, FakeRouter({"table_profile": [_profile_reply()]}), cache=cache,
                               log=lambda *_: None)
    await load_and_link(run, root_name="d", cache=cache, embed=embed, log=lambda *_: None)
    assert await n_entities() == before                       # load minted nothing
    refine = {"columns": [], "usage_notes": [], "common_questions": [],
              "measures": [{"id": "value", "label": "Value", "value_column": "value",
                            "identity_columns": ["name"], "period_columns": ["year"],
                            "qualifier_columns": []}]}
    await refine_registered(FakeRouter({"table_refine": [refine]}), verify=False, cache=cache, embed=embed,
                            log=lambda *_: None)
    await refine_registered(FakeRouter({}), verify=False, relink=True, cache=cache, embed=embed,
                            log=lambda *_: None)
    assert await n_entities() == before                       # refine + relink minted nothing
    async with db() as s:
        places = dict((await s.execute(text(
            "SELECT e.name, c.label FROM graphrag.graph_relationships r "
            "JOIN graphrag.entities e ON e.id = r.target_node_id JOIN graphrag.ontology_classes c ON c.id = e.class_id "
            "WHERE r.relationship_type = 'tablePlace'"))).all())
        assert places == {"Chile": "Country"}                 # known place linked, unknown unresolved
        times = sorted((await s.execute(text(
            "SELECT t.time_identifier FROM graphrag.graph_relationships r JOIN graphrag.time_instances t "
            "ON t.id = r.target_node_id WHERE r.relationship_type = 'tableTime'"))).scalars().all())
        assert times == ["YEAR_2022", "YEAR_2023"]
        meta = (await registered_tables(s))["hq"]["meta"]
    stats = meta["column_stats"]                       # measured on the loaded table
    assert stats["year"]["periods"] == {"year": [2022, 2023]} and stats["name"]["distinct"] == 3
    tags = next(it["tags"] for it in meta["filters"] if it["id"] == "by_country")
    assert tags["place_columns"] == ["hq_country"]


@db_only
async def test_place_column_without_a_place_word_links_by_its_values(db, tmp_path):
    """`reporter` / `economy_code` are named, described and typed with no
    place word; their values (country names, ISO-3 codes) still link the table
    to the country the graph holds -- and nothing is minted."""
    from sqlalchemy import text

    from backend.app.services.structured import cache as tc
    from backend.app.services.structured.db_load import load_and_link
    from backend.app.services.structured.profiler import profile_folder

    async def embed(texts):
        return [[0.01] * 1024 for _ in texts]

    root = tmp_path / "t"
    _write(root, "flows.csv", b"reporter,economy_code,year,value\nChile,CHL,2023,1\nJapan,JPN,2023,2\n"
                              b"Germany,DEU,2023,3\nPeru,PER,2024,4\n")
    reply = {"description": "trade by reporting economy", "grain": "one row per reporter and year",
             "columns": [{"name": c, "description": c, "role": r, "concept": None} for c, r in
                         [("reporter", "attribute"), ("economy_code", "identifier"),
                          ("year", "time"), ("value", "measure")]],
             "metrics": [], "filters": [], "queries": []}
    cache = tc.MetadataCache()
    run = await profile_folder(root, FakeRouter({"table_profile": [reply]}), cache=cache, log=lambda *_: None)
    lines: list[str] = []
    async with db() as s:
        before = (await s.execute(text("SELECT count(*) FROM graphrag.entities"))).scalar()
    await load_and_link(run, root_name="t", cache=cache, embed=embed, log=lines.append)
    async with db() as s:
        rows = (await s.execute(text(
            "SELECT e.name, r.extra_metadata FROM graphrag.graph_relationships r "
            "JOIN graphrag.entities e ON e.id = r.target_node_id WHERE r.relationship_type = 'tablePlace'"))).all()
        after = (await s.execute(text("SELECT count(*) FROM graphrag.entities"))).scalar()
    assert [r[0] for r in rows] == ["Chile"] and after == before          # known place only, nothing minted
    assert sorted(rows[0][1]["columns"]) == ["economy_code", "reporter"]
    assert any("found by their values: reporter, economy_code" in ln for ln in lines)

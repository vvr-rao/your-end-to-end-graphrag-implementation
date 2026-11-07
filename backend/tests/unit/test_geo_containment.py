"""The deterministic guard on what `enrich-geo` is allowed to write.

These edges carry no document behind them, so a wrong one is unfalsifiable
from the corpus -- which is exactly why the level arithmetic exists rather
than trusting the model. Both rejections below were produced by gpt-4o-mini on
a real 46-place list.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from backend.app.services import retrieval_sql as rsql
from backend.app.services.db_geo_enrich import (
    DEFAULT_GEO_CLASS_LABELS,
    NON_PLACE_CLASS_LABELS,
    _container_type,
    containment_is_plausible,
    plan_ontology_containment,
)
from backend.app.services.predicates import GRAPHRAG_LOCATED_IN


def test_accepts_a_strictly_upward_pair():
    assert containment_is_plausible("City", "Country")
    assert containment_is_plausible("City", "State")
    assert containment_is_plausible("AdministrativeArea", "Country")
    assert containment_is_plausible("Country", "Continent")


def test_rejects_same_level_pairs():
    # "Canada is in United States" and "United States is in United States of
    # America" -- both observed, both Country -> Country.
    assert not containment_is_plausible("Country", "Country")
    assert not containment_is_plausible("City", "City")


def test_rejects_downward_pairs():
    assert not containment_is_plausible("Country", "City")


def test_unknown_child_needs_a_specific_container():
    # A mistyped entity may still sit inside a country...
    assert containment_is_plausible("Stadium", "Country")
    assert containment_is_plausible(None, "Region")
    # ...but not inside something equally unspecific.
    assert not containment_is_plausible("Stadium", "Venue")


def test_unknown_parent_is_always_rejected():
    # The container is the claim; an unrecognised one is where a bogus edge
    # would get in.
    assert not containment_is_plausible("City", "Place")
    assert not containment_is_plausible("City", None)


def test_case_and_whitespace_insensitive():
    assert containment_is_plausible("  city ", "COUNTRY")


def test_spatialthing_is_not_a_default_seed_label():
    # foaf:Person is a subclass of geo:SpatialThing, so seeding from it pulled
    # 262 people into a 461-"place" list.
    assert "SpatialThing" not in DEFAULT_GEO_CLASS_LABELS
    assert "Person" in NON_PLACE_CLASS_LABELS


# --- the ontology step: containers come from `partOfClass`, not from a model ---

_IRI = "https://veerla-ramrao.ai/ontology/geography#"
_CLASSES = {
    _IRI + "Continent": {"label": "Continent", "part_of": []},
    _IRI + "Asia": {"label": "Asia", "part_of": [], "superclasses": [_IRI + "Continent"]},
    _IRI + "EastAsia": {"label": "East Asia", "part_of": [_IRI + "Asia"]},
    _IRI + "WesternAsia": {"label": "Western Asia", "part_of": [_IRI + "Asia"]},
    _IRI + "Japan": {"label": "Japan", "part_of": [_IRI + "EastAsia"]},
    _IRI + "SouthKorea": {"label": "South Korea", "part_of": [_IRI + "EastAsia"]},
    _IRI + "Georgia": {"label": "Georgia", "part_of": [_IRI + "WesternAsia"]},
}


def test_a_named_country_brings_its_containers():
    plan = plan_ontology_containment([("jp", ["Japan"], 3)], _CLASSES)
    assert set(plan.to_mint) == {_IRI + "EastAsia", _IRI + "Asia"}
    assert set(plan.links) == {(_IRI + "Japan", _IRI + "EastAsia"),
                               (_IRI + "EastAsia", _IRI + "Asia")}


def test_a_container_already_an_entity_is_not_minted():
    plan = plan_ontology_containment(
        [("jp", ["Japan"], 3), ("as", ["Asia"], 4)], _CLASSES)
    assert plan.to_mint == [_IRI + "EastAsia"]
    assert plan.entities_by_class[_IRI + "Asia"] == ["as"]


def test_unrelated_classes_do_not_become_nodes():
    plan = plan_ontology_containment([("jp", ["Japan"], 3)], _CLASSES)
    assert _IRI + "WesternAsia" not in plan.to_mint


def test_a_state_never_matches_a_country_of_the_same_name():
    # The US state must not land in Western Asia.
    plan = plan_ontology_containment([("ga", ["Georgia"], 2)], _CLASSES)
    assert not plan.links and not plan.to_mint


def test_an_alias_matches():
    plan = plan_ontology_containment([("kr", ["Korea", "South Korea"], 3)], _CLASSES)
    assert plan.entities_by_class[_IRI + "SouthKorea"] == ["kr"]


def test_a_chain_stops_at_a_class_the_prune_removed():
    classes = {k: v for k, v in _CLASSES.items() if k != _IRI + "Asia"}
    plan = plan_ontology_containment([("jp", ["Japan"], 3)], classes)
    assert plan.to_mint == [_IRI + "EastAsia"]


def test_a_cycle_ends():
    classes = {"a": {"label": "A", "part_of": ["b"]}, "b": {"label": "B", "part_of": ["a"]}}
    plan = plan_ontology_containment([("x", ["A"], 3)], classes)
    assert set(plan.links) == {("a", "b"), ("b", "a")}


def test_a_new_container_is_typed_with_its_level_class():
    assert _container_type(_CLASSES[_IRI + "Asia"], _CLASSES, _IRI + "Asia") == _IRI + "Continent"
    assert _container_type(_CLASSES[_IRI + "Japan"], _CLASSES, _IRI + "Japan") == _IRI + "Japan"


# --- retrieval: down only, like periods ---

class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows, self.params = rows, None

    async def execute(self, _sql, params):
        self.params = params
        return _Rows(self.rows)


def test_contained_places_score_by_depth_and_exclude_seeds():
    germany, hesse, europe = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _Session([(germany, 1), (hesse, 2), (europe, 1)])
    got = asyncio.run(rsql.expand_place_entities(session, [europe]))
    assert got[germany] == (0.8, 1)
    assert got[hesse] == (pytest.approx(0.64), 2)
    assert europe not in got


def test_no_seeds_no_query():
    session = _Session([])
    assert asyncio.run(rsql.expand_place_entities(session, [])) == {}
    assert session.params is None


def test_expansion_follows_edges_into_the_place_only():
    sql = rsql._PLACE_DESCEND_SQL.text
    assert "gr.target_node_id = d.id" in sql
    assert "gr.source_node_id = d.id" not in sql


def test_the_broad_walk_can_leave_containment_out():
    assert GRAPHRAG_LOCATED_IN in rsql._ENTITY_BFS_NO_CONTAINMENT_SQL.text
    assert GRAPHRAG_LOCATED_IN not in rsql._ENTITY_BFS_SQL.text

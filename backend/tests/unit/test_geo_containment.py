"""The deterministic guard on what `enrich-geo` is allowed to write.

These edges carry no document behind them, so a wrong one is unfalsifiable
from the corpus -- which is exactly why the level arithmetic exists rather
than trusting the model. Both rejections below were produced by gpt-4o-mini on
a real 46-place list.
"""
from __future__ import annotations

from backend.app.services.db_geo_enrich import (
    DEFAULT_GEO_CLASS_LABELS,
    NON_PLACE_CLASS_LABELS,
    containment_is_plausible,
)


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

"""Stage-2 entity-shaped class filter.

Covers `_looks_like_entity_not_class` (the heuristic) and
`_filter_entity_shaped_classes` (the demoter that promotes a class
proposal to MATCH NOT FOUND INSTANCES). The point of the filter is
to keep proper nouns -- company names, country names, named reports
-- out of the ontology's class set when gpt-4.1 occasionally leaks
one through despite the prompt's HARD RULE.
"""

from __future__ import annotations

import pytest

from backend.app.services.pipeline_llm import (
    _build_known_places_from_ontology,
    _compile_extra_suffix_regex,
    _compile_extra_word_regex,
    _filter_entity_shaped_classes,
    _looks_like_entity_not_class,
    _split_camel,
)


_DEFAULT_KNOWN_PLACES = frozenset({"myanmar", "vietnam", "asia", "tokyo",
                                    "united states", "hong kong"})


@pytest.mark.parametrize(
    "label,reason_prefix",
    [
        # Corporate suffix patterns
        ("BYD Company Ltd.", "corporate-suffix"),
        ("OCI N.V.", "corporate-suffix"),
        ("Apple Inc", "corporate-suffix"),
        ("Saudi Aramco Total Refinery & Petrochemicals Co.", "corporate-suffix"),
        ("Samsung Electronics Co., Ltd.", "corporate-suffix"),
        ("ThyssenKrupp AG", "corporate-suffix"),
        # Known proper-noun places (sourced from the dynamic set)
        ("Myanmar", "known-place"),
        ("Vietnam", "known-place"),
        ("Asia", "known-place"),
        ("Tokyo", "known-place"),
        ("United States", "known-place"),
        ("Hong Kong", "known-place"),
        # Document/report titles (whitespace-separated)
        ("Sovereign Risk Tracker", "document-title"),
        ("Vietnam's Manufacturing & Supply Chain Industry Report", "document-title"),
        ("Fertilizer Market Dashboard", "document-title"),
        ("Energy Outlook", "document-title"),
        # CamelCase document/tool titles (v2 catches these because the
        # CamelCase splitter exposes the tail word).
        ("FertilizerMarketDashboard", "document-title"),
        ("SovereignRiskTracker", "document-title"),
        ("EarlyWarningHub", "document-title"),
        ("GlobalDebtDatabase", "document-title"),
        ("FoodSecurityPortal", "document-title"),
        ("FuelShortageTracker", "document-title"),
        ("MonetaryPolicyTracker", "document-title"),
        ("CompanyDatabase", "document-title"),
        ("ReleaseCalendar", "document-title"),
        ("ConsumerPriceIndex", "document-title"),
        ("InflationDataSource", "document-title"),
        ("InvestorDirectory", "document-title"),
        ("AgricultureNewsPlatform", "document-title"),
        ("PublicOpinionSurvey", "document-title"),
        # Year-prefixed labels with document-title tail words are caught
        # by the document-title heuristic first (more specific reason).
        ("2025 Factbook", "document-title"),
        ("Q1 2024 Outlook", "document-title"),
        # Year-prefix-only fallback (no document-title tail word).
        ("2025 Strategic Review Conference", "year-prefix"),
        ("Q3 2024 Annual Summary", "year-prefix"),
    ],
)
def test_entity_shaped_labels_are_flagged(label: str, reason_prefix: str) -> None:
    is_entity, reason = _looks_like_entity_not_class(
        label, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is True, f"expected {label!r} to be flagged as entity"
    assert reason == reason_prefix, (
        f"expected reason {reason_prefix!r} for {label!r}, got {reason!r}"
    )


@pytest.mark.parametrize(
    "label",
    [
        # Abstract category names -- these are LEGITIMATE class proposals
        "CarManufacturer",
        "FertilizerProducer",
        "TradeAgreement",
        "SoutheastAsianCountry",
        "ChemicalElement",
        "Regulation",
        "SupplyChainRisk",
        "Person",                    # a class, not a person's name
        "Organization",
        "Country",
        # Bare tail-words alone are legitimate categories -- DO NOT flag.
        # ("Forecast" by itself = a kind of prediction; "Alert" by itself
        # = a category of warning.)
        "Forecast",
        "Alert",
        "Dashboard",
        "Tracker",
        "Monitor",
        "Report",
        # Edge cases that should pass
        "",
        "   ",
    ],
)
def test_category_labels_are_not_flagged(label: str) -> None:
    is_entity, reason = _looks_like_entity_not_class(
        label, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is False, (
        f"expected {label!r} to be kept as a class (got reason {reason!r})"
    )


def test_filter_promotes_entity_shaped_to_instances() -> None:
    stage2_result = {
        "MATCHES FOUND": [{"IRI": "ex:Country", "TEXT_SNIPPET": "Vietnam"}],
        "MATCH NOT FOUND": [
            {
                "LABEL": "BYD Company Ltd.",
                "DESCRIPTION": "A Chinese EV manufacturer.",
                "PARENT_LABEL": "Organization",
            },
            {
                "LABEL": "CarManufacturer",
                "DESCRIPTION": "A company that makes cars.",
                "PARENT_LABEL": "Organization",
            },
            {
                "LABEL": "Myanmar",
                "DESCRIPTION": "A Southeast Asian country.",
                "PARENT_LABEL": "Country",
            },
        ],
        "MATCH NOT FOUND INSTANCES": [
            {
                "LABEL": "Jan 2024",
                "CANONICAL_FORM": "January 2024",
                "TYPE_LABEL": "Month",
                "DESCRIPTION": "",
            }
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(
        stage2_result, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert updated is not None
    # Only the legitimate category survives in MATCH NOT FOUND
    surviving_labels = [c["LABEL"] for c in updated["MATCH NOT FOUND"]]
    assert surviving_labels == ["CarManufacturer"]
    # Both entity-shaped proposals were promoted to INSTANCES, preserving
    # the original temporal instance that was already there.
    instance_labels = [i["LABEL"] for i in updated["MATCH NOT FOUND INSTANCES"]]
    assert "Jan 2024" in instance_labels
    assert "BYD Company Ltd." in instance_labels
    assert "Myanmar" in instance_labels
    # Demotion records expose the reason for each move.
    demoted_labels = {d["label"]: d["reason"] for d in demotions}
    assert demoted_labels == {
        "BYD Company Ltd.": "corporate-suffix",
        "Myanmar": "known-place",
    }
    # Promoted instances inherit the PARENT_LABEL as their TYPE_LABEL.
    byd_instance = next(
        i for i in updated["MATCH NOT FOUND INSTANCES"] if i["LABEL"] == "BYD Company Ltd."
    )
    assert byd_instance["TYPE_LABEL"] == "Organization"
    assert byd_instance["CANONICAL_FORM"] == "BYD Company Ltd."
    assert byd_instance["DESCRIPTION"] == "A Chinese EV manufacturer."


def test_filter_is_safe_on_missing_keys() -> None:
    # No MATCH NOT FOUND at all -- returns the same dict, no demotions.
    result = {"MATCHES FOUND": []}
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated == {"MATCHES FOUND": []}
    assert demotions == []


def test_filter_is_safe_on_none() -> None:
    updated, demotions = _filter_entity_shaped_classes(None)
    assert updated is None
    assert demotions == []


def test_filter_is_idempotent_when_no_entities_to_promote() -> None:
    # All entries are legitimate categories -- the result is unchanged
    # and no demotions are produced.
    result = {
        "MATCH NOT FOUND": [
            {"LABEL": "TradeAgreement", "DESCRIPTION": "", "PARENT_LABEL": "NONE"},
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated["MATCH NOT FOUND"] == [
        {"LABEL": "TradeAgreement", "DESCRIPTION": "", "PARENT_LABEL": "NONE"}
    ]
    assert demotions == []
    # MATCH NOT FOUND INSTANCES was not touched (and was absent).
    assert "MATCH NOT FOUND INSTANCES" not in updated


def test_filter_preserves_non_dict_entries() -> None:
    # Stray malformed entries (strings, None) pass through untouched.
    result = {
        "MATCH NOT FOUND": [
            "stray-string",
            None,
            {"LABEL": "Apple Inc", "DESCRIPTION": "", "PARENT_LABEL": "Organization"},
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated["MATCH NOT FOUND"] == ["stray-string", None]
    assert len(demotions) == 1
    assert demotions[0]["label"] == "Apple Inc"


# ---------------------------------------------------------------------------
# v2 helpers: CamelCase split, dynamic known-places, config extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("FertilizerMarketDashboard", "Fertilizer Market Dashboard"),
        ("EarlyWarningHub", "Early Warning Hub"),
        ("GlobalDebtDatabase", "Global Debt Database"),
        ("WEO", "WEO"),                 # all-caps acronym preserved
        ("EUTrade", "EU Trade"),        # acronym + CamelCase
        ("Fertilizer Market Dashboard", "Fertilizer Market Dashboard"),  # already spaced
        ("2025Factbook", "2025 Factbook"),
        ("", ""),
        ("Person", "Person"),           # single-word category
    ],
)
def test_split_camel(label: str, expected: str) -> None:
    assert _split_camel(label) == expected


def _make_ontology_fixture() -> dict:
    """Tiny ontology mimicking the geography subtree pattern. Three layers:
    Continent / Region / Country at the top; specific continents (Asia),
    regions (SoutheastAsia), countries (Vietnam) below."""
    return {
        "classes_dict": {
            "ex:Continent": {"labels": ["Continent"], "superclasses": ["ex:Place"]},
            "ex:Region":    {"labels": ["Region"],    "superclasses": ["ex:Place"]},
            "ex:Country":   {"labels": ["Country"],   "superclasses": ["ex:Place"]},
            "ex:Place":     {"labels": ["Place"],     "superclasses": []},
            "ex:Asia":      {"labels": ["Asia"],      "superclasses": ["ex:Continent"]},
            "ex:Africa":    {"labels": ["Africa"],    "superclasses": ["ex:Continent"]},
            "ex:SoutheastAsia": {"labels": ["Southeast Asia"], "superclasses": ["ex:Region", "ex:Asia"]},
            "ex:Vietnam":   {"labels": ["Vietnam"],   "superclasses": ["ex:Country", "ex:SoutheastAsia"]},
            "ex:Myanmar":   {"labels": ["Myanmar"],   "superclasses": ["ex:Country"]},
            # Not a place
            "ex:CarManufacturer": {"labels": ["Car Manufacturer"], "superclasses": ["ex:Organization"]},
            "ex:Organization":    {"labels": ["Organization"],     "superclasses": []},
        }
    }


def test_build_known_places_from_ontology_discovers_geography_descendants() -> None:
    places = _build_known_places_from_ontology(_make_ontology_fixture())
    # All ancestors-of-Place classes contribute their labels (Place itself
    # is NOT a place-kind label, but Continent / Region / Country are).
    assert "continent" in places
    assert "country" in places
    assert "region" in places
    assert "asia" in places
    assert "vietnam" in places
    assert "myanmar" in places
    assert "southeast asia" in places
    # Non-geography classes are not included.
    assert "car manufacturer" not in places
    assert "organization" not in places


def test_build_known_places_returns_empty_when_no_geography_present() -> None:
    minimal = {"classes_dict": {"ex:Foo": {"labels": ["Foo"], "superclasses": []}}}
    places = _build_known_places_from_ontology(minimal)
    assert places == frozenset()


def test_build_known_places_unions_extras() -> None:
    minimal = {"classes_dict": {"ex:Foo": {"labels": ["Foo"], "superclasses": []}}}
    places = _build_known_places_from_ontology(
        minimal, extra_labels=["Atlantis", "Wakanda"]
    )
    assert places == frozenset({"atlantis", "wakanda"})


def test_build_known_places_safe_on_none() -> None:
    assert _build_known_places_from_ontology(None) == frozenset()
    assert _build_known_places_from_ontology({}) == frozenset()
    assert _build_known_places_from_ontology({"classes_dict": None}) == frozenset()


def test_known_places_default_skips_place_check() -> None:
    # When known_places is None or empty, "Myanmar" passes the heuristic
    # (the place check is the only branch that would catch it; corporate
    # suffix + doc tail + year prefix do not).
    is_entity, _ = _looks_like_entity_not_class("Myanmar", known_places=None)
    assert is_entity is False
    is_entity, _ = _looks_like_entity_not_class("Myanmar", known_places=frozenset())
    assert is_entity is False


def test_extra_corporate_suffix_regex_extends_builtins() -> None:
    extra = _compile_extra_suffix_regex(["KK", "OAO"])
    # A label using ONLY the extra suffix is now caught.
    is_entity, reason = _looks_like_entity_not_class(
        "Toyota KK", extra_corporate_suffix_re=extra,
    )
    assert is_entity is True
    assert reason == "corporate-suffix"
    # And built-ins still fire too.
    is_entity, reason = _looks_like_entity_not_class(
        "Toyota Inc", extra_corporate_suffix_re=extra,
    )
    assert is_entity is True
    assert reason == "corporate-suffix"


def test_extra_tail_word_regex_extends_builtins() -> None:
    extra = _compile_extra_word_regex(["Compendium", "Almanac"])
    # CamelCase + extra tail word now caught.
    is_entity, reason = _looks_like_entity_not_class(
        "FoodSecurityCompendium", extra_tail_word_re=extra,
    )
    assert is_entity is True
    assert reason == "document-title"
    # Built-in tail words still work.
    is_entity, _ = _looks_like_entity_not_class(
        "SovereignRiskTracker", extra_tail_word_re=extra,
    )
    assert is_entity is True


def test_compile_extras_return_none_on_empty_or_garbage() -> None:
    assert _compile_extra_suffix_regex(None) is None
    assert _compile_extra_suffix_regex([]) is None
    assert _compile_extra_suffix_regex([""]) is None
    assert _compile_extra_word_regex(None) is None
    assert _compile_extra_word_regex(["", "   "]) is None

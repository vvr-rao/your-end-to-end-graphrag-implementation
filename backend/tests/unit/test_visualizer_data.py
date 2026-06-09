"""Smoke tests for the visualizer's data layer.

Covers:
- discover_owl_files() finds known fixtures under source_ontologies/.
- load_ontology() loads skos.rdf into the canonical 4-dict shape with
  per-call isolated owlready2.World (no shared-state bleed across calls).
- is_too_large() flags DRON when present without trying to parse it.
- resolve_custom_path() rejects garbage but accepts a valid fixture path.
"""

from __future__ import annotations

from pathlib import Path

from visualizer.data import (
    OUTPUT_ROOT,
    SOURCE_ROOT,
    compute_stats,
    discover_owl_files,
    is_too_large,
    load_ontology,
    resolve_custom_path,
)


def test_discover_owl_files_finds_known_source_fixtures() -> None:
    files = discover_owl_files()
    labels = [f.label for f in files]
    # skos.rdf and index.rdf are tracked fixtures under source_ontologies/general_ontologies/.
    assert any(lbl.endswith("skos.rdf") for lbl in labels), labels
    assert any(lbl.endswith("index.rdf") for lbl in labels), labels
    # Every Generated entry must point at a path under output_ontologies/.
    for f in files:
        if f.group == "Generated":
            assert OUTPUT_ROOT in f.path.parents
        elif f.group == "Source Ontologies":
            assert SOURCE_ROOT in f.path.parents


def test_load_ontology_skos_has_expected_shape() -> None:
    skos = SOURCE_ROOT / "core_ontologies" / "skos.rdf"
    assert skos.exists()
    loaded = load_ontology(skos)
    assert loaded is not None
    for key in (
        "classes_dict",
        "object_properties_dict",
        "data_properties_dict",
        "instances_dict",
    ):
        assert key in loaded, key
    stats = compute_stats(loaded)
    # skos.rdf is small but non-empty; we don't pin exact counts (owlready2
    # version drift can shift them by one), only that load yields > 0 classes
    # AND > 0 object properties.
    assert stats["classes"] > 0
    assert stats["object_properties"] > 0


def test_load_ontology_isolates_across_calls() -> None:
    """Two separate loads must not share entities (the per-file World is the
    same isolation property tested by test_world_isolation, but exercised
    through the visualizer's cached entry point)."""
    skos = SOURCE_ROOT / "core_ontologies" / "skos.rdf"
    index = SOURCE_ROOT / "general_ontologies" / "index.rdf"
    assert skos.exists() and index.exists()
    a = load_ontology(skos)
    b = load_ontology(index)
    assert a is not None and b is not None
    a_iris = set(a["classes_dict"].keys())
    b_iris = set(b["classes_dict"].keys())
    # Both files declare some classes; classes from one must not bleed into
    # the other's dict.
    assert a_iris, "skos.rdf has no classes -- fixture broken"
    overlap = a_iris & b_iris
    assert not overlap, f"unexpected shared class IRIs: {next(iter(overlap))}"


def test_is_too_large_skips_dron_when_present() -> None:
    dron = SOURCE_ROOT / "pharma_ontologies" / "dron.owl"
    if not dron.exists():
        return  # fixture optional in some checkouts
    assert is_too_large(dron) is True
    # And load_ontology() refuses it without attempting a parse.
    assert load_ontology(dron) is None


def test_resolve_custom_path_validation(tmp_path: Path) -> None:
    bogus = tmp_path / "nope.owl"
    assert resolve_custom_path(str(bogus)) is None  # missing
    assert resolve_custom_path("") is None
    plain_txt = tmp_path / "file.txt"
    plain_txt.write_text("hello")
    assert resolve_custom_path(str(plain_txt)) is None  # wrong suffix
    skos = SOURCE_ROOT / "core_ontologies" / "skos.rdf"
    if skos.exists():
        resolved = resolve_custom_path(str(skos))
        assert resolved == skos.resolve()

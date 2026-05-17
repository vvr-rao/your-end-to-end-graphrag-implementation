"""Zip extraction, import URL sniffing, entity resolution, root finding."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from backend.app.services import ontology_io


@pytest.fixture
def tiny_owl_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two minimal RDF/XML files where alpha.owl imports beta.owl by basename."""
    alpha = tmp_path / "alpha.owl"
    beta = tmp_path / "beta.owl"
    alpha.write_text(
        """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://x/alpha"/>
    <owl:imports rdf:resource="http://x/beta.owl"/>
    <owl:Class rdf:about="http://x/A"/>
</rdf:RDF>"""
    )
    beta.write_text(
        """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://x/beta"/>
    <owl:Class rdf:about="http://x/B"/>
</rdf:RDF>"""
    )
    return alpha, beta


def test_enumerate_inputs_with_dir(tmp_path: Path, tiny_owl_pair) -> None:
    with ontology_io.enumerate_inputs([tmp_path]) as bundle:
        names = sorted(s.path.name for s in bundle.sources)
    assert names == ["alpha.owl", "beta.owl"]


def test_enumerate_inputs_with_zip(tmp_path: Path, tiny_owl_pair) -> None:
    alpha, beta = tiny_owl_pair
    z = tmp_path / "pkg.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(alpha, "alpha.owl")
        zf.write(beta, "sub/beta.owl")
    with ontology_io.enumerate_inputs([z]) as bundle:
        names = sorted(s.path.name for s in bundle.sources)
        labels = [s.label for s in bundle.sources]
    assert names == ["alpha.owl", "beta.owl"]
    assert any("pkg.zip::" in label for label in labels)


def test_resolve_entities_basic() -> None:
    text = """<!ENTITY root "file:/C:/X/">
    <!ENTITY a "&root;y/a.owl">
    <!ENTITY b "&root;z/b.owl">"""
    resolved = ontology_io._resolve_entities(text)
    assert resolved["a"] == "file:/C:/X/y/a.owl"
    assert resolved["b"] == "file:/C:/X/z/b.owl"
    assert "root" in resolved


def test_discover_imports_in_text_with_entities() -> None:
    text = """<!DOCTYPE rdf:RDF [
        <!ENTITY root "file:/C:/X/">
        <!ENTITY foo "&root;foo.owl">
    ]>
    <rdf:RDF>
        <owl:imports rdf:resource="&foo;"/>
    </rdf:RDF>"""
    imports = ontology_io._discover_imports_in_text(text)
    assert "file:/C:/X/foo.owl" in imports


def test_find_root_files_drops_imported(tmp_path: Path) -> None:
    a = tmp_path / "alpha.owl"
    b = tmp_path / "beta.owl"
    # alpha imports beta -> only alpha should be a root
    a.write_text(
        '<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<owl:imports rdf:resource="http://x/beta.owl"/></rdf:RDF>'
    )
    b.write_text('<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>')
    roots = ontology_io._find_root_files([a, b])
    assert a in roots
    assert b not in roots


def test_unsupported_input_raises(tmp_path: Path) -> None:
    f = tmp_path / "not_ontology.json"
    f.write_text("{}")
    with pytest.raises(ValueError):
        with ontology_io.enumerate_inputs([f]):
            pass


def test_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with ontology_io.enumerate_inputs([tmp_path / "does_not_exist.owl"]):
            pass

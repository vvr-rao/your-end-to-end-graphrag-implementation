"""User-reported 3-source merge: OCRe.zip + skos.rdf + hp.owl.

Regression test for the OOM-during-write_owl bug fixed by the streaming
RDF/XML export. Asserts all four canonical output files land AND
merged.owl reloads via rdflib without error.

Marked @pytest.mark.slow because hp.owl is 73 MB / 32 K classes -- this
takes ~75-90s on the dev machine and skipped by default in fast CI.
Runs on `pytest -m slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rdflib

from backend.app.services.pipeline import run_merge

REPO_ROOT = Path(__file__).resolve().parents[3]
OCRE_ZIP = REPO_ROOT / "source_ontologies" / "pharma_ontologies" / "OCRe.zip"
SKOS_RDF = REPO_ROOT / "source_ontologies" / "general_ontologies" / "skos.rdf"
HP_OWL = REPO_ROOT / "source_ontologies" / "pharma_ontologies" / "hp.owl"


@pytest.mark.slow
@pytest.mark.skipif(
    not (OCRE_ZIP.exists() and SKOS_RDF.exists() and HP_OWL.exists()),
    reason="one or more of OCRe.zip / skos.rdf / hp.owl missing",
)
def test_merge_ocre_skos_hp(tmp_path: Path) -> None:
    """Exact 3-source merge command the user hit. Must produce all four
    canonical output files (merged.json, merged.owl, manifest.json,
    stats.json) and the merged.owl must reload via rdflib."""
    out_root = tmp_path / "out"
    out_root.mkdir()
    version_dir = run_merge(
        input_ontologies=[OCRE_ZIP, SKOS_RDF, HP_OWL],
        output_root=out_root,
    )

    # All four canonical output files exist and are non-empty.
    for name in ("merged.json", "merged.owl", "manifest.json", "stats.json"):
        f = version_dir / name
        assert f.exists() and f.stat().st_size > 0, f"missing or empty: {name}"

    # merged.owl reloads via rdflib.
    g = rdflib.Graph()
    g.parse(str(version_dir / "merged.owl"), format="xml")
    assert len(g) > 0

    # Stats are sensible: HP alone contributes ~32 K classes; the total
    # should be at least dominated by HP.
    stats = json.loads((version_dir / "stats.json").read_text())
    counts = stats["counts"]
    assert counts["classes"] > 30_000, counts
    # OCRe contributes ~168 object properties; the total should reflect.
    assert counts["object_properties"] > 100, counts

    # Manifest records all three inputs.
    manifest = json.loads((version_dir / "manifest.json").read_text())
    input_names = {i["name"] for i in manifest["input_ontologies"]}
    assert {"OCRe.zip", "skos.rdf", "hp.owl"} <= input_names

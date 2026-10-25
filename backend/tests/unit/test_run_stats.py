"""graph-stats: ontology-folder counts and the per-database delta."""
from __future__ import annotations

import json

from backend.app.services import run_stats as rs

DC = "https://veerla-ramrao.ai/ontology/domain-concepts#"


def _cls(iri: str, parent: str | None) -> dict:
    return {"iri": iri, "superclasses": [{"iri": parent}] if parent else []}


def test_ontology_stats_counts_created_and_concept_classes(tmp_path) -> None:
    (tmp_path / "stats.json").write_text(json.dumps({
        "before": {"classes": 3, "instances": 10},
        "after": {"classes": 6, "instances": 14},
        "created_classes": [DC + "revenue", DC + "relatedpartyrevenue", DC + "acme"],
    }))
    classes = {
        DC + "Measure": _cls(DC + "Measure", None),
        DC + "Revenue": _cls(DC + "Revenue", DC + "Measure"),
        DC + "RelatedPartyRevenue": _cls(DC + "RelatedPartyRevenue", DC + "Revenue"),
        DC + "Acme": _cls(DC + "Acme", "http://example.org/Organization"),
    }
    (tmp_path / "merged.json").write_text(json.dumps({"classes_dict": classes}))
    s = rs.ontology_stats(tmp_path, ["Measure"])
    assert s["classes_created"] == 3
    assert s["instances_created"] == 4
    assert s["concept_classes"] == 3            # the root counts, as in the DB closure
    assert s["concept_classes_created"] == 2    # Acme is not under a concept root


def test_delta_is_none_first_then_the_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rs, "SNAPSHOT_DIR", tmp_path)
    first = {"entities": 10, "entity_entity_edges": 4}
    assert rs.diff_against_snapshot("db", first) is None
    assert rs.diff_against_snapshot("db", {"entities": 15, "entity_entity_edges": 4}) == {
        "entities": 5, "entity_entity_edges": 0}

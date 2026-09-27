"""Every ontology in `core_ontologies/` must actually be merged.

`geography_features_extension.owl` sat in that directory unmerged for months.
Its absence caused three distinct downstream failures, none of which pointed at
the merge step:

  1. `enrich-geo` rejected `California (Region) in United States (Country)` --
     44% of its rejections on the 42-doc run. The base geography ontology has no
     sub-national class, so US states typed as `Region`, which the containment
     level table ranks ABOVE `Country`.
  2. A utility 10-K typed California, Iowa, Nevada, Oregon, Utah, Wyoming and
     five more US states as `City` -- `City` was on the candidate menu and its
     sibling `AdministrativeArea` did not exist.
  3. `AdministrativeArea` and `Landform` were pinned in
     `extraction.pinned_class_labels`, pinning labels with nothing behind them.

A file present but unpinned is invisible: nothing errors, the merge just has
fewer classes, and the symptoms surface three pipeline stages later.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
CORE_DIR = ROOT / "source_ontologies" / "core_ontologies"
ONTOLOGY_SUFFIXES = {".owl", ".rdf", ".ttl", ".xml"}


def _pinned() -> set[str]:
    src = (ROOT / "scripts" / "merge_ontology.py").read_text()
    return set(re.findall(r'"source_ontologies/core_ontologies/([^"]+)"', src))


def test_every_core_ontology_on_disk_is_pinned():
    on_disk = {
        f.name for f in CORE_DIR.iterdir()
        if f.suffix.lower() in ONTOLOGY_SUFFIXES
    }
    unmerged = on_disk - _pinned()
    assert not unmerged, (
        f"these ontologies sit in core_ontologies/ but are never merged: "
        f"{sorted(unmerged)}. Add them to CORE in scripts/merge_ontology.py, or "
        f"move them out of core_ontologies/ if they are deliberately optional."
    )


def test_no_pinned_ontology_is_missing_from_disk():
    """The inverse: a pin pointing at nothing would fail the merge at runtime."""
    missing = {n for n in _pinned() if not (CORE_DIR / n).exists()}
    assert not missing, f"CORE pins files that do not exist: {sorted(missing)}"


def test_the_geography_extension_supplies_the_sub_national_class():
    """`AdministrativeArea` is what makes geographic containment work.

    Without it the only place classes are City / Country / Region / Continent,
    and `Region` outranks `Country` in the containment level table -- so no
    sub-national place can sit inside its own country.
    """
    text = (CORE_DIR / "geography_features_extension.owl").read_text()
    assert "AdministrativeArea" in text

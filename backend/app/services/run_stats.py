"""Build-step counts reported after every run (`graph-stats`).

Two sources:

* the database -- entities, entity->entity edges, concept classes (the
  closure under `extraction.concept_class_roots`), concept entities (entities
  typed to those classes), ontology classes and instances;
* a prune-expand version folder -- classes and instances before/after/created
  (stats.json) and the created classes that sit under a concept root
  (merged.json superclass chains).

Each DB report is saved per database under `.build/graph_stats/` so the next
report prints what the run in between CREATED (the delta), not only totals.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.services.retrieval_sql import _CONCEPT_CLASS_IDS_SQL

SNAPSHOT_DIR = Path(".build/graph_stats")

_GRAPH_SQL = text("""
WITH concept_cls AS (""" + _CONCEPT_CLASS_IDS_SQL.text + """),
concept_ent AS (
    SELECT e.id FROM graphrag.entities e
     WHERE e.status = 'ACTIVE' AND e.class_id IN (SELECT id FROM concept_cls)
)
SELECT
  (SELECT count(*) FROM graphrag.ontology_classes)                        AS ontology_classes,
  (SELECT count(*) FROM graphrag.ontology_instances)                      AS ontology_instances,
  (SELECT count(*) FROM concept_cls)                                      AS concept_classes,
  (SELECT count(*) FROM graphrag.entities WHERE status = 'ACTIVE')        AS entities,
  (SELECT count(*) FROM concept_ent)                                      AS concept_entities,
  (SELECT count(*) FROM graphrag.graph_relationships
    WHERE source_node_type = 'entity' AND target_node_type = 'entity')    AS entity_entity_edges,
  (SELECT count(*) FROM graphrag.graph_relationships
    WHERE source_node_type = 'entity' AND target_node_type = 'entity'
      AND (source_node_id IN (SELECT id FROM concept_ent)
           OR target_node_id IN (SELECT id FROM concept_ent)))            AS concept_entity_edges
""")

GRAPH_KEYS = (
    ("entities", "entities"),
    ("entity_entity_edges", "entity->entity edges"),
    ("concept_entity_edges", "  of which touch a concept"),
    ("concept_classes", "concept classes"),
    ("concept_entities", "concept entities"),
    ("ontology_classes", "ontology classes"),
    ("ontology_instances", "ontology instances"),
)


async def graph_stats(session, concept_roots: list[str]) -> dict[str, int]:
    labels = [r.strip().lower() for r in concept_roots if r and r.strip()]
    row = (await session.execute(_GRAPH_SQL, {"labels": labels})).mappings().one()
    return {k: int(v) for k, v in row.items()}


def _snapshot_path(db_name: str) -> Path:
    return SNAPSHOT_DIR / (re.sub(r"[^A-Za-z0-9_.-]+", "_", db_name) + ".json")


def diff_against_snapshot(db_name: str, now: dict[str, int], *, save: bool = True
                          ) -> dict[str, int] | None:
    """Delta vs the previous report for this database (None on the first)."""
    path = _snapshot_path(db_name)
    prev = None
    try:
        prev = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=1))
    if not isinstance(prev, dict):
        return None
    return {k: now[k] - int(prev.get(k, 0)) for k in now}


def format_graph_stats(now: dict[str, int], delta: dict[str, int] | None) -> str:
    lines = []
    for key, label in GRAPH_KEYS:
        d = "" if delta is None else f"   ({delta.get(key, 0):+d} since last report)"
        lines.append(f"  {label:<28}: {now[key]:>8}{d}")
    return "\n".join(lines)


def _local(iri: str) -> str:
    return re.split(r"[#/]", iri)[-1].lower()


def ontology_stats(version_dir: Path, concept_roots: list[str]) -> dict[str, Any]:
    """Counts for one prune-expand/build version folder."""
    stats = json.loads((version_dir / "stats.json").read_text())
    before, after = stats.get("before") or {}, stats.get("after") or {}
    created = list(stats.get("created_classes") or [])
    roots = {r.strip().lower() for r in concept_roots if r and r.strip()}
    classes: dict[str, Any] = {}
    try:
        classes = json.loads((version_dir / "merged.json").read_text()).get("classes_dict") or {}
    except (OSError, ValueError):
        pass
    parents = {iri: [s.get("iri") for s in (c.get("superclasses") or [])
                     if isinstance(s, dict) and s.get("iri")]
               for iri, c in classes.items()}

    def under_root(iri: str) -> bool:
        seen, todo = set(), [iri]
        while todo:
            cur = todo.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if _local(cur) in roots:   # a root counts, like the DB closure
                return True
            todo.extend(parents.get(cur) or [])
        return False

    concept_all = [i for i in classes if under_root(i)] if roots else []
    created_lc = {i.lower() for i in created}
    return {
        "classes_before": int(before.get("classes", 0)),
        "classes_after": int(after.get("classes", 0)),
        "classes_created": len(created),
        "instances_before": int(before.get("instances", 0)),
        "instances_after": int(after.get("instances", 0)),
        "instances_created": int(after.get("instances", 0)) - int(before.get("instances", 0)),
        "concept_classes": len(concept_all),
        "concept_classes_created": sum(1 for i in concept_all if i.lower() in created_lc),
    }


def format_ontology_stats(s: dict[str, Any]) -> str:
    return "\n".join([
        f"  ontology classes            : {s['classes_after']:>8}   "
        f"(before {s['classes_before']}, created {s['classes_created']})",
        f"  ontology instances          : {s['instances_after']:>8}   "
        f"(before {s['instances_before']}, net {s['instances_created']:+d})",
        f"  concept classes             : {s['concept_classes']:>8}   "
        f"(created {s['concept_classes_created']})",
    ])

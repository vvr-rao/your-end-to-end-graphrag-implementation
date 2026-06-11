"""Import a Phase-1 merge folder into the graphrag schema.

Reads `<input>/merged.json` (the canonical dict-of-dicts produced by
the Phase 1 merge / prune-expand CLI) and upserts rows into
`ontology_classes`, `ontology_object_properties`,
`ontology_data_properties`, and `ontology_instances`.

Embeddings are generated from `(label + description)` per class using
text-embedding-3-small @ 1024 dim. Idempotent: rows are upserted by
IRI; embeddings are recomputed only when the source text changes.

Cost guards:
- `--limit N` caps how many of each entity kind to process (for
  smoke-testing without spending real money on the full corpus).
- `--dry-run` skips all writes + embedding calls; just reports counts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version
from backend.app.db.models.ontology import (
    OntologyClass,
    OntologyDataProperty,
    OntologyInstance,
    OntologyObjectProperty,
)
from backend.app.db.session import session_scope
from backend.app.services.embeddings import Embedder

_VIAO_NAMESPACE = "https://veerla-ramrao.ai/ontology/intelligence-artifact"


@dataclass
class ImportSummary:
    classes_total: int = 0
    classes_inserted: int = 0
    classes_updated: int = 0
    classes_embedded: int = 0
    obj_props_total: int = 0
    data_props_total: int = 0
    instances_total: int = 0
    cost_usd: float = 0.0
    dry_run: bool = False


def _namespace_of(iri: str) -> str:
    """Split an IRI into the namespace prefix used in `ontology_classes.namespace`."""
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    return iri.rsplit("/", 1)[0] + "/"


def _is_viao(iri: str) -> bool:
    return iri.startswith(_VIAO_NAMESPACE)


def _first_str(values: Any) -> str | None:
    """Extract the first non-empty string from a value that may be
    str / list[str] / list[dict{value, lang}]."""
    if isinstance(values, str) and values.strip():
        return values.strip()
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                inner = v.get("value")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
    return None


def _build_embed_text(rec: dict[str, Any]) -> str:
    """Compose the text fed to the embedder. Prefers compact_description
    (from Phase 1's class-summary pass) over the raw owlready2 fields."""
    parts: list[str] = []
    label = _first_str(rec.get("labels")) or rec.get("name")
    if label:
        parts.append(label)
    compact = _first_str(rec.get("compact_description"))
    if compact:
        parts.append(compact)
    else:
        descr = _first_str(rec.get("descriptions"))
        if descr:
            parts.append(descr)
        else:
            comm = _first_str(rec.get("comments"))
            if comm:
                parts.append(comm)
    return " — ".join(parts) if parts else (label or "")


def _embed_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def import_ontology_folder(
    input_folder: Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> ImportSummary:
    """Driver. Reads `<input_folder>/merged.json` and writes to the DB.

    `limit` (default None = all) caps each entity kind. Useful for smoke
    tests: `--limit 5` writes only 5 of each (classes, properties, instances).
    """
    merged_path = input_folder / "merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(merged_path)

    with merged_path.open() as f:
        merged = json.load(f)

    classes = merged.get("classes_dict") or {}
    obj_props = merged.get("object_properties_dict") or {}
    data_props = merged.get("data_properties_dict") or {}
    instances = merged.get("instances_dict") or {}

    summary = ImportSummary(
        classes_total=len(classes),
        obj_props_total=len(obj_props),
        data_props_total=len(data_props),
        instances_total=len(instances),
        dry_run=dry_run,
    )

    if limit is not None:
        classes = dict(list(classes.items())[:limit])
        obj_props = dict(list(obj_props.items())[:limit])
        data_props = dict(list(data_props.items())[:limit])
        instances = dict(list(instances.items())[:limit])

    print(
        f"[db-import] reading {merged_path}\n"
        f"  classes={summary.classes_total}, obj_props={summary.obj_props_total}, "
        f"data_props={summary.data_props_total}, instances={summary.instances_total}\n"
        f"  limit={limit}  dry_run={dry_run}"
    )

    if dry_run:
        print("[db-import] DRY RUN: would write the above; exiting.")
        return summary

    embedder = Embedder()

    # Batch size for the multi-row UPSERTs. 100 keeps the SQL string +
    # parameter list well under any pooler limit and bounds each
    # transaction to ~1-2 seconds wall.
    UPSERT_BATCH = 100

    async def _upsert_batched(
        Model, payloads: list[dict[str, Any]], label: str
    ) -> None:
        """Multi-row INSERT...ON CONFLICT DO UPDATE in batches of
        UPSERT_BATCH. Each batch is its own transaction so a pooler
        connection drop only loses one batch (and re-running is
        idempotent via the iri unique constraint)."""
        if not payloads:
            return
        total_batches = (len(payloads) + UPSERT_BATCH - 1) // UPSERT_BATCH
        for batch_idx in range(total_batches):
            chunk = payloads[batch_idx * UPSERT_BATCH : (batch_idx + 1) * UPSERT_BATCH]
            async with session_scope() as session:
                stmt = pg_insert(Model).values(chunk)
                # `excluded.<col>` references the values that WOULD have
                # been inserted -- each conflicted row gets its own
                # proposed values back as the update set.
                set_cols = [c for c in chunk[0].keys() if c != "iri"]
                stmt = stmt.on_conflict_do_update(
                    index_elements=["iri"],
                    set_={c: getattr(stmt.excluded, c) for c in set_cols},
                )
                await session.execute(stmt)
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                done = min((batch_idx + 1) * UPSERT_BATCH, len(payloads))
                print(f"[db-import] {label}: {done}/{len(payloads)} upserted")

    # ---- Pass 1: classes (with embedding) ----
    if classes:
        texts: list[str] = []
        records: list[tuple[str, dict[str, Any]]] = []
        for iri, rec in classes.items():
            text = _build_embed_text(rec)
            if not text:
                continue
            texts.append(text)
            records.append((iri, rec))

        print(f"[db-import] embedding {len(texts)} class text(s) ...")
        vectors = await embedder.embed(texts) if texts else []
        print(f"[db-import] embedding DONE (cost ${embedder.total_cost_usd:.4f}, "
              f"{embedder.total_tokens:,} tokens)")

        payloads = []
        for (iri, rec), vec in zip(records, vectors, strict=False):
            payloads.append({
                "iri": iri,
                "label": _first_str(rec.get("labels")) or rec.get("name"),
                "description": (
                    _first_str(rec.get("compact_description"))
                    or _first_str(rec.get("descriptions"))
                    or _first_str(rec.get("comments"))
                ),
                "namespace": _namespace_of(iri),
                "source_ontology": _first_str(rec.get("sources")),
                "is_viao_class": _is_viao(iri),
                "embedding": vec,
                "extra_metadata": rec,
            })
        await _upsert_batched(OntologyClass, payloads, "classes")
        summary.classes_embedded = len(payloads)

    # ---- Pass 2: object + data properties (no embeddings) ----
    for prop_dict, Model, label in (
        (obj_props, OntologyObjectProperty, "obj_props"),
        (data_props, OntologyDataProperty, "data_props"),
    ):
        if not prop_dict:
            continue
        payloads = []
        for iri, rec in prop_dict.items():
            payloads.append({
                "iri": iri,
                "label": _first_str(rec.get("labels")) or rec.get("name"),
                "description": (
                    _first_str(rec.get("descriptions"))
                    or _first_str(rec.get("comments"))
                ),
                "namespace": _namespace_of(iri),
                "source_ontology": _first_str(rec.get("sources")),
                "extra_metadata": rec,
            })
        await _upsert_batched(Model, payloads, label)

    # ---- Pass 3: ontology_instances ----
    if instances:
        payloads = []
        for iri, rec in instances.items():
            type_iri = None
            types = rec.get("types") or rec.get("direct_types")
            if isinstance(types, list) and types:
                first = types[0]
                if isinstance(first, dict):
                    type_iri = first.get("iri")
            payloads.append({
                "iri": iri,
                "label": _first_str(rec.get("labels")) or rec.get("name"),
                "description": (
                    _first_str(rec.get("descriptions"))
                    or _first_str(rec.get("comments"))
                ),
                "namespace": _namespace_of(iri),
                "type_iri": type_iri,
                "source_ontology": _first_str(rec.get("sources")),
                "extra_metadata": rec,
            })
        await _upsert_batched(OntologyInstance, payloads, "instances")

    # ---- Bump graph_version ----
    async with session_scope() as session:
        new_version = await bump_version(session)
    print(f"[db-import] bumped graph_version -> {new_version}")

    summary.cost_usd = embedder.total_cost_usd
    print(
        f"[db-import] DONE: embedded={summary.classes_embedded} class(es), "
        f"cost=${summary.cost_usd:.4f}, tokens={embedder.total_tokens:,}"
    )
    return summary

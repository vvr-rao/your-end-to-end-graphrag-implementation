"""Pipeline orchestration — composes the I/O, merge, LLM, and export stages.

Each public entry point corresponds to one CLI subcommand. They all share
the same return contract: produce a new version folder under output_root
and return its Path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services import folder_io, ontology_export, ontology_io, ontology_merge, versioning


def run_merge(
    *,
    input_ontologies: list[Path],
    output_root: Path,
) -> Path:
    """`merge` subcommand: deterministic multi-ontology consolidation. Zero LLM."""
    output_root.mkdir(parents=True, exist_ok=True)
    version_dir = versioning.new_version_dir(output_root, "merge")

    print(f"[merge] expanding inputs: {[str(p) for p in input_ontologies]}")
    with ontology_io.enumerate_inputs(input_ontologies) as bundle:
        sources = bundle.sources
        print(f"[merge] found {len(sources)} ontology file(s)")
        loaded = ontology_merge.merge_sources(sources)

    counts = folder_io.count_entities(loaded)
    print(
        f"[merge] merged dict: {counts['classes']} classes, "
        f"{counts['object_properties']} object properties, "
        f"{counts['data_properties']} data properties, "
        f"{counts['instances']} instances"
    )

    json_path = folder_io.write_merged_json(version_dir, loaded)
    print(f"[merge] wrote {json_path.name} ({json_path.stat().st_size:,} bytes)")
    # `consume_dict=True`: drop in-memory dict entries as they're emitted to
    # the OWL. After write_merged_json the dict has already been persisted,
    # nothing further needs it, and on HP-scale merges the freed memory is
    # the difference between OOM and success.
    owl_path = ontology_export.write_owl(
        loaded, version_dir / folder_io.MERGED_OWL, consume_dict=True
    )
    print(f"[merge] wrote {owl_path.name} ({owl_path.stat().st_size:,} bytes)")

    versioning.write_manifest(
        version_dir,
        operation="merge",
        input_ontologies=list(input_ontologies),
    )
    versioning.write_stats(version_dir, {"counts": counts})
    versioning.ensure_audit_log(version_dir)
    return version_dir


# ---- LLM-using stages: implemented in a follow-up step in this same module.
# Stubs here so the CLI surface stays complete from day one.


async def run_prune(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import prune_only_async

    return await prune_only_async(**kwargs)


async def run_expand(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import expand_only_async

    return await expand_only_async(**kwargs)


async def run_prune_and_expand(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import prune_and_expand_async

    return await prune_and_expand_async(**kwargs)


async def run_build(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import build_async

    return await build_async(**kwargs)


async def run_summarize_descriptions(
    *,
    input_folder: Path,
    max_cost_usd: float = 5.0,
) -> dict[str, Any]:
    """`summarize-descriptions` subcommand: walk an existing merge folder,
    compress each class's descriptions+comments into a one-line
    compact_description, and overwrite merged.json + merged.owl in place.

    Idempotent: classes that already have a non-empty compact_description
    field are skipped. So this can be run again after a re-merge to fill
    in only the new classes.
    """
    from backend.app.core.config import get_settings
    from backend.app.services.llm_router import LLMRouter
    from backend.app.services.pipeline_llm import summarize_class_descriptions_async

    loaded = folder_io.load_version_folder(input_folder)
    settings = get_settings()
    router = LLMRouter(settings)
    classes_dict = loaded.get("classes_dict", {})

    summary = await summarize_class_descriptions_async(
        classes_dict=classes_dict,
        router=router,
        max_cost_usd=max_cost_usd,
    )

    # Write the updated dict back into the same folder so downstream
    # prune-expand runs see compact_description fields.
    folder_io.write_merged_json(input_folder, loaded)
    ontology_export.write_owl(loaded, input_folder / folder_io.MERGED_OWL)

    print(
        f"[compact-desc] DONE: summarized {summary['classes_summarized']} / "
        f"{summary['classes_total']} classes in {summary['llm_calls']} calls "
        f"(${summary['cost_usd']:.4f})"
    )
    return summary


async def run_audit_classifications(
    *,
    input_folder: Path,
) -> dict[str, Any]:
    """`audit-classifications` subcommand: walk an existing prune-expand
    folder, identify newly-created classes that look misclassified
    (parented under owl:Thing, corporate-suffix label not under
    Organization, event-keyword label not under Event, person-name
    label as a class, role label not under Role), and ask gpt-4o-mini
    to KEEP / RE_HOME / CONVERT_TO_INSTANCE each. Mutates merged.json +
    merged.owl in place.

    Standalone repair tool: lets the user fix an existing prune-expand
    output WITHOUT re-running the (expensive) full prune-expand. The
    same logic also runs automatically at the end of every prune-expand
    via the `classification_audit_enabled` config knob.
    """
    from backend.app.core.config import get_settings
    from backend.app.services.llm_router import LLMRouter
    from backend.app.services.pipeline_llm import run_classification_audit_async

    loaded = folder_io.load_version_folder(input_folder)
    settings = get_settings()
    router = LLMRouter(settings)
    app_cfg = settings.app_config
    base_iri = (app_cfg.get("ontology") or {}).get(
        "default_base_iri",
        "https://veerla-ramrao.ai/ontology/merged#",
    )

    summary = await run_classification_audit_async(
        classes_dict=loaded.setdefault("classes_dict", {}),
        instances_dict=loaded.setdefault("instances_dict", {}),
        router=router,
        default_base_iri=base_iri,
        concurrency=int((app_cfg.get("expansion") or {}).get("max_concurrent_llm_calls", 8)),
        use_cache=True,
    )

    folder_io.write_merged_json(input_folder, loaded)
    ontology_export.write_owl(loaded, input_folder / folder_io.MERGED_OWL)

    print(
        f"\nAUDIT DONE -> {input_folder}\n"
        f"  suspicious classes: {summary['suspicious']}\n"
        f"  decisions applied:  {summary['decisions']}\n"
        f"    kept:        {summary['kept']}\n"
        f"    rehomed:     {summary['rehomed']}\n"
        f"    converted:   {summary['converted']}\n"
        f"    noop:        {summary['noop']}\n"
        f"  LLM calls: {summary['llm_calls']}\n"
        f"  cost: ${summary['cost_usd']:.4f}"
    )
    return summary


# ---- Canonical vocabularies for upper ontologies (Fix D repair) ----

# Canonical class local-names per upper ontology. Anything in the
# corresponding namespace whose local-name is NOT in these sets is a
# squatter that was illegally minted by the pipeline before the Fix A
# guard landed. Sourced from the W3C / Friend-of-a-Friend specs.
_CANONICAL_FOAF_LOCALS: frozenset[str] = frozenset({
    "Agent", "Person", "Organization", "Group", "Document", "Image",
    "OnlineAccount", "OnlineChatAccount", "OnlineEcommerceAccount",
    "OnlineGamingAccount", "PersonalProfileDocument", "LabelProperty",
    "Project",
})
_CANONICAL_ORG_LOCALS: frozenset[str] = frozenset({
    "Organization", "FormalOrganization", "OrganizationalUnit",
    "Role", "Membership", "Site", "OrganizationalCollaboration",
    "ChangeEvent", "Post",
})
_CANONICAL_SKOS_LOCALS: frozenset[str] = frozenset({
    "Concept", "ConceptScheme", "Collection", "OrderedCollection",
})

# Map: namespace prefix -> canonical local-name set.
_UPPER_ONTOLOGY_CANONICAL: dict[str, frozenset[str]] = {
    "http://xmlns.com/foaf/0.1/": _CANONICAL_FOAF_LOCALS,
    "http://www.w3.org/ns/org#": _CANONICAL_ORG_LOCALS,
    "http://www.w3.org/2004/02/skos/core#": _CANONICAL_SKOS_LOCALS,
}


def _local_of(iri: str) -> str:
    """Get the local-name of an IRI. Handles both hash- and slash-IRIs."""
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _make_project_iri(local_name: str, default_base_iri: str) -> str:
    """Build an IRI under the project's default base. Handles both hash
    and slash forms of the base IRI."""
    if default_base_iri.endswith("#") or default_base_iri.endswith("/"):
        return default_base_iri + local_name
    # Default to hash form if base doesn't end with a separator.
    return default_base_iri.rstrip("#").rstrip("/") + "#" + local_name


def _walk_iris_in_record(rec: dict[str, Any]):
    """Yield (container, key/idx, iri_value) for every IRI reference
    embedded in a record. Used by the rebrand walker."""
    # Direct iri keys
    if isinstance(rec.get("iri"), str):
        yield (rec, "iri", rec["iri"])
    # Superclasses / equivalent_to / disjoint_with -- lists of entity_refs
    for field in ("superclasses", "equivalent_to", "disjoint_with"):
        for ref in (rec.get(field) or []):
            if isinstance(ref, dict) and isinstance(ref.get("iri"), str):
                yield (ref, "iri", ref["iri"])
    # Properties: domain + range
    for field in ("domain", "range"):
        for ref in (rec.get(field) or []):
            if isinstance(ref, dict) and isinstance(ref.get("iri"), str):
                yield (ref, "iri", ref["iri"])
    # Instances: types + direct_types
    for field in ("types", "direct_types"):
        for ref in (rec.get(field) or []):
            if isinstance(ref, dict) and isinstance(ref.get("iri"), str):
                yield (ref, "iri", ref["iri"])


def _rebrand_iri(old_iri: str, old_prefix: str, new_prefix: str) -> str:
    """If old_iri starts with old_prefix, swap to new_prefix; else
    return unchanged."""
    if old_iri.startswith(old_prefix):
        return new_prefix + old_iri[len(old_prefix):]
    return old_iri


async def run_repair_output(
    *,
    input_folder: Path,
    do_foaf_cleanup: bool = True,
    do_rebrand: bool = True,
    do_person_convert: bool = True,
) -> dict[str, Any]:
    """`repair-output` subcommand: repair an existing prune-expand folder
    in-place. Three deterministic ($0 LLM cost) passes:

    1. **FOAF/ORG/SKOS cleanup**: any class IRI in an upper-ontology
       namespace whose local-name is NOT canonical is a squatter; move
       it to default_base_iri. Updates every cross-reference via
       `_rewrite_iri_references`.

    2. **Brand rewrite**: every IRI starting with the deprecated
       `http://your-personal-ontologist.local/` placeholder is rewritten
       to `https://veerla-ramrao.ai/ontology/merged#`. Same for the
       custom annotation namespace. All cross-references updated.

    3. **Person-shape convert**: deterministic sweep using
       `_force_convert_person_shape`. Catches Elon-Musk-style entries
       that Layer H let slip past with KEEP.

    Writes merged.json + merged.owl back into the input folder.
    Returns summary stats.
    """
    from backend.app.core.config import get_settings
    from backend.app.helpers.ontology_pruning import _rewrite_iri_references
    from backend.app.services.pipeline_llm import (
        _force_convert_person_shape, _label_of, _looks_like_person_name,
    )

    loaded = folder_io.load_version_folder(input_folder)
    settings = get_settings()
    app_cfg = settings.app_config
    default_base_iri = (app_cfg.get("ontology") or {}).get(
        "default_base_iri", "https://veerla-ramrao.ai/ontology/merged#"
    )

    classes = loaded.setdefault("classes_dict", {})
    obj_props = loaded.setdefault("object_properties_dict", {})
    data_props = loaded.setdefault("data_properties_dict", {})
    instances = loaded.setdefault("instances_dict", {})

    summary: dict[str, Any] = {
        "foaf_cleanup_moved": 0,
        "rebrand_iris": 0,
        "person_converted": 0,
        "before_classes": len(classes),
        "before_instances": len(instances),
    }

    # ----- Pass 1: FOAF/ORG/SKOS cleanup -----
    if do_foaf_cleanup:
        moved = 0
        for prefix, canonical in _UPPER_ONTOLOGY_CANONICAL.items():
            squatters = [
                iri for iri in classes
                if iri.startswith(prefix) and _local_of(iri) not in canonical
            ]
            for old_iri in squatters:
                local = _local_of(old_iri)
                new_iri = _make_project_iri(local, default_base_iri)
                if new_iri in classes and new_iri != old_iri:
                    # Already exists at the target -- merge by deleting the
                    # squatter and letting cross-refs point to the canonical.
                    _rewrite_iri_references(
                        old_iri, new_iri, classes[new_iri].get("name") or local,
                        classes, obj_props, data_props, instances,
                    )
                    del classes[old_iri]
                else:
                    # Move the record to the new IRI.
                    rec = classes.pop(old_iri)
                    rec["iri"] = new_iri
                    classes[new_iri] = rec
                    _rewrite_iri_references(
                        old_iri, new_iri, rec.get("name") or local,
                        classes, obj_props, data_props, instances,
                    )
                moved += 1
        summary["foaf_cleanup_moved"] = moved
        print(f"[repair-output] FOAF/ORG/SKOS cleanup: {moved} squatter(s) moved to {default_base_iri}")

    # ----- Pass 2: brand rewrite -----
    # Old branding -> new branding. Covers both the default_base_iri and
    # the custom annotation namespace.
    REBRAND_MAP = [
        # (old_prefix, new_prefix)
        ("http://your-personal-ontologist.local/ontology/", default_base_iri),
        ("http://your-personal-ontologist.local/ann#", "https://veerla-ramrao.ai/ontology/ann#"),
    ]
    if do_rebrand:
        rebranded = 0
        for bucket in (classes, obj_props, data_props, instances):
            # First pass: collect renames (old_iri -> new_iri) so we can
            # rekey the dicts after iterating.
            renames: list[tuple[str, str]] = []
            for iri in list(bucket.keys()):
                new_iri = iri
                for old_prefix, new_prefix in REBRAND_MAP:
                    new_iri = _rebrand_iri(new_iri, old_prefix, new_prefix)
                if new_iri != iri:
                    renames.append((iri, new_iri))
            for old_iri, new_iri in renames:
                rec = bucket.pop(old_iri)
                rec["iri"] = new_iri
                # Local name on rec.name is likely already correct; just
                # rekey at the dict level.
                bucket[new_iri] = rec
                _rewrite_iri_references(
                    old_iri, new_iri, rec.get("name") or _local_of(new_iri),
                    classes, obj_props, data_props, instances,
                )
                rebranded += 1
        # Also rewrite any inline IRI strings in record fields (parent_iri etc.).
        # The entity-ref dict shape is already covered by _rewrite_iri_references
        # which is called per IRI above.
        summary["rebrand_iris"] = rebranded
        print(f"[repair-output] brand rewrite: {rebranded} IRI(s) updated")

    # ----- Pass 3: person-shape force-convert -----
    if do_person_convert:
        from backend.app.services.pipeline_llm import (
            _apply_audit_decision, _first_parent_label,
        )
        converted = 0
        # Iterate over a snapshot of keys -- the dict gets mutated.
        for iri in list(classes.keys()):
            rec = classes.get(iri)
            if rec is None:
                continue
            label = _label_of(rec)
            current_parent_label = _first_parent_label(rec, classes)
            if not _force_convert_person_shape(label, current_parent_label):
                continue
            # Apply a synthetic CONVERT_TO_INSTANCE decision (force_convert
            # logic in _apply_audit_decision will re-confirm).
            outcome = _apply_audit_decision(
                iri, rec,
                {"LABEL": label, "ACTION": "CONVERT_TO_INSTANCE", "NEW_PARENT": "Person"},
                classes, instances, default_base_iri,
                current_parent_label=current_parent_label,
            )
            if outcome == "converted":
                converted += 1
        summary["person_converted"] = converted
        print(f"[repair-output] person-shape convert: {converted} class(es) -> instances")

    # Write back.
    folder_io.write_merged_json(input_folder, loaded)
    ontology_export.write_owl(loaded, input_folder / folder_io.MERGED_OWL)

    summary["after_classes"] = len(classes)
    summary["after_instances"] = len(instances)
    print(
        f"\nREPAIR DONE -> {input_folder}\n"
        f"  FOAF/ORG/SKOS squatters moved : {summary['foaf_cleanup_moved']}\n"
        f"  IRIs rebranded                : {summary['rebrand_iris']}\n"
        f"  person-shape -> instance      : {summary['person_converted']}\n"
        f"  classes:  {summary['before_classes']} -> {summary['after_classes']}\n"
        f"  instances:{summary['before_instances']} -> {summary['after_instances']}"
    )
    return summary

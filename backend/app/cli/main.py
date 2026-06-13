"""CLI entry point.

    uv run python -m backend.app.cli <subcommand> [args]

Subcommands:
    merge          Multi-ontology consolidation (deterministic, zero LLM).
    prune          Drop classes unsupported by a documents folder (LLM-driven).
    expand         Propose new classes/relationships from documents (LLM-driven).
    prune-expand   Prune + expand in one pass (efficient: reuses LLM outputs).
    build          merge + prune-expand chained (one-shot end-to-end).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _add_output_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_ontologies"),
        help="Where to write the new version folder (default: output_ontologies/)",
    )


def _add_documents(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="Folder of PDF/TXT documents to drive pruning/expansion",
    )


def _add_input_folder(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A prior version folder (contains merged.json)",
    )


def _add_ontology_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ontology",
        type=Path,
        action="append",
        required=True,
        help="One ontology source (.owl, .rdf, .ttl, .xml, or .zip of these). Repeat for multiple.",
    )


def _add_llm_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-hops", type=int, default=None, help="Override prune_max_hops")
    p.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Override expansion.max_cost_usd; aborts if projected cost exceeds this cap",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only; no LLM calls and no writes")
    p.add_argument(
        "--use-owl",
        action="store_true",
        help="Force re-parsing the prior merged.owl instead of using the fast merged.json path",
    )


def _add_suggestions_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--suggested-new-classes",
        type=Path,
        default=None,
        help=(
            "Optional JSON file of user-supplied class suggestions to add IN ADDITION to "
            "whatever the LLM proposes. Format: "
            "[{CLASS_TYPE, CLASS_DESCRIPTION, PARENT_CLASS_TYPE}, ...]. "
            "Used by `expand`, `prune-expand`, and `build`."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ypo",
        description="your-personal-knowledge-graph-creator CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser(
        "merge",
        help="Consolidate multiple ontologies (no LLM)",
        description="Load + recursively merge .owl/.rdf/.ttl/.zip inputs by IRI. Deterministic, zero LLM calls.",
    )
    _add_ontology_inputs(p_merge)
    _add_output_dir(p_merge)
    p_merge.set_defaults(func=_cmd_merge)

    p_prune = sub.add_parser(
        "prune",
        help="Drop classes unsupported by documents (LLM)",
    )
    _add_input_folder(p_prune)
    _add_documents(p_prune)
    _add_output_dir(p_prune)
    _add_llm_flags(p_prune)
    p_prune.set_defaults(func=_cmd_prune)

    p_expand = sub.add_parser(
        "expand",
        help="Propose new classes/relationships from documents (LLM)",
    )
    _add_input_folder(p_expand)
    _add_documents(p_expand)
    _add_output_dir(p_expand)
    _add_llm_flags(p_expand)
    _add_suggestions_flag(p_expand)
    p_expand.set_defaults(func=_cmd_expand)

    p_pe = sub.add_parser(
        "prune-expand",
        help="Prune + expand in one pass (efficient)",
    )
    _add_input_folder(p_pe)
    _add_documents(p_pe)
    _add_output_dir(p_pe)
    _add_llm_flags(p_pe)
    _add_suggestions_flag(p_pe)
    p_pe.set_defaults(func=_cmd_prune_expand)

    p_build = sub.add_parser(
        "build",
        help="merge + prune-expand chained (end-to-end)",
    )
    _add_ontology_inputs(p_build)
    _add_documents(p_build)
    _add_output_dir(p_build)
    _add_llm_flags(p_build)
    _add_suggestions_flag(p_build)
    p_build.set_defaults(func=_cmd_build)

    p_sd = sub.add_parser(
        "summarize-descriptions",
        help=(
            "One-time compression of class descriptions+comments into a "
            "compact_description field on each class. Cuts Stage 2's per-"
            "class slice footprint by ~half for every subsequent prune-"
            "expand run against the same merge. Idempotent."
        ),
    )
    _add_input_folder(p_sd)
    p_sd.add_argument(
        "--max-cost-usd",
        type=float,
        default=5.0,
        help="Hard cost cap for the summarization pass (default $5).",
    )
    p_sd.set_defaults(func=_cmd_summarize_descriptions)

    p_audit = sub.add_parser(
        "audit-classifications",
        help=(
            "Layer H repair: scan an existing prune-expand folder for "
            "newly-created classes that look misclassified (owl:Thing "
            "parent, corporate-suffix label not under Organization, "
            "event-keyword label not under Event, person-name label "
            "as a class, role label not under Role) and ask gpt-4o-mini "
            "to KEEP / RE_HOME / CONVERT_TO_INSTANCE each. Modifies "
            "merged.json + merged.owl in place. Cached -- re-runs are "
            "free."
        ),
    )
    _add_input_folder(p_audit)
    p_audit.set_defaults(func=_cmd_audit_classifications)

    p_repair = sub.add_parser(
        "repair-output",
        help=(
            "Deterministic repair of an existing prune-expand folder: "
            "(1) move classes that squatted into FOAF/ORG/SKOS namespaces "
            "out to the project namespace; "
            "(2) rewrite the deprecated your-personal-ontologist.local "
            "placeholder IRIs to https://veerla-ramrao.ai/ontology/...; "
            "(3) convert person-name-shaped classes parented under "
            "Person/PersonRole/Role to instances of foaf:Person. $0 LLM "
            "cost. Modifies merged.json + merged.owl in place."
        ),
    )
    _add_input_folder(p_repair)
    p_repair.add_argument(
        "--skip-foaf-cleanup", action="store_true",
        help="Skip the FOAF/ORG/SKOS squatter cleanup pass.",
    )
    p_repair.add_argument(
        "--skip-rebrand", action="store_true",
        help="Skip the brand-rewrite pass.",
    )
    p_repair.add_argument(
        "--skip-person-convert", action="store_true",
        help="Skip the person-shape force-convert pass.",
    )
    p_repair.set_defaults(func=_cmd_repair_output)

    # ---------- Phase 2: import a merge folder into Postgres ----------
    p_import = sub.add_parser(
        "import-ontology",
        help=(
            "Phase 2: import a Phase-1 merge folder into the graphrag "
            "Postgres schema. Upserts classes/properties/instances by "
            "IRI. Embeds class (label + description) via "
            "text-embedding-3-small @ 1024 dim. Bumps graph_version on "
            "success. Use --limit for smoke testing and --dry-run for "
            "a no-write rehearsal."
        ),
    )
    _add_input_folder(p_import)
    p_import.add_argument(
        "--limit", type=int, default=None,
        help=(
            "Cap how many of each entity kind to process (classes, "
            "obj_props, data_props, instances). Use --limit 5 for the "
            "Milestone A smoke test."
        ),
    )
    p_import.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be imported; skip all DB writes and LLM calls.",
    )
    p_import.set_defaults(func=_cmd_import_ontology)

    # ---------- Phase 2: monitor DB size (500 MB cap) ----------
    p_dbsize = sub.add_parser(
        "db-size",
        help=(
            "Report total + per-table Postgres size for the graphrag "
            "schema. Exit 1 if total > 400 MB (close to the 500 MB cap)."
        ),
    )
    p_dbsize.set_defaults(func=_cmd_db_size)

    # ---------- Phase 2: migrate (wraps `alembic upgrade head`) ----------
    p_migrate = sub.add_parser(
        "db-migrate",
        help=(
            "Apply outstanding Alembic migrations to the Postgres "
            "schema (wraps `alembic upgrade head`). Idempotent: safe "
            "to run repeatedly."
        ),
    )
    p_migrate.add_argument(
        "--revision", default="head",
        help="Migration revision to upgrade to (default: head).",
    )
    p_migrate.set_defaults(func=_cmd_db_migrate)

    p_downgrade = sub.add_parser(
        "db-downgrade",
        help="Roll back Alembic migrations by one or more steps.",
    )
    p_downgrade.add_argument(
        "--revision", default="-1",
        help="Target revision. Default '-1' rolls back one step; 'base' clears everything.",
    )
    p_downgrade.set_defaults(func=_cmd_db_downgrade)

    p_status = sub.add_parser(
        "db-status",
        help=(
            "Report Phase 2 DB status: graph_version, alembic revision, "
            "per-table row counts."
        ),
    )
    p_status.set_defaults(func=_cmd_db_status)

    # ---------- Phase 2: one-shot init (migrate + optional import + status) ----------
    p_init = sub.add_parser(
        "db-init",
        help=(
            "One-shot Phase 2 setup. Runs db-migrate to create/upgrade "
            "the graphrag schema, optionally imports a Phase-1 "
            "merge/prune-expand folder via --input, then reports "
            "db-status. Idempotent: safe to re-run; upserts existing rows."
        ),
    )
    p_init.add_argument(
        "--input", type=Path, default=None,
        help=(
            "(Optional) Phase-1 version folder containing merged.json "
            "(e.g. output_ontologies/v...-prune-expand/). When set, "
            "imports the ontology after the migration step."
        ),
    )
    p_init.add_argument(
        "--mode", choices=("upsert", "replace"), default="upsert",
        help=(
            "How to apply the import (default: upsert). "
            "'upsert' = insert new rows + update existing rows by IRI, "
            "leave any rows whose IRI is no longer in the new ontology "
            "untouched (additive). "
            "'replace' = wipe ontology_classes + properties + instances "
            "first, then import. Use this when re-importing an ontology "
            "you've modified and want to drop classes that no longer "
            "exist. Refuses to run if dependent tables (entities, "
            "relationships, etc.) have rows -- delete documents first."
        ),
    )
    p_init.add_argument(
        "--limit", type=int, default=None,
        help="Cap how many of each entity kind to import (smoke testing).",
    )
    p_init.add_argument(
        "--dry-run", action="store_true",
        help="Migrate + report only; skip the import even if --input is set.",
    )
    p_init.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt for --mode replace.",
    )
    p_init.set_defaults(func=_cmd_db_init)

    # ---------- Phase 2: clear-corpus (wipe all corpus-derived rows) ----------
    p_clear = sub.add_parser(
        "clear-corpus",
        help=(
            "Wipe ALL corpus-derived rows (documents, chunks, "
            "time_instances, entities, intelligence_artifacts, "
            "artifact_sources, conversations, retrieval_runs, all "
            "graph_relationships except source='ONTOLOGY'). Leaves "
            "the ontology side untouched. graph_version_state is NOT "
            "reset -- the counter keeps moving forward."
        ),
    )
    p_clear.add_argument(
        "--yes", action="store_true",
        help="Required confirmation; refuses without it.",
    )
    p_clear.set_defaults(func=_cmd_clear_corpus)

    # ---------- Phase 2: Milestone B (ingestion + lifecycle) ----------
    p_reg = sub.add_parser(
        "register-documents",
        help=(
            "Milestone B: ingest a folder of .pdf/.txt files into the "
            "graphrag schema. Summarizes oversize docs via the Phase 1 "
            "disk cache, chunks, embeds, writes chunk-of edges."
        ),
    )
    p_reg.add_argument(
        "--input", type=Path, required=True,
        help="Folder of .pdf/.txt files to ingest. Generic: any corpus path.",
    )
    p_reg.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of docs to process (smoke testing).",
    )
    p_reg.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be ingested without writing.",
    )
    p_reg.add_argument(
        "--summarization-threshold", type=int, default=2000,
        help="Token threshold above which a doc gets pre-summarized.",
    )
    p_reg.add_argument(
        "--chunk-size", type=int, default=800,
        help="Target tokens per chunk.",
    )
    p_reg.add_argument(
        "--chunk-overlap", type=int, default=120,
        help="Token overlap between adjacent chunks.",
    )
    p_reg.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent LLM calls for summarization.",
    )
    p_reg.set_defaults(func=_cmd_register_documents)

    p_del = sub.add_parser(
        "delete-document",
        help="Soft-delete (default) or hard-delete a document by IRI.",
    )
    p_del.add_argument("--iri", required=True, help="Document IRI.")
    p_del.add_argument(
        "--hard", action="store_true",
        help="Hard delete (removes the row + cascades to chunks).",
    )
    p_del.set_defaults(func=_cmd_delete_document)

    p_upd = sub.add_parser(
        "update-document",
        help=(
            "Replace a document's content. No-op if the new file has "
            "the same sha256 as the existing one."
        ),
    )
    p_upd.add_argument("--iri", required=True, help="Existing document IRI.")
    p_upd.add_argument(
        "--path", type=Path, required=True,
        help="Path to the new file.",
    )
    p_upd.set_defaults(func=_cmd_update_document)

    p_list = sub.add_parser(
        "list-documents",
        help="List documents in the graphrag schema.",
    )
    p_list.add_argument(
        "--status", choices=("ACTIVE", "STALE", "DELETED"), default=None,
        help="Filter by status.",
    )
    p_list.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows returned.",
    )
    p_list.set_defaults(func=_cmd_list_documents)

    # ---------- Phase 2: Milestone D (temporal) ----------
    p_time = sub.add_parser(
        "enrich-time",
        help=(
            "Milestone D: scan chunks for date references, mint "
            "time_instances + chunk->time edges. No LLM cost."
        ),
    )
    p_time.add_argument(
        "--limit", type=int, default=None,
        help="Cap chunks scanned (smoke testing).",
    )
    p_time.add_argument(
        "--highest-level",
        choices=("year", "quarter", "month"), default="year",
        help="Top of the time hierarchy to materialize.",
    )
    p_time.add_argument(
        "--lowest-level",
        choices=("month", "day"), default="month",
        help="Bottom of the time hierarchy + gap-fill granularity.",
    )
    p_time.set_defaults(func=_cmd_enrich_time)

    # ---------- Phase 2: Milestone C (entity extraction) ----------
    p_ext = sub.add_parser(
        "extract-entities",
        help=(
            "Milestone C: extract named entities from chunks. For each "
            "chunk, vector-finds top-K candidate ontology classes, "
            "asks gpt-4o-mini to identify proper-noun entities + pick "
            "the right class IRI from that list. Mints entities, "
            "writes Chunk->viao:assertsAbout->Entity and "
            "Entity->rdf:type->OntologyClass edges. Idempotent."
        ),
    )
    p_ext.add_argument(
        "--scope-iri", default=None,
        help="Restrict to chunks of one document IRI.",
    )
    p_ext.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of chunks processed (smoke testing).",
    )
    p_ext.add_argument(
        "--candidate-classes", type=int, default=50,
        help="How many candidate ontology classes to give the LLM per chunk.",
    )
    p_ext.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent LLM calls.",
    )
    p_ext.add_argument(
        "--max-cost-usd", type=float, default=5.0,
        help="Abort if LLM spend exceeds this within the run.",
    )
    p_ext.set_defaults(func=_cmd_extract_entities)

    # ---------- Phase 2: Milestone E (artifacts) ----------
    p_art = sub.add_parser(
        "generate-artifacts",
        help=(
            "Milestone E: generate intelligence artifacts. Default = "
            "Claim+Finding+Observation per chunk plus per-doc Summary."
        ),
    )
    p_art.add_argument(
        "--type",
        choices=("Claim", "Finding", "Observation", "Summary"),
        default=None,
        action="append",
        help=(
            "Restrict to specific types (repeatable). Default = all 4. "
            "Insight + Recommendation are not implemented in v0."
        ),
    )
    p_art.add_argument(
        "--scope-iri", default=None,
        help="Restrict to one document IRI.",
    )
    p_art.add_argument(
        "--limit", type=int, default=None,
        help="Cap chunks/docs processed.",
    )
    p_art.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent LLM calls.",
    )
    p_art.add_argument(
        "--max-cost-usd", type=float, default=5.0,
        help="Abort if LLM spend exceeds this within the run.",
    )
    p_art.add_argument(
        "--no-entities", action="store_true",
        help=(
            "Skip entity grounding -- run with the generic prompt. "
            "By default generate-artifacts looks up each chunk's "
            "entities (from extract-entities) and feeds them to the "
            "LLM so artifact text names them specifically. Without "
            "this flag, if `graphrag.entities` is empty the command "
            "errors with a hint to run extract-entities first."
        ),
    )
    p_art.set_defaults(func=_cmd_generate_artifacts)

    return parser


# ---------- Subcommand handlers (lazy-import pipeline so `merge` doesn't pull
# in openai/groq SDKs and `pipeline.py` can stay self-contained).


def _cmd_merge(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_merge

    version_dir = run_merge(input_ontologies=args.ontology, output_root=args.output_dir)
    print(f"\nMERGED ontology written to: {version_dir}")
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_prune

    version_dir = asyncio.run(
        run_prune(
            input_folder=args.input,
            documents_dir=args.documents,
            output_root=args.output_dir,
            max_hops=args.max_hops,
            max_cost_usd=args.max_cost_usd,
            dry_run=args.dry_run,
        )
    )
    print(f"\nPRUNED ontology written to: {version_dir}")
    return 0


def _cmd_expand(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_expand

    version_dir = asyncio.run(
        run_expand(
            input_folder=args.input,
            documents_dir=args.documents,
            output_root=args.output_dir,
            max_hops=args.max_hops,
            max_cost_usd=args.max_cost_usd,
            dry_run=args.dry_run,
            suggested_new_classes=args.suggested_new_classes,
        )
    )
    print(f"\nEXPANDED ontology written to: {version_dir}")
    return 0


def _cmd_prune_expand(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_prune_and_expand

    version_dir = asyncio.run(
        run_prune_and_expand(
            input_folder=args.input,
            documents_dir=args.documents,
            output_root=args.output_dir,
            max_hops=args.max_hops,
            max_cost_usd=args.max_cost_usd,
            dry_run=args.dry_run,
            suggested_new_classes=args.suggested_new_classes,
        )
    )
    print(f"\nPRUNED+EXPANDED ontology written to: {version_dir}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_build

    version_dir = asyncio.run(
        run_build(
            input_ontologies=args.ontology,
            documents_dir=args.documents,
            output_root=args.output_dir,
            max_hops=args.max_hops,
            max_cost_usd=args.max_cost_usd,
            dry_run=args.dry_run,
            suggested_new_classes=args.suggested_new_classes,
        )
    )
    print(f"\nBUILT ontology written to: {version_dir}")
    return 0


def _cmd_summarize_descriptions(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_summarize_descriptions

    summary = asyncio.run(
        run_summarize_descriptions(
            input_folder=args.input,
            max_cost_usd=args.max_cost_usd,
        )
    )
    print(f"\nSUMMARIZE-DESCRIPTIONS DONE -> {args.input}")
    print(f"  classes total:       {summary['classes_total']}")
    print(f"  classes summarized:  {summary['classes_summarized']}")
    print(f"  classes skipped:     {summary['classes_skipped']}")
    print(f"  LLM calls:           {summary['llm_calls']}")
    print(f"  cost:                ${summary['cost_usd']:.4f}")
    return 0


def _cmd_audit_classifications(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_audit_classifications

    asyncio.run(run_audit_classifications(input_folder=args.input))
    return 0


def _cmd_repair_output(args: argparse.Namespace) -> int:
    from backend.app.services.pipeline import run_repair_output

    asyncio.run(run_repair_output(
        input_folder=args.input,
        do_foaf_cleanup=not args.skip_foaf_cleanup,
        do_rebrand=not args.skip_rebrand,
        do_person_convert=not args.skip_person_convert,
    ))
    return 0


def _cmd_import_ontology(args: argparse.Namespace) -> int:
    from backend.app.services.db_ontology_import import import_ontology_folder

    summary = asyncio.run(import_ontology_folder(
        input_folder=args.input,
        limit=args.limit,
        dry_run=args.dry_run,
    ))
    print(f"\nDB-IMPORT SUMMARY -> {args.input}")
    print(f"  classes seen      : {summary.classes_total}")
    print(f"  classes embedded  : {summary.classes_embedded}")
    print(f"  obj_props seen    : {summary.obj_props_total}")
    print(f"  data_props seen   : {summary.data_props_total}")
    print(f"  instances seen    : {summary.instances_total}")
    print(f"  embed cost        : ${summary.cost_usd:.4f}")
    if summary.dry_run:
        print(f"  (DRY RUN -- no DB writes happened)")
    return 0


def _cmd_db_size(args: argparse.Namespace) -> int:
    from backend.app.services.db_size import report_db_size

    asyncio.run(report_db_size())
    return 0


def _cmd_db_migrate(args: argparse.Namespace) -> int:
    """Wraps `alembic upgrade <revision>`."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    print(f"[db-migrate] upgrading to revision: {args.revision}")
    command.upgrade(cfg, args.revision)
    print(f"[db-migrate] DONE")
    return 0


def _cmd_db_downgrade(args: argparse.Namespace) -> int:
    """Wraps `alembic downgrade <revision>`."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    print(f"[db-downgrade] target revision: {args.revision}")
    command.downgrade(cfg, args.revision)
    print(f"[db-downgrade] DONE")
    return 0


def _cmd_db_status(args: argparse.Namespace) -> int:
    from backend.app.services.db_status import report_db_status

    asyncio.run(report_db_status())
    return 0


def _cmd_db_init(args: argparse.Namespace) -> int:
    """One-shot: migrate -> (optional) wipe -> (optional) import -> status.

    Idempotent. Safe to re-run -- the migration is no-op if already
    applied; default mode 'upsert' inserts new + updates existing rows.
    Use --mode replace to wipe ontology tables before reimporting (when
    you've modified the ontology and want stale rows removed).
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig

    from backend.app.db.engine import reset_engine_cache
    from backend.app.services.db_status import report_db_status

    print("=" * 64)
    print("DB-INIT: step 1/4 -- migrate (alembic upgrade head)")
    print("=" * 64)
    cfg = AlembicConfig("alembic.ini")
    command.upgrade(cfg, "head")
    print("[db-init] migrate DONE")
    print()

    # Alembic ran its own asyncio loop; that loop is now closed. Drop
    # the cached engine so the next get_engine() builds fresh against
    # the new loop. Can't await dispose_engine() here -- it would try
    # to close connections bound to the dead loop.
    reset_engine_cache()

    # ---- Optional wipe (--mode replace) ----
    if args.input is not None and args.mode == "replace" and not args.dry_run:
        print("=" * 64)
        print("DB-INIT: step 2/4 -- wipe ontology tables (--mode replace)")
        print("=" * 64)

        from backend.app.services.db_ontology_wipe import wipe_ontology_tables

        try:
            asyncio.run(wipe_ontology_tables(confirm=args.yes))
            reset_engine_cache()
        except RuntimeError as exc:
            print(f"\n[db-init] WIPE REFUSED: {exc}")
            print("[db-init] Either delete the dependent rows first, "
                  "or use --mode upsert.")
            return 1
        print()
    else:
        if args.input is not None and not args.dry_run:
            print("=" * 64)
            print("DB-INIT: step 2/4 -- mode=upsert, no wipe needed")
            print("=" * 64)
            print()

    if args.input is not None and not args.dry_run:
        from backend.app.services.db_ontology_import import import_ontology_folder
        print("=" * 64)
        print(f"DB-INIT: step 3/4 -- import from {args.input}")
        print("=" * 64)
        summary = asyncio.run(import_ontology_folder(
            input_folder=args.input,
            limit=args.limit,
            dry_run=False,
        ))
        print()
        reset_engine_cache()
    elif args.input is not None and args.dry_run:
        print("=" * 64)
        print("DB-INIT: step 3/4 -- import SKIPPED (--dry-run set)")
        print("=" * 64)
        print(f"  would have imported: {args.input}")
        print()
    else:
        print("=" * 64)
        print("DB-INIT: step 3/4 -- no --input given, skipping import")
        print("=" * 64)
        print()

    print("=" * 64)
    print("DB-INIT: step 4/4 -- report status")
    print("=" * 64)
    asyncio.run(report_db_status())
    return 0


def _cmd_clear_corpus(args: argparse.Namespace) -> int:
    from backend.app.services.db_corpus_wipe import wipe_corpus

    try:
        asyncio.run(wipe_corpus(confirm=args.yes))
    except RuntimeError as exc:
        print(f"[clear-corpus] REFUSED: {exc}")
        return 1
    return 0


def _cmd_register_documents(args: argparse.Namespace) -> int:
    from backend.app.services.db_document_ingest import ingest_documents_folder

    asyncio.run(
        ingest_documents_folder(
            folder=args.input,
            limit=args.limit,
            dry_run=args.dry_run,
            summarization_threshold=args.summarization_threshold,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            concurrency=args.concurrency,
        )
    )
    return 0


def _cmd_delete_document(args: argparse.Namespace) -> int:
    from backend.app.services.db_document_lifecycle import delete_document

    asyncio.run(delete_document(iri=args.iri, hard=args.hard))
    return 0


def _cmd_update_document(args: argparse.Namespace) -> int:
    from backend.app.services.db_document_lifecycle import update_document

    result = asyncio.run(update_document(iri=args.iri, new_path=args.path))
    print(result)
    return 0


def _cmd_list_documents(args: argparse.Namespace) -> int:
    from backend.app.services.db_document_lifecycle import list_documents

    rows = asyncio.run(list_documents(status=args.status, limit=args.limit))
    if not rows:
        print("(no documents)")
        return 0
    print(f"{'status':<10} {'v':<3} {'chunks':<7} {'created_at':<26} title")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['status']:<10} {r['version']:<3} {r['chunks']:<7} "
            f"{r['created_at']:<26} {r['title']}"
        )
    return 0


def _cmd_enrich_time(args: argparse.Namespace) -> int:
    from backend.app.services.db_temporal_enrich import enrich_temporal

    asyncio.run(
        enrich_temporal(
            limit=args.limit,
            highest_level=args.highest_level,
            lowest_level=args.lowest_level,
        )
    )
    return 0


def _cmd_extract_entities(args: argparse.Namespace) -> int:
    from backend.app.services.db_entity_extract import extract_entities

    asyncio.run(
        extract_entities(
            scope_document_iri=args.scope_iri,
            limit=args.limit,
            candidate_classes_per_chunk=args.candidate_classes,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost_usd,
        )
    )
    return 0


def _cmd_generate_artifacts(args: argparse.Namespace) -> int:
    from backend.app.db.engine import reset_engine_cache
    from backend.app.services.db_artifact_gen import (
        generate_document_summaries,
        generate_per_chunk_artifacts,
    )

    types_requested = tuple(args.type) if args.type else (
        "Claim", "Finding", "Observation", "Summary"
    )
    per_chunk = tuple(t for t in types_requested if t in ("Claim", "Finding", "Observation"))
    do_summary = "Summary" in types_requested

    use_entities = not args.no_entities
    if per_chunk:
        asyncio.run(
            generate_per_chunk_artifacts(
                scope_document_iri=args.scope_iri,
                limit=args.limit,
                types=per_chunk,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
                use_entities=use_entities,
            )
        )
        # The above asyncio.run's loop is now dead; drop the cached
        # engine so the next call builds fresh against its own loop.
        reset_engine_cache()

    if do_summary:
        asyncio.run(
            generate_document_summaries(
                scope_document_iri=args.scope_iri,
                limit=args.limit,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
            )
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

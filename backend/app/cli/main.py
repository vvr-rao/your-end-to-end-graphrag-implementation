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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

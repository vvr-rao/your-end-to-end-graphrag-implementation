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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

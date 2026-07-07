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
import json
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
        description="your-end-to-end-graphrag-implementation CLI",
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
    p_pe.add_argument(
        "--tables", action="store_true",
        help=(
            "Phase 2a (opt-in): extract structured tables from PDF documents "
            "during the run. Writes JSON-LD payloads to "
            "<output>/v<TS>-prune-expand/tables/<sha>.jsonld and to the "
            "user cache. Has no effect on non-PDF inputs. Default: OFF."
        ),
    )
    p_pe.add_argument(
        "--no-table-vision",
        dest="table_vision", action="store_false", default=True,
        help=(
            "Disable the vision-LLM route for complex tables (use only "
            "pdfplumber's flat extraction). Free, loses nested/merged "
            "tables. Effective only when --tables is set."
        ),
    )
    p_pe.add_argument(
        "--single-pass-summaries", action="store_true",
        help=(
            "Summarize documents with the legacy one-shot document_summarize "
            "instead of the default EVALUATED summarizer (summarize -> "
            "question-gen -> evaluate -> revise). The evaluated path is higher "
            "fidelity and its summaries are cached so register-documents reuses "
            "them for free; use this to fall back to the cheaper single-pass."
        ),
    )
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
    p_build.add_argument(
        "--tables", action="store_true",
        help="Phase 2a (opt-in): extract structured tables from PDF documents.",
    )
    p_build.add_argument(
        "--no-table-vision",
        dest="table_vision", action="store_false", default=True,
        help="Disable vision-LLM route for complex tables (uses pdfplumber-only).",
    )
    p_build.add_argument(
        "--single-pass-summaries", action="store_true",
        help="Use the legacy one-shot document summarizer instead of the default "
             "evaluated summarizer (see prune-expand --single-pass-summaries).",
    )
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
    p_reg.add_argument(
        "--tables", action="store_true",
        help=(
            "Phase 2a (opt-in): also ingest StructuredTable artifacts from "
            "PDF documents. Loads cached JSON-LD payloads when available "
            "(from a prior `prune-expand --tables` run) and extracts inline "
            "otherwise. Has no effect on non-PDF inputs. Default: OFF."
        ),
    )
    p_reg.add_argument(
        "--no-table-vision",
        dest="table_vision", action="store_false", default=True,
        help=(
            "Disable vision-LLM route during inline extraction. Free, loses "
            "complex tables. Effective only with --tables."
        ),
    )
    p_reg.add_argument(
        "--full-text-chunks", action="store_true",
        help=(
            "Also store verbatim full-text chunks (kind='fulltext') in addition "
            "to the summary chunks. Retrieval prefers full-text chunks (better "
            "recall + exact citations); entity/artifact extraction still use "
            "summary chunks. Increases DB size — smoke-test with --limit and "
            "check `db-size`. Default: OFF."
        ),
    )
    p_reg.add_argument(
        "--single-pass-summaries", action="store_true",
        help=(
            "Use the legacy one-shot document_summarize instead of the default "
            "EVALUATED summarizer (summarize -> question-gen -> evaluate -> revise). "
            "The evaluated path is higher-fidelity but ~6-9 LLM calls per chunk; "
            "use this to fall back to the cheaper single-pass summary."
        ),
    )
    p_reg.add_argument(
        "--eval-rounds", type=int, default=None,
        help=(
            "Evaluator revise rounds per chunk for the evaluated summarizer "
            "(default from config summarization.eval_rounds = 3). Higher = more "
            "fidelity + more cost. Ignored with --single-pass-summaries."
        ),
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
    p_time.add_argument(
        "--from-fulltext", action="store_true",
        help=(
            "Scan the verbatim FULL-TEXT chunks for dates instead of summary "
            "chunks, matching extract-entities / generate-artifacts so time "
            "edges anchor on the same chunk kind. Default: summary."
        ),
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
    p_ext.add_argument(
        "--from-fulltext", action="store_true",
        help=(
            "Mine entities from the verbatim FULL-TEXT chunks (kind='fulltext') "
            "instead of the summary chunks. Far more complete (captures every "
            "study/entity in the source), but ~18x more LLM calls + more DB rows. "
            "Requires the corpus was ingested with --full-text-chunks. Default: summary."
        ),
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
        choices=("Claim", "Finding", "Observation", "Event", "Summary",
                 "Insight", "Recommendation"),
        default=None,
        action="append",
        help=(
            "Restrict to specific types (repeatable). Default = the five "
            "ground-level types (Claim, Finding, Observation, Event, "
            "Summary). Insight + Recommendation must be opted in "
            "explicitly (they use gpt-4.1; cost ~$2 for full corpus)."
        ),
    )
    p_art.add_argument(
        "--min-claims-per-class", type=int, default=10,
        help="Insight only: minimum Claims+Findings per ontology class "
             "for a cluster to qualify.",
    )
    p_art.add_argument(
        "--top-insights", type=int, default=15,
        help="Recommendation only: how many top Insights to synthesize from.",
    )
    p_art.add_argument(
        "--theme-label", default="Corpus-wide synthesis",
        help="Recommendation only: thematic label for the synthesis run.",
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
    p_art.add_argument(
        "--from-fulltext", action="store_true",
        help=(
            "Extract per-chunk artifacts (Claim/Finding/Observation/Event) from "
            "the verbatim FULL-TEXT chunks instead of the summary chunks. More "
            "complete, but ~18x more LLM calls + more DB rows. Requires the corpus "
            "was ingested with --full-text-chunks. Default: summary."
        ),
    )
    # ---- Hierarchical clustered rollups (post-processing over existing artifacts) ----
    p_art.add_argument(
        "--rollup", action="store_true",
        help=(
            "After the base stages, cluster semantically-similar artifacts and "
            "mint NEW lossless rollup artifacts (originals retained; linked via "
            "viao:referencesArtifact). Gives retrieval a coarse-to-fine layer. "
            "Uses gpt-4.1 (artifact_merge). Smoke-test with --scope-iri/--limit "
            "+ --rollup-layers 1 and check db-size before a full run."
        ),
    )
    p_art.add_argument(
        "--rollup-layers", type=int, default=2,
        help="Rollup hierarchy depth: leaves -> L1 -> L2 ... (default 2).",
    )
    p_art.add_argument(
        "--rollup-threshold", type=float, default=0.35,
        help="Max L2 embedding distance for two artifacts to cluster "
             "(smaller = tighter/fewer merges). Default 0.35; tune per corpus.",
    )
    p_art.add_argument(
        "--rollup-min-cluster", type=int, default=2,
        help="Minimum artifacts in a cluster to produce a rollup (default 2).",
    )
    p_art.add_argument(
        "--rollup-max-neighbors", type=int, default=25,
        help="Max same-type neighbors fetched per artifact when clustering "
             "(default 25).",
    )
    p_art.add_argument(
        "--rollup-eval-rounds", type=int, default=None,
        help="Lossless-merge revise passes per rollup cluster (default from "
             "config summarization.rollup_eval_rounds = 1). After each merge an "
             "LLM checks for dropped facts and a reviser adds them back. 0 = merge "
             "only (faster, may lose data). Higher = safer + more LLM cost.",
    )
    p_art.add_argument(
        "--rollup-type",
        choices=("Claim", "Finding", "Observation", "Event", "Summary",
                 "Insight", "Recommendation", "StructuredTable"),
        default=None, action="append",
        help="Artifact types to roll up (repeatable). Default = ALL types "
             "except StructuredTable. Clustering is homogeneous within each type.",
    )
    p_art.set_defaults(func=_cmd_generate_artifacts)

    # ---------- Phase 2: Milestone F (retrieval + answer synthesis) ----------
    p_q = sub.add_parser(
        "query",
        help=(
            "Milestone F: run the 12-step retrieval pipeline against a "
            "natural-language question and synthesize an answer with "
            "citations. Persists retrieval_runs + retrieval_evidence."
        ),
    )
    p_q.add_argument("question", help="The user question, in quotes.")
    p_q.add_argument(
        "--mode",
        choices=("simple_qa", "deep_research", "artifact_only", "artifact-only"),
        default="deep_research",
        help=(
            "deep_research (default): structured 7-section output "
            "(SPECIFICS / ANALYSIS / ANSWER / CONTRADICTIONS / "
            "KEY CLAIMS / COVERAGE IMBALANCE / KEY INSIGHTS). "
            "simple_qa: tight 1-3 sentence direct answer. "
            "artifact_only: same structured output, but retrieves ONLY from the "
            "intelligence artifacts (all types, entity-linked + global vector), "
            "never chunks."
        ),
    )
    p_q.add_argument(
        "--top-k", type=int, default=None,
        help="How many candidates to surface as evidence. Defaults to "
             "30 for deep_research, 20 for simple_qa.",
    )
    p_q.add_argument(
        "--hops", type=int, default=None,
        help="Graph BFS depth from seed nodes. Default from config.yaml qa.hops.",
    )
    p_q.add_argument(
        "--max-cost-usd", type=float, default=1.0,
        help="Abort if LLM spend exceeds this within the run.",
    )
    p_q.add_argument(
        "--no-decompose", action="store_true",
        help="Skip step 9a query decomposition; use original query as single probe.",
    )
    p_q.add_argument(
        "--max-probes", type=int, default=5,
        help="Cap on number of vector-search probes (incl. the original query).",
    )
    p_q.add_argument(
        "--json", action="store_true",
        help="Print the full response envelope as JSON.",
    )
    p_q.add_argument(
        "--verbose", action="store_true",
        help="Print debug info from each pipeline step.",
    )
    p_q.add_argument(
        "--single-round", dest="multi_round", action="store_false", default=True,
        help=(
            "Disable the 2-round iterative retrieval (deep_research only). By "
            "default, a planner detects dependent multi-part questions (e.g. "
            "'compare X to its competitors, which is fastest?') and runs a bridge "
            "round to discover entities first. This flag forces a single round."
        ),
    )
    p_q.set_defaults(func=_cmd_query)

    # ---------- Phase 2: Eval framework (LLM-as-judge) ----------
    p_ev = sub.add_parser(
        "evaluate-queries",
        help=(
            "Run a question file through the F retrieval pipeline N "
            "times per question, then score each answer on 4 LLM-judged "
            "metrics (comprehensiveness, no-hallucination, consistency, "
            "gap detection) plus wall-time. Writes JSON + optional MD log."
        ),
    )
    p_ev.add_argument("--questions", type=Path, required=True,
                      help="Text file; one question per line; `#` comments; "
                           "`[gap]` tag marks questions the corpus shouldn't "
                           "be able to answer.")
    p_ev.add_argument(
        "--mode",
        choices=("simple_qa", "deep_research", "artifact_only", "artifact-only"),
        default="deep_research",
    )
    p_ev.add_argument("--runs-per-question", type=int, default=3)
    p_ev.add_argument(
        "--judge-model",
        choices=("gpt-4.1", "gpt-4o-mini"),
        default="gpt-4.1",
        help="Override the judge model. gpt-4o-mini is ~10x cheaper but less reliable.",
    )
    p_ev.add_argument("--output", type=Path, default=None,
                      help="Detailed JSON log.")
    p_ev.add_argument("--output-md", type=Path, default=None,
                      help="Human-readable markdown summary.")
    p_ev.add_argument("--max-cost-usd", type=float, default=10.0,
                      help="Whole-run cost cap; aborts mid-run.")
    p_ev.add_argument("--query-max-cost-usd", type=float, default=0.20,
                      help="Per-query cost cap inside the eval loop.")
    p_ev.add_argument("--concurrency", type=int, default=4)
    p_ev.add_argument("--verbose", action="store_true")
    p_ev.set_defaults(func=_cmd_evaluate_queries)

    # ---------- Phase 2: Milestone G (conversation-aware QA) ----------
    p_conv = sub.add_parser(
        "conversation",
        help=(
            "Multi-turn QA. `start` opens a conversation; `turn` adds "
            "a follow-up resolved against prior turns; `show` replays."
        ),
    )
    conv_sub = p_conv.add_subparsers(dest="conv_cmd", required=True)

    p_conv_start = conv_sub.add_parser("start", help="Open a new conversation.")
    p_conv_start.add_argument(
        "--title", default=None,
        help="Optional title (stored in conversations.extra_metadata.title).",
    )
    p_conv_start.add_argument(
        "--json", action="store_true",
        help="Print full JSON envelope instead of just the IRI.",
    )
    p_conv_start.set_defaults(func=_cmd_conversation_start)

    p_conv_turn = conv_sub.add_parser(
        "turn", help="Add one turn (a question) to a conversation."
    )
    p_conv_turn.add_argument("--conv", required=True, help="Conversation IRI.")
    p_conv_turn.add_argument("question", help="The user question in quotes.")
    p_conv_turn.add_argument(
        "--mode",
        choices=("simple_qa", "deep_research", "artifact_only", "artifact-only"),
        default="deep_research",
    )
    p_conv_turn.add_argument("--top-k", type=int, default=None)
    p_conv_turn.add_argument("--hops", type=int, default=None,
                             help="Graph BFS depth. Default from config.yaml qa.hops.")
    p_conv_turn.add_argument(
        "--max-cost-usd", type=float, default=0.50,
        help=(
            "Per-turn cost cap. Default 0.50: a deep_research turn runs the "
            "2-round bridge + several gpt-4.1 synthesis calls, which the old "
            "0.20 cap could truncate. simple_qa turns stay well under it."
        ),
    )
    p_conv_turn.add_argument("--no-decompose", action="store_true")
    p_conv_turn.add_argument("--max-probes", type=int, default=5)
    p_conv_turn.add_argument(
        "--history-window", type=int, default=3,
        help="How many prior turns to feed into follow-up resolution.",
    )
    p_conv_turn.add_argument("--json", action="store_true")
    p_conv_turn.add_argument("--verbose", action="store_true")
    p_conv_turn.set_defaults(func=_cmd_conversation_turn)

    p_conv_show = conv_sub.add_parser(
        "show", help="Replay a conversation."
    )
    p_conv_show.add_argument("--conv", required=True, help="Conversation IRI.")
    p_conv_show.add_argument("--json", action="store_true")
    p_conv_show.set_defaults(func=_cmd_conversation_show)

    # ---- Phase 3: Render lifecycle CLI -----------------------------------
    p_rb = sub.add_parser(
        "render-bootstrap",
        help=(
            "One-time setup helper: list Render owners to find your "
            "RENDER_OWNER_ID."
        ),
    )
    p_rb.add_argument("--json", action="store_true")
    p_rb.set_defaults(func=_cmd_render_bootstrap)

    p_ri = sub.add_parser(
        "render-init",
        help=(
            "First-time CLI deploy: create backend + frontend services on "
            "Render from render.yaml + .env, no dashboard click-through."
        ),
    )
    p_ri.add_argument(
        "--branch", default=None,
        help="Git branch to track (default: current local branch).",
    )
    p_ri.add_argument(
        "--repo", default=None,
        help="GitHub repo URL (default: parsed from 'origin' remote).",
    )
    p_ri.add_argument(
        "--no-deploy", action="store_true",
        help="Create services + env vars but skip the initial deploy trigger.",
    )
    p_ri.set_defaults(func=_cmd_render_init)

    p_rs = sub.add_parser(
        "render-status",
        help="Show the state of one or all Render services.",
    )
    p_rs.add_argument(
        "--service", default="all",
        help="Service name (backend / frontend) or 'all' (default).",
    )
    p_rs.add_argument("--json", action="store_true")
    p_rs.set_defaults(func=_cmd_render_status)

    p_rd = sub.add_parser(
        "render-deploy",
        help="Trigger a fresh deploy of a Render service.",
    )
    p_rd.add_argument(
        "--service", default="backend",
        help="Service name (backend / frontend). Default: backend.",
    )
    p_rd.add_argument(
        "--wait", action="store_true",
        help="Poll until the deploy reaches 'live' or 'failed'.",
    )
    p_rd.add_argument(
        "--clear-cache", action="store_true",
        help="Pass clearCache:'clear' to the deploy.",
    )
    p_rd.add_argument("--json", action="store_true")
    p_rd.set_defaults(func=_cmd_render_deploy)

    p_rsus = sub.add_parser(
        "render-suspend",
        help="Suspend a Render service (stops compute immediately).",
    )
    p_rsus.add_argument("--service", help="Service name.")
    p_rsus.add_argument(
        "--all", action="store_true",
        help="Suspend backend + frontend together.",
    )
    p_rsus.set_defaults(func=_cmd_render_suspend)

    p_rres = sub.add_parser(
        "render-resume",
        help="Resume a previously-suspended Render service.",
    )
    p_rres.add_argument("--service", help="Service name.")
    p_rres.add_argument(
        "--all", action="store_true",
        help="Resume backend + frontend together.",
    )
    p_rres.set_defaults(func=_cmd_render_resume)

    p_rl = sub.add_parser(
        "render-logs",
        help="Fetch recent log lines from a Render service.",
    )
    p_rl.add_argument("--service", required=True, help="Service name.")
    p_rl.add_argument(
        "--since", default="30m",
        help="Look back this far (e.g. 5m, 1h, 24h). Default: 30m.",
    )
    p_rl.add_argument("--limit", type=int, default=100)
    p_rl.add_argument("--json", action="store_true")
    p_rl.set_defaults(func=_cmd_render_logs)

    p_rtd = sub.add_parser(
        "render-takedown",
        help=(
            "Suspend (default) or delete the backend + frontend services. "
            "Requires --yes."
        ),
    )
    p_rtd.add_argument(
        "--hard", action="store_true",
        help="Delete the services entirely (irreversible). Default: suspend.",
    )
    p_rtd.add_argument(
        "--yes", action="store_true",
        help="Confirm the takedown without an interactive prompt.",
    )
    p_rtd.set_defaults(func=_cmd_render_takedown)

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
            extract_tables=getattr(args, "tables", False),
            table_vision=getattr(args, "table_vision", True),
            single_pass_summaries=getattr(args, "single_pass_summaries", False),
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
            extract_tables=getattr(args, "tables", False),
            table_vision=getattr(args, "table_vision", True),
            single_pass_summaries=getattr(args, "single_pass_summaries", False),
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
            extract_tables=getattr(args, "tables", False),
            table_vision=getattr(args, "table_vision", True),
            full_text_chunks=getattr(args, "full_text_chunks", False),
            summarization_method=(
                "single_pass" if getattr(args, "single_pass_summaries", False) else None
            ),
            eval_rounds=getattr(args, "eval_rounds", None),
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
            chunk_kind="fulltext" if getattr(args, "from_fulltext", False) else "summary",
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
            chunk_kind="fulltext" if getattr(args, "from_fulltext", False) else "summary",
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
        "Claim", "Finding", "Observation", "Event", "Summary"
    )
    per_chunk = tuple(
        t for t in types_requested
        if t in ("Claim", "Finding", "Observation", "Event")
    )
    do_summary = "Summary" in types_requested
    do_insight = "Insight" in types_requested
    do_recommendation = "Recommendation" in types_requested

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
                chunk_kind="fulltext" if getattr(args, "from_fulltext", False) else "summary",
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
        reset_engine_cache()

    if do_insight:
        from backend.app.services.db_insight_gen import generate_insights
        asyncio.run(
            generate_insights(
                min_claims_per_class=args.min_claims_per_class,
                limit=args.limit,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
            )
        )
        reset_engine_cache()

    if do_recommendation:
        from backend.app.services.db_recommendation_gen import generate_recommendations
        asyncio.run(
            generate_recommendations(
                top_insights=args.top_insights,
                max_cost_usd=args.max_cost_usd,
                theme_label=args.theme_label,
            )
        )
        reset_engine_cache()

    if getattr(args, "rollup", False):
        from backend.app.services.db_artifact_rollup import (
            ALL_ROLLUP_TYPES,
            generate_rollups,
        )
        # A prior stage's asyncio.run loop is now dead; rebuild the engine fresh.
        reset_engine_cache()
        from backend.app.core.config import get_settings
        rollup_types = tuple(args.rollup_type) if args.rollup_type else ALL_ROLLUP_TYPES
        _cfg_eval = int(
            get_settings().app_config.get("summarization", {}).get("rollup_eval_rounds", 1)
        )
        _eval_rounds = (
            args.rollup_eval_rounds if args.rollup_eval_rounds is not None else _cfg_eval
        )
        asyncio.run(
            generate_rollups(
                types=rollup_types,
                layers=args.rollup_layers,
                threshold=args.rollup_threshold,
                min_cluster=args.rollup_min_cluster,
                max_neighbors=args.rollup_max_neighbors,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
                scope_document_iri=args.scope_iri,
                eval_rounds=_eval_rounds,
                verbose=True,
            )
        )

    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    import json as _json
    from backend.app.services.retrieval import retrieve_and_answer

    # Accept the hyphenated spelling (artifact-only) as an alias for the
    # canonical underscore mode name used throughout the code.
    args.mode = args.mode.replace("-", "_")

    result = asyncio.run(
        retrieve_and_answer(
            args.question,
            mode=args.mode,
            top_k=args.top_k,
            hops=args.hops,
            max_cost_usd=args.max_cost_usd,
            decompose=not args.no_decompose,
            max_probes=args.max_probes,
            verbose=args.verbose,
            multi_round=getattr(args, "multi_round", True),
        )
    )

    if args.json:
        envelope = {
            "answer": result.answer,
            "mode": result.mode,
            "resolved_query": result.resolved_query,
            "evidence": result.evidence,
            "retrieval_run_id": str(result.retrieval_run_id) if result.retrieval_run_id else None,
            "parsed": result.parsed,
            "cost_usd": result.cost_usd,
            "wall_seconds": result.wall_seconds,
            "graph_version": result.graph_version,
        }
        print(_json.dumps(envelope, indent=2, default=str))
        return 0

    # Human-readable text output.
    print("=" * 72)
    print(f"MODE:     {result.mode}")
    print(f"QUERY:    {result.resolved_query}")
    print(f"COST:     ${result.cost_usd:.4f}   wall: {result.wall_seconds:.1f}s")
    print(f"RUN_ID:   {result.retrieval_run_id}")
    print("=" * 72)

    if result.answer:
        print("ANSWER:")
        print(result.answer)
        print()
    print(f"EVIDENCE ({len(result.evidence)} items):")
    for ev in result.evidence[:15]:
        kind = ev["kind"]
        line = f"  [{ev['rank']:>2}] [{kind}] {ev['iri']}  (score={ev['score']:.4f})"
        if kind == "chunk":
            line += f"\n        doc: {ev['document_title']}"
            snip = (ev["text"] or "").replace("\n", " ")[:140]
            line += f"\n        > {snip}"
        elif kind == "artifact":
            snip = (ev["text"] or "").replace("\n", " ")[:140]
            line += f"\n        ({ev['artifact_type']}) {snip}"
        print(line)
    return 0


def _cmd_evaluate_queries(args: argparse.Namespace) -> int:
    from backend.app.services.eval_judge import evaluate_questions

    # Accept the hyphenated spelling (artifact-only) as an alias for the
    # canonical underscore mode name used throughout the code.
    args.mode = args.mode.replace("-", "_")

    asyncio.run(
        evaluate_questions(
            questions_path=args.questions,
            mode=args.mode,
            runs_per_question=args.runs_per_question,
            judge_model=args.judge_model,
            output_json=args.output,
            output_md=args.output_md,
            max_cost_usd=args.max_cost_usd,
            query_max_cost_usd=args.query_max_cost_usd,
            concurrency=args.concurrency,
            verbose=args.verbose,
        )
    )
    return 0


def _cmd_conversation_start(args: argparse.Namespace) -> int:
    import json as _json
    from backend.app.services.db_conversation import start_conversation

    result = asyncio.run(start_conversation(title=args.title))
    if args.json:
        print(_json.dumps(result, indent=2))
    else:
        print(f"iri: {result['iri']}")
        if result.get("title"):
            print(f"title: {result['title']}")
    return 0


def _cmd_conversation_turn(args: argparse.Namespace) -> int:
    import json as _json
    from backend.app.services.db_conversation import add_turn

    # Accept the hyphenated spelling (artifact-only) as an alias for the
    # canonical underscore mode name used throughout the code.
    args.mode = args.mode.replace("-", "_")

    result = asyncio.run(
        add_turn(
            conversation_iri=args.conv,
            question=args.question,
            mode=args.mode,
            top_k=args.top_k,
            hops=args.hops,
            max_cost_usd=args.max_cost_usd,
            decompose=not args.no_decompose,
            max_probes=args.max_probes,
            history_window=args.history_window,
            verbose=args.verbose,
        )
    )
    if args.json:
        print(_json.dumps(result, indent=2, default=str))
        return 0
    print("=" * 72)
    print(f"TURN:     {result['turn_index']}")
    print(f"MODE:     {result['mode']}")
    print(f"ASKED:    {result['user_question']}")
    if result["follow_up_resolved"]:
        print(f"RESOLVED: {result['resolved_question']}")
    print(f"COST:     ${result['cost_usd']:.4f}   wall: {result['wall_seconds']:.1f}s")
    print(f"RUN_ID:   {result['retrieval_run_id']}")
    print("=" * 72)
    if result.get("answer"):
        print("ANSWER:")
        print(result["answer"])
    return 0


def _cmd_conversation_show(args: argparse.Namespace) -> int:
    import json as _json
    from backend.app.services.db_conversation import replay_conversation

    result = asyncio.run(replay_conversation(conversation_iri=args.conv))
    if args.json:
        print(_json.dumps(result, indent=2, default=str))
        return 0
    print("=" * 72)
    print(f"CONVERSATION: {result['iri']}")
    if result.get("title"):
        print(f"TITLE:        {result['title']}")
    print(f"TURNS:        {result['turn_count']}")
    print(f"STARTED:      {result['created_at']}")
    print("=" * 72)
    for t in result["turns"]:
        print()
        print(f"  --- turn {t['turn_index']} ({t['mode']}) ---")
        print(f"  ASKED:    {t['user_question']}")
        if t["follow_up_resolved"]:
            print(f"  RESOLVED: {t['resolved_question']}")
        ans = (t["answer"] or "").strip()
        if ans:
            print(f"  ANSWER:   {ans}")
    return 0


# ---------------------------------------------------------------------------
# Phase 3: Render lifecycle handlers
# ---------------------------------------------------------------------------


_RENDER_PHASE3_SERVICES = ("backend", "frontend")


def _iso_minutes_ago(spec: str) -> str:
    """Parse '30m' / '2h' / '24h' / '7d' into an RFC3339 timestamp in UTC."""
    import datetime as _dt
    import re

    m = re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", spec)
    if not m:
        raise SystemExit(f"--since must look like '30m' / '2h' / '7d', got {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    delta = {
        "m": _dt.timedelta(minutes=n),
        "h": _dt.timedelta(hours=n),
        "d": _dt.timedelta(days=n),
    }[unit]
    return (
        _dt.datetime.now(_dt.UTC) - delta
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_render_bootstrap(args: argparse.Namespace) -> int:
    from backend.app.services.render_client import RenderClient

    async def _run() -> list[dict]:
        client = RenderClient()
        return await client.list_owners()

    try:
        owners = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        _print_json(owners)
        return 0
    print("Render owners visible to this API key:\n")
    print(f"{'OWNER ID':<24} {'TYPE':<10} NAME")
    print("-" * 72)
    for o in owners:
        print(
            f"{o.get('id', ''):<24} {o.get('type', 'unknown'):<10} "
            f"{o.get('name', '?')}"
        )
    print("\nAdd the desired ID to .env as RENDER_OWNER_ID.")
    return 0


def _git_remote_url() -> str:
    """Read the 'origin' remote, normalize to https://github.com/.../foo."""
    import subprocess
    out = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if out.startswith("git@github.com:"):
        out = "https://github.com/" + out[len("git@github.com:"):]
    return out.removesuffix(".git")


def _git_current_branch() -> str:
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _read_env_file(path: str = ".env") -> dict[str, str]:
    """Minimal .env parser. Ignores comments + blank lines. Strips
    surrounding quotes from values. Doesn't expand variable references."""
    from pathlib import Path
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def _cmd_render_init(args: argparse.Namespace) -> int:
    """Create backend + frontend services on Render from render.yaml +
    .env, no dashboard click-through. Idempotent: skips services that
    already exist by name. Cross-links FRONTEND_ORIGIN <-> VITE_API_BASE_URL
    after both services have URLs."""
    import yaml
    from pathlib import Path
    from backend.app.services.render_client import RenderClient

    repo_root = Path(__file__).resolve().parents[3]
    rys = repo_root / "render.yaml"
    if not rys.exists():
        print(f"ERROR: {rys} not found.")
        return 1
    with rys.open() as f:
        blueprint = yaml.safe_load(f) or {}

    env_file = _read_env_file(str(repo_root / ".env"))
    required_secrets = ["DATABASE_URL", "OPENAI_API_KEY", "BEARER_TOKEN"]
    missing = [k for k in required_secrets if not env_file.get(k)]
    if missing:
        print(f"ERROR: missing required env vars in .env: {missing}")
        return 1

    try:
        repo_url = args.repo or _git_remote_url()
        branch = args.branch or _git_current_branch()
    except Exception as exc:
        print(f"ERROR: could not read git remote/branch: {exc}")
        return 1
    print(f"→ repo:   {repo_url}")
    print(f"→ branch: {branch}")

    async def _run() -> int:
        client = RenderClient()
        # Auto-discover owner if not set.
        if not client._owner_id:
            owners = await client.list_owners()
            if not owners:
                print("ERROR: no Render owners visible from this API key.")
                return 1
            if len(owners) > 1:
                print(
                    "Multiple Render owners visible. Set RENDER_OWNER_ID in "
                    ".env to disambiguate. Owners: "
                    + ", ".join(f"{o.get('name')} ({o.get('id')})" for o in owners)
                )
                return 1
            client._owner_id = owners[0]["id"]
            print(f"→ owner:  {owners[0].get('name')} ({client._owner_id})")

        existing = {
            s.get("name"): s for s in await client.list_services()
        }

        # Find the backend + frontend service defs from render.yaml.
        svc_defs: dict[str, dict] = {}
        for svc in blueprint.get("services", []):
            name = svc.get("name")
            if name in ("backend", "frontend"):
                svc_defs[name] = svc
        if "backend" not in svc_defs or "frontend" not in svc_defs:
            print("ERROR: render.yaml must declare both 'backend' and 'frontend' services.")
            return 1

        # --- 1. backend (or pick up existing) ---
        backend = existing.get("backend")
        if backend:
            print(f"= backend already exists: {backend.get('id')}")
        else:
            bdef = svc_defs["backend"]
            backend_env = [
                {"key": "DATABASE_URL",    "value": env_file["DATABASE_URL"]},
                {"key": "OPENAI_API_KEY",  "value": env_file["OPENAI_API_KEY"]},
                {"key": "BEARER_TOKEN",    "value": env_file["BEARER_TOKEN"]},
                {"key": "FRONTEND_ORIGIN", "value": "http://localhost:5173"},
                {"key": "LOG_LEVEL",       "value": "INFO"},
                {"key": "ENV",             "value": "production"},
            ]
            if env_file.get("GROQ_API_KEY"):
                backend_env.append(
                    {"key": "GROQ_API_KEY", "value": env_file["GROQ_API_KEY"]}
                )
            payload = {
                "type": "web_service",
                "name": "backend",
                "ownerId": client._owner_id,
                "repo": repo_url,
                "branch": branch,
                "autoDeploy": "yes",
                "rootDir": bdef.get("rootDir", "."),
                "envVars": backend_env,
                "serviceDetails": {
                    "env": "docker",
                    "plan": bdef.get("plan", "free"),
                    "region": bdef.get("region", "oregon"),
                    "healthCheckPath": bdef.get("healthCheckPath", "/health"),
                    # Docker-specific fields MUST nest under envSpecificDetails;
                    # the Render API silently ignores them at the top level and
                    # falls back to looking for ./Dockerfile in the repo root,
                    # which fails for our backend/Dockerfile layout.
                    "envSpecificDetails": {
                        "dockerfilePath": bdef.get(
                            "dockerfilePath", "./backend/Dockerfile"
                        ),
                        "dockerContext": ".",
                    },
                },
            }
            print("→ creating backend service...")
            backend = await client.create_service(payload)
            print(f"  created: id={backend.get('id')}")

        # Discover backend URL.
        backend_url = (
            backend.get("serviceDetails", {}).get("url")
            or f"https://{backend.get('name', 'backend')}.onrender.com"
        )
        print(f"→ backend URL: {backend_url}")

        # --- 2. frontend (or pick up existing) ---
        frontend = existing.get("frontend")
        if frontend:
            print(f"= frontend already exists: {frontend.get('id')}")
        else:
            fdef = svc_defs["frontend"]
            payload = {
                "type": "static_site",
                "name": "frontend",
                "ownerId": client._owner_id,
                "repo": repo_url,
                "branch": branch,
                "autoDeploy": "yes",
                "rootDir": fdef.get("rootDir", "frontend"),
                "envVars": [
                    {"key": "VITE_API_BASE_URL", "value": backend_url},
                ],
                "serviceDetails": {
                    "buildCommand": fdef.get("buildCommand", "npm ci && npm run build"),
                    "publishPath": fdef.get("staticPublishPath", "./dist"),
                    "pullRequestPreviewsEnabled": "no",
                },
            }
            print("→ creating frontend service...")
            frontend = await client.create_service(payload)
            print(f"  created: id={frontend.get('id')}")

        frontend_url = (
            frontend.get("serviceDetails", {}).get("url")
            or f"https://{frontend.get('name', 'frontend')}.onrender.com"
        )
        print(f"→ frontend URL: {frontend_url}")

        # --- 3. cross-link FRONTEND_ORIGIN on backend ---
        new_frontend_origin = f"http://localhost:5173,{frontend_url}"
        if existing.get("backend") is None or env_file.get(
            "FRONTEND_ORIGIN"
        ) != new_frontend_origin:
            print(
                f"→ updating backend FRONTEND_ORIGIN -> {new_frontend_origin}"
            )
            backend_env_full = [
                {"key": "DATABASE_URL",    "value": env_file["DATABASE_URL"]},
                {"key": "OPENAI_API_KEY",  "value": env_file["OPENAI_API_KEY"]},
                {"key": "BEARER_TOKEN",    "value": env_file["BEARER_TOKEN"]},
                {"key": "FRONTEND_ORIGIN", "value": new_frontend_origin},
                {"key": "LOG_LEVEL",       "value": "INFO"},
                {"key": "ENV",             "value": "production"},
            ]
            if env_file.get("GROQ_API_KEY"):
                backend_env_full.append(
                    {"key": "GROQ_API_KEY", "value": env_file["GROQ_API_KEY"]}
                )
            await client.update_env_vars(backend["id"], backend_env_full)

        # --- 4. optionally trigger first deploys ---
        if not args.no_deploy:
            print("→ triggering first deploys")
            try:
                await client.trigger_deploy(backend["id"])
                await client.trigger_deploy(frontend["id"])
            except Exception as exc:
                print(f"  (deploy trigger had a hiccup, ignore if auto-deploy is on: {exc})")

        print("\nDONE. Next steps:")
        print(f"  open {frontend_url} in a browser")
        print(f"  paste BEARER_TOKEN={env_file['BEARER_TOKEN'][:8]}... into /settings")
        print("  monitor build progress:")
        print("    uv run python -m backend.app.cli render-status")
        print("    uv run python -m backend.app.cli render-logs --service backend --since 10m")
        return 0

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1


def _cmd_render_status(args: argparse.Namespace) -> int:
    from backend.app.services.render_client import RenderClient

    async def _run() -> list[dict]:
        client = RenderClient()
        services = await client.list_services()
        if args.service != "all":
            services = [s for s in services if s.get("name") == args.service]
        return services

    try:
        services = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        _print_json(services)
        return 0
    if not services:
        print(f"No services found for --service {args.service!r}.")
        return 1
    print(f"{'NAME':<20} {'TYPE':<10} {'STATE':<14} URL")
    print("-" * 90)
    for s in services:
        state = s.get("suspended", "not_suspended")
        if state == "suspended":
            pretty = "suspended"
        else:
            pretty = s.get("serviceDetails", {}).get("buildStatus", "running")
        print(
            f"{s.get('name', '?'):<20} {s.get('type', '?'):<10} "
            f"{pretty:<14} {s.get('serviceDetails', {}).get('url', '')}"
        )
    return 0


def _cmd_render_deploy(args: argparse.Namespace) -> int:
    from backend.app.services.render_client import RenderClient

    async def _run() -> dict:
        client = RenderClient()
        svc = await client.resolve_service(args.service)
        sid = svc["id"]
        print(f"→ triggering deploy for {svc['name']} ({sid})")
        deploy = await client.trigger_deploy(sid, clear_cache=args.clear_cache)
        if not args.wait:
            return deploy
        deploy_id = deploy.get("id")
        if not deploy_id:
            print(f"WARN: deploy response missing id: {deploy}")
            return deploy
        last_status = ""
        import asyncio as _asy
        while True:
            d = await client.get_deploy(sid, deploy_id)
            status_ = d.get("status", "?")
            if status_ != last_status:
                print(f"  [{status_}]")
                last_status = status_
            if status_ in ("live", "deactivated", "build_failed",
                           "update_failed", "canceled"):
                return d
            await _asy.sleep(10)

    try:
        deploy = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        _print_json(deploy)
    else:
        print(
            f"\nDEPLOY ID:  {deploy.get('id', '?')}\n"
            f"STATUS:     {deploy.get('status', '?')}\n"
            f"COMMIT:     {(deploy.get('commit') or {}).get('id', '?')[:12]}"
        )
    return 0 if deploy.get("status") in ("live", "created", "build_in_progress",
                                          "update_in_progress", "queued") else 1


def _cmd_render_suspend(args: argparse.Namespace) -> int:
    return _render_lifecycle(args, action="suspend")


def _cmd_render_resume(args: argparse.Namespace) -> int:
    return _render_lifecycle(args, action="resume")


def _render_lifecycle(args: argparse.Namespace, *, action: str) -> int:
    """Shared driver for suspend/resume that accepts --service or --all."""
    from backend.app.services.render_client import RenderClient

    targets: list[str]
    if args.all:
        targets = list(_RENDER_PHASE3_SERVICES)
    elif args.service:
        targets = [args.service]
    else:
        print("ERROR: pass either --service NAME or --all.")
        return 1

    async def _run() -> list[tuple[str, str]]:
        client = RenderClient()
        results: list[tuple[str, str]] = []
        for name in targets:
            try:
                svc = await client.resolve_service(name)
                sid = svc["id"]
                if action == "suspend":
                    await client.suspend_service(sid)
                else:
                    await client.resume_service(sid)
                results.append((name, f"{action}ed"))
            except Exception as exc:
                results.append((name, f"ERROR: {exc}"))
        return results

    try:
        results = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    failed = False
    for name, msg in results:
        print(f"  {name:<20} {msg}")
        if msg.startswith("ERROR"):
            failed = True
    return 1 if failed else 0


def _cmd_render_logs(args: argparse.Namespace) -> int:
    from backend.app.services.render_client import RenderClient

    async def _run() -> dict:
        client = RenderClient()
        # The Render /v1/logs endpoint requires ownerId. If the user
        # didn't pin RENDER_OWNER_ID, auto-discover from the API key.
        if not client._owner_id:
            owners = await client.list_owners()
            if not owners:
                raise RuntimeError(
                    "no Render owners visible from this API key"
                )
            if len(owners) > 1:
                raise RuntimeError(
                    "multiple owners visible; pin RENDER_OWNER_ID in .env"
                )
            client._owner_id = owners[0]["id"]
        svc = await client.resolve_service(args.service)
        return await client.fetch_logs(
            svc["id"],
            start_time=_iso_minutes_ago(args.since),
            limit=args.limit,
        )

    try:
        data = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        _print_json(data)
        return 0
    logs = data.get("logs") if isinstance(data, dict) else []
    if not logs:
        print(f"(no log lines in the last {args.since} for {args.service})")
        return 0
    for entry in logs:
        ts = entry.get("timestamp", "")
        msg = entry.get("message", "")
        print(f"{ts}  {msg}")
    return 0


def _cmd_render_takedown(args: argparse.Namespace) -> int:
    """Suspend (default) or hard-delete the Phase-3 services. --yes required."""
    from backend.app.services.render_client import RenderClient

    if not args.yes:
        print(
            f"This will {'DELETE' if args.hard else 'SUSPEND'} the following "
            f"Render services: {', '.join(_RENDER_PHASE3_SERVICES)}.\n"
            "Re-run with --yes to confirm."
        )
        return 1

    async def _run() -> list[tuple[str, str]]:
        client = RenderClient()
        out: list[tuple[str, str]] = []
        for name in _RENDER_PHASE3_SERVICES:
            try:
                svc = await client.resolve_service(name)
                sid = svc["id"]
                if args.hard:
                    await client.delete_service(sid)
                    out.append((name, "deleted"))
                else:
                    await client.suspend_service(sid)
                    out.append((name, "suspended"))
            except Exception as exc:
                out.append((name, f"ERROR: {exc}"))
        return out

    try:
        results = asyncio.run(_run())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    failed = False
    for name, msg in results:
        print(f"  {name:<20} {msg}")
        if msg.startswith("ERROR"):
            failed = True
    if not args.hard:
        print(
            "\nTip: 'render-resume --all' brings them back when you're "
            "ready to use the app again."
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

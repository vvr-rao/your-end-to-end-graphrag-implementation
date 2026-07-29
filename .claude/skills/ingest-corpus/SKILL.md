---
name: ingest-corpus
description: >
  Load the prune-expand ontology into the database, then ingest the document
  corpus — chunk + embed, extract entities/relationships, enrich time, and
  generate intelligence artifacts. The paid, multi-hour steps run DETACHED so
  they survive a killed session, with --limit smoke-tests and a DB-size guardrail.
  Use after run-prune-expand, when building the knowledge graph from documents.
---

# Ingest the corpus into the knowledge graph

This is the corpus half of the build, symmetric with **run-prune-expand** (which
builds the ontology). It loads the ontology into Postgres, then runs the
document-ingestion pipeline. Several steps are **paid and multi-hour**, so they run
**detached** via the harness (`scripts/run_detached.py` + `scripts/job_status.py`)
and you monitor them — you never hold the process.

All commands use `uv run python …`, which works on Linux, macOS, and Windows.

## Preconditions
- `config/models.yaml` exists with keys present (**choose-llm-mode**).
- The database is set up and migrated (**setup-database**).
- A completed prune-expand version folder exists (**run-prune-expand**).

## Golden rules for the paid steps
- **Smoke-test first.** Run with `--limit 5` (or a small corpus), check the result
  and cost with the user, THEN scale to the full corpus. Never launch a full paid
  run without explicit go-ahead.
- **Watch DB size.** Before and after big steps: `uv run python -m backend.app.cli
  db-size`. Free-tier Postgres often caps around 500 MB — flag headroom as it grows.
- **Record each step** so the cross-session tracker can resume after a restart.

## Step 1 — Load the ontology into the DB
```bash
PE_DIR=$(uv run python scripts/latest_run_dir.py prune-expand)
uv run python -m backend.app.cli db-init --input "$PE_DIR"   # idempotent upsert
uv run python -m backend.app.cli db-status
uv run python scripts/build_state.py record db-init path="$PE_DIR"
```
(If the user hand-edited the ontology, re-merge first — see run-prune-expand
Step 1c — so `merged.json` reflects the edits before this import.)

## Step 2 — register-documents (chunk + embed; long, paid)
Confirm the corpus path (reuse the one from run-prune-expand if it's the same;
otherwise ask — never scan+pick). Discuss the options with the user:
- `--tables` — also extract structured tables from PDFs (retrievable rows/columns).
- `--full-text-chunks` — additionally store verbatim full-text chunks (better recall
  + exact citations) at the cost of DB size. Prereq for `--from-fulltext` later.

**Smoke-test** with `--limit 5`, review, then scale. Launch detached:
```bash
# choose a fresh RUN_ID, e.g. register_docs_<date+time>
uv run python scripts/run_detached.py <RUN_ID> \
  uv run python -m backend.app.cli register-documents --input "<DOCS>" [--tables] [--full-text-chunks]
uv run python scripts/job_status.py <RUN_ID> 40      # monitor
```
On completion, check `db-size`, then record (fill counts from the run output):
```bash
uv run python scripts/build_state.py record register-documents docs=<n> chunks=<n> [tables=<n>]
```

## Step 3 — extract-entities (entities + relationships; long, paid)
Mints entities/relationships per chunk — **no new ontology classes**. Add
`--from-fulltext` only if you ingested `--full-text-chunks` in Step 2.
```bash
uv run python scripts/run_detached.py <RUN_ID> \
  uv run python -m backend.app.cli extract-entities [--limit 5] [--from-fulltext] [--max-cost-usd <cap>]
uv run python scripts/job_status.py <RUN_ID> 40
uv run python scripts/build_state.py record extract-entities entities=<n>
```

## Step 4 — enrich-time (short)
Temporal enrichment (Year/Quarter/Month/Day, parent creation + gap-fill). Fast and
cheap — run in the foreground:
```bash
uv run python -m backend.app.cli enrich-time
uv run python scripts/build_state.py record enrich-time instances=<n>
```

## Step 5 — generate-artifacts (Claims/Findings/… ; long, paid)
Per-chunk `Claim`/`Finding`/`Observation`/`Event` + per-doc `Summary`. Opt-in
cross-cluster `Insight`/`Recommendation` via `--type` (gpt-4.1 — more expensive),
and `--rollup` for hierarchical consolidation. `--from-fulltext` if available.
```bash
uv run python scripts/run_detached.py <RUN_ID> \
  uv run python -m backend.app.cli generate-artifacts [--limit 5] [--type Insight --type Recommendation] [--rollup] [--from-fulltext] [--max-cost-usd <cap>]
uv run python scripts/job_status.py <RUN_ID> 40
uv run python -m backend.app.cli db-size
uv run python scripts/build_state.py record generate-artifacts artifacts=<n>
```

## Step 6 — Report + hand back
Report what's now queryable (documents, entities, artifacts), the final `db-size`,
and total cost. The corpus is now a knowledge graph — the user can `query` /
`conversation`, or proceed to the **deploy** skill (which needs a **cloud** DB).

## Notes
- **Scale ceiling:** at very large corpora both `extract-entities` and
  `generate-artifacts` currently fire all chunk LLM calls before committing, so
  memory grows with corpus size and a kill loses in-flight work. For a first build
  this is fine; for ~10M-token corpora it needs a batched/resumable driver.
- All long steps are detached, so a closed session never kills them; re-attach with
  `job_status.py` and the tracker tells you where you left off.

---
name: build-app
description: >
  Guide a user through building the whole GraphRAG app end-to-end after a git
  clone — pick LLM mode, set up the database, merge ontologies, run prune-expand,
  load the ontology, then (next) ingest documents and deploy. Orchestrates the
  other skills and the durable-run harness. Use when a user wants to "build",
  "set up", or "get started with" this app.
---

# Build the app, end to end

You are guiding a user who just cloned this repo to a working GraphRAG app,
running every step for them and monitoring the long ones. Be a calm operator:
do one step, confirm it worked, then move on. Never start a paid step until its
preconditions pass.

## Ground rules (always)
- **Credentials never touch the chat.** All API keys and the DB connection string
  go in `.env`, edited by the user. Never read `.env`, never ask for a secret in
  chat, only presence-check keys (see the choose-llm-mode skill).
- **Never auto-discover the ontology or the corpus.** When the user asks to build
  an ontology, SUGGEST a domain ontology (e.g. OCRe for pharma) and let them
  confirm — do not silently pick one. And ASK for the corpus folder path — never
  scan their existing folders/corpuses and choose one yourself. There is no
  default documents path.
- **Long steps run detached** via `scripts/run_detached.sh` and are monitored via
  `scripts/job_status.sh`, so a killed session never kills the work.
- **Bulk/paid runs are opt-in.** Smoke-test with `--limit` first where available,
  report cost, and get explicit go-ahead before a full corpus run.
- Watch the **500 MB** DB cap (`db-size` / db-status) as data grows.

## Starting from zero ("where do I start?" / "what do I do first?")
Assume the user knows nothing about this app. Don't dump the whole pipeline on
them. Orient in 2–3 sentences, then ask the ONE question that sets everything up:

> This tool turns a folder of your documents into a queryable knowledge graph you
> can ask questions against. To point it at the right vocabulary, tell me — what
> kind of documents are you working with?

Offer the supported domains and the fallback:
- **Pharma / clinical research** → uses the OCRe ontology (bundled)
- **Finance** → uses FIBO (downloaded live from EDM Council)
- **Manufacturing supply chain** → uses OntoCAPE (not bundled — you'll be asked to
  supply the zip, since RWTH gates its download)
- **Something else** → no domain ontology; we build with the core ontologies
  only (still fully works — you just don't get a domain-specific starting
  vocabulary)

Then walk the sequence below in order, one step at a time, confirming each before
moving on. Never race ahead to a paid step. If they ask what a step does or costs,
answer (or hand to **app-help**) before running it.

## The sequence

### 1. LLM mode + keys  →  invoke skill **choose-llm-mode**
Pick Groq+OpenAI / OpenAI-only / Anthropic+OpenAI, activate the preset, confirm
required keys are present in `.env`. (Every mode needs `OPENAI_API_KEY` for
embeddings.)

### 2. Database  →  invoke skill **setup-database**
Ask "have a Postgres DSN, or make a local one?" — warn that a local DB isn't
deploy-ready. Bring up docker Postgres if needed, then migrate + verify. Can run
concurrently with step 3 (prune-expand doesn't need the DB).

### 3. Merge + prune-expand  →  invoke skill **run-prune-expand**
**Suggest** a domain ontology and confirm (OCRe for pharma is the download-wired
suggestion today; FIBO/OntoCAPE vendored), **ask** for the corpus folder path,
then merge with VIAO + core and launch prune-expand **detached**, monitoring to
completion. This is the first multi-hour paid step. Capture the resulting
`v*-prune-expand/` folder.

### 4. Load the ontology into the DB
Once both the DB (step 2) and the prune-expand folder (step 3) are ready:
```bash
PE_DIR=$(ls -dt output_ontologies/v*-prune-expand/ | head -1)
uv run python -m backend.app.cli db-init --input "$PE_DIR"     # idempotent upsert
uv run python -m backend.app.cli db-status
```

### 5+. Corpus ingestion & deploy  →  (next — same detached pattern)
These reuse the exact Step-3 harness pattern (`run_detached.sh` + `job_status.sh`)
and are the next to be wired up. When you reach them, smoke-test with `--limit`
first, report, then scale on approval:
- `register-documents --documents <dir> [--tables]` — chunk + embed (long, paid).
- `extract-entities` — entities/relationships per chunk (long, paid).
- `enrich-time` — temporal enrichment (short).
- `generate-artifacts` — Claims/Findings/Insights/etc. (long, paid).
- `render-init` / `render-deploy --wait` / `render-status` — deploy (needs a
  Supabase DB, not local; walk the user through Render's manual bits).

For any of these today: run the CLI directly (or via the harness if long),
monitor, and surface errors — the orchestration pattern is identical to step 3.

## General questions
If the user asks what the app does, what a step costs, or how a tool works rather
than asking to build — invoke skill **app-help**.

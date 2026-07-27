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
- Watch database size (`db-size` / `db-status`) as data grows — free-tier Postgres
  plans often cap around 500 MB, so keep an eye on headroom.
- **Progress is tracked across sessions** in `.build/state.jsonl` (via
  `scripts/build_state.py`); each step records itself as it completes, and a
  SessionStart hook resurfaces it after a restart. If the user asks "what's the
  status / what did we do last / where were we", use the **build-status** skill.

## Starting from zero ("where do I start?" / "what do I do first?")
Assume the user knows nothing about this app. Don't dump the whole pipeline on
them. Orient in 2–3 sentences, then tell them the natural starting point:

> This tool turns a folder of your documents into a queryable knowledge graph. The
> best place to start is building the **ontology** — a MERGE — and then we run
> **prune-expand** against your documents to grow it. Let's start with the merge:
> where should the ontology come from?

Offer the four ontology sources (the **run-prune-expand** skill drives this — hand
off to it). Suggest based on what they tell you, and confirm; never pick silently:
- **A supported domain** — Pharma → OCRe (bundled), Finance → FIBO (downloaded
  live), Manufacturing/supply chain → OntoCAPE (not bundled; you'll be told how to
  supply it).
- **Another domain** → no domain ontology; merge the core ontologies only.
- **Their own ontology** → we accept `.owl` / `.rdf` (and `.ttl`/`.xml`/`.zip`);
  ask for a path and warn that very large ontologies are slow/memory-heavy.
- **Modify a previously generated ontology** → ask for the prior version folder
  (note the edit-and-resubmit gotcha the skill explains).

After the merge, give them the output location and ask them to verify `merged.owl`
in a third-party tool (Protégé) before the paid prune-expand. THEN ask for the
corpus, mention the `--tables` option and ask their preference, and run
prune-expand. Walk it one step at a time; never race ahead to a paid step.

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
Pick the ontology source (supported domain / other→core-only / their own .owl-.rdf
/ modify an existing folder), merge with the 7 core ontologies, then **report the
output and have them verify `merged.owl` in Protégé**. Next **ask** for the corpus
path, **mention the `--tables` flag and ask their preference**, then launch
prune-expand **detached**, monitoring to completion. First multi-hour paid step;
capture the resulting `v*-prune-expand/` folder. Note the edit-and-resubmit gotcha
(hand-edited `merged.owl` needs `--use-owl` or a re-merge, or edits are ignored).

### 4. Load the ontology into the DB
Once both the DB (step 2) and the prune-expand folder (step 3) are ready:
```bash
PE_DIR=$(ls -dt output_ontologies/v*-prune-expand/ | head -1)
uv run python -m backend.app.cli db-init --input "$PE_DIR"     # idempotent upsert
uv run python -m backend.app.cli db-status
```

### 5. Corpus ingestion  →  same detached pattern
These reuse the exact Step-3 harness pattern (`run_detached.sh` + `job_status.sh`).
Smoke-test with `--limit` first, check the result with the user, then scale up to
the full corpus. Run each via the harness (long) or directly (short), monitor, and
surface errors — the orchestration is identical to step 3:
- `register-documents --documents <dir> [--tables]` — chunk + embed (long, paid).
- `extract-entities` — entities/relationships per chunk (long, paid).
- `enrich-time` — temporal enrichment (short).
- `generate-artifacts` — Claims/Findings/Insights/etc. (long, paid).

### 6. Deploy to Render  →  invoke skill **deploy**
Guides the user through forking the repo, setting `RENDER_API_KEY` + generating a
`BEARER_TOKEN` in `.env`, `render-init` to create the backend + frontend services,
monitoring the build, and the final manual UI token paste. Requires a **cloud**
DB (not the local docker one). Records `deploy` (with URLs) to the tracker.

## General questions
If the user asks what the app does, what a step costs, or how a tool works rather
than asking to build — invoke skill **app-help**.

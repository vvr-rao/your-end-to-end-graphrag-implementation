---
name: app-help
description: >
  Answer general questions about this GraphRAG app — what it does, its pipeline
  stages, the CLI subcommands, the REST/MCP surface, costs, and constraints.
  Use when the user asks how something works or what's supported, rather than
  asking to build or run a step.
---

# App help — what this project is and how to use it

Answer from this overview and, when needed, by reading the code (the CLI parser
`backend/app/cli/main.py`, `README.md`, and `CLAUDE.local.md` are the best
sources). Prefer showing the real subcommand/flag over describing it.

## What it is
An end-to-end GraphRAG platform: ingest PDF/PPTX/DOCX/TXT, merge/curate an OWL
ontology, expand it via LLM analysis of the documents, persist everything in
Postgres + pgvector, and serve ontology-aware hybrid retrieval QA over REST, an
MCP server, and a React UI (deployable to Render). Postgres is external (Supabase)
in prod; local dev uses docker-compose.

## Supported domains
Out of the box the app targets three domains, each with a domain ontology, plus a
generic path:
- **Pharma / clinical research** → OCRe (bundled; BioPortal-gated upstream)
- **Finance** → FIBO (real live download from EDM Council; also bundled)
- **Manufacturing supply chain** → OntoCAPE (NOT bundled — GPL + RWTH form-gated;
  user supplies the zip when prompted)
- **Anything else** → core ontologies only (no domain ontology; still works)
Domain ontologies are fetched by `source_ontologies/fetch_ontology.py`; a fixed
set of 7 **core** ontologies (VIAO, FOAF, ORG, geography, time, SKOS,
domain_concepts) is ALWAYS merged in as well.

## The pipeline (order)
1. **Choose LLM mode** — Groq+OpenAI (default) / OpenAI-only / Anthropic+OpenAI;
   embeddings always OpenAI.
2. **Database** — Supabase (prod) or local docker Postgres; `db-init` migrates.
3. **Merge** ontologies (VIAO + 6 more core + a domain ontology) — deterministic, no LLM.
4. **prune-expand** — the paid, multi-hour ontology-building step (summarize →
   classify → propose → dedup → prune/extend). Caches summaries + tables on disk.
5. **db-init --input** — load the resulting ontology into the DB.
6. **register-documents** — chunk + embed the corpus.
7. **extract-entities** — mint entities + relationships per chunk (no new classes).
8. **enrich-time** — temporal enrichment (Year/Quarter/Month/Day).
9. **generate-artifacts** — Claims / Findings / Observations / Insights /
   Recommendations / Summaries (backed by VIAO).
10. **query / conversation** — 2 retrieval modes: `simple_qa` (tight factoid) and
    `deep_research` (structured 7-section answer). Full traceability answer →
    artifact → chunk → document.
11. **Deploy** — `render-init` / `render-deploy --wait` / `render-status`.

## CLI (single entrypoint)
`uv run python -m backend.app.cli <subcommand>`. Discover everything with
`uv run python -m backend.app.cli --help`. Groups: ontology (`merge`, `prune`,
`expand`, `prune-expand`, `build`), DB (`db-migrate`, `db-status`, `db-size`,
`db-init`, `clear-corpus`), corpus (`register-documents`, `list-documents`,
`update-document`, `delete-document`), enrichment (`extract-entities`,
`enrich-time`, `generate-artifacts`), retrieval (`query`, `evaluate-queries`,
`conversation start|turn|show`), deploy (`render-*`).

## Serving surfaces
- **REST** — FastAPI at `/`; `POST /qa`, `/conversations`, `/retrieval_runs/{id}`,
  browse routes. Bearer-token auth (`.env`).
- **MCP** — the same routes auto-exposed as MCP tools at `/mcp` (Streamable HTTP)
  via fastapi-mcp. Tool names come from each route's `operation_id`.
- **UI** — React + Vite (Phase 3), deployed alongside on Render.

## Costs & constraints (be upfront)
- Cheapest-viable-model per task; `gpt-4o-mini` default, escalate to `gpt-4.1`
  only for synthesis. prune-expand on a real corpus can run **hours** and cost
  tens of dollars — smoke-test first, report, then scale on approval.
- **500 MB** Supabase free-tier DB cap (`db-size` warns at 400 MB).
- Long steps run **detached** (`scripts/run_detached.sh`) so they survive a
  killed session; monitor with `scripts/job_status.sh`.
- Some steps have known caveats worth mentioning: garbled/unreadable PDFs are
  flagged by a preflight check before paid work; structured-table extraction is
  opt-in (`--tables`).

## How to help
- "What does X do / cost / support?" → answer here or from `--help`/README.
- "Build it / run step X" → hand off to **build-app** (or the specific skill).
- Keep credentials out of chat; never read `.env`.

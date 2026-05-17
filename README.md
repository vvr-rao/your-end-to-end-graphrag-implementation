# your-personal-knowledge-graph-creator

GraphRAG-based ontology and document management system. Ingests domain documents,
imports/merges OWL ontologies, expands them with LLMs, stores everything in
Postgres + pgvector, and answers questions with ontology-aware retrieval.

The same operations are exposed over **two parallel transports** so they can be
called by either an HTTP client or an LLM agent:

- **REST / OpenAPI** — `http://localhost:8000/docs`
- **MCP server** — `http://localhost:8000/mcp` (auto-generated from FastAPI routes via `fastapi-mcp`; both transports are served by a single uvicorn process)

A minimal React + Vite UI lives under `frontend/` and is deployed alongside the
backend on Render (see `render.yaml`).

## Status

**Phase 1 — ontology CLI.** Standalone command-line tool for ontology merge / prune / expand / build. Postgres persistence deferred to a later phase. See `CLAUDE.local.md` for the full project plan.

## CLI

Five subcommands. Each writes a fresh versioned folder under `output_ontologies/v<UTC-timestamp>-<subcommand>/` containing `merged.owl` (Protégé-readable), `merged.json` (canonical re-loadable form), `manifest.json` (provenance), `stats.json`, and `llm_audit.jsonl`.

```bash
# 1) merge — deterministic, ZERO LLM calls. Consolidate .owl/.rdf/.ttl/.zip inputs.
uv run python -m backend.app.cli merge \
  --ontology source_ontologies/pharma_ontologies/OCRe.zip \
  --ontology source_ontologies/general_ontologies/skos.rdf

# 2) prune — drop classes unsupported by your documents (LLM-driven).
uv run python -m backend.app.cli prune \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 3) expand — propose new classes/relationships from documents (LLM-driven).
uv run python -m backend.app.cli expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 4) prune-expand — both at once, more efficient (one LLM pass).
uv run python -m backend.app.cli prune-expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 5) build — merge + prune-expand end-to-end.
uv run python -m backend.app.cli build \
  --ontology source_ontologies/pharma_ontologies/OCRe.zip \
  --documents source_documents/
```

Optional flags: `--max-hops N`, `--max-cost-usd N`, `--dry-run`, `--output-dir DIR` (default `output_ontologies/`).

### Hand-suggesting additional classes (`--suggested-new-classes`)

The `expand`, `prune-expand`, and `build` subcommands accept an optional JSON file
of classes you want added in **addition** to whatever the LLM proposes from your
documents. The LLM is also told about them so it can avoid proposing near-duplicates;
any suggestions that don't already appear in `MATCH NOT FOUND` after dedup are
appended before the deterministic expansion step writes them out.

Copy the example template and edit:

```bash
cp suggested_new_classes.example.json suggested_new_classes.json
# Edit suggested_new_classes.json to list the classes you want added.
```

File format:

```json
[
  {
    "CLASS_TYPE": "Adverse Events",
    "CLASS_DESCRIPTION": "Adverse Events listed in a Drug or in a Study",
    "PARENT_CLASS_TYPE": "NONE"
  }
]
```

`PARENT_CLASS_TYPE` can be `"NONE"` (the class roots at the configured
`default_parent_iri`, typically `owl:Thing`) or the LABEL of another class in
the same file or the existing ontology.

Use it:

```bash
uv run python -m backend.app.cli expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/ \
  --suggested-new-classes suggested_new_classes.json
```

`suggested_new_classes.json` is gitignored; only the `*.example.json` template is tracked.

### How `merge` handles multi-file inputs

- A single `.owl`/`.rdf`/`.ttl`: parsed directly via owlready2.
- A `.zip` of many files: extracted to a temp directory; every `.owl`/`.rdf`/`.ttl` inside is enumerated. Cross-file `owl:imports` (including OntoCAPE's `file:/C:/...` Windows-style imports and XML `<!ENTITY>` references) are resolved to the local extracted copies via an IRI map. Only the **root** files (those not imported by anything in the set) are loaded; owlready2's import walker pulls in the rest in one pass per root — O(N) instead of O(N²).
- Imports that point to siblings the zip doesn't ship (e.g. OntoCAPE's reference to a separate `meta_model.owl` package) are stripped from the extracted copies so the rest of the load succeeds.
- All four sample ontology bundles work: SKOS, OCRe.zip, the OntoCAPE zip (slow but completes), the FIBO `prod.rdf.zip` (also slow), plus the giants (DRON 670MB, HP 73MB) which the `merge` command can parse but which we skip in the test suite.

## LLM providers

| Task | Provider | Default model |
|---|---|---|
| chunk_classification (document ingestion → existing ontology tagging) | Groq | `llama-3.3-70b-versatile` |
| class_proposal | OpenAI | `gpt-4.1` |
| match_dedup | OpenAI | `gpt-4.1` |
| class_summarization | OpenAI | `gpt-4o-mini` |
| qa_synthesis | OpenAI | `gpt-4.1` |
| embeddings | OpenAI | `text-embedding-3-small` (1536 dim) |

Both `OPENAI_API_KEY` and `GROQ_API_KEY` are required.

## First-run setup

```bash
# 1. Copy templates and fill in real values (OPENAI_API_KEY, GROQ_API_KEY,
#    and DATABASE_URL — paste your Supabase connection string as-is).
cp .env.example .env
cp config/config.example.yaml config/config.yaml
cp config/models.example.yaml config/models.yaml

# 2. Install Python dependencies
uv sync

# 3. Bring up Postgres + pgvector + Redis locally (skip Postgres if using Supabase)
docker compose up -d

# 4. Run the API + MCP server (single uvicorn process)
uv run uvicorn backend.app.main:app --reload
# → curl http://localhost:8000/health        (REST)
# → MCP discovery at http://localhost:8000/mcp

# 5. (Optional) Run the React UI
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Database URL

`DATABASE_URL` accepts the bare Supabase format directly:

```
postgresql://postgres:PASSWORD@db.<project>.supabase.co:5432/postgres
```

The app normalizes the scheme to `postgresql+asyncpg://` and appends
`?ssl=require` for `*.supabase.co` hosts internally. Supabase ships with
pgvector enabled, so no extra setup is required for the embedding tables.

## Helper scripts

### `source_ontologies/download_ontology.py`

Downloads industry-standard OWL ontologies (HPO, MAxO, ChEBI, OCRe, etc.) into
a local subfolder under `source_ontologies/`. Handy for seeding a domain
ontology before running an import.

The downloaded subfolder is **gitignored**; only the script itself is tracked.

```bash
# Example — Human Phenotype Ontology
uv run python source_ontologies/download_ontology.py \
  "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.owl" \
  "./source_ontologies/pharma_ontologies"
```

Optional flags:
- `--filename my.owl` — override the inferred filename
- `--extract` — auto-unzip if the URL serves a `.zip`

## Layout

```
backend/
  app/
    api/         # FastAPI routes (auto-exposed via MCP at /mcp)
    core/        # config, db, logging
    db/          # SQLAlchemy models + Alembic migrations
    helpers/     # ontology parsing + pruning helpers (relocated from repo root)
    jobs/        # arq workers
    ontology/    # OWL export, IRI utilities
    services/    # ontology I/O, persistence, embeddings, LLM router, retrieval
    cli/         # Phase 1 CLI: merge / prune / expand / prune-expand / build
  Dockerfile     # uv-based image used by Render web + worker services
frontend/              # Vite + React + TS + Tailwind UI (minimal stub; full build in Phase 3)
config/                # *.example.yaml tracked; *.yaml gitignored
source_ontologies/     # drop .owl source files here (subfolders gitignored, top-level files tracked)
source_documents/      # drop PDF/TXT documents here (everything except notes.md gitignored)
output_ontologies/     # versioned exports land here (subfolders gitignored)
render.yaml            # Render blueprint: backend web + worker + frontend static + managed Redis
scripts/
```

## Phase plan

- **Phase 1** (in progress) — standalone CLI: merge / prune / expand. Postgres persistence deferred.
- **Phase 2** — Postgres persistence + class summaries + GraphRAG QA API.
- **Phase 3** — React UI + graph viz + versioning + retirement + review workflow + Render deploy.

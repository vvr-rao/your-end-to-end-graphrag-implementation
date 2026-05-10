# your-personal-knowledge-graph-creator

GraphRAG-based ontology and document management system. Ingests domain documents,
imports/merges OWL ontologies, expands them with LLMs, stores everything in
Postgres + pgvector, and answers questions with ontology-aware retrieval.

The same operations are exposed over **two parallel transports** so they can be
called by either an HTTP client or an LLM agent:

- **REST / OpenAPI** — `http://localhost:8000/docs`
- **MCP server** — `http://localhost:8000/mcp` (auto-generated from FastAPI routes via `fastapi-mcp`; both transports are served by a single uvicorn process)

## Status

**Phase 0 — bootstrap.** See `CLAUDE.local.md` for the full project plan.

## First-run setup

```bash
# 1. Copy templates and fill in real values
cp .env.example .env
cp config/config.example.yaml config/config.yaml
cp config/models.example.yaml config/models.yaml

# 2. Install Python dependencies
uv sync

# 3. Bring up Postgres + pgvector + Redis locally
docker compose up -d

# 4. Run the API + MCP server (single uvicorn process)
uv run uvicorn backend.app.main:app --reload
# → curl http://localhost:8000/health        (REST)
# → MCP discovery at http://localhost:8000/mcp
```

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
backend/app/
  api/         # FastAPI routes (auto-exposed via MCP at /mcp)
  core/        # config, db, logging
  db/          # SQLAlchemy models + Alembic migrations
  helpers/     # ontology parsing + pruning helpers (relocated from repo root)
  jobs/        # arq workers
  ontology/    # OWL export, IRI utilities
  services/    # ontology I/O, persistence, embeddings, LLM router, retrieval
config/                # *.example.yaml tracked; *.yaml gitignored
source_ontologies/     # drop .owl source files here (subfolders gitignored, top-level files tracked)
output_ontologies/     # exports land here (subfolders gitignored)
scripts/
```

## Phase plan

- **Phase 1** — ontology import/export + Postgres schema + document ingestion + chunking + embeddings + ontology-to-chunk mapping.
- **Phase 2** — class summaries + GraphRAG QA API.
- **Phase 3** — React UI + graph viz + versioning + retirement + review workflow + Render deploy.

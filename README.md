# your-personal-ontologist

GraphRAG-based ontology and document management system. Ingests domain documents,
imports/merges OWL ontologies, expands them with LLMs, stores everything in
Postgres + pgvector, and answers questions with ontology-aware retrieval.

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

# 4. Run the API
uv run uvicorn backend.app.main:app --reload
# → curl http://localhost:8000/health
```

## Layout

```
backend/app/
  api/         # FastAPI routes
  core/        # config, db, logging
  db/          # SQLAlchemy models + Alembic migrations
  helpers/     # ontology parsing + pruning helpers (relocated from repo root)
  jobs/        # arq workers
  ontology/    # OWL export, IRI utilities
  services/    # ontology I/O, persistence, embeddings, LLM router, retrieval
config/        # *.example.yaml tracked; *.yaml gitignored
source_ontologies/   # drop .owl source files here (subfolders gitignored)
output_ontologies/   # exports land here (subfolders gitignored)
scripts/
```

## Phase plan

- **Phase 1** — ontology import/export + Postgres schema + document ingestion + chunking + embeddings + ontology-to-chunk mapping.
- **Phase 2** — class summaries + GraphRAG QA API.
- **Phase 3** — React UI + graph viz + versioning + retirement + review workflow + Render deploy.

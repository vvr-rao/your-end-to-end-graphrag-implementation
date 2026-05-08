"""
FastAPI entrypoint. Phase 0: only `/health` is wired. Phase 1+ adds routers
under `backend/app/api/` and includes them here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="your-personal-ontologist",
    version="0.0.1",
    description="GraphRAG-based ontology and document management system",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

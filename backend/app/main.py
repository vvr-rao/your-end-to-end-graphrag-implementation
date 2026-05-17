"""
FastAPI entrypoint with MCP server mounted in parallel.

Single source of truth: every API route declared on `app` is automatically
exposed to MCP-speaking agents at `/mcp` via fastapi-mcp. REST clients hit the
usual paths; agents (Claude Desktop, custom agents) connect to /mcp and call
the same operations as tools.

Phase 0: only /health is wired (excluded from MCP — not a useful tool).
Phase 1+ adds routers under backend/app/api/ and includes them here; they will
be picked up by MCP automatically.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP

from backend.app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="your-personal-knowledge-graph-creator",
    version="0.0.1",
    description=(
        "GraphRAG-based ontology and document management system. "
        "Same operations callable via REST or via MCP tools (mounted at /mcp)."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", operation_id="health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Phase 1: include API routers here. They'll be auto-exposed via MCP too.
# from backend.app.api import ontologies, documents, runs, qa
# app.include_router(ontologies.router)
# app.include_router(documents.router)
# ...

mcp = FastApiMCP(
    app,
    name="your-personal-knowledge-graph-creator",
    description="Ontology + document GraphRAG operations exposed as MCP tools.",
    exclude_operations=["health"],   # /health is not useful as an agent tool
)
mcp.mount_http()                     # MCP Streamable HTTP transport at /mcp

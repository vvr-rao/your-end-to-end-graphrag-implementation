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

from backend.app.api import browse, conversations as conv_routes, qa, trace
from backend.app.api.middleware.auth import BearerAuthMiddleware
from backend.app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="your-end-to-end-graphrag-implementation",
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
# Auth: every route except /health and the OpenAPI docs requires a
# bearer token matching settings.bearer_token.
app.add_middleware(BearerAuthMiddleware)


@app.get("/health", operation_id="health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Phase 2 / Milestone H: GraphRAG routes.
app.include_router(qa.router)
app.include_router(conv_routes.router)
app.include_router(trace.router)
app.include_router(browse.router)

mcp = FastApiMCP(
    app,
    name="your-end-to-end-graphrag-implementation",
    description="Ontology + document GraphRAG operations exposed as MCP tools.",
    exclude_operations=["health"],   # /health is not useful as an agent tool
)
mcp.mount_http()                     # MCP Streamable HTTP transport at /mcp

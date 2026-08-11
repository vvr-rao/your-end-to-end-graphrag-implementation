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
from urllib.parse import quote

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
    allow_origins=_settings.frontend_origins,
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

class _QuotingFastApiMCP(FastApiMCP):
    """Percent-encode path parameters before httpx parses the URL.

    fastapi-mcp 0.4.0 substitutes path params by raw string replacement
    (server.py:515: `path.replace(f"{{{name}}}", str(value))`) and hands the
    result straight to httpx. Every IRI in this system carries a '#'
    (…/entities#glp-1-93a560…), and in a URL a bare '#' starts the fragment,
    which is never transmitted. The server saw
    `/entities/https://veerla-ramrao.ai/ontology/entities` and correctly 404'd.

    That broke all six IRI-taking tools: document_get, entity_get, class_get,
    artifact_get, conversation_show, and conversation_turn -- the last
    returning 405 rather than 404 because the truncation also ate its trailing
    `/turns`, leaving `POST /conversations/{iri}`, which only accepts GET.

    safe="/:" keeps the path separators and the scheme colon intact, so
    Starlette's `{iri:path}` converter still matches and decodes the IRI
    whole. httpx preserves existing percent-encodings, so this does not
    double-encode.

    `_request` is private API; pyproject pins fastapi-mcp==0.4.0 and
    tests/unit/test_mcp_iri_routing.py fails loudly if this override stops
    being called.
    """

    async def _request(self, client, method, path, query, headers, body):  # type: ignore[override]
        return await super()._request(
            client, method, quote(path, safe="/:"), query, headers, body
        )


mcp = _QuotingFastApiMCP(
    app,
    name="your-end-to-end-graphrag-implementation",
    description="Ontology + document GraphRAG operations exposed as MCP tools.",
    exclude_operations=["health"],   # /health is not useful as an agent tool
)
mcp.mount_http()                     # MCP Streamable HTTP transport at /mcp

"""MCP tools receive whole IRIs, and conversation_turn exposes its body.

Two independent bugs, both in pinned fastapi-mcp 0.4.0:

1. server.py:515 interpolates path params by raw string replacement, then
   hands the result to httpx. A bare '#' starts a URL fragment and is never
   transmitted, so every IRI-taking tool received a truncated IRI and 404'd.
   conversation_turn returned 405 instead, because the truncation also ate
   its trailing '/turns'.

2. openapi/convert.py:181 only reads body properties from a schema with a
   top-level "properties" key. `TurnRequest | None` emits anyOf, which has
   none, so question/mode/top_k vanished from the tool schema.

These tests exist to fail loudly if a fastapi-mcp upgrade renames the private
`_request` hook the fix depends on.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx
import pytest

IRI = "https://veerla-ramrao.ai/ontology/entities#glp-1-93a560ce34387e93"


def test_bare_hash_in_a_path_is_lost_before_the_request_is_sent() -> None:
    """Demonstrates the upstream defect the override works around."""
    client = httpx.Client(base_url="http://apiserver")
    req = client.build_request("GET", f"/entities/{IRI}")
    # Everything from '#' onwards is treated as a fragment and dropped.
    assert req.url.path == "/entities/https://veerla-ramrao.ai/ontology/entities"
    assert "glp-1-93a560ce34387e93" not in req.url.path


def test_quoting_preserves_the_whole_iri_on_the_wire() -> None:
    client = httpx.Client(base_url="http://apiserver")
    raw = f"/entities/{IRI}"
    req = client.build_request("GET", quote(raw, safe="/:"))
    assert b"%23glp-1-93a560ce34387e93" in req.url.raw_path
    assert b"https://veerla-ramrao.ai/ontology/entities" in req.url.raw_path


def test_quoting_does_not_double_encode() -> None:
    """httpx preserves existing percent-encodings, so an already-safe path
    must survive unchanged rather than becoming %2523."""
    client = httpx.Client(base_url="http://apiserver")
    req = client.build_request("GET", quote(f"/entities/{IRI}", safe="/:"))
    assert b"%2523" not in req.url.raw_path


@pytest.mark.parametrize(
    "path",
    [
        "/documents/{iri}",
        "/entities/{iri}",
        "/classes/{iri}",
        "/artifacts/{iri}",
        "/conversations/{iri}",
        "/conversations/{iri}/turns",
    ],
)
def test_every_iri_route_shape_survives_quoting(path: str) -> None:
    substituted = path.replace("{iri}", IRI)  # what fastapi-mcp does
    quoted = quote(substituted, safe="/:")
    client = httpx.Client(base_url="http://apiserver")
    raw = client.build_request("GET", quoted).url.raw_path.decode()
    assert "%23" in raw, "the fragment marker must be encoded, not dropped"
    # The trailing segment must survive -- this is why conversation_turn 405'd.
    if path.endswith("/turns"):
        assert raw.endswith("/turns")


def test_override_is_wired_into_the_mounted_mcp_server() -> None:
    """If a fastapi-mcp upgrade renames _request, this fails."""
    from fastapi_mcp import FastApiMCP

    from backend.app.main import mcp

    assert type(mcp).__name__ == "_QuotingFastApiMCP"
    assert isinstance(mcp, FastApiMCP)
    # The override must actually shadow the base implementation.
    assert type(mcp)._request is not FastApiMCP._request


def test_conversation_turn_exposes_question_as_a_required_argument() -> None:
    from backend.app.main import app

    schema = app.openapi()
    op = schema["paths"]["/conversations/{iri}/turns"]["post"]
    body = op["requestBody"]
    assert body.get("required") is True, (
        "an optional body makes fastapi-mcp drop every field from the tool "
        "schema, leaving an agent unable to send a question"
    )
    ref = body["content"]["application/json"]["schema"]
    assert "$ref" in ref, "must be a plain $ref, not an anyOf with null"
    model = schema["components"]["schemas"]["TurnRequest"]
    assert "question" in model["properties"]
    assert "question" in model["required"]


def test_qa_ask_body_stays_required_too() -> None:
    from backend.app.main import app

    op = app.openapi()["paths"]["/qa"]["post"]
    assert op["requestBody"].get("required") is True

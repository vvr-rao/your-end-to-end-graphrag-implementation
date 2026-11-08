"""Test doubles for the structured-data downloaders (no network).

``FakeSession`` stands in for ``requests.Session``: each call is routed to a
handler ``(method, url, params, headers) -> FakeResponse`` and recorded, so
tests can assert on pagination, retry and header behaviour.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DOCS = REPO_ROOT / "source_documents"
FIXTURES = REPO_ROOT / "backend" / "tests" / "fixtures" / "structured"
if str(SOURCE_DOCS) not in sys.path:
    sys.path.insert(0, str(SOURCE_DOCS))


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes | str | Any = b"",
                 headers: dict[str, str] | None = None, chunks: list[bytes] | None = None,
                 fail_after: int | None = None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        self.content = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content)

    def iter_content(self, chunk_size: int = 1):
        chunks = self._chunks if self._chunks is not None else [
            self.content[i:i + chunk_size] for i in range(0, len(self.content), chunk_size)]
        for i, c in enumerate(chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise requests.ConnectionError("connection reset (simulated)")
            yield c

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, handler: Callable[..., FakeResponse]):
        self.handler = handler
        self.calls: list[dict] = []
        self.headers: dict[str, str] = {}

    def request(self, method, url, params=None, headers=None, json=None, stream=False,
                timeout=None):
        call = {"method": method, "url": url, "params": dict(params or {}),
                "headers": dict(headers or {}), "json": json}
        self.calls.append(call)
        result = self.handler(method, url, call["params"], call["headers"], json)
        if isinstance(result, Exception):
            raise result
        return result


def client_factory(session: FakeSession, sleeps: list[float] | None = None):
    """An ``execute(..., client_factory=...)`` hook using ``session``."""
    from structured._http import HttpClient

    def make(**kw):
        kw["log"] = lambda m: None
        return HttpClient(**kw, session=session,
                          sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
    return make


def parse(argv: list[str]):
    import structured_download
    return structured_download.build_parser().parse_args(argv)


def run_cli(argv: list[str], session: FakeSession) -> int:
    from structured._base import SOURCES, execute
    args = parse(argv)
    return execute(SOURCES[args.command], args.mode, args, client_factory=client_factory(session))


def load_manifest(root: Path) -> dict:
    found = list(root.rglob("manifest.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text(encoding="utf-8"))

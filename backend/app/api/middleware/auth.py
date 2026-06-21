"""Bearer-token auth middleware.

Single shared token from `settings.bearer_token`. Every request needs
an `Authorization: Bearer <token>` header except the paths in
`_PUBLIC_PATHS`.

Real per-user auth is deferred (single-tenant per spec).
"""
from __future__ import annotations

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from backend.app.core.config import get_settings


_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # CORS preflight requests must pass through unauthenticated --
        # browsers don't send auth headers on OPTIONS, so 401-ing the
        # preflight blocks the actual request from ever being sent.
        # The wrapped CORSMiddleware adds the allow-origin headers.
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        # Public paths bypass auth.
        if path in _PUBLIC_PATHS or path.startswith("/docs/"):
            return await call_next(request)
        # Middleware can't raise HTTPException because the global
        # exception handler doesn't run on raw ASGI middleware errors.
        # Return JSONResponse(401) directly.
        token = _extract_bearer(request)
        expected = get_settings().bearer_token
        if not token or token != expected:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid bearer token."},
                headers={"WWW-Authenticate": 'Bearer realm="api"'},
            )
        return await call_next(request)


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None

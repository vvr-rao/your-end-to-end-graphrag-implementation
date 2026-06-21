"""Thin async wrapper around the Render REST API (api.render.com/v1).

Used by the Phase 3 `render-*` CLI subcommands to deploy / suspend /
resume / take down the backend + frontend services declared in
`render.yaml`. Reads `RENDER_API_KEY` + `RENDER_OWNER_ID` from the
environment (settings.render_api_key + a one-off env lookup).

Service resolution: every helper accepts either a Render service ID
(`srv-abc123`) OR the human name from `render.yaml` (`backend`,
`frontend`). When given a name, we look it up via `list_services()`
and cache the result for the duration of the CLI invocation.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from backend.app.core.config import get_settings


_API_BASE = "https://api.render.com/v1"
_DEFAULT_TIMEOUT = 30.0


class RenderApiError(RuntimeError):
    """Raised when the Render API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Render API {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class RenderClient:
    """Async Render REST API client. Construct once per CLI invocation
    so the in-memory service-name → id cache survives between calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        owner_id: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.render_api_key or os.environ.get(
            "RENDER_API_KEY"
        )
        if not self._api_key:
            raise RuntimeError(
                "RENDER_API_KEY is not set. Add it to .env and re-run, or "
                "export it before invoking the CLI."
            )
        self._owner_id = owner_id or os.environ.get("RENDER_OWNER_ID")
        self._timeout = timeout
        self._service_cache: dict[str, dict[str, Any]] = {}  # name -> service dict

    # --- Low-level HTTP -----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{_API_BASE}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.request(
                method, url, headers=self._headers(), params=params, json=json,
            )
        if resp.status_code >= 300:
            raise RenderApiError(resp.status_code, resp.text)
        if not resp.content:
            return None
        return resp.json()

    # --- Owners -------------------------------------------------------------

    async def list_owners(self) -> list[dict[str, Any]]:
        """Return all owners (teams + personal accounts) the API key can see.

        Render wraps each item: [{"owner": {...}, "cursor": "..."}].
        We unwrap to a flat list of owner dicts.
        """
        data = await self._request("GET", "/owners")
        return [item.get("owner", item) for item in (data or [])]

    # --- Services -----------------------------------------------------------

    async def list_services(
        self, *, refresh: bool = False
    ) -> list[dict[str, Any]]:
        """List all services owned by `RENDER_OWNER_ID`. Caches by name."""
        if not refresh and self._service_cache:
            return list(self._service_cache.values())
        params: dict[str, Any] = {"limit": 100}
        if self._owner_id:
            params["ownerId"] = self._owner_id
        data = await self._request("GET", "/services", params=params)
        # Render wraps services the same way: [{"service": {...}, "cursor": ...}, ...]
        services = [item.get("service", item) for item in (data or [])]
        self._service_cache = {s.get("name", ""): s for s in services if s.get("name")}
        return services

    async def resolve_service(
        self, name_or_id: str
    ) -> dict[str, Any]:
        """Return the service dict for either a render.yaml name or a srv-* id."""
        if name_or_id.startswith("srv-"):
            data = await self._request("GET", f"/services/{name_or_id}")
            return data or {}
        services = await self.list_services()
        for svc in services:
            if svc.get("name") == name_or_id:
                return svc
        raise RuntimeError(
            f"No Render service named '{name_or_id}' found under owner. "
            f"Known: {sorted(self._service_cache.keys())}"
        )

    # --- Deploys ------------------------------------------------------------

    async def trigger_deploy(
        self,
        service_id: str,
        *,
        clear_cache: bool = False,
    ) -> dict[str, Any]:
        """Kick off a new deploy. Returns the deploy dict."""
        body: dict[str, Any] = {}
        if clear_cache:
            body["clearCache"] = "clear"
        return await self._request(
            "POST", f"/services/{service_id}/deploys", json=body,
        ) or {}

    async def get_deploy(
        self, service_id: str, deploy_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET", f"/services/{service_id}/deploys/{deploy_id}",
        ) or {}

    async def list_deploys(
        self, service_id: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        data = await self._request(
            "GET", f"/services/{service_id}/deploys",
            params={"limit": limit},
        )
        return [item.get("deploy", item) for item in (data or [])]

    # --- Lifecycle ----------------------------------------------------------

    async def suspend_service(self, service_id: str) -> None:
        await self._request("POST", f"/services/{service_id}/suspend")

    async def resume_service(self, service_id: str) -> None:
        await self._request("POST", f"/services/{service_id}/resume")

    async def delete_service(self, service_id: str) -> None:
        await self._request("DELETE", f"/services/{service_id}")

    # --- Create / mutate services ------------------------------------------

    async def create_service(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/services. `payload` follows Render's service-create
        schema (https://api-docs.render.com/reference/create-service).

        The wrapper accepts both wrapped (`{"service": {...}}`) and flat
        response shapes seen across API versions.
        """
        data = await self._request("POST", "/services", json=payload)
        if isinstance(data, dict) and "service" in data:
            return data["service"]
        return data or {}

    async def update_env_vars(
        self, service_id: str, env_vars: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """PUT /v1/services/{id}/env-vars — replaces the entire env-var
        set for the service. Each entry is {"key": "...", "value": "..."}
        or {"key": "...", "generateValue": "yes"}."""
        data = await self._request(
            "PUT", f"/services/{service_id}/env-vars", json=env_vars,
        )
        return data or []

    # --- Logs ---------------------------------------------------------------

    async def fetch_logs(
        self,
        service_id: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Fetch up to `limit` log lines via the Logs API.

        `start_time` / `end_time` are RFC3339 strings (e.g.
        "2026-06-20T00:00:00Z"). The wrapper at
        `_iso_minutes_ago` in the CLI handles the `--since 30m` parsing.
        """
        params: dict[str, Any] = {
            "resource": service_id,
            "limit": limit,
            "direction": "backward",
        }
        if self._owner_id:
            params["ownerId"] = self._owner_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("GET", "/logs", params=params) or {}

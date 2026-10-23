"""Is a DeepInfra DEDICATED deployment up right now?

A dedicated model id ("<username>/<deployment name>") only answers while its
deployment is running, and deployments are deleted after each build to stop
GPU billing. Tasks that can also run on a serverless model (`fallback_model`
in models.yaml, e.g. alias_judge) ask here first instead of waiting out a
404 through the router's retries.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

API = "https://api.deepinfra.com/deploy/list"
PLACEHOLDER = "__DEEPINFRA_FLASH_MODEL__"


def dedicated_status(model_id: str | None, *, timeout: float = 20.0) -> str:
    """'running', another deployment status ('deleted', 'failed',
    'deploying', ...), 'missing' (no such deployment), 'unset' (no id or the
    preset placeholder) or 'unknown' (no key / the API could not be asked).
    Never raises; never prints the key."""
    if not model_id or PLACEHOLDER in model_id:
        return "unset"
    from backend.app.core.config import get_settings

    key = get_settings().deepinfra_api_key
    if not key:
        return "unknown"
    req = urllib.request.Request(API, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rows = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return "unknown"
    rows = rows if isinstance(rows, list) else rows.get("deployments", [])
    did = model_id.split(":", 1)[1] if model_id.startswith("deploy_id:") else None
    mine = [r for r in rows if isinstance(r, dict)
            and (r.get("deploy_id") == did if did else r.get("model_name") == model_id)]
    if any(r.get("status") == "running" for r in mine):
        return "running"
    # Names are reused across attempts only by mistake; report the newest.
    mine.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""))
    return str(mine[-1].get("status") or "missing") if mine else "missing"

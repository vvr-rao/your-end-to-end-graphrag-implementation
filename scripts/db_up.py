#!/usr/bin/env python3
"""Bring up the local docker-compose Postgres and wait until it's healthy.
Cross-platform replacement for the bash `for i in $(seq ...); do docker inspect ...`
health-poll loop. Only used on the "local database" path.

    uv run python scripts/db_up.py

Docker itself is cross-platform; this just replaces the bash polling.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import ROOT  # noqa: E402

CONTAINER = "ontologist-postgres"
SERVICE = "postgres"
TIMEOUT_S = 90


def _health() -> str:
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", CONTAINER],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or "none"
    except Exception:
        return "none"


def main() -> int:
    print(f"[db-up] docker compose up -d {SERVICE}")
    up = subprocess.run(["docker", "compose", "up", "-d", SERVICE], cwd=str(ROOT))
    if up.returncode != 0:
        print("[db-up] `docker compose up` failed — is Docker Desktop running?", file=sys.stderr)
        return up.returncode

    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        status = _health()
        if status == "healthy":
            print("[db-up] Postgres is healthy.")
            return 0
        if status == "none":
            # No health field yet, or container not up — keep polling briefly.
            pass
        print(f"[db-up] waiting… ({status})")
        time.sleep(3)
    print(f"[db-up] timed out after {TIMEOUT_S}s waiting for '{CONTAINER}' to be healthy.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Top-level conftest — provides minimum env vars so Settings() loads in tests."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://x:y@localhost:5432/db")
os.environ.setdefault("BEARER_TOKEN", "test-bearer-token")

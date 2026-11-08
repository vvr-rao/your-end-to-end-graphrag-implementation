"""Unit-test setup: make the standalone downloader modules under
source_documents/ importable (they are scripts, not part of the package)."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_DOCS = Path(__file__).resolve().parents[3] / "source_documents"
if str(_SOURCE_DOCS) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DOCS))

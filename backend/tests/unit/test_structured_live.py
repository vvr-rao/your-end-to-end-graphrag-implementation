"""Opt-in live checks against the real public APIs (keyless sources only).

Run with:  YEGI_LIVE_TESTS=1 uv run pytest -m live backend/tests/unit/test_structured_live.py
Each test makes at most a handful of small requests (--limit 1-2) and never
downloads a bulk file.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("YEGI_LIVE_TESTS") != "1",
                       reason="set YEGI_LIVE_TESTS=1 to run live API tests"),
]

CASES = [
    ["clinicaltrials", "api", "--condition", "obesity", "--limit", "1", "--format", "csv"],
    ["drugsfda", "api", "--search", 'openfda.brand_name:"OZEMPIC"', "--limit", "1",
     "--format", "csv"],
    ["orangebook", "api", "--search", 'products.brand_name:"WEGOVY"', "--limit", "1"],
    ["gleif", "api", "--name", "Apple Inc", "--limit", "1", "--format", "csv"],
    ["chembl", "api", "--dataset", "molecule", "--chembl-id", "CHEMBL25", "--format", "csv"],
    ["opentargets", "api", "--dataset", "target", "--id", "ENSG00000169083", "--limit", "2"],
]


@pytest.mark.parametrize("argv", CASES, ids=[c[0] for c in CASES])
def test_live_bounded_sample(argv, tmp_path):
    import structured_download
    rc = structured_download.main([*argv, "-o", str(tmp_path), "--max-retries", "1"])
    assert rc == 0
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert m["status"] in ("complete", "sample")
    assert m["requests_made"] <= 6
    assert all(f["sha256"] and f["bytes"] > 0 for f in m["files"])

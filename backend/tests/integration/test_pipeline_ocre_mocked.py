"""Pipeline integration test against OCRe.zip with mocked LLM responses.

Verifies the full prune-expand wiring without making any real LLM calls.
The router is monkey-patched to return deterministic JSON for each task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.services import pipeline_llm
from backend.app.services.llm_router import ChatResult
from backend.app.services.pipeline import run_merge

REPO_ROOT = Path(__file__).resolve().parents[3]
OCRE_ZIP = REPO_ROOT / "source_ontologies" / "pharma_ontologies" / "OCRe.zip"


def _stub_chat_factory():
    """Return an async `chat` that picks a deterministic response per task."""

    async def stub_chat(self, task: str, *, system: str, user: str) -> ChatResult:
        if task == "chunk_classification":
            payload = {"relevant_iris": []}  # nothing relevant -> stages skip
        elif task == "class_proposal":
            payload = {
                "MATCHES FOUND": [],
                "MATCH NOT FOUND": [
                    {"LABEL": "MockedNewConcept", "DESCRIPTION": "Mock new class from documents."}
                ],
            }
        elif task == "match_dedup":
            payload = json.loads(user.split("INPUT:\n", 1)[1].split("\n\nReturn", 1)[0])
        else:
            payload = {}
        return ChatResult(
            text=json.dumps(payload),
            model="mock",
            provider="mock",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0,
        )

    return stub_chat


@pytest.mark.asyncio
@pytest.mark.skipif(not OCRE_ZIP.exists(), reason="OCRe.zip not available")
async def test_build_pipeline_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text(
        "Clinical trials use randomized controlled designs to evaluate intervention safety and efficacy.\n\n"
        "Adverse events must be tracked throughout the study period to ensure participant safety."
    )

    # Patch the router's chat method to deterministic responses.
    from backend.app.services.llm_router import LLMRouter

    monkeypatch.setattr(LLMRouter, "chat", _stub_chat_factory())

    out_root = tmp_path / "out"
    out_root.mkdir()

    # Step 1: deterministic merge (no LLM)
    merged_dir = run_merge(input_ontologies=[OCRE_ZIP], output_root=out_root)
    assert (merged_dir / "merged.owl").exists()

    # Step 2: prune-expand with mocked LLM
    result_dir = await pipeline_llm.prune_and_expand_async(
        input_folder=merged_dir,
        documents_dir=docs_dir,
        output_root=out_root,
        max_hops=1,
        max_cost_usd=10.0,
        dry_run=False,
    )

    assert (result_dir / "merged.owl").exists()
    assert (result_dir / "merged.json").exists()
    manifest = json.loads((result_dir / "manifest.json").read_text())
    assert manifest["operation"] == "prune-expand"
    assert manifest["parent_version"] == str(merged_dir.resolve())

    stats = json.loads((result_dir / "stats.json").read_text())
    assert "before" in stats and "after" in stats

    # The audit log should have at least the dedup record + chunk records.
    audit_lines = (result_dir / "llm_audit.jsonl").read_text().splitlines()
    assert len(audit_lines) >= 1

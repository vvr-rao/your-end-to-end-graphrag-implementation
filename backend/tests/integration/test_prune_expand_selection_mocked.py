"""prune-expand --select-subset wiring, with every LLM + embedding call mocked.

Two things must hold, and they pull in opposite directions:

  1. WITH --select-subset, the paid stages must see ONLY the selected
     documents. If selection writes a nice report but the pipeline goes on to
     classify all 348 documents anyway, the feature costs money and saves
     none.
  2. WITHOUT it, the document set must be byte-identical to before. This is
     the non-breaking guarantee the whole change rests on.

Also covers the preview stop: the default (no --yes) must write selection.json
and then spend nothing, because the run it precedes costs $20+.
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

# Three clearly-different "document types", 4 documents each. Near-duplicates
# within a type, so clustering has an obvious structure to find.
_CORPORA = {
    "fin": "Quarterly revenue rose. Operating margin and free cash flow improved. "
           "The balance sheet carries goodwill from prior acquisitions.",
    "drug": "Adverse reactions include nausea. Dosage and administration follow "
            "renal function. Contraindicated in hepatic impairment.",
    "news": "The minister said talks would continue next week. Analysts expect "
            "the dispute to affect shipping rates across the region.",
}


def _write_corpus(docs_dir: Path, per_type: int = 4) -> None:
    """Each document carries its own stem IN THE BODY.

    Stage 1 and the doc-type labeller both receive chunk/summary TEXT, never
    a filename, so a marker in the body is the only way a stub can tell which
    source document a given call came from.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    for kind, body in _CORPORA.items():
        for i in range(per_type):
            stem = f"{kind}_{i:02d}"
            (docs_dir / f"{stem}.txt").write_text(
                f"MARKER-{stem}\n\n{body}\n\nDocument {i} of the {kind} series.\n\n{body}"
            )


def _stub_chat_factory(classified: list[str]):
    """Deterministic per-task responses; records every classified chunk's
    source document so the test can assert which documents were paid for."""

    async def stub_chat(self, task: str, *, system: str, user: str, **kw) -> ChatResult:
        if task == "chunk_classification":
            classified.append(user)
            payload = {"relevant_iris": ["http://purl.org/net/OCRe/OCRe.owl#OCREEntity"]}
        elif task == "class_proposal":
            payload = {
                "MATCHES FOUND": [],
                "MATCH NOT FOUND": [
                    {"LABEL": "MockedConcept", "DESCRIPTION": "Mock class."},
                ],
                "MATCH NOT FOUND RELATIONS": [],
            }
        elif task == "match_dedup":
            payload = json.loads(user.split("INPUT:\n", 1)[1].split("\n\nReturn", 1)[0])
        elif task in ("document_type_label", "class_summarization",
                      "document_summarize"):
            # Selection resolves the cheapest configured task, so accept any
            # of the three it may land on. Serves BOTH the doc-type label and
            # the consolidation call.
            if "canonicalize" in system or "canonical form" in user:
                payload = {}  # identity mapping; consolidate_labels fills it in
            else:
                kind = next((k for k in _CORPORA if f"MARKER-{k}_" in user), "other")
                payload = {"doc_type": f"{kind} report", "region": "", "confidence": 0.9}
        else:
            payload = {}
        return ChatResult(
            text=json.dumps(payload), model="mock", provider="mock",
            prompt_tokens=10, completion_tokens=5, cost_usd=0.0,
        )

    return stub_chat


def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic vectors clustered by document type -- no OpenAI call.

    `__post_init__` is neutralised as well as `embed`. Embedder builds a real
    AsyncOpenAI client at CONSTRUCTION time, which raises without an API key
    even though this test never sends a request -- so stubbing only `embed`
    left a hidden dependency on the developer's .env that passed locally and
    failed in CI.
    """
    from backend.app.services.embeddings import Embedder

    monkeypatch.setattr(Embedder, "__post_init__", lambda self: None)

    async def fake_embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * 8
            for i, kind in enumerate(_CORPORA):
                if f"MARKER-{kind}_" in t:
                    vec[i] = 1.0
                    break
            else:
                vec[7] = 1.0
            out.append(vec)
        return out

    monkeypatch.setattr(Embedder, "embed", fake_embed)


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from backend.app.services import corpus_selection
    from backend.app.services.llm_router import LLMRouter

    classified: list[str] = []
    monkeypatch.setattr(LLMRouter, "chat", _stub_chat_factory(classified))
    _stub_embedder(monkeypatch)
    # Never touch the user's real profile cache from a test.
    monkeypatch.setattr(
        corpus_selection, "_profile_cache_dir", lambda: tmp_path / "profile_cache"
    )
    (tmp_path / "profile_cache").mkdir(parents=True, exist_ok=True)

    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)
    out_root = tmp_path / "out"
    out_root.mkdir()
    merged_dir = run_merge(input_ontologies=[OCRE_ZIP], output_root=out_root)
    return docs_dir, out_root, merged_dir, classified


def _docs_touched(classified: list[str]) -> set[str]:
    """Which source documents reached Stage 1, by filename stem."""
    touched = set()
    for chunk in classified:
        for kind in _CORPORA:
            for i in range(20):
                if f"MARKER-{kind}_{i:02d}" in chunk:
                    touched.add(f"{kind}_{i:02d}")
    return touched


@pytest.mark.asyncio
@pytest.mark.skipif(not OCRE_ZIP.exists(), reason="OCRe.zip not available")
async def test_preview_stops_before_the_paid_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir, out_root, merged_dir, classified = _prepare(tmp_path, monkeypatch)

    result_dir = await pipeline_llm.prune_and_expand_async(
        input_folder=merged_dir, documents_dir=docs_dir, output_root=out_root,
        max_hops=1, max_cost_usd=10.0, dry_run=False,
        select_subset=True, selection_yes=False,
    )

    report = json.loads((result_dir / "selection.json").read_text())
    assert report["n_documents"] == 12
    assert 0 < report["n_selected"] <= 12

    # The whole point of the preview: nothing downstream ran.
    assert classified == [], "Stage 1 fired during a preview-only run"
    assert not (result_dir / "merged.owl").exists()
    assert not (result_dir / "stats.json").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(not OCRE_ZIP.exists(), reason="OCRe.zip not available")
async def test_selected_subset_is_the_only_thing_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir, out_root, merged_dir, classified = _prepare(tmp_path, monkeypatch)

    result_dir = await pipeline_llm.prune_and_expand_async(
        input_folder=merged_dir, documents_dir=docs_dir, output_root=out_root,
        max_hops=1, max_cost_usd=10.0, dry_run=False,
        select_subset=True, selection_yes=True,
    )

    report = json.loads((result_dir / "selection.json").read_text())
    selected = {Path(d["path"]).stem for d in report["documents"] if d["selected"]}
    touched = _docs_touched(classified)

    assert touched, "no document reached Stage 1"
    assert touched <= selected, (
        f"paid to classify documents that were NOT selected: {touched - selected}"
    )
    assert len(selected) < 12, "selection reduced nothing"
    # The ontology was still built.
    assert (result_dir / "merged.owl").exists()
    assert (result_dir / "stats.json").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(not OCRE_ZIP.exists(), reason="OCRe.zip not available")
async def test_every_document_type_reaches_the_ontology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage guarantee, end to end: all three genres must contribute."""
    docs_dir, out_root, merged_dir, _ = _prepare(tmp_path, monkeypatch)

    result_dir = await pipeline_llm.prune_and_expand_async(
        input_folder=merged_dir, documents_dir=docs_dir, output_root=out_root,
        max_hops=1, max_cost_usd=10.0, dry_run=False,
        select_subset=True, selection_yes=True,
    )

    report = json.loads((result_dir / "selection.json").read_text())
    assert len(report["doc_types"]) == 3, report["doc_types"]
    for doc_type, row in report["doc_types"].items():
        assert row["selected"] >= 1, f"type {doc_type!r} contributed nothing"

    selected_kinds = {
        Path(d["path"]).stem.split("_")[0]
        for d in report["documents"] if d["selected"]
    }
    assert selected_kinds == set(_CORPORA)


@pytest.mark.asyncio
@pytest.mark.skipif(not OCRE_ZIP.exists(), reason="OCRe.zip not available")
async def test_without_the_flag_every_document_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-breaking guarantee. Default behaviour must be unchanged."""
    docs_dir, out_root, merged_dir, classified = _prepare(tmp_path, monkeypatch)

    result_dir = await pipeline_llm.prune_and_expand_async(
        input_folder=merged_dir, documents_dir=docs_dir, output_root=out_root,
        max_hops=1, max_cost_usd=10.0, dry_run=False,
    )

    assert not (result_dir / "selection.json").exists()
    assert _docs_touched(classified) == {
        f"{kind}_{i:02d}" for kind in _CORPORA for i in range(4)
    }
    assert (result_dir / "merged.owl").exists()

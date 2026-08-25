"""Corpus selection: clustering numerics + the three selection rules.

The rules exist to defend specific failure modes, and each test below names
the one it guards:

  - Cluster representatives must be stable run-to-run, or a preview and the
    run that confirms it would build different ontologies.
  - Type coverage is a HARD guarantee. A corpus with one EU drug label among
    500 US ones must still contribute EU regulatory vocabulary; clustering
    alone will never pick it.
  - Outliers are measured against the nearest SELECTED document, not the
    nearest centroid. Distance-to-centroid inverts on exactly the documents
    the rule exists to catch: a true outlier lands in its own singleton
    cluster where it IS the centroid and scores 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.services.corpus_selection import (
    DocProfile,
    SelectionConfig,
    _kmeans,
    _knee_index,
    _silhouette,
    choose_k,
    consolidate_labels,
    format_selection_summary,
    normalize_label,
    select_representatives,
    selection_report,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _blobs(n_clusters: int = 3, per: int = 20, dim: int = 16, spread: float = 0.3, seed: int = 0):
    """Well-separated gaussian blobs -- the easy case any k-means must pass."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_clusters):
        centre = np.zeros(dim)
        centre[c * (dim // max(n_clusters, 1))] = 10.0
        out.append(rng.normal(centre, spread, size=(per, dim)))
    return np.vstack(out).astype(np.float32)


def _profiles(mat, doc_types: list[str] | None = None) -> list[DocProfile]:
    types = doc_types or ["type-a"] * mat.shape[0]
    return [
        DocProfile(
            path=Path(f"/corpus/doc{i:03d}.txt"),
            doc_type=types[i],
            doc_type_raw="",
            vector=mat[i].tolist(),
        )
        for i in range(mat.shape[0])
    ]


class _StubRouter:
    """Duck-typed router (the house pattern -- see test_summarizer_concurrency)."""

    def __init__(self, text: str, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    async def chat(self, task, *, system, user, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")

        class _R:
            pass

        r = _R()
        r.text = self.text
        return r


# --------------------------------------------------------------------------- #
# k-means numerics
# --------------------------------------------------------------------------- #


def test_kmeans_recovers_well_separated_blobs() -> None:
    mat = _blobs(3, per=20)
    labels, _, _ = _kmeans(mat, 3, restarts=5, seed=42)
    # Each true blob must map onto exactly one predicted label.
    for c in range(3):
        block = labels[c * 20 : (c + 1) * 20]
        assert len(set(block.tolist())) == 1, "a true cluster was split"
    assert len(set(labels.tolist())) == 3, "two blobs collapsed onto one centre"


def test_kmeans_is_deterministic_for_a_fixed_seed() -> None:
    """A preview and the run confirming it must select the same documents."""
    mat = _blobs(4, per=15)
    a = _kmeans(mat, 4, restarts=5, seed=42)
    b = _kmeans(mat, 4, restarts=5, seed=42)
    assert np.array_equal(a[0], b[0])
    assert a[2] == pytest.approx(b[2])


def test_kmeans_handles_k_larger_than_n() -> None:
    mat = _blobs(1, per=3, dim=4)
    labels, centres, _ = _kmeans(mat, 10, restarts=2, seed=1)
    assert centres.shape[0] == 3  # clamped to n
    assert labels.shape[0] == 3


def test_choose_k_finds_the_true_cluster_count() -> None:
    mat = _blobs(3, per=20)
    k, inertia, silhouette = choose_k(
        mat, k_min=2, k_max=8, restarts=5, seed=42, sample_cap=2000
    )
    assert k == 3
    assert set(inertia) == set(silhouette) == set(range(2, 9))
    # Inertia must fall sharply at the true k -- that is what the knee detects.
    assert inertia[3] < inertia[2] * 0.5


def test_choose_k_degrades_sanely_on_a_single_blob() -> None:
    """No structure to find: must still return a valid k, not crash or hang."""
    rng = np.random.default_rng(3)
    mat = rng.normal(0.0, 1.0, size=(40, 8)).astype(np.float32)
    k, _, _ = choose_k(mat, k_min=2, k_max=6, restarts=3, seed=42, sample_cap=2000)
    assert 2 <= k <= 6


def test_knee_index_picks_the_elbow() -> None:
    ks = [2, 3, 4, 5, 6]
    # Sharp drop between k=2 and k=3, flat after.
    inertias = [1000.0, 100.0, 90.0, 85.0, 82.0]
    assert ks[_knee_index(ks, inertias)] == 3


def test_silhouette_subsamples_above_the_cap() -> None:
    """The full (n,n) matrix is ~800 MB at 10k docs on a 2.7 GB box."""
    mat = _blobs(3, per=500, dim=8, spread=0.5, seed=7)
    labels, _, _ = _kmeans(mat, 3, restarts=2, seed=1)
    score = _silhouette(mat, labels, sample_cap=200, seed=1)
    assert 0.5 < score <= 1.0  # well-separated blobs score high even subsampled


def test_silhouette_is_zero_without_two_clusters() -> None:
    mat = _blobs(1, per=10, dim=4)
    assert _silhouette(mat, np.zeros(10, dtype=np.int32)) == 0.0


# --------------------------------------------------------------------------- #
# Selection rules
# --------------------------------------------------------------------------- #


def test_type_coverage_is_guaranteed_for_a_singleton_type() -> None:
    """A type present in ONE document must survive, even buried inside a
    large cluster of another type where clustering would never surface it."""
    mat = _blobs(3, per=20)
    types = ["common"] * 60
    types[5] = "rare-eu-label"  # mid-blob, not an outlier -- clustering misses it
    profiles = _profiles(mat, types)

    result = select_representatives(profiles, cfg=SelectionConfig())

    rare = profiles[5]
    assert rare.path in result.reasons, "the only document of its type was dropped"
    assert any(r.startswith("type-coverage") for r in result.reasons[rare.path])


def test_outliers_are_kept_even_when_they_cluster_together() -> None:
    """Two mutually-close outliers far from everything else. The first tends
    to become its own cluster representative; the SECOND is the regression --
    under a distance-to-centroid rule it sits close to that new centroid and
    is silently dropped."""
    mat = _blobs(3, per=20)
    far1 = np.zeros((1, 16), dtype=np.float32)
    far1[0, 15] = 99.0
    far2 = far1.copy()
    far2[0, 15] = 97.0
    mat = np.vstack([mat, far1, far2])

    profiles = _profiles(mat, ["type-a"] * mat.shape[0])
    result = select_representatives(profiles, cfg=SelectionConfig())

    assert profiles[-1].path in result.reasons
    assert profiles[-2].path in result.reasons


def test_lower_outlier_sigma_keeps_more_documents() -> None:
    mat = _blobs(3, per=20, spread=1.5, seed=11)
    profiles = _profiles(mat)
    strict = select_representatives(profiles, cfg=SelectionConfig(outlier_sigma=3.0))
    loose = select_representatives(profiles, cfg=SelectionConfig(outlier_sigma=0.5))
    assert len(loose.selected) > len(strict.selected)


def test_selection_is_deterministic() -> None:
    mat = _blobs(4, per=12, seed=5)
    profiles = _profiles(mat)
    a = select_representatives(profiles, cfg=SelectionConfig())
    b = select_representatives(profiles, cfg=SelectionConfig())
    assert a.selected == b.selected
    assert a.k == b.k


def test_tiny_corpus_selects_everything() -> None:
    mat = _blobs(1, per=2, dim=4)
    profiles = _profiles(mat)
    result = select_representatives(profiles, cfg=SelectionConfig())
    assert len(result.selected) == 2
    assert result.k == 0
    assert "clustering skipped" in result.note


def test_empty_corpus_is_not_a_crash() -> None:
    result = select_representatives([], cfg=SelectionConfig())
    assert result.selected == []
    assert result.note


def test_every_selected_document_has_a_reason() -> None:
    mat = _blobs(3, per=20)
    profiles = _profiles(mat, ["a"] * 40 + ["b"] * 20)
    result = select_representatives(profiles, cfg=SelectionConfig())
    assert set(result.selected) == set(result.reasons)
    assert all(result.reasons[p] for p in result.selected)


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("doc_type", "region", "subject", "expected"),
    [
        ("Financial Report", "US", "", "financial report (US)"),
        ("  drug   label ", "EU", "", "drug label (EU)"),
        ("research paper", "", "oncology", "research paper (oncology)"),
        ("whitepaper", "", "", "whitepaper"),
        ("", "", "", "unclassified"),
        ("News Article.", "", "", "news article"),
    ],
)
def test_normalize_label(doc_type, region, subject, expected) -> None:
    assert normalize_label(doc_type, region, subject) == expected


async def test_consolidate_labels_merges_variants() -> None:
    router = _StubRouter(json.dumps({
        "U.S. financial report": "financial report (US)",
        "financial report (US)": "financial report (US)",
        "drug label (EU)": "drug label (EU)",
    }))
    mapping = await consolidate_labels(
        ["U.S. financial report", "financial report (US)", "drug label (EU)"], router
    )
    assert mapping["U.S. financial report"] == "financial report (US)"
    assert mapping["drug label (EU)"] == "drug label (EU)"


async def test_consolidate_labels_falls_back_to_identity_on_failure() -> None:
    """An un-consolidated label set over-selects slightly. Aborting the run
    over a cosmetic call would be far worse."""
    router = _StubRouter("", fail=True)
    labels = ["financial report (US)", "drug label (EU)"]
    assert await consolidate_labels(labels, router) == {label: label for label in labels}


async def test_consolidate_labels_never_drops_a_label() -> None:
    router = _StubRouter(json.dumps({"a": "canon-a"}))  # incomplete response
    mapping = await consolidate_labels(["a", "b", "c"], router)
    assert set(mapping) == {"a", "b", "c"}
    assert mapping["b"] == "b"


async def test_label_document_survives_non_json() -> None:
    from backend.app.services.corpus_selection import _label_document

    label, _ = await _label_document("text", _StubRouter("I think this is a 10-K."))
    assert label == "unclassified"


async def test_label_document_strips_markdown_fences() -> None:
    from backend.app.services.corpus_selection import _label_document

    fenced = '```json\n{"doc_type": "drug label", "region": "EU"}\n```'
    label, _ = await _label_document("text", _StubRouter(fenced))
    assert label == "drug label (EU)"


# --------------------------------------------------------------------------- #
# Task resolution
# --------------------------------------------------------------------------- #


class _SpecRouter:
    """Router exposing only task_spec, with a controllable task table."""

    def __init__(self, tasks: dict[str, dict]) -> None:
        self._tasks = tasks

    def task_spec(self, task: str) -> dict:
        if task not in self._tasks:
            raise KeyError(task)
        return self._tasks[task]


def test_label_task_prefers_the_dedicated_cheap_task() -> None:
    from backend.app.services.corpus_selection import resolve_label_task

    router = _SpecRouter({
        "document_type_label": {"model": "gpt-4o-mini", "max_tokens": 256},
        "class_summarization": {"model": "gpt-4o-mini", "max_tokens": 1024},
        "document_summarize": {"model": "gpt-4.1", "max_tokens": 8192},
    })
    assert resolve_label_task(router) == "document_type_label"


def test_label_task_falls_back_on_an_older_models_yaml() -> None:
    """A live deployment predating this feature must keep working."""
    from backend.app.services.corpus_selection import resolve_label_task

    router = _SpecRouter({
        "class_summarization": {"model": "gpt-4o-mini", "max_tokens": 1024},
        "document_summarize": {"model": "gpt-4.1", "max_tokens": 8192},
    })
    assert resolve_label_task(router) == "class_summarization"


def test_compress_task_refuses_a_task_that_cannot_emit_the_target() -> None:
    """THE REGRESSION. Compression was routed through document_type_label,
    whose max_tokens=256 silently truncated a 32,614-token summary down to
    260 tokens -- destroying the document vector it was meant to preserve.
    A task must be rejected on max_tokens, never assumed adequate."""
    from backend.app.services.corpus_selection import resolve_compress_task

    router = _SpecRouter({
        "embed_compress": {"model": "gpt-4o-mini", "max_tokens": 256},  # too small
        "document_summarize": {"model": "gpt-4.1", "max_tokens": 8192},
    })
    assert resolve_compress_task(router, 6000) == "document_summarize"


def test_compress_task_uses_the_cheap_task_when_it_is_big_enough() -> None:
    from backend.app.services.corpus_selection import resolve_compress_task

    router = _SpecRouter({
        "embed_compress": {"model": "gpt-4o-mini", "max_tokens": 8192},
        "document_summarize": {"model": "gpt-4.1", "max_tokens": 8192},
    })
    assert resolve_compress_task(router, 6000) == "embed_compress"


def test_shipped_presets_can_all_run_selection() -> None:
    """Every shipped preset must define both tasks, and embed_compress must be
    able to emit the configured target -- otherwise selection silently
    degrades on that provider mode."""
    import yaml

    repo = Path(__file__).resolve().parents[3]
    target = int(
        (yaml.safe_load((repo / "config" / "config.example.yaml").read_text())
         .get("corpus_selection") or {}).get("embed_target_tokens", 6000)
    )
    presets = list((repo / "config").glob("models*.example.yaml"))
    assert presets, "no model presets found"
    for preset in presets:
        tasks = yaml.safe_load(preset.read_text())["tasks"]
        assert "document_type_label" in tasks, f"{preset.name} lacks document_type_label"
        assert "embed_compress" in tasks, f"{preset.name} lacks embed_compress"
        cap = int(tasks["embed_compress"]["max_tokens"])
        assert cap >= target, (
            f"{preset.name}: embed_compress max_tokens={cap} < "
            f"embed_target_tokens={target}; compression would be truncated"
        )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_selection_report_shape_and_serializability() -> None:
    mat = _blobs(3, per=20)
    profiles = _profiles(mat, ["fin (US)"] * 40 + ["drug (EU)"] * 20)
    cfg = SelectionConfig()
    result = select_representatives(profiles, cfg=cfg)
    report = selection_report(result, Path("/corpus"), cfg)

    json.dumps(report)  # must round-trip to selection.json
    assert report["n_documents"] == 60
    assert report["n_selected"] == len(result.selected)
    assert set(report["doc_types"]) == {"fin (US)", "drug (EU)"}
    for row in report["doc_types"].values():
        assert row["selected"] >= 1, "every type must appear in the subset"
    doc = report["documents"][0]
    assert {"path", "name", "doc_type", "cluster", "selected", "reasons"} <= set(doc)
    assert sum(1 for d in report["documents"] if d["selected"]) == report["n_selected"]


def test_summary_renders_without_error() -> None:
    mat = _blobs(2, per=10)
    cfg = SelectionConfig()
    result = select_representatives(_profiles(mat), cfg=cfg)
    text = format_selection_summary(selection_report(result, Path("/corpus"), cfg))
    assert "CORPUS SELECTION" in text
    assert "PROJECTED prune-expand cost" in text

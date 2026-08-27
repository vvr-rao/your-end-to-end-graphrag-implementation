"""Pick a representative subset of a corpus for ontology building.

Why: prune-expand's Stage 2 (`class_proposal`) and Stage 3 (`match_dedup`) are
~95% of its cost and ~66% of its wall time, and both scale with chunk count,
which scales with document count. Measured on a 30-doc run: $15.77 of $23.06
went to Stage 2 alone. Building the ontology from a well-chosen subset cuts
both roughly proportionally.

What this does NOT do: reduce ingestion. `register-documents` still ingests
the whole corpus. Only the ONTOLOGY is built from the subset, and the document
summaries computed here are written to the shared `eval_summaries/` cache, so
the later full ingest reuses them for free.

Selection has three rules, applied in order and union'd:

  1. Cluster representatives -- k-means over document embeddings, then the
     document closest to each centroid.
  2. Type coverage -- every distinct document type contributes at least one
     document, even if clustering never picked one. This is a hard guarantee:
     US and EU financial filings use different regulatory vocabulary, and a
     subset missing one of them yields an ontology missing those classes.
  3. Outliers -- every document unusually far from the nearest ALREADY-SELECTED
     document is kept. Outliers are the documents least represented by anything
     else, so dropping them is exactly the wrong trade. Note the metric: the
     obvious "distance to its own centroid" inverts here, because a true
     outlier forms its own singleton cluster where it IS the centroid and
     scores zero -- the lowest value on the measure meant to flag it.

No reduction target is enforced. The caller previews the result and decides.

k-means is hand-rolled in numpy rather than pulled from scikit-learn: numpy is
already a dependency, the corpus sizes here are trivial for it, and the
precedent is `semantic_grouping._greedy_cluster`, which was hand-vectorised
for the same reason.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.services import document_io
from backend.app.services.document_io import LoadedDocument
from backend.app.services.embed_compress import compress_for_embed, count_embed_tokens
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter, _strip_json_fences
from backend.app.services.prompts import PROMPTS

# Bump when the profiling inputs change in a way that invalidates cached
# vectors or labels (prompt wording, embed model, compression target).
#   v4: region qualifiers leaked onto non-jurisdiction genres (whitepaper
#       (Taiwan) vs (US)) and annual reports split across two genres.
#   v3: compression was routed through a 256-token task, collapsing long
#       summaries to ~260 tokens; v2 vectors are unusable.
#   v2: tightened the doc-type prompts. v1 over-fragmented types --
#       "news article" and "news article (fertilizer industry)" survived as
#       separate types, and each spurious variant claimed its own
#       guaranteed coverage slot, inflating the subset.
PROFILE_VERSION = "v4"

_UNCLASSIFIED = "unclassified"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class SelectionConfig:
    """Resolved `corpus_selection` config. Read the house way -- plain dict
    out of app_config, no pydantic model (see backend/app/core/config.py)."""

    k_min: int = 2
    k_max: int = 40
    kmeans_restarts: int = 5
    random_seed: int = 42
    outlier_sigma: float = 2.0
    silhouette_sample_cap: int = 2000
    embed_target_tokens: int = 6000
    doc_type_labeling: bool = True
    use_profile_cache: bool = True
    label_concurrency: int = 8

    @classmethod
    def from_app_config(
        cls,
        app_cfg: dict[str, Any],
        *,
        k_max: int | None = None,
        outlier_sigma: float | None = None,
        label_concurrency: int | None = None,
    ) -> SelectionConfig:
        raw = (app_cfg.get("corpus_selection", {}) or {})
        cfg = cls(
            k_min=int(raw.get("k_min", 2)),
            k_max=int(raw.get("k_max", 40)),
            kmeans_restarts=int(raw.get("kmeans_restarts", 5)),
            random_seed=int(raw.get("random_seed", 42)),
            outlier_sigma=float(raw.get("outlier_sigma", 2.0)),
            silhouette_sample_cap=int(raw.get("silhouette_sample_cap", 2000)),
            embed_target_tokens=int(raw.get("embed_target_tokens", 6000)),
            doc_type_labeling=bool(raw.get("doc_type_labeling", True)),
            use_profile_cache=bool(raw.get("use_profile_cache", True)),
            label_concurrency=int(raw.get("label_concurrency", 8)),
        )
        # CLI overrides win over config.
        if k_max is not None:
            cfg.k_max = int(k_max)
        if outlier_sigma is not None:
            cfg.outlier_sigma = float(outlier_sigma)
        if label_concurrency is not None:
            cfg.label_concurrency = int(label_concurrency)
        return cfg


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class DocProfile:
    """One document reduced to what selection needs: a vector and a type."""

    path: Path
    doc_type: str
    doc_type_raw: str
    vector: list[float]
    tokens: int = 0
    compressed: bool = False

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class SelectionResult:
    profiles: list[DocProfile]
    selected: list[Path]
    k: int
    cluster_of: dict[Path, int] = field(default_factory=dict)
    distance_to_centroid: dict[Path, float] = field(default_factory=dict)
    # Distance to the nearest SELECTED document -- how well the subset covers
    # each document. 0 for the selected ones themselves. Drives outliers.
    distance_to_selected: dict[Path, float] = field(default_factory=dict)
    reasons: dict[Path, list[str]] = field(default_factory=dict)
    inertia_curve: dict[int, float] = field(default_factory=dict)
    silhouette_curve: dict[int, float] = field(default_factory=dict)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    note: str = ""

    @property
    def reduction_ratio(self) -> float:
        return (len(self.profiles) / len(self.selected)) if self.selected else 1.0


# --------------------------------------------------------------------------- #
# Profile cache
# --------------------------------------------------------------------------- #


def _profile_cache_dir() -> Path:
    root = (
        Path.home()
        / ".cache"
        / "your-end-to-end-graphrag-implementation"
        / "corpus_profiles"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _profile_cache_key(text: str, model: str, target_tokens: int) -> str:
    """Keyed on the ORIGINAL document text, so a cache entry survives changes
    to summarizer settings that don't change the document itself."""
    h = hashlib.sha256()
    for part in (PROFILE_VERSION, model, str(target_tokens), text):
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _profile_cache_load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("vector"):
        return None
    return data


def _profile_cache_save(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write -- a half-written cache file would poison every later run.

    Scoped by pid (matching the eval_summaries cache) so two concurrent runs
    cannot collide on the temp name; `id()` is reused within a process and
    means nothing across them.
    """
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Document-type labelling
# --------------------------------------------------------------------------- #


def _task_exists(router: LLMRouter, task: str) -> dict[str, Any] | None:
    try:
        return router.task_spec(task)
    except KeyError:
        return None


def resolve_label_task(router: LLMRouter) -> str:
    """Cheapest configured task for the per-document type label.

    Selection makes ONE call per document, so the model dominates its cost.
    Measured the hard way: routing through `document_summarize` -- which maps
    to gpt-4.1, NOT the gpt-4o-mini its name suggests -- made labelling 70% of
    a 10-document run ($0.55 of $0.79).

    Falls back through `class_summarization` (gpt-4o-mini / haiku in every
    shipped preset) so a deployment on an older models.yaml keeps working
    instead of dying on a KeyError. The output is a tiny JSON object, so a
    small max_tokens is correct here.
    """
    for task in ("document_type_label", "class_summarization", "document_summarize"):
        if _task_exists(router, task) is not None:
            return task
    return "document_summarize"


def resolve_compress_task(router: LLMRouter, target_tokens: int) -> str:
    """Cheapest configured task that can actually EMIT `target_tokens`.

    Unlike the label task, this one has a hard floor: the compressor must be
    able to write a summary of roughly `target_tokens`, so a task whose
    `max_tokens` is below that silently truncates the compression. That is the
    very data loss this whole feature exists to prevent, just moved one step
    earlier -- a 32,614-token summary once collapsed to 260 tokens by being
    routed through a 256-token task.

    Candidates are therefore filtered on max_tokens, never assumed.
    """
    for task in ("embed_compress", "document_summarize", "evaluated_summary_chunk"):
        spec = _task_exists(router, task)
        if spec is None:
            continue
        cap = int(spec.get("max_tokens") or 0)
        if cap >= target_tokens:
            return task
        print(
            f"[select] task '{task}' max_tokens={cap} < target {target_tokens}; "
            "skipping it for compression"
        )
    print(
        "[select] WARNING: no configured task can emit "
        f"{target_tokens} tokens; compression may be truncated"
    )
    return "document_summarize"


def normalize_label(doc_type: str, region: str = "", subject: str = "") -> str:
    """Fold the LLM's three fields into one canonical coverage key.

    Deterministic and cheap; the LLM consolidation pass afterwards handles
    what string normalization cannot (synonyms, word order).
    """
    base = re.sub(r"\s+", " ", (doc_type or "").strip().lower())
    base = base.strip(" .,;:-")
    if not base:
        return _UNCLASSIFIED
    qualifier = (region or "").strip() or (subject or "").strip()
    if qualifier:
        qualifier = re.sub(r"\s+", " ", qualifier).strip(" .,;:-")
        # "US" stays upper, "oncology" stays lower -- preserve the model's case
        # for short codes, which are conventionally capitalised.
        return f"{base} ({qualifier})"
    return base


async def _label_document(
    text: str, router: LLMRouter, task: str = "document_summarize"
) -> tuple[str, str]:
    """One cheap classification call. Returns (normalized, raw_json_label).

    Never raises: an unparseable or failed response yields `unclassified`
    rather than killing a run that has already paid for summarization.
    """
    system, user = PROMPTS["document_type_label"](text)
    try:
        result = await router.chat(task, system=system, user=user)
    except Exception as exc:
        print(f"[select] doc-type LLM call failed: {exc}")
        return _UNCLASSIFIED, ""
    raw = _strip_json_fences(result.text or "")
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError):
        print(f"[select] doc-type response was not JSON: {raw[:120]!r}")
        return _UNCLASSIFIED, raw[:200]
    label = normalize_label(
        str(payload.get("doc_type", "")),
        str(payload.get("region", "") or ""),
        str(payload.get("subject", "") or ""),
    )
    return label, label


async def consolidate_labels(
    labels: list[str], router: LLMRouter, task: str = "document_summarize"
) -> dict[str, str]:
    """One call over the DISTINCT labels, merging near-duplicates.

    Costs one call per corpus regardless of size. Falls back to identity on
    any failure -- an un-consolidated label set over-selects slightly, which
    is a far better failure than aborting.
    """
    distinct = sorted({label for label in labels if label and label != _UNCLASSIFIED})
    if len(distinct) < 2:
        return {label: label for label in distinct}

    system, user = PROMPTS["document_type_consolidate"](distinct)
    try:
        result = await router.chat(task, system=system, user=user)
        payload = json.loads(_strip_json_fences(result.text or ""))
        if not isinstance(payload, dict):
            raise ValueError("not an object")
    except Exception as exc:
        print(f"[select] label consolidation failed ({exc}); keeping raw labels")
        return {label: label for label in distinct}

    mapping: dict[str, str] = {}
    for label in distinct:
        canon = payload.get(label)
        mapping[label] = (
            str(canon).strip() if isinstance(canon, str) and canon.strip() else label
        )
    merged = len(distinct) - len(set(mapping.values()))
    if merged:
        print(f"[select] consolidated {len(distinct)} raw label(s) -> {len(set(mapping.values()))}")
    return mapping


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


async def profile_corpus(
    paths: list[Path],
    router: LLMRouter,
    embedder: Embedder,
    *,
    cfg: SelectionConfig,
    summarize: Any,
    batch_size: int = 16,
) -> tuple[list[DocProfile], list[tuple[Path, str]]]:
    """Summarize -> compress -> label -> embed every document.

    Walks in batches so peak memory stays flat on large corpora (the same
    reason `stream_evaluated_summarize_and_chunk_async` does).

    `summarize` is an async callable taking `list[LoadedDocument]` and
    returning objects with `.path` and `.combined` -- normally a partial over
    `evaluated_summarize_documents_async`, injected so tests need no LLM.

    Returns (profiles, skipped). Never raises for one bad document.
    """
    profiles: list[DocProfile] = []
    skipped: list[tuple[Path, str]] = []
    cache_dir = _profile_cache_dir() if cfg.use_profile_cache else None
    label_task = resolve_label_task(router)
    compress_task = resolve_compress_task(router, cfg.embed_target_tokens)
    print(
        f"[select] labelling on '{label_task}' "
        f"({router.task_spec(label_task).get('model')}); "
        f"compression on '{compress_task}' "
        f"({router.task_spec(compress_task).get('model')}, "
        f"max_tokens={router.task_spec(compress_task).get('max_tokens')})"
    )
    sem = asyncio.Semaphore(max(1, cfg.label_concurrency))
    n_cached = 0

    step = batch_size if batch_size > 0 else len(paths)
    total_batches = max(1, math.ceil(len(paths) / step)) if paths else 0

    for batch_idx in range(total_batches):
        batch_paths = paths[batch_idx * step : (batch_idx + 1) * step]
        loaded: list[LoadedDocument] = []
        for p in batch_paths:
            try:
                doc = document_io.load_document(p)
            except Exception as exc:
                skipped.append((p, f"unreadable: {exc}"))
                continue
            if not doc.text.strip():
                skipped.append((p, "empty after text extraction"))
                continue
            loaded.append(doc)
        if not loaded:
            continue

        # Cache first: a hit needs neither summarization nor an LLM label nor
        # an embedding call, which is what makes the confirm-after-preview
        # re-run free.
        pending: list[LoadedDocument] = []
        keys: dict[Path, str] = {}
        for doc in loaded:
            key = _profile_cache_key(doc.text, embedder.model, cfg.embed_target_tokens)
            keys[doc.path] = key
            hit = _profile_cache_load(cache_dir / f"{key}.json") if cache_dir else None
            if hit is not None:
                profiles.append(
                    DocProfile(
                        path=doc.path,
                        doc_type=str(hit.get("doc_type") or _UNCLASSIFIED),
                        doc_type_raw=str(hit.get("doc_type_raw") or ""),
                        vector=list(hit["vector"]),
                        tokens=int(hit.get("tokens") or 0),
                        compressed=bool(hit.get("compressed")),
                    )
                )
                n_cached += 1
            else:
                pending.append(doc)

        if pending:
            summaries = await summarize(pending)
            by_path = {s.path: s for s in summaries}

            async def _prepare(
                doc: LoadedDocument, _by_path=by_path
            ) -> tuple[LoadedDocument, str, str, str]:
                s = _by_path.get(doc.path)
                base = (getattr(s, "combined", "") or doc.text) if s else doc.text
                # The semaphore spans compression AND labelling, not just the
                # label call: compress_for_embed itself makes up to four LLM
                # calls for a long document, so gating only the label would let
                # a whole batch fire compressions unthrottled.
                async with sem:
                    compressed = await compress_for_embed(
                        base,
                        router,
                        target_tokens=cfg.embed_target_tokens,
                        log_prefix="select",
                        task=compress_task,
                    )
                    if cfg.doc_type_labeling:
                        label, raw = await _label_document(
                            compressed, router, label_task
                        )
                    else:
                        label, raw = _UNCLASSIFIED, ""
                return doc, compressed, label, raw

            prepared = await asyncio.gather(*[_prepare(d) for d in pending])
            vectors = await embedder.embed([c for _, c, _, _ in prepared])

            for (doc, compressed, label, raw), vec in zip(prepared, vectors, strict=True):
                tok = count_embed_tokens(compressed) or 0
                prof = DocProfile(
                    path=doc.path,
                    doc_type=label,
                    doc_type_raw=raw,
                    vector=list(vec),
                    tokens=tok,
                    compressed=compressed != doc.text,
                )
                profiles.append(prof)
                if cache_dir is not None:
                    _profile_cache_save(
                        cache_dir / f"{keys[doc.path]}.json",
                        {
                            "doc_type": prof.doc_type,
                            "doc_type_raw": prof.doc_type_raw,
                            "vector": prof.vector,
                            "tokens": prof.tokens,
                            "compressed": prof.compressed,
                        },
                    )

        print(
            f"[select] profiled batch {batch_idx + 1}/{total_batches}; "
            f"{len(profiles)} doc(s) so far ({n_cached} from cache)"
        )

    # One consolidation pass over the distinct labels, then rewrite.
    if cfg.doc_type_labeling and profiles:
        mapping = await consolidate_labels(
            [p.doc_type for p in profiles], router, label_task
        )
        for prof in profiles:
            prof.doc_type = mapping.get(prof.doc_type, prof.doc_type)

    # Deterministic order regardless of gather completion order.
    profiles.sort(key=lambda p: str(p.path))
    return profiles, skipped


# --------------------------------------------------------------------------- #
# Numerics -- hand-rolled k-means (numpy only)
# --------------------------------------------------------------------------- #


def _as_matrix(profiles: list[DocProfile]):
    import numpy as np

    return np.asarray([p.vector for p in profiles], dtype=np.float32)


def _sqdist(mat, centres):
    """Squared euclidean distances, shape (n, k), via the expansion
    ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2. Clipped at 0 because floating-point
    cancellation can otherwise produce small negatives that become NaN
    under sqrt."""
    import numpy as np

    d = (
        (mat * mat).sum(1)[:, None]
        - 2.0 * (mat @ centres.T)
        + (centres * centres).sum(1)[None, :]
    )
    return np.maximum(d, 0.0)


def _kmeans_plusplus(mat, k: int, rng):
    """k-means++ seeding: each new centre is drawn with probability
    proportional to its squared distance from the nearest chosen centre.
    Plain random seeding routinely collapses two centres onto one blob."""
    import numpy as np

    n = mat.shape[0]
    centres = np.empty((k, mat.shape[1]), dtype=mat.dtype)
    centres[0] = mat[rng.integers(n)]
    closest = _sqdist(mat, centres[:1]).ravel()
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # All remaining points coincide with a chosen centre.
            centres[i] = mat[rng.integers(n)]
        else:
            centres[i] = mat[int(rng.choice(n, p=closest / total))]
        closest = np.minimum(closest, _sqdist(mat, centres[i : i + 1]).ravel())
    return centres


def _kmeans(mat, k: int, *, restarts: int = 5, seed: int = 42, max_iter: int = 100):
    """Lloyd's algorithm, best-of-`restarts` by inertia.

    Deterministic for a given (X, k, seed). Returns (labels, centres, inertia).
    """
    import numpy as np

    n = mat.shape[0]
    k = max(1, min(k, n))
    best: tuple[Any, Any, float] | None = None

    for r in range(max(1, restarts)):
        rng = np.random.default_rng(seed + r)
        centres = _kmeans_plusplus(mat, k, rng)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            new_labels = _sqdist(mat, centres).argmin(1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = mat[labels == j]
                if members.shape[0]:
                    centres[j] = members.mean(0)
                else:
                    # Empty cluster: re-seed onto the point furthest from its
                    # own centre, rather than leaving a dead centre behind.
                    far = _sqdist(mat, centres).min(1).argmax()
                    centres[j] = mat[far]
        inertia = float(_sqdist(mat, centres)[np.arange(n), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, centres, inertia)

    assert best is not None
    return best


def _silhouette(mat, labels, *, sample_cap: int = 2000, seed: int = 42) -> float:
    """Mean silhouette score, subsampled above `sample_cap`.

    The full computation needs an (n, n) distance matrix -- at 10k documents
    that is 800 MB in float64, which the 2.7 GB dev box will not survive.
    Subsampling keeps the estimate honest at a bounded cost.
    """
    import numpy as np

    n = mat.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2 or n < 3:
        return 0.0

    if n > sample_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_cap, replace=False)
        mat, labels = mat[idx], labels[idx]
        n = sample_cap
        uniq = np.unique(labels)
        if uniq.size < 2:
            return 0.0

    dmat = np.sqrt(_sqdist(mat, mat))
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels == labels[i]
        own_n = int(own.sum())
        if own_n <= 1:
            scores[i] = 0.0  # a singleton cluster is neither well nor badly placed
            continue
        a = float((dmat[i][own].sum()) / (own_n - 1))  # excludes self (dmat[i,i] == 0)
        b = min(
            float(dmat[i][labels == c].mean())
            for c in uniq
            if c != labels[i] and int((labels == c).sum()) > 0
        )
        denom = max(a, b)
        scores[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(scores.mean())


def _knee_index(ks: list[int], inertias: list[float]) -> int:
    """Kneedle: the point of maximum distance from the chord joining the
    first and last points of the inertia curve, after normalising both axes
    to [0, 1] so the units cannot dominate."""
    import numpy as np

    if len(ks) < 3:
        return 0
    x = np.asarray(ks, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)
    x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0.0
    y = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0.0
    # Distance below the chord from (0,1)-ish start to end; for a decreasing
    # convex curve the knee maximises (chord - curve).
    chord = x * (y[-1] - y[0]) + y[0]
    return int(np.argmax(chord - y))


def choose_k(
    mat, *, k_min: int, k_max: int, restarts: int, seed: int, sample_cap: int
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Sweep k, pick the inertia knee, break ties on silhouette.

    Elbow alone is unreliable on text embeddings, where inertia often falls
    smoothly with no sharp knee and the argmax lands somewhere arbitrary.
    Silhouette is used only to choose among candidates whose inertia is
    within 5% of the knee's, so the elbow still drives the decision.
    """
    n = mat.shape[0]
    k_max = max(k_min, min(k_max, n - 1))
    ks = list(range(k_min, k_max + 1))
    if not ks:
        return 1, {}, {}

    inertia_curve: dict[int, float] = {}
    silhouette_curve: dict[int, float] = {}
    for k in ks:
        labels, _, inertia = _kmeans(mat, k, restarts=restarts, seed=seed)
        inertia_curve[k] = inertia
        silhouette_curve[k] = _silhouette(mat, labels, sample_cap=sample_cap, seed=seed)

    if len(ks) == 1:
        return ks[0], inertia_curve, silhouette_curve

    knee_i = _knee_index(ks, [inertia_curve[k] for k in ks])
    knee_k = ks[knee_i]
    knee_inertia = inertia_curve[knee_k]

    # Candidates within 5% of the knee's inertia; among them, best silhouette.
    tol = abs(knee_inertia) * 0.05
    candidates = [k for k in ks if abs(inertia_curve[k] - knee_inertia) <= tol]
    if not candidates:
        candidates = [knee_k]
    best_k = max(candidates, key=lambda k: (silhouette_curve[k], -k))
    return best_k, inertia_curve, silhouette_curve


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_representatives(
    profiles: list[DocProfile],
    *,
    cfg: SelectionConfig,
    skipped: list[tuple[Path, str]] | None = None,
) -> SelectionResult:
    """Cluster representatives + full type coverage + all outliers."""
    import numpy as np

    skipped = list(skipped or [])
    if not profiles:
        return SelectionResult(
            profiles=[], selected=[], k=0, skipped=skipped,
            note="no profilable documents",
        )

    n = len(profiles)
    # Too few documents to cluster meaningfully -- take everything and say so.
    if n <= max(cfg.k_min, 2):
        return SelectionResult(
            profiles=profiles,
            selected=[p.path for p in profiles],
            k=0,
            reasons={p.path: ["all-documents"] for p in profiles},
            skipped=skipped,
            note=f"only {n} document(s); clustering skipped, all selected",
        )

    mat = _as_matrix(profiles)
    # Cap k against the corpus: asking for 40 clusters over 50 documents makes
    # "representatives" meaningless. 2*sqrt(n) is the usual rule of thumb.
    k_ceiling = max(cfg.k_min, min(cfg.k_max, n - 1, math.ceil(math.sqrt(n)) * 2))
    k, inertia_curve, silhouette_curve = choose_k(
        mat,
        k_min=cfg.k_min,
        k_max=k_ceiling,
        restarts=cfg.kmeans_restarts,
        seed=cfg.random_seed,
        sample_cap=cfg.silhouette_sample_cap,
    )
    labels, centres, _ = _kmeans(
        mat, k, restarts=cfg.kmeans_restarts, seed=cfg.random_seed
    )

    dists = np.sqrt(_sqdist(mat, centres))
    own_dist = dists[np.arange(n), labels]

    cluster_of = {profiles[i].path: int(labels[i]) for i in range(n)}
    distance_to_centroid = {profiles[i].path: float(own_dist[i]) for i in range(n)}
    reasons: dict[Path, list[str]] = {}

    def _mark(idx: int, reason: str) -> None:
        reasons.setdefault(profiles[idx].path, [])
        if reason not in reasons[profiles[idx].path]:
            reasons[profiles[idx].path].append(reason)

    # 1. Cluster representatives -- closest document to each centroid.
    for j in range(k):
        members = np.flatnonzero(labels == j)
        if members.size == 0:
            continue
        _mark(int(members[own_dist[members].argmin()]), "cluster-representative")

    # 2. Type coverage -- a hard guarantee, one document per distinct type.
    by_type: dict[str, list[int]] = {}
    for i, prof in enumerate(profiles):
        by_type.setdefault(prof.doc_type or _UNCLASSIFIED, []).append(i)
    for doc_type, idxs in sorted(by_type.items()):
        if any(profiles[i].path in reasons for i in idxs):
            continue
        # The most typical document OF THIS TYPE: closest to its own centroid.
        type_centre = mat[idxs].mean(0, keepdims=True)
        best = int(idxs[int(_sqdist(mat[idxs], type_centre).ravel().argmin())])
        _mark(best, f"type-coverage:{doc_type}")

    # 3. Outliers -- documents the subset chosen so far fails to represent.
    #
    # Measured against the nearest SELECTED document, NOT the nearest centroid.
    # Distance-to-own-centroid inverts on exactly the documents this rule
    # exists to catch: a true outlier usually lands in its own singleton
    # cluster, where it IS the centroid and scores 0 -- the lowest possible
    # value on the metric meant to flag it. "How far is my nearest surviving
    # proxy?" is the question that actually matters, and it has no such
    # blind spot.
    chosen_idx = [i for i in range(n) if profiles[i].path in reasons]
    remaining = [i for i in range(n) if profiles[i].path not in reasons]
    dist_to_selected: dict[Path, float] = {profiles[i].path: 0.0 for i in chosen_idx}
    if remaining and chosen_idx:
        gap = np.sqrt(_sqdist(mat[remaining], mat[chosen_idx])).min(1)
        for pos, i in enumerate(remaining):
            dist_to_selected[profiles[i].path] = float(gap[pos])
        mean_g = float(gap.mean())
        std_g = float(gap.std())
        threshold = mean_g + cfg.outlier_sigma * std_g
        if std_g > 0:
            for pos in np.flatnonzero(gap > threshold):
                _mark(int(remaining[int(pos)]), "outlier")

    selected = sorted((p for p in reasons), key=str)
    return SelectionResult(
        profiles=profiles,
        selected=selected,
        k=k,
        cluster_of=cluster_of,
        distance_to_centroid=distance_to_centroid,
        reasons=reasons,
        inertia_curve=inertia_curve,
        silhouette_curve=silhouette_curve,
        distance_to_selected=dist_to_selected,
        skipped=skipped,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

# Measured on output_ontologies/v20260819T145736Z-prune-expand: $23.06 over
# 212 chunks. A rough per-chunk rate is enough to tell $2 from $200 before
# the user commits, which is the decision the preview exists to support.
USD_PER_CHUNK = 23.064504 / 212


def selection_report(result: SelectionResult, corpus_dir: Path, cfg: SelectionConfig) -> dict[str, Any]:
    """The `selection.json` payload."""
    type_counts: dict[str, dict[str, int]] = {}
    for prof in result.profiles:
        row = type_counts.setdefault(prof.doc_type, {"total": 0, "selected": 0})
        row["total"] += 1
        if prof.path in result.reasons:
            row["selected"] += 1

    return {
        "mode": "representative-subset",
        "profile_version": PROFILE_VERSION,
        "corpus_dir": str(corpus_dir),
        "n_documents": len(result.profiles),
        "n_selected": len(result.selected),
        "reduction_ratio": round(result.reduction_ratio, 2),
        "k": result.k,
        "outlier_sigma": cfg.outlier_sigma,
        "note": result.note,
        "inertia_curve": {str(k): round(v, 4) for k, v in sorted(result.inertia_curve.items())},
        "silhouette_curve": {
            str(k): round(v, 4) for k, v in sorted(result.silhouette_curve.items())
        },
        "doc_types": dict(sorted(type_counts.items())),
        "skipped": [{"path": str(p), "reason": r} for p, r in result.skipped],
        "documents": [
            {
                "path": str(prof.path),
                "name": prof.name,
                "doc_type": prof.doc_type,
                "cluster": result.cluster_of.get(prof.path, -1),
                "distance": round(result.distance_to_centroid.get(prof.path, 0.0), 4),
                "distance_to_nearest_selected": round(
                    result.distance_to_selected.get(prof.path, 0.0), 4
                ),
                "selected": prof.path in result.reasons,
                "reasons": result.reasons.get(prof.path, []),
                "embed_tokens": prof.tokens,
                "compressed": prof.compressed,
            }
            for prof in result.profiles
        ],
    }


def format_selection_summary(report: dict[str, Any]) -> str:
    """Human-readable preview -- what the user approves or rejects."""
    lines: list[str] = []
    n_docs = report["n_documents"]
    n_sel = report["n_selected"]
    lines.append("")
    lines.append("=" * 66)
    lines.append("CORPUS SELECTION — representative subset")
    lines.append("=" * 66)
    if report.get("note"):
        lines.append(f"note: {report['note']}")
    lines.append(
        f"documents: {n_sel} selected of {n_docs}  "
        f"(reduction {report['reduction_ratio']}x)   clusters k={report['k']}"
    )
    if report["skipped"]:
        lines.append(f"skipped:   {len(report['skipped'])} unreadable/empty document(s)")

    lines.append("")
    lines.append(f"{'document type':<42}{'selected':>10}{'total':>8}")
    lines.append("-" * 66)
    for doc_type, row in report["doc_types"].items():
        lines.append(f"{doc_type[:41]:<42}{row['selected']:>10}{row['total']:>8}")

    by_reason: dict[str, int] = {}
    for doc in report["documents"]:
        for reason in doc["reasons"]:
            key = reason.split(":", 1)[0]
            by_reason[key] = by_reason.get(key, 0) + 1
    if by_reason:
        lines.append("")
        lines.append("selected because:")
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:<28} {count}")

    lines.append("")
    lines.append("PROJECTED prune-expand cost on this subset:")
    lines.append(
        f"  ~${n_sel * 7.07 * USD_PER_CHUNK:,.2f}  "
        f"(vs ~${n_docs * 7.07 * USD_PER_CHUNK:,.2f} for the full corpus)"
    )
    lines.append(
        "  Rough: 7.07 chunks/doc and $0.109/chunk, measured on a 30-doc run. "
        "Chunk count varies with document length."
    )
    lines.append("=" * 66)
    return "\n".join(lines)

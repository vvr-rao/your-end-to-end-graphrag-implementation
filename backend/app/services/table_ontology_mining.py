"""Phase 2a follow-up: classify extracted tables into the 6 anchor
buckets in `domain_concepts.owl` and emit Stage-2-shaped class
proposals for `match_dedup` (Stage 3).

Two-layer matching (user-locked 2026-06-15):
    1. MATCH against existing ontology classes first.  Each per-table
       candidate (table-class OR column-class) gets fuzzy-matched
       against `loaded_ontology['classes_dict']` labels.  When a
       similar-enough match exists, the candidate is REUSED (emitted
       as a `MATCHES FOUND` entry against the existing IRI), not
       proposed anew.
    2. GROUP across tables (collapse duplicates).  Surviving
       candidates from all tables get normalised
       `(label, parent_iri)` keys; per group, ONE canonical proposal
       is emitted as a `MATCH NOT FOUND` entry.

The output shape (`MATCHES FOUND` + `MATCH NOT FOUND` lists) plugs
directly into `_run_llm_stages`' Stage-3 `match_dedup` pass via the
existing recursive-merge path -- no new merge code is needed.

Runs only during `prune-expand --tables`.  Not part of the Phase 2
`register-documents` ingest.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS


# The 6 anchor bucket IRIs in domain_concepts.owl (added in Phase A).
ANCHOR_BUCKETS: list[dict[str, str]] = [
    {
        "iri": "https://veerla-ramrao.ai/ontology/domain-concepts#FinancialTable",
        "label": "FinancialTable",
        "description": (
            "A table of financial values extracted from a source document; "
            "parent of newly-minted per-table-type subclasses."
        ),
    },
    {
        "iri": (
            "https://veerla-ramrao.ai/ontology/domain-concepts#"
            "FinancialObservation"
        ),
        "label": "FinancialObservation",
        "description": "A factual reading from a row+column intersection.",
    },
    {
        "iri": "https://veerla-ramrao.ai/ontology/domain-concepts#Metric",
        "label": "Metric",
        "description": (
            "A derived numerical value (computed from measures); e.g. Net "
            "Margin %, ROE."
        ),
    },
    {
        "iri": "https://veerla-ramrao.ai/ontology/domain-concepts#Dimension",
        "label": "Dimension",
        "description": (
            "A categorical axis for slicing metrics/measures; e.g. "
            "Segment, Geography."
        ),
    },
    {
        "iri": "https://veerla-ramrao.ai/ontology/domain-concepts#Measure",
        "label": "Measure",
        "description": (
            "A raw measured quantity with units; e.g. RevenueUSDM, UnitsSold."
        ),
    },
    {
        "iri": "https://veerla-ramrao.ai/ontology/domain-concepts#TimePeriod",
        "label": "TimePeriod",
        "description": (
            "A bounded reporting interval used as a slicing axis; e.g. FY2024."
        ),
    },
]

ANCHOR_IRIS = frozenset(b["iri"] for b in ANCHOR_BUCKETS)
_IRI_TO_LABEL = {b["iri"]: b["label"] for b in ANCHOR_BUCKETS}

_TABLE_TYPE_BUCKET_IRI = (
    "https://veerla-ramrao.ai/ontology/domain-concepts#FinancialTable"
)

# Cache version: bump when this module's prompt / routing changes
# materially.  Combined with the table's @id into the on-disk cache key.
_GROUPING_VERSION = "p2a-follow-1"

# Label-similarity threshold for layer-1 "reuse existing class" match.
_EXISTING_MATCH_RATIO = 0.85

# Suffix strip-off list for label normalisation in layer-2 dedup.
_UNIT_SUFFIXES = (
    " usd m",
    " usd mn",
    " usd",
    " million",
    " mn",
    " m",
    " bn",
    " billion",
    " thousands",
    " thousand",
    " k",
    " %",
    " percent",
    " count",
)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _normalize_label(label: str) -> str:
    s = label.strip().lower()
    # Punctuation first so "(USD M)" becomes " usd m " before suffix-stripping.
    s = _PUNCT_RE.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    # Iterate until no more unit-suffixes match (handles compound trails
    # like "revenue usd m thousand").
    changed = True
    while changed:
        changed = False
        for suf in _UNIT_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)].rstrip()
                changed = True
                break
    return s


def _all_class_labels(classes_dict: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return a list of (iri, label, normalized_label) for every class
    in the loaded ontology.  Used by layer-1 existing-match."""
    out: list[tuple[str, str, str]] = []
    for iri, rec in classes_dict.items():
        names: list[str] = []
        for lab in rec.get("labels") or []:
            if isinstance(lab, str) and lab:
                names.append(lab)
            elif isinstance(lab, dict):
                v = lab.get("value")
                if isinstance(v, str) and v:
                    names.append(v)
        name = rec.get("name")
        if isinstance(name, str) and name:
            names.append(name)
        for n in names:
            out.append((iri, n, _normalize_label(n)))
    return out


def _existing_match(
    candidate_label: str,
    parent_iri: str,
    classes_index: list[tuple[str, str, str]],
) -> str | None:
    """Layer 1: return the IRI of an existing class whose label is
    similar enough to `candidate_label`, or None.

    Conservative: trigram ratio + exact normalised-label equality.
    We do NOT consider the parent constraint here -- a label match
    is global to avoid duplicating semantically-identical classes
    that already exist under a different bucket.
    """
    norm = _normalize_label(candidate_label)
    if not norm:
        return None
    # Exact normalised match is the fast path.
    for iri, _, nlab in classes_index:
        if nlab == norm:
            return iri
    # Fall back to trigram ratio above threshold.
    best_iri: str | None = None
    best_ratio = 0.0
    for iri, _, nlab in classes_index:
        if not nlab:
            continue
        # Cheap pre-filter: trigram only if first chars match.
        if nlab[0] != norm[0]:
            continue
        r = difflib.SequenceMatcher(a=norm, b=nlab).ratio()
        if r > best_ratio:
            best_ratio = r
            best_iri = iri
    if best_ratio >= _EXISTING_MATCH_RATIO:
        return best_iri
    return None


def _grouping_cache_key(table_id: str) -> str:
    h = hashlib.sha256()
    h.update(_GROUPING_VERSION.encode("utf-8"))
    h.update(b"|")
    h.update(table_id.encode("utf-8"))
    return h.hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"grouping_{key}.json"


def _cache_load(cache_dir: Path | None, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_save(cache_dir: Path | None, key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, key)
    try:
        p.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        # Cache write failure is non-fatal.
        pass


def _load_tables_from_dir(tables_dir: Path) -> list[dict[str, Any]]:
    """Walk `tables_dir/*.jsonld` (one file per PDF, each holding a
    list of table payloads in the on-disk cache format) and return a
    flat list of every individual table payload."""
    out: list[dict[str, Any]] = []
    if not tables_dir.exists():
        return out
    for fp in sorted(tables_dir.glob("*.jsonld")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # The on-disk format is a `{doc_sha, tables: [...], manifest: {...}}`
        # bundle written by table_cache.save.
        tables = payload.get("tables") if isinstance(payload, dict) else None
        if not isinstance(tables, list):
            continue
        for t in tables:
            if isinstance(t, dict):
                out.append(t)
    return out


def _extract_table_fields(table: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """Pull (table_id, caption, columns, row_label_samples) from one
    JSON-LD table payload.

    `columns` is a list of {column_index, label} dicts.
    `row_label_samples` is up to 3 row labels (rowLabel field), skipping
    empty / header rows.
    """
    tid = str(table.get("@id") or "")
    caption = str(table.get("caption") or "")
    cols_payload = table.get("columns") or []
    columns: list[dict[str, Any]] = []
    for c in cols_payload:
        if not isinstance(c, dict):
            continue
        idx = c.get("columnIndex")
        # JSON-LD persists column header under `columnLabel` (matching VIAO's
        # viao:columnLabel data property). Fall back to `label` for back-
        # compat with any test fixtures that used the shorter key.
        label = c.get("columnLabel") or c.get("label") or ""
        if not isinstance(label, str):
            label = str(label)
        if idx is None:
            continue
        columns.append({"column_index": int(idx), "label": label.strip()})

    rows_payload = table.get("rows") or []
    row_labels: list[str] = []
    for r in rows_payload:
        if not isinstance(r, dict):
            continue
        if r.get("isHeaderRow"):
            continue
        lab = r.get("rowLabel") or ""
        if isinstance(lab, str) and lab.strip():
            row_labels.append(lab.strip())
        if len(row_labels) >= 3:
            break

    return tid, caption, columns, row_labels


async def _classify_one_table(
    table: dict[str, Any],
    router: LLMRouter,
    cache_dir: Path | None,
) -> dict[str, Any] | None:
    """One LLM call (cached) per table.  Returns the parsed JSON dict
    with `table_class` + `columns` keys, or None on failure.
    """
    table_id, caption, columns, row_labels = _extract_table_fields(table)
    if not table_id or not columns:
        return None

    cache_key = _grouping_cache_key(table_id)
    cached = _cache_load(cache_dir, cache_key)
    if cached is not None:
        return cached

    system, user = PROMPTS["table_concept_grouping"](
        caption,
        columns,
        row_labels,
        ANCHOR_BUCKETS,
    )
    try:
        result = await router.chat("table_concept_grouping", system=system, user=user)
    except Exception as exc:  # noqa: BLE001
        print(f"[table-mining] LLM failed for table {table_id}: {exc}")
        return None
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        # Fall back to permissive extraction (matches db_artifact_gen pattern).
        m = re.search(r"\{.*\}", result.text or "", re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None

    _cache_save(cache_dir, cache_key, parsed)
    return parsed


def _validate_parent_iri(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if v in ANCHOR_IRIS else None


def _stage2_match_entry(iri: str, snippet: str) -> dict[str, str]:
    return {"IRI": iri, "TEXT_SNIPPET": snippet}


def _stage2_proposal_entry(
    label: str,
    description: str,
    parent_label: str,
) -> dict[str, str]:
    return {
        "LABEL": label,
        "DESCRIPTION": description,
        "PARENT_LABEL": parent_label,
    }


async def mine_table_concepts_async(
    tables_dir: Path,
    loaded_ontology: dict[str, Any],
    router: LLMRouter,
    *,
    cache_dir: Path | None = None,
    audit_callback: Any = None,
) -> dict[str, Any]:
    """Return a Stage-2-shaped dict ready to feed Stage 3 `match_dedup`.

    `tables_dir` is the run-folder cache (`<run>/tables/`); it must
    contain the per-PDF `.jsonld` bundles written by the extractor.
    `cache_dir` is an OPTIONAL user-level cache for the per-table
    grouping LLM call.

    `audit_callback`, when provided, is called as
    `audit_callback(task_name: str, payload: dict)` for each decision
    (raw LLM output, layer-1 reuse, layer-2 collapse, final tally).
    """
    tables = _load_tables_from_dir(tables_dir)
    if not tables:
        print("[table-mining] no tables in run cache; skipping")
        return {
            "MATCHES FOUND": [],
            "MATCH NOT FOUND": [],
            "MATCH NOT FOUND RELATIONS": [],
        }

    print(
        f"[table-mining] classifying {len(tables)} table(s) "
        f"into {len(ANCHOR_BUCKETS)} anchor bucket(s)"
    )

    classes_dict = loaded_ontology.get("classes_dict") or {}
    classes_index = _all_class_labels(classes_dict)

    # Run the LLM passes with bounded concurrency to keep memory + spend low.
    sem = asyncio.Semaphore(4)

    async def _one(t: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            return await _classify_one_table(t, router, cache_dir)

    raw_results = await asyncio.gather(*[_one(t) for t in tables])

    matches: list[dict[str, str]] = []
    raw_candidates: list[dict[str, Any]] = []  # {label, description, parent_iri, source_id}

    n_reused = 0
    n_dropped_bad_parent = 0

    for table, parsed in zip(tables, raw_results, strict=False):
        if parsed is None:
            continue
        table_id, caption, _cols, _rows = _extract_table_fields(table)
        if audit_callback is not None:
            audit_callback(
                "table_concept_grouping",
                {"table_id": table_id, "caption": caption, "llm_output": parsed},
            )

        # ----- table-class candidate -----
        tc = parsed.get("table_class")
        if isinstance(tc, dict):
            label = (tc.get("proposed_label") or "").strip()
            definition = (tc.get("definition") or "").strip()
            # The table-class anchor is always FinancialTable; even if the
            # LLM emits a different one we coerce it.
            parent_iri = _TABLE_TYPE_BUCKET_IRI
            if label:
                existing = _existing_match(label, parent_iri, classes_index)
                if existing is not None:
                    matches.append(_stage2_match_entry(existing, caption or label))
                    n_reused += 1
                    if audit_callback is not None:
                        audit_callback(
                            "table_concept_match",
                            {"label": label, "matched_iri": existing,
                             "via": "table_class"},
                        )
                else:
                    raw_candidates.append({
                        "label": label,
                        "description": definition or f"Subclass of FinancialTable for {label}.",
                        "parent_iri": parent_iri,
                        "source_id": table_id,
                        "snippet": caption or label,
                    })

        # ----- column-class candidates -----
        cols = parsed.get("columns")
        if isinstance(cols, list):
            for c in cols:
                if not isinstance(c, dict):
                    continue
                label = (c.get("proposed_label") or "").strip()
                definition = (c.get("definition") or "").strip()
                parent_iri = _validate_parent_iri(c.get("parent_iri"))
                if not label or parent_iri is None:
                    if parent_iri is None and label:
                        n_dropped_bad_parent += 1
                    continue
                existing = _existing_match(label, parent_iri, classes_index)
                if existing is not None:
                    matches.append(_stage2_match_entry(existing, label))
                    n_reused += 1
                    if audit_callback is not None:
                        audit_callback(
                            "table_concept_match",
                            {"label": label, "matched_iri": existing,
                             "via": "column"},
                        )
                else:
                    raw_candidates.append({
                        "label": label,
                        "description": definition or f"Subclass of {_IRI_TO_LABEL[parent_iri]} for {label}.",
                        "parent_iri": parent_iri,
                        "source_id": table_id,
                        "snippet": label,
                    })

    # ----- Layer 2: across-table collapse -----
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in raw_candidates:
        key = (_normalize_label(cand["label"]), cand["parent_iri"])
        if key in grouped:
            grouped[key]["occurrences"].append(cand["source_id"])
            continue
        grouped[key] = {
            "label": cand["label"],
            "description": cand["description"],
            "parent_iri": cand["parent_iri"],
            "occurrences": [cand["source_id"]],
        }

    if audit_callback is not None and grouped:
        collapsed = {
            f"{key[0]}|{_IRI_TO_LABEL[key[1]]}": g["occurrences"]
            for key, g in grouped.items()
            if len(g["occurrences"]) > 1
        }
        if collapsed:
            audit_callback(
                "table_concept_collapse",
                {"collapsed_groups": collapsed, "n_groups": len(collapsed)},
            )

    # ----- Emit Stage-2-shaped proposals -----
    proposals: list[dict[str, str]] = []
    for entry in grouped.values():
        parent_label = _IRI_TO_LABEL[entry["parent_iri"]]
        proposals.append(_stage2_proposal_entry(
            label=entry["label"],
            description=entry["description"],
            parent_label=parent_label,
        ))

    out = {
        "MATCHES FOUND": matches,
        "MATCH NOT FOUND": proposals,
        "MATCH NOT FOUND RELATIONS": [],
    }

    print(
        f"[table-mining] done: {len(matches)} reused, {len(proposals)} new "
        f"proposed, {n_dropped_bad_parent} dropped (bad parent_iri), "
        f"from {len(tables)} tables"
    )
    if audit_callback is not None:
        audit_callback(
            "table_concept_summary",
            {
                "n_tables": len(tables),
                "n_reused": n_reused,
                "n_proposed": len(proposals),
                "n_dropped_bad_parent": n_dropped_bad_parent,
            },
        )

    return out

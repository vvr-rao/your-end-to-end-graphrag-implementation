"""Shared helpers: which entities are concepts or places, and where reports go."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.core.config import get_settings
from backend.app.services.retrieval_sql import _CONCEPT_CLASS_IDS_SQL

REPORT_ROOT = Path(".build/evals")


def concept_roots() -> list[str]:
    cfg = get_settings().app_config.get("extraction", {}) or {}
    return [str(r) for r in cfg.get("concept_class_roots") or ()]


def geo_labels() -> tuple[str, ...]:
    from backend.app.services.db_geo_enrich import DEFAULT_GEO_CLASS_LABELS

    cfg = get_settings().app_config.get("geo_enrichment", {}) or {}
    return tuple(cfg.get("class_labels") or DEFAULT_GEO_CLASS_LABELS)


async def concept_class_ids(session: Any) -> set[uuid.UUID]:
    labels = [r.strip().lower() for r in concept_roots() if r.strip()]
    if not labels:
        return set()
    rows = await session.execute(_CONCEPT_CLASS_IDS_SQL, {"labels": labels})
    return {r[0] for r in rows.all()}


async def place_class_ids(session: Any) -> set[uuid.UUID]:
    """Geographic classes, minus anything that is also an agent, organization
    or event (the same rule enrich-geo uses)."""
    from backend.app.services.db_geo_enrich import NON_PLACE_CLASS_LABELS, _class_closure

    ids = await _class_closure(session, geo_labels())
    return ids - await _class_closure(session, NON_PLACE_CLASS_LABELS)


async def entity_kinds(session: Any) -> dict[uuid.UUID, tuple[str, uuid.UUID]]:
    """{entity id: (kind, class id)} for every ACTIVE entity. Kind is `concept`
    (typed under a concept root and not judged a named thing), `place` (typed
    under a geographic class) or `named` (everything else)."""
    concept_cls = await concept_class_ids(session)
    place_cls = await place_class_ids(session)
    rows = await session.execute(text("""
        SELECT id, class_id, extra_metadata->>'concept_kind' = 'named'
          FROM graphrag.entities WHERE status = 'ACTIVE'
    """))
    out: dict[uuid.UUID, tuple[str, uuid.UUID]] = {}
    for eid, cid, judged_named in rows.all():
        if cid in concept_cls and not judged_named:
            kind = "concept"
        elif cid in place_cls:
            kind = "place"
        else:
            kind = "named"
        out[eid] = (kind, cid)
    return out


def database_name() -> str:
    from sqlalchemy.engine import make_url

    url = make_url(get_settings().database_url)
    return f"{url.host or 'local'}_{url.port or ''}_{url.database or ''}"


def configured_batch_size(default: int = 25) -> int:
    """Questions per child process in eval-questions (`evals.batch_size`)."""
    cfg = get_settings().app_config.get("evals", {}) or {}
    value = cfg.get("batch_size")
    return default if value is None else max(0, int(value))


def save_report(kind: str, report: dict[str, Any], out: Path | None = None) -> Path:
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = REPORT_ROOT / database_name() / f"{kind}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return out


def buckets(counts: list[int]) -> dict[str, int]:
    """How many values are 0, 1, 2, 3 and more than 3."""
    out = {"0": 0, "1": 0, "2": 0, "3": 0, ">3": 0}
    for n in counts:
        out[str(n) if n <= 3 else ">3"] += 1
    return out


def count_buckets(counts: list[int], top: int = 5) -> dict[str, int]:
    """How many values are 0, 1, ..., `top` and more than `top`."""
    out = {str(i): 0 for i in range(top + 1)}
    out[f">{top}"] = 0
    for n in counts:
        out[str(n) if n <= top else f">{top}"] += 1
    return out


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    mid = len(v) // 2
    return float(v[mid]) if len(v) % 2 else (v[mid - 1] + v[mid]) / 2

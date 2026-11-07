"""Tell named things apart from real concepts among entities typed under the
concept roots (`extraction.concept_class_roots`).

Being typed under a concept root does not make an entity a concept. Trials sit
under `Process` and vehicles under `TechnologyConcept`, so `SUSTAIN 6` and
`BMW iX3` counted as concepts next to `hypoglycemia` and `operating margin`.
Measured on the 2026-10-24 bench graphs, by hand: 21% of the finance concept
entities and 10% of the pharma ones were named things. That is not cosmetic:
retrieval never walks THROUGH a concept (`qa.concept_terminal`), so a walk
stopped at the trial or product it most needed to pass through.

The class cannot decide it (`cardiovascular outcomes trial` and `SUSTAIN 6`
share one) and neither can capitalisation (tested: it could not decide 30-50%
of entities and misses lowercase drug names). So a cheap judge reads each name.

The verdict is stored as `extra_metadata.concept_kind`. Only a `named` verdict
changes anything: an entity with no verdict stays a concept, so a graph that
never ran this behaves as before, and a failed call just leaves entities for
the next run. Runs after `extract-entities` and as `classify-concepts`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import text

from backend.app.db.session import session_scope
from backend.app.services.retrieval_sql import _CONCEPT_CLASS_IDS_SQL

TASK = "concept_kind_judge"
KEY = "concept_kind"
NAMED = "named"
CONCEPT = "concept"
_KINDS = frozenset({NAMED, CONCEPT})


def _judge_prompt(prompt_set: str | None):
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_concept_kind_judge

        return deepseek_concept_kind_judge
    if prompt_set in (None, "", "default"):
        from backend.app.services.prompts import concept_kind_judge

        return concept_kind_judge
    return None


def parse_decisions(got: Any, batch: list[dict[str, Any]]) -> dict[str, str]:
    """{entity id: kind} for the valid answers. A bad id or an unknown kind is
    left out, so that entity keeps today's behaviour and is asked again."""
    out: dict[str, str] = {}
    for d in got if isinstance(got, list) else []:
        if not isinstance(d, dict):
            continue
        try:
            item = batch[int(d["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        kind = str(d.get("kind") or "").strip().lower()
        if kind in _KINDS:
            out[item["entity_id"]] = kind
    return out


async def _load(roots: list[str], *, limit: int | None, recheck: bool) -> list[dict[str, Any]]:
    labels = [r.strip().lower() for r in roots if r and r.strip()]
    sql = text("""
        SELECT e.id::text AS entity_id, e.name, oc.label AS class_label
          FROM graphrag.entities e
          JOIN graphrag.ontology_classes oc ON oc.id = e.class_id
         WHERE e.status = 'ACTIVE'
           AND e.class_id IN (SELECT id FROM (""" + _CONCEPT_CLASS_IDS_SQL.text + """) c)
           AND (:recheck OR NOT e.extra_metadata ? 'concept_kind')
         ORDER BY e.name, e.id
         LIMIT :limit
    """)
    async with session_scope() as session:
        rows = await session.execute(
            sql, {"labels": labels, "recheck": recheck, "limit": limit})
        return [dict(r) for r in rows.mappings().all()]


async def _save(verdicts: dict[str, str]) -> None:
    sql = text("""
        UPDATE graphrag.entities
           SET extra_metadata = extra_metadata || jsonb_build_object('concept_kind', CAST(:kind AS text))
         WHERE id = ANY(CAST(:ids AS uuid[]))
    """)
    async with session_scope() as session:
        for kind in _KINDS:
            ids = [i for i, k in verdicts.items() if k == kind]
            if ids:
                await session.execute(sql, {"kind": kind, "ids": ids})


async def classify_concept_kinds(
    router: Any,
    roots: list[str],
    *,
    limit: int | None = None,
    recheck: bool = False,
    dry_run: bool = False,
    log=print,
) -> dict[str, Any]:
    """Judge the entities under `roots` that have no verdict yet. Each batch is
    saved as it finishes, so an interrupted run keeps what it paid for."""
    stats: dict[str, Any] = {"entities": 0, NAMED: 0, CONCEPT: 0, "unanswered": 0,
                             "failed_calls": 0, "cost_usd": 0.0, "named_sample": []}
    try:
        spec = router.task_spec(TASK)
    except (AttributeError, KeyError):
        return stats
    if not spec or spec.get("enabled") is False or not roots:
        return stats
    build = _judge_prompt(spec.get("prompt_set"))
    if build is None:
        log(f"[concept-kinds] skipped: no {TASK} prompt for prompt_set={spec.get('prompt_set')!r}")
        return stats
    items = await _load(roots, limit=limit, recheck=recheck)
    stats["entities"] = len(items)
    if not items:
        log("[concept-kinds] nothing to classify")
        return stats
    from backend.app.services.db_artifact_gen import _extract_json

    size = int(spec.get("entities_per_call", 25))
    sem = asyncio.Semaphore(int(spec.get("concurrency", 8)))
    start_cost = router.total_cost_usd

    async def _one(batch: list[dict[str, Any]]) -> None:
        ask = [{"id": k, "name": it["name"], "class": it["class_label"]}
               for k, it in enumerate(batch)]
        system, user = build(ask)
        async with sem:
            try:
                res = await router.chat(TASK, system=system, user=user)
                parsed = _extract_json(res.text)
            except Exception as exc:  # no verdict = still a concept; asked again next run
                stats["failed_calls"] += 1
                stats["unanswered"] += len(batch)
                log(f"[concept-kinds] judge call failed ({type(exc).__name__}); "
                    f"{len(batch)} entities left unclassified")
                return
            verdicts = parse_decisions(
                parsed.get("decisions") if isinstance(parsed, dict) else None, batch)
            if verdicts and not dry_run:
                await _save(verdicts)
        stats["unanswered"] += len(batch) - len(verdicts)
        for it in batch:
            kind = verdicts.get(it["entity_id"])
            if kind:
                stats[kind] += 1
            if kind == NAMED:
                stats["named_sample"].append(it["name"])

    await asyncio.gather(*(_one(items[i:i + size]) for i in range(0, len(items), size)))
    stats["cost_usd"] = router.total_cost_usd - start_cost
    log(f"[concept-kinds] {stats['entities']} entities: {stats[NAMED]} named, "
        f"{stats[CONCEPT]} concepts, {stats['unanswered']} unanswered, "
        f"{stats['failed_calls']} failed call(s), cost=${stats['cost_usd']:.4f}"
        + (" (dry run, nothing saved)" if dry_run else ""))
    return stats

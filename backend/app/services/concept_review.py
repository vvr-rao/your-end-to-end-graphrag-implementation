"""Review the concepts a chunk produced: keep, re-type or drop each one.

Nothing else checks these. The concept call's reply goes through a code filter
that only asks whether the class was on the menu, `--validate-entities` runs
before that call and is off by default, and `concept_kind_judge` reads names
only. So "a tradition" typed `EconomicConcept` reached the graph unchallenged,
and most concepts appear in one chunk, so voting across chunks never helps.

Runs per chunk, after the concept call and before name-level merging, on what
counts as a concept whichever call produced it: everything the concept call
added plus entity-call results typed under a concept class.

`no_class` does not delete: every entity needs a class, and rejecting one is not
proof the concept is bad, so the concept stays and is flagged
`extra_metadata.concept_class_doubtful`. Retrieval leaves flagged concepts out of
concept seeding; they can still be reached by a walk or matched by exact name.

Fails OPEN, unlike `relationship_verify`, where a missing verdict discards one
claim. Here it would empty a chunk's concepts, so a failed call, a missing
verdict or an unusable suggestion leaves the concept as it was.

Gated by the `concept_review` task: absent or `enabled: false` skips the step.

Off in every preset for now. Blind test on 101 concepts (11 poor, 2026-11-03),
counting a rejection as a catch: DeepSeek-V4.1-Flash caught 4 with few rejections
of sound concepts (2 of 90) or 9 with many (21); Kimi-K3 caught 10 and rejected 11.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

TASK = "concept_review"
SAMPLE_CAP = 60
DOUBTFUL_KEY = "concept_class_doubtful"
_VERDICTS = frozenset({"keep", "move", "drop", "no_class"})


def new_stats() -> dict[str, int]:
    return {
        "review_calls": 0, "review_failed": 0, "review_checked": 0, "review_kept": 0,
        "review_moved": 0, "review_dropped": 0, "review_flagged": 0,
        "review_unanswered": 0, "review_bad_move": 0,
    }


def _prompt_for(prompt_set: str | None):
    if prompt_set == "kimi":
        from backend.app.services.prompts_kimi import kimi_concept_review

        return kimi_concept_review
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_concept_review

        return deepseek_concept_review
    if prompt_set in (None, "", "default"):
        from backend.app.services.prompts import concept_review

        return concept_review
    return None


def resolve_prompt(router: Any, log=print):
    """The prompt builder to use, or None when the step should not run."""
    try:
        spec = router.task_spec(TASK)
    except (AttributeError, KeyError):
        return None
    if not spec or spec.get("enabled") is False:
        return None
    build = _prompt_for(spec.get("prompt_set"))
    if build is None:
        log(f"[concept-review] skipped: no {TASK} prompt for "
            f"prompt_set={spec.get('prompt_set')!r}")
    return build


def _menu_lookup(menu: list[dict[str, str]]) -> dict[str, str]:
    """IRI or label -> IRI. Models routinely write the label in the IRI field."""
    out: dict[str, str] = {}
    for c in menu:
        label = (c.get("label") or "").strip().lower()
        if label:
            out.setdefault(label, c["iri"])
    for c in menu:
        out[c["iri"].lower()] = c["iri"]
    return out


def apply_verdicts(
    parsed: Any,
    concepts: list[dict[str, Any]],
    menu: list[dict[str, str]],
    stats: dict[str, int],
    samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any] | None]:
    """One result per concept, in order: the concept (class changed or flagged if
    so judged), or None if dropped. Anything unusable keeps it unchanged."""
    verdicts: dict[int, dict[str, Any]] = {}
    raw = parsed.get("verdicts") if isinstance(parsed, dict) else None
    for v in raw if isinstance(raw, list) else []:
        if not isinstance(v, dict):
            continue
        try:
            verdicts.setdefault(int(v.get("id")), v)
        except (TypeError, ValueError):
            continue
    lookup = _menu_lookup(menu)
    results: list[dict[str, Any] | None] = []
    for i, concept in enumerate(concepts):
        stats["review_checked"] += 1
        v = verdicts.get(i)
        verdict = str(v.get("verdict") or "").strip().lower() if v else ""
        if verdict not in _VERDICTS:
            stats["review_unanswered"] += 1
            results.append(concept)
            continue
        reason = str(v.get("reason") or "").strip()[:200]
        target = None
        if verdict == "move":
            target = lookup.get(str(v.get("better_class_iri") or "").strip().lower())
            if target is None:
                stats["review_bad_move"] += 1
                verdict = "keep"
            elif target == concept["class_iri"]:
                verdict = "keep"
        if verdict == "keep":
            stats["review_kept"] += 1
            results.append(concept)
            continue
        if samples is not None and len(samples) < SAMPLE_CAP:
            samples.append({"name": concept.get("canonical_name"), "verdict": verdict,
                            "from": concept["class_iri"], "to": target, "reason": reason})
        if verdict == "move":
            stats["review_moved"] += 1
            results.append({**concept, "class_iri": target})
        elif verdict == "no_class":
            stats["review_flagged"] += 1
            results.append({**concept, DOUBTFUL_KEY: True})
        else:
            stats["review_dropped"] += 1
            results.append(None)
    return results


def doubtful_names(
    mentions: Iterable[tuple[str, str, bool]], primary_class: Mapping[str, str]
) -> set[str]:
    """Names to flag, from (name, class IRI, flagged) mentions. A name becomes one
    node with one main class, so only mentions of that class count, and most of
    them must have been flagged."""
    seen: dict[str, list[int]] = {}
    for name, class_iri, flagged in mentions:
        if class_iri == primary_class.get(name, class_iri):
            counts = seen.setdefault(name, [0, 0])
            counts[0] += 1 if flagged else 0
            counts[1] += 1
    return {name for name, (hits, total) in seen.items() if hits * 2 > total}


async def review_concepts(
    router: Any,
    build: Any,
    chunk_text: str,
    concepts: list[dict[str, Any]],
    menu: list[dict[str, str]],
    stats: dict[str, int],
    samples: list[dict[str, Any]] | None = None,
    log=print,
) -> list[dict[str, Any] | None]:
    """Ask the reviewer about `concepts`; see `apply_verdicts` for the result."""
    if not concepts:
        return []
    from backend.app.services.db_artifact_gen import _extract_json

    labels = {c["iri"]: c.get("label") or "" for c in menu}
    items = [{"id": i, "name": c.get("canonical_name"), "class_iri": c["class_iri"],
              "class_label": labels.get(c["class_iri"], "")}
             for i, c in enumerate(concepts)]
    stats["review_calls"] += 1
    try:
        system, user = build(chunk_text, menu, items)
        out = await router.chat(TASK, system=system, user=user)
        parsed = _extract_json(out.text)
    except Exception as exc:
        stats["review_failed"] += 1
        log(f"[concept-review] call failed ({type(exc).__name__}); "
            f"{len(concepts)} concept(s) kept unreviewed")
        return list(concepts)
    if not isinstance(parsed, dict):
        stats["review_failed"] += 1
        return list(concepts)
    return apply_verdicts(parsed, concepts, menu, stats, samples)

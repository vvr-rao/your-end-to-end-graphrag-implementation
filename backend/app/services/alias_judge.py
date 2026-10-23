"""LLM judge for corpus-mined synonym pairs (task `alias_judge`).

`alias_mining` reads "NAME (other name)" and "X, also known as Y" off the
text. That is safe in FDA labels, where `BRAND (generic)` is a fixed
convention, but not in scraped web text, where the bracket as often names
an owner, a place or a note. On the 2026-10-19 web-search build ~440 of 614
mined pairs were false -- `Germu\\nRefinery (PetroChina)` gave Refinery <->
PetroChina (and 62 more "Refinery" pairs), `U.S. Department of Energy (DoE)`
gave Energy <-> DoE -- and retrieval applies every pair to the probes, the
graph seeds and the answer prompt. Shape rules cannot tell "another name
for" from "owned by", so a model makes that call.

Contract:
  * Only `parenthetical` and `phrase` pairs are judged. `ontology` pairs are
    curated annotations; `manual` rows never pass through the miner.
  * Verdicts are cached in `term_alias_judgements`, keyed by judge model and
    JUDGE_VERSION, so each pair is paid for once -- not on every ingest --
    and switching LLM mode or prompt re-judges with the new judge.
  * Fail closed and atomic: if any pair is left unjudged (a failed call, an
    id missing from the reply) the refresh raises AliasJudgeIncompleteError
    BEFORE `term_aliases` is touched. Verdicts already made are kept, so a
    re-run only asks for the rest.
  * The model comes from the active preset (`alias_judge` task): OpenAI and
    Groq+OpenAI gpt-4.1, Anthropic claude-sonnet-4-6, DeepInfra
    Qwen3.8-27B -- on the DEDICATED deployment when it is running, else
    serverless (`fallback_model`; serverless 27B decodes ~20 tok/s vs ~70+
    dedicated, but delete-document refreshes aliases with no GPU up).
    Both are the same weights, so they share one cache key (`verdict_key`).
    Not a small model: on the 614 web-search pairs gpt-4.1-mini kept ~85
    wrong pairs (76 refinery <-> owner) where gpt-4.1, Sonnet 4.6 and
    Qwen3.8-27B each kept ~5 (2026-10-22).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql_text

TASK = "alias_judge"
# Bump when either prompt changes meaning: cached verdicts are keyed by it.
JUDGE_VERSION = "v1"
JUDGED_KINDS: frozenset[str] = frozenset({"parenthetical", "phrase"})

_CONTEXT_BEFORE = 160
_CONTEXT_AFTER = 80


class AliasJudgeNotConfiguredError(RuntimeError):
    """The active models.yaml has no usable `alias_judge` task."""


class AliasJudgeIncompleteError(RuntimeError):
    """Some pairs got no verdict; term_aliases was left unchanged."""


@dataclass
class JudgeReport:
    model: str = ""              # the model actually called (or cache key)
    hosting: str = ""            # 'dedicated' / 'serverless' / '' (not routed)
    enabled: bool = True
    judged_new: int = 0          # verdicts bought this run
    cached: int = 0              # verdicts reused from term_alias_judgements
    exempt: int = 0              # ontology pairs, not judged
    rejected: int = 0            # judged DIFFERENT (new + cached)
    unjudged: int = 0            # no verdict (dry run, or failed calls)
    cost_usd: float = 0.0
    rejected_samples: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def judge_spec(router: Any) -> dict[str, Any] | None:
    """The task spec, or None when the preset turns the judge off."""
    try:
        spec = router.task_spec(TASK)
    except KeyError as exc:
        raise AliasJudgeNotConfiguredError(
            f"models.yaml has no `{TASK}` task. Re-activate your LLM mode "
            "(scripts/select_llm_mode.py <mode>) to pick up the preset that "
            "defines it, or pass --no-judge to write unjudged pairs."
        ) from exc
    if spec.get("enabled", True) is False:
        return None
    return spec


def judge_prompt(prompt_set: str | None) -> Callable[[list[dict[str, str]]], tuple[str, str]]:
    """Each model family has its own prompt; an unknown set is an error,
    never a silent fallback to another model's wording."""
    if prompt_set == "qwen":
        from backend.app.services.prompts_qwen import qwen_alias_judge

        return qwen_alias_judge
    if prompt_set in (None, "", "default"):
        from backend.app.services.prompts import alias_judge

        return alias_judge
    raise AliasJudgeNotConfiguredError(
        f"no {TASK} prompt for prompt_set={prompt_set!r} (have: default, qwen)"
    )


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def context_snippet(text: str | None, surface_a: str, surface_b: str) -> str | None:
    """The passage a pair was mined from, whitespace-collapsed.

    Anchored on the bracketed side when there is one -- "(PetroChina)" is
    far rarer in a chunk than "Refinery" -- else on either surface.
    """
    if not text:
        return None
    low = text.lower()
    pos = -1
    for s in (surface_b, surface_a):
        pos = low.find(f"({s.lower()}")
        if pos >= 0:
            break
    if pos < 0:
        for s in (surface_a, surface_b):
            pos = low.find(s.lower())
            if pos >= 0:
                break
    if pos < 0:
        return None
    snip = text[max(0, pos - _CONTEXT_BEFORE): pos + _CONTEXT_AFTER]
    return re.sub(r"\s+", " ", snip).strip()


def parse_verdicts(raw: str, n: int) -> dict[int, bool]:
    """{"same": [ids], "different": [ids]} -> {id: same}. Ids outside
    0..n-1 are ignored; an id in BOTH lists is dropped (no verdict)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

    def ids(key: str) -> set[int]:
        out: set[int] = set()
        for v in obj.get(key) or []:
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                out.add(i)
        return out

    same, diff = ids("same"), ids("different")
    both = same & diff
    return {**{i: True for i in same - both}, **{i: False for i in diff - both}}


def resolve_route(
    spec: dict[str, Any], *, status: Callable[[str], str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[str, int, str]:
    """(model, concurrency, hosting) for this run.

    Without `fallback_model` the task's model is used as is. With one, the
    task's model is a dedicated deployment: use it while it is `running`,
    otherwise the serverless fallback at `fallback_concurrency`.
    """
    model = str(spec["model"])
    conc = int(spec.get("concurrency", 8))
    fallback = spec.get("fallback_model")
    if not fallback:
        return model, conc, ""
    if status is None:
        from backend.app.services.deepinfra_dedicated import dedicated_status as status
    st = status(model)
    if st == "running":
        log(f"[alias-judge] dedicated deployment {model} is running; using it")
        return model, conc, "dedicated"
    log(f"[alias-judge] dedicated deployment {st} ({model}); "
        f"falling back to serverless {fallback}")
    return str(fallback), int(spec.get("fallback_concurrency", conc)), "serverless"


async def judge_batches(
    router: Any,
    spec: dict[str, Any],
    items: list[dict[str, str]],
    *,
    model: str | None = None,
    concurrency: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[int, bool]:
    """Ask the judge about `items` ([{"a", "b", "context"}]); returns
    {index into items: same}. A failed call leaves its pairs out."""
    build = judge_prompt(spec.get("prompt_set"))
    size = max(1, int(spec.get("pairs_per_call", 40)))
    sem = asyncio.Semaphore(max(1, concurrency or int(spec.get("concurrency", 8))))
    verdicts: dict[int, bool] = {}

    async def one(start: int) -> None:
        chunk = items[start:start + size]
        ask = [{"id": str(k), **it} for k, it in enumerate(chunk)]
        system, user = build(ask)
        async with sem:
            try:
                res = await router.chat(TASK, system=system, user=user, model=model)
            except Exception as exc:
                log(f"[alias-judge] call failed ({type(exc).__name__}: {exc}); "
                    f"{len(chunk)} pair(s) left unjudged")
                return
        got = parse_verdicts(res.text, len(chunk))
        if len(got) < len(chunk):
            log(f"[alias-judge] reply covered {len(got)}/{len(chunk)} pair(s)")
        for k, same in got.items():
            verdicts[start + k] = same

    await asyncio.gather(*(one(s) for s in range(0, len(items), size)))
    return verdicts


# --------------------------------------------------------------------------
# DB-backed pass
# --------------------------------------------------------------------------

_LOAD_SQL = """
SELECT term_a, term_b, same FROM graphrag.term_alias_judgements
 WHERE judge_model = :m AND judge_version = :v
   AND (term_a, term_b) IN (
     SELECT * FROM unnest(CAST(:a AS text[]), CAST(:b AS text[])))
"""

_CONTEXT_SQL = """
SELECT chunk_identifier, text FROM graphrag.chunks
 WHERE chunk_identifier = ANY(CAST(:refs AS text[]))
"""

_STORE_SQL = """
INSERT INTO graphrag.term_alias_judgements
  (term_a, term_b, judge_model, judge_version, same, surface_a, surface_b, context)
VALUES (:term_a, :term_b, :m, :v, :same, :surface_a, :surface_b, :context)
ON CONFLICT (term_a, term_b, judge_model, judge_version) DO UPDATE SET
  same = EXCLUDED.same, surface_a = EXCLUDED.surface_a,
  surface_b = EXCLUDED.surface_b, context = EXCLUDED.context, judged_at = now()
"""


async def judge_mined_pairs(
    pairs: list[Any],
    *,
    router: Any = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[Any], JudgeReport]:
    """Filter MinedPairs down to the ones the judge calls SAME.

    Dry run: cached verdicts only, no LLM calls; uncached pairs are kept
    in the listing and counted as `unjudged`.
    """
    from backend.app.db.session import session_scope

    if router is None:
        from backend.app.services.llm_router import LLMRouter

        router = LLMRouter()
    report = JudgeReport()
    spec = judge_spec(router)
    if spec is None:
        report.enabled = False
        log(f"[alias-judge] disabled in models.yaml ({TASK}.enabled: false); "
            "pairs written unjudged")
        return pairs, report
    judge_prompt(spec.get("prompt_set"))          # fail before any DB work
    # Cache key: one per set of weights, not per hosting route.
    model = str(spec.get("verdict_key") or spec["model"])
    report.model = model

    todo = [p for p in pairs if p.evidence_kind in JUDGED_KINDS]
    report.exempt = len(pairs) - len(todo)
    verdict: dict[tuple[str, str], bool] = {}
    if todo:
        async with session_scope() as s:
            rows = (await s.execute(sql_text(_LOAD_SQL), {
                "m": model, "v": JUDGE_VERSION,
                "a": [p.term_a for p in todo], "b": [p.term_b for p in todo],
            })).all()
        verdict = {(a, b): bool(same) for a, b, same in rows}
    # One verdict per term pair, even if it was mined by two sources.
    uncached: dict[tuple[str, str], Any] = {}
    for p in todo:
        key = (p.term_a, p.term_b)
        if key not in verdict and key not in uncached:
            uncached[key] = p
    report.cached = len({(p.term_a, p.term_b) for p in todo}) - len(uncached)

    if uncached and not dry_run:
        refs = sorted({p.evidence_ref for p in uncached.values() if p.evidence_ref})
        texts: dict[str, str] = {}
        if refs:
            async with session_scope() as s:
                texts = dict((await s.execute(sql_text(_CONTEXT_SQL), {"refs": refs})).all())
        keys = list(uncached)
        items = []
        for key in keys:
            p = uncached[key]
            items.append({
                "a": p.surface_a, "b": p.surface_b,
                "context": context_snippet(texts.get(p.evidence_ref or ""),
                                           p.surface_a, p.surface_b) or "",
            })
        use_model, conc, report.hosting = resolve_route(spec, log=log)
        report.model = use_model
        cost0 = float(getattr(router, "total_cost_usd", 0.0) or 0.0)
        got = await judge_batches(router, spec, items, model=use_model,
                                  concurrency=conc, log=log)
        report.cost_usd = float(getattr(router, "total_cost_usd", 0.0) or 0.0) - cost0
        rows = []
        for idx, same in got.items():
            key = keys[idx]
            verdict[key] = same
            p = uncached[key]
            rows.append({"term_a": key[0], "term_b": key[1], "m": model,
                         "v": JUDGE_VERSION, "same": same,
                         "surface_a": p.surface_a, "surface_b": p.surface_b,
                         "context": items[idx]["context"] or None})
        report.judged_new = len(rows)
        if rows:
            # Committed on its own, before any check below can raise: a
            # partial run still banks what it paid for.
            async with session_scope() as s:
                await s.execute(sql_text(_STORE_SQL), rows)

    kept: list[Any] = []
    for p in pairs:
        if p.evidence_kind not in JUDGED_KINDS:
            kept.append(p)
            continue
        v = verdict.get((p.term_a, p.term_b))
        if v is None:
            report.unjudged += 1
            if dry_run:
                kept.append(p)
        elif v:
            kept.append(p)
        else:
            report.rejected += 1
            if len(report.rejected_samples) < 50:
                report.rejected_samples.append((p.surface_a, p.surface_b))
    if report.unjudged and not dry_run:
        raise AliasJudgeIncompleteError(
            f"{report.unjudged} pair(s) got no verdict from {model}; "
            "term_aliases left unchanged. Re-run `mine-aliases`: verdicts "
            f"already made ({report.judged_new} this run) are cached."
        )
    return kept, report

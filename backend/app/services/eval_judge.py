"""Eval framework: LLM-as-judge against the F retrieval pipeline.

Run a question set through `retrieve_and_answer` N times per question,
then score each answer on 4 metrics (comprehensiveness, no
hallucination, consistency, gap detection) plus wall-time. Output is a
detailed JSON log and an optional markdown summary.

This file is the smoke-test gate for F before each retrieval mode
ships -- the same rubric also benchmarks regressions over time.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS
from backend.app.services.retrieval import (
    RetrievalResult,
    _CITATION_TOKEN_RE,
    _citation_key,
    retrieve_and_answer,
)


def _citation_validity(res: RetrievalResult) -> float | None:
    """Fraction of emitted citations that resolve to retrieved evidence.

    1.0 = every citation resolves; None = the answer cited nothing, which
    is not a failure and must not be averaged in as a zero.

    `retrieve_and_answer` already strips unresolvable citations, so the
    numerator counts what survived and `res.invalid_citations` supplies
    what was removed. Purely arithmetic -- no judge call, no cost.
    """
    kept = len(_CITATION_TOKEN_RE.findall(res.answer or ""))
    removed = len(res.invalid_citations)
    total = kept + removed
    if total == 0:
        return None
    return kept / total


def _as_synthesized(evidence: list[dict]) -> list[dict]:
    """Evidence as the synthesis saw it: each text cut to
    `qa.evidence_char_cap`, the same cap `retrieve_and_answer` applies.
    The judge prompts no longer clip, so this is the only cut."""
    cap = int(get_settings().app_config.get("qa", {}).get("evidence_char_cap", 5000))
    return [
        {**ev, "text": ev["text"][:cap]} if len(ev.get("text") or "") > cap else ev
        for ev in evidence
    ]


@dataclass
class QuestionResult:
    question: str
    expected_gap: bool
    mode: str
    runs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def parse_question_file(path: Path) -> list[tuple[str, bool]]:
    """Read a question file. Skip comments and blank lines.

    Returns list of (question, expected_gap) tuples. A leading `[gap]`
    tag (with optional surrounding spaces) marks `expected_gap=True`.
    """
    out: list[tuple[str, bool]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[gap\]\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            out.append((m.group(1).strip(), True))
        else:
            out.append((line, False))
    return out


JUDGE_TASKS = ("judge_comprehensiveness", "judge_no_hallucination",
               "judge_gap_detection", "judge_consistency")


def infer_provider(model: str) -> str | None:
    """Provider for a bare model name, or None when the name does not say.

    `claude-*` -> anthropic; `gpt-*`, `o<digit>*`, `chatgpt-*` -> openai; an
    `org/model` id (e.g. `Qwen/...`, `deepseek-ai/...`) -> deepinfra. Anything
    else (e.g. Groq's `llama-*`) needs an explicit `provider:model`.
    """
    m = model.strip().lower()
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith(("gpt-", "chatgpt-")) or re.match(r"o\d", m):
        return "openai"
    if "/" in m:
        return "deepinfra"
    return None


def override_judge_model(router: Any, judge_model: str) -> None:
    """Point every judge_* task at `judge_model` for this run (in memory only;
    models.yaml on disk is untouched).

    Accepts `model` or `provider:model`. Same provider as the preset's judge
    task: only the model changes (timeouts, caps and streaming are kept).
    Different provider: settings that belong to the old provider are dropped
    -- `reasoning_effort` (the new model's default applies), `thinking` and
    `prompt_set` (DeepInfra-only), `temperature` on Anthropic, `stream` on
    Groq -- and the router's own config checks run again, so a bad choice
    fails here rather than mid-run.
    """
    provider, _, model = judge_model.partition(":") if ":" in judge_model else ("", "", judge_model)
    provider, model = provider.strip().lower(), model.strip()
    if not model:
        raise ValueError(f"--judge-model {judge_model!r}: no model name")
    for tname in JUDGE_TASKS:
        if tname not in router._tasks:
            continue
        spec = dict(router._tasks[tname])
        # explicit provider, else inferred from the name, else the preset's
        new_provider = provider or infer_provider(model) or spec.get("provider")
        if new_provider not in router._providers:
            raise ValueError(
                f"--judge-model {judge_model!r} needs provider {new_provider!r}, which is not "
                f"configured (set {str(new_provider).upper()}_API_KEY, or pass provider:model)")
        if new_provider != spec.get("provider"):
            for k in ("reasoning_effort", "thinking", "prompt_set"):
                spec.pop(k, None)
            if new_provider == "anthropic":
                spec.pop("temperature", None)
            if new_provider == "groq":
                spec.pop("stream", None)
        elif spec.get("model") != model:
            spec.pop("reasoning_effort", None)   # tuned for the preset's model
        spec["provider"], spec["model"] = new_provider, model
        router._tasks[tname] = spec
    router._validate_stream_support()
    router._validate_thinking()
    router._validate_reasoning_effort()


def describe_judge_models(router: Any) -> str:
    """`provider:model` of the judge tasks (comma-joined if they differ)."""
    seen = []
    for tname in JUDGE_TASKS:
        spec = getattr(router, "_tasks", {}).get(tname)
        if spec:
            d = f"{spec.get('provider')}:{spec.get('model')}"
            if d not in seen:
                seen.append(d)
    return ", ".join(seen) or "unknown"


def _judge_prompt(router: Any, task_name: str) -> Any:
    """`prompt_set: deepseek` on the judge task selects DeepSeek's own judge
    prompt (prompts_deepseek.py); otherwise the shared one."""
    try:
        prompt_set = router.task_spec(task_name).get("prompt_set")
    except (AttributeError, KeyError):  # test doubles without a task map
        prompt_set = None
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import DEEPSEEK_JUDGES

        return DEEPSEEK_JUDGES[task_name]
    return PROMPTS[task_name]


async def _judge(
    router: LLMRouter,
    task_name: str,
    prompt_args: tuple,
) -> dict[str, Any]:
    """One judge call. Returns {score, justification} (or {score=None,
    justification=error message} on failure)."""
    try:
        system, user = _judge_prompt(router, task_name)(*prompt_args)
    except Exception as exc:
        return {"score": None, "justification": f"prompt builder error: {exc}"}
    try:
        out = await router.chat(task_name, system=system, user=user)
        parsed = _extract_json(out.text) or {}
        score = parsed.get("score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        return {
            "score": score,
            "justification": parsed.get("justification") or "",
        }
    except Exception as exc:
        return {"score": None, "justification": f"judge call failed: {exc}"}


async def evaluate_questions(
    *,
    questions_path: Path,
    mode: str = "simple_qa",
    runs_per_question: int = 3,
    judge_model: str | None = None,
    use_structured_data: bool | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
    max_cost_usd: float = 10.0,
    concurrency: int = 4,
    query_max_cost_usd: float = 0.20,
    verbose: bool = False,
) -> dict[str, Any]:
    """Driver. See `evaluate-queries` CLI for argument semantics.

    `judge_model` None = the judge_* tasks run on the models configured in
    `config/models.yaml`. Any other value (`model` or `provider:model`)
    replaces the model of all four judge tasks for this run only; see
    `override_judge_model`.
    """
    questions = parse_question_file(questions_path)
    if not questions:
        raise ValueError(f"no usable questions in {questions_path}")

    router = LLMRouter()
    if judge_model:
        override_judge_model(router, judge_model)
    judge_model = describe_judge_models(router)
    print(
        f"[eval] {len(questions)} question(s) x {runs_per_question} run(s) "
        f"in mode={mode} with judge={judge_model}"
    )

    cost_before = router.total_cost_usd
    cost_limit_hit = asyncio.Event()
    sem = asyncio.Semaphore(concurrency)

    all_results: list[QuestionResult] = []
    t0 = time.time()

    async def _one_question(q: str, expected_gap: bool) -> QuestionResult:
        if cost_limit_hit.is_set():
            return QuestionResult(question=q, expected_gap=expected_gap, mode=mode)
        async with sem:
            if cost_limit_hit.is_set():
                return QuestionResult(question=q, expected_gap=expected_gap, mode=mode)

            qres = QuestionResult(question=q, expected_gap=expected_gap, mode=mode)
            # Run N times sequentially within this slot.
            for i in range(runs_per_question):
                if cost_limit_hit.is_set():
                    break
                rt0 = time.time()
                try:
                    res = await retrieve_and_answer(
                        q, mode=mode,
                        max_cost_usd=query_max_cost_usd,
                        use_structured_data=use_structured_data,
                    )
                    qres.runs.append({
                        "answer": res.answer,
                        "evidence": res.evidence,
                        "evidence_count": len(res.evidence),
                        "retrieval_run_id": str(res.retrieval_run_id)
                                            if res.retrieval_run_id else None,
                        "wall_seconds": res.wall_seconds,
                        "cost_usd": res.cost_usd,
                        # Deterministic, no LLM call. `no_hallucination` asks
                        # whether each CLAIM is grounded and never looks at
                        # IRIs, so an answer carrying citations that resolve
                        # to nothing scores a clean 1.0 on it -- exactly how
                        # four dead ids passed the 2026-08-19 audit. This
                        # counts them instead.
                        "citation_validity": _citation_validity(res),
                    })
                except Exception as exc:
                    qres.runs.append({
                        "answer": f"(query failed: {exc})",
                        "evidence": [], "evidence_count": 0,
                        "retrieval_run_id": None,
                        "wall_seconds": time.time() - rt0,
                        "cost_usd": 0.0,
                        "citation_validity": None,
                    })

            # Judge calls -- parallel, one per metric per run + 1
            # consistency call across the run set.
            judge_tasks: list[tuple[str, str, tuple]] = []
            for run in qres.runs:
                evidence = _as_synthesized(run["evidence"])
                judge_tasks.append((
                    "comprehensiveness",
                    "judge_comprehensiveness",
                    (q, evidence, run["answer"]),
                ))
                judge_tasks.append((
                    "no_hallucination",
                    "judge_no_hallucination",
                    (q, evidence, run["answer"]),
                ))
                judge_tasks.append((
                    "gap_detection",
                    "judge_gap_detection",
                    (q, evidence, run["answer"], expected_gap),
                ))
            if len(qres.runs) >= 2:
                judge_tasks.append((
                    "consistency",
                    "judge_consistency",
                    (q, [r["answer"] for r in qres.runs]),
                ))

            judge_results = await asyncio.gather(*[
                _judge(router, task_name, args) for _, task_name, args in judge_tasks
            ])

            # Aggregate per metric across runs.
            comp_scores: list[tuple[float, str]] = []
            hall_scores: list[tuple[float, str]] = []
            gap_scores: list[tuple[float, str]] = []
            consistency: dict[str, Any] | None = None

            for (metric, _, _), jr in zip(judge_tasks, judge_results, strict=True):
                if metric == "consistency":
                    consistency = jr
                    continue
                if jr["score"] is None:
                    continue
                if metric == "comprehensiveness":
                    comp_scores.append((jr["score"], jr["justification"]))
                elif metric == "no_hallucination":
                    hall_scores.append((jr["score"], jr["justification"]))
                elif metric == "gap_detection":
                    gap_scores.append((jr["score"], jr["justification"]))

            def _agg(scores: list[tuple[float, str]]) -> dict[str, Any]:
                vals = [s for s, _ in scores]
                if not vals:
                    return {"mean": None, "min": None, "max": None,
                            "justifications": []}
                return {
                    "mean": statistics.mean(vals),
                    "min":  min(vals),
                    "max":  max(vals),
                    "justifications": [j for _, j in scores],
                }

            walls = [r["wall_seconds"] for r in qres.runs]
            qres.metrics = {
                "comprehensiveness": _agg(comp_scores),
                "no_hallucination":  _agg(hall_scores),
                "gap_detection":     _agg(gap_scores),
                "consistency": (consistency or {"score": None, "justification": ""}),
                "wall_seconds": {
                    "mean": statistics.mean(walls) if walls else None,
                    "min":  min(walls) if walls else None,
                    "max":  max(walls) if walls else None,
                },
            }

            if router.total_cost_usd - cost_before > max_cost_usd:
                if not cost_limit_hit.is_set():
                    cost_limit_hit.set()
                    print(f"[eval] HALT: cost cap ${max_cost_usd} reached")
            if verbose:
                comp_mean = qres.metrics["comprehensiveness"]["mean"]
                hall_mean = qres.metrics["no_hallucination"]["mean"]
                print(
                    f"  [{q[:60]}] comp={comp_mean} "
                    f"no_hall={hall_mean} cost=${router.total_cost_usd-cost_before:.3f}"
                )
            return qres

    # Run questions in parallel with bounded concurrency.
    results = await asyncio.gather(*[
        _one_question(q, eg) for q, eg in questions
    ])
    all_results = list(results)

    # Judge cost (tracked on the local router) + query cost (tracked
    # per-run on each retrieve_and_answer invocation, which uses its
    # OWN router instance internally).
    judge_cost = router.total_cost_usd - cost_before
    query_cost = sum(
        run.get("cost_usd", 0.0)
        for qr in all_results
        for run in qr.runs
    )
    total_cost = judge_cost + query_cost
    wall = time.time() - t0

    # ---- aggregate across the question set ----
    def _mean_metric(metric_path: str, expected_gap_filter: bool | None = None) -> dict[str, Any]:
        vals = []
        below = 0
        for qr in all_results:
            if expected_gap_filter is not None and qr.expected_gap != expected_gap_filter:
                continue
            m = qr.metrics.get(metric_path, {})
            if isinstance(m, dict):
                v = m.get("mean") if "mean" in m else m.get("score")
                if v is None:
                    continue
                vals.append(v)
                if v < 0.7:
                    below += 1
        if not vals:
            return {"mean": None, "questions_below_0.7": 0}
        return {"mean": statistics.mean(vals), "questions_below_0.7": below}

    walls = [
        qr.metrics["wall_seconds"]["mean"]
        for qr in all_results
        if qr.metrics.get("wall_seconds", {}).get("mean") is not None
    ]
    aggregate = {
        "comprehensiveness": _mean_metric("comprehensiveness"),
        "no_hallucination":  _mean_metric("no_hallucination"),
        "gap_detection": {
            "mean": _mean_metric("gap_detection")["mean"],
            "by_expected_gap": {
                "true":  _mean_metric("gap_detection", expected_gap_filter=True)["mean"],
                "false": _mean_metric("gap_detection", expected_gap_filter=False)["mean"],
            },
        },
        "consistency": _mean_metric("consistency"),
        "wall_seconds": {
            "mean": statistics.mean(walls) if walls else None,
            "p95":  statistics.quantiles(walls, n=20)[-1] if len(walls) >= 20 else (max(walls) if walls else None),
        },
        "total_cost_usd": total_cost,
        "judge_cost_usd": judge_cost,
        "query_cost_usd": query_cost,
        "wall_seconds_total": wall,
        "questions_count": len(all_results),
    }

    envelope = {
        "config": {
            "questions_path": str(questions_path),
            "mode": mode,
            "runs_per_question": runs_per_question,
            "judge_model": judge_model,
            "use_structured_data": use_structured_data,
        },
        "results": [
            {
                "question": qr.question,
                "expected_gap": qr.expected_gap,
                "runs": qr.runs,
                "metrics": qr.metrics,
            }
            for qr in all_results
        ],
        "aggregate": aggregate,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(envelope, indent=2, default=str))
        print(f"[eval] wrote {output_json}")

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(envelope))
        print(f"[eval] wrote {output_md}")

    _print_summary(envelope)

    # Standing question evaluation (backend/app/evals), from the persisted runs:
    # what each question ties to and what the walk found. Free; fails soft.
    from backend.app.evals.question_eval import report_from_run_ids

    run_ids = [r["retrieval_run_id"] for qr in all_results for r in qr.runs[:1]
               if r.get("retrieval_run_id")]
    if run_ids:
        await report_from_run_ids(run_ids, label="evaluate-queries", mode=mode)
    return envelope


def _render_markdown(envelope: dict[str, Any]) -> str:
    a = envelope["aggregate"]
    cfg = envelope["config"]
    lines = [
        f"# Eval results - {cfg['mode']} (judge={cfg['judge_model']})",
        "",
        f"- Questions: {a['questions_count']}",
        f"- Runs per question: {cfg['runs_per_question']}",
        f"- Total cost: ${a['total_cost_usd']:.4f}",
        f"- Total wall time: {a['wall_seconds_total']:.1f}s",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean | Notes |",
        "|---|---:|---|",
    ]
    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

    for label, key in [
        ("comprehensiveness", "comprehensiveness"),
        ("no_hallucination",  "no_hallucination"),
        ("consistency",       "consistency"),
    ]:
        m = a[key]
        lines.append(
            f"| {label} | {_fmt(m['mean'])} | "
            f"{m['questions_below_0.7']} below 0.7 |"
        )
    g = a["gap_detection"]["by_expected_gap"]
    lines.append(
        f"| gap_detection (overall) | {_fmt(a['gap_detection']['mean'])} | "
        f"expected_gap=true: {_fmt(g.get('true'))}, "
        f"expected_gap=false: {_fmt(g.get('false'))} |"
    )
    w = a["wall_seconds"]
    lines.append(
        f"| wall_seconds | mean={_fmt(w['mean'])}s | p95={_fmt(w['p95'])}s |"
    )
    lines.append("")

    # Flagged questions (any metric < 0.7).
    flagged: list[str] = []
    for r in envelope["results"]:
        m = r["metrics"]
        flags = []
        for k in ("comprehensiveness", "no_hallucination", "gap_detection"):
            mean = m.get(k, {}).get("mean")
            if isinstance(mean, (int, float)) and mean < 0.7:
                flags.append(f"{k}={mean:.2f}")
        cs = m.get("consistency", {}).get("score")
        if isinstance(cs, (int, float)) and cs < 0.7:
            flags.append(f"consistency={cs:.2f}")
        if flags:
            flagged.append(f"- `{r['question']}` :: {', '.join(flags)}")

    if flagged:
        lines.append("## Flagged questions (any metric < 0.7)")
        lines.append("")
        lines.extend(flagged)

    return "\n".join(lines) + "\n"


def _print_summary(envelope: dict[str, Any]) -> None:
    a = envelope["aggregate"]
    cfg = envelope["config"]
    print()
    print("=" * 72)
    print(f"EVAL SUMMARY  -- mode={cfg['mode']}  judge={cfg['judge_model']}")
    print("=" * 72)
    print(f"  questions:            {a['questions_count']}")
    print(f"  runs / question:      {cfg['runs_per_question']}")
    print(f"  total cost:           ${a['total_cost_usd']:.4f}")
    print(f"  total wall:           {a['wall_seconds_total']:.1f}s")
    print()
    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    for label in ("comprehensiveness", "no_hallucination", "consistency"):
        m = a[label]
        print(
            f"  {label:22s} mean={_fmt(m['mean'])}   "
            f"<0.7: {m['questions_below_0.7']}"
        )
    g = a["gap_detection"]
    by = g["by_expected_gap"]
    print(
        f"  gap_detection          mean={_fmt(g['mean'])}   "
        f"expected_gap=true:{_fmt(by.get('true'))} "
        f"expected_gap=false:{_fmt(by.get('false'))}"
    )
    w = a["wall_seconds"]
    print(f"  wall_seconds           mean={_fmt(w['mean'])}s   p95={_fmt(w['p95'])}s")
    print("=" * 72)

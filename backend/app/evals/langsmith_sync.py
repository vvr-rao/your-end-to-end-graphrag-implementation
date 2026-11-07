"""Test sets and evaluation runs in LangSmith.

* A test set becomes a LangSmith dataset named `<langsmith.dataset_prefix>-<name>`:
  one example per question, inputs `{question}`, outputs
  `{gold_documents, supporting_documents}`, metadata `{id, tags, corpus, user}`.
  Pushing adds only the questions the dataset does not already hold (by id).
* A dataset already in LangSmith can be pulled back into a test set.
* An evaluation run becomes a LangSmith experiment on that dataset. Each trace
  carries the answer (when one was generated), the ranked evidence as document
  names and chunk ids, and the scores: Recall@K, Precision@K, MRR@K and the
  coverage metrics. Chunk text is never uploaded.

* A corpus evaluation (no questions, so no dataset) becomes one run in the
  tracing project `<langsmith.corpus_project>` (default
  `<dataset_prefix>-corpus-evals`), named `corpus-eval: <label or database>`:
  inputs `{evaluation, database, label}`, outputs the full report, metadata
  `{user, database, label, evaluation, git_commit, app_config, models_config}`,
  and every number in the report attached as a feedback
  score (`entities`, `summary.chunks_without_entities_or_concepts`,
  `summary.by_entities_and_concepts.0`, ...). The project's runs table and
  monitoring charts then track each metric across builds; filter by
  `metadata.database` or `metadata.label` to follow one corpus. The judge
  evaluations go to the same project as `typing-eval: <label or database>`
  and `ontology-eval: <label or folder>` (the ontology run records the
  prune-expand folder, not a database).

LangSmith is OPTIONAL. With `LANGSMITH_API_KEY` in `.env` (next to the other
API keys), every question evaluation uploads its run automatically; without
one nothing is sent and nothing fails. The rest is the `langsmith:` block of
config.yaml; `user` is stamped on every dataset and experiment, and uploads
are skipped (with a message) until it is set. `eval-questions --no-langsmith`
skips one run.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.evals.testset import TestQuestion, TestSet

DEFAULT_ENDPOINT = "https://api.smith.langchain.com"

# Coverage scores sent with every trace, besides the retrieval metrics.
COVERAGE_KEYS = ("entities", "concepts", "places", "time_periods", "entity_classes",
                 "concept_classes", "place_classes", "time_levels", "walk_entities",
                 "walk_summary_chunks", "walk_summary_chunks_in_evidence")


class LangSmithNotConfiguredError(RuntimeError):
    pass


def settings() -> dict[str, Any]:
    cfg = dict(get_settings().app_config.get("langsmith", {}) or {})
    cfg.setdefault("endpoint", DEFAULT_ENDPOINT)
    cfg.setdefault("dataset_prefix", "graphrag")
    return cfg


def enabled() -> bool:
    """True when an API key is configured: uploads happen automatically."""
    return bool(get_settings().langsmith_api_key)


def maybe_upload(report: dict[str, Any], ts: TestSet, *, mode: str, log=print) -> None:
    """Upload when LangSmith is configured; otherwise do nothing. Never raises."""
    if not enabled():
        return
    try:
        out = upload_experiment(report, ts, mode=mode)
        log(f"[langsmith] dataset {out['dataset']}: {out['uploaded']} trace(s)"
            + (f", experiment {out['experiment']}" if out.get("experiment") else ""))
    except Exception as exc:
        log(f"[langsmith] upload skipped: {type(exc).__name__}: {exc}")


def user() -> str:
    name = str(settings().get("user") or "").strip()
    if not name:
        raise LangSmithNotConfiguredError(
            "set langsmith.user in config/config.yaml (who is running this, e.g. a name)")
    return name


def client() -> Any:
    key = get_settings().langsmith_api_key
    if not key:
        raise LangSmithNotConfiguredError(
            "LANGSMITH_API_KEY is not set: add it to .env next to the other API keys")
    from langsmith import Client

    return Client(api_url=settings()["endpoint"], api_key=key)


def dataset_name(testset_name: str) -> str:
    prefix = str(settings().get("dataset_prefix") or "").strip()
    return f"{prefix}-{testset_name}" if prefix else testset_name


def example_payload(q: TestQuestion, corpus: str, who: str) -> dict[str, Any]:
    return {"inputs": {"question": q.question},
            "outputs": {"gold_documents": q.gold_documents,
                        "supporting_documents": q.supporting_documents,
                        **({"answer_strings": q.answer_strings,
                            "answer_match": q.answer_match} if q.answer_strings else {})},
            "metadata": {"id": q.id, "tags": q.tags, "corpus": corpus, "user": who}}


def push_testset(ts: TestSet, ls: Any | None = None) -> dict[str, Any]:
    """Create the dataset if missing and add the questions it lacks."""
    ls = ls or client()
    who = user()
    name = dataset_name(ts.name)
    updated = 0
    if ls.has_dataset(dataset_name=name):
        ds = ls.read_dataset(dataset_name=name)
        existing = {(ex.metadata or {}).get("id"): ex for ex in ls.list_examples(dataset_id=ds.id)}
        have = set(existing)
        created = False
        # Questions already there whose gold or answer strings changed are updated.
        by_id = {q.id: q for q in ts.questions}
        for qid, ex in existing.items():
            q = by_id.get(qid)
            if q is None:
                continue
            want = example_payload(q, ts.corpus, who)
            if (ex.outputs or {}) != want["outputs"] or (ex.inputs or {}) != want["inputs"]:
                ls.update_example(example_id=ex.id, inputs=want["inputs"],
                                  outputs=want["outputs"], metadata=want["metadata"])
                updated += 1
    else:
        ds = ls.create_dataset(name, description=ts.description or f"Test set {ts.name}",
                               metadata={"corpus": ts.corpus, "format": "graphrag-testset/1",
                                         "user": who})
        have, created = set(), True
    new = [example_payload(q, ts.corpus, who) for q in ts.questions if q.id not in have]
    if new:
        ls.create_examples(dataset_id=ds.id, examples=new)
    return {"dataset": name, "created": created, "added": len(new), "updated": updated,
            "already_there": len(ts.questions) - len(new)}


def list_datasets(ls: Any | None = None) -> list[dict[str, Any]]:
    ls = ls or client()
    return [{"name": d.name, "examples": getattr(d, "example_count", None),
             "description": d.description or ""} for d in ls.list_datasets()]


def pull_testset(name: str, ls: Any | None = None) -> TestSet:
    """A test set rebuilt from a LangSmith dataset (ours, or any dataset whose
    examples have a `question` input)."""
    ls = ls or client()
    ds = ls.read_dataset(dataset_name=name)
    questions: list[TestQuestion] = []
    for i, ex in enumerate(ls.list_examples(dataset_id=ds.id), 1):
        meta, out = ex.metadata or {}, ex.outputs or {}
        q = str((ex.inputs or {}).get("question") or "").strip()
        if not q:
            continue
        questions.append(TestQuestion(
            id=str(meta.get("id") or f"q{i:03d}"), question=q,
            gold_documents=list(out.get("gold_documents") or []),
            supporting_documents=list(out.get("supporting_documents") or []),
            tags=list(meta.get("tags") or []),
            answer_strings=list(out.get("answer_strings") or []),
            answer_match=str(out.get("answer_match") or "all")))
    questions.sort(key=lambda q: q.id)
    prefix = str(settings().get("dataset_prefix") or "")
    short = name[len(prefix) + 1:] if prefix and name.startswith(prefix + "-") else name
    return TestSet(name=short, questions=questions,
                   corpus=str((ds.metadata or {}).get("corpus") or ""),
                   description=ds.description or "")


def trace_outputs(row: dict[str, Any], k: int) -> dict[str, Any]:
    """What one question's trace carries. No chunk text."""
    scores = {key: row.get(key, 0) for key in COVERAGE_KEYS}
    scores["doc_arm_fired"] = int(bool(row.get("doc_arm_fired")))
    rm = row.get("retrieval")
    if rm:
        for key in (f"recall@{k}", f"precision@{k}", f"mrr@{k}"):
            scores[key] = rm[key]
    am = row.get("answers")
    if am:
        for key in ("answer_recall", "answer_found", "answer_mrr", "answer_in_reply"):
            if key in am:
                scores[key] = am[key]
    return {"answer": row.get("answer", ""),
            "ranked_documents": row.get("ranked_documents", []),
            "evidence_chunk_ids": row.get("evidence_chunk_ids", []),
            "gold_found": (rm or {}).get("gold_found", []),
            "gold_missed": (rm or {}).get("gold_missed", []),
            "answers_found": (am or {}).get("answers_found", []),
            "answers_missed": (am or {}).get("answers_missed", []),
            "scores": scores}


def _scores_evaluator(run: Any, example: Any) -> dict[str, Any]:
    scores = ((run.outputs or {}).get("scores") or {})
    return {"results": [{"key": key, "score": value} for key, value in scores.items()]}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def upload_experiment(report: dict[str, Any], ts: TestSet, *, mode: str,
                      ls: Any | None = None) -> dict[str, Any]:
    """One LangSmith experiment on the test set's dataset; one trace per question
    in the report. Pushes the test set first if the dataset lacks questions."""
    from langsmith.evaluation import evaluate

    from backend.app.evals.common import database_name

    ls = ls or client()
    who = user()
    push = push_testset(ts, ls)
    k = report["summary"]["k"]
    by_question = {r["question"]: trace_outputs(r, k) for r in report["questions"]
                   if not r.get("missing_trace")}
    examples = [ex for ex in ls.list_examples(dataset_name=push["dataset"])
                if (ex.inputs or {}).get("question") in by_question]
    if not examples:
        return {"dataset": push["dataset"], "uploaded": 0}

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return by_question[inputs["question"]]

    results = evaluate(
        target, data=examples, evaluators=[_scores_evaluator], client=ls,
        experiment_prefix=f"{who}-{ts.name}-{mode}",
        metadata={"user": who, "testset": ts.name, "corpus": ts.corpus, "mode": mode,
                  "k": k, "database": database_name(), "git_commit": _git_commit(),
                  "app_config": os.environ.get("APP_CONFIG_PATH", "config/config.yaml"),
                  "models_config": os.environ.get("MODELS_CONFIG_PATH", "config/models.yaml")},
        max_concurrency=4)
    return {"dataset": push["dataset"], "uploaded": len(examples),
            "experiment": getattr(results, "experiment_name", "")}


# ---------------------------------------------------------------- corpus evaluations


def corpus_project() -> str:
    cfg = settings()
    name = str(cfg.get("corpus_project") or "").strip()
    if name:
        return name
    prefix = str(cfg.get("dataset_prefix") or "").strip()
    return f"{prefix}-corpus-evals" if prefix else "corpus-evals"


def corpus_scores(report: dict[str, Any]) -> dict[str, float]:
    """Every number in a corpus report as a flat {feedback key: score}.
    Lists (the largest classes) are left in the run outputs only."""
    out: dict[str, float] = {}
    for key in ("entities", "places", "concepts"):
        out[key] = report.get(key, 0)
    for ckind, c in (report.get("chunks") or {}).items():
        for key, value in c.items():
            if key == "chunks_by_count":
                for what, b in value.items():
                    for bucket, n in b.items():
                        out[f"{ckind}.by_{what}.{bucket}"] = n
            elif isinstance(value, int | float):
                out[f"{ckind}.{key}"] = value
    for key, value in (report.get("relationships") or {}).items():
        out[f"relationships.{key}"] = value
    for side, c in (report.get("classes") or {}).items():
        for key, value in c.items():
            if isinstance(value, int | float):
                out[f"classes.{side}.{key}"] = value
    for side, b in (report.get("relationships_per_node") or {}).items():
        for bucket, n in b.items():
            out[f"relationships_per_node.{side}.{bucket}"] = n
    return out


def _tracing_client() -> Any:
    key = get_settings().langsmith_api_key
    if not key:
        raise LangSmithNotConfiguredError(
            "LANGSMITH_API_KEY is not set: add it to .env next to the other API keys")
    from langsmith import Client

    # Synchronous posting: the feedback must find its run.
    return Client(api_url=settings()["endpoint"], api_key=key, auto_batch_tracing=False)


def upload_eval_run(kind: str, report: dict[str, Any], scores: dict[str, float], *,
                    label: str = "", inputs: dict[str, Any] | None = None,
                    ls: Any | None = None) -> dict[str, Any]:
    """One run in the corpus-eval tracing project, with every score as feedback.
    `kind` is corpus / typing / ontology (a tag, and the run-name suffix)."""
    from backend.app.evals.common import database_name

    ls = ls or _tracing_client()
    who = user()
    # What was evaluated: a database (corpus, typing) or an ontology folder.
    target = (Path(str(report.get("folder"))).name if kind == "ontology"
              else database_name())
    where = "folder" if kind == "ontology" else "database"
    project = corpus_project()
    if ls.has_project(project):
        proj = ls.read_project(project_name=project)
    else:
        proj = ls.create_project(project, description=(
            "GraphRAG corpus evaluations (eval-corpus, eval-typing, eval-ontology): one "
            "run per evaluation, every metric as a feedback score."))
    run_id = uuid.uuid4()
    now = datetime.now(UTC)
    name = f"{kind}-eval: {label or target}"
    ls.create_run(name, inputs or {"evaluation": kind, where: target, "label": label}, "chain",
                  project_name=project, id=run_id, start_time=now, end_time=now,
                  outputs=report,
                  extra={"metadata": {
                      "user": who, where: target, "label": label, "evaluation": kind,
                      "git_commit": _git_commit(),
                      "app_config": os.environ.get("APP_CONFIG_PATH", "config/config.yaml"),
                      "models_config": os.environ.get("MODELS_CONFIG_PATH",
                                                      "config/models.yaml")}},
                  tags=[f"{kind}-eval", target] + ([label] if label else []))
    for key, value in scores.items():
        ls.create_feedback(run_id, key, score=value, session_id=proj.id)
    return {"project": project, "run": name, "run_id": str(run_id), "scores": len(scores)}


def upload_corpus_eval(report: dict[str, Any], *, label: str = "",
                       ls: Any | None = None) -> dict[str, Any]:
    """One run in the corpus-eval tracing project, with every metric as feedback."""
    return upload_eval_run("corpus", report, corpus_scores(report), label=label, ls=ls)


def maybe_upload_judge(report: dict[str, Any], *, label: str = "", log=print) -> None:
    """eval-typing / eval-ontology: upload when configured. Never raises."""
    if not enabled():
        return
    from backend.app.evals.judge_evals import judge_scores

    try:
        out = upload_eval_run(report["evaluation"], report, judge_scores(report), label=label)
        log(f"[langsmith] project {out['project']}: run {out['run']} "
            f"with {out['scores']} metric(s)")
    except Exception as exc:
        log(f"[langsmith] upload skipped: {type(exc).__name__}: {exc}")


def maybe_upload_corpus(report: dict[str, Any], *, label: str = "", log=print) -> None:
    """Upload when LangSmith is configured; otherwise do nothing. Never raises."""
    if not enabled():
        return
    try:
        out = upload_corpus_eval(report, label=label)
        log(f"[langsmith] project {out['project']}: run {out['run']} "
            f"with {out['scores']} metric(s)")
    except Exception as exc:
        log(f"[langsmith] upload skipped: {type(exc).__name__}: {exc}")

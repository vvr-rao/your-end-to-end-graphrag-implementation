"""Question evaluation: what each question ties to in the graph, what the graph
walk finds for it, and -- when the test set names gold documents -- Recall@K,
Precision@K and MRR@K over the documents retrieval returned.

Three ways in:
  * a test set or question file (`eval-questions FILE`): runs retrieval for each
    question, without the answer call unless `--answer`. Costs the question
    parse only (a fraction of a cent per question on the cheap presets).
  * finished runs (`eval-questions --run-ids ...` or `--logs ...`): reads the
    trace every persisted run stores in `retrieval_plan.eval_trace` and the
    ranked evidence in `retrieval_evidence`. Free.
  * any test runner holding `RetrievalResult`s: `report_from_results`.

Metric 1 -- what the question ties to (its seeds): named entities, concepts,
  places (seeded, or named in the question text) and time periods; and their
  classes: entity classes, concept classes, place classes, time levels. Also the
  ontology classes the parse matched directly and the concept-area classes its
  terms name.
Metric 2 -- what the graph walk found: entities the walk reached, the SUMMARY
  chunks those entities are mentioned in, and how many reached the evidence.
Retrieval metrics -- see `retrieval_metrics.py`.
Also recorded per question: whether the document arm fired and why, and what
the full-text bridge's document limit cut, by the route that brought each
document in. deep_research reports the final round; round 1 is kept in JSON.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.db.session import session_scope
from backend.app.evals.common import entity_kinds, mean, median
from backend.app.evals.retrieval_metrics import (
    answer_metrics,
    configured_k,
    ranked_documents,
    retrieval_metrics,
)
from backend.app.evals.testset import TestSet

_SCORES = ("entities", "concepts", "places", "time_periods", "entity_classes",
           "concept_classes", "place_classes", "time_levels", "matched_classes",
           "concept_area_classes", "walk_entities", "walk_summary_chunks",
           "walk_summary_chunks_in_evidence")


def read_questions(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def run_ids_from_logs(paths: list[Path]) -> list[str]:
    """RUN_ID lines printed by `query`, in file order."""
    return [r for r, _ in runs_from_logs(paths)]


def runs_from_logs(paths: list[Path]) -> list[tuple[str, str]]:
    """(run id, answer) from `query` output files, in file order."""
    out: list[tuple[str, str]] = []
    for p in paths:
        s = p.read_text(encoding="utf-8")
        m = re.search(r"^RUN_ID:\s+(\S+)", s, re.M)
        if not m:
            continue
        a = re.search(r"^ANSWER:\n(.*?)^EVIDENCE", s, re.S | re.M)
        out.append((m.group(1), a.group(1).strip() if a else ""))
    return out


def trace_from_result(question: str, result: Any) -> dict[str, Any]:
    """The evaluation trace of one in-process `RetrievalResult`."""
    chunks = sorted((ev for ev in result.evidence if ev.get("kind") == "chunk"),
                    key=lambda ev: ev.get("rank") or 0)
    return {"question": question, "answer": result.answer or "",
            "run_id": str(result.retrieval_run_id) if result.retrieval_run_id else None,
            "evidence_chunk_ids": [str(ev["node_id"]) for ev in chunks],
            **(result.parsed.get("eval_trace") or {})}


async def traces_by_retrieval(questions: list[str], *, mode: str = "simple_qa",
                              answer: bool = False, log=print
                              ) -> tuple[list[dict[str, Any]], float]:
    from backend.app.services.retrieval import retrieve_and_answer

    traces: list[dict[str, Any]] = []
    cost = 0.0
    for i, q in enumerate(questions, 1):
        res = await retrieve_and_answer(q, mode=mode, persist=False, answer_question=answer,
                                        multi_round=answer)
        cost += res.cost_usd
        traces.append(trace_from_result(q, res))
        log(f"[eval-questions] {i}/{len(questions)} traced")
    return traces, cost


def traces_in_batches(questions: list[str], *, mode: str = "simple_qa",
                      answer: bool = False, batch_size: int = 25,
                      traces_dir: Path, log=print) -> tuple[list[dict[str, Any]], float]:
    """Run retrieval `batch_size` questions per child process, saving each batch's
    traces under `traces_dir`. One process for a whole test set holds its memory
    for hours (two at once took down a 2.7 GB machine); a crash here loses one
    batch, and re-running with the same `traces_dir` skips finished batches."""
    import subprocess
    import sys

    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest = traces_dir / "questions.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != questions:
            raise ValueError(f"{traces_dir} holds batches of a different question list")
    else:
        manifest.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    size = max(1, batch_size)
    traces: list[dict[str, Any]] = []
    cost = 0.0
    n_batches = (len(questions) + size - 1) // size
    for b in range(n_batches):
        out = traces_dir / f"batch-{b + 1:03d}.json"
        if not out.exists():
            batch = questions[b * size:(b + 1) * size]
            qfile = traces_dir / f"batch-{b + 1:03d}.questions.json"
            qfile.write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
            cmd = [sys.executable, "-m", "backend.app.cli", "eval-questions",
                   "--batch-questions", str(qfile), "--batch-out", str(out), "--mode", mode]
            if answer:
                cmd.append("--answer")
            log(f"[eval-questions] batch {b + 1}/{n_batches} "
                f"(questions {b * size + 1}-{b * size + len(batch)})")
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0 or not out.exists():
                raise RuntimeError(f"batch {b + 1} failed (exit {rc}); re-run with "
                                   f"--traces-dir {traces_dir} to resume")
        else:
            log(f"[eval-questions] batch {b + 1}/{n_batches} already done")
        data = json.loads(out.read_text(encoding="utf-8"))
        traces += data["traces"]
        cost += data["cost_usd"]
    return traces, cost


async def run_batch_file(questions_file: Path, out: Path, *, mode: str,
                         answer: bool) -> None:
    """Child side of traces_in_batches: trace one batch and write it atomically."""
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    traces, cost = await traces_by_retrieval(questions, mode=mode, answer=answer)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"traces": traces, "cost_usd": cost}, default=str,
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


async def traces_from_runs(runs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Traces of finished runs, from (run id, answer) pairs."""
    ids = [r for r, _ in runs]
    async with session_scope() as session:
        rows = await session.execute(text("""
            SELECT id::text, resolved_query, retrieval_plan->'eval_trace', retrieval_mode
              FROM graphrag.retrieval_runs WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": ids})
        found = {rid: (q, t, m) for rid, q, t, m in rows.all()}
        ev = await session.execute(text("""
            SELECT retrieval_run_id::text, evidence_uuid::text
              FROM graphrag.retrieval_evidence
             WHERE retrieval_run_id = ANY(CAST(:ids AS uuid[])) AND evidence_kind = 'chunk'
             ORDER BY retrieval_run_id, rank
        """), {"ids": ids})
        evidence: dict[str, list[str]] = {}
        for rid, cid in ev.all():
            evidence.setdefault(rid, []).append(cid)
    out = []
    for rid, answer in runs:
        q, t, mode = found.get(rid, ("", None, None))
        out.append({"question": q, "run_id": rid, "answer": answer, "mode": mode,
                    "missing_trace": t is None, "evidence_chunk_ids": evidence.get(rid, []),
                    **(t or {})})
    return out


async def describe(traces: list[dict[str, Any]], testset: TestSet | None = None,
                   k: int | None = None) -> dict[str, Any]:
    """Per-question metrics and the summary over the set."""
    k = k or configured_k()
    gold_by_q = testset.by_question() if testset else {}
    chunk_ids = {c for t in traces for c in t.get("evidence_chunk_ids") or []}
    async with session_scope() as session:
        kinds = await entity_kinds(session)
        levels = {str(tid): lvl for tid, lvl in (await session.execute(
            text("SELECT id, time_level FROM graphrag.time_instances"))).all()}
        names = {str(eid): n for eid, n in (await session.execute(
            text("SELECT id, name FROM graphrag.entities WHERE status = 'ACTIVE'"))).all()}
        doc_of: dict[str, str] = {}
        if chunk_ids:
            doc_of = {str(c): f for c, f in (await session.execute(text("""
                SELECT c.id, d.file_name FROM graphrag.chunks c
                  JOIN graphrag.documents d ON d.id = c.document_id
                 WHERE c.id = ANY(CAST(:ids AS uuid[]))
            """), {"ids": list(chunk_ids)})).all()}
        # Chunk text only for questions that carry answer strings; it is used
        # here and never leaves this process (nothing is uploaded).
        need_text = {c for t in traces
                     if (tq := gold_by_q.get(t.get("question", ""))) and tq.answer_strings
                     for c in t.get("evidence_chunk_ids") or []}
        text_of: dict[str, str] = {}
        if need_text:
            text_of = {str(c): x for c, x in (await session.execute(text("""
                SELECT id, text FROM graphrag.chunks WHERE id = ANY(CAST(:ids AS uuid[]))
            """), {"ids": list(need_text)})).all()}
    rows = [_row(t, gold_by_q.get(t.get("question", "")), kinds, levels, names, doc_of, k,
                 text_of)
            for t in traces]
    return {"summary": _summary(rows, k, testset), "questions": rows}


def _row(t: dict[str, Any], tq: Any, kinds: dict, levels: dict, names: dict,
         doc_of: dict, k: int, text_of: dict | None = None) -> dict[str, Any]:
    by_kind: dict[str, list[str]] = {"named": [], "concept": [], "place": []}
    classes: dict[str, set] = {"named": set(), "concept": set(), "place": set()}
    for eid in list(dict.fromkeys((t.get("seed_entities") or []) + (t.get("named_places") or []))):
        kd = kinds.get(uuid.UUID(eid))
        if not kd:
            continue
        by_kind[kd[0]].append(eid)
        classes[kd[0]].add(kd[1])
    times = t.get("seed_times") or []
    arm = t.get("doc_arm") or {}
    ranked = ranked_documents([doc_of.get(c) for c in t.get("evidence_chunk_ids") or []])
    gold = list(tq.gold_documents) if tq else []
    row = {
        "id": tq.id if tq else None,
        "question": t.get("question", ""),
        "run_id": t.get("run_id"),
        "mode": t.get("mode"),
        "answer": t.get("answer", ""),
        "missing_trace": bool(t.get("missing_trace")),
        "seeding": t.get("seeding"),
        "entities": len(by_kind["named"]),
        "concepts": len(by_kind["concept"]),
        "places": len(by_kind["place"]),
        "places_inside": t.get("contained_places", 0),
        "time_periods": len(times),
        "entity_classes": len(classes["named"]),
        "concept_classes": len(classes["concept"]),
        "place_classes": len(classes["place"]),
        "time_levels": len({levels.get(x) for x in times if levels.get(x)}),
        "matched_classes": len(t.get("matched_classes") or []),
        "concept_area_classes": len(t.get("concept_area_classes") or []),
        "seed_names": {kk: [names.get(e, e) for e in v][:15] for kk, v in by_kind.items()},
        "walk_entities": t.get("walk_entities", 0),
        # Runs traced before the summary-only count existed fall back to all kinds.
        "walk_summary_chunks": t.get("walk_summary_chunks", t.get("walk_chunks", 0)),
        "walk_summary_chunks_in_evidence": t.get(
            "walk_summary_chunks_in_evidence", t.get("walk_chunks_in_evidence", 0)),
        "walk_chunks_all_kinds": t.get("walk_chunks", 0),
        "evidence_chunks": t.get("evidence_chunks", 0),
        "doc_arm_fired": bool(arm.get("fired")),
        "doc_arm_reason": arm.get("reason", ""),
        "doc_arm_documents": arm.get("documents", 0),
        "bridge": t.get("fulltext_bridge") or {},
        "ranked_documents": ranked[:max(k, 20)],
        "evidence_chunk_ids": t.get("evidence_chunk_ids") or [],
        "gold_documents": gold,
        "retrieval": retrieval_metrics(ranked, gold, k) if gold else None,
        "answers": (answer_metrics(
            [(text_of or {}).get(c, "") for c in t.get("evidence_chunk_ids") or []],
            tq.answer_strings, match=tq.answer_match, reply=t.get("answer", ""))
            if tq and tq.answer_strings else None),
    }
    if isinstance(t.get("round1"), dict):
        r1 = t["round1"]
        row["round1"] = {"question": r1.get("question"),
                         "seed_entities": [names.get(e, e) for e in r1.get("seed_entities") or []],
                         "seed_times": len(r1.get("seed_times") or []),
                         "walk_summary_chunks": r1.get("walk_summary_chunks",
                                                       r1.get("walk_chunks", 0))}
    return row


def _summary(rows: list[dict[str, Any]], k: int, testset: TestSet | None) -> dict[str, Any]:
    ok = [r for r in rows if not r["missing_trace"]]
    scored = [r["retrieval"] for r in rows if r["retrieval"]]
    return {
        "testset": testset.name if testset else None,
        "k": k,
        "questions": len(rows),
        "without_trace": len(rows) - len(ok),
        "mean": {key: mean([r[key] for r in ok]) for key in _SCORES},
        "median_walk_summary_chunks": median([r["walk_summary_chunks"] for r in ok]),
        "with_no_entity_concept_place_or_time": sum(
            1 for r in ok if not (r["entities"] or r["concepts"] or r["places"] or r["time_periods"])),
        "with_no_entities": sum(1 for r in ok if not r["entities"]),
        "with_no_concepts": sum(1 for r in ok if not r["concepts"]),
        "with_places": sum(1 for r in ok if r["places"]),
        "with_time": sum(1 for r in ok if r["time_periods"]),
        "with_no_walk_chunks": sum(1 for r in ok if not r["walk_summary_chunks"]),
        "doc_arm_fired": sum(1 for r in ok if r["doc_arm_fired"]),
        "bridge_cut_graph_documents": sum(
            1 for r in ok if (r["bridge"] or {}).get("graph_documents_cut")),
        "retrieval": {
            "scored_questions": len(scored),
            "without_gold": len(rows) - len(scored),
            f"recall@{k}": mean([m[f"recall@{k}"] for m in scored]),
            f"precision@{k}": mean([m[f"precision@{k}"] for m in scored]),
            f"mrr@{k}": mean([m[f"mrr@{k}"] for m in scored]),
        },
        "answers": _answer_summary(rows),
    }


def _answer_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Means over the questions with answer strings; None when none have any."""
    scored = [r["answers"] for r in rows if r.get("answers")]
    if not scored:
        return None
    out = {"scored_questions": len(scored),
           "without_answer_strings": len(rows) - len(scored),
           "answer_recall": mean([a["answer_recall"] for a in scored]),
           "answer_found": mean([a["answer_found"] for a in scored]),
           "answer_mrr": mean([a["answer_mrr"] for a in scored])}
    replies = [a["answer_in_reply"] for a in scored if "answer_in_reply" in a]
    if replies:
        out["answer_in_reply"] = mean(replies)
    return out


def format_question_eval(report: dict[str, Any]) -> str:
    s = report["summary"]
    m = s["mean"]
    k = s.get("k", 10)
    ret = s.get("retrieval") or {}
    lines = [
        f"{s['questions']} questions ({s['without_trace']} without a trace)"
        + (f", test set {s['testset']}" if s.get("testset") else ""),
        f"RETRIEVAL (K={k}, {ret.get('scored_questions', 0)} questions with gold, "
        f"{ret.get('without_gold', 0)} without): recall@{k} {ret.get(f'recall@{k}', 0)}, "
        f"precision@{k} {ret.get(f'precision@{k}', 0)}, mrr@{k} {ret.get(f'mrr@{k}', 0)}",
        *([f"ANSWERS IN EVIDENCE ({ans['scored_questions']} questions with answer strings, "
           f"{ans['without_answer_strings']} without): recall {ans['answer_recall']}, "
           f"all found {ans['answer_found']}, mrr {ans['answer_mrr']}"
           + (f", in reply {ans['answer_in_reply']}" if "answer_in_reply" in ans else "")]
          if (ans := s.get("answers")) else []),
        "TIES TO (mean per question): "
        f"entities {m['entities']}, concepts {m['concepts']}, places {m['places']}, "
        f"time periods {m['time_periods']}; entity classes {m['entity_classes']}, "
        f"concept classes {m['concept_classes']}, place classes {m['place_classes']}, "
        f"time levels {m['time_levels']}; matched classes {m['matched_classes']}, "
        f"concept-area classes {m['concept_area_classes']}",
        f"  no entity/concept/place/time {s['with_no_entity_concept_place_or_time']}, "
        f"no entities {s['with_no_entities']}, no concepts {s['with_no_concepts']}, "
        f"with places {s['with_places']}, with time {s['with_time']}",
        f"GRAPH WALK: entities reached {m['walk_entities']}, summary chunks "
        f"{m['walk_summary_chunks']} (median {s['median_walk_summary_chunks']}), in final "
        f"evidence {m['walk_summary_chunks_in_evidence']}; no walk chunks {s['with_no_walk_chunks']}",
        f"document arm fired {s['doc_arm_fired']}; bridge limit cut graph documents on "
        f"{s['bridge_cut_graph_documents']}",
        "",
        "  #  ent con plc tim | eCls cCls pCls tLvl | walkEnt walkChk inEv | docArm |"
        f"  R@{k}  P@{k} MRR | ans | bridge cand/graph/armOnly/kept/graphCut",
    ]
    for i, r in enumerate(report["questions"], 1):
        b = r["bridge"] or {}
        rm = r.get("retrieval")
        score = (f"{rm[f'recall@{k}']:5.2f} {rm[f'precision@{k}']:5.2f} {rm[f'mrr@{k}']:4.2f}"
                 if rm else "    -     -    -")
        lines.append(
            f"{i:3} {r['entities']:4} {r['concepts']:3} {r['places']:3} {r['time_periods']:3} |"
            f" {r['entity_classes']:4} {r['concept_classes']:4} {r['place_classes']:4}"
            f" {r['time_levels']:4} | {r['walk_entities']:7} {r['walk_summary_chunks']:7}"
            f" {r['walk_summary_chunks_in_evidence']:4} | {'yes' if r['doc_arm_fired'] else 'no ':6} |"
            f" {score} | {_ans_cell(r.get('answers'))} | {b.get('candidate_documents', '-')}/{b.get('graph_documents', '-')}"
            f"/{b.get('doc_arm_only_documents', '-')}/{b.get('kept', '-')}"
            f"/{b.get('graph_documents_cut', '-')}  {r['question'][:60]}")
    return "\n".join(lines)


def _ans_cell(a: dict[str, Any] | None) -> str:
    if not a:
        return "  -"
    n = len(a["answers_found"]) + len(a["answers_missed"])
    return f"{len(a['answers_found'])}/{n}".rjust(3)


def _testset_for(questions: list[str], testset: TestSet | None, label: str) -> TestSet:
    """The test set to upload under: the given one, or the questions themselves
    (no gold) named after the runner."""
    if testset is not None:
        return testset
    import hashlib

    from backend.app.evals.testset import TestQuestion

    # Ids from the question text, not the position: the same runner label is
    # reused across runs, and a positional id would collide with a different
    # question already in the dataset, so the new one would never be added.
    unique = list(dict.fromkeys(q for q in questions if q))
    return TestSet(name=label, questions=[
        TestQuestion(id="q-" + hashlib.sha1(q.encode()).hexdigest()[:12], question=q)
        for q in unique])


async def report_from_results(pairs: list[tuple[str, Any]], *, testset: TestSet | None = None,
                              k: int | None = None, label: str = "questions",
                              mode: str = "simple_qa",
                              out: Path | None = None, log=print) -> dict[str, Any] | None:
    """For any test runner: (question, RetrievalResult) pairs -> printed and saved
    report. Fails soft -- a report error never fails the test run."""
    from backend.app.evals.common import save_report

    try:
        report = await describe([trace_from_result(q, r) for q, r in pairs], testset, k)
        log("QUESTION EVALUATION")
        log(format_question_eval(report))
        log(f"[eval-questions] saved {save_report(label, report, out)}")
    except Exception as exc:
        log(f"[eval-questions] skipped: {type(exc).__name__}: {exc}")
        return None
    from backend.app.evals.langsmith_sync import maybe_upload

    maybe_upload(report, _testset_for([q for q, _ in pairs], testset, label), mode=mode, log=log)
    return report


async def report_from_run_ids(run_ids: list[str], *, testset: TestSet | None = None,
                              k: int | None = None, label: str = "questions",
                              mode: str = "simple_qa",
                              out: Path | None = None, log=print) -> dict[str, Any] | None:
    """For runners that persist their runs: run ids -> printed and saved report.
    Fails soft."""
    from backend.app.evals.common import save_report

    try:
        report = await describe(await traces_from_runs([(r, "") for r in run_ids]), testset, k)
        log("QUESTION EVALUATION")
        log(format_question_eval(report))
        log(f"[eval-questions] saved {save_report(label, report, out)}")
    except Exception as exc:
        log(f"[eval-questions] skipped: {type(exc).__name__}: {exc}")
        return None
    from backend.app.evals.langsmith_sync import maybe_upload

    maybe_upload(report, _testset_for([r["question"] for r in report["questions"]], testset,
                                      label), mode=mode, log=log)
    return report

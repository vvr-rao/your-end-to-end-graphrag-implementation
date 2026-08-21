#!/usr/bin/env python
"""Re-run the retrieval audit question set and report on the D1-D4 fixes.

Read-only against the database (`--no-persist`), so it can be pointed at a
deployed corpus. Costs real LLM spend -- roughly $0.07 per deep_research
question.

Usage:
    uv run python scripts/run_audit_questions.py [--only 2,6,9,18] [--mode deep_research]

Writes a JSON envelope per question plus a summary table, so the run can be
diffed against docs/retrieval-audit-2026-08-19.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.retrieval import retrieve_and_answer  # noqa: E402

# (id, question, what the audit expects now). Ids match the audit document.
QUESTIONS: list[tuple[int, str, str]] = [
    # --- Batch 1: baseline ---
    (1, "Compare the dosing schedules of tirzepatide and dulaglutide",
     "tirzepatide 2.5-15mg, dulaglutide 0.75-4.5mg"),
    (2, "What competitors to Ozempic and Wegovy emerged in the GLP-1 weight-loss market by 2025?",
     "D2+D3: zero invalid citations; 2026 approvals flagged as out of window"),
    (3, "What is Ozempic's active ingredient?", "semaglutide"),
    (4, "How do GLP-1 receptor medications work?", "cAMP/PKA, glucose-dependent insulin secretion"),
    # --- Batch 2: acronym pairs, asked both ways ---
    (5, "Which weight-loss drug is approved for OSA?", "ZEPBOUND"),
    (8, "Which weight-loss drug is approved for obstructive sleep apnea?", "pair of Q5 - must converge"),
    (6, "What is the MRHD used in the animal studies?", "D1: must converge with Q9"),
    (9, "What is the maximum recommended human dose used in animal studies?", "pair of Q6"),
    (7, "How do ACE inhibitors work?", "less angiotensin II, more bradykinin"),
    (10, "How do angiotensin-converting enzyme inhibitors work?", "pair of Q7"),
    # --- Batch 3: typos, coverage, acronyms ---
    (11, "What is the maximum dose of Mounjaro?", "15 mg weekly"),
    (13, "What is the maximum dose of Mounjarro?", "typo - must still give 15 mg"),
    (12, "Compare the dosing schedules of Zepbound and Trulicity", "brand-vocabulary form of Q1"),
    (14, "What are the side effects of semaglutid?", "typo - nausea etc"),
    (15, "What are ARBs used for?", "hypertension, heart failure, CKD"),
    (16, "What is the maximum dose of liraglutide?", "3 mg Saxenda / 1.8 mg Victoza"),
    (17, "What are the main types of diuretics?", "thiazide, loop, K-sparing..."),
    (18, "Which drugs in this corpus contain semaglutide?", "D4: must now include RYBELSUS"),
    (19, "What is the risk of MTC with GLP-1 drugs?", "rodent C-cell tumours, human relevance unknown"),
    (20, "What is the risk of medullary thyroid carcinoma with GLP-1 drugs?", "pair of Q19"),
    (21, "Which drug targets GIP and what does GIP stand for?", "tirzepatide; glucose-dependent insulinotropic polypeptide"),
    # --- Batch 4: trends over time (new) ---
    # These exercise D3 hardest: the parser reduces every relational form
    # ("since", "between", "after", "before") to a bare YEAR_YYYY, so the
    # window can only be honoured by the evidence time tags + the prompt.
    (22, "What have been the clinical studies studying diabetes and hypertension medications since 2020?",
     "TREND: must respect 'since 2020' or say what falls outside"),
    (23, "How has GLP-1 dosing guidance evolved between 2020 and 2025?",
     "TREND: two-sided window"),
    (24, "What safety warnings were added to GLP-1 labels after 2023?",
     "TREND: 'after' relation"),
    (25, "Which cardiovascular outcome trials were reported before 2020?",
     "TREND: 'before' relation - inverse window"),
    # Sub-year granularity. `_ontology_match` regexes only the YEAR out of a
    # time term, so "June 2026" and "Q2 2026" both seed YEAR_2026 -- broad,
    # never wrong. The month can only be honoured downstream, via the
    # per-chunk time tags (82 chunks carry month labels) plus the verbatim
    # term in the prompt. NOTE no chunk links to a quarter node at all, so
    # Q27 requires the model to infer the quarter from month labels.
    (26, "What happened in June 2026 in the GLP-1 market?",
     "MONTH: 13 chunks carry a 'June 2026' tag; seeding is year-level"),
    (27, "What developments occurred in Q2 2026?",
     "QUARTER: zero chunks link to a quarter node - must infer from months"),
    # --- Batch 5: the temporal bug report's own questions ---------------------
    # docs/temporal-retrieval-issue-2026-08-21.md, verbatim. Numbered 100+ so
    # they stay visually distinct from the audit set. t10/t11/t12 and c3 were
    # the failures; the rest were passing and are regression guards.
    (100, "What clinical studies of diabetes and hypertension medications have been done since 2020?",
     "t10 FORWARD - was a false 'no evidence' denial"),
    (101, "How has GLP-1 dosing guidance evolved between 2020 and 2025?",
     "t11 FORWARD - was a false 'no evidence' denial"),
    (102, "What safety warnings were added to GLP-1 labels after 2023?",
     "t12 FORWARD - was a false 'no evidence' denial"),
    (103, "Which cardiovascular outcome trials were reported before 2020?",
     "t13 BACKWARD - was PASS, 9 trials"),
    (104, "What happened in June 2026 in the GLP-1 market?",
     "t14 EXACT LABEL - was PASS"),
    (105, "What developments occurred in Q2 2026?",
     "t15 EXACT LABEL - was PASS"),
    (106, "What safety warnings were added to GLP-1 labels?",
     "c1 NO DATE - was PASS (control)"),
    (107, "What safety warnings were in GLP-1 labels before 2026?",
     "c2 BACKWARD - was PASS (control)"),
    (108, "Which cardiovascular outcome trials were reported after 2020?",
     "c3 FORWARD - was PARTIAL, 1 trial only"),
    (109, "What recent changes were made to the ZEPBOUND prescribing information?",
     "c4 'recent' - was PASS, change-log at rank 3"),
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="deep_research")
    ap.add_argument("--only", default=None, help="Comma-separated question ids.")
    ap.add_argument("--out", default="docs/audit-rerun.jsonl")
    args = ap.parse_args()

    wanted = (
        {int(x) for x in args.only.split(",")} if args.only else None
    )
    qs = [q for q in QUESTIONS if wanted is None or q[0] in wanted]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")

    total_cost = 0.0
    total_invalid = 0
    t0 = time.time()
    rows: list[tuple] = []

    for qid, question, expectation in qs:
        print(f"\n{'=' * 78}\nQ{qid}: {question}\n  expect: {expectation}", flush=True)
        try:
            res = await retrieve_and_answer(
                question, mode=args.mode, persist=False,
            )
        except Exception as exc:
            print(f"  !! FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append((qid, "ERROR", 0, 0.0, 0.0, {}, []))
            continue

        aliases = res.parsed.get("aliases") or {}
        times = res.parsed.get("time_terms") or []
        # Document spread: the concentration defect showed up as every
        # evidence item coming from one document, so track it per query.
        titles = {
            e.get("document_title")
            for e in res.evidence
            if e.get("kind") == "chunk" and e.get("document_title")
        }
        dated = sum(1 for e in res.evidence if e.get("times"))
        total_cost += res.cost_usd
        total_invalid += len(res.invalid_citations)
        rows.append((
            qid, "ok", len(res.evidence), res.cost_usd, res.wall_seconds,
            aliases, res.invalid_citations,
        ))

        print(f"  cost ${res.cost_usd:.4f}  wall {res.wall_seconds:.0f}s  "
              f"evidence {len(res.evidence)}  docs {len(titles)}  "
              f"dated {dated}", flush=True)
        if aliases:
            print("  aliases: " + "; ".join(
                f"{k} -> {', '.join(v[:3])}" for k, v in aliases.items()), flush=True)
        if times:
            print(f"  time_terms: {times}", flush=True)
        if res.invalid_citations:
            print(f"  !! INVALID CITATIONS: {res.invalid_citations}", flush=True)
        print(f"  --- answer ---\n{res.answer}", flush=True)

        fh.write(json.dumps({
            "id": qid, "question": question, "expectation": expectation,
            "answer": res.answer, "evidence_count": len(res.evidence),
            "aliases": aliases, "time_terms": times,
            "invalid_citations": res.invalid_citations,
            "cost_usd": res.cost_usd, "wall_seconds": res.wall_seconds,
            "evidence_iris": [e.get("iri") for e in res.evidence],
            "distinct_documents": len(titles),
            "dated_evidence": dated,
        }, default=str) + "\n")
        fh.flush()

    fh.close()
    print(f"\n{'=' * 78}\nSUMMARY  ({len(qs)} questions, {time.time() - t0:.0f}s)")
    print(f"{'Q':>3}  {'st':<5} {'ev':>3} {'cost':>8} {'wall':>6}  aliases / invalid")
    for qid, st, ev, cost, wall, aliases, invalid in rows:
        flag = f"  !!{len(invalid)} INVALID" if invalid else ""
        alias_note = f"  aliases={len(aliases)}" if aliases else ""
        print(f"{qid:>3}  {st:<5} {ev:>3} ${cost:>7.4f} {wall:>5.0f}s{alias_note}{flag}")
    print(f"\ntotal cost ${total_cost:.4f}   invalid citations: {total_invalid}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

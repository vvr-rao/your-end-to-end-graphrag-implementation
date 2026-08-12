#!/usr/bin/env python3
"""End-to-end sweep of every CLI subcommand against a small corpus.

Runs the full pipeline in dependency order, then the read-only and
lifecycle commands, recording exit code + wall time + cost per step. One
failure does not stop the sweep -- the point is to learn which commands are
broken, not to stop at the first.

    uv run python scripts/e2e_sweep.py <label> <merge_dir> <docs_dir>

Excluded on purpose:
  render-*        would mutate the deployed Render services
  db-downgrade    known broken: 0005's downgrade violates a CHECK constraint
  build           == merge + prune-expand; doubles the most expensive step
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ["uv", "run", "python", "-m", "backend.app.cli"]

COST_RE = re.compile(r"\$\s*([0-9]+\.[0-9]+)")


def run(name: str, args: list[str], *, timeout: int = 3600) -> dict:
    t0 = time.monotonic()
    print(f"\n{'=' * 70}\n>>> {name}\n    {' '.join(args)}\n{'=' * 70}", flush=True)
    try:
        p = subprocess.run(CLI + args, cwd=ROOT, capture_output=True,
                           text=True, timeout=timeout)
        out, code = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", 124
    dt = time.monotonic() - t0
    # Keep HEAD as well as tail: config echoes ("concurrency = 32") print first
    # and were being truncated away, while summary lines print last.
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) > 40:
        shown = lines[:12] + [f"    ... {len(lines) - 32} lines omitted ..."] + lines[-20:]
    else:
        shown = lines
    print("\n".join(shown), flush=True)
    costs = [float(c) for c in COST_RE.findall(out)]
    return {"name": name, "exit": code, "seconds": round(dt, 1),
            "cost_seen": max(costs) if costs else 0.0,
            # Tail of the RAW output, not the elided view -- on a failure the
            # traceback is the last thing printed and must survive intact.
            "err": "" if code == 0 else "\n".join(lines)[-600:]}


def capture(args: list[str]) -> str:
    p = subprocess.run(CLI + args, cwd=ROOT, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def main() -> int:
    label, merge_dir, docs = sys.argv[1], sys.argv[2], sys.argv[3]
    results: list[dict] = []

    # --- ontology side -----------------------------------------------------
    results.append(run("db-migrate", ["db-migrate"]))
    results.append(run("db-status", ["db-status"]))
    results.append(run("db-size", ["db-size"]))
    results.append(run("summarize-descriptions",
                       ["summarize-descriptions", "--input", merge_dir,
                        "--max-cost-usd", "2.0"]))
    results.append(run("prune-expand",
                       ["prune-expand", "--input", merge_dir, "--documents", docs,
                        "--max-cost-usd", "12.0", "--tables"], timeout=5400))

    latest = subprocess.run(["uv", "run", "python", "scripts/latest_run_dir.py",
                             "prune-expand"], cwd=ROOT, capture_output=True, text=True)
    pe_dir = (latest.stdout or "").strip().splitlines()[-1] if latest.stdout.strip() else ""
    print(f"\n[sweep] prune-expand folder = {pe_dir}", flush=True)

    results.append(run("audit-classifications",
                       ["audit-classifications", "--input", pe_dir]))
    results.append(run("repair-output", ["repair-output", "--input", pe_dir]))

    # --- database + corpus -------------------------------------------------
    results.append(run("clear-corpus", ["clear-corpus", "--yes"]))
    results.append(run("db-init(replace)",
                       ["db-init", "--input", pe_dir, "--mode", "replace", "--yes"],
                       timeout=2400))
    results.append(run("import-ontology",
                       ["import-ontology", "--input", pe_dir, "--limit", "5"]))
    results.append(run("register-documents",
                       ["register-documents", "--input", docs, "--limit", "4",
                        "--tables"], timeout=3600))
    results.append(run("list-documents", ["list-documents"]))
    results.append(run("extract-entities",
                       ["extract-entities", "--max-cost-usd", "5.0"], timeout=3600))
    results.append(run("enrich-time", ["enrich-time"], timeout=1800))
    results.append(run("generate-artifacts",
                       ["generate-artifacts", "--max-cost-usd", "5.0"], timeout=3600))

    # --- retrieval ---------------------------------------------------------
    q = "Which GLP-1 drugs are described and what are their main side effects?"
    for mode in ("simple_qa", "deep_research", "artifact_only"):
        results.append(run(f"query({mode})",
                           ["query", q, "--mode", mode, "--max-cost-usd", "2.0"],
                           timeout=1800))

    # --- conversation ------------------------------------------------------
    conv = capture(["conversation", "start", "--title", f"e2e {label}", "--json"])
    m = re.search(r"(https?://[^\s\"',}\]]+#[^\s\"',}\]]+)", conv)
    iri = m.group(1) if m else ""
    print(f"\n[sweep] conversation iri = {iri or '(NOT FOUND)'}", flush=True)
    results.append({"name": "conversation start", "exit": 0 if iri else 1,
                    "seconds": 0.0, "cost_seen": 0.0,
                    "err": "" if iri else conv[-300:]})
    if iri:
        results.append(run("conversation turn",
                           ["conversation", "turn", "--conv", iri, "What about dosing?",
                            "--mode", "simple_qa", "--max-cost-usd", "2.0"], timeout=1800))
        results.append(run("conversation show", ["conversation", "show", "--conv", iri]))

    # --- lifecycle ---------------------------------------------------------
    # list-documents prints status/version/chunks/title -- NO iri column -- so
    # the IRI has to come from the DB. Scraping stdout silently yielded "" and
    # skipped the two lifecycle commands with no failure recorded.
    doc_iri = subprocess.run(
        ["uv", "run", "python", "-c",
         "import asyncio;from sqlalchemy import text;"
         "from backend.app.db.session import session_scope\n"
         "async def m():\n"
         "    async with session_scope() as s:\n"
         "        r=await s.execute(text(\"SELECT document_identifier FROM graphrag.documents "
         "WHERE status='ACTIVE' ORDER BY created_at LIMIT 1\"))\n"
         "        row=r.first()\n"
         "        print(row[0] if row else '')\n"
         "asyncio.run(m())"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    doc_iri = doc_iri[-1].strip() if doc_iri else ""
    print(f"\n[sweep] document iri = {doc_iri or '(NOT FOUND)'}", flush=True)
    if doc_iri:
        results.append(run("delete-document(soft)",
                           ["delete-document", "--iri", doc_iri]))
        results.append(run("regenerate-stale-artifacts",
                           ["regenerate-stale-artifacts", "--max-cost-usd", "3.0"],
                           timeout=2400))

    results.append(run("db-size(final)", ["db-size"]))

    # --- report ------------------------------------------------------------
    out = ROOT / f".e2e_{label}.json"
    out.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if r["exit"] == 0]
    bad = [r for r in results if r["exit"] != 0]
    print(f"\n\n{'=' * 70}\nE2E SWEEP [{label}]: {len(ok)}/{len(results)} passed\n{'=' * 70}")
    print(f"{'cmd':<30} {'exit':>4} {'secs':>8}")
    for r in results:
        flag = "" if r["exit"] == 0 else "   <-- FAIL"
        print(f"{r['name']:<30} {r['exit']:>4} {r['seconds']:>8.1f}{flag}")
    if bad:
        print("\nFAILURES:")
        for r in bad:
            print(f"\n--- {r['name']} (exit {r['exit']})\n{r['err']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

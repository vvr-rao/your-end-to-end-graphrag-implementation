---
name: run-prune-expand
description: >
  Merge the chosen ontologies with VIAO + the core ontologies, then run the
  long prune-expand pipeline DETACHED so it survives a killed session, and
  monitor it to completion. Use when building the ontology from documents (the
  first heavy, paid, multi-hour step). Preserves existing table/summary caching.
---

# Merge + prune-expand (detached, monitored)

This is the first long, paid step (can run for **hours**). It runs **detached**
via the durable-run harness so a dead Claude session never kills it, and you
re-attach by reading its status/log — you do not hold the process.

## Preconditions (check first)
- `config/models.yaml` exists and required keys pass (run **choose-llm-mode** if not).
- The database step can run in parallel or after; prune-expand itself does NOT
  need the DB (it writes a version folder on disk). DB import happens later.

## Step 0 — Start with the MERGE: choose an ontology source (never auto-pick)
If the user asks "where do I start / what do I do first", tell them plainly: we
**start by building the ontology (a MERGE), then run prune-expand against a
document corpus.** First decide where the ontology comes from. Offer these four,
suggest based on their corpus, and CONFIRM — never choose silently, never scan
their folders and pick one:

**1. A supported domain** — pharma, manufacturing / supply chain, or finance. If
they accept, use the matching ontology and tell them how it's obtained:

| Domain | Ontology | How it's obtained (verified 2026-07-21) |
|---|---|---|
| Pharma / clinical | OCRe | Bundled in the repo; real download if `BIOPORTAL_API_KEY` is set |
| Finance | FIBO | Downloaded live from EDM Council (bundled fallback) |
| Manufacturing / supply chain | OntoCAPE | **NOT bundled** — `fetch_ontology.py` prints the RWTH form URL + where to drop the zip; **pause until they supply it**, don't substitute another ontology |

**2. Another domain** — no domain ontology; merge the 7 core ontologies only. Still
fully works, just without a domain-specific starting vocabulary.

**3. Their own ontology** — we accept `.owl` and `.rdf` (also `.ttl`, `.xml`, or a
`.zip` of these). Ask them to place the file somewhere and give you the path.
**WARN about size:** very large ontologies (hundreds of MB) are slow to import —
owlready2 walks imports — and can exceed memory; smoke-test a big one before
committing to a full run.

**4. Modify a previously generated ontology** — ask for the prior version folder
this tool produced (`output_ontologies/v*-.../`). Read "Step 1c — Re-ingesting an
edited ontology" first: whether their hand-edits are honored depends on HOW it's
fed back.

The 7 core ontologies are ALWAYS merged in, on every path.

## Step 1 — Merge (fast, foreground, no LLM)
Run ONE of the following, matching the Step-0 choice. Each merges the pinned 7 core
ontologies + your choice, records the merge, and **prints the version-folder path on
its last line** — read that path from the command output and use it as `MERGE_DIR`
in Step 3 (no shell-variable capture needed, so this works in any shell):
```
# Path 1 — a supported domain (fetches OCRe / FIBO / OntoCAPE):
uv run python scripts/merge_ontology.py --domain <pharma|finance|manufacturing>
# Path 2 — another domain (core ontologies only):
uv run python scripts/merge_ontology.py --core-only
# Path 3 — the user's own ontology (.owl/.rdf/.ttl/.xml/.zip; repeat --ontology for several):
uv run python scripts/merge_ontology.py --ontology "<path the user gave>"
# Path 4 — modify a previously generated ontology: see Step 1c
```
The 7 core ontologies are always included. Merge is deterministic; if a specific
input fails, the error names it. Large zips like FIBO are slow (~4 min) — expected
owlready2 import-walking, not a hang.

## Step 1b — Verify in Protégé before spending
`merge_ontology.py` already recorded the merge to the tracker and printed the merge
folder path (`MERGE_DIR`). Have the user eyeball the result before the paid step:
Say: *"Open `<MERGE_DIR>/merged.owl` in Protégé (https://protege.stanford.edu/) to
check the class hierarchy looks right before we run the paid prune-expand."* Pause
for their confirmation.

## Step 1c — Re-ingesting an edited ontology (IMPORTANT — silent-edit-loss gotcha)
A version folder holds TWO representations: **`merged.json`** (the fast machine
path downstream steps read BY DEFAULT) and **`merged.owl`** (what a human edits in
Protégé). They can diverge. If a user edits `merged.owl` and feeds the folder back:
- `prune-expand --input <folder>` alone reads `merged.json` → **the `.owl` edits are
  silently ignored.**
- `prune-expand --input <folder> --use-owl` re-parses the edited `merged.owl` → edits
  honored, and it writes a fresh, consistent folder.
- Universal + safest: re-merge the edited OWL —
  `uv run python scripts/merge_ontology.py --ontology <edited_merged.owl>` — which
  regenerates `merged.json` FROM the edited OWL, consistent for EVERY downstream
  step (prune-expand AND `db-init`).
So on Path 4, if they hand-edited the OWL, **re-merge it (or pass `--use-owl` to
prune-expand); never just resubmit the folder and assume the edits took.**

## Step 1d — Ask for the corpus (now that the ontology is ready)
Ask the user for the path to their documents folder — **REQUIRED, no default,
never scan their folders and pick one.** Confirm the folder exists and list its
contents back to them (use your own file-listing tools) so they confirm it's the
right corpus. Hold the confirmed path as `DOCS` for Step 3.

## Step 2 — Mention the `--tables` option and ask what they prefer
Proactively tell the user that prune-expand has an optional `--tables` flag, and
ask their preference — don't silently pick. Lay out the trade-off:
- **Without `--tables` (default):** faster and free; tables inside PDFs get read as
  run-on text and lose their row/column structure.
- **With `--tables`:** extracts structured tables from PDFs so their rows/columns
  become retrievable. Costs ~$0.02–0.04 per typical 10-K (vision). Worth it for
  data-heavy corpora — financial filings, spec sheets, anything numeric.
- **`--tables --no-table-vision`:** tables via pdfplumber only — free, but loses
  nested / merged-cell tables.

Only matters for PDF corpora (no effect on plain text). Set the flags in Step 3
to match their answer.

## Step 3 — Launch prune-expand DETACHED
Choose a fresh, unique `RUN_ID` (e.g. `prune_expand_<date+time>`). Launch it detached
via the harness as a **single-line command** (works in any shell). Append `--tables`
only if the user opted in, and `--no-table-vision` to skip vision:
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli prune-expand --input "<MERGE_DIR>" --documents "<DOCS>" --output-dir output_ontologies
```
The harness runs this command **unchanged**, so the two-tier table cache
(`output_ontologies/.../tables/` + `~/.cache/.../tables/`) and the evaluated-
summary cache (`~/.cache/.../eval_summaries/`) work exactly as before. A re-run
after a crash reuses everything already computed — it does not re-pay.

## Step 4 — Monitor (re-attach any time, even after a session death)
```bash
uv run python scripts/job_status.py <RUN_ID> 40      # state + last 40 log lines
```
Watch for:
- **`[preflight] WARNING: ... does not look like language`** — a garbled/
  unreadable PDF (broken font encoding). This fires **before** any paid table
  work. Surface it loudly and recommend the user re-source/OCR or remove that
  file; every downstream stage pays to process noise otherwise.
- **`[evaluated-summary] ... window X/Y`** — live summarization progress.
- **`HALT: cost cap ... reached`** — the `--max-cost-usd` cap tripped.
- **state = DIED** — killed/crashed before finishing; inspect the log tail, fix,
  and re-launch with a fresh RUN_ID (caches make the re-run cheap).

Do not block the session waiting. Check back periodically; the job runs
regardless of whether Claude is alive.

## Step 5 — Confirm completion + report
Run this — it checks the success sentinels (`stats/manifest/cost.json`, written only
on success), prints the cost, records `prune-expand`, and **prints the folder on its
last line** (read it from the output as `PE_DIR`):
```
uv run python scripts/pe_report.py
```
If it reports NOT complete, the run is still going or DIED — check Step 4's
`job_status.py` and the log tail before re-launching (caches make a re-run cheap).

Report: version-folder path, total cost, any preflight warnings, and that a re-run
would hit caches. Hand the prune-expand folder path to the orchestrator — the
ontology import into the DB is the first step of the **ingest-corpus** skill
(`db-init --input <prune-expand-folder>`).

## Notes
- prune-expand `--input` also accepts any prior version folder containing
  `merged.json`, so you can iterate without re-merging.
- Ontology sourcing is handled by `source_ontologies/fetch_ontology.py` (a small
  registry). FIBO downloads for real; OCRe (BioPortal-gated) and OntoCAPE
  (RWTH form-gated) fall back to the vendored zips — the fetcher prints why. To
  get a fresh OCRe, set `BIOPORTAL_API_KEY` in `.env`. Downloaded files land in
  `source_ontologies/downloaded/<domain>/` (gitignored); vendored zips are the
  guaranteed fallback so a clean clone never breaks.

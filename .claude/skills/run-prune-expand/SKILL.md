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
The 7 core ontologies, pinned by name (not globbed) so the set is explicit:
```bash
CORE=(
  source_ontologies/core_ontologies/viao_intelligence_artifact_ontology_v2.owl
  source_ontologies/core_ontologies/foaf.rdf
  source_ontologies/core_ontologies/org.ttl
  source_ontologies/core_ontologies/geography_ontology.owl
  source_ontologies/core_ontologies/time.ttl
  source_ontologies/core_ontologies/skos.rdf
  source_ontologies/core_ontologies/domain_concepts.owl
)
```
Set `DOMAIN_FILE` according to the Step-0 choice:
```bash
# Path 1 (supported domain) — fetch it; the ready-to-merge path is the LAST line:
DOMAIN_FILE=$(uv run python source_ontologies/fetch_ontology.py <pharma|finance|manufacturing> | tail -1)
# Path 2 (another domain):     DOMAIN_FILE=""                       # core only
# Path 3 (their own ontology): DOMAIN_FILE="<path the user gave>"   # .owl/.rdf/.ttl/.xml/.zip
# Path 4 (modify existing):    usually re-merge the edited merged.owl — see Step 1c
```
Build the args (core + domain if present) and run:
```bash
args=(); for f in "${CORE[@]}" ${DOMAIN_FILE:+"$DOMAIN_FILE"}; do args+=(--ontology "$f"); done
uv run python -m backend.app.cli merge "${args[@]}" --output-dir output_ontologies
MERGE_DIR=$(ls -dt output_ontologies/v*-merge/ 2>/dev/null | head -1); echo "$MERGE_DIR"
```
Merge is deterministic. If it errors on a specific ontology, report which
`--ontology` input failed. Large zips like FIBO are slow (~4 min for full FIBO) —
expected owlready2 import-walking, not a hang.

## Step 1b — Report the result + ask the user to verify in Protégé
Tell the user WHERE the merged ontology is, and have them eyeball it in a
third-party tool BEFORE any paid step:
```bash
echo "Merged ontology written to: $MERGE_DIR/merged.owl"
```
Say: *"Open `$MERGE_DIR/merged.owl` in Protégé (https://protege.stanford.edu/) to
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
- Universal + safest: re-run `merge --ontology <edited_merged.owl>` (plus the 7
  core) → a new folder with `merged.json` regenerated FROM the edited OWL,
  consistent for EVERY downstream step (prune-expand AND `db-init`).
So on Path 4, if they hand-edited the OWL, **re-merge it (or pass `--use-owl`);
never just resubmit the folder and assume the edits took.**

## Step 1d — Ask for the corpus (now that the ontology is ready)
```bash
DOCS="<path the user gives you>"   # REQUIRED — ask; never assume source_documents/, never scan+pick
[ -d "$DOCS" ] || echo "corpus folder not found: $DOCS — ask the user to confirm the path"
ls "$DOCS" | head                  # show them what you found so they confirm the corpus
```

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
```bash
RUN_ID="prune_expand_$(date -u +%Y%m%d_%H%M%S)"
scripts/run_detached.sh "$RUN_ID" \
  uv run python -m backend.app.cli prune-expand \
    --input "$MERGE_DIR" \
    --documents "$DOCS" \
    --output-dir output_ontologies \
    --tables            # include only if the user opted in; add --no-table-vision to skip vision
echo "$RUN_ID"
```
The harness runs this command **unchanged**, so the two-tier table cache
(`output_ontologies/.../tables/` + `~/.cache/.../tables/`) and the evaluated-
summary cache (`~/.cache/.../eval_summaries/`) work exactly as before. A re-run
after a crash reuses everything already computed — it does not re-pay.

## Step 4 — Monitor (re-attach any time, even after a session death)
```bash
scripts/job_status.sh "$RUN_ID" 40      # state + last 40 log lines
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
Completion sentinels are written only on success into the new version folder:
```bash
PE_DIR=$(ls -dt output_ontologies/v*-prune-expand/ 2>/dev/null | head -1)
ls "$PE_DIR"/{stats.json,manifest.json,cost.json} 2>/dev/null && echo "COMPLETE: $PE_DIR"
# report the actual spend
python3 -c "import json,sys; d=json.load(open('$PE_DIR/cost.json')); print('cost report:', json.dumps(d, indent=2))" 2>/dev/null
```
Also confirm caches populated (sanity):
```bash
ls ~/.cache/your-end-to-end-graphrag-implementation/eval_summaries/ 2>/dev/null | wc -l
```
Report: version-folder path, total cost, any preflight warnings, and that a
re-run would hit caches. Hand `PE_DIR` back to the orchestrator — the ontology
import into the DB is `db-init --input "$PE_DIR"` (run after setup-database).

## Notes
- prune-expand `--input` also accepts any prior version folder containing
  `merged.json`, so you can iterate without re-merging.
- Ontology sourcing is handled by `source_ontologies/fetch_ontology.py` (a small
  registry). FIBO downloads for real; OCRe (BioPortal-gated) and OntoCAPE
  (RWTH form-gated) fall back to the vendored zips — the fetcher prints why. To
  get a fresh OCRe, set `BIOPORTAL_API_KEY` in `.env`. Downloaded files land in
  `source_ontologies/downloaded/<domain>/` (gitignored); vendored zips are the
  guaranteed fallback so a clean clone never breaks.

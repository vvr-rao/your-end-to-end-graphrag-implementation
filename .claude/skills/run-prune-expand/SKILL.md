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

## Step 0 — Ask what to build (NEVER auto-discover)
You MUST get two things from the user **explicitly**. Do not guess, do not scan
their existing corpuses/folders and pick one, do not assume a default path.

**(a) Which domain — suggest, then confirm.** We support three domains, each with
a domain ontology, plus an "other" path that uses only the core ontologies. Ask
which fits their corpus and confirm:

| Domain | Ontology | Download status (verified 2026-07-21) |
|---|---|---|
| **Pharma / clinical** | OCRe | Bundled in repo (BioPortal-gated upstream); real download if `BIOPORTAL_API_KEY` is set |
| **Finance** | FIBO | **Real direct download** from EDM Council (also bundled as fallback) |
| **Manufacturing supply chain** | OntoCAPE | **NOT bundled** (GPL + RWTH form-gated) — the fetcher tells the user where to place the zip |
| **Other / none of these** | (none) | Core ontologies only — see Step 1 |

If the user picks manufacturing and OntoCAPE isn't present, `fetch_ontology.py`
exits with instructions (the RWTH form URL + the path to drop the zip). Relay
those and pause until they've supplied it — don't fall back to another ontology.

Suggest based on what the corpus looks like, e.g. *"These look like clinical
trial documents — I suggest the OCRe ontology. Good?"* Never pick silently.

**(b) Where the corpus is — ask for the path. There is NO default.**
```bash
DOCS="<path the user gives you>"      # REQUIRED — ask; never assume source_documents/
[ -d "$DOCS" ] || { echo "corpus folder not found: $DOCS — ask the user to confirm the path"; }
ls "$DOCS" | head    # show them what you found so they confirm it's the right corpus
```

## Step 1 — Fetch domain ontology + merge (fast, foreground, no LLM)
**The 7 core ontologies below are ALWAYS merged**, for every domain including
"other". They are pinned by name (not globbed) so the set is explicit and stable:
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
For a supported domain, fetch its ontology (downloads FIBO for real; OCRe/OntoCAPE
fall back to the vendored copy with an explanation). The fetcher prints the
ready-to-merge path as its LAST line:
```bash
# domain is one of: pharma | finance | manufacturing  (from Step 0)
DOMAIN_FILE=$(uv run python source_ontologies/fetch_ontology.py <domain> | tail -1)
```
For the **"other"** path, skip the fetch — `DOMAIN_FILE` stays empty and only the
core ontologies are merged.

Build the merge args (core + domain if present) and run:
```bash
args=(); for f in "${CORE[@]}" ${DOMAIN_FILE:+"$DOMAIN_FILE"}; do args+=(--ontology "$f"); done
uv run python -m backend.app.cli merge "${args[@]}" --output-dir output_ontologies
MERGE_DIR=$(ls -dt output_ontologies/v*-merge/ 2>/dev/null | head -1); echo "$MERGE_DIR"
```
Merge is deterministic. If it errors on a specific ontology, report which
`--ontology` input failed. Large zips like FIBO are slow (~4 min for full FIBO) —
that's expected owlready2 import-walking, not a hang.

## Step 2 — Decide on tables (cost note)
Ask the user whether to extract structured tables (`--tables`, opt-in, default
OFF). It adds ~$0.02–0.04 per typical 10-K in vision costs but makes financial
tables retrievable. On this corpus, pdfplumber alone is fabrication-free; the
vision route adds nested-table coverage at some cost. `--no-table-vision` keeps
tables via pdfplumber only (free). Match the flags to the user's answer.

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

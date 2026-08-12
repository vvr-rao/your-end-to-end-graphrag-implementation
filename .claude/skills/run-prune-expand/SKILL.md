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

## Step 2b — Compact the class descriptions first (cheap; cuts the big step)
**Always run this before prune-expand.** It is fast, costs cents, and lowers the
most expensive stage that follows:

```
uv run python -m backend.app.cli summarize-descriptions --input "<MERGE_DIR>" --max-cost-usd 2.0
```

It compresses each class's verbose `descriptions`/`comments` into a one-line
`compact_description`, written back into the merge folder in place. Stage 2's
`class_proposal` prompt then ships the compact text instead of the verbose
original (`pipeline_llm._slice_ontology`).

Measured on the 757-class pharma merge: **647 classes compacted for $0.025**,
a **78% reduction** in description tokens, and prune-expand cost fell
**$1.79 → $1.44** with the prompt-cache hit rate rising to 37%. Stage 2 is
63-70% of prune-expand's spend, so this is the highest-leverage cheap step in
the build.

Notes:
- **Idempotent.** Classes that already have a `compact_description` are skipped,
  so re-running is free. Safe to run even if you are unsure.
- It **modifies the merge folder in place** (merged.json + merged.owl). If the
  user wants the original preserved, copy the folder first.
- Skip only if the user explicitly wants the verbose descriptions in Stage 2.


### Step 2c — Size the run against THIS machine (always, before launching)
```
uv run python scripts/tpm_check.py "<DOCS>"
```
One command answers three questions: provider rate limits, how much RAM this
box actually has free, and how the corpus divides into batches. It prints a
concrete suggestion for each knob. **Report what it says and propose values --
do not silently apply them.**

**This is a REQUIRED step, not optional.** Run it, read the output back to the
user, and propose a full settings block before launching. The tool reads free
memory, swap, the provider's own rate-limit headers, and the corpus shape.

**What it reads from the provider:** the account's real limits, not a guess --
`x-ratelimit-limit-tokens` (**TPM**) and `x-ratelimit-limit-requests` (**RPM**),
plus remaining headroom, per model in `config/models.yaml`. Both bound the
suggestion: a stage is capped by whichever runs out first, which is why the
mini-model stages land at 32 (TPM-rich) while `class_proposal` stays at 4. Cost
is one 10-token probe per model.

**Platform:** the rate-limit half is a plain HTTPS call, so it behaves
identically on Linux, macOS and Windows. Only the *memory* half is
OS-specific -- `/proc/meminfo`, `vm_stat`+`sysctl`, and `GlobalMemoryStatusEx`
respectively, all built in. If a stage's advice is missing, it is memory
detection that failed, never the limits; use the fallback table below and keep
the rate-limit numbers as printed.

Knobs, grouped by what actually constrains them:

| Group | Knobs | Constraint | Shipped |
|---|---|---|---|
| **Memory** | `concurrency.table_extraction` | ~152 MB per PDF subprocess | 1 |
| **Memory (cheap) + caps concurrency** | `chunking.streaming_batch_size` | ~50 MB at 16 docs | 8 |
| **Mini-model rate limit (~10M TPM)** | `summarization`, `chunk_classification`, `entity_extraction`, `artifact_generation`, `table_mining` | measured ~3% of tier at 32 | 32 |
| **Big-model rate limit (~2M TPM, 32k requests)** | `class_proposal`, `dedup` | measured **6.2% of tier at 4** | 4 |

Per-run flags on prune-expand/build (each falls back to its config key):
```
--summarization-concurrency N   --classification-concurrency N   (Stage 1)
--proposal-concurrency N        (Stage 2)   --dedup-concurrency N
--table-mining-concurrency N    --expansion-concurrency N        (legacy: both stages)
```

**Stage 1 and Stage 2 are separate on purpose.** They shared one semaphore,
which pinned `chunk_classification` (gpt-4.1-mini, 10M tier) to
`class_proposal`'s safe value (gpt-4.1 @ 32k, 2M tier). Measured cost: 197
chunks took ~8 minutes where ~1 would do. Never propose raising them together.

**Cost concentration, from a measured 30-doc run ($25.71 total):**
`class_proposal` **65%**, `dedup` **22%** — so the big-model group is 87% of
prune-expand spend. Raising it saves the most TIME but changes no cost.

Three knobs, three different constraints:

| Knob | Bound by | Default | Notes |
|---|---|---|---|
| `concurrency.table_extraction` | **MEMORY** | 1 | Each PDF = one subprocess, ~152 MB measured. Serial extraction is often the LONG POLE: 18 PDFs at 1 is ~100 min. |
| `chunking.streaming_batch_size` | memory (cheap) | 8 | Holds N documents' text (~50 MB at 16). Also CAPS effective concurrency. |
| `concurrency.summarization` | **RATE LIMIT** | 32 | Measured ~0.5% of a 10M TPM tier; rarely the constraint on tier 3+. |

Say it plainly, e.g.
> *"You have 2.0 GB free and no swap. Table extraction is set to 1 PDF at a
> time — with 18 PDFs that's ~100 minutes, and it's the biggest single cost.
> Your memory supports 7 in parallel, cutting it to ~15 minutes. Batch size 8
> is also capping your concurrency at ~19 of 32, so I'd raise that to 16-32.
> **None of this reduces the bill** — same tokens, less waiting."*

**If `tpm_check.py` cannot read memory** (it prints "skipping memory-based
advice" on an unrecognised platform), fall back to the OS's own command --
and note that these are NOT interchangeable:

| OS | Command | Gotcha |
|---|---|---|
| Linux | `free -m` | "available", not "free" |
| macOS | `vm_stat` + `sysctl -n hw.memsize` | **Do NOT use "Pages free" alone** -- macOS parks most RAM in `inactive` (evictable), so free-only under-reports several-fold. Add free + inactive + speculative, times the page size from the header. |
| Windows | `powershell -c "Get-CimInstance Win32_OperatingSystem \| Select FreePhysicalMemory,TotalVisibleMemorySize"` | values are in KB |

`free -m` does not exist on macOS, so never issue it unconditionally.

**Memory rules:**
- If free RAM is under ~1.2 GB, say so and suggest closing other applications
  (an IDE easily holds 1.5+ GB) before a long paid run.
- On a **swapless** box (the tool reports this) an over-commit is a HARD KILL
  mid-run, not a slowdown. Prefer under-shooting.
- `concurrency.table_extraction` exists precisely because the in-process
  extractor OOMed partway through a corpus; 1 is a deliberate safe default, not
  an oversight. Raise it only against measured free memory.

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

### Concurrency — check it before a multi-hour run, and tell the user
`prune-expand` takes **three separate flags**, one per stage, because the stages
run on different models with different rate limits:

```
--summarization-concurrency N   --table-mining-concurrency N   --expansion-concurrency N
```

Each falls back to its config key when unset. Append them to the detached
command above when the user wants a per-run value; otherwise edit
`config/config.yaml`. The run prints what it resolved — quote it back:

```
[llm] concurrency: summarization=32 expansion(stage1/2)=4
[table-mining] concurrency = 32
```

| Knob | Drives | Model / limit | Sane value |
|---|---|---|---|
| `concurrency.summarization` | evaluated summarizer | mini-class, ~10M TPM | 32 (tier 3+), 8 (tier 1-2) |
| `concurrency.table_mining` | one call per table, `--tables` only | mini-class, ~10M TPM | 32 (tier 3+), 8 (tier 1-2) |
| `expansion.max_concurrent_llm_calls` | Stage 1/2 `class_proposal` | gpt-4.1 @ 32k max_tokens, ~2M TPM | **leave at 4-8** |

Summarization dominates wall time. It scales close to linearly: a 1.6M-token
corpus took **~4 hours at concurrency 4** and is **~25 minutes at 32**, at
identical cost.

**Before launching, run the planner and TALK THE USER THROUGH IT:**
```
uv run python scripts/tpm_check.py "<DOCS>"
```
Passing the corpus makes it size `chunking.streaming_batch_size` as well as
concurrency, because **the two interact and batch size usually wins.**

### Explain both knobs before proposing numbers
Users reasonably assume "concurrency" is the speed dial. It is only half of it:

- **`chunking.streaming_batch_size` (default 8) = how many DOCUMENTS are loaded
  and summarized at a time.** Batches are a hard barrier: `pipeline_llm` awaits
  each one fully before loading the next.
- **`concurrency.summarization` (default 32) = how many LLM calls may be in
  flight.** But only the CURRENT batch's windows exist, so:

      effective concurrency = min(concurrency, windows in this batch)
      windows per batch     ≈ streaming_batch_size × ~2.5 windows/doc

  At batch 8 that is ~19-20 windows, so **a concurrency above ~20 is idle** --
  raising it alone changes nothing. Measured on a 30-doc corpus: 13 of 32 slots
  sat unused.

Say this in plain terms, e.g.
> *"Two settings drive the time. `streaming_batch_size` is how many documents
> are worked on at once (currently 8); `concurrency` is how many LLM calls run
> in parallel (currently 32). Because only the current batch's work exists, the
> batch size is capping you at ~19 parallel calls — 13 of the 32 are idle.
> Raising batch size to 16 unlocks the concurrency you already have. It costs
> ~50 MB of memory and **does not change the bill** — same tokens, less waiting."*

**Keep 8 and 32 as the shipped defaults.** They are safe on a small-RAM box and
on low provider tiers. RECOMMEND changes; do not apply them unprompted. Both are
per-run overridable (`--summarization-concurrency`); `streaming_batch_size` is
config-only, so an edit to `config/config.yaml` is needed for that one.

Also flag the barrier effect when the corpus has outliers: a batch runs at the
speed of its slowest document, so one 60k-token doc holds up its whole batch.

It reads OpenAI's own rate-limit headers and suggests a concurrency per stage.
Then say what you found and propose raising it, e.g.
> *"Your gpt-4.1-mini limit is 10M TPM; summarization is set to 32, using well
> under 1%. I can take it to 64-125 and cut wall time roughly proportionally.
> **It won't cost less** — same tokens either way — it just finishes sooner."*

Always state that concurrency buys **SPEED, not savings**. Users assume a
tuning knob saves money; this one doesn't (it's ~2% *more*, from a slightly
lower prompt-cache hit rate — measured 69% → 49% at 4 → 32).

Measured at concurrency 32: **~0.5% sustained TPM** and **zero** burst pressure
on the token bucket. On tier 3+ the provider limit is not the constraint.

Do **not** raise `expansion.max_concurrent_llm_calls` to match — it drives
gpt-4.1 at 32k max_tokens on a 5×-tighter tier, where concurrent large calls
throttle and time out.

Also relevant to throughput: `chunking.streaming_batch_size` (default 8) is a
hard barrier — documents process in batches and batch N+1 cannot start until N
fully drains. Raising it trades memory for fewer stalls; keep it low on small-RAM
boxes.

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

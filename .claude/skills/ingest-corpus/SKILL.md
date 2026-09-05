---
name: ingest-corpus
description: >
  Load the prune-expand ontology into the database, then ingest the document
  corpus — chunk + embed, extract entities/relationships, enrich time, and
  generate intelligence artifacts. The paid, multi-hour steps run DETACHED so
  they survive a killed session, with --limit smoke-tests and a DB-size guardrail.
  Use after run-prune-expand, when building the knowledge graph from documents.
---

# Ingest the corpus into the knowledge graph

This is the corpus half of the build, symmetric with **run-prune-expand** (which
builds the ontology). It loads the ontology into Postgres, then runs the
document-ingestion pipeline. Several steps are **paid and multi-hour**, so they run
**detached** via the harness (`scripts/run_detached.py` + `scripts/job_status.py`)
and you monitor them — you never hold the process.

All commands use `uv run python …`, which works on Linux, macOS, and Windows.

## Preconditions
- `config/models.yaml` exists with keys present (**choose-llm-mode**).
- The database is set up and migrated (**setup-database**).
- A completed prune-expand version folder exists (**run-prune-expand**).

## Golden rules for the paid steps
- **Smoke-test first.** Run with `--limit 5` (or a small corpus), check the result
  and cost with the user, THEN scale to the full corpus. Never launch a full paid
  run without explicit go-ahead.
- **Watch DB size.** Before and after big steps: `uv run python -m backend.app.cli
  db-size`. Free-tier Postgres often caps around 500 MB — flag headroom as it grows.
- **Record each step** so the cross-session tracker can resume after a restart.
- **Full-text consistency (important).** `--full-text-chunks` at Step 2 stores BOTH
  summary and full-text chunks — but `extract-entities` and `generate-artifacts`
  default to the *summary* chunks. So **whenever full-text was ingested, you MUST
  pass `--from-fulltext` to Steps 3, 4, AND 5** (`extract-entities`, `enrich-time`,
  `generate-artifacts`), or the full-text chunks you paid to store are never mined.
  The choice is stored in the tracker as `fulltext=yes|no` on the
  `register-documents` step — check it (`build_state.py show`) and apply
  `--from-fulltext` consistently across all three downstream steps.
- **Check rate-limit headroom and RECOMMEND raising concurrency. Do this before
  every long paid step.** Run:
  ```
  uv run python scripts/tpm_check.py "<DOCS>"
  ```
  It reads the account's own rate-limit headers -- `x-ratelimit-limit-tokens`
  (**TPM**) and `x-ratelimit-limit-requests` (**RPM**), plus remaining headroom
  -- per model in `config/models.yaml`, and prints the current config next to a
  suggested value per stage. Both limits bound the suggestion; a stage is capped
  by whichever runs out first. Passing the corpus also sizes
  `chunking.streaming_batch_size`. Costs ~nothing (one 10-token probe per model).

  **Platform:** the rate-limit probe is a plain HTTPS call and behaves the same
  on Linux, macOS and Windows. Only the *memory* half is OS-specific
  (`/proc/meminfo`, `vm_stat`+`sysctl`, `GlobalMemoryStatusEx` -- all built in).
  If it prints "skipping memory-based advice", it is memory detection that
  failed, not the limits: keep the printed TPM/RPM numbers and get RAM from the
  OS's own command (see **run-prune-expand** Step 2c for the per-OS table and
  the macOS "Pages free" trap).

  **Explain BOTH knobs, because batch size usually wins.** `streaming_batch_size`
  (default 8) is how many DOCUMENTS are summarized at a time, and batches are a
  hard barrier -- only the current batch's work exists. So
  `effective concurrency = min(concurrency, windows in this batch)`, roughly
  `batch_size x 2.5`. At batch 8 that caps you near 20 parallel calls, and a
  concurrency above that is simply idle. Raising concurrency alone does nothing
  until the batch size moves.

  Keep 8 / 32 as the shipped defaults -- safe on small-RAM boxes and low tiers.
  RECOMMEND changes, do not apply them unprompted.

  **The tool also reads this machine's free memory** and suggests
  `concurrency.table_extraction` (default 1, MEMORY-bound: one subprocess per
  PDF, ~152 MB each). With `--tables` on a PDF-heavy corpus that serial
  extraction is often the single biggest time cost -- 18 PDFs at 1 is ~100
  minutes. Report free RAM, note if there is NO SWAP (an over-commit is then a
  hard kill), and suggest closing other apps if under ~1.2 GB free.

  Then **tell the user what you found and propose a number**, e.g.
  > *"Your gpt-4.1-mini limit is 10M TPM and we're set to 32 — that's using well
  > under 1% of it. I can raise it to 64-125 and cut the wall time roughly
  > proportionally. **This won't reduce cost at all** — the same tokens get sent
  > either way; it only makes the run finish sooner. Want me to?"*

  **Present it as a CHOICE, not a number.** Give conservative / recommended /
  aggressive options with what each buys AND costs, then ask which they want,
  and make clear they can name their own value or keep the defaults. See
  **run-prune-expand** Step 2c for the option table and the four effects to
  state (speed-not-savings, ~2% more spend from cache dilution, 429 risk on the
  gpt-4.1 stages, HARD KILL risk on `table_extraction`).

  **If their TIER is what caps the number** -- not memory, not corpus size --
  say so explicitly and point them at platform.openai.com → Settings → Limits
  to request an increase. A constraint nobody names is one the user cannot act
  on.

  Be explicit every time that **concurrency buys SPEED, not savings.** Users
  reasonably assume a tuning knob saves money; this one does not. If anything it
  costs ~2% more, because a higher value slightly lowers the prompt-cache hit
  rate (measured 69% → 49% going 4 → 32).

- **The numbers.** Steps 2, 3 and 5 read
  `concurrency.{summarization,entity_extraction,artifact_generation}` from
  `config/config.yaml`; each also takes `--concurrency N` for a single run. Wall
  time scales close to linearly: a 1.6M-token corpus took ~4 hours at 4 and is
  ~25 minutes at 32. Measured utilisation at 32 was **~0.5% sustained** on a 10M
  TPM tier, with **zero** burst pressure on the token bucket — so on tier 3+ the
  limit is not the constraint, our setting is.
  - `config.example.yaml` ships a conservative **8** for unknown tiers.
  - **Lower to 4-8 on tier 1-2**, where several large concurrent calls throttle.
  - Do NOT raise `expansion.max_concurrent_llm_calls` to match. It drives
    `class_proposal` on gpt-4.1 at `max_tokens: 32768` against a ~2M TPM tier
    (5× tighter); 4-8 is correct there.
  - Each command prints its resolved value on startup
    (`[extract-entities] concurrency = 32`) — quote it back when reporting.

## Step 1 — Load the ontology into the DB
First get the prune-expand folder path (the helper prints it):
```
uv run python scripts/latest_run_dir.py prune-expand
```
Then, substituting that printed path for `<PE_DIR>`, load it and record the step
(db-init is an idempotent upsert):
```
uv run python -m backend.app.cli db-init --input "<PE_DIR>"
uv run python -m backend.app.cli db-status
uv run python scripts/build_state.py record db-init path="<PE_DIR>"
```
(If the user hand-edited the ontology, re-merge first — see run-prune-expand
Step 1c — so `merged.json` reflects the edits before this import.)

## Step 2 — register-documents (chunk + embed; long, paid)
Confirm the corpus path (reuse the one from run-prune-expand if it's the same;
otherwise ask — never scan+pick). Discuss the options with the user:
- `--tables` — also extract structured tables from PDFs (retrievable rows/columns).
- `--full-text-chunks` — additionally store verbatim full-text chunks (better recall
  + exact citations) at the cost of DB size. **If you use it, Steps 3 and 5 must run
  with `--from-fulltext`** (see the full-text-consistency rule above) — decide this
  once, here.

**Smoke-test** with `--limit 5`, review, then scale. Add `--no-mine-aliases` to the
smoke test only (see the alias note below). Launch detached (single-line command;
choose a fresh `RUN_ID`, e.g. `register_docs_<date+time>`):
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli register-documents --input "<DOCS>" [--tables] [--full-text-chunks]
uv run python scripts/job_status.py <RUN_ID> 40
```
On completion, check `db-size`, then record — **including whether full-text was used**
(`fulltext=yes` if you passed `--full-text-chunks`, else `fulltext=no`) so downstream
steps and future sessions know to apply `--from-fulltext`:
```
uv run python scripts/build_state.py record register-documents docs=<n> chunks=<n> fulltext=<yes|no> tables=<n>
```

**Corpus synonyms refresh automatically here.** A successful (non-dry-run) ingest
ends by re-running `mine-aliases`, which extracts brand/generic pairs, acronym
expansions and alias phrases straight out of the text — "MOUNJARO (tirzepatide)
injection", "Securities and Exchange Commission (SEC)". Retrieval uses these so a
question asking "tirzepatide" still reaches chunks that only ever say "MOUNJARO".
It is free ($0, no LLM), takes ~1 min per 2,500 chunks, and always rescans the
whole corpus, so:
- **Pass `--no-mine-aliases` on `--limit` smoke tests** — a full rescan there is
  wasted work.
- **Never pass it on the real run.** Skipping it leaves the alias table stale for
  exactly the documents you just added, and nothing at query time says so.
- If the step reports a WARNING (e.g. migration `0006_term_aliases` not applied),
  the ingest still succeeded — apply migrations and run `mine-aliases` by hand.
- Look at the pair count it prints. Zero pairs on a real corpus means the guards
  are rejecting everything; run `mine-aliases --dry-run --show 40` to inspect.

## Step 3 — extract-entities (entities + relationships; long, paid)
Mints entities/relationships per chunk — **no new ontology classes**. **If Step 2
recorded `fulltext=yes` (check `uv run python scripts/build_state.py show`), you MUST
add `--from-fulltext`** so entities come from the full-text chunks, not the summary
ones. Single-line launch:
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli extract-entities [--limit 5] [--from-fulltext (REQUIRED if fulltext=yes)] [--max-cost-usd <cap>]
uv run python scripts/job_status.py <RUN_ID> 40
uv run python scripts/build_state.py record extract-entities entities=<n>
```

**Entity-to-entity relationships (TWO LLM calls per chunk).** This step now
also mints typed `entity -> entity` edges. Pass 1 extracts entities as before;
pass 2 asks for relationships between them, using a predicate menu narrowed to
those entities' actual classes. The run prints a line to report back:
```
[extract-entities] entity->entity relationships: N written, M dropped
    (unresolved=.., bad_predicate=.., domain_range=.., no_evidence=..)
```

**Before launching, confirm all THREE relationship tasks are in
`config/models.yaml`.** They are NEW tasks; a config predating them makes the
step print a one-line notice and quietly do less -- easy to miss in a long log,
because the run still looks successful:
```
uv run python -c "import yaml;t=yaml.safe_load(open('config/models.yaml'))['tasks'];\
print({k:(k in t) for k in ('relationship_extract','relationship_verify','relationship_repair')})"
```
Any False: run **choose-llm-mode** to refresh the preset, or copy the missing
blocks from `config/models.example.yaml`. What each one costs you if absent:
- `relationship_extract` missing -> **no entity->entity edges at all**.
- `relationship_verify` missing -> edges are written **without the
  support check**, so quotes that contradict their own claim get through.
- `relationship_repair` missing -> only matters with `--rescue-relationships`,
  which is off by default.

What to tell the user afterwards:
- **It is not more expensive.** Measured on 442 chunks: 305 edges at $0.31 as
  two passes, versus 84 edges at $0.32 as one call. The entity prompt shrank
  and the second call is skipped when a chunk has under two entities or no
  fitting predicate.
- **Every edge carries a verified evidence quote**, stored on the edge, so an
  edge can be audited without re-reading the chunk.
- **0 written is a real signal, not necessarily a bug** -- usually the ontology
  declares no object property whose domain AND range both match this corpus's
  classes. The run says so explicitly; surface it rather than calling it done.
- **Quality is good, not perfect.** Three gates now stand between a proposal
  and an edge: the quote must occur in the chunk, must NAME BOTH ends, and a
  third LLM pass judges whether it actually supports the claim in that
  direction. Hand-checked precision rose from ~50% to ~85% across three
  corpora. It is not 100% -- do not present the edges as clean.
- **Acceptance is deliberately low.** Roughly 15-22% of proposals become edges.
  The largest rejection bucket is `domain_range`: the ontology offers no
  predicate whose declared domain AND range fit that pair. That is usually a
  vocabulary gap, not a bad extraction.
- `--no-relationships` reproduces the older entity-only behaviour.
- `--no-verify-relationships` skips the third pass. Only for reproducing
  pre-verification behaviour; it lets contradicting quotes through.
- `--rescue-relationships` (**off by default, it measured WORSE**) re-asks for
  type-rejected claims using a wider either-end predicate menu. It rescued 74
  claims on one corpus and precision fell ~75% -> ~45%, because the wider menu
  admits predicates whose OTHER end is nonsense. Do not enable it casually.
- `--entity-identity name|name-class`. Default `name` gives ONE node per name,
  primary class by majority vote, other observed classes kept as extra
  `rdf:type` edges. `name-class` restores the old behaviour, which split one
  entity into a node per class -- measured 18 separate "Google" nodes on a
  30-doc corpus, which breaks multi-hop traversal. Use `name-class` only if
  your corpus has genuine homonyms that must stay distinct.
- **Watch the identity line in the log.** `name-level identity: N name(s) seen
  with more than one class collapsed to a single node` tells you how much
  fragmentation was repaired. Zero on a large corpus is suspicious.
- An EXISTING graph has none of these edges until extract-entities re-runs.

## Step 4 — enrich-time (short)
Temporal enrichment (Year/Quarter/Month/Day, parent creation + gap-fill). Fast and
cheap — run in the foreground. **Add `--from-fulltext` if Step 2 recorded
`fulltext=yes`** (scan the full-text chunks for dates, consistent with Steps 3 & 5):
```
uv run python -m backend.app.cli enrich-time [--from-fulltext (REQUIRED if fulltext=yes)]
uv run python scripts/build_state.py record enrich-time instances=<n>
```

## Step 5 — generate-artifacts (Claims/Findings/… ; long, paid)
Per-chunk `Claim`/`Finding`/`Observation`/`Event` + per-doc `Summary`. Opt-in
cross-cluster `Insight`/`Recommendation` via `--type` (gpt-4.1 — more expensive),
and `--rollup` for hierarchical consolidation. **Add `--from-fulltext` if Step 2
recorded `fulltext=yes`** (consistent with Steps 3 & 4). Single-line launch:
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli generate-artifacts [--limit 5] [--type Insight --type Recommendation] [--rollup] [--from-fulltext (REQUIRED if fulltext=yes)] [--max-cost-usd <cap>]
uv run python scripts/job_status.py <RUN_ID> 40
uv run python -m backend.app.cli db-size
uv run python scripts/build_state.py record generate-artifacts artifacts=<n>
```

## Step 6 — Report + hand back
Report what's now queryable (documents, entities, artifacts), the final `db-size`,
and total cost. The corpus is now a knowledge graph — the user can `query` /
`conversation`, or proceed to the **deploy** skill (which needs a **cloud** DB).

Before handing back, confirm synonyms are populated — Step 2 does this
automatically, but a skipped or failed refresh is invisible until a query
silently under-retrieves:
```
uv run python -m backend.app.cli mine-aliases --dry-run --show 15
```
If the printed pair count is 0, or far below what the corpus should yield, say so
rather than reporting a clean finish.

## Notes
- **Scale ceiling:** at very large corpora both `extract-entities` and
  `generate-artifacts` currently fire all chunk LLM calls before committing, so
  memory grows with corpus size and a kill loses in-flight work. For a first build
  this is fine; for ~10M-token corpora it needs a batched/resumable driver.
- All long steps are detached, so a closed session never kills them; re-attach with
  `job_status.py` and the tracker tells you where you left off.

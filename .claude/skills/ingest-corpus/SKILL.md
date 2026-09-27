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
- **Full-text handling — know the three different defaults.** Changed
  2026-09-27; older transcripts describe the previous behaviour.

  | stage | shipped default | knob |
  |---|---|---|
  | Step 2 `register-documents` | **full-text ON** — stores BOTH summary and verbatim chunks | `chunking.full_text_chunks: true`, or `--no-full-text-chunks` |
  | Step 3 `extract-entities` | **summary** (opt in per run) | `extraction.from_fulltext: false`, or `--from-fulltext` |
  | Step 4 `enrich-time` | **follows the corpus** | auto; force with `--from-fulltext` / `--no-from-fulltext` |
  | Step 5 `generate-artifacts` | **follows EXTRACTION** (Step 3) | auto; force with `--from-fulltext` / `--no-from-fulltext` |

  `enrich-time` **follows the corpus**: it checks whether `kind='fulltext'`
  chunks exist and uses them if so, falling back to summary. It only needs text,
  so that is the right rule for it.

  `generate-artifacts` **follows EXTRACTION** instead — whichever chunk kind
  carries `viao:assertsAbout` edges. Corrected 2026-09-27 after a measured
  failure: a corpus ingested with full-text chunks but extracted from summary
  chunks sent artifacts at the 583 full-text chunks, which reported
  `0/583 chunks have >=1 entity` and wrote 7,384 artifacts of which **0.9%**
  were entity-linked. Re-running over the 218 summary chunks took that to
  **25.3%**. Artifacts no entity can reach are invisible to the graph arm of
  retrieval, so following the corpus was wrong for this stage.

  Either way you no longer pass `--from-fulltext` by hand, and the old
  instruction that it was "REQUIRED if fulltext=yes" is obsolete.

  **`extract-entities` is deliberately NOT driven by ingestion, because it is
  the most expensive switch in the pipeline.** Measured 2026-09-26 on the
  42-doc `mixed-regression` corpus (40 news articles + 2 long PDFs):

  | | chunks | cost | wall |
  |---|---|---|---|
  | summary (default) | 282 | **$8.75** | 17 min @ concurrency 12 |
  | full-text | 1,222 | **~$38** (extrapolated at $0.031/chunk) | ~70 min |

  That is ~4.3x on a mixed corpus and ~18x on corpora of long reports, where
  summaries compress hardest. Two traps before you pass `--from-fulltext`:
  - **`--max-cost-usd` defaults to 5.0**, so a full-text run trips the cap
    almost immediately. Raise it explicitly and say the number to the user.
  - **Extraction is all-or-nothing** — it writes everything at the end, so a
    cap trip or a kill loses the whole run's spend.

  **What you actually give up by leaving it OFF is narrower than it sounds.**
  Retrieval reaches the full-text chunks regardless, through the
  document-mediated full-text bridge — citations still quote verbatim text.
  What you lose is *direct entity edges into* verbatim text, i.e. entity recall
  for facts that appear only deep inside a long document and never reach its
  summary. Turn it on for that recall, never for citation quality. Do NOT
  describe the default as leaving full-text chunks "never mined" — they are
  embedded and retrieved; only extraction skips them.

  Step 2 still records the choice in the tracker as `fulltext=yes|no`; it is
  now what Steps 4 and 5 detect automatically.
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

  **It now reports THREE limits, not two.** The third is the DATABASE connection
  pool, and it is the one that has actually broken a run. The pool cannot
  outgrow what the server accepts: Supabase free tier in SESSION mode caps the
  whole PROJECT at 15 client connections. Measured 2026-09-26:
  `extract-entities --concurrency 32` lost 62 of its first 71 chunks to
  `(EMAXCONNSESSION) max clients reached in session mode` -- because the POOL was
  grown to 32 connections. Keep `database.pool_size` + `max_overflow` at or
  under 15 on Supabase; that is what prevents it.

  | limit | bounds | set by |
  |---|---|---|
  | **TPM / RPM** | every LLM stage | the provider, read from response headers |
  | **Memory** | `table_extraction`, `streaming_batch_size` | this machine |
  | **DB pool** | `evaluation` (worker concurrency); every stage (pool size vs the server's client cap) | `database.pool_size` + `max_overflow` |

  So the rate-limit suggestion is an UPPER BOUND, not an instruction: a
  pool-capped stage runs at `min(suggestion, pool ceiling)` however high you set
  it, and the CLI prints the reduction when it applies. `summarization`,
  `entity_extraction` and `artifact_generation` are NOT capped -- their workers
  hold no DB session while waiting on the LLM (checked by a unit test that scans
  for awaits inside `session_scope()`), so 64 workers share a 12-connection pool
  in millisecond slices. Every `prune-expand` stage is exempt too, because it
  never touches the database. Raise `database.pool_size` only if the SERVER
  allows it.

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
- `--full-text-chunks` — **ON by default** (`chunking.full_text_chunks: true`); pass
  `--no-full-text-chunks` for summary-only. Additionally stores verbatim full-text chunks (better recall
  + exact citations) at the cost of DB size. **If you use it, Steps 3 and 5 must run
  with `--from-fulltext`** (see the full-text-consistency rule above) — decide this
  once, here.

**Smoke-test** with `--limit 5`, review, then scale. Add `--no-mine-aliases` to the
smoke test only (see the alias note below). Launch detached (single-line command;
choose a fresh `RUN_ID`, e.g. `register_docs_<date+time>`):
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli register-documents --input "<DOCS>" [--tables] [--no-full-text-chunks]
uv run python scripts/job_status.py <RUN_ID> 40
```
On completion, check `db-size`, then record — **including whether full-text was used**
(`fulltext=yes` unless you passed `--no-full-text-chunks`). Steps 4 and 5 now
detect this from the corpus themselves; the record is for humans and for
`extract-entities`, which stays opt-in:
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
**Use `--batch-size` on any run you would mind losing.** Without it the run is
all-or-nothing: every LLM call happens first and everything is written once at
the end, so a kill, a crash or a `--max-cost-usd` trip at 90% discards the whole
spend (measured: 282 chunks, $8.75, 17 minutes, nothing durable until the final
second). With `--batch-size N` each batch commits, and a re-run resumes by
itself — chunk selection already skips chunks that have entity edges, so the
graph is the progress marker and there is no checkpoint file. Prefer the
LARGEST batch you can afford to lose (100-200 on a multi-thousand-chunk run):
variant-spelling collapse and the class plurality vote see one batch rather than
the whole run, so a spelling split across batches may not merge. Default is 0
(single-shot) via `extraction.batch_size`.

Chunks that yield NO entities never get an edge and so can never be marked done;
the streaming loop attempts each once, then steps past it, and reports the count
at the end as an ontology-fit signal.

`--from-fulltext` is OPT-IN and costs ~4.3x (~$38 vs $8.75 on the 42-doc
reference corpus) — see the full-text table at the top. If you pass it, also raise
`--max-cost-usd` (default 5.0 trips almost immediately). Default takes entities from the summary
ones. Single-line launch:
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli extract-entities [--limit 5] [--batch-size 100] [--from-fulltext (OPT-IN: ~4.3x cost, raise --max-cost-usd)] [--max-cost-usd <cap>]
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

### ALWAYS report two coverage percentages, not just the raw counts
The run prints absolute numbers; a count alone does not tell the user whether
the ontology fits their corpus. Compute both and report **percentage AND
absolute**, because each answers a different question -- the percentage says how
well the ontology fits, the absolute says how much was actually lost.

**1. Entities dropped for want of a class.** From the `entity drops:` line and
the `DONE: entities (minted=N` line:
```
abstained % = abstained / (minted + abstained) * 100
```
Report as: `"X% of entity mentions were dropped (N of M) because no ontology
class fit them"`, and name 3-5 of the examples the run prints. Then say which
of the two causes it is -- they need different fixes and the log's own wording
picks the wrong one often enough to check:
- If the `recovered ... abstention(s) whose proposed_type named a class that
  EXISTS` count is **also high**, the menu is too narrow, NOT the ontology. A
  prune-expand run would not help; `--candidate-classes` or
  `extraction.pinned_class_labels` would.
- If the abstained examples are things the ontology plainly *should* cover
  (an organisation when `Organization` is pinned), it is neither -- flag it as
  unexplained rather than recommending a prune-expand run.
Anything **over ~15%** is worth calling out explicitly as a coverage problem.
Measured reference: a 66-doc corpus with a corpus-fitted 2,160-class ontology
still abstained on **12.5% (459 of 3,680)**, so double digits is normal and a
fitted ontology does not drive it to zero.

**2. Relationships that fell back to the generic predicate.** From the
`relatedTo (no ontology predicate fitted)` line:
```
relatedTo % = relatedTo / total edges written * 100
```
Report as: `"Y% of relationships (N of M) had no matching ontology predicate and
were recorded as graphrag:relatedTo, keeping the passage's own phrase"`. These
edges are NOT lost -- they are traversable and vector-searchable on that phrase
-- so present this as a vocabulary gap, not a failure. The run also histograms
the most common phrases; quote the top few, because they are a concrete
shopping list of predicates the ontology is missing. Measured reference: the
same 66-doc run routed **15.2% (178 of 1,173)** to relatedTo, with
`has capacity` x24 and `ranked above` x12 at the top.

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
- `--orphan-batch-size N` (default 8) -- orphans per `relationship_orphan_check`
  call. The check is what rescues entities both relationship passes missed, and
  its recall collapses on a long list: measured, a 31-orphan call proposed 2
  rescues where the same model asked about ONE found the edge at 0.95 confidence.
  Batching at 8 moved edges 192 -> 226 (+18%) and still-unlinked 1066 -> 926 on a
  30-document corpus, for +48% on this step's cost ($2.09 -> $3.10). Raise it to
  cut cost at the price of recall; lower it to spend more for more edges.
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
cheap — run in the foreground. Chunk kind is automatic (follows the corpus); the
old rule below is obsolete. **Formerly: add `--from-fulltext` if Step 2 recorded
`fulltext=yes`** (scan the full-text chunks for dates, consistent with Steps 3 & 5):
```
uv run python -m backend.app.cli enrich-time            # chunk kind follows the corpus automatically
uv run python scripts/build_state.py record enrich-time instances=<n>
```

## Step 4b — enrich-geo + embed-relationships (short, ~cents)
Two small steps that the relationship-aware retrieval depends on. Both are cheap
and fast; run them in the foreground.

**`enrich-geo`** mints geographic containment (`Bangalore -> Republic of India`)
between places the corpus ALREADY names. Prose never states these, so extraction
never finds them: a 40-document news corpus had exactly ONE `locatedin` edge, and
"which cities are in India" had no edge to walk. It adds EDGES, NEVER NODES -- a
container the corpus does not name is skipped -- and the edges are marked
`evidence_kind: world_knowledge`, so an answer may use them but they are labelled
rather than presented as something a document said.
```
uv run python -m backend.app.cli enrich-geo [--dry-run] [--limit N]
```
Report `created` and `rejected_by_level`. A high `unresolved` count means the
model named containers the corpus has no entity for -- expected, not a fault. If
`places=` is near zero on a corpus full of places, the ontology's place classes
are named something this pass does not recognise; set
`geo_enrichment.class_labels`.

**`embed-relationships`** vectorises each extracted edge as
`<source> <relation> <target>`, which is what the relationship-aware walk matches
a question against. `extract-entities` runs it automatically for the edges IT
writes, so you only need it explicitly:
- after `enrich-geo` (its edges are new and unembedded), and
- on any graph built before migration 0008 (idempotent -- embeds only NULLs).
```
uv run python -m backend.app.cli embed-relationships [--dry-run]
uv run python scripts/build_state.py record enrich-geo edges=<n>
```
Costs embeddings only: a few cents per 100k edges.

**Both are required for the graph to be walkable by relation.** Skip them and
retrieval still works, but it falls back to the broad neighbourhood walk and
loses the relation-matched hop -- silently, with no error.

## Step 5 — generate-artifacts (Claims/Findings/… ; long, paid)
Per-chunk `Claim`/`Finding`/`Observation`/`Event` + per-doc `Summary`. Opt-in
cross-cluster `Insight`/`Recommendation` via `--type` (gpt-4.1 — more expensive),
and `--rollup` for hierarchical consolidation. Chunk kind is automatic (follows the
corpus). **Formerly: add `--from-fulltext` if Step 2
recorded `fulltext=yes`** (consistent with Steps 3 & 4). Single-line launch:
```
uv run python scripts/run_detached.py <RUN_ID> uv run python -m backend.app.cli generate-artifacts [--limit 5] [--type Insight --type Recommendation] [--rollup] [--max-cost-usd <cap>]   # chunk kind follows the corpus
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

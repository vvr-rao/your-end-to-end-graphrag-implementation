---
name: app-help
description: >
  Answer general questions about this GraphRAG app — what it does, its pipeline
  stages, the CLI subcommands, the REST/MCP surface, costs, and constraints.
  Use when the user asks how something works or what's supported, rather than
  asking to build or run a step.
---

# App help — what this project is and how to use it

Answer from this overview and, when needed, by reading the code (the CLI parser
`backend/app/cli/main.py` and `README.md` are the best sources). Prefer showing
the real subcommand/flag over describing it.

## What it is
An end-to-end GraphRAG platform: ingest PDF/PPTX/DOCX/TXT, merge/curate an OWL
ontology, expand it via LLM analysis of the documents, persist everything in
Postgres + pgvector, and serve ontology-aware hybrid retrieval QA over REST, an
MCP server, and a React UI (deployable to Render). Postgres can be a hosted
instance (e.g. Supabase) or a local docker-compose database.

## Supported domains
Out of the box the app targets three domains, each with a domain ontology, plus a
generic path:
- **Pharma / clinical research** → OCRe (bundled; BioPortal-gated upstream)
- **Finance** → FIBO (real live download from EDM Council; also bundled)
- **Manufacturing supply chain** → OntoCAPE (NOT bundled — GPL + RWTH form-gated;
  user supplies the zip when prompted)
- **Anything else** → core ontologies only (no domain ontology; still works)
Domain ontologies are fetched by `source_ontologies/fetch_ontology.py`; a fixed
set of 7 **core** ontologies (VIAO, FOAF, ORG, geography, time, SKOS,
domain_concepts) is ALWAYS merged in as well.

## The pipeline (order)
1. **Choose LLM mode** — Groq+OpenAI (default) / OpenAI-only / Anthropic+OpenAI;
   embeddings always OpenAI.
2. **Database** — Supabase (prod) or local docker Postgres; `db-init` migrates.
3. **Merge** ontologies (VIAO + 6 more core + a domain ontology) — deterministic, no LLM.
3b. **summarize-descriptions** (optional but recommended, cents) — compacts each
   class's verbose descriptions into one line, which Stage 2 of prune-expand then
   ships instead. Measured: 78% fewer description tokens, prune-expand $1.79 →
   $1.44. Idempotent; edits the merge folder in place.
4. **prune-expand** — the paid, multi-hour ontology-building step (summarize →
   classify → propose → dedup → prune/extend). Caches summaries + tables on disk.
   Optionally (`--select-subset`) builds the ontology from a REPRESENTATIVE
   SUBSET rather than the whole corpus — see below.
5. **db-init --input** — load the resulting ontology into the DB.
6. **register-documents** — chunk + embed the corpus.
7. **extract-entities** — mint entities + relationships per chunk (no new classes).
8. **enrich-time** — temporal enrichment (Year/Quarter/Month/Day).
9. **generate-artifacts** — Claims / Findings / Observations / Insights /
   Recommendations / Summaries (backed by VIAO).
10. **query / conversation** — 2 retrieval modes: `simple_qa` (tight factoid) and
    `deep_research` (structured 7-section answer). Full traceability answer →
    artifact → chunk → document.
11. **Deploy** — `render-init` / `render-deploy --wait` / `render-status`.

## Handling a large corpus: representative-subset selection
`prune-expand --select-subset` builds the ontology from a chosen sample instead
of every document. It exists because Stage 2 (`class_proposal`) and Stage 3
(`match_dedup`) are **~95% of prune-expand's cost** and ~66% of its wall time,
and both scale with chunk count, hence with document count. Measured on a
30-doc run: $15.77 of $23.06 was Stage 2 alone.

How the subset is chosen:
1. Every document is summarized, compressed to ~6000 tokens, type-labelled by a
   cheap LLM call, and embedded.
2. k-means over the embeddings; **k** comes from the inertia elbow with a
   silhouette tiebreak. The document nearest each centroid is kept.
3. **Every document type is guaranteed at least one document** — US vs EU
   filings, drug labels by country, papers by area are each their own type.
4. **Every outlier is kept**, measured by distance to the nearest already-
   selected document (not to a centroid — a true outlier forms its own cluster,
   where it IS the centroid and would score zero).

Points users regularly get wrong:
- **Ingestion is unaffected.** `register-documents` still loads every document.
  Only the ontology comes from the subset.
- **Summarization is not wasted.** It writes the shared `eval_summaries/` cache
  that the later full ingest reads.
- **It previews by default.** Without `--yes` it writes `selection.json`, prints
  the subset, and stops before the $20+ ontology build.
- **The preview is not free on a cold corpus.** Clustering needs an embedding of
  every document, so the preview summarizes the WHOLE corpus first. With a warm
  `eval_summaries/` cache that is cents (measured $0.01 on 10 docs); cold, it is
  full summarization — roughly $0.03 per 12k-token window, so ~$14 for a 5M-token
  corpus. That spend carries over to `register-documents`, but it is real money
  up front and on a big cold corpus it dwarfs the selection itself.
- **What it can lose**: a concept appearing in exactly one ordinary (non-outlier)
  document. Tune with `--selection-k-max` (more clusters) and `--outlier-sigma`
  (lower keeps more outliers).
- Re-running selection is free — per-document vectors and labels are cached in
  `~/.cache/.../corpus_profiles/`, keyed on the document text.

One thing subsetting does NOT speed up: `stage4-apply` (~20% of wall time) is
pure CPU and scales with ONTOLOGY size, not document count.

## Retrieval precision: the `--rerank` flag

`query --rerank` re-sorts the fused chunk pool by distance to the **question**
embedding before truncating to `--top-k`.

Why it helps: step 9 vector-reranks per PROBE, then step 10 fuses those
rankings with RRF -- and RRF combines rank ORDER only, discarding the distances
(`_trim_by_distance` drops them explicitly). So a chunk that merely placed well
across several probes can outrank one that is genuinely closer to what was
asked. The flag restores that discarded signal as a final pass.

It reranks a pool WIDER than `top_k` (`qa.rerank_pool_multiplier`, default 3x)
and truncates afterwards, so it can pull a better chunk INTO the evidence set
rather than only reordering the ones RRF already chose.

- **Costs no extra LLM call** -- the question is already embedded for step 9.
  One additional SQL query.
- **Off by default** (`qa.rerank_after_fusion: false`). It changes which
  evidence a query returns, so it is opt-in until measured on your corpus.
- **Caveat worth knowing:** on a reranked run `retrieval_evidence.score` stays
  the RRF score, so it is NOT monotonic with rank -- sorting evidence rows by
  score does not reproduce the delivered order. The run records
  `retrieval_runs.retrieval_plan.rerank` to say so.

Reach for it when answers cite loosely-related passages, or when the corpus is
large and topically mixed so the candidate pool is broad.

## Graph relationships in the answer

Every query now carries a THIRD kind of evidence beside chunks and artifacts:
`entity -> entity` triples, rendered as a sentence plus the verbatim quote
extraction verified. They appear in citations as `[viao:Relationship_NNNN]`.

Why they are appended rather than fused: RRF ranks ONE flat id space (chunk
UUIDs), so a triple has no representation in it. Entities are projected down to
chunks at step 8 and the relationship was dropped there -- which is why a graph
holding 14 `defeated` edges once answered "which teams defeated which other
teams" with "no information". Triples now bypass fusion entirely.

- **Both modes get them.** `simple_qa` and `deep_research` alike; the count is
  `qa.max_relationship_evidence` (default 25) and does not vary by mode.
- `qa.relationship_scope` (default `either`) decides how strictly a triple must
  sit inside the BFS neighbourhood. `both` requires both endpoints to have been
  reached -- safe but lossy: on one question it admitted 44 of 132 edges and
  excluded every relevant one. `either` is safe only because triples are RANKED
  against the question before the cap applies.
- **They are ranked lexically**, predicate weighted 3x the endpoint names, so a
  question saying "signed with" promotes a `signedWith` edge. A question with
  no vocabulary overlap falls back to support-count order.
- **A triple's quote is trimmed** to the span that asserts the claim. Without
  that, a correct edge (`Windfield Holdings subsidiaryOf Albemarle`) whose quote
  also carried Windfield's Perth address made the answer report Perth as
  ALBEMARLE's headquarters. A prompt rule alone did NOT fix this.

If a relationship question returns "no information", check the verbose line
`graph relationships added: N of M edge(s) among the K entities BFS reached`.
`M` small means BFS never reached the right part of the graph (a seeding
problem); `N of M` with M large means ranking put the wrong triples first.

## Multi-hop questions need `--mode deep_research`

A question like "which clubs did X play against", where X's club must be
resolved first, needs the round-2 bridge: a planner runs a cheap round 1 to
discover entities, then rewrites round 2 with them substituted.

**That planner is `deep_research`-only.** `simple_qa` ignores `multi_round`
entirely -- passing it changes nothing. In `simple_qa` a 2-hop question only
works when a single chunk happens to resolve the second hop.

Cost measured on the same questions: **deep_research is 25-30x simple_qa**
($0.0012-0.0021 vs $0.035-0.061 per question) and 1.2-2x the wall time, because
it synthesises with gpt-4.1 at 4x the token ceiling over 50% more evidence, and
pays an extra retrieval round when the bridge fires. Both are cheap in absolute
terms; the ratio only matters at volume.

## Probe filtering

Query decomposition makes several probes; each votes into RRF. A probe that
matched nothing still votes -- and it votes for whatever is topically nearest,
manufacturing the consensus RRF trusts. `qa.probe_rel_band` (default 0.10)
drops a probe only if its best hit is worse than the BEST PROBE IN THIS QUERY
plus that band.

It replaced an absolute `probe_dist_floor`, which could not tell "matched
nothing" from "question phrased abstractly": five probes for one question
spanned 1.0222-1.0973 and the fixed 1.05 discarded four of them.
`qa.probe_dist_ceiling` (1.25) remains as a backstop for when every probe is
hopeless.

## Class matching

`qa.class_descend_depth` / `class_lexical_per_term` control how a question's
category words reach the classes entities actually carry. Matching is now
vector + lexical + **head-noun** (`player` -> `BaseballPlayer`, `NFLPlayer`) +
taxonomy descent.

The head-noun rule does most of the work, and deliberately ignores the
hierarchy -- generated ontologies get it locally wrong. Measured: one ontology
put `Player` under `Organization` while `BaseballPlayer` sat under `Person`, so
no downward walk could connect them.

## Entity-to-entity relationships

`extract-entities` mints typed `entity -> entity` edges. It runs as **TWO LLM
calls per chunk**: pass 1 extracts entities (unchanged), pass 2 asks for
relationships between the entities pass 1 just found.

Why two calls rather than one: the candidate-predicate menu can only be
narrowed to the entities' ACTUAL classes once those entities exist. Deriving it
from the chunk's top-50 candidate CLASSES instead offered a median of 12
predicates against 4-5 entities, most inapplicable -- measured 669 of 1,140
proposals rejected on domain/range alone.

**Measured on 442 chunks (2026-08-29), same corpus, same models:**

| | one call | **two passes** |
|---|---:|---:|
| edges written | 84 | **305** |
| invented predicate IRIs | 45 | **0** |
| acceptance rate | 7.4% | **42%** |
| cost | $0.32 | **$0.31** |

It is not more expensive: the entity prompt shrank (no predicate list) and the
second call is skipped whenever a chunk has fewer than two entities or no
predicate fits their classes.

**Every edge carries a verified evidence quote.** The model must quote the span
asserting the relationship, and the span is checked to really occur in the
chunk before the edge is written. This replaces protection lost by narrowing
the menu (offer and check now share a basis, so the type check is weaker).

**Quality is good, not perfect -- say so when asked.** Spot-checked sample:
- Sound: `COVID-19 lockdown measures -prevents-> COVID-19`
- Wrong: `MECP2 Duplication Syndrome -associatedWith-> S.pneumoniae`, whose
  quote is real and in the chunk but never mentions MECP2. The check verifies
  the quote EXISTS, not that it supports the claim.
- Over-broad: generic `foaf:`/`org:` predicates fit almost anything and get
  over-selected, e.g. `Ozempic -foaf:...-> acute pancreatitis`.

Other things worth knowing:
- `--no-relationships` reproduces the older entity-only behaviour.
- An existing graph gets none until `extract-entities` is re-run.
- Needs the `relationship_extract` task in `models.yaml`. On an older config
  the step says so once and continues with entities only.
- If a run reports 0 relationships, the usual cause is that the ontology
  declares no object property whose domain AND range both match this corpus's
  classes -- not that the documents assert nothing.

## CLI (single entrypoint)
`uv run python -m backend.app.cli <subcommand>`. Discover everything with
`uv run python -m backend.app.cli --help`. Groups: ontology (`merge`, `prune`,
`expand`, `prune-expand`, `build`), DB (`db-migrate`, `db-status`, `db-size`,
`db-init`, `clear-corpus`), corpus (`register-documents`, `list-documents`,
`update-document`, `delete-document`), enrichment (`extract-entities`,
`enrich-time`, `generate-artifacts`), retrieval (`query`, `evaluate-queries`,
`conversation start|turn|show`), deploy (`render-*`).

## Serving surfaces
- **REST** — FastAPI at `/`; `POST /qa`, `/conversations`, `/retrieval_runs/{id}`,
  browse routes. Bearer-token auth (`.env`).
- **MCP** — the same routes auto-exposed as MCP tools at `/mcp` (Streamable HTTP)
  via fastapi-mcp. Tool names come from each route's `operation_id`.
- **UI** — React + Vite, deployed alongside on Render.

## Costs & constraints (be upfront)
- The app uses a cheapest-viable-model-per-task configuration to keep costs down.
  prune-expand and the corpus steps on a real corpus can run **hours** and cost
  tens of dollars — smoke-test with `--limit` first, check the result, then scale.
- Keep an eye on database size with `db-size` / `db-status`; free-tier Postgres
  plans often cap around 500 MB, so watch headroom as the corpus grows.
- Long steps run **detached** (`scripts/run_detached.py`) so they survive a
  killed session; monitor with `scripts/job_status.py`. The harness is pure
  Python, so it works on **Linux and macOS**; on **Windows use WSL2**.
- Some steps have known caveats worth mentioning: garbled/unreadable PDFs are
  flagged by a preflight check before paid work; structured-table extraction is
  opt-in (`--tables`).
- **If the user asks why a run is slow:** run `uv run python scripts/tpm_check.py`
  first, then recommend raising concurrency. It reads OpenAI's rate-limit headers
  and suggests a value per stage. Measured on a tier-4 account at concurrency 32:
  ~0.5% sustained TPM and ZERO burst pressure, so on tier 3+ the provider limit
  is almost never the constraint -- the config is.
  - **Always say that concurrency buys SPEED, not savings.** Users assume a
    tuning knob lowers the bill; this one does not (~2% more, from a lower
    prompt-cache hit rate). Wall time scales close to linearly: a 1.6M-token
    corpus took ~4h to summarize at 4 and ~25 min at 32.
  - Settings live under `concurrency:` in `config/config.yaml`
    (`summarization`, `entity_extraction`, `artifact_generation`,
    `table_mining`, `evaluation`). Suggest 32-64 on tier 3+, 4-8 on tier 1-2.
  - `expansion.max_concurrent_llm_calls` is deliberately SEPARATE and stays at
    4-8 -- it drives gpt-4.1 at 32k max_tokens against a ~2M TPM tier.
  - Flags: most commands take `--concurrency N`; `prune-expand` and `build` take
    three (`--summarization-concurrency`, `--table-mining-concurrency`,
    `--expansion-concurrency`) since they drive three stages on different models.
  - **Batch size caps concurrency.** `chunking.streaming_batch_size` (default 8)
    is how many DOCUMENTS are summarized at once, and batches are a barrier, so
    `effective concurrency = min(concurrency, ~2.5 x batch_size)`. At batch 8
    that is ~20 -- a concurrency above that is idle. Pass the corpus to
    `tpm_check.py <docs-dir>` and it will size both.
  - See the README's "Tuning throughput" section.

## How to help
- "What does X do / cost / support?" → answer here or from `--help`/README.
- "Build it / run step X" → hand off to **build-app** (or the specific skill).
- "Add / update / delete a document" (after the build) → hand off to **manage-documents**.
- Keep credentials out of chat; never read `.env`.

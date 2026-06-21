# Your End-to-End GraphRAG Implementation

GraphRAG-based ontology and document management system. Ingests domain documents,
imports/merges OWL ontologies, expands them with LLMs, stores everything in
Postgres + pgvector, and answers questions with ontology-aware retrieval.

The same operations are exposed over **two parallel transports** so they can be
called by either an HTTP client or an LLM agent:

- **REST / OpenAPI** — `http://localhost:8000/docs`
- **MCP server** — `http://localhost:8000/mcp` (auto-generated from FastAPI routes via `fastapi-mcp`; both transports are served by a single uvicorn process)

A minimal React + Vite UI lives under `frontend/` and is deployed alongside the
backend on Render (see `render.yaml`).

## Status

- **Phase 1 — Ontology CLI** ✅ shipped. `merge` / `prune` / `expand` / `prune-expand` / `build` produce versioned ontology folders.
- **Phase 2 — Persistent GraphRAG platform** ✅ shipped. Postgres + pgvector knowledge graph; ingestion → chunking → entity extraction → temporal enrichment → intelligence artifacts (Claim / Finding / Observation / Summary / Insight / Recommendation) → retrieval. FastAPI + MCP routes wrap every CLI op.
- **Phase 3 — UI + versioning + Render deploy** — pending.

See `CLAUDE.local.md` for the full project plan.

## Phase 2 quick start (retrieval)

The platform supports **two retrieval modes**:

| Mode | What it produces | When to use it |
|---|---|---|
| `simple_qa` | Tight 1-3 sentence direct answer with citations. | Direct factoid lookup. |
| `deep_research` (**default**) | Structured 7-section answer: **SPECIFICS → ANALYSIS → ANSWER → CONTRADICTIONS → KEY CLAIMS (with evidence status) → COVERAGE IMBALANCE → KEY INSIGHTS**. | Comparisons, listings, synthesis, deep-dives. |

```bash
# One-shot question (default mode = deep_research)
uv run python -m backend.app.cli query \
  "What are the regulations around EV production in Asia?"

# Tight factoid answer
uv run python -m backend.app.cli query \
  --mode simple_qa \
  "What is OCI N.V.'s annual nitrogen fertilizer production capacity?"

# Multi-turn conversation with automatic follow-up resolution
CONV=$(uv run python -m backend.app.cli conversation start | grep '^iri:' | awk '{print $2}')
uv run python -m backend.app.cli conversation turn --conv "$CONV" \
  "What does the corpus say about EV production?"
uv run python -m backend.app.cli conversation turn --conv "$CONV" \
  "And how does Vietnam compare specifically?"     # 'and how does ... compare' is resolved against the prior turn's answer
uv run python -m backend.app.cli conversation show --conv "$CONV"

# Build / refresh the corpus (idempotent; safe to re-run)
uv run python -m backend.app.cli register-documents \
  --input source_documents/<your-corpus>
uv run python -m backend.app.cli enrich-time
uv run python -m backend.app.cli extract-entities
uv run python -m backend.app.cli generate-artifacts                # Claim / Finding / Observation / Event + per-doc Summary
uv run python -m backend.app.cli generate-artifacts --type Insight  # cross-document synthesis (gpt-4.1)
uv run python -m backend.app.cli generate-artifacts --type Recommendation  # cross-Insight (gpt-4.1)

# Evaluate the retrieval quality (5 metrics: comprehensiveness,
# no_hallucination, consistency, gap_detection, time)
uv run python -m backend.app.cli evaluate-queries \
  --questions eval_questions/v1_smoke.txt \
  --runs-per-question 3 \
  --output /tmp/eval.json --output-md /tmp/eval.md
```

See [eval_questions/README.md](eval_questions/README.md) for the
methodology, retrieval pipeline diagram, per-question cost estimates,
and what each metric measures.

## Document summarization (pre-ingestion)

Documents above `chunking.summarization_threshold_tokens` (default 2,000 tokens
in [config/config.yaml](config/config.yaml)) are compressed via gpt-4o-mini
before chunking + embedding. The summary — not the raw text — is what gets
embedded and stored on `chunks.text`; `documents.file_path` still points at
the original so citations can resolve back to source.

**Designed to keep** (per the prompt):

- Named entities (countries, regions, companies, products, people, regulations, dates, monetary amounts, measurements)
- Conceptual categories (industries, sectors, technologies, materials, processes, frameworks)
- Relationships between entities (X causes Y, X is part of Y, X exports Y, etc.)
- Numerical specifics tied to a named thing
- Intelligence-bearing fragments rendered as standalone sentences: **Events** (with dates), **Claims** (with source attribution), **Findings**, **Risks**, **Insights**

The prompt lives in [`document_summarize`](backend/app/services/prompts.py#L711)
at [prompts.py:711-792](backend/app/services/prompts.py#L711-L792). Users may
want to look at it and edit it to retain specific information their corpus
needs. When you change the prompt, bump `_DOC_SUMMARY_PROMPT_VERSION` at
[pipeline_llm.py:1518](backend/app/services/pipeline_llm.py#L1518) to
invalidate the on-disk summary cache.

## Intelligence artifacts

The retrieval layer reads from a typed library of **Intelligence Artifacts** — rows in `graphrag.intelligence_artifacts`, each a typed instance of a VIAO class. Eight types are extracted today (`Fact`, `Conclusion`, `Risk` are also in VIAO but reserved for future passes).

### The 8 supported types

| Type | VIAO class | Definition | Generator | Model |
|---|---|---|---|---|
| **Claim** | `viao:Claim` | A factual assertion the source MAKES (e.g. *"BYD owns 30% of the Vietnamese EV market"*). Specific + verifiable. | `artifact_chunk_extract_with_entities` (per chunk) | gpt-4o-mini |
| **Finding** | `viao:Finding` | An analytical conclusion or insight (e.g. *"the trend suggests EV demand is accelerating in ASEAN"*). Goes beyond raw facts. | same prompt, single LLM call | gpt-4o-mini |
| **Observation** | `viao:Observation` | A raw factual statement directly visible in the text (e.g. *"price rose 5% in March"*). The most concrete of the assertion types. | same prompt | gpt-4o-mini |
| **Event** | `viao:Event` | A happening anchored to a date or date range (study publication, election, founding, regulation effective date, crisis incident). Carries `event_date` / `event_start_date` / `event_end_date` / `event_category` metadata. | same prompt | gpt-4o-mini |
| **Summary** | `viao:Summary` | Condensed 200-word representation of one Document. | `artifact_document_summary` (per doc, once) | gpt-4o-mini |
| **StructuredTable** | `viao:StructuredTable` | JSON-LD representation of a table extracted from a PDF (caption + columns + rows + cells). Full payload lives in `extra_metadata` JSONB. | `register-documents --tables` (PDFs only; reads pre-extracted JSON-LD from `~/.cache/.../tables/`) | pdfplumber + gpt-4o-mini vision (during prune-expand) |
| **Insight** | `viao:Insight` | Cross-class synthesis. Non-obvious pattern across many Claims/Findings tied to the same ontology class. Opt-in via `--type Insight`. | `insight_gen` clusters by `class_id` of attached entities | **gpt-4.1** |
| **Recommendation** | `viao:Recommendation` | Actionable judgment derived from clustered Insights. Opt-in via `--type Recommendation`. | `recommendation_gen` clusters Insights by embedding k-means | **gpt-4.1** |

### Which nodes get artifacts derived from them

Artifacts are derived from **3 corpus-side node types** + the artifact graph itself. Ontology-side nodes (classes, properties, instances) are *referenced* by artifacts but don't produce them.

| Node type | Artifacts derived FROM each node |
|---|---|
| **Document** (1 row per ingested file) | 1 `Summary` (always) · N `StructuredTable`s (PDFs only — typically 50–300 per annual report) |
| **Chunk** (1+ per document after summarization + chunking) | ~0–8 of each of: `Claim`, `Finding`, `Observation`, `Event` per chunk (single LLM call returns all 4 types as JSON) |
| **Cluster of Claim+Finding** (grouped by ontology class via the entity's `class_id`) | 1–3 `Insight`s per qualifying cluster (≥10 attached artifacts by default) |
| **Cluster of Insights** (grouped by embedding k-means) | 1–3 `Recommendation`s per theme |
| Ontology classes / properties / instances | ❌ no artifacts derived |
| Entities | ❌ no artifacts derived. Entities are *subjects* of artifacts via `viao:assertsAbout` edges |
| Time instances | ❌ no artifacts derived. Chunks link to them via `time:inPeriod` |

### The derivation hierarchy (mirrors VIAO predicates)

```
            Document
              │ ─────────────────────────┐
              │ has chunks               │ derivedFromDocument / summarizes
              ▼                          ▼
            Chunk                  Summary  +  StructuredTable (PDFs only)
              │
              │ derivedFromChunk
              ▼
       ┌──────┼──────┬──────────┐
       ▼      ▼      ▼          ▼
     Claim  Finding  Observation  Event
       │      │
       │   insightBasedOn
       └──────┘
           │
           ▼
        Insight                   ── cross-class synthesis (opt-in, gpt-4.1)
           │
           │ recommendationBasedOn
           ▼
     Recommendation               ── cross-Insight (opt-in, gpt-4.1)
```

### Per-artifact metadata captured in `extra_metadata` JSONB

| Type | Fields |
|---|---|
| `Claim` / `Finding` / `Observation` | `evidence_status` (`"backed"`/`"partial"`/`"unbacked"` — whether the chunk supplies reasoning) · `claim_source` (who made the claim, e.g. *"BHP's CEO"*) · `time_scope` (what period the claim applies to) |
| `Event` | `event_date` / `event_start_date` / `event_end_date` (YYYY-MM-DD strings) · `event_category` (free-text label like *"study"*, *"election"*, *"founding"*) |
| `StructuredTable` | Full JSON-LD bundle: `caption`, `pageNumber`, `extractionMethod`, `columns[]`, `rows[]`, `cells[]` |
| `Insight` / `Recommendation` | `cluster_class` (the ontology-class the cluster came from) · base-Claim/Insight references via `artifact_sources` M2M |

### Entity grounding

When a chunk has entities attached (from `extract-entities`), the per-chunk prompt is `artifact_chunk_extract_with_entities` — it lists the chunk's entities in the prompt and **requires** the LLM to use exact canonical names (no *"the company"* / *"the manufacturer"* substitution). For each entity whose canonical name then appears in the artifact text, an `Artifact → viao:assertsAbout → Entity` edge is written. That gives you the citation chain *artifact → asserted entity → typed ontology class* used by `deep_research`.

### Cardinality expectations on a ~80-doc / ~90-chunk financial-reports corpus

| Type | Approximate count |
|---|---:|
| `Summary` | ~80 (1 per doc) |
| `StructuredTable` | 1,500–3,000 (depends heavily on PDF density) |
| `Claim` | ~250–400 |
| `Finding` | ~150–300 |
| `Observation` | ~200–400 |
| `Event` | ~50–150 |
| `Insight` (opt-in) | ~20–40 |
| `Recommendation` (opt-in) | ~10–20 |

### Storage shape

Every artifact lives in `graphrag.intelligence_artifacts` with `artifact_identifier` matching `viao:<Type>_<uuid16>`, a `vector(1024)` embedding over `(title + text)`, a `graph_version` stamp, and `status` in `{ACTIVE, STALE, RETIRED, DELETED}`. Traceability flows through:

```
Answer → retrieval_evidence → intelligence_artifacts → artifact_sources → chunks → documents → file_path
```

## Phase 1 — ontology CLI

Five subcommands. Each writes a fresh versioned folder under `output_ontologies/v<UTC-timestamp>-<subcommand>/` containing `merged.owl` (Protégé-readable), `merged.json` (canonical re-loadable form), `manifest.json` (provenance), `stats.json`, and `llm_audit.jsonl`.

```bash
# 1) merge — deterministic, ZERO LLM calls. Consolidate .owl/.rdf/.ttl/.zip inputs.
uv run python -m backend.app.cli merge \
  --ontology source_ontologies/pharma_ontologies/OCRe.zip \
  --ontology source_ontologies/general_ontologies/skos.rdf

# 2) prune — drop classes unsupported by your documents (LLM-driven).
uv run python -m backend.app.cli prune \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 3) expand — propose new classes/relationships from documents (LLM-driven).
uv run python -m backend.app.cli expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 4) prune-expand — both at once, more efficient (one LLM pass).
uv run python -m backend.app.cli prune-expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/

# 5) build — merge + prune-expand end-to-end.
uv run python -m backend.app.cli build \
  --ontology source_ontologies/pharma_ontologies/OCRe.zip \
  --documents source_documents/
```

Optional flags: `--max-hops N`, `--max-cost-usd N`, `--dry-run`, `--output-dir DIR` (default `output_ontologies/`).

### Hand-suggesting additional classes (`--suggested-new-classes`)

The `expand`, `prune-expand`, and `build` subcommands accept an optional JSON file
of classes you want added in **addition** to whatever the LLM proposes from your
documents. The LLM is also told about them so it can avoid proposing near-duplicates;
any suggestions that don't already appear in `MATCH NOT FOUND` after dedup are
appended before the deterministic expansion step writes them out.

Copy the example template and edit:

```bash
cp suggested_new_classes.example.json suggested_new_classes.json
# Edit suggested_new_classes.json to list the classes you want added.
```

File format:

```json
[
  {
    "CLASS_TYPE": "Adverse Events",
    "CLASS_DESCRIPTION": "Adverse Events listed in a Drug or in a Study",
    "PARENT_CLASS_TYPE": "NONE"
  }
]
```

`PARENT_CLASS_TYPE` can be `"NONE"` (the class roots at the configured
`default_parent_iri`, typically `owl:Thing`) or the LABEL of another class in
the same file or the existing ontology.

Use it:

```bash
uv run python -m backend.app.cli expand \
  --input output_ontologies/v<...>-merge/ \
  --documents source_documents/ \
  --suggested-new-classes suggested_new_classes.json
```

`suggested_new_classes.json` is gitignored; only the `*.example.json` template is tracked.

### How `merge` handles multi-file inputs

- A single `.owl`/`.rdf`/`.ttl`: parsed directly via owlready2 in its own isolated `World()`.
- A `.zip` of many files: extracted to a temp directory; every `.owl`/`.rdf`/`.ttl` inside is enumerated. Each file is loaded into its **own per-file owlready2.World()** so triples don't accumulate across files (this is what kept FIBO and OntoCAPE merges from hanging — previously the shared `default_world` plus per-call `rdf_graph` snapshot made total extraction O(N²)).
- Cross-file `owl:imports` (including OntoCAPE's `file:/C:/...` Windows-style imports baked via XML `<!ENTITY>` references, and FIBO's OASIS `catalog-v001.xml` files) are resolved to the local extracted copies via an IRI map.
- HTTP(S) imports that owlready2 doesn't already know about (FIBO's `https://www.omg.org/...`, `https://spec.edmcouncil.org/...`) are stripped from the extracted copies so owlready2 doesn't hang on a TCP SYN trying to fetch them.
- `file:` imports that point to siblings the zip doesn't ship (OntoCAPE's reference to a separate `meta_model.owl` package) are also stripped.
- Per-file failures (a defective XML file mid-zip; an owlready2 incompatibility) are logged and skipped so one bad file doesn't kill the whole merge.
- Verified merges on the dev machine:
  - OCRe.zip: 389 classes, ~6s.
  - HP.owl: 32,085 classes, ~75s.
  - OntoCAPE zip: 790 classes, 63/64 files (1 skipped due to an orphan `-->` in the source archive), ~50s.
  - FIBO `prod.rdf.zip`: 2,237 classes / 222 files, ~254s.
  - DRON.owl (670MB): excluded from automated tests — owlready2 needs ~3GB RAM to parse, which exceeds the 2.7GB ceiling on a typical dev laptop. Verifiable manually on bigger hardware via the CLI.

### How the LLM pipeline works (`prune`, `expand`, `prune-expand`, `build`)

The LLM-driven subcommands all run the same 4-stage pipeline; they differ only in which deterministic transformation Stage 4 applies at the end. Stages 1–3 produce a single `{MATCHES FOUND, MATCH NOT FOUND, MATCH NOT FOUND RELATIONS}` dict; Stage 4 turns that into actual changes on the canonical dict-of-dicts.

Lives in [backend/app/services/pipeline_llm.py](backend/app/services/pipeline_llm.py). The four stages:

#### Stage 1 — `chunk_classification` (Groq · llama-3.3-70b-versatile)

Per doc chunk, asks the model which **top-level ontology branches** (typically 100–250 root classes — one per `subClassOf` tree) the chunk is plausibly relevant to. Returns a short IRI list.

This is the narrowing step. The full `classes_dict` for a mid-size ontology is well over 1M tokens, so Stage 2 can't see it whole on every chunk; Stage 1 picks the slices Stage 2 should actually look at.

The "top-level branch" detection treats `owl:Thing` as outside-the-ontology when checking superclass containment — otherwise domain roots (VIAO `InformationSource`, geography `GeographicEntity`, W3C-time `DayOfWeek`, etc.) that all declare `owl:Thing` as super get filtered out and Stage 1 never sees them. `_top_level_branches` in `pipeline_llm.py` skips a configurable `_GENERIC_TOP_TYPES` set during root detection. Includes a retry-on-429 path that respects Groq's `Please try again in Xs` hint for transient TPM bursts on the Dev tier.

Why Groq + a 70B model: classification is cheap, and the 70B beats the 8B at disambiguating similar branch labels.

#### Stage 2 — `class_proposal` (OpenAI · gpt-4.1)

Per chunk, with the chunk text plus a **sliced sub-ontology** (every class within `max_hops` of any IRI Stage 1 returned, stripped to `name / iri / labels / comments / descriptions / superclasses`), asks the model:

1. Which IRIs from the slice does this chunk talk about? (`MATCHES FOUND`)
2. What new classes does the chunk need that aren't in the slice? (`MATCH NOT FOUND`, each entry has `LABEL` + `DESCRIPTION`)
3. What relationships does the chunk assert between classes? (`MATCH NOT FOUND RELATIONS`, each entry has `LABEL` + `DOMAIN` + `RANGE`)

Each `MATCHES FOUND` IRI must be an exact key of the sliced ontology. New-class proposals and relation endpoints may reference labels of other proposals in the same response — Stage 4 resolves them after Stage 3 dedup. The Stage 2 prompt is recall-biased for class matching (geographic + temporal mentions in particular MUST match existing classes when available) and precision-biased for relations (because hallucinated endpoints like `DOMAIN: "Chinese government"` would clutter the ontology and clutter the skip list).

#### Stage 3 — `match_dedup` (OpenAI · gpt-4.1, one call total)

After all Stage 2 outputs are merged, one consolidation pass collapses:
- `MATCH NOT FOUND` entries that propose the same concept under different labels.
- `MATCH NOT FOUND` entries that duplicate something already in `MATCHES FOUND`.
- `MATCH NOT FOUND RELATIONS` entries with the same `LABEL` + `DOMAIN` + `RANGE` or trivially paraphrased verb labels.

`MATCHES FOUND` entries are never modified.

#### Stage 4 — deterministic prune / expand (pure Python, no LLM)

For `prune`, `prune-expand`, and `build`:

Build the **keep-set** as:

1. Detected IRIs from `MATCHES FOUND` (the seed set).
2. The **full ancestor + descendant transitive closure** of every seed via `subClassOf` (`collect_full_class_hierarchy` in `backend/app/helpers/ontology_pruning.py`). This guarantees every kept class's place in the taxonomy is unambiguous — siblings of seeds are NOT pulled in just because they share a parent.
3. **Relationship partners**: for every object/data property whose domain or range touches the keep-set, the OTHER endpoint joins the keep-set (no orphan `range=[]`).
4. **Protected IRI prefixes**: every class whose IRI starts with one of the `ontology.protected_iri_prefixes` strings from `config.yaml` is force-included regardless of detection. See "Protecting ontologies from prune" below.

Then drop every class/property/instance not in the keep-set. Properties also drop if both domain and range are pruned.

For `expand`, `prune-expand`, and `build`:

After prune, the new-class and new-relation proposals are minted:

- New classes get IRIs of the form `<default_base_iri><slug>` (e.g. `http://your-personal-ontologist.local/ontology/electric_vehicle`) and the configured `default_parent_iri` (`owl:Thing` unless overridden) as superclass, unless the LLM proposed a parent label that resolves to another existing or just-proposed class.
- New relations get fresh property IRIs; their `DOMAIN` and `RANGE` are resolved against existing-class labels and the just-minted classes. A relation is skipped (and logged) if either endpoint can't be resolved — these come from the model proposing junk endpoints like `DOMAIN: "platforms and mechanisms"` that aren't real classes.

#### One-glance flow

```
docs → chunks (paragraph-first tiktoken split, ~800 tok)
                        │
                        ▼
   Stage 1: per-chunk Groq call (narrowing)
                        │
                        ▼
   _slice_ontology: per-chunk Python (no LLM)
                        │
                        ▼
   Stage 2: per-chunk OpenAI gpt-4.1 call
                        │ (all chunks merged into one dict)
                        ▼
   Stage 3: ONE OpenAI gpt-4.1 dedup call
                        │
                        ▼
   Stage 4: deterministic Python
            Phase A: prune (keep-set + IS-A closure + partners + protected)
            Phase B: expand (mint new classes/relations)
                        │
                        ▼
   write merged.json + merged.owl + manifest + stats + llm_audit.jsonl
```

The Stage 1 → Stage 2 narrowing is the whole reason this scales: without it, every chunk would either need to see the full classes_dict (won't fit even at gpt-4.1's 1M-token context for mid-size ontologies) or have no ontology context at all (which collapses prune/expand into raw generation).

### Protecting ontologies from prune (`protected_iri_prefixes`)

Sometimes you want a particular ontology to **always survive prune**, regardless of whether the document corpus happens to mention it. Configure that in `config/config.yaml`:

```yaml
ontology:
  default_base_iri: http://your-personal-ontologist.local/ontology/
  default_parent_iri: http://www.w3.org/2002/07/owl#Thing
  # IRI prefixes that prune will NEVER remove. Any class whose IRI starts
  # with one of these is force-included in the keep-set regardless of
  # whether the document corpus surfaced it. Property survival follows
  # automatically: any object/data property whose domain or range touches
  # a protected class also survives.
  protected_iri_prefixes:
    - https://your-domain.example.com/ontology/your-curated-ontology
```

Useful for in-house ontologies you maintain by hand (e.g. an `intelligence-artifact` schema) that you want preserved across every `prune` and `prune-expand` run. Match is by exact IRI-prefix `startswith` — pick a stable namespace.

## Inspecting an ontology

Generated `merged.owl` files open directly in [Protégé](https://protege.stanford.edu/) for visual inspection. The earlier in-repo Dash viewer has been moved to `archive/visualizer/`; use Protégé going forward.

## Source-document downloaders

Three standalone CLI utilities live in `source_documents/` for grabbing input documents from free public sources. Each accepts the same `--search` / `--output` / `--max` interface and writes to a destination folder (default: `source_documents/<tool>_<slug-of-search>/`).

### 1) DailyMed — drug Patient-Information PDFs

Searches [DailyMed](https://dailymed.nlm.nih.gov/) (NLM) for drug labels matching a condition and downloads each match's Patient-Information PDF.

```bash
uv run python source_documents/dailymed_download.py \
  --search "diabetes" \
  --output source_documents/pharma_documents \
  --max 10
```

### 2) Web search — top-N pages as plain text

Hits DuckDuckGo's HTML SERP (`https://duckduckgo.com/html/?q=...`), takes the top `--max` results, fetches each page, extracts visible text via BeautifulSoup, and writes one `.txt` per result plus an `_index.json` manifest.

```bash
uv run python source_documents/websearch_download.py \
  --search "GraphRAG ontology techniques" \
  --max 5
```

### 3) SEC EDGAR — financial-report PDFs

Searches SEC EDGAR full-text index for filings matching a company name or ticker (forms `10-K`, `10-Q`, `20-F`, `40-F`, `8-K` by default; override with `--forms`). For each matching filing, walks the filing's index for documents with a `.pdf` extension and downloads them.

Most US 10-Ks ship as iXBRL only — they contain no `.pdf` attachments. Pass `--allow-html` to fall back to the primary HTML 10-K body and convert it to PDF via [WeasyPrint](https://weasyprint.org/) when no native PDF exists. If conversion fails (malformed markup, missing font), the raw `.htm` is written instead so the filing content is preserved.

```bash
# PDF-only (may yield zero files for iXBRL-only issuers):
uv run python source_documents/financial_report_download.py \
  --search "Apple Inc" --max 5

# Permissive: PDFs when available, HTML→PDF conversion otherwise:
uv run python source_documents/financial_report_download.py \
  --search "Apple Inc" --max 5 --allow-html
```

### Shared flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--search` / `-q` | string | *required* | Search term (condition / company / query). |
| `--output` / `-o` | path | per-tool slug | Destination folder. |
| `--max` / `-n` | int | `10` | Cap on matches. |
| `--overwrite` | flag | off | Redownload even if the destination file already exists. |

EDGAR-only: `--allow-html` (see above), `--forms <CSV>` (default `10-K,10-Q,20-F,40-F,8-K`).

## LLM providers

Task → model routing lives in `config/models.yaml`. Phase 1 uses Groq + OpenAI; Phase 2 uses OpenAI exclusively. Defaults:

| Phase | Task | Provider | Model |
|---|---|---|---|
| 1 | `chunk_classification` (document → existing-ontology tagging) | Groq | `llama-3.3-70b-versatile` |
| 1 | `class_proposal` | OpenAI | `gpt-4.1` |
| 1 | `match_dedup` | OpenAI | `gpt-4.1` |
| 1+2 | `class_summarization`, `document_summarize`, `compact_description` | OpenAI | `gpt-4o-mini` |
| 2 | `entity_extract`, `question_parse`, `concept_expansion`, `query_decompose`, `follow_up_resolution` | OpenAI | `gpt-4o-mini` |
| 2 | `artifact_chunk_extract_with_entities` (Claim / Finding / Observation extraction) | OpenAI | `gpt-4o-mini` |
| 2 | `answer_simple_qa`, `answer_conversation_turn` | OpenAI | `gpt-4o-mini` |
| 2 | `answer_deep_research`, `insight_gen`, `recommendation_gen` | OpenAI | `gpt-4.1` |
| 2 | LLM-as-judge (4 metrics) | OpenAI | `gpt-4.1` (override to `gpt-4o-mini` via `--judge-model`) |
| 2 | `embeddings` | OpenAI | `text-embedding-3-small` (**1024 dim** — Phase 2 default) |

`OPENAI_API_KEY` is required; `GROQ_API_KEY` is only needed if you run Phase 1 with `chunk_classification` enabled.

## First-run setup

```bash
# 1. Copy templates and fill in real values (OPENAI_API_KEY, GROQ_API_KEY,
#    and DATABASE_URL — paste your Supabase connection string as-is).
cp .env.example .env
cp config/config.example.yaml config/config.yaml
cp config/models.example.yaml config/models.yaml

# 2. Install Python dependencies
uv sync

# 3. Bring up Postgres + pgvector + Redis locally (skip Postgres if using Supabase)
docker compose up -d

# 4. Run the API + MCP server (single uvicorn process)
uv run uvicorn backend.app.main:app --reload
# → curl http://localhost:8000/health        (REST)
# → MCP discovery at http://localhost:8000/mcp

# 5. (Optional) Run the React UI
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Database URL

`DATABASE_URL` accepts the bare Supabase format directly:

```
postgresql://postgres:PASSWORD@db.<project>.supabase.co:5432/postgres
```

The app normalizes the scheme to `postgresql+asyncpg://` and appends
`?ssl=require` for `*.supabase.co` hosts internally. Supabase ships with
pgvector enabled, so no extra setup is required for the embedding tables.

## Deploying to Render

Two services are created on Render — a Docker web service (FastAPI + MCP at `/mcp`) and a static site (the React UI). Postgres stays external at Supabase. All driven by a single CLI command — no dashboard click-through except the one-time GitHub App install below.

### One-time GitHub App install

Render needs read access to your repo before its API can fetch from it. If `render-init` fails with `Render API 400: ... unfetchable: https://github.com/...`, the GitHub App isn't installed yet. Two paths — whichever you find faster:

**Path A — install directly from GitHub** (most reliable):

1. Go to https://github.com/apps/render
2. Click the green **Install** (or **Configure** if it's already there)
3. Pick your GitHub account/org
4. Either "All repositories" or pick `your-end-to-end-graphrag-implementation`
5. Click **Install** / **Save**

**Path B — trigger the OAuth from Render's New-Service flow:**

1. Render dashboard → click **New +** (top right) → **Web Service**
2. The page should show a "Connect GitHub" or "Configure GitHub App" button — click it
3. Authorize → install on the repo
4. Don't actually create the service from this page — just close the tab once GitHub is connected

After either path, Render's GitHub App is installed on your repo permanently — you won't need to do it again for future deploys.

### Deploy

Required env vars in `.env`: `RENDER_API_KEY`, `DATABASE_URL`, `OPENAI_API_KEY`, `BEARER_TOKEN` (plus optional `GROQ_API_KEY`, `RENDER_OWNER_ID`).

```bash
uv run python -m backend.app.cli render-init
```

One command. It reads `render.yaml` + `.env` + your git origin/branch, creates both services on Render, sets env vars, cross-links `FRONTEND_ORIGIN` ↔ `VITE_API_BASE_URL`, and triggers the first deploys. Idempotent — safe to re-run if anything errored partway through.

Add `--no-deploy` to create the services without auto-triggering builds (useful if you want to review env vars first).

### Lifecycle commands

```bash
uv run python -m backend.app.cli render-status                          # state of both services
uv run python -m backend.app.cli render-deploy --service backend --wait # force a fresh build
uv run python -m backend.app.cli render-logs --service backend --since 10m
uv run python -m backend.app.cli render-suspend --all                   # park overnight to save Free-tier hours
uv run python -m backend.app.cli render-resume --all
uv run python -m backend.app.cli render-takedown --yes                  # soft (suspend both)
uv run python -m backend.app.cli render-takedown --hard --yes           # delete services entirely
```

### Render Free-tier limits

- **Web services**: 750 instance-hours per workspace per month (one always-on web service uses ~720 hr; sleeps after 15 min idle).
- **Static sites**: 100 GB bandwidth + 500 build minutes/month; never sleeps.
- **No free Postgres, Redis, or persistent disks** — Postgres lives at Supabase; Redis is dropped from `render.yaml` until the worker ships; the backend disk was removed since Phase 3 doesn't ingest from the web.
- First request after sleep takes 30-60s to wake (the UI's `LoadingSpinner` switches copy to warn about this).

## Helper scripts

### `source_ontologies/download_ontology.py`

Downloads industry-standard OWL ontologies (HPO, MAxO, ChEBI, OCRe, etc.) into
a local subfolder under `source_ontologies/`. Handy for seeding a domain
ontology before running an import.

The downloaded subfolder is **gitignored**; only the script itself is tracked.

```bash
# Example — Human Phenotype Ontology
uv run python source_ontologies/download_ontology.py \
  "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.owl" \
  "./source_ontologies/pharma_ontologies"
```

Optional flags:
- `--filename my.owl` — override the inferred filename
- `--extract` — auto-unzip if the URL serves a `.zip`

## Layout

```
backend/
  app/
    api/         # FastAPI routes (auto-exposed via MCP at /mcp)
    core/        # config, db, logging
    db/          # SQLAlchemy models + Alembic migrations
    helpers/     # ontology parsing + pruning helpers (relocated from repo root)
    jobs/        # arq workers
    ontology/    # OWL export, IRI utilities
    services/    # ontology I/O, persistence, embeddings, LLM router, retrieval
    cli/         # Phase 1 CLI: merge / prune / expand / prune-expand / build
  Dockerfile     # uv-based image used by Render web + worker services
frontend/              # Vite + React + TS + Tailwind UI (minimal stub; full build in Phase 3)
config/                # *.example.yaml tracked; *.yaml gitignored
source_ontologies/     # drop .owl source files here (subfolders gitignored, top-level files tracked)
source_documents/      # drop PDF/TXT documents here (everything except notes.md gitignored)
output_ontologies/     # versioned exports land here (subfolders gitignored)
render.yaml            # Render blueprint: backend web + worker + frontend static + managed Redis
scripts/
```

## Phase plan

- **Phase 1** ✅ — standalone CLI: `merge` / `prune` / `expand` / `prune-expand` / `build` with versioned output folders.
- **Phase 2** ✅ — Postgres + pgvector knowledge graph. Document ingestion, chunking, embeddings, entity extraction with entity-grounded artifact prompts, temporal enrichment, intelligence artifacts (Claim / Finding / Observation / Summary / Insight / Recommendation). Two-mode retrieval (`simple_qa` / `deep_research`) with multi-probe vector rerank, RRF fusion, and conversation-aware QA. LLM-as-judge eval framework. FastAPI routes + MCP exposure at `/mcp`.
- **Phase 3** (pending) — React UI + graph viz + versioning + retirement + review workflow + Render deploy.

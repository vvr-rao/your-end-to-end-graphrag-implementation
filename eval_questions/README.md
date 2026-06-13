# Evaluation framework

What we measure when we say "the retrieval layer is working" — and
how to run the evaluator yourself against your corpus.

## What we are measuring

Each evaluation run grades every answer on **five tracking metrics**.
Four use an LLM-as-judge (gpt-4.1 by default); one is deterministic.

| # | Metric | Question it answers | Pass threshold |
|---|---|---|---|
| 1 | **Comprehensiveness** | Did the answer actually address every part of what was asked? | mean ≥ 0.80 |
| 2 | **No hallucination** | Are the answer's claims grounded in the retrieved evidence? Penalize claims that go beyond what evidence supports. | mean ≥ 0.90 (the most dangerous failure) |
| 3 | **Consistency** | Run the same question N times (default 3) — are the answers semantically equivalent? | mean ≥ 0.80 |
| 4 | **Gap detection** | When the corpus has NO answer, does the LLM say so instead of fabricating? When it DOES have evidence, does it engage instead of refusing? | mean ≥ 0.85 on `[gap]`-tagged questions |
| 5 | **Time** | Deterministic wall-time per query. | p95 ≤ 10s `simple_qa`; ≤ 60s `deep_research` |

Each metric is a float in `[0.0, 1.0]`. The judge prompts (see
[`backend/app/services/prompts.py`](../backend/app/services/prompts.py))
spell out the score-bands precisely — e.g., for "no hallucination":
1.0 = every claim cited, 0.5 = several claims unsupported, 0.0 =
fabrication.

## The two retrieval modes

The platform supports two modes, both running the same 12-step
retrieval pipeline and diverging only at the answer-synthesis step.
**`deep_research` is the default** if `--mode` is omitted.

| Mode | What it produces | When to use it | Cost / query | Wall |
|---|---|---|---:|---:|
| `simple_qa` | A tight, direct answer in 1-3 sentences. No padding, no broader framing. If the evidence does not answer, it says so in one sentence. | Direct factoid lookup. "What was K+S's potash volume in 2022?" | ~$0.006 | ~30s |
| `deep_research` (DEFAULT) | A structured 7-section answer: **SPECIFICS** → **ANALYSIS** → **ANSWER** → **CONTRADICTIONS** → **KEY CLAIMS (with evidence status)** → **COVERAGE IMBALANCE** → **KEY INSIGHTS**. | Everything that needs structure: comparisons, listings, synthesis, time-anchored questions, broad investigations. | ~$0.07 | ~50-90s |

The 7-section deep_research output is **always rendered**, even when
sections are empty. Empty sections say "None identified" so the
shape stays predictable.

### What each deep_research section contains

| Section | Content |
|---|---|
| **SPECIFICS** | Enumerate the named entities, regulations, events, people, places, dates, and figures from the evidence. Each line cited by IRI. If the user asks about *regulations*, list each one with its name + who passed it + when. If *companies*, list with specific dates/numbers. Verbatim from evidence — no summarizing away. |
| **ANALYSIS** | Synthesis pulling from Finding + Insight artifacts in the evidence pool. Connects the SPECIFICS into a coherent picture. Cited. |
| **ANSWER** | A direct, focused answer to the question, building on the ANALYSIS. Leads with the actual answer — yes/no, the list, the comparison, the cause — whatever was asked for. 2-5 sentences with key specifics inline. The section a reader can consume alone. If the corpus does not address the question, says so explicitly here. |
| **CONTRADICTIONS** | Where two or more sources disagree, names them: *"[doc X] states A, while [doc Y] states B."* If none found: *"None identified in the evidence retrieved."* |
| **KEY CLAIMS (with evidence status)** | Significant claims surfaced for the question. Every claim stated regardless of whether evidence backs it. Two badges per line: (a) who made the claim, (b) whether the source provided supporting evidence (BACKED / PARTIAL / UNBACKED). Backed and unbacked claims mixed together — not split. Uses the new `evidence_status` + `claim_source` metadata on Claim artifacts. |
| **COVERAGE IMBALANCE** | Anywhere the corpus has substantially more material on one side than another. The LLM picks the axes: sub-topics, viewpoints, geographies, time periods, organizations, dimensions, etc. Format: *"The corpus contains N sources on topic A but only M on topic B; <observation about why this matters>."* |
| **KEY INSIGHTS** | 1-2 sentence standout patterns. Cross-period trends (year-over-year, month-over-month). Geographic / organizational patterns. Sudden or unusual changes. Flagged as judgement: *"This pattern suggests..."* Uses the new `time_scope` metadata on Claim artifacts to support cross-period trends. |

## Retrieval methodology (what's being graded)

Every query — both modes — runs the same 12-step pipeline implemented
in [`backend/app/services/retrieval.py`](../backend/app/services/retrieval.py).
Vector search happens **after** the entity and ontology graph has
narrowed the candidate set, not before:

```
1.  (Conversation only) Follow-up resolution → standalone question.
2.  Mode selected by caller.
3.  Question parse → {entities, classes, time_terms, intent}  (gpt-4o-mini, JSON).
4.  Ontology match (vector + pg_trgm) on classes + entities + time_instances.
5.  Concept expansion → +5-15 related class IRIs               (gpt-4o-mini, JSON).
6.  Seed nodes = union of (4) + (5).
7.  Graph BFS through `graph_relationships` up to --hops          (recursive CTE in SQL).
8.  Candidate retrieval — chunks + artifacts touched by the
    expanded node set.
9.  MULTI-PROBE vector rerank (handles complex/comparative queries):
      9a. Query decompose → 1-5 atomic sub-questions             (gpt-4o-mini, JSON).
      9b. Embed all probes (original + sub-questions).
      9c. For each probe, SQL `ORDER BY embedding <-> probe`
          against ONLY the candidate set from step 8.
      9d. Per-candidate, per-probe scores feed step 10.
10. Reciprocal Rank Fusion (RRF) merging:
      - per-probe vector ranks
      - graph-distance ranks (BFS hops)
      - entity-coverage ranks (how many query entities the chunk asserts about)
    Output: top-K candidates. K = 30 for deep_research, 20 for simple_qa.
11. Context engineering — pack top-K with snippets + IRIs into
    a mode-specific prompt.
12. Answer generation. simple_qa → gpt-4o-mini, tight 1-3 sentence.
                        deep_research → gpt-4.1, structured 7-section.
```

Then persist `retrieval_runs` (1 row) + `retrieval_evidence` (top-K
rows). Every answer has a `retrieval_run_id` that resolves the full
evidence chain back through `chunks` to `documents.file_path`.

**Key design constraint**: vector search is constrained to the
already-narrowed candidate set from step 8. We never run a vector
search against the full chunk table for a complex query — that's how
comparative questions like *"How does Vietnam compare to the rest of
Asia?"* historically miss evidence. Decomposition in step 9 sharpens
ranking within the BFS-narrowed set.

## Intelligence-artifact metadata

The Claim/Finding/Observation artifacts that power deep_research's
KEY CLAIMS section carry three metadata fields (in
`intelligence_artifacts.extra_metadata`):

| Field | Values | Used in |
|---|---|---|
| `evidence_status` | `backed` / `partial` / `unbacked` — does the source text supply reasoning or data supporting the claim? | deep_research § KEY CLAIMS (the badge after each claim) |
| `claim_source` | who made the claim (e.g. `"the report itself"`, `"BYD's CEO"`, `"a 2024 Reuters article"`) | deep_research § KEY CLAIMS + § CONTRADICTIONS (attribution) |
| `time_scope` | the period the claim applies to (e.g. `"2024"`, `"first half of 2022"`, `"Q1 2024"`) | deep_research § KEY INSIGHTS (cross-period trends) |

These fields are populated by the
`artifact_chunk_extract_with_entities@v2` prompt during
`generate-artifacts`. If you ingested artifacts under v1 of the
prompt (before 2026-06-13), the fields will be null and deep_research
will fall back gracefully — but the KEY CLAIMS section will lose its
evidence-status badges. Re-run `generate-artifacts` to populate.

## Repo convention — what's tracked vs ignored

This directory ships **`.example.txt` templates** in git but
gitignores any other `*.txt`. Same pattern as
`config/models.example.yaml`.

- `v1_smoke.example.txt` — tracked; the canonical starter set.
- `v1_smoke.txt` (if you create it) — gitignored; your local copy.

To start using an example set:

```bash
cp eval_questions/v1_smoke.example.txt eval_questions/v1_smoke.txt
# edit eval_questions/v1_smoke.txt freely; it won't get committed
```

That keeps your corpus-specific question evolution out of the repo,
while the example stays visible as a starting point for anyone cloning
fresh.

## Running an eval

```bash
# Full smoke against deep_research (default mode), 3 runs each, gpt-4.1 judge.
# 19 questions × 3 runs ≈ $6.50 with gpt-4.1 judge, ~$1.20 with mini.
uv run python -m backend.app.cli evaluate-queries \
  --questions eval_questions/v1_smoke.txt \
  --mode deep_research \
  --runs-per-question 3 \
  --judge-model gpt-4.1 \
  --output /tmp/eval_deep_research.json \
  --output-md /tmp/eval_deep_research.md \
  --max-cost-usd 10.0
```

Faster sanity runs (mini judge, fewer runs):

```bash
uv run python -m backend.app.cli evaluate-queries \
  --questions eval_questions/v1_smoke.txt \
  --mode simple_qa \
  --runs-per-question 2 \
  --judge-model gpt-4o-mini \
  --max-cost-usd 1.0
```

You can also run a tight check against `simple_qa` if you only care
about the factoid questions (the deep_research-shaped questions still
work in `simple_qa` mode, they just won't get the structured output).

## Cost per question

Three factors drive cost: (a) the **mode**, (b) `--runs-per-question
N` (every metric is judged on every run), (c) the **judge model**.

**Cost per ONE question at default `--runs-per-question 3`**:

| Mode | Query × 3 | Judges (gpt-4.1) | Per-question total | With `--judge-model gpt-4o-mini` |
|---|---:|---:|---:|---:|
| `simple_qa` | ~$0.018 | ~$0.27 | **~$0.29** | **~$0.04** |
| `deep_research` | ~$0.21 | ~$0.27 | **~$0.48** | **~$0.23** |

Each "Judges (gpt-4.1)" column = 10 judge calls per question:
3 × `comprehensiveness` + 3 × `no_hallucination` + 3 × `gap_detection`
+ 1 × `consistency` (one call across all 3 answers). ~$0.025-$0.04
per judge call with gpt-4.1, vs ~$0.0025-$0.004 with mini.

**Cost per ONE question at `--runs-per-question 1`** (skips
consistency metric, single answer):

| Mode | Query × 1 | Judges (gpt-4.1) | Per-question total | Mini judge |
|---|---:|---:|---:|---:|
| `simple_qa` | ~$0.006 | ~$0.075 | **~$0.08** | **~$0.01** |
| `deep_research` | ~$0.07 | ~$0.075 | **~$0.15** | **~$0.08** |

**Quick scaling rule**:

| Setup | Per-question rough cost | 19 questions |
|---|---:|---:|
| `simple_qa` × 3 runs × mini judge | ~$0.04 | ~$0.80 |
| `simple_qa` × 3 runs × gpt-4.1 judge | ~$0.30 | ~$5.85 |
| `deep_research` × 3 runs × mini judge | ~$0.23 | ~$4.40 |
| `deep_research` × 3 runs × gpt-4.1 judge | ~$0.48 | ~$9.10 |

The `--max-cost-usd` flag is a hard ceiling — the run aborts as soon
as cumulative spend (queries + judges) crosses it.

**Recommendations**:

- **Iterating on a question set**: `--runs-per-question 1
  --judge-model gpt-4o-mini`. Sub-$0.10/question; fast feedback.
- **Pre-ship sign-off**: `--runs-per-question 3 --judge-model
  gpt-4.1`. The thresholds in this README assume this setup.
- **Daily / weekly regression**: `--runs-per-question 3 --judge-model
  gpt-4o-mini` is usually enough. Trustworthy on comprehensiveness +
  gap_detection; less reliable on no_hallucination (mini-judge has
  trouble tracing citation IRIs back to evidence text).

## Question file format

Plain text, one question per line. Two tag conventions:

```
# Lines starting with `#` are comments.
What is OCI N.V.'s annual nitrogen fertilizer production capacity?

# `[gap]` marks a question the corpus is NOT expected to answer.
# Used by judge_gap_detection: a correct answer says so explicitly.
[gap] What is the current price of Bitcoin?
```

Place files in this directory. The starter set is
`v1_smoke.example.txt` (and your gitignored `v1_smoke.txt` after the
`cp`).

## What the judge actually sees

For each metric, the judge receives:

- `judge_comprehensiveness` → question + answer (no evidence — purely
  about scope coverage).
- `judge_no_hallucination` → question + retrieved evidence + answer.
  Asks: is every claim cited?
- `judge_gap_detection` → question + evidence + answer +
  `expected_gap` boolean. Behavior flips: if `expected_gap=true`,
  refusing scores high; if `expected_gap=false`, refusing scores low.
- `judge_consistency` → question + ordered list of N answers (across
  the N runs). Asks: are these semantically equivalent?

All four prompts return JSON `{score: 0.0-1.0, justification: "..."}`
so per-question justifications carry into the output log for human
review of any flagged question.

## Output

**JSON** (one per eval run, full detail):

```json
{
  "config": {"mode": "deep_research", "runs_per_question": 3, "judge_model": "gpt-4.1",
             "graph_version": 19},
  "results": [
    {
      "question": "...",
      "expected_gap": false,
      "runs": [
        {"answer": "...", "evidence_count": 28, "retrieval_run_id": "...",
         "wall_seconds": 65.2, "cost_usd": 0.071},
        ...
      ],
      "metrics": {
        "comprehensiveness": {"mean": 0.93, "min": 0.85, "max": 1.0, "justifications": ["...", ...]},
        "no_hallucination":  {"mean": 1.0, ...},
        "gap_detection":     {"mean": 0.5, ...},
        "consistency":       {"score": 0.95, "justification": "..."},
        "wall_seconds":      {"mean": 62.1, "min": 58.0, "max": 67.2}
      }
    }
  ],
  "aggregate": {
    "comprehensiveness": {"mean": 0.88, "questions_below_0.7": 2},
    "no_hallucination":  {"mean": 0.96, "questions_below_0.7": 1},
    "gap_detection":     {"mean": 0.81, "by_expected_gap": {"true": 0.92, "false": 0.74}},
    "consistency":       {"mean": 0.89},
    "wall_seconds":      {"mean": 4.3, "p95": 7.1},
    "total_cost_usd":    9.10,
    "questions_count":   19
  }
}
```

**Markdown** (optional `--output-md`): aggregate table + a "flagged"
list of any question where any metric came in below 0.7 — quick
human-review surface.

## Authoring a new question set

1. Decide on the mode mix. `deep_research` exercises every section
   of the structured output; `simple_qa` tests the tight-answer path.
2. Write 5-20 questions in the style appropriate for each mode
   (factoid for `simple_qa`; comparison / listing / synthesis /
   time-anchored for `deep_research`).
3. Include at least 2 `[gap]` lines so `gap_detection` has signal.
4. Save as `eval_questions/<name>.txt` (gitignored) — keep an
   `eval_questions/<name>.example.txt` (tracked) if you want to share
   it.
5. Run the evaluator; iterate on the question set or on the prompts
   until you meet the pass thresholds above.

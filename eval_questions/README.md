# Evaluation framework

What we measure when we say "Milestone F is working" — and how to run
the evaluator yourself against your corpus.

## What we are measuring

Each evaluation run grades every answer on **five tracking metrics**.
Four use an LLM-as-judge (gpt-4.1 by default); one is deterministic.

| # | Metric | Question it answers | Pass threshold |
|---|---|---|---|
| 1 | **Comprehensiveness** | Did the answer actually address every part of what was asked? | mean ≥ 0.80 |
| 2 | **No hallucination** | Are the answer's claims grounded in the retrieved evidence? Penalize claims that go beyond what evidence supports. | mean ≥ 0.90 (the most dangerous failure) |
| 3 | **Consistency** | Run the same question N times (default 3) — are the answers semantically equivalent? | mean ≥ 0.80 |
| 4 | **Gap detection** | When the corpus has NO answer, does the LLM say so instead of fabricating? When it DOES have evidence, does it engage instead of refusing? | mean ≥ 0.85 on `[gap]`-tagged questions |
| 5 | **Time** | Deterministic wall-time per query. | p95 ≤ 10s `simple_qa`; ≤ 30s `deep_research` |

Each metric is a float in `[0.0, 1.0]`. The judge prompts (see
[`backend/app/services/prompts.py`](../backend/app/services/prompts.py))
spell out the score-bands precisely (e.g., for "no hallucination":
1.0 = every claim cited, 0.5 = several claims unsupported, 0.0 =
fabrication).

## Retrieval methodology (what's being graded)

Every query — regardless of mode — runs the same 12-step pipeline
implemented in
[`backend/app/services/retrieval.py`](../backend/app/services/retrieval.py).
Vector search happens **after** the entity and ontology graph has
narrowed the candidate set, not before:

```
1. (Conversation only) Follow-up resolution → standalone question.
2. Mode selected by caller.
3. Question parse → {entities, classes, time_terms, intent}  (gpt-4o-mini, JSON).
4. Ontology match (vector + pg_trgm) on classes + entities + time_instances.
5. Concept expansion → +5-15 related class IRIs               (gpt-4o-mini, JSON).
6. Seed nodes = union of (4) + (5).
7. Graph BFS through `graph_relationships` up to --hops          (recursive CTE in SQL).
8. Candidate retrieval — chunks + artifacts touched by the
   expanded node set.
9. MULTI-PROBE vector rerank (handles complex/comparative queries):
     9a. Query decompose → 1-5 atomic sub-questions             (gpt-4o-mini, JSON).
     9b. Embed all probes (original + sub-questions).
     9c. For each probe, SQL `ORDER BY embedding <-> probe`
         against ONLY the candidate set from step 8.
     9d. Per-candidate, per-probe scores feed into step 10.
10. Reciprocal Rank Fusion (RRF) merging:
       - per-probe vector ranks
       - graph-distance ranks (BFS hops)
       - entity-coverage ranks (how many query entities the chunk asserts about)
     Output: top-K candidates.
11. Context engineering — pack top-K with snippets + IRIs
    into a mode-specific prompt.
12. Answer generation                                            (mode-specific model).
```

Then persist `retrieval_runs` (1 row) + `retrieval_evidence` (top-K
rows). Every answer has a `retrieval_run_id` that resolves the full
evidence chain back through `chunks` to `documents.file_path`.

**Key design constraint**: vector search is constrained to the
already-narrowed candidate set from step 8. We never run a vector
search against the full chunk table for a complex query — that's
how comparative questions like *"How does Vietnam compare to the
rest of Asia?"* historically miss evidence. Decomposition in step 9
sharpens ranking within the BFS-narrowed set.

## The six retrieval modes

All modes share steps 1-10 above; they differ at steps 11 (context
assembly) and 12 (answer model + prompt). Mode selected via
`--mode`.

| Mode | What it's for | What gets ranked higher | Model | Typical cost |
|---|---|---|---|---|
| `simple_qa` | Short factoid answers | Chunks (vector dominates) | gpt-4o-mini | ~$0.006/q |
| `summarize` | Thematic summary across docs | Doc-level + Summary artifacts | gpt-4o-mini | ~$0.006/q |
| `deep_research` | Long-form comparative synthesis | Mix of chunks + Finding/Insight artifacts. **Map-reduce relevance filter** at step 11 — filter retrieved chunks down to relevant extracts before stuffing the gpt-4.1 context (handles long-context queries safely). | gpt-4.1 | ~$0.07/q |
| `insights` | Surface non-obvious patterns | Insight + Finding artifacts. Same map-reduce. | gpt-4.1 | ~$0.07/q |
| `knowledge_gaps` | What does the corpus NOT cover? | LLM identifies sub-questions; reports which find ≥1 high-rel chunk | gpt-4o-mini | ~$0.009/q |
| `exhaustive_search` | Enumeration — "find ALL X" | Hard intersection of entity + class + time constraints. **NO ranking** — every match is returned, grouped by document. One gpt-4o-mini caption per group. **NO synthesis.** | gpt-4o-mini per caption | ~$0.03-0.10/q |

`exhaustive_search` is the only mode where step 9 is **skipped** —
we want every match, not top-K. It also returns a different
envelope shape (no `answer` field; `exhaustive_results` list
instead).

## Running an eval

```bash
# Full smoke (~$5.85 with gpt-4.1 judge, ~$0.85 with mini judge)
uv run python -m backend.app.cli evaluate-queries \
  --questions eval_questions/v1_smoke.txt \
  --mode simple_qa \
  --runs-per-question 3 \
  --judge-model gpt-4.1 \
  --output /tmp/eval_simple_qa.json \
  --output-md /tmp/eval_simple_qa.md \
  --max-cost-usd 8.0
```

Sanity / smoke runs (a few questions, mini judge):

```bash
uv run python -m backend.app.cli evaluate-queries \
  --questions eval_questions/v1_smoke.txt \
  --mode simple_qa \
  --runs-per-question 2 \
  --judge-model gpt-4o-mini \
  --max-cost-usd 1.0
```

## Question file format

Plain text, one question per line. Two tag conventions:

```
# Lines starting with `#` are comments.
What is OCI N.V.'s annual nitrogen fertilizer production capacity?

# `[gap]` marks a question the corpus is NOT expected to answer.
# Used by judge_gap_detection: a correct answer says so explicitly.
[gap] What is the current price of Bitcoin?
```

Place files in this directory. The starter set is `v1_smoke.txt`.

## What the judge actually sees

For each metric, the judge receives:

- `judge_comprehensiveness` → question + answer (no evidence — purely
  about scope coverage).
- `judge_no_hallucination` → question + retrieved evidence + answer.
  Asks: is every claim cited?
- `judge_gap_detection` → question + evidence + answer + `expected_gap`
  boolean. Behavior flips: if `expected_gap=true`, refusing scores
  high; if `expected_gap=false`, refusing scores low.
- `judge_consistency` → question + ordered list of N answers (across
  the N runs). Asks: are these semantically equivalent?

All four prompts return JSON `{score: 0.0-1.0, justification: "..."}`
so we can carry per-question justifications into the output log for
human review of any flagged question.

## Output

**JSON** (one per eval run, full detail):
```json
{
  "config": {"mode": "simple_qa", "runs_per_question": 3, "judge_model": "gpt-4.1",
             "graph_version": 19},
  "results": [
    {
      "question": "...",
      "expected_gap": false,
      "runs": [
        {"answer": "...", "evidence_count": 8, "retrieval_run_id": "...",
         "wall_seconds": 4.2, "cost_usd": 0.0061},
        ...
      ],
      "metrics": {
        "comprehensiveness": {"mean": 0.93, "min": 0.85, "max": 1.0,
                              "justifications": ["...", "...", "..."]},
        "no_hallucination":  {"mean": 1.0, ...},
        "gap_detection":     {"mean": 0.5,  ...},
        "consistency":       {"score": 0.95, "justification": "..."},
        "wall_seconds":      {"mean": 4.17, "min": 3.9, "max": 4.4}
      }
    }
  ],
  "aggregate": {
    "comprehensiveness": {"mean": 0.88, "questions_below_0.7": 2},
    "no_hallucination":  {"mean": 0.96, "questions_below_0.7": 1},
    "gap_detection":     {"mean": 0.81,
                          "by_expected_gap": {"true": 0.92, "false": 0.74}},
    "consistency":       {"mean": 0.89},
    "wall_seconds":      {"mean": 4.3, "p95": 7.1},
    "total_cost_usd":    5.91,
    "questions_count":   19
  }
}
```

**Markdown** (optional `--output-md`): aggregate table + a "flagged"
list of any question where any metric came in below 0.7 — quick
human-review surface.

## Authoring a new question set

1. Pick the mode you're targeting.
2. Write 5-20 questions in the style appropriate for that mode
   (factoid for `simple_qa`, comparative for `deep_research`,
   enumeration for `exhaustive_search`, etc.).
3. Include at least 2 `[gap]` lines so gap_detection has signal.
4. Save as `eval_questions/<name>.txt`.
5. Run the evaluator; iterate on the question set or on F until you
   meet the pass thresholds above.

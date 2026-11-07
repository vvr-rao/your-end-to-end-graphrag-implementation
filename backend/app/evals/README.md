# Standing evaluations

Two evaluations, run every time a graph is built and every time a set of
questions is tested. Reports are printed and saved as JSON under
`.build/evals/<database>/` (gitignored); `--out` writes elsewhere.

## Corpus evaluation (`eval-corpus`)

Runs automatically at the end of `extract-entities`. Run it again after any
later step that changes entities or relationships (enrich-geo, a re-extract,
document deletes). Free: database queries only.

```
uv run python -m backend.app.cli eval-corpus
```

Entities are split three ways: **concepts** (typed under
`extraction.concept_class_roots`, minus those judged named things),
**places** (typed under a geographic class, `geo_enrichment.class_labels`) and
other **named entities**. "Entities" below means named entities including
places.

| Metric | Definition |
|---|---|
| Entities / concepts per chunk | Mean and median, per chunk kind (summary, fulltext), counted from `viao:assertsAbout` mention edges |
| Chunks without entities or concepts | Per chunk kind; also without entities, and without concepts |
| Relationships | Entity-to-entity edges, split entity-entity, entity-concept, concept-concept; place-containment edges counted separately |
| Entities per class | Mean and median over classes with at least one entity, for entity classes, concept classes and both |
| Classes with >50 entities | Count and the largest classes |
| Relationships per node | Entities and concepts with 0, 1, 2, 3 and more than 3 relationships |

## Question evaluation (`eval-questions`)

Run it when a question set is created, and on every question test.

```
# a test set (questions + gold documents): runs retrieval without the answer call
uv run python -m backend.app.cli eval-questions eval_questions/testsets/fingen.json [--k 10] [--answer]
# a plain question file works too (no gold, so no Recall/Precision/MRR)
uv run python -m backend.app.cli eval-questions eval_questions/my_set.txt
# a test set kept in LangSmith
uv run python -m backend.app.cli eval-questions --dataset graphrag-fingen

# a finished test: reads the traces stored with each query run (free)
uv run python -m backend.app.cli eval-questions --logs results/q*.log
uv run python -m backend.app.cli eval-questions --run-ids <id> <id> ...
# add gold to finished runs
uv run python -m backend.app.cli eval-questions --logs results/q*.log --testset eval_questions/testsets/fingen.json
```

`evaluate-queries`, `scripts/run_audit_questions.py` and `scripts/bench/eval_arm.py`
print and save this report at the end of every run; `extract-entities` and
`enrich-geo` print the corpus evaluation.

Long test sets run in batches: `evals.batch_size` (25) questions per child
process, each saving its traces under `--traces-dir` (default a new folder in
`.build/evals/<database>/`). Memory stays flat, a crash loses one batch, and
re-running with the same `--traces-dir` skips finished batches. `--batch-size 0`
runs everything in one process. Run one test set at a time on the 2.7 GB dev VM.

The question-file form costs the question parse only (a fraction of a cent
per question on the cheap presets). Every `query` run stores its trace in
`retrieval_runs.retrieval_plan.eval_trace`; runs made before 2026-11-07 have
none and are reported as "without a trace".

**Metric 1: what each question ties to.** Its seeds: named entities, concepts,
places (seeded, or named in the question text) and time periods; and their
classes: entity classes, concept classes, place classes and time levels. Also
the ontology classes the parse matched directly and the concept-area classes
its terms name. The summary counts questions that tie to nothing, to no
entities, to no concepts, and to places and time.

**Metric 2: what the graph walk found.** Entities the walk reached, the
summary chunks those entities are mentioned in, and how many of those chunks
reached the final evidence. deep_research reports its final round; round 1 is
kept in the JSON.

**Retrieval metrics (test sets with gold documents).** The ranked evidence
chunks become a ranked list of distinct documents (a document counts at its
first chunk; artifacts are not counted). K comes from `evals.k` (10) or `--k`.

| Metric | Definition |
|---|---|
| Recall@K | gold documents in the top K / all gold documents |
| Precision@K | gold documents in the top K / K |
| MRR@K | 1 / rank of the first gold document within the top K, else 0 |

Questions with no gold documents get the coverage metrics and are left out of
these means; the report says how many.

Also per question: whether the document arm fired and why, and what the
full-text bridge's document limit cut, split by the route that brought each
document in (graph routes, the document arm only, or neither).

## Test sets (`eval_questions/testsets/<corpus>.json`)

One file per corpus: each question with the documents that answer it.

```json
{"format": "graphrag-testset/1", "name": "fingen",
 "corpus": "source_documents/fin-gen-subset", "description": "...",
 "questions": [{"id": "q001", "question": "...",
   "gold_documents": ["2025_1Q_LGES_Audit_Report_FS_en.pdf"],
   "supporting_documents": [], "tags": ["finance"]}]}
```

- Documents are named by file name, so a test set works on any database built
  from that corpus.
- `gold_documents` (one or more) are scored. `supporting_documents` are
  optional and not scored. A question may have no gold.
- The folder is gitignored: test sets stay private to the working copy (and to
  your LangSmith workspace if you upload them). Local sets: `websearch3x`,
  `pharma12` (converted from the graded gold files: grade 2 -> gold,
  1 -> supporting), `fingen`, `finqa`, `multihop40`.

## LangSmith (optional)

With `LANGSMITH_API_KEY` in `.env` (next to the other API keys), every question
evaluation is uploaded automatically: the test set becomes a dataset
`<langsmith.dataset_prefix>-<name>` (only missing questions are added), and the
run becomes an experiment named `<langsmith.user>-<test set>-<mode>`. Each trace
carries the answer (when one was generated), the ranked evidence as document
names and chunk ids, and every score above as a feedback column. Chunk text is
never uploaded. Without a key nothing is sent and nothing fails;
`--no-langsmith` skips one run. Set `langsmith.user` in config.yaml (uploads
are skipped with a message until it is).

```
uv run python -m backend.app.cli eval-dataset push eval_questions/testsets/fingen.json
uv run python -m backend.app.cli eval-dataset list
uv run python -m backend.app.cli eval-dataset pull graphrag-fingen [--out FILE]
```

"""Standing evaluations, run after every graph build and every question test.

* `corpus_eval`   -- how well the graph covers the corpus (`eval-corpus`).
* `question_eval` -- what a question set ties to in the graph and what the
  walk finds for it (`eval-questions`).

Reports are printed and saved as JSON under `.build/evals/<database>/`.
See README.md in this folder for the metric definitions.
"""

"""Standing evaluations: the pure parts (database parts run in eval-corpus /
eval-questions)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.evals import langsmith_sync
from backend.app.evals.common import buckets, mean, median
from backend.app.evals.corpus_eval import format_corpus_eval
from backend.app.evals.question_eval import (
    _row,
    _summary,
    format_question_eval,
    read_questions,
    run_ids_from_logs,
    runs_from_logs,
    trace_from_result,
)
from backend.app.evals.retrieval_metrics import ranked_documents, retrieval_metrics
from backend.app.evals.testset import (
    TestQuestion,
    TestSet,
    from_dict,
    from_graded_gold,
    load_testset,
)

# ---------------------------------------------------------------- helpers


def test_relationship_counts_fall_into_five_buckets():
    assert buckets([0, 0, 1, 2, 3, 4, 9]) == {"0": 2, "1": 1, "2": 1, "3": 1, ">3": 2}


def test_mean_and_median_of_nothing_are_zero():
    assert mean([]) == 0.0
    assert median([]) == 0.0
    assert median([1, 3, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5


# ---------------------------------------------------------------- retrieval metrics


def test_documents_are_ranked_by_first_chunk_once():
    assert ranked_documents(["a", "b", "a", None, "c", "b"]) == ["a", "b", "c"]


def test_gold_at_rank_one():
    m = retrieval_metrics(["g", "x", "y"], ["g"], 10)
    assert m["recall@10"] == 1.0
    assert m["precision@10"] == 0.1
    assert m["mrr@10"] == 1.0


def test_gold_just_past_k_does_not_count():
    ranked = [f"d{i}" for i in range(1, 11)] + ["g"]
    m = retrieval_metrics(ranked, ["g"], 10)
    assert m["recall@10"] == 0.0
    assert m["mrr@10"] == 0.0
    assert m["gold_missed"] == ["g"]


def test_partial_recall_and_second_rank():
    m = retrieval_metrics(["x", "g1", "y"], ["g1", "g2"], 3)
    assert m["recall@3"] == 0.5
    assert m["precision@3"] == pytest.approx(0.3333, abs=1e-4)
    assert m["mrr@3"] == 0.5


def test_precision_divides_by_k_even_when_fewer_documents():
    assert retrieval_metrics(["g"], ["g"], 10)["precision@10"] == 0.1


def test_no_gold_is_not_scored():
    assert retrieval_metrics(["a"], [], 10) is None


# ---------------------------------------------------------------- test sets


def _ts_dict():
    return {"format": "graphrag-testset/1", "name": "demo", "corpus": "c",
            "questions": [{"id": "q1", "question": "Who?", "gold_documents": ["a.txt"],
                           "tags": ["x"]},
                          {"question": "Where?"}]}


def test_a_test_set_round_trips(tmp_path: Path):
    ts = from_dict(_ts_dict())
    p = tmp_path / "t.json"
    ts.save(p)
    back = load_testset(p)
    assert back.to_dict() == ts.to_dict()
    assert back.questions[1].id == "q002"
    assert back.questions[1].gold_documents == []


@pytest.mark.parametrize("bad, msg", [
    ({"name": "x", "questions": []}, "no questions"),
    ({"questions": [{"question": "a"}]}, "no name"),
    ({"name": "x", "format": "other", "questions": [{"question": "a"}]}, "unknown format"),
    ({"name": "x", "questions": [{"question": "a"}, {"question": "a"}]}, "duplicate question"),
    ({"name": "x", "questions": [{"id": "1", "question": "a"},
                                 {"id": "1", "question": "b"}]}, "duplicate id"),
    ({"name": "x", "questions": [{"question": "a", "gold_documents": "a.txt"}]}, "list of"),
])
def test_invalid_test_sets_say_why(bad, msg):
    with pytest.raises(ValueError, match=msg):
        from_dict(bad)


def test_a_question_file_becomes_a_test_set_without_gold(tmp_path: Path):
    p = tmp_path / "my set.txt"
    p.write_text("# set\nFirst?\n\n  Second?  \n", encoding="utf-8")
    ts = load_testset(p)
    assert [q.question for q in ts.questions] == ["First?", "Second?"]
    assert ts.name == "my-set"
    assert read_questions(p) == ["First?", "Second?"]


def test_graded_gold_converts_grades():
    gold = {"_grades": "...", "set": {"Q?": {"D01": 2, "D02": 1, "D03": 0}}}
    ts = from_graded_gold("g", gold, {"D01": "a.txt", "D02": "b.txt"})
    q = ts.questions[0]
    assert (q.gold_documents, q.supporting_documents, q.tags) == (["a.txt"], ["b.txt"], ["set"])


def test_shipped_test_sets_are_valid():
    for p in Path("eval_questions/testsets").glob("*.json"):
        assert load_testset(p).questions, p


# ---------------------------------------------------------------- question report


def test_run_ids_and_answers_are_read_from_query_logs(tmp_path: Path):
    a = tmp_path / "a.log"
    a.write_text("MODE: simple_qa\nRUN_ID:   abc-1\nANSWER:\nIt is X.\nEVIDENCE (2 items):\n",
                 encoding="utf-8")
    b = tmp_path / "b.log"
    b.write_text("no run here\n", encoding="utf-8")
    assert run_ids_from_logs([a, b]) == ["abc-1"]
    assert runs_from_logs([a]) == [("abc-1", "It is X.")]


def _fake_result():
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    return SimpleNamespace(
        answer="ans", retrieval_run_id=None,
        evidence=[{"kind": "chunk", "node_id": c2, "rank": 2},
                  {"kind": "artifact", "node_id": uuid.uuid4(), "rank": 1},
                  {"kind": "chunk", "node_id": c1, "rank": 1}],
        parsed={"eval_trace": {"seed_entities": [], "walk_summary_chunks": 4}}), c1, c2


def test_a_result_trace_keeps_chunk_order_and_drops_artifacts():
    res, c1, c2 = _fake_result()
    t = trace_from_result("Q?", res)
    assert t["evidence_chunk_ids"] == [str(c1), str(c2)]
    assert t["walk_summary_chunks"] == 4
    assert t["answer"] == "ans"


def test_a_row_scores_gold_and_keeps_round_one():
    e = uuid.uuid4()
    res, c1, c2 = _fake_result()
    t = trace_from_result("Q?", res) | {
        "seed_entities": [str(e)], "seed_times": ["t1"],
        "round1": {"question": "R1?", "seed_entities": [str(e)], "walk_chunks": 3}}
    tq = TestQuestion(id="q1", question="Q?", gold_documents=["b.txt"])
    row = _row(t, tq, {e: ("place", "cls")}, {"t1": "year"}, {str(e): "Asia"},
               {str(c1): "a.txt", str(c2): "b.txt"}, 10)
    assert (row["places"], row["place_classes"], row["time_periods"], row["time_levels"]) == (1, 1, 1, 1)
    assert row["ranked_documents"] == ["a.txt", "b.txt"]
    assert row["retrieval"]["mrr@10"] == 0.5
    assert row["round1"] == {"question": "R1?", "seed_entities": ["Asia"], "seed_times": 0,
                             "walk_summary_chunks": 3}
    report = {"summary": _summary([row], 10, TestSet("demo", [tq])), "questions": [row]}
    text = format_question_eval(report)
    assert "recall@10 1.0" in text
    assert "test set demo" in text


def test_a_question_without_gold_is_left_out_of_the_means():
    row = _row({"question": "Q?"}, None, {}, {}, {}, {}, 10)
    s = _summary([row], 10, None)
    assert s["retrieval"]["scored_questions"] == 0
    assert s["retrieval"]["without_gold"] == 1
    assert "    -     -    -" in format_question_eval({"summary": s, "questions": [row]})


# ---------------------------------------------------------------- corpus report


def test_corpus_report_formats():
    chunk = {"chunks": 10, "entities_per_chunk_mean": 2.0, "entities_per_chunk_median": 2,
             "places_per_chunk_mean": 0.5, "concepts_per_chunk_mean": 1.0,
             "concepts_per_chunk_median": 1, "chunks_without_entities_or_concepts": 1,
             "chunks_without_entities": 2, "chunks_without_concepts": 3}
    cls = {"classes_with_entities": 4, "entities_per_class_mean": 3.0,
           "classes_over_50": 1, "classes_over_50_list": [("Organization", 60)]}
    text = format_corpus_eval({
        "entities": 12, "places": 3, "concepts": 5, "chunks": {"summary": chunk},
        "relationships": {"total": 7, "entity_entity": 4, "entity_concept": 2,
                          "concept_concept": 1, "of_which_place_containment": 2},
        "classes": {"entity": cls, "concept": cls, "all": cls},
        "relationships_per_node": {"entities": buckets([0, 1]), "concepts": buckets([5])},
    })
    assert "Organization 60" in text
    assert "entity-concept 2" in text
    assert "without entities or concepts 1" in text


# ---------------------------------------------------------------- LangSmith (no network)


class _StubClient:
    def __init__(self, existing: list[dict] | None = None):
        self.created: list[dict] = []
        self.existing = existing
        self.datasets: dict[str, SimpleNamespace] = {}

    def has_dataset(self, *, dataset_name):
        return self.existing is not None

    def read_dataset(self, *, dataset_name):
        return SimpleNamespace(id="ds1", name=dataset_name, description="d",
                               metadata={"corpus": "c"})

    def create_dataset(self, name, *, description, metadata):
        return SimpleNamespace(id="ds1", name=name)

    def list_examples(self, dataset_id=None, dataset_name=None):
        return [SimpleNamespace(**e) for e in (self.existing or [])]

    def create_examples(self, *, dataset_id, examples):
        self.created.extend(examples)

    def update_example(self, *, example_id, inputs, outputs, metadata):
        self.updated = [*getattr(self, "updated", []), (example_id, outputs)]


@pytest.fixture
def ls_config(monkeypatch):
    monkeypatch.setattr(langsmith_sync, "settings",
                        lambda: {"user": "tester", "dataset_prefix": "graphrag",
                                 "endpoint": "x"})


def test_push_creates_the_dataset_with_every_question(ls_config):
    stub = _StubClient()
    out = langsmith_sync.push_testset(from_dict(_ts_dict()), stub)
    assert out == {"dataset": "graphrag-demo", "created": True, "added": 2, "updated": 0,
                   "already_there": 0}
    ex = stub.created[0]
    assert ex["inputs"] == {"question": "Who?"}
    assert ex["outputs"]["gold_documents"] == ["a.txt"]
    assert ex["metadata"] == {"id": "q1", "tags": ["x"], "corpus": "c", "user": "tester"}


def test_push_adds_only_missing_questions(ls_config):
    same = langsmith_sync.example_payload(from_dict(_ts_dict()).questions[0], "c", "tester")
    stub = _StubClient(existing=[{"id": "e1", **same}])
    out = langsmith_sync.push_testset(from_dict(_ts_dict()), stub)
    assert (out["created"], out["added"], out["updated"], out["already_there"]) == (
        False, 1, 0, 1)


def test_push_updates_questions_whose_gold_changed(ls_config):
    stub = _StubClient(existing=[{"id": "e1", "metadata": {"id": "q1"},
                                  "inputs": {"question": "Who?"}, "outputs": {}}])
    data = _ts_dict()
    data["questions"][0]["answer_strings"] = ["42"]
    out = langsmith_sync.push_testset(from_dict(data), stub)
    assert out["updated"] == 1
    assert stub.updated[0][1]["answer_strings"] == ["42"]


def test_pull_rebuilds_a_test_set(ls_config):
    stub = _StubClient(existing=[
        {"metadata": {"id": "q2", "tags": []}, "inputs": {"question": "B?"}, "outputs": {}},
        {"metadata": {"id": "q1", "tags": ["t"]}, "inputs": {"question": "A?"},
         "outputs": {"gold_documents": ["a.txt"]}}])
    ts = langsmith_sync.pull_testset("graphrag-demo", stub)
    assert ts.name == "demo"
    assert [q.id for q in ts.questions] == ["q1", "q2"]
    assert ts.questions[0].gold_documents == ["a.txt"]


def test_no_user_means_no_upload(monkeypatch):
    monkeypatch.setattr(langsmith_sync, "settings", lambda: {"user": ""})
    with pytest.raises(langsmith_sync.LangSmithNotConfiguredError, match=r"langsmith\.user"):
        langsmith_sync.user()


def test_trace_outputs_carry_scores_but_no_chunk_text():
    row = {"answer": "A", "ranked_documents": ["a.txt"], "evidence_chunk_ids": ["c1"],
           "doc_arm_fired": True, "entities": 2,
           "retrieval": {"recall@10": 1.0, "precision@10": 0.1, "mrr@10": 1.0,
                         "gold_found": ["a.txt"], "gold_missed": []}}
    out = langsmith_sync.trace_outputs(row, 10)
    assert out["scores"]["recall@10"] == 1.0
    assert out["scores"]["doc_arm_fired"] == 1
    assert out["scores"]["entities"] == 2
    assert "text" not in json.dumps(out)
    keys = [r["key"] for r in langsmith_sync._scores_evaluator(
        SimpleNamespace(outputs=out), None)["results"]]
    assert "mrr@10" in keys


def test_without_a_key_nothing_is_uploaded(monkeypatch):
    calls = []
    monkeypatch.setattr(langsmith_sync, "enabled", lambda: False)
    monkeypatch.setattr(langsmith_sync, "upload_experiment", lambda *a, **k: calls.append(1))
    langsmith_sync.maybe_upload({}, from_dict(_ts_dict()), mode="simple_qa")
    assert calls == []


def test_an_upload_error_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    logged = []
    monkeypatch.setattr(langsmith_sync, "enabled", lambda: True)
    monkeypatch.setattr(langsmith_sync, "upload_experiment", boom)
    langsmith_sync.maybe_upload({}, from_dict(_ts_dict()), mode="simple_qa", log=logged.append)
    assert "network down" in logged[0]


def test_adhoc_question_ids_come_from_the_text():
    from backend.app.evals.question_eval import _testset_for

    a = _testset_for(["Q1?", "Q2?"], None, "runs")
    b = _testset_for(["Q2?"], None, "runs")
    assert a.questions[1].id == b.questions[0].id
    assert a.questions[0].id != b.questions[0].id


# --- batched eval-questions ---------------------------------------------------

def _fake_child(calls):
    import json as _json

    def run(cmd, check=False):
        qfile = cmd[cmd.index("--batch-questions") + 1]
        out = cmd[cmd.index("--batch-out") + 1]
        qs = _json.loads(open(qfile, encoding="utf-8").read())
        calls.append(qs)
        with open(out, "w", encoding="utf-8") as f:
            _json.dump({"traces": [{"question": q} for q in qs], "cost_usd": 0.01}, f)

        class R:
            returncode = 0
        return R()
    return run


def test_batches_run_in_order_and_merge(tmp_path, monkeypatch):
    import subprocess

    from backend.app.evals.question_eval import traces_in_batches

    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_child(calls))
    qs = [f"q{i}" for i in range(7)]
    traces, cost = traces_in_batches(qs, batch_size=3, traces_dir=tmp_path, log=lambda m: None)
    assert [len(c) for c in calls] == [3, 3, 1]
    assert [t["question"] for t in traces] == qs
    assert cost == pytest.approx(0.03)


def test_batches_resume_skips_finished(tmp_path, monkeypatch):
    import subprocess

    from backend.app.evals.question_eval import traces_in_batches

    calls: list = []
    monkeypatch.setattr(subprocess, "run", _fake_child(calls))
    qs = [f"q{i}" for i in range(6)]
    traces_in_batches(qs, batch_size=3, traces_dir=tmp_path, log=lambda m: None)
    (tmp_path / "batch-002.json").unlink()  # pretend batch 2 was lost in a crash
    calls.clear()
    traces, _ = traces_in_batches(qs, batch_size=3, traces_dir=tmp_path, log=lambda m: None)
    assert calls == [["q3", "q4", "q5"]]
    assert [t["question"] for t in traces] == qs


def test_batches_refuse_a_different_question_list(tmp_path, monkeypatch):
    import subprocess

    from backend.app.evals.question_eval import traces_in_batches

    monkeypatch.setattr(subprocess, "run", _fake_child([]))
    traces_in_batches(["a", "b"], batch_size=1, traces_dir=tmp_path, log=lambda m: None)
    with pytest.raises(ValueError):
        traces_in_batches(["a", "c"], batch_size=1, traces_dir=tmp_path, log=lambda m: None)


def test_failed_batch_stops_with_resume_hint(tmp_path, monkeypatch):
    import subprocess

    from backend.app.evals.question_eval import traces_in_batches

    class R:
        returncode = 1
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: R())
    with pytest.raises(RuntimeError, match="--traces-dir"):
        traces_in_batches(["a", "b"], batch_size=1, traces_dir=tmp_path, log=lambda m: None)


# --- answer strings (passage-level, optional) --------------------------------

def test_answer_strings_are_optional_and_validated():
    data = _ts_dict()
    assert from_dict(data).questions[0].answer_strings == []
    data["questions"][0]["answer_strings"] = ["1,234"]
    data["questions"][0]["answer_match"] = "any"
    q = from_dict(data).questions[0]
    assert (q.answer_strings, q.answer_match) == (["1,234"], "any")
    data["questions"][0]["answer_match"] = "some"
    with pytest.raises(ValueError, match="answer_match"):
        from_dict(data)
    data["questions"][0]["answer_match"] = "all"
    data["questions"][0]["answer_strings"] = ["ok", ""]
    with pytest.raises(ValueError, match="answer_strings"):
        from_dict(data)


def test_answer_metrics_all_any_rank_and_reply():
    from backend.app.evals.retrieval_metrics import answer_metrics

    ev = ["intro text", "Revenue was  29,031,560 KRW", "profit 2,177,068"]
    m = answer_metrics(ev, ["29,031,560", "2,177,068"])
    assert (m["answer_recall"], m["answer_found"], m["answer_rank"], m["answer_mrr"]) == (
        1.0, 1, 2, 0.5)
    m = answer_metrics(ev[:2], ["29,031,560", "2,177,068"])
    assert (m["answer_recall"], m["answer_found"], m["answers_missed"]) == (
        0.5, 0, ["2,177,068"])
    m = answer_metrics(ev[:2], ["29,031,560", "2,177,068"], match="any")
    assert (m["answer_recall"], m["answer_found"]) == (1.0, 1)
    assert answer_metrics(["REVENUE WAS 29,031,560"], ["revenue was 29,031,560"])[
        "answer_found"] == 1
    assert "answer_in_reply" not in m
    m = answer_metrics(ev, ["2,177,068"], reply="It made 2,177,068.")
    assert m["answer_in_reply"] == 1.0
    assert answer_metrics(ev, []) is None
    m = answer_metrics(["nothing"], ["42"])
    assert (m["answer_found"], m["answer_mrr"], m["answer_rank"]) == (0, 0.0, None)


def test_answer_summary_only_over_questions_that_have_strings():
    from backend.app.evals.question_eval import _answer_summary

    rows = [{"answers": {"answer_recall": 1.0, "answer_found": 1, "answer_mrr": 0.5}},
            {"answers": {"answer_recall": 0.0, "answer_found": 0, "answer_mrr": 0.0}},
            {"answers": None}]
    s = _answer_summary(rows)
    assert (s["scored_questions"], s["without_answer_strings"]) == (2, 1)
    assert (s["answer_recall"], s["answer_found"], s["answer_mrr"]) == (0.5, 0.5, 0.25)
    assert "answer_in_reply" not in s
    assert _answer_summary([{"answers": None}]) is None


def test_trace_outputs_carry_answer_scores_only_when_present():
    base = {"answer": "", "ranked_documents": [], "evidence_chunk_ids": []}
    assert "answer_recall" not in langsmith_sync.trace_outputs(base, 10)["scores"]
    out = langsmith_sync.trace_outputs(base | {"answers": {
        "answer_recall": 0.5, "answer_found": 0, "answer_mrr": 1.0,
        "answers_found": ["a"], "answers_missed": ["b"]}}, 10)
    assert out["scores"]["answer_recall"] == 0.5
    assert out["answers_missed"] == ["b"]

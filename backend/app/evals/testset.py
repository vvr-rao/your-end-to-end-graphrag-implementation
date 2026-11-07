"""Per-corpus test sets: questions with the documents that answer them.

Format (`eval_questions/testsets/<corpus>.json`):

    {"format": "graphrag-testset/1",
     "name": "websearch3x",
     "corpus": "source_documents/websearch-geo-time-3x",
     "description": "...",
     "questions": [
        {"id": "q001", "question": "...",
         "gold_documents": ["001_....txt"],        # hold the answer; scored
         "supporting_documents": ["..."],          # optional, not scored
         "answer_strings": ["29,031,560"],         # optional: text the answer holds
         "answer_match": "all",                    # or "any"
         "tags": ["region"]}]}

Documents are named by their file name (`documents.file_name`), so a test set
works on any database built from that corpus. A question may have no gold
documents: it still gets the coverage metrics and is left out of the retrieval
metrics.

`answer_strings` is optional: verbatim text (a figure, a name) that a passage
holding the answer contains. When a question has them, the evaluation checks
whether the retrieved evidence contains them -- passage-level, where the
document metrics only see whole documents. `answer_match` says whether every
string is needed ("all", the default) or any one ("any").
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "graphrag-testset/1"
ANSWER_MATCH = ("all", "any")


@dataclass
class TestQuestion:
    __test__ = False  # not a pytest class
    id: str
    question: str
    gold_documents: list[str] = field(default_factory=list)
    supporting_documents: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    answer_strings: list[str] = field(default_factory=list)
    answer_match: str = "all"


@dataclass
class TestSet:
    __test__ = False  # not a pytest class
    name: str
    questions: list[TestQuestion]
    corpus: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "name": self.name, "corpus": self.corpus,
                "description": self.description,
                "questions": [asdict(q) for q in self.questions]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    def by_question(self) -> dict[str, TestQuestion]:
        return {q.question: q for q in self.questions}


def _names(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{where}: must be a list of document file names")
    return [v.strip() for v in value]


def _strings(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{where}: must be a list of non-empty strings")
    return [v.strip() for v in value]


def from_dict(data: dict[str, Any]) -> TestSet:
    """Validate and build a test set. Raises ValueError naming the problem."""
    if not isinstance(data, dict):
        raise ValueError("a test set is a JSON object")
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError(f"unknown format {data.get('format')!r}; expected {FORMAT}")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError("test set has no name")
    raw = data.get("questions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("test set has no questions")
    questions: list[TestQuestion] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for i, q in enumerate(raw, 1):
        if not isinstance(q, dict):
            raise ValueError(f"question {i}: must be an object")
        text = str(q.get("question") or "").strip()
        if not text:
            raise ValueError(f"question {i}: empty question")
        qid = str(q.get("id") or f"q{i:03d}").strip()
        if qid in seen_ids:
            raise ValueError(f"question {i}: duplicate id {qid!r}")
        if text in seen_text:
            raise ValueError(f"question {i}: duplicate question {text[:60]!r}")
        seen_ids.add(qid)
        seen_text.add(text)
        tags = q.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError(f"question {i}: tags must be a list")
        match = str(q.get("answer_match") or "all").strip().lower()
        if match not in ANSWER_MATCH:
            raise ValueError(f"question {i}: answer_match must be one of {ANSWER_MATCH}")
        questions.append(TestQuestion(
            id=qid, question=text,
            gold_documents=_names(q.get("gold_documents"), f"question {i} gold_documents"),
            supporting_documents=_names(
                q.get("supporting_documents"), f"question {i} supporting_documents"),
            tags=[str(t) for t in tags],
            answer_strings=_strings(q.get("answer_strings"), f"question {i} answer_strings"),
            answer_match=match))
    return TestSet(name=name, questions=questions, corpus=str(data.get("corpus") or ""),
                   description=str(data.get("description") or ""))


def load_testset(path: Path) -> TestSet:
    """A `.json` test set, or a plain question file (one per line, # comments),
    which becomes a test set with no gold documents."""
    if path.suffix.lower() == ".json":
        return from_dict(json.loads(path.read_text(encoding="utf-8")))
    return from_txt(path)


def from_txt(path: Path) -> TestSet:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    return TestSet(name=re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem),
                   questions=[TestQuestion(id=f"q{i:03d}", question=q)
                              for i, q in enumerate(lines, 1)])


def from_graded_gold(name: str, gold: dict[str, Any], doc_keys: dict[str, str], *,
                     corpus: str = "", description: str = "") -> TestSet:
    """Convert the older graded gold files (`*_gold_docs.json` + `*_doc_keys.json`):
    grade 2 (holds the answer) -> gold, grade 1 (supporting) -> supporting."""
    questions: list[TestQuestion] = []
    for set_name, block in gold.items():
        if set_name.startswith("_") or not isinstance(block, dict):
            continue
        for question, grades in block.items():
            g = [doc_keys.get(k, k) for k, v in sorted(grades.items()) if v == 2]
            s = [doc_keys.get(k, k) for k, v in sorted(grades.items()) if v == 1]
            questions.append(TestQuestion(id=f"q{len(questions) + 1:03d}", question=question,
                                          gold_documents=g, supporting_documents=s,
                                          tags=[set_name]))
    return TestSet(name=name, questions=questions, corpus=corpus, description=description)

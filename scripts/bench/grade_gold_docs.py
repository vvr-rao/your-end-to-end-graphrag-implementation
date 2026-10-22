#!/usr/bin/env python3
"""Grade which documents answer which questions, with ONE LLM reading each
ORIGINAL document against every question (PAID, ~$1 for 66 docs x 25 qs).

    uv run python scripts/bench/grade_gold_docs.py --docs source_documents/websearch-geo-time-3x \
        --questions eval_questions/test_questions_websearch_corpus.txt \
        --out eval_questions/websearch3x_gold_docs.json [--model gpt-4.1] [--max-doc-tokens 30000] --go

Writes the retrieval_quality.py gold format: {"_doc_keys", "_grades",
"<question file stem>": {question: {"D01": 2, "D07": 1, ...}}} with
D01..Dnn = the documents in sorted file order (also written to
<out stem minus _gold_docs>_doc_keys.json). The previous gold file is kept
as <out>.bak. Grades: 2 = the document holds (part of) the answer,
1 = relevant background, 0 = omitted.

One call per document, all questions in it: the model judges each
question against the same text, so grades are consistent across
questions and the cost is per document, not per pair. Documents longer
than --max-doc-tokens are cut there (the head carries the substance of
web articles); the cut is reported.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SYSTEM = (
    "You build gold relevance labels for a retrieval benchmark. You get ONE "
    "document and a numbered list of questions. For EACH question decide how "
    "useful this document is for answering it:\n"
    "  2 = the document states facts, figures, analysis or claims that directly "
    "answer the question or a substantial part of it;\n"
    "  1 = the document is on the question's topic and gives useful background "
    "or a minor supporting fact, but does not answer it;\n"
    "  0 = not useful (merely mentioning a keyword is 0).\n"
    "Be strict: most documents are 0 for most questions. Judge only by what the "
    "document says, not by what you know.\n"
    'Return ONE JSON object: {"grades": [{"q": <question number>, "g": 0|1|2}, ...]} '
    "covering every question number."
)

DOC_SUFFIXES = {".pdf", ".pptx", ".docx", ".txt", ".md"}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def read_questions(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


async def main_async(argv: list[str]) -> int:
    from openai import AsyncOpenAI

    from backend.app.core.config import get_settings
    from backend.app.services.document_io import load_document
    from backend.app.services.evaluated_summarizer import get_encoder

    docs_dir, qpath, out = Path(_arg(argv, "--docs")), Path(_arg(argv, "--questions")), Path(_arg(argv, "--out"))
    model = _arg(argv, "--model", "gpt-4.1")
    cap = int(_arg(argv, "--max-doc-tokens", "30000"))
    budget = float(_arg(argv, "--max-cost-usd", "5"))
    price_in, price_out = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6)}.get(model, (2.0, 8.0))
    questions = read_questions(qpath)
    files = sorted(p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    keys = {f"D{i:02d}": p.name for i, p in enumerate(files, 1)}
    enc = get_encoder()
    qlist = "\n".join(f"{n}. {q}" for n, q in enumerate(questions, 1))
    client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    sem = asyncio.Semaphore(8)
    spent = [0.0]
    grades: dict[str, dict[str, int]] = {q: {} for q in questions}
    failed: list[str] = []

    async def one(key: str, path: Path) -> None:
        text = load_document(path, warn_unreadable=False).text
        toks = enc.encode(text)
        if len(toks) > cap:
            print(f"  {key} {path.name[:50]}: {len(toks)} tokens, cut to {cap}")
            text = enc.decode(toks[:cap])
        async with sem:
            if spent[0] > budget:
                failed.append(key)
                return
            for _attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0, response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": f"DOCUMENT:\n{text}\n\nQUESTIONS:\n{qlist}"}])
                    spent[0] += (r.usage.prompt_tokens * price_in + r.usage.completion_tokens * price_out) / 1e6
                    for g in json.loads(r.choices[0].message.content).get("grades", []):
                        q, v = int(g["q"]), int(g["g"])
                        if 1 <= q <= len(questions) and v in (1, 2):
                            grades[questions[q - 1]][key] = v
                    return
                except Exception as exc:  # retry, then record
                    print(f"  {key}: {type(exc).__name__}: {str(exc)[:100]}")
            failed.append(key)

    await asyncio.gather(*[one(f"D{i:02d}", p) for i, p in enumerate(files, 1)])
    if failed:
        print(f"FAILED documents (not graded): {failed}")
        return 1
    if out.exists():
        shutil.copy(out, out.with_suffix(out.suffix + ".bak"))
    keys_path = out.with_name(out.name.replace("_gold_docs.json", "_doc_keys.json"))
    keys_path.write_text(json.dumps(keys, indent=1))
    out.write_text(json.dumps({
        "_doc_keys": f"D01..D{len(files):02d} = files of {docs_dir} in sorted order (see {keys_path.name})",
        "_grades": f"2 = holds the answer; 1 = relevant/supporting. Graded by {model} reading each original "
                   f"document (first {cap} tokens) against every question.",
        qpath.stem: grades,
    }, indent=1))
    for q, g in grades.items():
        print(f"  {sum(v == 2 for v in g.values()):>2} x2 {sum(v == 1 for v in g.values()):>2} x1  {q[:80]}")
    print(f"wrote {out} ({len(questions)} questions, {len(files)} documents), cost ${spent[0]:.3f}")
    return 0


def main(argv: list[str]) -> int:
    if not all(_arg(argv, k) for k in ("--docs", "--questions", "--out")):
        print(__doc__)
        return 2
    if "--go" not in argv:
        print("(dry run) add --go to grade (PAID)")
        return 0
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

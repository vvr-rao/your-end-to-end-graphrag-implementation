"""Recall@K, Precision@K and MRR@K over the documents retrieval returned.

The evidence is a ranked list of chunks. It is collapsed to distinct documents
in rank order (a document counts at the rank of its first chunk); artifacts are
not counted. Gold = the test set's `gold_documents`. Ported from
`scripts/bench/retrieval_quality.py` (`doc_metrics`, `mrr`).

  Recall@K    = gold documents in the top K / all gold documents
  Precision@K = gold documents in the top K / K
  MRR@K       = 1 / rank of the first gold document within the top K, else 0

Passage level, for questions with `answer_strings` (`answer_metrics`):

  answer_recall  = answer strings found in the evidence / strings needed
                   (needed = all of them, or 1 when answer_match is "any")
  answer_found   = 1 when every needed string was found, else 0
  answer_mrr     = 1 / rank of the first evidence chunk holding any string, else 0
  answer_in_reply = the same recall over the generated answer (only with --answer)

Matching is case-insensitive substring after collapsing whitespace; figures
are matched as written ("29,031,560"), as the source documents write them.
"""
from __future__ import annotations

from typing import Any

from backend.app.core.config import get_settings

DEFAULT_K = 10


def configured_k() -> int:
    cfg = get_settings().app_config.get("evals", {}) or {}
    return int(cfg.get("k") or DEFAULT_K)


def ranked_documents(chunk_documents: list[str | None]) -> list[str]:
    """Document names in rank order, each once, skipping unknowns."""
    out: list[str] = []
    seen: set[str] = set()
    for d in chunk_documents:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def retrieval_metrics(ranked: list[str], gold: list[str], k: int) -> dict[str, Any] | None:
    """None when there is no gold to score against."""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return None
    top = ranked[:k]
    hits = [d for d in top if d in gold_set]
    first = next((i for i, d in enumerate(top, 1) if d in gold_set), None)
    return {
        f"recall@{k}": round(len(hits) / len(gold_set), 4),
        f"precision@{k}": round(len(hits) / k, 4),
        f"mrr@{k}": round(1 / first, 4) if first else 0.0,
        "gold_found": hits,
        "gold_missed": sorted(gold_set - set(hits)),
    }


def _norm(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def answer_metrics(evidence_texts: list[str], answer_strings: list[str], *,
                   match: str = "all", reply: str = "") -> dict[str, Any] | None:
    """Passage-level answer check over the ranked evidence texts. None when the
    question has no answer strings."""
    wanted = [a for a in answer_strings if a and a.strip()]
    if not wanted:
        return None
    needed = 1 if match == "any" else len(wanted)
    texts = [_norm(t) for t in evidence_texts]
    found = [a for a in wanted if any(_norm(a) in t for t in texts)]
    first = next((i for i, t in enumerate(texts, 1)
                  if any(_norm(a) in t for a in wanted)), None)
    out: dict[str, Any] = {
        "answer_recall": round(min(len(found), needed) / needed, 4),
        "answer_found": int(len(found) >= needed),
        "answer_mrr": round(1 / first, 4) if first else 0.0,
        "answer_rank": first,
        "answers_found": found,
        "answers_missed": [a for a in wanted if a not in found],
    }
    if reply:
        in_reply = [a for a in wanted if _norm(a) in _norm(reply)]
        out["answer_in_reply"] = round(min(len(in_reply), needed) / needed, 4)
    return out

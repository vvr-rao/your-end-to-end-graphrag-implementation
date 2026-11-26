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

Matching is case-insensitive substring after collapsing whitespace and
folding typography (curly quotes and dashes, accents, non-breaking spaces).
An answer string that is a plain number ("7.7", "84", "6.0%", "$31.4",
"29,031,560") also matches any number in the text that gives it when rounded
to the same decimal places -- read as written, or scaled to thousands /
millions / billions / trillions ("$7,709,020,032" and "7.71 billion" both
match "7.7"; "83.82%" matches "84"). Nothing here depends on a dataset.
"""
from __future__ import annotations

import re
import unicodedata
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


_TYPO = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
                       "\u2032": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
                       "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
                       "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u202f": " ",
                       "\u2009": " "})


def _norm(text: str) -> str:
    folded = unicodedata.normalize("NFKD", (text or "").translate(_TYPO))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(folded.split()).casefold()


# A plain number, optionally with a currency sign and/or percent sign.
_EXPECTED_NUMBER = re.compile(r"\s*(?:us)?[$\u20ac\u00a3\u00a5]?\s*(-?\d[\d,]*(?:\.\d+)?)\s*%?\s*", re.I)
# Numbers in running text: grouped thousands or plain digits, optional scale word.
_TEXT_NUMBER = re.compile(
    r"(?<![\w.])(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(trillion|tn|billion|bn|million|mn|mln|thousand|k)\b)?", re.I)
_SCALE = {"trillion": 1e12, "tn": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6,
          "mn": 1e6, "mln": 1e6, "thousand": 1e3, "k": 1e3}


def _expected_number(answer: str) -> tuple[float, int] | None:
    """(value, decimal places) when the answer string is a plain number."""
    m = _EXPECTED_NUMBER.fullmatch(answer or "")
    if not m:
        return None
    raw = m.group(1)
    if "," in raw and not re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        return None
    raw = raw.replace(",", "")
    return float(raw), len(raw.split(".")[1]) if "." in raw else 0


def _number_in(answer: str, text: str) -> bool:
    """True when some number in `text`, read as written or scaled to a larger
    unit, rounds to the expected number at its own decimal places."""
    exp = _expected_number(answer)
    if exp is None:
        return False
    want, places = exp
    for m in _TEXT_NUMBER.finditer(text or ""):
        value = float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1.0)
        for unit in (1.0, 1e3, 1e6, 1e9, 1e12):
            if unit > 1 and abs(value) < unit:
                break
            if round(value / unit, places) == round(want, places):
                return True
    return False


def _contains(answer: str, normed_text: str) -> bool:
    if _expected_number(answer) is not None:     # whole numbers only: "31.4" is not in "131.4"
        return _number_in(answer, normed_text)
    return _norm(answer) in normed_text


def answer_metrics(evidence_texts: list[str], answer_strings: list[str], *,
                   match: str = "all", reply: str = "") -> dict[str, Any] | None:
    """Passage-level answer check over the ranked evidence texts. None when the
    question has no answer strings."""
    wanted = [a for a in answer_strings if a and a.strip()]
    if not wanted:
        return None
    needed = 1 if match == "any" else len(wanted)
    texts = [_norm(t) for t in evidence_texts]
    found = [a for a in wanted if any(_contains(a, t) for t in texts)]
    first = next((i for i, t in enumerate(texts, 1)
                  if any(_contains(a, t) for a in wanted)), None)
    out: dict[str, Any] = {
        "answer_recall": round(min(len(found), needed) / needed, 4),
        "answer_found": int(len(found) >= needed),
        "answer_mrr": round(1 / first, 4) if first else 0.0,
        "answer_rank": first,
        "answers_found": found,
        "answers_missed": [a for a in wanted if a not in found],
    }
    if reply:
        normed_reply = _norm(reply)
        in_reply = [a for a in wanted if _contains(a, normed_reply)]
        out["answer_in_reply"] = round(min(len(in_reply), needed) / needed, 4)
    return out

"""Answer-string matching in the question eval: typography folding and
number-aware matching. Generic fixtures only."""

from __future__ import annotations

import pytest

from backend.app.evals.retrieval_metrics import _contains, _norm, answer_metrics


@pytest.mark.parametrize("answer, text", [
    ("7.7", "India imported US$7,709,020,032.69 in 2024"),
    ("2.56", "Thailand: $2,561,055,865.11"),
    ("71.8", "about $71.81 billion"),
    ("85.3", "85,334,419,380"),
    ("84", "together 83.82% of imports"),
    ("6.0", "growth of 6% in 2025"),
    ("4.0", "Nigeria | 4% |"),
    ("31.4", "approximately US$31.45 billion"),
    ("$31.4", "31.4 bn"),
    ("29,031,560", "a total of 29,031,560 units"),
    ("200", "about 200 days"),
])
def test_numbers_match_in_any_written_form(answer, text):
    assert _contains(answer, _norm(text))


@pytest.mark.parametrize("answer, text", [
    ("7.7", "7.8 billion"),
    ("84", "82.4%"),
    ("200", "2,000 days"),
    ("6.0", "16% growth"),
    ("4.0", "2024"),
    ("31.4", "131.4"),
])
def test_numbers_do_not_match_other_values(answer, text):
    assert not _contains(answer, _norm(text))


def test_typography_is_folded():
    assert _contains("C\u00f4te d'Ivoire", _norm("Cote d\u2019Ivoire grew"))
    assert _contains("high-risk", _norm("the high\u2013risk zone"))


def test_text_strings_still_need_the_words():
    assert not _contains("Saudi Arabia", _norm("the United Arab Emirates"))


def test_reply_recall_uses_number_matching():
    out = answer_metrics([], ["India", "7.7"], reply="India: $7,709,020,032")
    assert out["answer_in_reply"] == 1.0

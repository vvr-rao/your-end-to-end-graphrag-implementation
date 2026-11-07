"""Places are read from the question text, because the parse often misses them."""
from __future__ import annotations

import pytest

from backend.app.services.question_places import places_in_question

PLACES = [
    ("asia", "Asia", []),
    ("eu", "Europe", []),
    ("east", "Eastern Europe", []),
    ("us", "United States", ["U.S.", "US", "USA"]),
    ("sk", "South Korea", ["Korea"]),
    ("rk", "Republic of Korea", ["Korea"]),
    ("de", "Germany", []),
    ("jp", "Japan", []),
    ("in", "India", ["Indian"]),
]


@pytest.mark.parametrize("question,expected", [
    ("Which companies are in Asia?", ["asia"]),
    ("What revenue did companies report in Europe?", ["eu"]),
    ("Compare the financials of companies in consumer electronics in Asia vs the US",
     ["asia", "us"]),
    ("How do Asian manufacturers compare with US manufacturers?", ["asia", "us"]),
    ("What did European automakers report?", ["eu"]),
    ("What are German automakers doing?", ["de"]),
    ("what is planned in asia?", ["asia"]),
    ("Asia is growing, why?", ["asia"]),
])
def test_finds_places_the_parse_can_miss(question, expected):
    assert sorted(places_in_question(question, PLACES)) == sorted(expected)


def test_the_word_us_is_not_the_united_states():
    assert places_in_question("Tell us about revenue", PLACES) == []


def test_the_longest_name_wins():
    assert places_in_question("Investment in Eastern Europe", PLACES) == ["east"]


def test_an_alias_shared_by_two_entities_finds_both():
    assert sorted(places_in_question("Companies in Korea", PLACES)) == ["rk", "sk"]


@pytest.mark.parametrize("question", [
    "What did Forbes Asia report?",
    "Who runs LG Energy Solution Japan?",
    "What does the Asian Development Bank do?",
    "Is Japan Exchange Group listed?",
])
def test_a_place_inside_another_name_is_not_a_place(question):
    assert places_in_question(question, PLACES) == []


def test_a_word_containing_a_name_does_not_match():
    assert places_in_question("Is Asiatic or Japanned stock up?", PLACES) == []


def test_no_places_means_nothing():
    assert places_in_question("What was operating profit?", []) == []

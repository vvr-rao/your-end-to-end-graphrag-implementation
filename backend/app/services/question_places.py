"""Find the places a question names, with rules rather than the question parse.

The parse often does not report a region as an entity: "Which companies are in
Asia?" returned `Asia` as a class term and "What revenue did companies report in
Europe?" did not list it at all (measured 2026-11-04, 4 of 7 region questions), so
the expansion never ran. Time terms avoid this by being read from the question
text with the function ingestion uses; this does the same for places.

Only names and aliases of entities are looked for, so a question cannot name a
place the graph does not have.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# Adjectives that are not the name plus -n or -an (Asian, European, Korean).
_IRREGULAR: dict[str, tuple[str, ...]] = {
    "china": ("Chinese",),
    "germany": ("German",),
    "japan": ("Japanese",),
    "france": ("French",),
    "spain": ("Spanish",),
    "united kingdom": ("British",),
    "united states": ("American",),
    "vietnam": ("Vietnamese",),
    "taiwan": ("Taiwanese",),
    "portugal": ("Portuguese",),
    "hungary": ("Hungarian",),
    "poland": ("Polish",),
    "netherlands": ("Dutch",),
    "sweden": ("Swedish",),
    "denmark": ("Danish",),
    "finland": ("Finnish",),
    "norway": ("Norwegian",),
    "switzerland": ("Swiss",),
    "ireland": ("Irish",),
    "turkey": ("Turkish",),
    "thailand": ("Thai",),
}
_WORD = r"[^\W_]"


def _forms(name: str) -> list[str]:
    """A name and its adjectives. Only a name ending in a vowel takes a suffix:
    `Japann` is not a word, and a consonant-ending name is in `_IRREGULAR`."""
    forms = [name]
    lowered = name.lower()
    if lowered.endswith("a"):
        forms.append(name + "n")
    elif lowered.endswith("e"):
        forms.append(name + "an")
    forms.extend(_IRREGULAR.get(lowered, ()))
    return forms


def _exact_case(form: str) -> bool:
    """`US` must not match the word "us"."""
    return len(form) <= 3 or form.isupper()


def _is_capitalized(word: str) -> bool:
    return bool(word) and word[0].isupper()


def _part_of_a_longer_name(question: str, start: int, end: int) -> bool:
    """`Forbes Asia`, `LG Energy Solution Japan`, `Japan Exchange Group`: a
    capitalized neighbour means the place is inside some other name. The first
    word of the question is capitalized anyway, so it does not count."""
    before = question[:start].rstrip()
    if before and before[-1].isalnum() and before == question[:start - 1]:
        words = before.split()
        if len(words) > 1 and _is_capitalized(words[-1]):
            return True
    after = question[end:]
    if after.startswith(" "):
        words = after.split()
        if words and _is_capitalized(words[0]) and words[0].lower() not in {"vs", "vs."}:
            return True
    return False


def places_in_question(
    question: str, places: Iterable[tuple[str, str, Iterable[str]]]
) -> list[str]:
    """Ids of the `(id, name, aliases)` places the question names, longest name
    first: "Eastern Europe" is not also a question about Europe."""
    forms: list[tuple[str, str]] = []
    for pid, name, aliases in places:
        for label in (name, *aliases):
            if label and label.strip():
                forms.extend((f, pid) for f in _forms(label.strip()))
    forms.sort(key=lambda item: -len(item[0]))

    found: list[str] = []
    taken: list[tuple[int, int]] = []
    for form, pid in forms:
        flags = 0 if _exact_case(form) else re.IGNORECASE
        pattern = rf"(?<!{_WORD}){re.escape(form)}(?!{_WORD})"
        for match in re.finditer(pattern, question, flags):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                # The same words under another entity (two share `Korea`) count;
                # a shorter name inside a longer one does not.
                if (start, end) not in taken:
                    continue
            elif _part_of_a_longer_name(question, start, end):
                continue
            taken.append((start, end))
            if pid not in found:
                found.append(pid)
    return found

"""Deterministic synonym mining from the corpus. No LLM, no embeddings, $0.

Motivation
----------
A query asking about "tirzepatide" could not reach chunks that only ever
say "MOUNJARO" -- the two words are unrelated in embedding space, so the
right chunk sat at rank #886 of 2,498 and the synthesis answered with a
*different drug's* dose. FDA labels use brand and generic names
interchangeably by design, so this hits every drug in the corpus.

The equivalence is not in the ontology, not in the graph (there is no
`sameAs` predicate), and emphatically not in the embeddings. But it IS
in the text: labels state the pairing themselves, in a fixed convention.

    MOUNJARO (tirzepatide) injection, for subcutaneous use

So we mine it, rather than asking a model what Mounjaro is. Three
sources, in descending yield:

  1. `parenthetical` -- apposition, both directions, plus the classic
     acronym rule (`World Health Organization (WHO)`).
  2. `phrase`        -- "also known as", "marketed as", "sold under the
     brand name", "formerly known as".
  3. `ontology`      -- `skos:altLabel` / `oboInOwl:hasExactSynonym`
     annotations, which the ontology importer already parked in
     `ontology_classes.extra_metadata->'annotations'` and nothing has
     ever read.

The cost of determinism is recall: a pair the corpus never appositions
is not found, and those queries behave exactly as they do today. That
gap is *reported* by `mine-aliases`, not hidden.

Precision is the priority throughout -- a wrong pair produces exactly
the failure this whole change exists to fix, just pointed at a different
drug. Every guard below is there to reject rather than to reach, and
`mine-aliases --dry-run` prints the pairs so they can be eyeballed
before anything is written.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.graph_version import current_version
from backend.app.db.session import session_scope

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

# MUST stay semantically identical to `db_entity_extract._normalize_name`.
# Duplicated rather than imported so this module stays dependency-light (it
# is pulled in by the retrieval path); `test_alias_mining.py` asserts the two
# agree, so divergence fails the suite rather than silently splitting the
# vocabulary between ingest and query time.
def normalize_term(name: str) -> str:
    """Lowercase + strip non-alphanumerics (except internal spaces)."""
    s = re.sub(r"[^\w\s]", "", name, flags=re.UNICODE).strip().lower()
    return re.sub(r"\s+", " ", s)


def _strip_marks(s: str) -> str:
    """Drop trademark/registered marks and collapse whitespace.

    Whitespace collapsing matters for the surface form specifically: PDF
    text wraps mid-term, and "beats per\\nminute" would otherwise be
    stored (and embedded) with the newline in it.
    """
    return re.sub(r"\s+", " ", re.sub(r"[®™©]", "", s)).strip()


# --------------------------------------------------------------------------
# Precision guards
# --------------------------------------------------------------------------

# A parenthetical or alias candidate that starts with one of these is a
# clause, a cross-reference, or a hedge -- not a name.
_LEAD_STOPWORDS: frozenset[str] = frozenset({
    "see", "eg", "ie", "including", "include", "includes", "such",
    "and", "or", "but", "if", "when", "where", "which", "that", "who",
    "from", "to", "per", "up", "at", "in", "on", "of", "for", "with",
    "approximately", "about", "based", "defined", "measured", "adjusted",
    "table", "figure", "fig", "section", "page", "appendix", "exhibit",
    "continued", "cont", "all", "each", "any", "both", "either",
    "total", "mean", "median", "range", "versus", "vs", "compared",
    "not", "no", "yes", "excluding", "except", "other", "others",
})

# Bare unit / measure parentheticals. The digit guard kills "(95% CI)" and
# "(n=234)"; these are the ones that carry no digits at all.
_UNIT_TOKENS: frozenset[str] = frozenset({
    "mg", "ml", "mcg", "kg", "g", "l", "iu", "u", "mmol", "mol", "meq",
    "mmhg", "kgm", "bmi", "sd", "se", "sem", "ci", "iqr", "na", "nd",
    "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "cny",
    "millions", "million", "billions", "billion", "thousands", "thousand",
    "percent", "pct", "each", "week", "weekly", "daily", "monthly",
    "yearly", "annual", "annually", "hour", "hours", "day", "days",
    "month", "months", "year", "years", "unaudited", "audited",
    "restated", "revised", "pro forma", "as adjusted", "loss", "net",
})

# Words too generic to be half of a synonym pair. Prevents
# "Eli Lilly and Company (Lilly)" from minting `company <-> lilly` once the
# capitalized-run rule stops at the lowercase "and".
#
# The second block was added after a --dry-run over the live corpus surfaced
# them as high-frequency false positives -- mostly clinical-table headers
# and label boilerplate sitting next to a parenthesis.
_GENERIC_TERMS: frozenset[str] = frozenset({
    "company", "companies", "corporation", "corp", "inc", "incorporated",
    "ltd", "limited", "plc", "group", "holdings", "holding", "llc", "llp",
    "sa", "ag", "gmbh", "nv", "bv", "spa", "se", "asa", "ab", "oy", "kg",
    "study", "studies", "trial", "trials", "patients", "patient",
    "subjects", "subject", "treatment", "treatments", "dose", "doses",
    "dosage", "placebo", "control", "controls", "group", "groups",
    "table", "figure", "section", "product", "products", "drug", "drugs",
    "the", "this", "that", "these", "those", "it", "its",
    # Observed in the live dry run:
    "source", "sources", "item", "code", "codes", "baseline", "change",
    "changes", "incidence", "discontinuation", "arm", "arms", "labeling",
    "label", "labels", "pen", "rate", "rates", "mean", "means", "median",
    "blockers", "blocker", "inhibitors", "inhibitor", "agonists", "agonist",
    "rotate", "administration", "description", "indication", "indications",
    "information", "instructions", "reference", "references", "column",
    "row", "value", "values", "level", "levels", "week", "weeks",
    "prescription drug", "patient information", "adverse reactions",
    # Drug-label boilerplate that prefixes a brand name on the carton /
    # "how supplied" panels: "Rx Only NDC Zepbound", "HANDLING How
    # Supplied TRULICITY", "brand name Norvasc".
    "rx", "ndc", "only", "use", "uses", "used", "seal", "handling",
    "supplied", "how", "brand", "trade", "generic", "name", "names",
    "tablets", "tablet", "injection", "injections", "capsule", "capsules",
    "vial", "vials", "carton", "storage", "store", "keep", "read",
    "medication", "guide", "usage", "supplied", "dispense", "pharmacist",
    # Dosage FORMS are not synonyms of the drug: "Wegovy (pill)" describes
    # a presentation, and admitting it lets "pill" ride into probe text as
    # though it named the product.
    "pill", "pills", "shot", "shots", "syringe", "needle", "prefilled",
    "solution", "suspension", "powder", "oral", "subcutaneous",
    # Delivery DEVICES are not the drug either. "MOUNJARO KwikPen" is a
    # presentation of tirzepatide, not another name for it, and admitting
    # it burns an alias slot that a real brand name should hold.
    "kwikpen", "flexpen", "flextouch", "autoinjector", "auto-injector",
    "singledose", "multidose", "cartridge", "ampoule", "ampule",
})

# Terms whose SHAPE marks them as not-a-name, checked before the word lists.
#   - a slash: units ("mg/dL", "mL/min"), never a synonym
#   - a long lowercase run glued to an uppercase run: URL fragments
#     scraped into the text ("drugsatfdaCOZAAR", "drugsatfdaPRINIVIL").
#     The 6-char minimum on the lowercase side keeps real mixed-case
#     terms ("mTOR", "mITT", "pH") safe.
# Route / formulation modifiers that legitimately PRECEDE a generic name:
# "Rybelsus (oral semaglutide)". These are stripped from the front of a
# candidate term before validation, so the real name underneath is judged
# on its own merits.
#
# Deliberately much narrower than "strip any leading generic word" -- that
# broader rule would resurrect junk like "Product Characteristics Color
# WHITE" by peeling "Product" off the front. Only route/form words qualify.
_FORM_MODIFIERS: frozenset[str] = frozenset({
    "oral", "orally", "injectable", "injected", "subcutaneous",
    "intravenous", "intramuscular", "topical", "inhaled", "nasal",
    "sublingual", "transdermal", "chewable", "effervescent",
    "extended-release", "immediate-release", "delayed-release",
    "sustained-release", "controlled-release", "long-acting",
    "short-acting", "rapid-acting", "micronized", "generic", "recombinant",
})


def _strip_form_modifiers(s: str) -> str:
    """Peel leading route/formulation words off a candidate term.

    "oral semaglutide" -> "semaglutide". Returns the input unchanged when
    nothing strips, or when stripping would leave nothing behind.
    """
    parts = s.split()
    i = 0
    while i < len(parts) - 1 and parts[i].lower().strip(",") in _FORM_MODIFIERS:
        i += 1
    return " ".join(parts[i:]) if i else s


# An FDA label pronunciation respelling: "TRULICITY (TRU-li-si-tee)",
# "OZEMPIC (oh-ZEM-pick)", "amlodipine (am loe' di peen)". These are
# notation, not synonyms, and they were riding into live probe expansions.
#
# The discriminator is that respelling syllables are ALL SHORT. Trigram
# similarity does not work here -- a respelling is often paired with the
# GENERIC name, so "oh-ZEM-pick"/"semaglutide" scores 0.000. The <=5 rule
# is what protects real hyphenated names: "PEG-loxenatide" (10) and
# "non-HDL cholesterol" (11) both survive, where a shape-only test kills them.
_MAX_RESPELLING_SYLLABLE = 5


def _is_pronunciation_respelling(s: str) -> bool:
    """True for FDA-style phonetic respellings."""
    parts = [p for p in re.split(r"[\s\-]+", s.strip()) if p]
    if len(parts) < 2:
        return False
    if max(len(p) for p in parts) > _MAX_RESPELLING_SYLLABLE:
        return False
    # An apostrophe stress mark ("loe'") is conclusive on its own.
    if any(p.endswith("'") for p in parts):
        return True
    # Otherwise: the tell-tale ALL-CAPS / all-lowercase syllable alternation.
    has_upper = any(p.isupper() and len(p) >= 2 for p in parts)
    has_lower = any(p.islower() and len(p) >= 2 for p in parts)
    return has_upper and has_lower


def _strip_inline_respelling(s: str) -> str:
    """Drop a respelling that trails the name it respells.

    Label headers run the two together -- "TRULICITY TRU-li-si-tee
    (dulaglutide)" -- so the head comes back as both words and the pair
    stores a surface no query will ever match. Removing the respelling
    recovers the CORRECT pair (`TRULICITY <-> dulaglutide`) rather than
    discarding the mention.

    Only strips when another token in the term shares the respelling's
    first two letters, which is what makes it a respelling *of that name*.
    That guard is why "non-HDL cholesterol" survives intact: "non-HDL" is
    respelling-shaped, but "cholesterol" does not start with "no".
    """
    parts = s.split()
    if len(parts) < 2:
        return s
    keep: list[str] = []
    for i, tok in enumerate(parts):
        if _is_pronunciation_respelling(tok):
            head2 = normalize_term(tok)[:2]
            others = [p for j, p in enumerate(parts) if j != i]
            if head2 and any(normalize_term(o).startswith(head2) for o in others):
                continue
        keep.append(tok)
    return " ".join(keep) if keep else s


_UNIT_SLASH_RE = re.compile(r"/")
# Applied per whitespace/hyphen-separated token, not across the whole
# string: "lossMounjaro" and "drugsatfdaCOZAAR" are single glued tokens,
# whereas a hyphenated pronunciation guide ("mown-JAHR-OH") is not one and
# must survive.
_GLUED_TOKEN_RE = re.compile(r"^[a-z]{4,}[A-Z]")

_MIN_TERM_CHARS = 3
_MAX_TERM_WORDS = 4
# An acronym EXPANSION is legitimately longer than an ordinary name --
# "earnings before interest and taxes", "Securities and Exchange
# Commission" -- so the word cap is relaxed when the other side of the
# pair is an acronym vouching for it.
_MAX_ACRONYM_EXPANSION_WORDS = 6
_MAX_TERM_CHARS = 60

# Words an acronym conventionally skips. "Securities and Exchange
# Commission" spells SEC, not SAEC; "earnings before interest and taxes"
# spells EBIT, not EBIAT. Without this the initials rule rejects most
# real-world acronyms outside the tidy three-capitalised-words case.
_ACRONYM_SKIP_WORDS: frozenset[str] = frozenset({
    "and", "of", "the", "for", "in", "on", "to", "a", "an", "at",
    "by", "with", "from", "or", "de", "la", "le",
})

_WORD_RE = re.compile(r"[A-Za-z][\w\-']*")


def is_acronym(s: str) -> bool:
    """All-caps token of 2-6 letters, e.g. WHO, FDA, GLP."""
    return bool(re.fullmatch(r"[A-Z]{2,6}", s))


def _has_upper(s: str) -> bool:
    return any(c.isupper() for c in s)


def is_valid_term(s: str, *, max_words: int = _MAX_TERM_WORDS) -> bool:
    """Term-shape guard, applied to BOTH sides of every candidate pair.

    Rejects clauses, cross-references, numbers, units and generic nouns.
    Deliberately strict: a false pair is worse than a missed one.

    `max_words` is relaxed for acronym expansions by `is_valid_pair`.
    """
    s = _strip_marks(s)
    if not s:
        return False
    if len(s) > _MAX_TERM_CHARS:
        return False
    # A phonetic respelling is notation, not a synonym.
    if _is_pronunciation_respelling(s):
        return False
    # Numbers anywhere: "(2024)", "(n=234)", "(95% CI)", "(NCT01234)".
    if any(c.isdigit() for c in s):
        return False
    # Commas / terminal punctuation mean a clause, not a term.
    if re.search(r"[,;:.!?]", s):
        return False
    # Units ("mg/dL", "mL/min") and scraped/glued fragments
    # ("drugsatfdaCOZAAR", "lossMounjaro") -- both from the live dry run.
    if _UNIT_SLASH_RE.search(s):
        return False
    if any(_GLUED_TOKEN_RE.match(tok) for tok in re.split(r"[\s\-]+", s)):
        return False
    words = _WORD_RE.findall(s)
    if not words or len(words) > max_words:
        return False
    # Must be mostly letters -- rejects "%", "+/-", stray symbols.
    if not re.fullmatch(r"[A-Za-z\s\-'&]+", s):
        return False
    norm = normalize_term(s)
    if len(norm.replace(" ", "")) < _MIN_TERM_CHARS:
        return False
    # Lead-stopword rejection kills clauses ("who received placebo",
    # "see Table 3"). A bare acronym is exempt, or "WHO" would be thrown
    # out for colliding with the relative pronoun.
    if words[0].lower() in _LEAD_STOPWORDS and not (
        len(words) == 1 and is_acronym(_strip_marks(s))
    ):
        return False
    if norm in _UNIT_TOKENS or norm in _GENERIC_TERMS:
        return False
    # Every word generic ("the company", "study group") -> not a name.
    if all(w.lower() in _GENERIC_TERMS or w.lower() in _UNIT_TOKENS for w in words):
        return False
    # A name does not START with a generic noun. This is what separates
    # "Mounjaro" from the table-header run "Dose NDC Mounjaro" that the
    # capitalized-run head rule otherwise picks up in clinical tables.
    if len(words) > 1 and words[0].lower() in _GENERIC_TERMS:
        return False
    # A repeated adjacent word is a PDF-scrape artifact, never a name:
    # "TRULICITY TRULICITY", "SEAL SEAL TRULICITY", "Ozempic Ozempic".
    lowered = [w.lower() for w in words]
    if any(a == b for a, b in zip(lowered, lowered[1:])):
        return False
    return True


def _acronym_matches(head: str, acronym: str) -> bool:
    """Does `head` expand to `acronym`? Three accepted conventions.

      initials          World Health Organization  -> WHO
      skipping stopwords Securities and Exchange Commission -> SEC
                         earnings before interest and taxes -> EBIT
      prefix            BHP Group Limited -> BHP

    The prefix form matters for company names, where the short form is
    the leading token rather than a set of initials.
    """
    words = _WORD_RE.findall(head)
    if not words:
        return False
    target = acronym.upper()
    if "".join(w[0].upper() for w in words) == target:
        return True
    kept = [w for w in words if w.lower() not in _ACRONYM_SKIP_WORDS]
    if kept and "".join(w[0].upper() for w in kept) == target:
        return True
    # Prefix form: only when the head is more than its first word, or the
    # "pair" is just one name against itself.
    if len(words) > 1 and words[0].upper() == target:
        return True
    return False


def _acronym_applies(other: str, acronym: str) -> bool:
    """Whether the initials rule should be enforced for this pair at all.

    Only when the other side is a multi-word phrase that could plausibly
    expand to the acronym -- i.e. it has at least as many words as the
    acronym has letters, once stopwords are dropped, OR it leads with the
    acronym itself.

    Without this gate an all-caps BRAND of 2-6 letters looks like an
    acronym and gets rejected against its own generic: `COZAAR (losartan
    potassium)` would demand that "losartan potassium" spell C-O-Z-A-A-R.
    """
    words = _WORD_RE.findall(other)
    if len(words) < 2:
        return False
    if words[0].upper() == acronym.upper():
        return True
    kept = [w for w in words if w.lower() not in _ACRONYM_SKIP_WORDS]
    return len(kept) == len(acronym)


def is_valid_pair(left: str, right: str) -> bool:
    """Both sides valid, distinct, and at least one is name-like.

    The name-like requirement is what rejects prose apposition such as
    "administered in the morning (once daily)" -- two lowercase common
    phrases that happen to sit next to a parenthesis.
    """
    # An acronym on one side vouches for a longer phrase on the other.
    left_is_acr = is_acronym(_strip_marks(left))
    right_is_acr = is_acronym(_strip_marks(right))
    left_max = _MAX_ACRONYM_EXPANSION_WORDS if right_is_acr else _MAX_TERM_WORDS
    right_max = _MAX_ACRONYM_EXPANSION_WORDS if left_is_acr else _MAX_TERM_WORDS
    if not (
        is_valid_term(left, max_words=left_max)
        and is_valid_term(right, max_words=right_max)
    ):
        return False
    ln, rn = normalize_term(left), normalize_term(right)
    if ln == rn or not ln or not rn:
        return False
    # One side must carry a capital (a proper name / brand / acronym),
    # OR the pair is an acronym expansion.
    if not (_has_upper(_strip_marks(left)) or _has_upper(_strip_marks(right))):
        return False
    right_bare = _strip_marks(right)
    left_bare = _strip_marks(left)
    # A 3+ letter acronym cannot expand to a SINGLE word. "MRHD (monkey)"
    # and "MRHD (rabbit)" are species qualifiers on a dose, and they were
    # being minted as expansions because `_acronym_applies` returns False
    # for a one-word other and so skipped the initials check entirely.
    # "ACE (kininase II)" is two words and unaffected.
    for acr, other in ((left_bare, right_bare), (right_bare, left_bare)):
        if not is_acronym(acr):
            continue
        # A 3+ letter acronym cannot expand to a SINGLE word. "MRHD
        # (monkey)" and "MRHD (rabbit)" are species qualifiers on a dose,
        # and they were being minted as expansions because
        # `_acronym_applies` returns False for a one-word other and so
        # skipped the initials check entirely. "ACE (kininase II)" is two
        # words and unaffected.
        if len(acr) >= 3 and len(_WORD_RE.findall(other)) < 2:
            return False
        # An expansion may LEAD with its short form -- "BHP Group Limited
        # (BHP)" -- but repeating it later means the "expansion" is really
        # a description of the word itself: "WHITE (White to off white)",
        # "BLUE (Light blue to blue)". Those slip past the initials rule
        # because their word counts happen not to match the acronym length.
        other_words = normalize_term(other).split()
        acr_norm = normalize_term(acr)
        if any(w == acr_norm for w in other_words[1:]):
            return False
    if (
        is_acronym(right_bare)
        and _acronym_applies(left_bare, right_bare)
        and not _acronym_matches(left_bare, right_bare)
    ):
        return False
    if (
        is_acronym(left_bare)
        and _acronym_applies(right_bare, left_bare)
        and not _acronym_matches(right_bare, left_bare)
    ):
        return False
    return True


# --------------------------------------------------------------------------
# Source 1: parenthetical apposition
# --------------------------------------------------------------------------

_PAREN_RE = re.compile(r"\(\s*([^()]{2,60}?)\s*\)")

# Copulas / connectives that sit between a name and an alias phrase
# ("semaglutide IS marketed as Ozempic"). Dropped from the trailing edge
# before the head is read, or the head comes back as "is".
_CONNECTIVES: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "also", "and", "or", "then", "thus", "now", "which", "that",
    "commonly", "widely", "generally", "sometimes", "often",
})


# A separator that means two adjacent capitalised words belong to DIFFERENT
# items, even though the word tokenizer cannot see it:
#   digits      list/section markers -- "Ozempic 3.1 Mounjaro (tirzepatide)"
#               merged into the head "Ozempic Mounjaro"
#   . + space   sentence or list end -- "in Wegovy 8. Zepbound (Tirzepatide)"
#   newline/pipe/bullet  table cells and list items
# A bare period is NOT a boundary, or "Alphabet Inc." and "U.S." would split.
_ITEM_BOUNDARY_RE = re.compile(r"\d|\.\s|[\n|•;]")


def _clean_tail_tokens(prefix: str) -> list[str]:
    """Trailing word tokens of `prefix`, cut at the last item boundary.

    Without this the capitalised-run rule reads straight through a
    numbered list and invents a name from two neighbouring entries.
    """
    matches = list(_WORD_RE.finditer(prefix))
    if not matches:
        return []
    start_idx = 0
    for i in range(1, len(matches)):
        gap = prefix[matches[i - 1].end() : matches[i].start()]
        if _ITEM_BOUNDARY_RE.search(gap):
            start_idx = i
    return [m.group(0) for m in matches[start_idx:]]


def _head_before(text: str, end: int, *, n_words: int | None = None) -> str:
    """The name immediately preceding position `end`.

    Three shapes have to work:

      "The maximum dosage of MOUNJARO (tirzepatide)"  -> "MOUNJARO"
      "dosage of tirzepatide (Mounjaro)"              -> "tirzepatide"
      "semaglutide is marketed as Ozempic"            -> "semaglutide"

    Collect the preceding word tokens, drop trailing connectives, then:
    if the trailing run is capitalized take the whole run (this is what
    stops at the lowercase "of"); otherwise take just the last token.

    `n_words` forces an exact-length head, used by the acronym rule where
    an N-letter acronym must be matched against exactly N words.
    """
    # NOTE: read boundaries from the RAW slice, not `_strip_marks` output --
    # that collapses whitespace and would turn ". " into "." , hiding a
    # sentence boundary.
    prefix = text[:end]
    tokens = _clean_tail_tokens(prefix)
    while tokens and tokens[-1].lower() in _CONNECTIVES:
        tokens.pop()
    if not tokens:
        return ""
    if n_words is not None:
        if len(tokens) < n_words:
            return ""
        return " ".join(tokens[-n_words:])
    tail = tokens[-_MAX_TERM_WORDS:]
    # Longest trailing run of capitalized/all-caps tokens.
    run: list[str] = []
    for tok in reversed(tail):
        if tok[:1].isupper():
            run.insert(0, tok)
        else:
            break
    if run:
        return " ".join(run)
    return tokens[-1]


def _head_after(text: str, start: int) -> str:
    """Mirror of `_head_before` for the text FOLLOWING position `start`.

      "... marketed as Ozempic for type 2 diabetes"  -> "Ozempic"
      "... also known as semaglutide, a GLP-1"       -> "semaglutide"
    """
    suffix = _strip_marks(text[start:])
    tokens = _WORD_RE.findall(suffix)[:_MAX_TERM_WORDS]
    if not tokens:
        return ""
    run: list[str] = []
    for tok in tokens:
        if tok[:1].isupper():
            run.append(tok)
        else:
            break
    if run:
        return " ".join(run)
    return tokens[0]


def extract_parenthetical_pairs(text: str) -> list[tuple[str, str]]:
    """`NAME (other-name)` in either direction. Returns surface-form pairs."""
    out: list[tuple[str, str]] = []
    for m in _PAREN_RE.finditer(text):
        inner = _strip_marks(m.group(1))
        # "Rybelsus (oral semaglutide)" -- the parenthetical is a descriptor
        # plus the generic name. Peel the route/form word so the real name
        # underneath is judged (and stored) on its own.
        inner = _strip_form_modifiers(inner)
        inner = _strip_inline_respelling(inner)
        if not inner:
            continue
        if is_acronym(inner):
            # An N-letter acronym expands from AT LEAST N words, and more
            # when it skips stopwords ("Securities and Exchange Commission"
            # -> SEC needs 4). Try increasing head lengths and take the
            # first that actually spells the acronym; a greedy capitalized
            # run instead would trip on the leading "The".
            head = ""
            for n in range(len(inner), _MAX_ACRONYM_EXPANSION_WORDS + 1):
                cand = _head_before(text, m.start(), n_words=n)
                if cand and _acronym_matches(cand, inner):
                    head = cand
                    break
        else:
            head = _head_before(text, m.start())
        if not head:
            continue
        head = _strip_inline_respelling(head)
        if not head:
            continue
        if is_valid_pair(head, inner):
            out.append((head, inner))
    return out


# --------------------------------------------------------------------------
# Source 2: explicit alias phrases
# --------------------------------------------------------------------------

_ALIAS_PHRASES = (
    r"also known as",
    r"also called",
    r"also referred to as",
    r"otherwise known as",
    r"marketed as",
    r"marketed under the (?:brand |trade )?names?",
    r"sold under the (?:brand |trade )?names?",
    r"known commercially as",
    r"formerly known as",
    r"previously known as",
    r"generic name:?",
    r"brand name:?",
)

# Only the connective phrase is matched; the two names are then read off
# either side with the same head logic the parenthetical source uses.
# Capturing the names inside the regex instead made it greedy -- "is
# marketed as Ozempic for type 2 diabetes" yielded the pair
# ("semaglutide is", "Ozempic for type").
_PHRASE_RE = re.compile(
    r"[,(]?\s*(?:" + "|".join(_ALIAS_PHRASES) + r")\s+", re.IGNORECASE
)


def extract_phrase_pairs(text: str) -> list[tuple[str, str]]:
    """"X, also known as Y" and friends. Returns surface-form pairs."""
    out: list[tuple[str, str]] = []
    for m in _PHRASE_RE.finditer(text):
        a = _head_before(text, m.start())
        b = _head_after(text, m.end())
        if not a or not b:
            continue
        if is_valid_pair(a, b):
            out.append((a, b))
    return out


# --------------------------------------------------------------------------
# Source 3: ontology synonym annotations (already in the DB, never read)
# --------------------------------------------------------------------------

# Annotation predicates the ontology parser preserves verbatim into
# `ontology_classes.extra_metadata->'annotations'`, keyed by full URI.
_SYNONYM_PREDICATE_SUFFIXES = (
    "altLabel",
    "prefLabel",
    "hasExactSynonym",
    "hasRelatedSynonym",
    "hasSynonym",
    "IAO_0000118",          # OBO "alternative term"
    "alternative_term",
)


def extract_ontology_pairs(
    label: str | None, annotations: dict[str, Any] | None
) -> list[tuple[str, str]]:
    """Pair a class `label` with each of its synonym annotations."""
    if not label or not isinstance(annotations, dict):
        return []
    out: list[tuple[str, str]] = []
    for pred, values in annotations.items():
        if not any(pred.endswith(sfx) for sfx in _SYNONYM_PREDICATE_SUFFIXES):
            continue
        if isinstance(values, (str, bytes)):
            values = [values]
        if not isinstance(values, Iterable):
            continue
        for v in values:
            if not isinstance(v, str):
                continue
            cand = _strip_marks(v)
            if is_valid_pair(label, cand):
                out.append((label, cand))
    return out


# --------------------------------------------------------------------------
# Pair assembly
# --------------------------------------------------------------------------


@dataclass
class MinedPair:
    term_a: str
    term_b: str
    surface_a: str
    surface_b: str
    evidence_kind: str
    evidence_ref: str | None = None
    occurrences: int = 1


def make_pair(
    left: str, right: str, kind: str, ref: str | None = None
) -> MinedPair | None:
    """Normalize + order a surface pair into a storable row.

    Ordering by normalized term (`term_a < term_b`) is what lets ONE row
    answer a lookup from either direction; the DB enforces it with a CHECK.
    """
    ln, rn = normalize_term(left), normalize_term(right)
    if not ln or not rn or ln == rn:
        return None
    if ln < rn:
        return MinedPair(ln, rn, _strip_marks(left), _strip_marks(right), kind, ref)
    return MinedPair(rn, ln, _strip_marks(right), _strip_marks(left), kind, ref)


def mine_text(text: str, ref: str | None = None) -> list[MinedPair]:
    """All deterministic text sources over one chunk."""
    pairs: list[MinedPair] = []
    for left, right in extract_parenthetical_pairs(text):
        p = make_pair(left, right, "parenthetical", ref)
        if p:
            pairs.append(p)
    for left, right in extract_phrase_pairs(text):
        p = make_pair(left, right, "phrase", ref)
        if p:
            pairs.append(p)
    return pairs


def _merge(acc: dict[tuple[str, str, str], MinedPair], pairs: Iterable[MinedPair]) -> None:
    """Accumulate pairs, counting occurrences. First surface form wins --
    it is as good as any, and keeping it stable makes re-runs idempotent."""
    for p in pairs:
        key = (p.term_a, p.term_b, p.evidence_kind)
        existing = acc.get(key)
        if existing is None:
            acc[key] = p
        else:
            existing.occurrences += p.occurrences


# --------------------------------------------------------------------------
# DB pass
# --------------------------------------------------------------------------


@dataclass
class MineSummary:
    chunks_scanned: int = 0
    classes_scanned: int = 0
    pairs_found: int = 0
    pairs_written: int = 0
    pairs_pruned: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    wall_seconds: float = 0.0
    dry_run: bool = False
    samples: list[MinedPair] = field(default_factory=list)


# Prefer verbatim full-text chunks; fall back to summary chunks only for
# documents that have none. Load-bearing: summarization would plausibly
# collapse "MOUNJARO (tirzepatide)" down to just "Mounjaro", destroying the
# very apposition this pass depends on.
_CHUNK_PAGE_SQL = """
SELECT c.id, c.chunk_identifier, c.text
  FROM graphrag.chunks c
 WHERE c.status = 'ACTIVE'
   AND c.text IS NOT NULL
   AND (c.kind = 'fulltext' OR NOT EXISTS (
         SELECT 1 FROM graphrag.chunks f
          WHERE f.document_id = c.document_id
            AND f.kind = 'fulltext'
            AND f.status = 'ACTIVE'))
   AND c.id > :after
 ORDER BY c.id
 LIMIT :page
"""

_CLASS_PAGE_SQL = """
SELECT id, iri, label, extra_metadata -> 'annotations' AS annotations
  FROM graphrag.ontology_classes
 WHERE label IS NOT NULL
   AND extra_metadata ? 'annotations'
   AND id > :after
 ORDER BY id
 LIMIT :page
"""

_UPSERT_SQL = """
INSERT INTO graphrag.term_aliases (
  id, term_a, term_b, surface_a, surface_b,
  evidence_kind, evidence_ref, occurrences, graph_version, created_at
) VALUES (
  :id, :term_a, :term_b, :surface_a, :surface_b,
  :evidence_kind, :evidence_ref, :occurrences, :gv, now()
)
ON CONFLICT (term_a, term_b, evidence_kind) DO UPDATE SET
  occurrences   = EXCLUDED.occurrences,
  surface_a     = EXCLUDED.surface_a,
  surface_b     = EXCLUDED.surface_b,
  evidence_ref  = EXCLUDED.evidence_ref,
  graph_version = EXCLUDED.graph_version
"""

# Keyset pagination, not OFFSET: bounded memory on a 2.7 GB dev box and no
# re-scan cost per page.
_PAGE = 500
_WRITE_BATCH = 200


async def mine_aliases(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    min_occurrences: int = 1,
    verbose: bool = False,
) -> MineSummary:
    """Scan the corpus + ontology and upsert every mined synonym pair.

    Idempotent: `occurrences` is REPLACED with the freshly-counted total
    rather than incremented, so re-running after an ingest converges
    instead of inflating.
    """
    t0 = time.time()
    summary = MineSummary(dry_run=dry_run)
    acc: dict[tuple[str, str, str], MinedPair] = {}

    # ---- text sources ----
    after = uuid.UUID(int=0)
    async with session_scope() as session:
        while True:
            page = _PAGE if limit is None else min(_PAGE, limit - summary.chunks_scanned)
            if page <= 0:
                break
            rows = (
                await session.execute(
                    sql_text(_CHUNK_PAGE_SQL), {"after": after, "page": page}
                )
            ).all()
            if not rows:
                break
            for cid, ciri, ctext in rows:
                after = cid
                summary.chunks_scanned += 1
                _merge(acc, mine_text(ctext or "", ref=ciri))
            if verbose:
                print(
                    f"[mine-aliases] scanned {summary.chunks_scanned:,} chunk(s); "
                    f"{len(acc):,} distinct pair(s) so far"
                )

    # ---- ontology annotations (already imported, never read until now) ----
    after = uuid.UUID(int=0)
    async with session_scope() as session:
        while True:
            rows = (
                await session.execute(
                    sql_text(_CLASS_PAGE_SQL), {"after": after, "page": _PAGE}
                )
            ).all()
            if not rows:
                break
            for cls_id, iri, label, annotations in rows:
                after = cls_id
                summary.classes_scanned += 1
                pairs = extract_ontology_pairs(label, annotations)
                _merge(
                    acc,
                    [
                        p
                        for p in (
                            make_pair(a, b, "ontology", iri) for a, b in pairs
                        )
                        if p
                    ],
                )

    # ---- filter + report ----
    kept = [p for p in acc.values() if p.occurrences >= min_occurrences]
    kept.sort(key=lambda p: (-p.occurrences, p.term_a, p.term_b))
    summary.pairs_found = len(kept)
    for p in kept:
        summary.by_kind[p.evidence_kind] = summary.by_kind.get(p.evidence_kind, 0) + 1
    summary.samples = kept[:50]

    if dry_run or not kept:
        summary.wall_seconds = time.time() - t0
        return summary

    # ---- write ----
    async with session_scope() as session:
        gv = await current_version(session)
    async with session_scope() as session:
        for i in range(0, len(kept), _WRITE_BATCH):
            batch = kept[i : i + _WRITE_BATCH]
            await session.execute(
                sql_text(_UPSERT_SQL),
                [
                    {
                        "id": uuid.uuid4(),
                        "term_a": p.term_a,
                        "term_b": p.term_b,
                        "surface_a": p.surface_a,
                        "surface_b": p.surface_b,
                        "evidence_kind": p.evidence_kind,
                        "evidence_ref": p.evidence_ref,
                        "occurrences": p.occurrences,
                        "gv": gv,
                    }
                    for p in batch
                ],
            )
            summary.pairs_written += len(batch)

        # `term_aliases` is a DERIVED view of the corpus, so a full scan
        # replaces it wholesale: pairs that no longer qualify (tightened
        # precision guards, or text removed from the corpus) must not
        # linger and keep steering probes. Skipped under --limit, where
        # the scan is a sample and pruning would delete almost everything.
        #
        # `evidence_kind='manual'` is exempt. Those rows are hand-curated
        # for pairs the corpus never states outright, and since this
        # refresh runs automatically after every ingest, pruning them
        # would delete an operator's work with no warning.
        if limit is None:
            pruned = await session.execute(
                sql_text("""
                DELETE FROM graphrag.term_aliases
                 WHERE evidence_kind <> 'manual'
                   AND (term_a, term_b, evidence_kind) NOT IN (
                     SELECT * FROM unnest(
                       CAST(:a AS text[]), CAST(:b AS text[]), CAST(:k AS text[])
                     )
                 )
                """),
                {
                    "a": [p.term_a for p in kept],
                    "b": [p.term_b for p in kept],
                    "k": [p.evidence_kind for p in kept],
                },
            )
            summary.pairs_pruned = pruned.rowcount or 0

    summary.wall_seconds = time.time() - t0
    return summary

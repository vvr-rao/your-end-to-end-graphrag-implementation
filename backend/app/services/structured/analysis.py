"""How a loaded table's columns interact -- measured with SQL, no LLM.

A column description says what a column is; it does not say how to query
the table correctly. These facts come from the data itself and hold for any
table:

* **grain** -- a small set of columns that identifies one row (greedy
  search for a unique key).
* **functional dependencies** -- ``a -> b`` when every value of ``a`` has one
  value of ``b`` (a code and its name, a report id and its date).
* **repeated values** -- a measure that stays the same when only some
  columns change (and the columns that depend on them), with how often
  (share of groups with several rows, share of those whose rows agree). This
  is EVIDENCE that the column may identify the REPORT or VERSION a value came
  from (a filing, vintage, revision or snapshot restating the same figure) --
  but a real dimension can repeat too (a price unchanged for years). Nothing
  is decided here: a column is treated as a version only when a measure
  definition names it, with a reason (``refine``).
* **item columns** -- a text column in the grain whose values name different
  quantities (a long-format table); its most widely shared values are listed
  so the measures can be defined on them.
* **total rows** -- a value of a grain column ('Overall', 'All', 'Total',
  'World'...) whose rows hold the SUM of the other rows of their group (the
  rest of the grain). Published tables often ship totals next to their parts;
  adding both counts everything twice. The query builder keeps only the total
  rows when a query does not use that column and drops them when it does.

The result is stored in ``meta["analysis"]`` and drives both the LLM step
that defines measures (``refine``) and the deterministic tests of every
stored query.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text as sql_text

log = logging.getLogger(__name__)

MAX_KEY_COLUMNS = 8
MAX_CANDIDATES = 16
REPEAT_MIN_SHARE = 0.75       # of multi-row groups where the measure is constant
REPEAT_MIN_GROUPS = 0.02      # multi-row groups / all groups, to count at all
TOP_VALUES = 60
TIMEOUT_MS = 30000
NEAR_DEP_MIN_VALUES = 50     # near-dependency only over columns this varied
NEAR_DEP_SHARE = 0.02         # ... and at most this share of exceptions
SPARSE_NULL_SHARE = 0.3       # columns this often null are annotations, not identity
TOTAL_CANDIDATE_VALUES = 5    # per grain column, the values on the most rows (+ as many named as totals)
TOTAL_MIN_COVERAGE = 0.1      # a total may cover only some groups (a section's totals); the builder reads parts where it is absent
TOTAL_FULL_COVERAGE = 0.5     # below this coverage a total needs TOTAL_MIN_GROUPS_PARTIAL compared groups
TOTAL_MIN_GROUPS = 5          # groups where total and (>= 2) parts can be compared
TOTAL_MIN_GROUPS_PARTIAL = 10 # ... for a total present in only some groups
TOTAL_MIN_SHARE = 0.9         # ... and where the total equals the sum of the parts
TOTAL_TOLERANCE = 0.005       # relative difference still counted as equal (rounding)
TOTAL_MAX_MEASURES = 6
TOTAL_MAX_GROUPS = 20000      # groups read per check
TOTAL_ROUNDED_MAX = 0.1       # a rounded total may differ from its parts by at most this share
MIRROR_MIN_SHARE = 0.5        # a text column restating a measure has about as many values as it
MIRROR_MAX_SPLIT = 0.05       # ... allowing this share of numbers with several spellings
LABEL_MIN_SHARE = 0.8         # a column is a total's label when this share of its values is the total's own
COARSER_MAX = 4               # columns a dropped label column determines, added back to the grouping
# words that nominate a value as a total candidate (the numeric check decides)
TOTAL_WORDS = r"\m(total|all|world|overall|aggregate|sum|grand)\M"


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _cols(cols: list[str]) -> str:
    return ", ".join(_q(c) for c in cols)


async def _scalar(session, sql: str, binds: dict | None = None):
    try:
        async with session.begin_nested():
            await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
            return (await session.execute(sql_text(sql), binds or {})).scalar()
    except Exception as exc:  # a slow or odd column must not stop the analysis
        log.info("analysis query skipped: %s", str(exc).splitlines()[0][:200])
        return None


def candidate_columns(meta: dict) -> list[dict]:
    """Columns that can identify or classify a row (not measures, not free text)."""
    out = []
    for c in meta.get("columns") or []:
        if c.get("role") in {"measure", "text"}:
            continue
        out.append(c)
    return out[:MAX_CANDIDATES * 2]


def measure_columns(meta: dict) -> list[str]:
    return [c["name"] for c in meta.get("columns") or []
            if c.get("role") == "measure" and c.get("type") in {"integer", "numeric"}]


async def _distinct(session, ref: str, cols: list[str]) -> int | None:
    if not cols:
        return 1
    return await _scalar(session, f"SELECT COUNT(*) FROM (SELECT DISTINCT {_cols(cols)} FROM {ref}) d")


async def analyze_table(session, meta: dict) -> dict[str, Any]:
    table = meta["table_name"]
    ref = f"graphrag.{_q(table)}"
    rows = await _scalar(session, f"SELECT COUNT(*) FROM {ref}") or 0
    out: dict[str, Any] = {"row_count": rows, "grain": [], "grain_unique": False,
                           "dependencies": {}, "near_dependencies": {}, "repeats": {}, "item_columns": {},
                           # the data this analysis measured: a reload makes it stale
                           "file_sha": (meta.get("source") or {}).get("file_sha")}
    if rows == 0:
        return out

    cands = candidate_columns(meta)
    distinct = {}
    nulls = {}
    for c in cands:
        n = await _distinct(session, ref, [c["name"]])
        if n is not None:
            distinct[c["name"]] = n
            nulls[c["name"]] = await _scalar(
                session, f"SELECT COUNT(*) FROM {ref} WHERE {_q(c['name'])} IS NULL") or 0
    # Columns unique and filled on EVERY row: row keys. They are left out of
    # the grain search below (a unique column trivially "explains" the rows),
    # but recorded so a per-row measure can be keyed by the subject's own
    # identifier instead of a combination of descriptive columns.
    out["row_keys"] = [c["name"] for c in cands if distinct.get(c["name"]) == rows
                       and not nulls.get(c["name"])]
    # A column unique on (nearly) every row is a row id, not part of a grain.
    usable = [c["name"] for c in cands if c["name"] in distinct and 1 < distinct[c["name"]] < 0.95 * rows]
    usable = sorted(usable, key=lambda n: -distinct[n])[:MAX_CANDIDATES]

    # ---- grain: greedy unique key
    key: list[str] = []
    best = 1
    while len(key) < MAX_KEY_COLUMNS and best < rows:
        step = None
        for c in usable:
            if c in key:
                continue
            n = await _distinct(session, ref, [*key, c])
            if n is not None and n > best and (step is None or n > step[1]):
                step = (c, n)
        if step is None:
            break
        key.append(step[0])
        best = step[1]
    out["grain"] = key
    out["grain_unique"] = best >= rows
    out["grain_distinct"] = best

    # ---- functional dependencies a -> b (only where |b| <= |a|)
    deps: dict[str, list[str]] = {}
    near: dict[str, list[str]] = {}
    for a in usable:
        for b in usable:
            if a == b or distinct[b] > distinct[a]:
                continue
            n = await _distinct(session, ref, [a, b])
            if n is not None and n == distinct[a]:
                deps.setdefault(a, []).append(b)
            elif (n is not None and distinct[a] >= NEAR_DEP_MIN_VALUES
                  and n - distinct[a] <= NEAR_DEP_SHARE * distinct[a]):
                # all but a few values of a have one b (a spelling variant, a
                # one-off change): b describes a, it does not split it
                near.setdefault(a, []).append(b)
    out["dependencies"] = deps
    out["near_dependencies"] = near

    # ---- repeated measure values across report/version columns.
    # A mostly-null column can be an annotation set in only ONE of the reports
    # that restate a value; it splits the restated rows apart. So the test is
    # also run with each such column left out (one at a time: another sparse
    # column may be a real part of the identity, e.g. a start date that only
    # some rows have), and the column that reveals the repeat is reported with
    # the report columns.
    sparse = [c for c in usable if nulls.get(c, 0) >= SPARSE_NULL_SHARE * rows]
    out["sparse"] = sparse

    async def _repeat(m: str, group: list[str]):
        if not group:
            return None
        stats = await _scalar(session, f"""
            SELECT json_build_object(
                'multi', COUNT(*) FILTER (WHERE n > 1),
                'groups', COUNT(*),
                'same', COUNT(*) FILTER (WHERE n > 1 AND nd = 1))
              FROM (SELECT COUNT(*) AS n, COUNT(DISTINCT {_q(m)}) AS nd
                      FROM {ref} WHERE {_q(m)} IS NOT NULL GROUP BY {_cols(group)}) g""")
        if not stats or not stats.get("groups"):
            return None
        multi, groups, same = stats["multi"], stats["groups"], stats["same"]
        if multi and multi / groups >= REPEAT_MIN_GROUPS and same / multi >= REPEAT_MIN_SHARE:
            return round(multi / groups, 3), round(same / multi, 3)
        return None

    roles = {c["name"]: c.get("role") for c in cands}
    measures = measure_columns(meta)
    for m in measures:
        found = []
        for c in usable:
            if c in sparse:
                continue
            # columns a report column determines move with it (its date, its
            # type...), except the identifiers of what the value is ABOUT (a
            # report belongs to one entity; that entity stays in the group)
            keep = {b for b in deps.get(c, []) if roles.get(b) == "identifier"}
            keep |= {x for k in keep for x in deps.get(k, [])}
            moved = [c, *[b for b in deps.get(c, []) if b not in keep]]
            base = [x for x in usable if x not in moved]
            hit, extra = await _repeat(m, base), []
            for sp in [x for x in sparse if x in base] if hit is None else []:
                hit = await _repeat(m, [x for x in base if x != sp])
                if hit is not None:
                    extra = [sp]
                    break
            if hit is not None:
                found.append({"column": c, "with": moved[1:] + extra,
                              "share_of_groups_repeated": hit[0], "share_identical": hit[1]})
        if found:
            out["repeats"][m] = found

    # ---- item columns: text grain columns with many values; their most
    # widely shared values (present for the most entities, then most rows)
    ent = next((c["name"] for c in cands if c.get("role") == "identifier" and c["name"] in usable), None)
    domains = meta.get("value_domains") or {}
    types = {c["name"]: c.get("type") for c in cands}
    for c in key:
        if types.get(c) != "text" or "values" in domains.get(c, {}):
            continue
        order = f"COUNT(DISTINCT {_q(ent)}) DESC, " if ent and ent != c else ""
        try:
            async with session.begin_nested():
                await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
                vals = (await session.execute(sql_text(
                    f"SELECT {_q(c)} FROM {ref} WHERE {_q(c)} IS NOT NULL GROUP BY 1 "
                    f"ORDER BY {order}COUNT(*) DESC, 1 LIMIT {TOP_VALUES}"))).scalars().all()
        except Exception:
            vals = []
        if vals:
            out["item_columns"][c] = {"distinct": distinct.get(c), "top_values": [str(v) for v in vals]}

    out["total_rows"] = await detect_total_rows(session, meta, key, types, measures)
    return out


async def detect_total_rows(session, meta: dict, key: list[str], types: dict[str, str],
                            measures: list[str]) -> list[dict]:
    """Values of a grain column whose rows are the totals of the other rows of
    their group: [{"column", "value", "groups_by", "groups_compared",
    "share_equal", "measure"}]. SQL only.

    A value qualifies when, for at least one measure, its row equals the sum of
    the group's other rows in nearly every group where it can be compared:

    * groups are formed by the rest of the grain, minus columns that only
      MIRROR the measure (its value written as text: grouping by it pairs a
      total with its parts only when they are written alike), minus columns
      whose values on the candidate's rows never occur on other rows (the
      total's own label, e.g. a detail column reading "production: rounded");
    * a group is compared only when it has ONE candidate row and at least TWO
      other rows (a total over a single part proves nothing: when one entity
      is the only producer, its row equals the world total);
    * equal = within TOTAL_TOLERANCE, or within half the total's own rounding
      unit (a published "190" covers 185-195), capped at TOTAL_ROUNDED_MAX;
    * of a column's qualifying values, the best supported one is kept.
    Candidates are the values on the most rows plus values whose words name a
    total (TOTAL_WORDS); only the numeric check decides."""
    table = meta["table_name"]
    ref = f"graphrag.{_q(table)}"
    found: list[dict] = []
    if not measures:
        return found
    mirrors: dict[str, set[str]] = {}
    for m in measures[:TOTAL_MAX_MEASURES]:
        mirrors[m] = {g for g in key if types.get(g) == "text" and await _mirrors(session, ref, g, m)}
    for c in key:
        if types.get(c) != "text":
            continue
        try:
            async with session.begin_nested():
                await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
                top = (await session.execute(sql_text(
                    f"SELECT {_q(c)} FROM {ref} WHERE {_q(c)} IS NOT NULL GROUP BY 1 "
                    f"ORDER BY COUNT(*) DESC, 1 LIMIT {TOTAL_CANDIDATE_VALUES}"))).scalars().all()
                named = (await session.execute(sql_text(
                    f"SELECT DISTINCT {_q(c)} FROM {ref} WHERE {_q(c)} ~* :w LIMIT {TOTAL_CANDIDATE_VALUES}"),
                    {"w": TOTAL_WORDS})).scalars().all()
        except Exception as exc:
            log.info("total-row candidates skipped: %s", str(exc).splitlines()[0][:200])
            continue
        best = None
        for value in list(dict.fromkeys([*named, *top])):
            labels = [g for g in key if g != c and await _label_only(session, ref, c, value, g)]
            # a dropped label column still carries a coarser distinction (a
            # "production: rounded" detail is still production): keep the
            # columns it determines that the candidate's rows share with others
            coarser: list[str] = []
            for g in labels:
                for x in await _coarser(session, ref, meta, g, c, value, [*key, *coarser]):
                    coarser.append(x)
            for m in measures[:TOTAL_MAX_MEASURES]:
                group = [g for g in key if g != c and g not in mirrors[m] and g not in labels] + coarser
                if not group:
                    continue
                st = await _compare_total(session, ref, c, value, m, group)
                if st is None:
                    continue
                compared, equal, eligible = st
                need = TOTAL_MIN_GROUPS if compared >= TOTAL_FULL_COVERAGE * eligible else TOTAL_MIN_GROUPS_PARTIAL
                if compared < need or compared < TOTAL_MIN_COVERAGE * eligible:
                    continue
                share = equal / compared
                if share >= TOTAL_MIN_SHARE and (best is None or (share, compared) > (best[1], best[2])):
                    best = (str(value), share, compared, m, group)
        if best is not None:
            found.append({"column": c, "value": best[0], "groups_by": best[4],
                          "groups_compared": best[2], "share_equal": round(best[1], 3),
                          "measure": best[3]})
    return await _distinct_totals(session, ref, found)


async def _distinct_totals(session, ref: str, found: list[dict]) -> list[dict]:
    """One marker per set of total rows: a total's own label (a detail column
    reading "production: rounded") marks the very same rows as the total value
    of another column, and two markers on one set of rows would make a query
    that groups by one of them read nothing. Keeps the value whose words name a
    total, then the better supported one."""
    def rank(t: dict) -> tuple:
        return (bool(re.search(TOTAL_WORDS.replace(r"\m", r"\b").replace(r"\M", r"\b"), t["value"], re.I)),
                t["groups_compared"], t["share_equal"])

    kept: list[dict] = []
    for t in sorted(found, key=rank, reverse=True):
        same = False
        for k in kept:
            diff = await _scalar(session, f"""
                SELECT COUNT(*) FROM {ref}
                 WHERE ({_q(t["column"])} = :a) IS DISTINCT FROM ({_q(k["column"])} = :b)""",
                {"a": t["value"], "b": k["value"]})
            if diff == 0:
                same = True
                break
        if not same:
            kept.append(t)
    return [t for t in found if t in kept]


async def _mirrors(session, ref: str, col: str, measure: str) -> bool:
    """Whether a text column only restates a measure: (nearly) every value of
    the measure has one text -- spelling variants such as "1,000" / "1000" are
    tolerated -- and the text has about as many values as the measure."""
    st = await _scalar(session, f"""
        SELECT json_build_object(
          'split', (SELECT COUNT(*) FROM (SELECT {_q(measure)} FROM {ref} WHERE {_q(measure)} IS NOT NULL
                    GROUP BY 1 HAVING COUNT(DISTINCT {_q(col)}) > 1) x),
          'nm', (SELECT COUNT(DISTINCT {_q(measure)}) FROM {ref}),
          'nc', (SELECT COUNT(DISTINCT {_q(col)}) FROM {ref} WHERE {_q(measure)} IS NOT NULL))""")
    if not st or not st.get("nm"):
        return False
    return st["split"] <= MIRROR_MAX_SPLIT * st["nm"] and st["nc"] >= MIRROR_MIN_SHARE * st["nm"]


async def _label_only(session, ref: str, c: str, value: Any, g: str) -> bool:
    """Whether column g's values on the candidate's rows are (nearly all) its
    own: values no other row has -- the total's label, not a shared dimension."""
    st = await _scalar(session, f"""
        SELECT json_build_object('n', COUNT(*), 'shared', COUNT(*) FILTER (WHERE EXISTS (
                   SELECT 1 FROM {ref} b WHERE b.{_q(g)} = a.v AND b.{_q(c)} IS DISTINCT FROM :v)))
          FROM (SELECT DISTINCT {_q(g)} AS v FROM {ref} WHERE {_q(c)} = :v AND {_q(g)} IS NOT NULL) a""",
        {"v": value})
    if not st or not st.get("n"):
        return False
    return st["n"] - st["shared"] >= LABEL_MIN_SHARE * st["n"]


async def _coarser(session, ref: str, meta: dict, g: str, c: str, value: Any,
                   have: list[str]) -> list[str]:
    """Columns that label column g determines (each g value has one of their
    values, with few exceptions) and that the candidate's rows share with
    other rows: the distinction g carries besides the total's label."""
    out = []
    for col in meta.get("columns") or []:
        x = col.get("name")
        if (not x or x in have or x in (g, c) or col.get("role") in {"measure", "text"}
                or col.get("type") != "text"):
            continue
        st = await _scalar(session, f"""
            SELECT json_build_object(
              'n', (SELECT COUNT(DISTINCT {_q(g)}) FROM {ref}),
              'split', (SELECT COUNT(*) FROM (SELECT {_q(g)} FROM {ref} GROUP BY 1
                        HAVING COUNT(DISTINCT {_q(x)}) > 1) y),
              'filled', (SELECT COUNT(*) FROM {ref} WHERE {_q(c)} = :v AND {_q(x)} IS NOT NULL))""",
            {"v": value})
        if not st or not st.get("n") or not st.get("filled") or st["split"] > NEAR_DEP_SHARE * st["n"]:
            continue
        if await _label_only(session, ref, c, value, x):
            continue
        out.append(x)
        if len(out) >= COARSER_MAX:
            break
    return out


async def _compare_total(session, ref: str, c: str, value: Any, m: str,
                         group: list[str]) -> tuple[int, int, int] | None:
    """(groups compared, groups where total == sum of parts, groups with >= 2
    parts) for one candidate value, measure and grouping."""
    try:
        async with session.begin_nested():
            await session.execute(sql_text(f"SET LOCAL statement_timeout = {TIMEOUT_MS}"))
            rows = (await session.execute(sql_text(f"""
                SELECT tot, rest, nt FROM (
                  SELECT SUM({_q(m)}) FILTER (WHERE {_q(c)} = :v) AS tot,
                         SUM({_q(m)}) FILTER (WHERE {_q(c)} IS DISTINCT FROM :v) AS rest,
                         COUNT({_q(m)}) FILTER (WHERE {_q(c)} = :v) AS nt,
                         COUNT({_q(m)}) FILTER (WHERE {_q(c)} IS DISTINCT FROM :v) AS nr
                    FROM {ref} GROUP BY {_cols(group)}) g
                 WHERE nr >= 2 LIMIT {TOTAL_MAX_GROUPS}"""), {"v": value})).all()
    except Exception as exc:
        log.info("total-row check skipped: %s", str(exc).splitlines()[0][:200])
        return None
    compared = equal = 0
    for tot, rest, nt in rows:
        if nt != 1 or tot is None or rest is None:
            continue
        compared += 1
        equal += _total_matches(float(tot), float(rest))
    return compared, equal, len(rows)


def _total_matches(tot: float, rest: float) -> bool:
    """A total equals its parts' sum: within TOTAL_TOLERANCE, or within half the
    total's rounding unit as written (190 -> 10s, 380000 -> 10,000s), capped."""
    diff = abs(tot - rest)
    if diff <= TOTAL_TOLERANCE * max(abs(tot), abs(rest), 1e-9):
        return True
    if tot == 0 or tot != int(tot):
        return False
    unit, t = 1, int(abs(tot))
    while t % 10 == 0:
        t //= 10
        unit *= 10
    return unit > 1 and diff <= unit / 2 and diff <= TOTAL_ROUNDED_MAX * abs(tot)


def total_groups(meta: dict | None) -> dict[str, list[str]]:
    """{column: the columns its total rows are grouped by} (from the analysis):
    a row's group HAS a total when a total row shares its values on them."""
    out: dict[str, list[str]] = {}
    for t in ((meta or {}).get("analysis") or {}).get("total_rows") or []:
        if isinstance(t, dict) and t.get("column") and t.get("groups_by"):
            out.setdefault(str(t["column"]), [str(g) for g in t["groups_by"]])
    return out


def total_markers(meta: dict | None) -> dict[str, str]:
    """{column: total value} from a table's analysis (empty when none)."""
    out: dict[str, str] = {}
    for t in ((meta or {}).get("analysis") or {}).get("total_rows") or []:
        if isinstance(t, dict) and t.get("column") and t.get("value") is not None:
            out.setdefault(str(t["column"]), str(t["value"]))
    return out


def report_columns(analysis: dict, measure: str | None = None) -> list[str]:
    """Columns across which a measure's value often REPEATS (plus the columns
    they determine): candidates for report/version columns, not a verdict --
    see :func:`version_columns` for the confirmed ones."""
    out: list[str] = []
    reps = analysis.get("repeats") or {}
    for m, found in reps.items():
        if measure and m != measure:
            continue
        for f in found:
            for c in [f["column"], *f.get("with", [])]:
                if c not in out:
                    out.append(c)
    return out


def version_columns(meta: dict, analysis: dict | None = None) -> list[str]:
    """Columns CONFIRMED to identify only the report/version of a value: the
    ``version_column`` a measure definition names (with a reason), plus the
    columns the analysis saw repeat together with it. Repeated values alone
    are evidence, never a confirmation (:func:`report_columns` lists those)."""
    an = analysis if analysis is not None else (meta.get("analysis") or {})
    out: list[str] = []
    for it in meta.get("measures") or []:
        m = it.get("measure") or {}
        v = m.get("version_column") or m.get("latest_by")
        if not v:
            continue
        moved = [v]
        for found in (an.get("repeats") or {}).values():
            for f in found:
                if f.get("column") == v:
                    moved += list(f.get("with") or [])
        for c in moved:
            if c not in out:
                out.append(c)
    return out


def summary_for_prompt(analysis: dict) -> dict:
    """The parts of the analysis the LLM and the selection model need."""
    out = {"grain": analysis.get("grain"), "grain_unique": analysis.get("grain_unique")}
    if analysis.get("row_keys"):
        out["row_keys"] = analysis["row_keys"]
    if analysis.get("repeats"):
        out["values_repeated_across"] = {
            m: [{"columns": [f["column"], *f.get("with", [])],
                 "share_of_groups_repeated": f.get("share_of_groups_repeated"),
                 "share_identical": f.get("share_identical")} for f in found]
            for m, found in analysis["repeats"].items()}
    if analysis.get("dependencies"):
        out["determines"] = analysis["dependencies"]
    if analysis.get("near_dependencies"):
        out["mostly_determines"] = analysis["near_dependencies"]
    if analysis.get("item_columns"):
        out["item_columns"] = analysis["item_columns"]
    if analysis.get("total_rows"):
        out["total_rows"] = [
            {"column": t["column"], "value": t["value"],
             "meaning": (f"rows with {t['column']} = {t['value']!r} hold the total of the other rows "
                         f"with the same {', '.join(t.get('groups_by') or [])}")}
            for t in analysis["total_rows"]]
    return out

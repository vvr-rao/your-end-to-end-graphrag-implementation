"""Whole-page vision for borderless financial statements.

2026-10-23, fin-gen-subset: pdfplumber finds tables by ruling lines, so the
A$ filing's 37 "tables" were 1-4 row header strips like
['', 'Note', "A$'000", '', "A$'000"] and the LGES / SDI audit reports gave
none. A figure-dense page with no real table now goes whole to the vision
model, and every figure it returns must be printed on the page."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from backend.app.services import table_extract as te
from backend.app.services.prompts import PROMPTS

STATEMENT = """Consolidated statement of financial position
Note 2025 2024
A$'000 A$'000
Cash and cash equivalents 7 12,345 10,987
Trade and other receivables 8 4,210 3,876
Inventories 9 1,050 998
Property, plant and equipment 10 25,600 24,100
Intangible assets 11 3,300 3,410
Total assets 46,505 43,371
Trade and other payables 12 5,120 4,870
Borrowings 13 9,000 8,500
Total liabilities 14,120 13,370
Net assets 32,385 30,001
"""


def _table(rows: list[tuple[str, list[str]]], *, index: int = 0) -> dict:
    body = {
        "caption": "Statement of financial position",
        "columns": [{"columnIndex": i, "columnLabel": lab}
                    for i, lab in enumerate(["2025", "2024"])],
        "rows": [{"rowIndex": ri, "rowLabel": label,
                  "cells": [{"columnIndex": ci, "cellValue": v} for ci, v in enumerate(vals)]}
                 for ri, (label, vals) in enumerate(rows)],
    }
    return te._vision_body_to_jsonld(body, doc_sha="a" * 64, table_index=index,
                                     page_number=1, caption_hint=None)


GROUNDED = [("Cash and cash equivalents", ["12,345", "10,987"]),
            ("Inventories", ["1,050", "998"]),
            ("Total assets", ["46,505", "43,371"])]


# ---- helpers ----------------------------------------------------------------

def test_header_strip_is_header_only() -> None:
    assert te._is_header_only([["", "Note", "A$'000", "", "A$'000"]])
    assert te._is_header_only([["", "2025", "2024"], ["", "A$'000", "A$'000"]])


def test_statement_rows_are_not_header_only() -> None:
    grid = [["", "2025", "2024"],
            ["Cash", "12,345", "10,987"],
            ["Inventories", "1,050", "998"]]
    assert not te._is_header_only(grid)


def test_figure_lines_counts_statement_lines_not_prose() -> None:
    assert te._figure_lines(STATEMENT) >= 8
    prose = "In 2025 the group grew revenue by 12% across 3 regions.\n" * 20
    assert te._figure_lines(prose) == 0


def test_grounded_table_has_zero_ungrounded_share() -> None:
    assert te._ungrounded_share(_table(GROUNDED), STATEMENT) == 0.0


def test_invented_figures_raise_the_ungrounded_share() -> None:
    invented = GROUNDED[:1] + [("Goodwill", ["77,777", "66,666"])]
    assert te._ungrounded_share(_table(invented), STATEMENT) == 0.5


def test_bracketed_negatives_and_unicode_minus_are_grounded() -> None:
    text = "Net finance costs (1,234) (987)\nOther −2.50 −1.75\n"
    t = _table([("Net finance costs", ["(1,234)", "(987)"]),
                ("Other", ["−2.50", "−1.75"])])
    assert te._ungrounded_share(t, text) == 0.0


def test_notes_column_references_are_grounded() -> None:
    """LGES balance sheet: 'Cash and cash equivalents 3,5,6 ￦ 1,158,885'. The
    Notes cell must match the printed token, not runs of single digits."""
    text = "Cash and cash equivalents 3,5,6 ￦ 1,158,885 ￦ 849,894\n"
    t = _table([("Cash and cash equivalents", ["3,5,6", "1,158,885", "849,894"])])
    assert te._ungrounded_share(t, text) == 0.0


def test_currency_signs_and_ranges_are_grounded() -> None:
    """LGES p15/p20: cells as the model writes them, all printed on the page."""
    text = ("Increase ₩ (2,700) - (2,700)\n"
            "Discount rate 3.00%–3.97% 3.17%–4.12%\n")
    t = _table([("Increase", ["₩ (2,700)", "₩ -"]),
                ("Discount rate", ["3.00% – 3.97%", "3.17% – 4.12%"])])
    assert te._ungrounded_share(t, text) == 0.0
    wrong = _table([("Increase", ["₩ (2,900)", "₩ -"])])
    assert te._ungrounded_share(wrong, text) == 1.0


def test_table_without_numbers_counts_as_ungrounded() -> None:
    t = _table([("Basis of preparation", ["see note", "see note"])])
    assert te._ungrounded_share(t, STATEMENT) == 1.0


def test_page_prompt_allows_no_table_and_several_tables() -> None:
    system, user = PROMPTS["table_extract_page_vision"](page_number=4)
    assert '"no_table": true' in system + user
    assert '"tables"' in system + user


def test_settings_defaults_and_override(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(app_config={}))
    assert te.page_vision_settings() == {
        "enabled": True, "min_figure_lines": 8, "max_pages": 80, "max_ungrounded": 0.1}
    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(app_config={
        "tables": {"page_vision": False, "page_vision_max_pages_per_doc": 3}}))
    s = te.page_vision_settings()
    assert s["enabled"] is False and s["max_pages"] == 3


# ---- _extract_page_tables_via_vision -----------------------------------------

def _patch_vision_reply(monkeypatch, reply: str) -> None:
    monkeypatch.setattr(te, "_vision_spec", lambda task, fallback=None: {"model": "m"})
    monkeypatch.setattr(te, "_render_png_b64", lambda page, bbox: "png")

    async def fake_call(spec, system, user, png):
        return reply, 0.01
    monkeypatch.setattr(te, "_call_vision", fake_call)


def _page_tables(**kw):
    return asyncio.run(te._extract_page_tables_via_vision(
        object(), page_text=STATEMENT, doc_sha="a" * 64, first_table_index=5,
        page_number=1, max_ungrounded=0.1, **kw))


def test_page_vision_keeps_grounded_and_rejects_invented(monkeypatch) -> None:
    import json
    good = {"caption": "SOFP", "columns": [{"columnIndex": 0, "columnLabel": "2025"}],
            "rows": [{"rowIndex": 0, "rowLabel": "Cash",
                      "cells": [{"columnIndex": 0, "cellValue": "12,345"}]},
                     {"rowIndex": 1, "rowLabel": "Borrowings",
                      "cells": [{"columnIndex": 0, "cellValue": "9,000"}]}]}
    bad = {"caption": "Made up", "columns": [{"columnIndex": 0, "columnLabel": "2025"}],
           "rows": [{"rowIndex": 0, "rowLabel": "Goodwill",
                     "cells": [{"columnIndex": 0, "cellValue": "88,888"}]}]}
    _patch_vision_reply(monkeypatch, json.dumps({"tables": [good, bad]}))
    found, cost, rejected, blanked = _page_tables()
    assert [t["caption"] for t in found] == ["SOFP"]
    assert rejected == 1 and cost == 0.01 and blanked == 0
    assert found[0]["extractionMethod"] == "vision-llm"


def test_misread_cell_in_a_kept_table_is_blanked(monkeypatch) -> None:
    """SDI: revenue read as ...884,368 where the page prints ...884,388. The
    table passes the share limit; the wrong cell must not survive."""
    import json
    rows = [("Cash", "12,345"), ("Receivables", "4,210"), ("Inventories", "1,050"),
            ("PP&E", "25,600"), ("Intangibles", "3,300"), ("Total assets", "46,505"),
            ("Payables", "5,120"), ("Borrowings", "9,000"), ("Liabilities", "14,120"),
            ("Net assets", "32,358")]  # misread: page prints 32,385
    body = {"caption": "SOFP", "columns": [{"columnIndex": 0, "columnLabel": "2025"}],
            "rows": [{"rowIndex": i, "rowLabel": lab,
                      "cells": [{"columnIndex": 0, "cellValue": v}]}
                     for i, (lab, v) in enumerate(rows)]}
    _patch_vision_reply(monkeypatch, json.dumps({"tables": [body]}))
    found, _, rejected, blanked = _page_tables()
    assert rejected == 0 and blanked == 1
    values = {r["rowLabel"]: r["cells"][0]["cellValue"] for r in found[0]["rows"]}
    assert values["Net assets"] == "" and values["Total assets"] == "46,505"


def test_blank_keeps_dashes_and_text_cells() -> None:
    t = _table([("Cash", ["12,345", "-"]), ("Basis", ["see note", "n/a"])])
    assert te._blank_ungrounded_cells(t, STATEMENT) == 0


def test_page_vision_no_table_and_bad_json(monkeypatch) -> None:
    _patch_vision_reply(monkeypatch, '{"no_table": true}')
    assert _page_tables() == ([], 0.01, 0, 0)
    _patch_vision_reply(monkeypatch, "not json")
    assert _page_tables() == ([], 0.01, 0, 0)


# ---- extract_tables_async integration (fake pdfplumber) ----------------------

class _Detected:
    """A pdfplumber table whose cells are the fixture's header strip."""
    def extract(self):
        return [["", "Note", "A$'000", "", "A$'000"]]


class _Page:
    width, height = 600.0, 800.0

    def __init__(self, text: str, n_tables: int) -> None:
        self._text, self._n = text, n_tables

    def find_tables(self):
        return [_Detected()] * self._n

    def extract_text(self, **kw):
        self.text_kwargs = kw
        return self._text

    def flush_cache(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_pdf(monkeypatch, tmp_path):
    """One page whose only detected 'table' is a header strip."""
    pages: list[_Page] = []

    class _Pdf:
        def __init__(self, _path):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(sys.modules, "pdfplumber", types.SimpleNamespace(open=_Pdf))
    strip = [["", "Note", "A$'000", "", "A$'000"]]
    monkeypatch.setattr(te, "_merge_row_fragments",
                        lambda tables: [((50, 100, 550, 120), strip, None, 1)] * len(tables))
    monkeypatch.setattr(te, "_bbox_filter_reason", lambda *a, **k: None)
    monkeypatch.setattr(te, "_content_filter_reason", lambda *a, **k: None)
    monkeypatch.setattr(te, "_caption_hint", lambda *a, **k: None)
    monkeypatch.setattr(te, "_is_simple_table", lambda grid: True)
    header = _table([("", ["A$'000", "A$'000"])])
    monkeypatch.setattr(te, "_build_jsonld_from_grid", lambda grid, **k: dict(header))
    monkeypatch.setattr(te, "page_vision_settings", lambda: {
        "enabled": True, "min_figure_lines": 8, "max_pages": 80, "max_ungrounded": 0.1})
    pdf = tmp_path / "filing.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf, pages, tmp_path


def _extract(pdf: Path, tmp: Path, *, use_vision: bool = True):
    return asyncio.run(te.extract_tables_async(
        pdf, router=None, run_cache_dir=tmp / "run", user_cache_dir=tmp / "user",
        use_vision=use_vision))


def _patch_page_vision(monkeypatch, result=None, exc: Exception | None = None):
    calls: list[int] = []

    async def fake(page, **kw):
        calls.append(kw["page_number"])
        if exc is not None:
            raise exc
        return result if result is not None else ([], 0.0, 0, 0)
    monkeypatch.setattr(te, "_extract_page_tables_via_vision", fake)
    return calls


def test_header_strip_is_replaced_by_the_page_vision_table(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=1))
    statement = _table(GROUNDED, index=1)
    calls = _patch_page_vision(monkeypatch, result=([statement], 0.008, 0, 0))
    res = _extract(pdf, tmp)
    assert calls == [1]
    assert [t["extractionMethod"] for t in res.tables] == ["vision-llm"]
    assert res.manifest["n_page_vision_tables"] == 1
    assert res.manifest["n_header_only_dropped"] == 1


def test_no_table_answer_keeps_the_header_strip(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=1))
    _patch_page_vision(monkeypatch)
    res = _extract(pdf, tmp)
    assert len(res.tables) == 1 and res.manifest["n_header_only_dropped"] == 0


def test_vision_failure_keeps_the_strip_and_counts_an_error(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=1))
    _patch_page_vision(monkeypatch, exc=te.VisionCallError("503"))
    res = _extract(pdf, tmp)
    assert len(res.tables) == 1
    assert res.manifest["n_vision_errors"] == 1


def test_prose_page_is_not_sent(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page("Directors' report.\nThe group traded well.\n" * 10, n_tables=1))
    calls = _patch_page_vision(monkeypatch)
    res = _extract(pdf, tmp)
    assert calls == [] and len(res.tables) == 1


def test_figure_dense_page_with_no_detected_table_is_sent(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=0))
    calls = _patch_page_vision(monkeypatch, result=([_table(GROUNDED)], 0.008, 0, 0))
    res = _extract(pdf, tmp)
    assert calls == [1] and len(res.tables) == 1


def test_page_cap_limits_vision_calls(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.extend(_Page(STATEMENT, n_tables=0) for _ in range(5))
    monkeypatch.setattr(te, "page_vision_settings", lambda: {
        "enabled": True, "min_figure_lines": 8, "max_pages": 2, "max_ungrounded": 0.1})
    calls = _patch_page_vision(monkeypatch)
    _extract(pdf, tmp)
    assert calls == [1, 2]


def test_vision_off_never_calls_page_vision(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=1))
    calls = _patch_page_vision(monkeypatch)
    res = _extract(pdf, tmp, use_vision=False)
    assert calls == [] and len(res.tables) == 1


@pytest.mark.parametrize("preset", ["models.example.yaml", "models.openai.example.yaml",
                                    "models.anthropic.example.yaml",
                                    "models.deepinfra.example.yaml"])
def test_every_preset_routes_page_vision_like_crop_vision(preset) -> None:
    import yaml

    root = Path(__file__).resolve().parents[3]
    tasks = yaml.safe_load((root / "config" / preset).read_text(encoding="utf-8"))["tasks"]
    page, crop = tasks["table_extract_page_vision"], tasks["table_extract_vision"]
    assert (page["provider"], page["model"]) == (crop["provider"], crop["model"])
    assert page["max_tokens"] >= crop.get("max_tokens", 0)


def test_page_text_uses_tight_word_gap(monkeypatch, fake_pdf) -> None:
    """At pdfplumber's default 3 pt, 44 of 148 FinQA pages ran words together."""
    pdf, pages, tmp = fake_pdf
    pages.append(_Page(STATEMENT, n_tables=0))
    _patch_page_vision(monkeypatch)
    _extract(pdf, tmp)
    assert pages[0].text_kwargs == {"x_tolerance": 1}


# ---- pages in flight (concurrency.table_page_vision) -------------------------

def _patch_slow_vision(monkeypatch, conc: int) -> dict:
    """A vision call that takes a moment and returns the grounded statement;
    records how many were in flight at once."""
    import json
    seen = {"now": 0, "max": 0, "pages": []}
    body = {"tables": [{"caption": "SOFP",
                        "columns": [{"columnIndex": 0, "columnLabel": "2025"}],
                        "rows": [{"rowIndex": 0, "rowLabel": "Cash",
                                  "cells": [{"columnIndex": 0, "cellValue": "12,345"}]}]}]}
    monkeypatch.setattr(te, "_vision_spec", lambda task, fallback=None: {"model": "m"})
    monkeypatch.setattr(te, "_render_png_b64", lambda page, bbox: "png")

    async def fake_call(spec, system, user, png):
        seen["now"] += 1
        seen["max"] = max(seen["max"], seen["now"])
        seen["pages"].append(user)
        await asyncio.sleep(0.02)
        seen["now"] -= 1
        return json.dumps(body), 0.01
    monkeypatch.setattr(te, "_call_vision", fake_call)
    monkeypatch.setattr(te, "page_vision_concurrency", lambda: conc)
    return seen


def _five_page_filing(pages: list) -> None:
    pages.extend([_Page(STATEMENT, n_tables=0), _Page("Directors' report.\n" * 10, n_tables=0),
                  _Page(STATEMENT, n_tables=1), _Page(STATEMENT, n_tables=0),
                  _Page(STATEMENT, n_tables=0)])


def test_pages_in_flight_give_identical_tables(monkeypatch, fake_pdf) -> None:
    import json
    pdf, pages, tmp = fake_pdf
    _five_page_filing(pages)
    out = {}
    for conc in (1, 4):
        seen = _patch_slow_vision(monkeypatch, conc)
        res = _extract(pdf, tmp / f"c{conc}")
        out[conc] = (json.dumps(res.tables, sort_keys=True), seen["max"], res.manifest)
    assert out[1][0] == out[4][0]
    assert out[1][1] == 1 and out[4][1] > 1, "conc 4 must overlap page calls"
    assert out[4][2]["n_page_vision_pages"] == out[1][2]["n_page_vision_pages"] == 4
    assert out[4][2]["page_vision_concurrency"] == 4
    assert out[4][2]["n_page_vision_prefetch_wasted"] == 0
    assert out[4][2]["cost_usd"] == out[1][2]["cost_usd"]


def test_prefetch_respects_the_page_cap(monkeypatch, fake_pdf) -> None:
    pdf, pages, tmp = fake_pdf
    _five_page_filing(pages)
    monkeypatch.setattr(te, "page_vision_settings", lambda: {
        "enabled": True, "min_figure_lines": 8, "max_pages": 2, "max_ungrounded": 0.1})
    seen = _patch_slow_vision(monkeypatch, 4)
    res = _extract(pdf, tmp)
    assert len(seen["pages"]) == 2 and res.manifest["n_page_vision_pages"] == 2


def test_prefetched_page_with_a_real_table_is_discarded(monkeypatch, fake_pdf) -> None:
    """The loop only sends a page whose pdfplumber tables were all strips; a
    page with a real table keeps it, and its prefetched answer is dropped."""
    import json
    pdf, pages, tmp = fake_pdf
    pages.extend([_Page(STATEMENT, n_tables=1), _Page(STATEMENT, n_tables=1)])
    monkeypatch.setattr(te, "_build_jsonld_from_grid", lambda grid, **k: _table(GROUNDED))
    monkeypatch.setattr(_Detected, "extract", lambda self: [["Cash", "12,345"], ["Inventories", "1,050"]])
    res, sent = {}, {}
    for conc in (1, 4):
        sent[conc] = _patch_slow_vision(monkeypatch, conc)["pages"]
        res[conc] = _extract(pdf, tmp / f"c{conc}")
    assert sent[4] == [], "a page with a real ruled table must not be prefetched"
    assert json.dumps(res[1].tables, sort_keys=True) == json.dumps(res[4].tables, sort_keys=True)
    assert res[1].manifest["n_page_vision_pages"] == res[4].manifest["n_page_vision_pages"] == 0
    assert len(res[4].tables) == 2


def test_page_vision_concurrency_default_and_override(monkeypatch) -> None:
    from types import SimpleNamespace
    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(app_config={}))
    assert te.page_vision_concurrency() == 1
    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(
        app_config={"concurrency": {"table_page_vision": 6}}))
    assert te.page_vision_concurrency() == 6

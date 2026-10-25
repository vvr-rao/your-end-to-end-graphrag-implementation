"""Unit tests for the evaluated summarizer building blocks (no LLM/DB): section
coverage, section-boundary size-cap splitting, token windowing, prompt/task
registration, and the configurable evidence-char cap."""

from __future__ import annotations

from pathlib import Path

from backend.app.services.evaluated_summarizer import (
    EvaluatedDocSummary,
    _cache_load,
    _cache_save,
    _line_split,
    _split_on_sections,
    _split_with_context,
    _summary_subject,
    _token_windows,
    check_section_coverage,
    evaluated_result_to_chunks,
    get_encoder,
)
from backend.app.services.prompts import (
    PROMPTS,
    REQUIRED_SECTIONS,
    _format_evidence_block,
    answer_deep_research,
)

_ENC = get_encoder()

_SECTIONS = [
    "CLAIMS", "EVIDENCE", "FINDINGS", "OBSERVATIONS", "EVENTS",
    "ENTITIES", "DATES", "REGULATORY / SAFETY / WARNINGS", "LISTS",
]


def _fake_summary(words_per_section: int = 5) -> str:
    return "\n".join(f"{h}\n" + ("word " * words_per_section) for h in _SECTIONS)


def test_section_coverage_detects_present_and_absent() -> None:
    cov = check_section_coverage("CLAIMS\nfoo\nEVIDENCE\nbar\nLISTS\nnone")
    assert cov["CLAIMS"] and cov["EVIDENCE"] and cov["LISTS"]
    assert not cov["DATES"] and not cov["EVENTS"]
    # REGULATORY/SAFETY/WARNINGS matches any of its aliases.
    assert not cov["REGULATORY_SAFETY_WARNINGS"]
    cov2 = check_section_coverage("SAFETY\nboxed warning")
    assert cov2["REGULATORY_SAFETY_WARNINGS"]


def test_all_required_sections_present() -> None:
    cov = check_section_coverage(_fake_summary())
    assert all(cov.values())
    assert len(REQUIRED_SECTIONS) == 9


def test_split_on_sections_respects_cap_and_is_lossless() -> None:
    big = "\n".join(f"{h}\n" + ("word " * 60) for h in _SECTIONS)
    cap = 50
    pieces = _split_on_sections(big, cap, _ENC)
    assert len(pieces) > 1
    # Every piece is within the cap (allow small tokenizer slack).
    assert all(len(_ENC.encode(p)) <= cap + 5 for p in pieces)
    # Lossless: every section header survives somewhere across the pieces.
    joined = "\n".join(pieces)
    for h in _SECTIONS:
        assert h.split(" /")[0] in joined


def test_split_on_sections_small_input_single_piece() -> None:
    s = "CLAIMS\nshort"
    assert _split_on_sections(s, 1000, _ENC) == [s]


def test_split_on_sections_oversized_single_section() -> None:
    # One section far bigger than the cap must be hard-split, not dropped.
    s = "CLAIMS\n" + ("word " * 200)
    pieces = _split_on_sections(s, 40, _ENC)
    assert len(pieces) >= 2
    assert all(len(_ENC.encode(p)) <= 45 for p in pieces)


_TITLE = "**LG ENERGY SOLUTION, LTD. Interim Condensed Separate Financial Statements (Q1 2025)**"
_STATEMENT = _TITLE + "\n\n" + "\n".join(
    f"*   **Line item {i}:** 13,543,750 (Mar 25) / 11,923,730 (Dec 24)" for i in range(60)
) + "\n\nCLAIMS\n- The Company reported a net loss.\n"


def test_oversized_block_is_cut_between_lines_never_inside_one() -> None:
    # The LGES statement narrative sits before the first section header; the
    # old token split cut it mid-number ("13,543" | ",750").
    lines = set(_STATEMENT.splitlines())
    pieces = _split_on_sections(_STATEMENT, 120, _ENC)
    assert len(pieces) > 3
    for p in pieces:
        assert len(_ENC.encode(p)) <= 125
        assert all(line in lines for line in p.splitlines())


def test_line_split_still_cuts_a_single_overlong_line() -> None:
    pieces = _line_split("word " * 300, 40, _ENC)
    assert len(pieces) >= 7 and all(len(_ENC.encode(p)) <= 45 for p in pieces)


def test_every_piece_carries_the_document_line_within_the_cap() -> None:
    subject = _summary_subject([_STATEMENT], Path("2025_1Q_LGES_Audit_Report_FS_en.pdf"))
    assert subject == "LG ENERGY SOLUTION, LTD. Interim Condensed Separate Financial Statements (Q1 2025)"
    pieces = _split_with_context([_STATEMENT, "CLAIMS\n- second window"], subject, 120, _ENC)
    assert all(p.startswith(f"[Document: {subject}]\n") for p in pieces)
    assert all(len(_ENC.encode(p)) <= 125 for p in pieces)
    assert pieces[-1].endswith("- second window")


def test_subject_falls_back_to_the_file_name_but_not_a_hash() -> None:
    no_title = ["CLAIMS\n- Ozempic reshaped the diabetes market."]
    assert _summary_subject(no_title, Path("SDI_Audit report_2025.02.21_eng.pdf")) == "SDI Audit report 2025.02.21 eng"
    assert _summary_subject(no_title, Path("24220c53-5b7f-417d-9a9f-c4d6267d4c11.pdf")) is None
    assert _split_with_context(no_title, None, 1000, _ENC) == [no_title[0]]


def test_cache_holds_window_summaries_and_rejoins_old_split_pieces(tmp_path) -> None:
    path = tmp_path / "doc.json"
    _cache_save(path, ["window one", "window two"])
    assert _cache_load(path, 1200, _ENC) == (["window one", "window two"], None)
    # Old format: the split pieces. Token-cut boundaries rejoin with no gap.
    cap = 60
    toks = _ENC.encode(_STATEMENT)
    legacy = [_ENC.decode(toks[i:i + cap]) for i in range(0, len(toks), cap)]
    path.write_text(__import__("json").dumps(legacy))
    (rejoined,), subject = _cache_load(path, cap, _ENC)
    assert subject is None
    assert "13,543,750" in rejoined
    assert rejoined.split() == _STATEMENT.split()


def test_token_windows_overlap() -> None:
    text = "token " * 1000
    windows = _token_windows(text, 300, 50, _ENC)
    assert len(windows) == 5  # step 250 over ~1000 tokens
    assert _token_windows("tiny", 300, 50, _ENC) == ["tiny"]


def test_summarizer_prompts_registered() -> None:
    for k in ("evaluated_summary_chunk", "summary_question_gen",
              "summary_evaluate", "summary_revise"):
        assert k in PROMPTS
    # Source is supplied out-of-band via the router's cache_prefix (a cached
    # leading block), so the builders take no source arg and reference the source
    # in the system context rather than inlining it.
    s, u = PROMPTS["evaluated_summary_chunk"]()
    assert "CLAIMS" in u and "SOURCE CHUNK provided in the system context above" in u
    _qs, qu = PROMPTS["summary_question_gen"](5)
    assert "SOURCE TEXT provided in the system context above" in qu
    _rs, ru = PROMPTS["summary_revise"]("CURRENT SUMMARY", '{"passed": false}')
    assert "ORIGINAL SOURCE CHUNK provided in the system context above" in ru
    assert "CURRENT SUMMARY" in ru


def test_evaluated_result_to_chunks() -> None:
    r = EvaluatedDocSummary(
        path=Path("/tmp/doc.txt"),
        chunk_summaries=["CLAIMS a", "EVIDENCE b", "LISTS c"],
        combined="x", summarized=True,
    )
    chunks = evaluated_result_to_chunks(r)
    assert len(chunks) == 3
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(c.source_name == "doc.txt" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)
    assert chunks[0].text == "CLAIMS a"


def test_evidence_char_cap_configurable() -> None:
    ev = [{"kind": "chunk", "iri": "x", "text": "A" * 3000}]
    assert len(_format_evidence_block(ev)) < 700              # legacy default 600
    long = _format_evidence_block(ev, 1500)
    assert 1400 < len(long) < 1700
    _, user = answer_deep_research("q", ev, 1500)
    assert user.count("A") > 1000                            # cap honored downstream


# --------------------------------------------------------------------------- #
# document_subject: one call per document, cached with the summary.
# --------------------------------------------------------------------------- #
import asyncio as _asyncio
import json as _json
import types as _types

from backend.app.services import evaluated_summarizer as _es
from backend.app.services.document_io import LoadedDocument

_LGES_SUBJECT = {
    "subject": "LG Energy Solution, Ltd. -- interim condensed separate financial statements, Q1 2025",
    "organization": "LG Energy Solution, Ltd.", "document_type": "interim financial statements",
    "period": "Q1 2025",
}


class _SubjectRouter:
    def __init__(self, reply: str | Exception, configured: bool = True) -> None:
        self.reply, self.configured, self.calls = reply, configured, []
        self.total_cost_usd = self.cache_read_tokens = self.total_input_tokens = 0

    def task_spec(self, task: str) -> dict:
        if task == "document_subject" and not self.configured:
            raise KeyError(task)
        return {"model": "m"}

    async def chat(self, task: str, *, system: str, user: str, **_: object):
        self.calls.append((task, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return _types.SimpleNamespace(text=self.reply)


def _doc(name: str = "2025_1Q_LGES_Audit_Report_FS_en.pdf") -> LoadedDocument:
    return LoadedDocument(path=Path(name), text="LG Energy Solution, Ltd. Report on review. " * 800)


def test_document_subject_reads_the_source_opening_and_file_name() -> None:
    router = _SubjectRouter(_json.dumps(_LGES_SUBJECT))
    info = _asyncio.run(_es._document_subject(router, _doc(), _ENC))
    assert info == _LGES_SUBJECT
    (task, user), = router.calls
    assert task == "document_subject"
    assert "FILE NAME: 2025_1Q_LGES_Audit_Report_FS_en.pdf" in user
    assert len(_ENC.encode(user)) < _es._SUBJECT_OPENING_TOKENS + 100   # opening only


def test_document_subject_is_optional_and_soft() -> None:
    unconfigured = _SubjectRouter("{}", configured=False)
    assert _asyncio.run(_es._document_subject(unconfigured, _doc(), _ENC)) is None
    assert unconfigured.calls == []
    for reply in (RuntimeError("timeout"), "not json", '{"subject": ""}'):
        assert _asyncio.run(_es._document_subject(_SubjectRouter(reply), _doc(), _ENC)) is None


def test_cached_summary_gets_its_subject_once(tmp_path, monkeypatch) -> None:
    # A summary cached before the subject call existed (old split-piece format,
    # no title line -- like every gpt-4.1-mini summary): one call, then cached.
    monkeypatch.setattr(_es, "_cache_dir", lambda: tmp_path)
    doc = _doc()
    key = _es._cache_key(doc.text, "m", 3, 12000, 1200)
    (tmp_path / f"{key}.json").write_text(_json.dumps(["CLAIMS\n- Revenue: 2,177,068 (2025)."]))
    router = _SubjectRouter(_json.dumps(_LGES_SUBJECT))

    def run():
        return _asyncio.run(_es.evaluated_summarize_documents_async([doc], router))[0]

    first = run()
    assert first.chunk_summaries == [f"[Document: {_LGES_SUBJECT['subject']}]\nCLAIMS\n- Revenue: 2,177,068 (2025)."]
    assert first.audit["subject"] == _LGES_SUBJECT
    saved = _json.loads((tmp_path / f"{key}.json").read_text())
    assert saved["format"] == 2 and saved["subject"] == _LGES_SUBJECT
    second = run()
    assert second.chunk_summaries == first.chunk_summaries
    assert len(router.calls) == 1


def test_without_the_task_the_file_name_is_the_subject(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_es, "_cache_dir", lambda: tmp_path)
    doc = _doc("SDI_Audit report_2025.02.21_eng.pdf")
    key = _es._cache_key(doc.text, "m", 3, 12000, 1200)
    _es._cache_save(tmp_path / f"{key}.json", ["CLAIMS\n- Net profit rose."])
    router = _SubjectRouter("{}", configured=False)
    out = _asyncio.run(_es.evaluated_summarize_documents_async([doc], router))[0]
    assert out.chunk_summaries[0].startswith("[Document: SDI Audit report 2025.02.21 eng]\n")
    assert out.audit["subject"]["source"] == "fallback" and router.calls == []

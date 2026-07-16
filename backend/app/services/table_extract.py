"""Phase 2a — extract `viao:StructuredTable` JSON-LD payloads from PDFs.

Hybrid strategy:

1. **Detect** via `pdfplumber.Page.find_tables()` — fast, free, deterministic.
2. **Score complexity** per detected table. A table is "simple" when
   pdfplumber's flat extract returns a clean rectangular grid with no
   missing cells. Complex tables (None cells, merged headers, sub-tables)
   route to the vision LLM.
3. **Build JSON-LD** per the `table_jsonld` schema. Caption hint comes
   from the line of text immediately above the table region.
4. **Validate** + drop invalid payloads with an audit-log entry. The
   pipeline must not raise on bad PDFs.

The extractor is **opt-in** at the caller's discretion. When the caller
omits `--tables` the extractor never runs and `summarize_long_documents_async`
+ chunking are unchanged.

Caller integration shape:

    from backend.app.services.table_extract import extract_tables_async

    result = await extract_tables_async(
        pdf_path,
        router=router,
        run_cache_dir=Path("output_ontologies/.../tables"),
        # OR omit run_cache_dir to use only the user cache.
    )
    # result.tables is a list of validated JSON-LD payloads.
    # result.manifest carries cost + counts for audit.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services import table_cache, table_jsonld
from backend.app.services.llm_router import (
    LLMRouter,
    _anthropic_accepts_temperature,
    _strip_json_fences,
)
from backend.app.services.llm_router import _estimate_cost as _estimate_cost_by_model
from backend.app.services.prompts import PROMPTS


# Pages with NO detectable tables incur tiny pdfplumber time; we still
# walk them. Cap on tables per doc as a runaway-cost guard.
_MAX_TABLES_PER_DOC = 400

# Memory-safety guards. The dev sandbox has a ~2.7 GiB ceiling and
# pdfplumber loads the full parse tree per page (chars, lines, rects,
# curves) -- a 150-page 10-K can easily blow past 1 GiB if state isn't
# released between pages. These guards keep extraction within budget:
#   - reject PDFs over `_MAX_PDF_BYTES`; log a soft skip
#   - stop after `_MAX_PAGES_PER_DOC` pages; log a partial-extraction warning
#   - flush each page's cache + force GC every `_GC_EVERY_N_PAGES`
_MAX_PDF_BYTES = 80 * 1024 * 1024     # 80 MB
_MAX_PAGES_PER_DOC = 300
_GC_EVERY_N_PAGES = 10

# Vision LLM defaults if config/models.yaml lacks the task entry.
_DEFAULT_VISION_MODEL = "gpt-4o-mini"
_DEFAULT_VISION_TIMEOUT = 90
_DEFAULT_VISION_TEMP = 0.0
_DEFAULT_VISION_MAX_TOKENS = 4096

# Rendering DPI for the cropped table PNG sent to vision. Lower than the
# typical 200 DPI default since cropped regions are small and 150 keeps
# the PNG well under 1 MB on financial-table widths.
_RENDER_DPI = 150


@dataclass
class TableExtractionResult:
    tables: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


# Hash chunk size for the streaming PDF hash. 1 MiB is small enough that
# the buffer churn is negligible but large enough to avoid syscall noise.
_HASH_CHUNK_BYTES = 1 * 1024 * 1024


def _hash_pdf_streaming(path: Path) -> tuple[str, str]:
    """Compute (doc_sha, cache_key) for a PDF without reading the whole
    file into memory. Mirrors `table_cache.doc_sha256` and
    `table_cache.doc_cache_key` byte-for-byte but uses a streaming read.

    Returns (doc_sha256_hex, sha256_of_version_pipe_bytes_hex)."""
    import hashlib

    h_sha = hashlib.sha256()  # raw doc sha (matches table_cache.doc_sha256)
    h_key = hashlib.sha256()  # cache key (matches table_cache.doc_cache_key)
    h_key.update(table_cache.EXTRACTOR_VERSION.encode("utf-8"))
    h_key.update(b"|")
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h_sha.update(chunk)
            h_key.update(chunk)
    return h_sha.hexdigest(), h_key.hexdigest()


# Module-level singleton OpenAI client for the vision route. Sharing one
# client across all vision calls avoids leaking a fresh connection pool
# (sockets + TLS contexts) per call -- previously this accumulated ~10
# MB per call until GC ran, contributing to OOM on 1000+-call runs.
_VISION_CLIENT: Any = None
_ANTHROPIC_VISION_CLIENT: Any = None


def _get_vision_client() -> Any:
    """Return the shared `AsyncOpenAI` client, creating it on first use."""
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        settings = get_settings()
        if not settings.openai_api_key:
            return None
        from openai import AsyncOpenAI

        _VISION_CLIENT = AsyncOpenAI(api_key=settings.openai_api_key)
    return _VISION_CLIENT


def _get_anthropic_vision_client() -> Any:
    """Return the shared `AsyncAnthropic` client for the vision route,
    creating it on first use. Used when `table_extract_vision.provider`
    is `anthropic` (fully OpenAI-free table extraction)."""
    global _ANTHROPIC_VISION_CLIENT
    if _ANTHROPIC_VISION_CLIENT is None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            return None
        from anthropic import AsyncAnthropic

        _ANTHROPIC_VISION_CLIENT = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _ANTHROPIC_VISION_CLIENT


async def close_vision_client() -> None:
    """Explicitly close the shared vision client(s) + drop the module-level
    references. Call this once after all table extraction work is done so
    the connection pool + TLS contexts get released before downstream
    phases (LLM stages) start consuming memory."""
    global _VISION_CLIENT, _ANTHROPIC_VISION_CLIENT
    clients = [c for c in (_VISION_CLIENT, _ANTHROPIC_VISION_CLIENT) if c is not None]
    _VISION_CLIENT = None
    _ANTHROPIC_VISION_CLIENT = None
    for client in clients:
        try:
            # AsyncOpenAI / AsyncAnthropic both expose an async close().
            await client.close()
        except Exception:
            pass


# ---------- Complexity heuristic ----------------------------------------------


def _is_simple_table(grid: list[list[Any]]) -> bool:
    """Return True when pdfplumber's flat extract is structurally sound
    (rectangular grid, >= 2 cols) so we can use the free pdfplumber path.

    Empty cells (None or "") are NORMAL in financial tables (a year column
    where a segment had no value) -- coerce them to "" rather than
    escalating. Earlier behaviour escalated almost every table to vision
    because of routine empty cells.

    Only escalate to vision when the grid is genuinely degenerate:
      - empty / single row
      - single column (almost always a misdetection)
      - ragged width (rows of different lengths -- suggests merged cells)
      - all cells empty (no data)"""
    if not grid or len(grid) < 2:
        return False
    width = max((len(r) for r in grid), default=0)
    if width < 2:
        return False
    non_empty_cells = 0
    for row in grid:
        if len(row) != width:
            return False
        for i, cell in enumerate(row):
            if cell is None:
                row[i] = ""
            elif isinstance(cell, str) and cell.strip():
                non_empty_cells += 1
            elif not isinstance(cell, str):
                non_empty_cells += 1
    # Require some signal: at least 30% of cells non-empty AND >= 4 total.
    if non_empty_cells < 4:
        return False
    if non_empty_cells < int(0.3 * len(grid) * width):
        return False
    return True


# ---------- Pre-extraction filters for non-data tables ----------------------
# Cheap structural checks that run BEFORE either extraction route. Catch
# headers, footers, tables-of-contents, indexes, and bibliographies --
# all of which pdfplumber's `find_tables` reports as tables but which
# carry no useful data for downstream retrieval.

# Bibliography indicator tokens. Multi-row tables with several of these
# in their cells are almost certainly a references section.
_BIB_TOKENS = re.compile(
    r"\b(?:et al\.?|vol\.?|pp\.?|ibid\.?|op\. cit\.?|"
    r"doi:|http[s]?://|ISBN|ISSN|journal|press|proceedings)\b",
    re.IGNORECASE,
)

# Page-number-ish cell values: a few digits, optional Roman numerals, dashes.
_PAGE_NUMBER_RE = re.compile(r"^\s*[ivxlcdmIVXLCDM\d\-,\s]+\s*$")

# Section / heading prefix tokens commonly used in TOCs and indexes.
_TOC_HEAD_TOKENS = re.compile(
    r"\b(?:Table of Contents|Index|Bibliography|References)\b",
    re.IGNORECASE,
)


def _bbox_filter_reason(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> str | None:
    """Reject ONLY paper-thin slivers of running text that pdfplumber
    flags as a table. Tables near the top/bottom of the page that have
    real tabular structure get a second-chance content check downstream
    (so e.g. SEC cover-page disclosure tables survive). Returns a short
    reason string when the table should be skipped; None to keep."""
    x0, top, x1, bottom = bbox
    h = max(0.0, bottom - top)
    if page_height <= 0:
        return None
    rel_h = h / page_height
    # 2% page height is paper-thin: ~16 pt on US Letter. No real data
    # table fits in that vertical space; almost always a running-text
    # band misclassified as a table.
    if rel_h < 0.02:
        return "thin-band"
    return None


def _content_filter_reason(
    grid: list[list[Any]],
    *,
    page_number: int,
    caption_hint: str | None,
    bbox: tuple[float, float, float, float] | None = None,
    page_height: float | None = None,
) -> str | None:
    """Inspect the extracted grid for TOC / index / bibliography signatures
    PLUS header/footer-shaped grids (single short row near a page edge).
    Returns a reason string to drop the table, or None to keep."""
    # Caption check first -- "Table of Contents" / "Index" / "References" /
    # "Bibliography" is the strongest signal and fires regardless of how
    # many rows pdfplumber sees.
    if caption_hint and _TOC_HEAD_TOKENS.search(caption_hint):
        return "caption-toc-index-bib"

    # Header/footer detection (content-aware): single-row or two-row
    # grids near the top/bottom of the page with only one substantive
    # cell are almost always a misclassified page header/footer line.
    # The cover-page disclosure table on an SEC 10-K survives because
    # it has >=2 substantive cells.
    if bbox is not None and page_height and page_height > 0:
        x0, top, x1, bottom = bbox
        near_top = top < page_height * 0.08
        near_bottom = (page_height - bottom) < page_height * 0.08
        if (near_top or near_bottom) and len(grid) <= 2:
            non_empty = sum(
                1 for row in grid for c in row
                if isinstance(c, str) and c.strip()
            )
            if non_empty < 3:
                return "header-band" if near_top else "footer-band"

    if not grid or len(grid) < 3:
        return None  # too small to confidently classify; pdfplumber path handles
    width = max((len(r) for r in grid), default=0)
    if width < 1:
        return None
    flat: list[str] = []
    last_col: list[str] = []
    for row in grid:
        for cell in row:
            if isinstance(cell, str):
                flat.append(cell.strip())
        if row:
            tail = row[-1]
            last_col.append(tail.strip() if isinstance(tail, str) else "")
    joined = " | ".join(flat).strip()
    if not joined:
        return "empty-cells"

    # TOC / index: last column dominated by page numbers (>= 70%).
    non_empty_last = [c for c in last_col if c]
    if non_empty_last:
        looks_pagenum = sum(
            1 for c in non_empty_last
            if c.isdigit() or _PAGE_NUMBER_RE.match(c)
        )
        if looks_pagenum / len(non_empty_last) >= 0.7 and len(non_empty_last) >= 5:
            return "toc-or-index"

    # Bibliography: many cells contain reference tokens (et al., vol., DOI,
    # http://, ISBN, etc.). Tolerates rotation pickups by pdfplumber.
    matches = sum(1 for c in flat if c and _BIB_TOKENS.search(c))
    if matches >= 4 and matches / max(1, sum(1 for c in flat if c)) >= 0.25:
        return "bibliography"

    return None


# ---------- pdfplumber path: simple flat extraction ---------------------------


def _build_jsonld_from_grid(
    grid: list[list[Any]],
    *,
    doc_sha: str,
    table_index: int,
    caption: str | None,
    page_number: int,
) -> dict[str, Any]:
    """Convert a clean rectangular grid into JSON-LD.

    Heuristic for header/row-label detection: the first row is treated
    as the column-header row. The first column is treated as a row-label
    column when its non-header cells are all non-numeric (lets the
    `pdfplumber` route cope with the common "metric name in col 0,
    numbers in col 1..N" pattern of financial tables)."""
    iri = table_jsonld.build_table_iri(doc_sha, table_index)
    payload = table_jsonld.empty_payload(
        doc_sha, table_index,
        extraction_method="pdfplumber",
        caption=caption,
        page_number=page_number,
    )

    first_row = [str(c).strip() for c in grid[0]]
    data_rows = grid[1:]
    width = len(first_row)

    # Decide whether column 0 is a row-label column.
    first_col_label_like = all(
        not _looks_numeric(str(r[0]).strip()) and str(r[0]).strip()
        for r in data_rows
    ) if data_rows else False

    col_start = 1 if first_col_label_like else 0
    columns: list[dict[str, Any]] = []
    for ci_emit, ci_raw in enumerate(range(col_start, width)):
        col_iri = table_jsonld.build_column_iri(iri, ci_emit)
        columns.append({
            "@id": col_iri,
            "@type": "viao:TableColumn",
            "columnIndex": ci_emit,
            "columnLabel": first_row[ci_raw] or None,
        })
    payload["columns"] = columns

    rows: list[dict[str, Any]] = []
    for ri, raw_row in enumerate(data_rows):
        row_iri = table_jsonld.build_row_iri(iri, ri)
        row_label = str(raw_row[0]).strip() if first_col_label_like else None
        cells: list[dict[str, Any]] = []
        for ci_emit, ci_raw in enumerate(range(col_start, width)):
            cell_iri = table_jsonld.build_cell_iri(iri, ri, ci_emit)
            cells.append({
                "@id": cell_iri,
                "@type": "viao:TableCell",
                "inColumn": columns[ci_emit]["@id"],
                "cellValue": str(raw_row[ci_raw]).strip(),
            })
        rows.append({
            "@id": row_iri,
            "@type": "viao:TableRow",
            "rowIndex": ri,
            "rowLabel": row_label,
            "isHeaderRow": False,
            "cells": cells,
        })
    payload["rows"] = rows
    return payload


def _looks_numeric(s: str) -> bool:
    if not s:
        return False
    cleaned = (
        s.replace(",", "")
         .replace("$", "")
         .replace("€", "")
         .replace("£", "")
         .replace("¥", "")
         .replace("%", "")
         .replace("(", "")
         .replace(")", "")
         .strip()
    )
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


# ---------- Caption hint: text immediately above the table -------------------


def _caption_hint(page: Any, bbox: tuple[float, float, float, float]) -> str | None:
    """Pull the line of text immediately ABOVE the table's top edge as a
    cheap caption hint. Useful for the vision prompt + as a fallback
    caption when extraction is otherwise label-less."""
    x0, top, x1, bottom = bbox
    band_top = max(0.0, top - 36.0)  # ~half-inch above
    try:
        crop = page.crop((x0, band_top, x1, top))
        text = crop.extract_text() or ""
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    # Last non-empty line of the band is most likely the caption.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    if len(candidate) > 200:
        candidate = candidate[:200].rsplit(" ", 1)[0] + " ..."
    return candidate


# ---------- Vision LLM path: cropped PNG → JSON-LD ----------------------------


async def _vision_call_openai(
    system: str,
    user: str,
    png_b64: str,
    *,
    model: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, float]:
    """OpenAI vision call. Returns (json_text, cost_usd) or (None, 0.0)."""
    client = _get_vision_client()
    if client is None:
        return None, 0.0
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format={"type": "json_object"},
        )
    except Exception:
        return None, 0.0
    text = (resp.choices[0].message.content or "").strip()
    return text, _estimate_cost(resp)


async def _vision_call_anthropic(
    system: str,
    user: str,
    png_b64: str,
    *,
    model: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, float]:
    """Anthropic vision call (base64 image content block). Returns
    (json_text, cost_usd) or (None, 0.0). Anthropic has no
    response_format=json_object, so we instruct bare JSON in the system
    prompt and strip any ```json fences before returning."""
    client = _get_anthropic_vision_client()
    if client is None:
        return None, 0.0
    sys_prompt = (
        f"{system}\n\nOutput only a single valid JSON object. "
        "Do not include any prose, explanation, or markdown code fences."
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": sys_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": png_b64,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
        "timeout": timeout,
    }
    # Opus 4.6+/Fable/Mythos 400 on sampling params; Sonnet/Haiku accept them.
    if _anthropic_accepts_temperature(model):
        kwargs["temperature"] = temperature
    try:
        resp = await client.messages.create(**kwargs)
    except Exception:
        return None, 0.0
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    text = _strip_json_fences(text)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", None) if usage else None
    out_tok = getattr(usage, "output_tokens", None) if usage else None
    cost = _estimate_cost_by_model(model, in_tok, out_tok) or 0.0
    return text, cost


async def _extract_table_via_vision(
    page: Any,
    bbox: tuple[float, float, float, float],
    *,
    doc_sha: str,
    table_index: int,
    page_number: int,
    caption_hint: str | None,
    router: LLMRouter,
) -> tuple[dict[str, Any] | None, float]:
    """Render the table region to PNG, send to gpt-4o-mini vision, parse
    JSON back into a StructuredTable payload. Returns (payload, cost_usd)
    or (None, 0.0) on any failure (caller logs + drops)."""
    settings = get_settings()

    spec = settings.models_config.get("tasks", {}).get("table_extract_vision", {})
    provider = str(spec.get("provider", "openai")).lower()
    model = spec.get("model", _DEFAULT_VISION_MODEL)
    timeout = int(spec.get("timeout", _DEFAULT_VISION_TIMEOUT))
    temperature = float(spec.get("temperature", _DEFAULT_VISION_TEMP))
    max_tokens = int(spec.get("max_tokens", _DEFAULT_VISION_MAX_TOKENS))

    # Render the cropped region to PNG bytes (shared across providers).
    try:
        img = page.crop(bbox).to_image(resolution=_RENDER_DPI)
        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        png_b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
    except Exception:
        return None, 0.0

    system, user = PROMPTS["table_extract_vision"](
        page_number=page_number, caption_hint=caption_hint,
    )

    # Dispatch to the provider configured in models.yaml. Both helpers
    # return (json_text, cost_usd) or (None, 0.0) on any failure. The
    # anthropic path keeps table extraction fully OpenAI-free.
    if provider == "anthropic":
        text, cost = await _vision_call_anthropic(
            system, user, png_b64, model=model, timeout=timeout,
            temperature=temperature, max_tokens=max_tokens,
        )
    else:
        text, cost = await _vision_call_openai(
            system, user, png_b64, model=model, timeout=timeout,
            temperature=temperature, max_tokens=max_tokens,
        )
    if text is None:
        return None, 0.0

    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None, 0.0

    # The detector hands vision crops that contain no table (prose, page
    # fragments, wide multi-column layouts). The prompt used to assert "this
    # image contains a table" with no way to disagree, so the model complied the
    # only way it could -- by inventing one. Audit of 46 vision tables: 17%
    # fabricated outright (an IEA EV-policy page yielding an "EPA | Total Budget
    # | Employees" table), mean 53% of numbers not present on their own page.
    # pdfplumber over the same doc: 0% fabricated. Fabrications are worse than
    # noise -- they are well-formed, embedded, and citable as fact.
    if not isinstance(body, dict) or body.get("no_table") is True:
        return None, cost

    payload = _vision_body_to_jsonld(
        body,
        doc_sha=doc_sha,
        table_index=table_index,
        page_number=page_number,
        caption_hint=caption_hint,
    )
    if payload is not None and _looks_fabricated(payload):
        return None, cost
    return payload, cost


# Placeholder tokens a model reaches for when asked to describe a table it
# cannot actually read. Real financial/policy tables do not label their rows
# "Category A". Backstop for models that ignore the no_table instruction.
_PLACEHOLDER_LABELS = (
    "category a", "category b", "category c",
    "column 1", "column 2", "column 3",
    "row 1", "row 2", "row 3",
    "item 1", "item 2",
    "value 1", "value 2",
    "example", "placeholder", "lorem ipsum",
)


def _looks_fabricated(payload: dict[str, Any]) -> bool:
    """True when an extracted table shows the signature of invented content."""
    labels: list[str] = []
    for col in payload.get("columns", []) or []:
        v = col.get("columnLabel")
        if isinstance(v, str):
            labels.append(v.strip().lower())
    for row in payload.get("rows", []) or []:
        v = row.get("rowLabel")
        if isinstance(v, str):
            labels.append(v.strip().lower())
        for cell in row.get("cells", []) or []:
            cv = cell.get("cellValue")
            if isinstance(cv, str):
                labels.append(cv.strip().lower())
    if not labels:
        return False
    hits = sum(1 for lab in labels if lab in _PLACEHOLDER_LABELS)
    # Two independent placeholder tokens is not a coincidence.
    return hits >= 2


def _vision_body_to_jsonld(
    body: dict[str, Any],
    *,
    doc_sha: str,
    table_index: int,
    page_number: int,
    caption_hint: str | None,
) -> dict[str, Any] | None:
    """Promote the vision LLM's JSON output (which uses short keys for
    token economy) to a full JSON-LD payload. Returns None on shape
    mismatch."""
    if not isinstance(body, dict):
        return None
    iri = table_jsonld.build_table_iri(doc_sha, table_index)
    payload = table_jsonld.empty_payload(
        doc_sha, table_index,
        extraction_method="vision-llm",
        caption=body.get("caption") or caption_hint,
        page_number=page_number,
    )
    cols_in = body.get("columns") or []
    rows_in = body.get("rows") or []
    if not isinstance(cols_in, list) or not isinstance(rows_in, list):
        return None
    columns: list[dict[str, Any]] = []
    for c in cols_in:
        if not isinstance(c, dict):
            continue
        ci = c.get("columnIndex")
        if not isinstance(ci, int):
            continue
        columns.append({
            "@id": table_jsonld.build_column_iri(iri, ci),
            "@type": "viao:TableColumn",
            "columnIndex": ci,
            "columnLabel": (c.get("columnLabel") or None),
        })
    col_iri_by_index = {c["columnIndex"]: c["@id"] for c in columns}
    payload["columns"] = columns

    rows: list[dict[str, Any]] = []
    for r in rows_in:
        if not isinstance(r, dict):
            continue
        ri = r.get("rowIndex")
        if not isinstance(ri, int):
            continue
        cells_in = r.get("cells") or []
        cells: list[dict[str, Any]] = []
        for c in cells_in:
            if not isinstance(c, dict):
                continue
            ci = c.get("columnIndex")
            if ci not in col_iri_by_index:
                # Drop cells that reference a non-existent column rather
                # than fabricate one — keeps the payload internally
                # consistent for the validator.
                continue
            cells.append({
                "@id": table_jsonld.build_cell_iri(iri, ri, ci),
                "@type": "viao:TableCell",
                "inColumn": col_iri_by_index[ci],
                "cellValue": "" if c.get("cellValue") is None else str(c.get("cellValue")),
            })
        rows.append({
            "@id": table_jsonld.build_row_iri(iri, ri),
            "@type": "viao:TableRow",
            "rowIndex": ri,
            "rowLabel": r.get("rowLabel") or None,
            "isHeaderRow": bool(r.get("isHeaderRow", False)),
            "cells": cells,
        })
    payload["rows"] = rows
    return payload


def _estimate_cost(resp: Any) -> float:
    """Best-effort cost estimate for the vision call from the OpenAI
    response object. Mirrors the routing in `llm_router._estimate_cost`
    but image input tokens follow the standard text-input price band on
    gpt-4o-mini (no separate image-token billing tier at this scale)."""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    # gpt-4o-mini: $0.15 / M input, $0.60 / M output (2026 pricing).
    return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000


# ---------- Top-level entry point --------------------------------------------


async def extract_tables_async(
    pdf_path: Path,
    *,
    router: LLMRouter,
    run_cache_dir: Path | None = None,
    use_vision: bool = True,
    user_cache_dir: Path | None = None,
    max_pages: int | None = None,
) -> TableExtractionResult:
    """Extract every detectable table from `pdf_path` and return them as
    JSON-LD payloads + an audit manifest.

    Cache-first: returns the cached payload list when the disk cache
    has an entry under the (doc_bytes + extractor_version) hash. Skips
    non-PDF inputs silently. Soft-fails on any per-table error; surviving
    tables still return."""
    started = time.monotonic()
    p = Path(pdf_path)
    if p.suffix.lower() != ".pdf" or not p.is_file():
        return TableExtractionResult(
            tables=[],
            manifest={"reason": "not-a-pdf-or-missing", "path": str(p)},
        )

    # Memory-safety guard: refuse to open very large PDFs. pdfplumber
    # loads the full parse tree; on a 2.7 GiB sandbox a 100+ MB PDF can
    # OOM the process. Caller gets an audit-friendly skip.
    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes > _MAX_PDF_BYTES:
        return TableExtractionResult(
            tables=[],
            manifest={
                "source": "skipped",
                "reason": f"pdf too large ({size_bytes / 1024 / 1024:.1f} MB > "
                          f"{_MAX_PDF_BYTES / 1024 / 1024:.0f} MB cap)",
                "size_bytes": size_bytes,
            },
        )

    # Hash the PDF in a streaming pass instead of reading it all into
    # memory at once. A 30 MB PDF would otherwise allocate 30 MB on the
    # Python heap that doesn't get returned to the OS even after `del`
    # -- over 17 sequential PDFs that churns hundreds of MB.
    doc_sha, cache_key = _hash_pdf_streaming(p)

    # User-cache fallback dir (None = don't write user cache; default = use it).
    if user_cache_dir is None:
        user_cache_dir = table_cache.user_cache_dir()

    hit = table_cache.two_tier_load(run_cache_dir, user_cache_dir, cache_key)
    if hit is not None:
        # When the hit came from the user cache but a run cache is
        # configured, copy the payload into the run cache so downstream
        # passes (e.g. table_ontology_mining) that read run-folder files
        # find a complete record of every PDF processed by this run.
        # Skipped silently if the run-cache file already exists or write
        # fails (cache writes are always best-effort).
        if run_cache_dir is not None:
            from pathlib import Path as _Path
            run_file = _Path(run_cache_dir) / f"{cache_key}.jsonld"
            if not run_file.exists():
                try:
                    table_cache.save(
                        run_cache_dir, cache_key,
                        doc_sha=hit.doc_sha,
                        doc_path=str(p),
                        tables=hit.tables,
                        manifest=hit.manifest,
                    )
                except Exception:
                    pass
        return TableExtractionResult(
            tables=hit.tables,
            manifest={**hit.manifest, "source": "cache", "doc_sha": hit.doc_sha},
        )

    # Cold path: real extraction.
    try:
        import pdfplumber
    except ImportError as exc:
        return TableExtractionResult(
            tables=[],
            manifest={"reason": f"pdfplumber missing: {exc}"},
        )

    import gc

    payloads: list[dict[str, Any]] = []
    n_pdfplumber = 0
    n_vision = 0
    n_dropped = 0
    cost_usd = 0.0
    table_index = 0
    pages_processed = 0
    pages_truncated = False

    n_skipped_filter = 0
    skip_reasons: dict[str, int] = {}

    try:
        with pdfplumber.open(str(p)) as pdf:
            n_pages = len(pdf.pages)
            page_cap = max_pages if max_pages is not None else _MAX_PAGES_PER_DOC
            print(
                f"[tables] {p.name}: starting "
                f"({n_pages} pages, cap {min(n_pages, page_cap)}, "
                f"vision={'ON' if use_vision else 'OFF'})",
                flush=True,
            )
            last_progress_t = time.monotonic()
            for page_no, page in enumerate(pdf.pages, start=1):
                if table_index >= _MAX_TABLES_PER_DOC:
                    break
                if page_no > page_cap:
                    pages_truncated = True
                    break
                page_w = float(getattr(page, "width", 0.0) or 0.0)
                page_h = float(getattr(page, "height", 0.0) or 0.0)
                try:
                    tables = page.find_tables() or []
                except Exception:
                    tables = []
                page_kept_before = len(payloads)
                page_skipped_filter = 0
                for tbl in tables:
                    if table_index >= _MAX_TABLES_PER_DOC:
                        break
                    bbox = tbl.bbox

                    # Filter 1: bbox geometry (header/footer/thin bands)
                    reason = _bbox_filter_reason(bbox, page_w, page_h)
                    if reason:
                        n_skipped_filter += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        page_skipped_filter += 1
                        table_index += 1
                        continue

                    try:
                        grid = tbl.extract() or []
                    except Exception:
                        grid = []
                    caption = _caption_hint(page, bbox)

                    # Filter 2: content patterns (TOC/index/bibliography +
                    # content-aware header/footer detection)
                    reason = _content_filter_reason(
                        grid, page_number=page_no, caption_hint=caption,
                        bbox=bbox, page_height=page_h,
                    )
                    if reason:
                        n_skipped_filter += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        page_skipped_filter += 1
                        table_index += 1
                        continue

                    payload: dict[str, Any] | None = None
                    if _is_simple_table(grid):
                        payload = _build_jsonld_from_grid(
                            grid,
                            doc_sha=doc_sha,
                            table_index=table_index,
                            caption=caption,
                            page_number=page_no,
                        )
                        n_pdfplumber += 1
                    elif use_vision:
                        payload, vision_cost = await _extract_table_via_vision(
                            page,
                            bbox,
                            doc_sha=doc_sha,
                            table_index=table_index,
                            page_number=page_no,
                            caption_hint=caption,
                            router=router,
                        )
                        cost_usd += vision_cost
                        if payload is not None:
                            n_vision += 1
                        else:
                            n_dropped += 1

                    if payload is None:
                        table_index += 1
                        continue

                    errors = table_jsonld.validate_table_jsonld(payload)
                    if errors:
                        # Invalid payload -- drop with an audit-friendly note
                        # (caller writes the audit line).
                        n_dropped += 1
                        table_index += 1
                        continue

                    payloads.append(payload)
                    table_index += 1

                # Per-page memory release. pdfplumber caches a lot of
                # geometric primitives on each Page; without an explicit
                # flush, those stay live until the `with` block exits --
                # which on a 150-page 10-K accumulates to >1 GiB.
                try:
                    page.flush_cache()
                    page.close()
                except Exception:
                    pass
                pages_processed += 1
                if pages_processed % _GC_EVERY_N_PAGES == 0:
                    gc.collect()

                # Progress log: every 10 pages OR every 20 seconds, whichever
                # comes first. Tight enough to see liveness on a 200-page
                # PDF but not so chatty it dominates the log.
                now = time.monotonic()
                kept_this_page = len(payloads) - page_kept_before
                show_progress = (
                    kept_this_page > 0
                    or page_skipped_filter > 0
                    or pages_processed % 20 == 0
                    or (now - last_progress_t) >= 20.0
                )
                if show_progress:
                    print(
                        f"[tables]   {p.name} p{page_no}/{n_pages}: "
                        f"+{kept_this_page} kept, "
                        f"{page_skipped_filter} filtered "
                        f"(cumulative: kept={len(payloads)} "
                        f"pdfplumber={n_pdfplumber} vision={n_vision} "
                        f"filter-skip={n_skipped_filter} "
                        f"vision-drop={n_dropped}, "
                        f"cost=${cost_usd:.4f})",
                        flush=True,
                    )
                    last_progress_t = now
    except Exception as exc:
        return TableExtractionResult(
            tables=payloads,
            manifest={
                "reason": f"pdfplumber-open-failed: {type(exc).__name__}: {exc}",
                "n_tables": len(payloads),
                "n_pdfplumber": n_pdfplumber,
                "n_vision_llm": n_vision,
                "n_dropped": n_dropped,
                "cost_usd": round(cost_usd, 5),
                "wall_seconds": round(time.monotonic() - started, 3),
                "doc_sha": doc_sha,
                "pages_processed": pages_processed,
            },
        )
    finally:
        gc.collect()

    wall = round(time.monotonic() - started, 3)
    manifest = {
        "source": "fresh",
        "doc_sha": doc_sha,
        "n_tables": len(payloads),
        "n_pdfplumber": n_pdfplumber,
        "n_vision_llm": n_vision,
        "n_dropped": n_dropped,
        "n_skipped_filter": n_skipped_filter,
        "skip_reasons": skip_reasons,
        "cost_usd": round(cost_usd, 5),
        "wall_seconds": wall,
        "pages_processed": pages_processed,
        "pages_truncated": pages_truncated,
    }
    print(
        f"[tables] {p.name}: DONE -- {len(payloads)} kept "
        f"(pdfplumber={n_pdfplumber}, vision={n_vision}, "
        f"filter-skip={n_skipped_filter}, vision-drop={n_dropped}) "
        f"over {pages_processed} pages in {wall:.1f}s, cost=${cost_usd:.4f}",
        flush=True,
    )

    # Persist to both cache tiers (run cache + user cache).
    for tier_dir in (run_cache_dir, user_cache_dir):
        if tier_dir is not None:
            try:
                table_cache.save(
                    tier_dir, cache_key,
                    doc_sha=doc_sha,
                    doc_path=str(p),
                    tables=payloads,
                    manifest=manifest,
                )
            except Exception:
                # Cache writes are best-effort; never propagate.
                pass

    return TableExtractionResult(tables=payloads, manifest=manifest)


async def extract_tables_for_paths_subprocess(
    pdf_paths: list[Path],
    *,
    run_cache_dir: Path | None = None,
    use_vision: bool = True,
    concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    """Phase 2a v2: extract tables from each given PDF by spawning
    `python -m backend.app.services.table_extract_worker` per PDF.

    Same memory-isolation story as `extract_tables_for_folder_subprocess`
    but operates over a caller-supplied list of paths -- useful when the
    caller has already filtered to "PDFs in this batch" (e.g.
    register-documents post-dup-check)."""
    pdfs = [Path(p) for p in pdf_paths if Path(p).suffix.lower() == ".pdf"]
    if not pdfs:
        return {}

    print(
        f"[tables/worker] scanning {len(pdfs)} PDF(s) "
        f"(concurrency={concurrency}, vision={'ON' if use_vision else 'OFF'}, "
        f"isolation=subprocess)",
        flush=True,
    )
    return await _drive_subprocess_workers(
        pdfs,
        run_cache_dir=run_cache_dir,
        use_vision=use_vision,
        concurrency=concurrency,
    )


async def extract_tables_for_folder_subprocess(
    folder: Path,
    *,
    run_cache_dir: Path | None = None,
    use_vision: bool = True,
    limit: int | None = None,
    concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    """Phase 2a v2: extract tables from every PDF in `folder` by spawning
    `python -m backend.app.services.table_extract_worker` per PDF.

    Why subprocess: the in-process extractor accumulates heap
    fragmentation on constrained hosts (~2.7 GiB sandbox), reliably
    OOMing partway through the corpus. A fresh process per PDF
    guarantees the kernel reclaims all per-PDF memory unconditionally
    when each worker exits.

    Returns a mapping of `str(pdf_path) -> manifest`, where each
    manifest is whatever the worker wrote to the cache file. The
    returned dict only carries lightweight manifests -- the full
    JSON-LD payloads stay on disk in the run-cache + user-cache.

    `concurrency=1` is the safe default; raise it on roomier hosts."""
    folder = Path(folder)
    pdfs = sorted(folder.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    if not pdfs:
        return {}

    print(
        f"[tables/worker] scanning {len(pdfs)} PDF(s) under {folder} "
        f"(concurrency={concurrency}, vision={'ON' if use_vision else 'OFF'}, "
        f"isolation=subprocess)",
        flush=True,
    )
    return await _drive_subprocess_workers(
        pdfs,
        run_cache_dir=run_cache_dir,
        use_vision=use_vision,
        concurrency=concurrency,
    )


async def _drive_subprocess_workers(
    pdfs: list[Path],
    *,
    run_cache_dir: Path | None,
    use_vision: bool,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Shared worker-driving loop used by both folder + paths variants."""
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0
    folder_started = time.monotonic()
    summary_stats = {"n_tables": 0, "cost_usd": 0.0, "n_cached": 0,
                     "n_failed": 0}
    results: dict[str, dict[str, Any]] = {}

    cmd_base = [sys.executable, "-u", "-m",
                "backend.app.services.table_extract_worker"]

    async def _one(p: Path) -> tuple[str, dict[str, Any]]:
        nonlocal done
        async with sem:
            cmd = list(cmd_base) + ["--pdf", str(p.resolve())]
            if run_cache_dir is not None:
                cmd += ["--run-cache-dir", str(Path(run_cache_dir).resolve())]
            if not use_vision:
                cmd += ["--no-vision"]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except Exception as exc:  # noqa: BLE001
                done += 1
                summary_stats["n_failed"] += 1
                print(
                    f"[tables/worker] spawn failed for {p.name}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                return str(p), {
                    "source": "spawn-failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            # Stream worker output through to parent stdout so the user
            # still sees per-PDF progress lines from the worker.
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                # The worker prefixes its own lines with [tables] /
                # [worker]; pass through verbatim.
                sys.stdout.write(line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            rc = await proc.wait()
            done += 1

            # Look up the just-written manifest from the user-cache. We
            # need to re-compute the cache key to find it.
            try:
                doc_sha, cache_key = _hash_pdf_streaming(p)
            except Exception:
                cache_key = None
                doc_sha = None

            manifest: dict[str, Any] = {"exit_code": rc}
            if cache_key is not None:
                from backend.app.services import table_cache  # local import
                hit = table_cache.two_tier_load(
                    run_cache_dir, table_cache.user_cache_dir(), cache_key,
                )
                if hit is not None:
                    manifest.update(hit.manifest)
                    manifest["doc_sha"] = hit.doc_sha
                    summary_stats["n_tables"] += int(
                        hit.manifest.get("n_tables", 0) or 0
                    )
                    summary_stats["cost_usd"] += float(
                        hit.manifest.get("cost_usd", 0.0) or 0.0
                    )
                    if hit.manifest.get("source") == "cache":
                        summary_stats["n_cached"] += 1
            if rc != 0:
                summary_stats["n_failed"] += 1
                manifest.setdefault("source", "worker-failed")
            print(
                f"[tables/worker] folder progress: {done}/{len(pdfs)} done "
                f"({time.monotonic() - folder_started:.0f}s, "
                f"running totals: tables={summary_stats['n_tables']} "
                f"cost=${summary_stats['cost_usd']:.4f} "
                f"failures={summary_stats['n_failed']})",
                flush=True,
            )
            return str(p), manifest

    pairs = await asyncio.gather(*[_one(p) for p in pdfs])
    for k, v in pairs:
        results[k] = v
    print(
        f"[tables/worker] folder DONE: {summary_stats['n_tables']} tables "
        f"across {len(pdfs)} PDF(s) "
        f"({summary_stats['n_cached']} cache-hits, "
        f"{summary_stats['n_failed']} failed), "
        f"{time.monotonic() - folder_started:.0f}s, "
        f"cost=${summary_stats['cost_usd']:.4f}",
        flush=True,
    )
    return results


async def extract_tables_for_folder_async(
    folder: Path,
    *,
    router: LLMRouter,
    run_cache_dir: Path | None = None,
    use_vision: bool = True,
    limit: int | None = None,
    concurrency: int = 1,
) -> dict[str, TableExtractionResult]:
    """Convenience wrapper: extract tables from every PDF under `folder`.

    Returns a mapping of doc-path-str -> TableExtractionResult. Used by
    the prune-expand integration to batch-process a corpus folder."""
    folder = Path(folder)
    pdfs = sorted(folder.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    if not pdfs:
        return {}

    print(
        f"[tables] folder scan: {len(pdfs)} PDF(s) under {folder} "
        f"(concurrency={concurrency}, vision={'ON' if use_vision else 'OFF'})",
        flush=True,
    )

    import gc

    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0
    folder_started = time.monotonic()
    pairs: list[tuple[str, TableExtractionResult]] = []
    # Aggregate counters; we DO NOT keep every TableExtractionResult in
    # memory because each holds the full tables list (sometimes MBs of
    # JSON-LD). Manifests are persisted to disk anyway.
    summary_stats = {"n_tables": 0, "cost_usd": 0.0, "n_cached": 0}

    async def _one(p: Path) -> tuple[str, TableExtractionResult]:
        nonlocal done
        async with sem:
            r = await extract_tables_async(
                p, router=router,
                run_cache_dir=run_cache_dir,
                use_vision=use_vision,
            )
            done += 1
            # Update running totals, then DROP the heavy fields so the
            # caller's dict doesn't pin them in memory while later PDFs
            # are still being processed. The full payloads are already
            # on disk in the run-cache + user-cache; rehydrate from
            # there in downstream code instead.
            summary_stats["n_tables"] += len(r.tables)
            summary_stats["cost_usd"] += float(r.manifest.get("cost_usd", 0.0) or 0.0)
            if r.manifest.get("source") == "cache":
                summary_stats["n_cached"] += 1
            slim = TableExtractionResult(tables=[], manifest=dict(r.manifest))
            r.tables = []
            gc.collect()
            print(
                f"[tables] folder progress: {done}/{len(pdfs)} PDF(s) done "
                f"({time.monotonic() - folder_started:.0f}s elapsed, "
                f"running totals: {summary_stats['n_tables']} tables, "
                f"${summary_stats['cost_usd']:.4f})",
                flush=True,
            )
            return str(p), slim

    pairs = await asyncio.gather(*[_one(p) for p in pdfs])
    print(
        f"[tables] folder DONE: {summary_stats['n_tables']} tables across "
        f"{len(pdfs)} PDF(s) ({summary_stats['n_cached']} cache-hits), "
        f"{time.monotonic() - folder_started:.0f}s, "
        f"cost=${summary_stats['cost_usd']:.4f}",
        flush=True,
    )
    return dict(pairs)

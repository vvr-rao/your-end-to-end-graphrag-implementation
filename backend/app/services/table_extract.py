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

    Returns (doc_sha256_hex, sha256_of_key_prefix_plus_bytes_hex)."""
    import hashlib

    h_sha = hashlib.sha256()  # raw doc sha (matches table_cache.doc_sha256)
    h_key = hashlib.sha256()  # cache key (matches table_cache.doc_cache_key)
    h_key.update(table_cache.key_prefix())
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
_DEEPINFRA_VISION_CLIENT: Any = None

# Providers the vision route can call. Anything else is a config error: it
# used to fall through to the OpenAI client with a non-OpenAI model id, every
# call failed, and the tables were silently dropped (2026-10-17 deepinfra
# benchmark lost all 35 vision tables).
_VISION_PROVIDERS = ("openai", "anthropic", "deepinfra")


class VisionCallError(Exception):
    """The vision request itself failed (transport, auth, unknown model) --
    as opposed to the model answering that the crop holds no table."""


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


def _get_deepinfra_vision_client() -> Any:
    """Return the shared DeepInfra client (OpenAI-compatible endpoint) for
    the vision route, creating it on first use."""
    global _DEEPINFRA_VISION_CLIENT
    if _DEEPINFRA_VISION_CLIENT is None:
        settings = get_settings()
        if not settings.deepinfra_api_key:
            return None
        from openai import AsyncOpenAI

        from backend.app.services.llm_router import DEEPINFRA_BASE_URL

        _DEEPINFRA_VISION_CLIENT = AsyncOpenAI(
            api_key=settings.deepinfra_api_key, base_url=DEEPINFRA_BASE_URL,
        )
    return _DEEPINFRA_VISION_CLIENT


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
    global _VISION_CLIENT, _ANTHROPIC_VISION_CLIENT, _DEEPINFRA_VISION_CLIENT
    clients = [c for c in (_VISION_CLIENT, _ANTHROPIC_VISION_CLIENT, _DEEPINFRA_VISION_CLIENT)
               if c is not None]
    _VISION_CLIENT = None
    _ANTHROPIC_VISION_CLIENT = None
    _DEEPINFRA_VISION_CLIENT = None
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

# Page-number-ish cell values: a bare integer of at most 4 digits, a Roman
# numeral, or a hyphenated range of either ("12-15", "iv-vi").
#
# Deliberately does NOT admit thousands separators or decimals. The earlier
# pattern (`[ivxlcdmIVXLCDM\d\-,\s]+`) accepted any run of digits, commas and
# spaces, so a financial table whose last column held plain figures --
# '3,324', '2,667', '1,363' -- read as a table of contents and was dropped.
# That never fired while single-row fragments were being discarded upstream;
# reassembling them into whole tables exposed it. A real page number is a
# small integer; a comma-grouped figure is data.
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:[ivxlcdm]+|\d{1,4})(?:\s*[-–]\s*(?:[ivxlcdm]+|\d{1,4}))?\s*$",  # noqa: RUF001
    re.IGNORECASE,
)

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
        # `_PAGE_NUMBER_RE` already covers bare digits; testing `isdigit()`
        # separately would re-admit the long unseparated figures ('48047')
        # that the tightened pattern exists to exclude.
        looks_pagenum = sum(
            1 for c in non_empty_last if _PAGE_NUMBER_RE.match(c)
        )
        if looks_pagenum / len(non_empty_last) >= 0.7 and len(non_empty_last) >= 5:
            return "toc-or-index"

    # Bibliography: many cells contain reference tokens (et al., vol., DOI,
    # http://, ISBN, etc.). Tolerates rotation pickups by pdfplumber.
    matches = sum(1 for c in flat if c and _BIB_TOKENS.search(c))
    if matches >= 4 and matches / max(1, sum(1 for c in flat if c)) >= 0.25:
        return "bibliography"

    return None


# ---------- Fragment merging: reassemble row-per-candidate tables ------------
# SEC-style filings underline each row of figures and draw no outer box and no
# vertical rules. A line-based detector therefore has no evidence that
# consecutive ruled rows belong to one table, so it reports each row as its own
# single-row candidate about 15 pt tall -- just under the 2% thin-band guard,
# which then drops all of them. Measured over a 148-PDF SEC corpus: 62 of 83
# candidates were single-row, 73.5% fell under the guard, and 137 documents
# yielded no table at all.
#
# Lowering the guard would be the wrong fix -- it would admit exactly the
# running-text bands the guard exists to reject. Instead reassemble the
# fragments BEFORE filtering: consecutive single-row candidates that share
# column boundaries and sit within a normal line gap are one table.
#
# The merge deliberately only ever combines SINGLE-ROW fragments. Any candidate
# that already extracts >= 2 rows passes through untouched, so corpora whose
# tables are fully ruled (drug labels, boxed 10-K tables) are unaffected.
_MERGE_X_TOL = 3.0        # column-boundary alignment tolerance, pt
_MERGE_GAP_TOL = 30.0     # max vertical gap between stacked fragments, pt
_HEADER_BAND_PT = 60.0    # how far above a merged table to look for headers
_HEADER_CLEARANCE_PT = 2.0  # lift the caption band clear of the header line


def _column_bounds(table: Any) -> list[float] | None:
    """The x edges delimiting each column of `table`'s first row, as
    `len(columns) + 1` ascending floats. Returns None when the row carries
    a merged (None) cell, since then the column grid is ambiguous and the
    candidate must not be merged with anything."""
    try:
        cells = table.rows[0].cells
    except (AttributeError, IndexError):
        return None
    if not cells or any(c is None for c in cells):
        return None
    try:
        bounds = [float(c[0]) for c in cells]
        bounds.append(float(cells[-1][2]))
    except (TypeError, IndexError, ValueError):
        return None
    return bounds


def _bounds_match(a: list[float] | None, b: list[float] | None) -> bool:
    """True when two candidates share the same column grid within tolerance.
    Stricter than comparing column COUNTS: two unrelated tables that happen
    to have the same number of columns will not be merged unless their
    column edges actually line up."""
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(x - y) <= _MERGE_X_TOL for x, y in zip(a, b, strict=True))


def _merge_row_fragments(
    tables: list[Any],
) -> list[tuple[tuple[float, float, float, float], list[list[Any]],
               list[float] | None, int]]:
    """Group consecutive single-row candidates into whole tables.

    Returns `(bbox, grid, col_bounds, n_fragments)` per surviving candidate.
    `n_fragments > 1` marks an assembled table -- those have no header row
    (the header sits above the ruled region, outside every fragment), which
    is why the caller recovers one via `_recover_header_row`."""
    entries: list[dict[str, Any]] = []
    for tbl in tables:
        try:
            grid = tbl.extract() or []
        except Exception:
            grid = []
        single = len(grid) == 1
        entries.append({
            "bbox": tuple(float(v) for v in tbl.bbox),
            "grid": grid,
            # Column geometry is only needed for merge decisions, so skip
            # the work for candidates that can never be merged.
            "bounds": _column_bounds(tbl) if single else None,
            "single": single,
        })
    entries.sort(key=lambda e: e["bbox"][1])

    out: list[tuple[tuple[float, float, float, float], list[list[Any]],
                    list[float] | None, int]] = []
    i = 0
    while i < len(entries):
        e = entries[i]
        if not e["single"] or e["bounds"] is None:
            out.append((e["bbox"], e["grid"], e["bounds"], 1))
            i += 1
            continue
        group = [e]
        j = i + 1
        while j < len(entries):
            nxt = entries[j]
            if not nxt["single"] or nxt["bounds"] is None:
                break
            # Fragments are stacked top-to-bottom; a small negative gap is
            # tolerated because adjacent rules can share a y coordinate.
            gap = nxt["bbox"][1] - group[-1]["bbox"][3]
            if gap > _MERGE_GAP_TOL or gap < -1.0:
                break
            if not _bounds_match(group[-1]["bounds"], nxt["bounds"]):
                break
            group.append(nxt)
            j += 1
        if len(group) == 1:
            out.append((e["bbox"], e["grid"], e["bounds"], 1))
            i += 1
            continue
        merged_bbox = (
            min(g["bbox"][0] for g in group),
            min(g["bbox"][1] for g in group),
            max(g["bbox"][2] for g in group),
            max(g["bbox"][3] for g in group),
        )
        merged_grid = [row for g in group for row in g["grid"]]
        out.append((merged_bbox, merged_grid, group[0]["bounds"], len(group)))
        i = j
    return out


def _recover_header_row(
    page: Any,
    bbox: tuple[float, float, float, float],
    col_bounds: list[float],
) -> tuple[list[str], float] | None:
    """Read column labels out of the text band immediately above a merged
    table.

    Fragment merging assembles DATA rows only -- in an unboxed filing the
    column header ("(in millions) 2018 2017 2016") sits above the first
    rule, so it belongs to no fragment. Without this the first data row
    would be misread as the header and its values lost.

    Words on the bottom-most text line of the band are bucketed into
    columns by their horizontal midpoint.

    Returns `(labels, header_top)` -- the second element is the y of the
    top of the header line, which the caller needs so `_caption_hint`
    can look for the caption ABOVE the header instead of returning the
    header itself. Returns None when nothing usable is above the table."""
    x0, top, x1, _bottom = bbox
    band_top = max(0.0, top - _HEADER_BAND_PT)
    if band_top >= top:
        return None
    try:
        words = page.crop((x0, band_top, x1, top)).extract_words() or []
    except Exception:
        return None
    if not words:
        return None
    # The header is the line resting directly on the table, not the prose
    # further up the band.
    bottom_most = max(float(w["bottom"]) for w in words)
    line = [w for w in words if bottom_most - float(w["bottom"]) <= 3.0]
    if not line:
        return None

    n_cols = len(col_bounds) - 1
    buckets: list[list[tuple[float, str]]] = [[] for _ in range(n_cols)]
    for w in line:
        mid = (float(w["x0"]) + float(w["x1"])) / 2.0
        for ci in range(n_cols):
            lo = col_bounds[ci] - _MERGE_X_TOL
            hi = col_bounds[ci + 1] + _MERGE_X_TOL
            if lo <= mid < hi:
                buckets[ci].append((float(w["x0"]), str(w.get("text") or "")))
                break
    labels = [
        " ".join(t for _, t in sorted(b)).strip() if b else ""
        for b in buckets
    ]
    if not any(labels):
        return None
    header_top = min(float(w["top"]) for w in line)
    return labels, header_top


# ---------- pdfplumber path: simple flat extraction ---------------------------


def _build_jsonld_from_grid(
    grid: list[list[Any]],
    *,
    doc_sha: str,
    table_index: int,
    caption: str | None,
    page_number: int,
    header_row: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a clean rectangular grid into JSON-LD.

    Heuristic for header/row-label detection: the first row is treated
    as the column-header row. The first column is treated as a row-label
    column when its non-header cells are all non-numeric (lets the
    `pdfplumber` route cope with the common "metric name in col 0,
    numbers in col 1..N" pattern of financial tables).

    `header_row`, when supplied, carries column labels recovered from the
    text band above the table (see `_recover_header_row`). It is passed
    only for candidates assembled by `_merge_row_fragments`, whose rows are
    ALL data rows -- taking `grid[0]` as the header there would silently
    eat the first line item. Ignored unless its width matches the grid, so
    a bad recovery degrades to the normal heuristic rather than
    misaligning every column."""
    iri = table_jsonld.build_table_iri(doc_sha, table_index)
    payload = table_jsonld.empty_payload(
        doc_sha, table_index,
        extraction_method="pdfplumber",
        caption=caption,
        page_number=page_number,
    )

    grid_width = max((len(r) for r in grid), default=0)
    if header_row is not None and len(header_row) == grid_width:
        first_row = [str(c or "").strip() for c in header_row]
        data_rows = grid
    else:
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


# A caption is prose; a header line is mostly periods, dates and units.
# Used to walk back past a multi-line header block (a spanning header can
# run three lines: "Fiscal Years Ended | Percentage of Revenues" /
# "March 31, March 31, April 1," / "(in millions) 2018 2017(1) 2016(1)").
_MONTH_RE = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE
)
# How many header-ish lines to skip before giving up and taking what's there.
_MAX_HEADER_LINES = 4


def _looks_header_like(line: str) -> bool:
    """True when a line reads as part of a column-header block rather than
    a caption: mostly numbers, month names, or one/two-character tokens."""
    tokens = line.split()
    if not tokens:
        return True
    headerish = 0
    for t in tokens:
        stripped = t.strip("(),.$%:;")
        if not stripped or len(stripped) <= 2:
            headerish += 1
        elif _looks_numeric(stripped) or _MONTH_RE.match(stripped):
            headerish += 1
    return headerish / len(tokens) >= 0.5


def _caption_hint(
    page: Any,
    bbox: tuple[float, float, float, float],
    *,
    skip_header_lines: bool = False,
) -> str | None:
    """Pull the line of text immediately ABOVE the table's top edge as a
    cheap caption hint. Useful for the vision prompt + as a fallback
    caption when extraction is otherwise label-less.

    `skip_header_lines` is set for merged candidates, whose bbox top is a
    DATA row: the lines directly above are the column-header block, and
    returning one of those makes the caption a restatement of the headers
    ("2013 2012 2011") rather than a description of the table. Walks back
    past header-shaped lines to the first line that reads as prose."""
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
    if skip_header_lines:
        for ln in reversed(lines[-(_MAX_HEADER_LINES + 1):]):
            if not _looks_header_like(ln):
                candidate = ln
                break
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
    provider: str = "openai",
) -> tuple[str | None, float]:
    """OpenAI-compatible vision call (OpenAI, or DeepInfra's endpoint).
    Returns (json_text, cost_usd); raises VisionCallError when the request
    fails."""
    client = _get_deepinfra_vision_client() if provider == "deepinfra" else _get_vision_client()
    if client is None:
        raise VisionCallError(f"no API key for vision provider {provider!r}")
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
    except Exception as exc:
        raise VisionCallError(f"{type(exc).__name__}: {exc}") from exc
    text = (resp.choices[0].message.content or "").strip()
    if provider == "deepinfra":
        usage = getattr(resp, "usage", None)
        cost = _estimate_cost_by_model(
            model,
            getattr(usage, "prompt_tokens", None) if usage else None,
            getattr(usage, "completion_tokens", None) if usage else None,
        ) or 0.0
        return _strip_json_fences(text), cost
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
        raise VisionCallError("no API key for vision provider 'anthropic'")
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
    except Exception as exc:
        raise VisionCallError(f"{type(exc).__name__}: {exc}") from exc
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    text = _strip_json_fences(text)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", None) if usage else None
    out_tok = getattr(usage, "output_tokens", None) if usage else None
    cost = _estimate_cost_by_model(model, in_tok, out_tok) or 0.0
    return text, cost


def _vision_spec(task: str, fallback: str | None = None) -> dict[str, Any]:
    """Provider/model/limits for a vision task from models.yaml (`fallback`
    is read when `task` is absent, so older configs keep working)."""
    tasks = get_settings().models_config.get("tasks", {}) or {}
    spec = tasks.get(task) or (tasks.get(fallback) if fallback else None) or {}
    provider = str(spec.get("provider", "openai")).lower()
    if provider not in _VISION_PROVIDERS:
        raise ValueError(
            f"{task}.provider={provider!r} is not supported "
            f"(use one of {', '.join(_VISION_PROVIDERS)})"
        )
    return {
        "provider": provider,
        "model": spec.get("model", _DEFAULT_VISION_MODEL),
        "timeout": int(spec.get("timeout", _DEFAULT_VISION_TIMEOUT)),
        "temperature": float(spec.get("temperature", _DEFAULT_VISION_TEMP)),
        "max_tokens": int(spec.get("max_tokens", _DEFAULT_VISION_MAX_TOKENS)),
    }


def _render_png_b64(page: Any, bbox: tuple[float, float, float, float] | None) -> str | None:
    """`bbox` region (whole page when None) as a base64 PNG, or None."""
    try:
        region = page if bbox is None else page.crop(bbox)
        img = region.to_image(resolution=_RENDER_DPI)
        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        return base64.b64encode(png_buf.getvalue()).decode("ascii")
    except Exception:
        return None


async def _call_vision(
    spec: dict[str, Any], system: str, user: str, png_b64: str,
) -> tuple[str | None, float]:
    """Dispatch to the provider configured in models.yaml. Both helpers
    return (json_text, cost_usd) or raise VisionCallError. The anthropic path
    keeps table extraction fully OpenAI-free."""
    kw = {"model": spec["model"], "timeout": spec["timeout"],
          "temperature": spec["temperature"], "max_tokens": spec["max_tokens"]}
    if spec["provider"] == "anthropic":
        return await _vision_call_anthropic(system, user, png_b64, **kw)
    return await _vision_call_openai(system, user, png_b64, provider=spec["provider"], **kw)


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
    """Render the table region to PNG, send it to the configured vision
    model, parse JSON back into a StructuredTable payload. Returns
    (payload, cost_usd); (None, cost) when the crop holds no usable table.
    Raises VisionCallError when the request itself fails, so the caller can
    keep a failed extraction out of the cache."""
    spec = _vision_spec("table_extract_vision")

    # Render the cropped region to PNG bytes (shared across providers).
    png_b64 = _render_png_b64(page, bbox)
    if png_b64 is None:
        return None, 0.0

    system, user = PROMPTS["table_extract_vision"](
        page_number=page_number, caption_hint=caption_hint,
    )
    text, cost = await _call_vision(spec, system, user, png_b64)
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


# ---------- Whole-page vision: borderless financial statements ---------------
# Audited statements draw no grid lines, so find_tables() keeps only the ruled
# header strip above each statement. Measured 2026-10-23 on fin-gen-subset:
# all 37 "tables" of one A$ filing were 1-4 row strips like
# ['', 'Note', "A$'000", '', "A$'000"] (no labels, no figures), and the LGES
# (58 pp) and SDI (99 pp) audit reports yielded none at all. A page that is
# dense with figures but produced no real table is sent WHOLE to the vision
# model, and every figure it returns must appear in the page's text layer.

# A figure as printed in a statement: thousands-grouped or decimal, optional
# parentheses/sign. Bare integers ("2025", note numbers) do not count.
_FIGURE_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+\.\d+\)?")
# Any number, separators included ("1,158,885", "3.97", "3,5,6").
_NUMBER_RE = re.compile(r"\d[\d,.]*")
# pdfplumber's word-gap threshold (points) for the page-vision text layer.
_PAGE_TEXT_X_TOLERANCE = 1


def _data_row_count(grid: list[list[Any]]) -> int:
    """Rows holding a text label AND at least one numeric value."""
    n = 0
    for row in grid:
        cells = [c.strip() for c in row if isinstance(c, str) and c.strip()]
        if any(_looks_numeric(c) for c in cells) and any(not _looks_numeric(c) for c in cells):
            n += 1
    return n


def _payload_grid(payload: dict[str, Any]) -> list[list[str]]:
    """A StructuredTable payload as plain rows: [rowLabel, cell, cell, ...]."""
    return [
        [str(r.get("rowLabel") or "")]
        + [str(c.get("cellValue") or "") for c in r.get("cells", []) or []]
        for r in payload.get("rows", []) or []
    ]


def _is_header_only(grid: list[list[Any]]) -> bool:
    """A detected 'table' that is only a column-header strip: fewer than two
    rows pairing a label with a value (years and units alone don't count)."""
    return _data_row_count(grid) < 2


def _figure_lines(text: str) -> int:
    """How many lines of page text carry two or more figures."""
    return sum(1 for ln in text.splitlines() if len(_FIGURE_RE.findall(ln)) >= 2)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _printed_numbers(page_text: str) -> set[str]:
    """Digit strings of every number printed on the page."""
    # Compare NUMBERS, not cells: the model writes "₩ (2,700)" and
    # "3.00% – 3.97%", which a whole-cell numeric test skipped -- 34 of 35
    # LGES rejections (2026-10-23) were real tables scored as having no
    # figures at all. A Notes cell "3,5,7,29" is one number, "35729".
    printed = {_digits(m) for m in _NUMBER_RE.findall(page_text)}
    printed |= {_digits(t) for t in page_text.split()}
    return printed


def _ungrounded_share(payload: dict[str, Any], page_text: str) -> float:
    """Share of the table's numbers whose digits do not occur in the
    page's text layer (0.0 = every figure is printed on the page)."""
    printed = _printed_numbers(page_text)
    nums = [
        _digits(m)
        for r in payload.get("rows", []) or []
        for c in r.get("cells", []) or []
        for m in _NUMBER_RE.findall(str(c.get("cellValue") or ""))
    ]
    if not nums:
        return 1.0
    return sum(1 for d in nums if d not in printed) / len(nums)


def _blank_ungrounded_cells(payload: dict[str, Any], page_text: str) -> int:
    """Empty every cell holding a number the page does not print; returns how
    many. A kept table may still carry misreads under the ungrounded limit
    (SDI 2024 revenue read as ...884,368, printed ...884,388: 1-4% of figures
    in the 2026-10-23 smoke) -- a gap is better than a citable wrong figure."""
    printed = _printed_numbers(page_text)
    n = 0
    for r in payload.get("rows", []) or []:
        for c in r.get("cells", []) or []:
            nums = _NUMBER_RE.findall(str(c.get("cellValue") or ""))
            if any(_digits(m) not in printed for m in nums):
                c["cellValue"] = ""
                n += 1
    return n


def page_vision_settings() -> dict[str, Any]:
    """tables.page_vision* from config.yaml, with defaults."""
    t = (get_settings().app_config.get("tables", {}) or {})
    return {
        "enabled": bool(t.get("page_vision", True)),
        "min_figure_lines": int(t.get("page_vision_min_figure_lines", 8)),
        "max_pages": int(t.get("page_vision_max_pages_per_doc", 80)),
        "max_ungrounded": float(t.get("page_vision_max_ungrounded", 0.1)),
    }


def page_vision_concurrency() -> int:
    """Whole-page vision calls in flight per PDF (concurrency.table_page_vision).

    Unset = 1: pages go one at a time, as before. Sized from free memory by
    scripts/tpm_check.py and scripts/deepinfra_sizing.py together with
    concurrency.table_extraction (see table_concurrency_for_budget). Each call
    is decode-bound -- serverless Qwen3-VL ~33 tok/s per stream, unchanged at 8
    in flight (2026-10-24) -- so pages in parallel cut a PDF's wall time
    without changing what is sent or kept."""
    cfg = get_settings().app_config
    return max(1, int((cfg.get("concurrency", {}) or {}).get("table_page_vision", 1)))


async def _page_vision_request(page: Any, page_number: int) -> tuple[str | None, float]:
    """The slow half of page vision: render the page and ask the model.
    (None, 0.0) when the page cannot be rendered; raises VisionCallError."""
    spec = _vision_spec("table_extract_page_vision", fallback="table_extract_vision")
    png_b64 = _render_png_b64(page, None)
    if png_b64 is None:
        return None, 0.0
    system, user = PROMPTS["table_extract_page_vision"](page_number=page_number)
    return await _call_vision(spec, system, user, png_b64)


def _flush(page: Any) -> None:
    try:
        page.flush_cache()
    except Exception:
        pass


class _PageVisionPrefetch:
    """Starts whole-page vision requests ahead of the page loop.

    The loop stays sequential -- each page's result sets the table numbering
    of every later page and whether its header strips are kept -- but the
    REQUEST for a figure-dense page depends only on the page, so it can be in
    flight while earlier pages are still being processed. The loop then
    `take`s each page's answer in order, so requests, post-processing and
    numbering are what the one-at-a-time loop produces. A prefetched page the
    loop does not send (it had real pdfplumber tables, or the cap was hit) is
    cancelled or, if it already finished, counted in `wasted`.

    Pages are touched only AHEAD of the loop (`main_pos`), and everything runs
    on the one event-loop thread, so pdfplumber never sees two users of a page.
    """

    def __init__(self, pdf: Any, *, page_cap: int, pv: dict[str, Any], concurrency: int) -> None:
        self._pdf = pdf
        self._page_cap = page_cap
        self._pv = pv
        self._slots = asyncio.Semaphore(concurrency)
        self._tasks: dict[int, asyncio.Task] = {}
        self._scheduled = 0
        self._direct = 0
        self.texts: dict[int, str] = {}       # page text read here, reused by the loop
        self.main_pos = 0
        self.wasted = 0
        self.wasted_cost = 0.0
        self._runner: asyncio.Task | None = None

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run())

    async def _run(self) -> None:
        n_pages = len(self._pdf.pages)
        for page_no in range(1, min(n_pages, self._page_cap) + 1):
            # Shares the per-PDF cap with the loop's own (non-prefetched) calls.
            if self._scheduled + self._direct >= self._pv["max_pages"]:
                return
            await self._slots.acquire()          # released when the loop takes/discards it
            if page_no <= self.main_pos:         # the loop already passed it
                self._slots.release()
                continue
            page = self._pdf.pages[page_no - 1]
            try:
                text = page.extract_text(x_tolerance=_PAGE_TEXT_X_TOLERANCE) or ""
            except Exception:
                text = ""
            self.texts[page_no] = text
            if _figure_lines(text) < self._pv["min_figure_lines"] or self._has_real_table(page):
                self._slots.release()
                _flush(page)
                continue
            self._tasks[page_no] = asyncio.create_task(_page_vision_request(page, page_no))
            self._scheduled += 1
            _flush(page)
            await asyncio.sleep(0)

    @staticmethod
    def _has_real_table(page: Any) -> bool:
        """The loop sends a figure-dense page only when none of its tables is
        kept. A ruled table with data rows is (nearly always) kept, so such a
        page is left to the loop: on 14 pharma labels 88 of 129 figure-dense
        pages were of this kind (2026-10-24). A wrong guess costs nothing in
        output -- an unsent page is asked directly, a sent one is discarded."""
        try:
            tables = page.find_tables() or []
            return any(not _is_header_only(t.extract() or []) for t in tables)
        except Exception:
            return False

    async def take(self, page: Any, page_no: int) -> tuple[str | None, float]:
        """This page's answer: the prefetched one, or a direct request."""
        task = self._tasks.pop(page_no, None)
        if task is None:
            self._direct += 1
            return await _page_vision_request(page, page_no)
        try:
            return await task
        finally:
            self._slots.release()

    def discard(self, page_no: int) -> None:
        task = self._tasks.pop(page_no, None)
        if task is None:
            return
        if task.done() and not task.cancelled() and task.exception() is None:
            self.wasted += 1
            self.wasted_cost += float(task.result()[1] or 0.0)
        else:
            task.cancel()
        self._slots.release()

    async def close(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        for page_no in list(self._tasks):
            self.discard(page_no)
        await asyncio.gather(*(t for t in [self._runner] if t), return_exceptions=True)


def _page_vision_payloads(
    text: str | None,
    *,
    page_text: str,
    doc_sha: str,
    first_table_index: int,
    page_number: int,
    max_ungrounded: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """The deterministic half of page vision: keep the returned tables whose
    figures are printed on the page, blank any cell whose number is not.
    Returns (payloads, n_rejected_ungrounded, n_cells_blanked)."""
    if text is None:
        return [], 0, 0
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return [], 0, 0
    if not isinstance(body, dict) or body.get("no_table") is True:
        return [], 0, 0
    out: list[dict[str, Any]] = []
    rejected = 0
    blanked = 0
    for raw in body.get("tables") or []:
        payload = _vision_body_to_jsonld(
            raw, doc_sha=doc_sha, table_index=first_table_index + len(out) + rejected,
            page_number=page_number, caption_hint=None,
        )
        if payload is None or not payload.get("rows"):
            continue
        if _looks_fabricated(payload) or _ungrounded_share(payload, page_text) > max_ungrounded:
            rejected += 1
            continue
        blanked += _blank_ungrounded_cells(payload, page_text)
        out.append(payload)
    return out, rejected, blanked


async def _extract_page_tables_via_vision(
    page: Any,
    *,
    page_text: str,
    doc_sha: str,
    first_table_index: int,
    page_number: int,
    max_ungrounded: float,
    prefetch: _PageVisionPrefetch | None = None,
) -> tuple[list[dict[str, Any]], float, int, int]:
    """Render the whole page, ask the vision model for every table on it, and
    keep the ones whose figures are printed on the page, with any cell whose
    number is not printed blanked. Returns (payloads, cost_usd,
    n_rejected_ungrounded, n_cells_blanked); raises VisionCallError on a
    failed request so the caller keeps the PDF out of the cache. With
    `prefetch`, the request may already be in flight."""
    text, cost = (await prefetch.take(page, page_number) if prefetch is not None
                  else await _page_vision_request(page, page_number))
    out, rejected, blanked = _page_vision_payloads(
        text, page_text=page_text, doc_sha=doc_sha, first_table_index=first_table_index,
        page_number=page_number, max_ungrounded=max_ungrounded,
    )
    return out, cost, rejected, blanked


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


def _mirror_to_run_cache(run_cache_dir: Path | None, cache_key: str, hit: Any, pdf: Path) -> None:
    """Copy a user-cache hit into the run cache (if configured and absent)."""
    if run_cache_dir is None or (Path(run_cache_dir) / f"{cache_key}.jsonld").exists():
        return
    try:
        table_cache.save(
            run_cache_dir, cache_key,
            doc_sha=hit.doc_sha,
            doc_path=str(pdf),
            tables=hit.tables,
            manifest=hit.manifest,
        )
    except Exception:
        pass


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
        _mirror_to_run_cache(run_cache_dir, cache_key, hit, p)
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
    n_vision_errors = 0
    cost_usd = 0.0
    table_index = 0
    pages_processed = 0
    pages_truncated = False

    n_skipped_filter = 0
    n_merged = 0          # tables assembled from >1 detected fragment
    n_headers_recovered = 0
    skip_reasons: dict[str, int] = {}
    pv = page_vision_settings() if use_vision else {"enabled": False}
    n_page_vision_pages = 0
    n_page_vision_tables = 0
    n_page_vision_ungrounded = 0
    n_page_vision_cells_blanked = 0
    n_header_only_dropped = 0
    prefetch: _PageVisionPrefetch | None = None
    page_conc = page_vision_concurrency() if pv["enabled"] else 1

    try:
        with pdfplumber.open(str(p)) as pdf:
            n_pages = len(pdf.pages)
            page_cap = max_pages if max_pages is not None else _MAX_PAGES_PER_DOC
            if page_conc > 1:
                prefetch = _PageVisionPrefetch(pdf, page_cap=page_cap, pv=pv, concurrency=page_conc)
                prefetch.start()
                await asyncio.sleep(0)            # let it claim the first pages
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
                if prefetch is not None:
                    prefetch.main_pos = page_no
                page_w = float(getattr(page, "width", 0.0) or 0.0)
                page_h = float(getattr(page, "height", 0.0) or 0.0)
                try:
                    tables = page.find_tables() or []
                except Exception:
                    tables = []
                page_kept_before = len(payloads)
                page_skipped_filter = 0
                # Page vision needs the text layer: to spot figure-dense pages
                # and to check every figure the model returns. x_tolerance=1:
                # at the default 3 pt, 44 of 148 tightly set FinQA pages ran
                # words together ("Devonhasaccessto$5.0billion"); fin-gen
                # text is unchanged and no words split apart (2026-10-23).
                page_text = ""
                figure_dense = False
                if pv["enabled"]:
                    if prefetch is not None and page_no in prefetch.texts:
                        page_text = prefetch.texts.pop(page_no)
                    else:
                        try:
                            page_text = page.extract_text(x_tolerance=_PAGE_TEXT_X_TOLERANCE) or ""
                        except Exception:
                            page_text = ""
                    figure_dense = _figure_lines(page_text) >= pv["min_figure_lines"]
                header_only: list[dict[str, Any]] = []
                # Reassemble row-per-candidate fragments BEFORE any filter --
                # each fragment is individually thin enough to be dropped by
                # the bbox guard, while the table they form is not.
                candidates = _merge_row_fragments(tables)
                for bbox, grid, col_bounds, n_fragments in candidates:
                    if table_index >= _MAX_TABLES_PER_DOC:
                        break

                    # Filter 1: bbox geometry (header/footer/thin bands)
                    reason = _bbox_filter_reason(bbox, page_w, page_h)
                    if reason:
                        n_skipped_filter += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        page_skipped_filter += 1
                        table_index += 1
                        continue

                    # Recover the header BEFORE the caption. Merging pulls
                    # the bbox top down to the first DATA row, so the line
                    # immediately above it is the column header -- reading
                    # the caption from there returned the header verbatim
                    # ("2013 2012 2011"), which is also what we feed the
                    # mining prompt as the table's description. Knowing
                    # where the header sits lets the caption band start
                    # above it instead.
                    header_row: list[str] | None = None
                    header_top: float | None = None
                    if n_fragments > 1 and col_bounds is not None:
                        n_merged += 1
                        recovered = _recover_header_row(page, bbox, col_bounds)
                        if recovered is not None:
                            header_row, header_top = recovered
                            n_headers_recovered += 1

                    # Nudge the band above the header's own top edge:
                    # pdfplumber's crop keeps objects that INTERSECT the
                    # box, so cutting exactly at `header_top` re-admits the
                    # header line (clipped to a fragment like '2013').
                    caption_bbox = (
                        bbox if header_top is None
                        else (bbox[0], header_top - _HEADER_CLEARANCE_PT,
                              bbox[2], bbox[3])
                    )
                    caption = _caption_hint(
                        page, caption_bbox,
                        skip_header_lines=header_top is not None,
                    )

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
                            header_row=header_row,
                        )
                        n_pdfplumber += 1
                    elif use_vision:
                        try:
                            payload, vision_cost = await _extract_table_via_vision(
                                page,
                                bbox,
                                doc_sha=doc_sha,
                                table_index=table_index,
                                page_number=page_no,
                                caption_hint=caption,
                                router=router,
                            )
                        except VisionCallError as exc:
                            payload, vision_cost = None, 0.0
                            n_vision_errors += 1
                            if n_vision_errors == 1:
                                print(f"[tables] {p.name}: vision call FAILED ({exc}); "
                                      f"this PDF will not be cached", flush=True)
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

                    if figure_dense and _is_header_only(_payload_grid(payload)):
                        # A header strip above a borderless statement; page
                        # vision below replaces it with the real table.
                        header_only.append(payload)
                        table_index += 1
                        continue

                    payloads.append(payload)
                    table_index += 1

                # Whole-page vision for a figure-dense page that produced no
                # real table (see _extract_page_tables_via_vision).
                if header_only or figure_dense:
                    vision_page = (
                        figure_dense
                        and len(payloads) == page_kept_before
                        and n_page_vision_pages < pv["max_pages"]
                        and table_index < _MAX_TABLES_PER_DOC
                    )
                    replaced = False
                    if vision_page:
                        n_page_vision_pages += 1
                        try:
                            found, page_cost, rejected, blanked = await _extract_page_tables_via_vision(
                                page, page_text=page_text, doc_sha=doc_sha,
                                first_table_index=table_index, page_number=page_no,
                                max_ungrounded=pv["max_ungrounded"], prefetch=prefetch,
                            )
                        except VisionCallError as exc:
                            n_vision_errors += 1
                            if n_vision_errors == 1:
                                print(f"[tables] {p.name}: page vision FAILED ({exc}); "
                                      f"this PDF will not be cached", flush=True)
                        else:
                            # Strips are dropped only when vision returned the
                            # statement; a "no table" answer keeps them.
                            replaced = bool(found)
                            cost_usd += page_cost
                            n_page_vision_ungrounded += rejected
                            n_page_vision_cells_blanked += blanked
                            table_index += len(found) + rejected
                            for t in found:
                                if not table_jsonld.validate_table_jsonld(t):
                                    payloads.append(t)
                                    n_page_vision_tables += 1
                                    n_vision += 1
                    elif prefetch is not None:
                        prefetch.discard(page_no)
                    if replaced:
                        n_header_only_dropped += len(header_only)
                    else:
                        # Not sent (cap reached, or the page has real tables)
                        # or the request failed: keep the fragments as before.
                        payloads.extend(header_only)

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
        if prefetch is not None:
            await prefetch.close()
        gc.collect()

    if prefetch is not None:
        cost_usd += prefetch.wasted_cost
    wall = round(time.monotonic() - started, 3)
    manifest = {
        "source": "fresh",
        "doc_sha": doc_sha,
        "n_tables": len(payloads),
        "n_pdfplumber": n_pdfplumber,
        "n_vision_llm": n_vision,
        "n_dropped": n_dropped,
        "n_vision_errors": n_vision_errors,
        "n_skipped_filter": n_skipped_filter,
        "n_merged": n_merged,
        "n_headers_recovered": n_headers_recovered,
        "skip_reasons": skip_reasons,
        "n_page_vision_pages": n_page_vision_pages,
        "n_page_vision_tables": n_page_vision_tables,
        "n_page_vision_ungrounded": n_page_vision_ungrounded,
        "n_page_vision_cells_blanked": n_page_vision_cells_blanked,
        "n_header_only_dropped": n_header_only_dropped,
        "page_vision_concurrency": page_conc,
        "n_page_vision_prefetch_wasted": prefetch.wasted if prefetch is not None else 0,
        "cost_usd": round(cost_usd, 5),
        "wall_seconds": wall,
        "pages_processed": pages_processed,
        "pages_truncated": pages_truncated,
    }
    print(
        f"[tables] {p.name}: DONE -- {len(payloads)} kept "
        f"(pdfplumber={n_pdfplumber}, vision={n_vision}, "
        f"merged={n_merged}, headers-recovered={n_headers_recovered}, "
        f"filter-skip={n_skipped_filter}, vision-drop={n_dropped}, "
        f"page-vision={n_page_vision_tables} from {n_page_vision_pages} page(s), "
        f"ungrounded={n_page_vision_ungrounded}, cells-blanked={n_page_vision_cells_blanked}, header-only-dropped={n_header_only_dropped}) "
        f"over {pages_processed} pages in {wall:.1f}s, cost=${cost_usd:.4f}",
        flush=True,
    )

    # Persist to both cache tiers (run cache + user cache) -- unless a
    # vision call failed: caching that result would freeze the missing
    # tables in, and every later run would hit the cache and never retry.
    for tier_dir in ((run_cache_dir, user_cache_dir) if not n_vision_errors else ()):
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


def table_extraction_concurrency() -> int:
    """How many PDFs to extract in parallel (concurrency.table_extraction).

    Defaults to 1: each PDF runs in its own subprocess because the in-process
    extractor accumulated heap fragmentation and OOMed partway through a corpus
    on a ~2.7 GB host. Measured at ~152 MB RSS and ~340 s per 125-page PDF, so
    18 PDFs serially is ~100 minutes -- worth raising on a roomier machine, but
    only after checking free memory, since one pathological PDF (large page
    rasters for the vision route) can cost several times the average.
    """
    from backend.app.core.config import get_settings

    cfg = get_settings().app_config
    return max(1, int((cfg.get("concurrency", {}) or {}).get("table_extraction", 1)))


# A worker's stdout is read until EOF and then its exit status awaited, both
# with no deadline. On 2026-10-23 a register-documents run sat for 25 minutes
# (dedicated GPU billing) after its third worker had exited: the parent never
# saw EOF/exit, no child process existed, and the worker had already written
# its cache entry. Not reproduced in 40 isolated cached runs, so instead of
# trusting EOF the pump polls: on every quiet interval it checks whether the
# worker is still alive, and a live worker silent for too long is killed.
# Either way the caller then reads the cache, which recovers that case.
_WORKER_POLL_S = 30.0
_WORKER_EXIT_WAIT_S = 30.0
_DEFAULT_WORKER_MAX_SILENT_S = 1800.0


def _worker_max_silent_s() -> float:
    """tables.worker_max_silent_seconds: kill a worker that is alive but has
    printed nothing for this long (it prints at least every 20 s per page
    unless a vision call is outstanding, and those time out at 300 s)."""
    cfg = get_settings().app_config
    tables = cfg.get("tables", {}) or {}
    return float(tables.get("worker_max_silent_seconds", _DEFAULT_WORKER_MAX_SILENT_S))


def _worker_gone(proc: Any) -> bool:
    """True when the worker has exited, even if asyncio has not noticed.
    Linux reads /proc (a zombie counts as gone); elsewhere only the
    returncode asyncio recorded is trusted -- os.kill(pid, 0) would
    terminate the process on Windows."""
    if proc.returncode is not None:
        return True
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open(f"/proc/{proc.pid}/stat", "rb") as fh:
            state = fh.read().rsplit(b")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state in (b"Z", b"X")


async def _pump_worker(proc: Any, pdf: Path, *, max_silent_s: float) -> int:
    """Forward the worker's output and return its exit code (-1 if unknown,
    -9 if it was killed). Never blocks forever: see the note above."""
    assert proc.stdout is not None
    silent = 0.0
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=_WORKER_POLL_S)
        except asyncio.TimeoutError:
            silent += _WORKER_POLL_S
            if _worker_gone(proc):
                print(f"[tables/worker] {pdf.name}: worker exited without closing its "
                      f"output; reading its result from the cache", flush=True)
                break
            if silent >= max_silent_s:
                print(f"[tables/worker] {pdf.name}: no output for {silent:.0f}s; "
                      f"killing the worker", flush=True)
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                break
            continue
        if not line:
            break
        silent = 0.0
        # The worker prefixes its own lines with [tables] / [worker]; pass
        # through verbatim.
        sys.stdout.write(line.decode("utf-8", errors="replace"))
        sys.stdout.flush()
    try:
        return await asyncio.wait_for(proc.wait(), timeout=_WORKER_EXIT_WAIT_S)
    except asyncio.TimeoutError:
        print(f"[tables/worker] {pdf.name}: exit status never arrived; "
              f"treating it as unknown", flush=True)
        return -1 if proc.returncode is None else proc.returncode


def _find_pdfs(folder: Path) -> list[Path]:
    """Every PDF under `folder`, recursively and case-insensitively.

    Must match how documents themselves are discovered -- ontology_io
    .iter_documents walks with rglob and lowercases the suffix. These two
    used `glob("*.pdf")`, so pointing --input at a corpus whose PDFs sit one
    level down ingested every document while extracting tables from none of
    them, and returned {} without a word. A paid, opt-in feature did nothing
    and said nothing.
    """
    pdfs = sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdfs:
        print(
            f"[tables] WARNING: no PDFs found under {folder} -- table "
            f"extraction was requested but has nothing to do. Check the path.",
            flush=True,
        )
    return pdfs


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
    paths: list[Path] | None = None,
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

    `concurrency=1` is the safe default; raise it on roomier hosts.

    `paths`, when given, restricts extraction to those files instead of
    scanning `folder` -- corpus selection passes its chosen subset so a run
    does not pay vision costs for documents the ontology will never see."""
    folder = Path(folder)
    if paths is not None:
        pdfs = sorted(p for p in paths if p.suffix.lower() == ".pdf")
    else:
        pdfs = _find_pdfs(folder)
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
                     "n_failed": 0, "cost_saved_usd": 0.0}
    results: dict[str, dict[str, Any]] = {}

    cmd_base = [sys.executable, "-u", "-m",
                "backend.app.services.table_extract_worker"]

    spawn_error: dict[Path, str] = {}

    async def _spawn_worker(p: Path) -> int | None:
        """Run the extraction worker for one PDF; its exit code, or None when
        it could not be started (reason in `spawn_error`)."""
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
            spawn_error[p] = f"{type(exc).__name__}: {exc}"
            print(f"[tables/worker] spawn failed for {p.name}: {spawn_error[p]}", flush=True)
            return None
        # Stream worker output through to parent stdout so the user
        # still sees per-PDF progress lines from the worker.
        return await _pump_worker(proc, p, max_silent_s=_worker_max_silent_s())

    async def _one(p: Path) -> tuple[str, dict[str, Any]]:
        nonlocal done
        async with sem:
            # Sample cache presence BEFORE the worker runs. Afterwards the
            # entry exists either way, so nothing downstream can tell whether
            # this run paid for it.
            was_cached = False
            try:
                _, _pre_key = _hash_pdf_streaming(p)
                pre_hit = table_cache.two_tier_load(
                    run_cache_dir, table_cache.user_cache_dir(), _pre_key,
                )
                was_cached = pre_hit is not None
            except Exception:  # noqa: BLE001 -- accounting only, never fatal
                was_cached = False

            if was_cached:
                # A cache hit needs no worker: the worker would only copy
                # the entry into the run cache. Spawning one per PDF anyway
                # cost 632 of register-documents' 670 s on 148 cached FinQA
                # PDFs (2026-10-23) -- about 10 min of paid GPU clock.
                _mirror_to_run_cache(run_cache_dir, _pre_key, pre_hit, p)
                rc = 0
            else:
                rc = await _spawn_worker(p)
                if rc is None:
                    done += 1
                    summary_stats["n_failed"] += 1
                    return str(p), {"source": "spawn-failed", "reason": spawn_error[p]}
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
                hit = table_cache.two_tier_load(
                    run_cache_dir, table_cache.user_cache_dir(), cache_key,
                )
                if hit is not None:
                    manifest.update(hit.manifest)
                    manifest["doc_sha"] = hit.doc_sha
                    summary_stats["n_tables"] += int(
                        hit.manifest.get("n_tables", 0) or 0
                    )
                    cached_cost = float(hit.manifest.get("cost_usd", 0.0) or 0.0)
                    # Cache-hit detection cannot read the manifest's own
                    # "source": that field is written as "fresh" when the entry
                    # is PERSISTED, and the "cache" value is only attached to
                    # the in-memory return value (see the cold-path return
                    # above). Reloading the file therefore always saw "fresh",
                    # so n_cached was permanently 0 on this path -- an 18-PDF
                    # re-run reported "0 cache-hits" while finishing in 66s
                    # instead of 526s.
                    #
                    # `was_cached` is sampled BEFORE the worker is spawned,
                    # which is the only observation that actually distinguishes
                    # "the entry already existed" from "the worker just wrote
                    # it".
                    if was_cached:
                        summary_stats["n_cached"] += 1
                        # Not spent THIS run. Adding it overstated the bill by
                        # the full original extraction cost on every re-run --
                        # two consecutive runs both reported $0.8145 when the
                        # second paid nothing.
                        summary_stats["cost_saved_usd"] += cached_cost
                        manifest["source"] = "cache"
                    else:
                        summary_stats["cost_usd"] += cached_cost
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
        f"cost=${summary_stats['cost_usd']:.4f}"
        + (
            f" (${summary_stats['cost_saved_usd']:.4f} saved by cache)"
            if summary_stats.get("cost_saved_usd") else ""
        ),
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
    pdfs = _find_pdfs(folder)
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

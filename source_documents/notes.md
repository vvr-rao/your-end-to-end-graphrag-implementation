## Drop documents here

Files in this folder feed the `prune` and `expand` CLIs (see top-level
README.md). Subfolders are gitignored so anything bulky stays local.

## Bundled downloaders

Three standalone scripts live in this folder. Each accepts `--search <q>`
and `--output <path>` (default: a sluggified subfolder under
`source_documents/`).

```bash
# 1) DailyMed -- Patient-Information PDFs by condition.
uv run python source_documents/dailymed_download.py \
  --search "diabetes" --output source_documents/pharma_documents --max 10

# 2) Web search -- DuckDuckGo HTML SERP, top-N pages extracted as .txt.
uv run python source_documents/websearch_download.py \
  --search "GraphRAG ontology techniques" --max 5

# 3) SEC EDGAR -- financial-report PDFs (10-K / 10-Q / 20-F / 40-F / 8-K).
#    Most US 10-Ks ship as iXBRL only; pass --allow-html to fetch the
#    primary HTML 10-K body and convert it to PDF via WeasyPrint.
uv run python source_documents/financial_report_download.py \
  --search "Apple Inc" --max 5 --allow-html
```

Each tool prints a per-result progress line, then a one-line summary.
SEC EDGAR filings are mostly HTML/iXBRL -- only filings that include a
PDF attachment land in the output folder; the rest are reported and
skipped. With `--allow-html`, the primary 10-K HTML body is fetched and
rendered to PDF via WeasyPrint; if conversion fails the raw `.htm` is
saved instead so the filing content isn't lost.

Other useful sources of information
- HKEX for Hong Kong listed securities. No API Available that I know of but you can go to their site and do a keyword search (e.g. "Annual Report") - https://www1.hkexnews.hk/search/titlesearch.xhtml?category=0&lang=EN&market=SEHK&stockId=130186
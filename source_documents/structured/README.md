# Structured-data downloads

`source_documents/structured_download.py` downloads **bounded samples** (and, when you
ask for them explicitly, bulk snapshots) of public finance and pharma datasets into
local files. Each run also writes a `manifest.json`. A later YEGI upgrade will ingest
these files. This tool only acquires, extracts and (optionally) normalizes data. It
makes no LLM calls and does not write to the database.

| Domain | Source | API mode | Bulk / archive mode |
|---|---|---|---|
| finance | `sec`: SEC EDGAR | Company Facts, submissions per CIK | Financial Statement Data Sets (quarter), Notes data sets, nightly `companyfacts.zip` / `submissions.zip` |
| finance | `fred`: FRED / ALFRED | series metadata, observations, real-time periods, vintages | none (FRED publishes no bulk files) |
| finance | `gleif`: GLEIF | LEI records + parent/child relationships + reporting exceptions | Golden Copy (full) or delta files: `lei2`, `rr`, `repex` |
| finance | `ffiec`: FFIEC CDR / Fed NIC | CDR REST API: reporting periods, panel of reporters, filers since, facsimiles | **archive**: CDR "Call Reports -- Single Period" zip, NIC zip (you download them in a browser) |
| pharma | `clinicaltrials`: ClinicalTrials.gov / AACT | API v2 studies | **archive**: AACT flat-file or pg_dump zip (needs an AACT account) |
| pharma | `drugsfda`: Drugs@FDA | openFDA `drug/drugsfda` | FDA Drugs@FDA data files (tab), openFDA bulk JSON |
| pharma | `orangebook`: FDA Orange Book | openFDA `drug/orangebook` | FDA Orange Book data files (tilde), openFDA bulk JSON |
| pharma | `chembl`: ChEMBL | molecule / target / assay / activity / document with filters | release assets (SQLite, PostgreSQL, SDF, chemreps…), checked against SHA-256 |
| pharma | `opentargets`: Open Targets | GraphQL: target / disease / drug (+ paged associations) | Parquet datasets of a pinned release |

```bash
uv run python source_documents/structured_download.py list            # sources, modes, formats, credentials
uv run python source_documents/structured_download.py info gleif      # datasets, notes, docs + terms links
uv run python source_documents/structured_download.py gleif api --help
```

## Setup and credentials

Credentials are read from environment variables, or from the repo's `.env`. Existing
environment variables take precedence over `.env`. Never pass a credential on the
command line. API keys are added to requests when they are sent and are redacted from
logs, error messages and manifests.

| Variable | Needed for | Notes |
|---|---|---|
| `SEC_USER_AGENT` | `sec` (all modes) | Your organisation and a **real** contact email, e.g. `Acme Research jane@acme.org`. SEC fair-access rules require it, and placeholder values are rejected. |
| `FRED_API_KEY` | `fred` | Free key from <https://fredaccount.stlouisfed.org/apikeys>. |
| `OPENFDA_API_KEY` | `drugsfda`, `orangebook` (optional) | Without a key, openFDA allows 1,000 requests/day per IP. |
| `FFIEC_USER_ID`, `FFIEC_TOKEN` | `ffiec api` | CDR PWS account (<https://cdr.ffiec.gov/public/PWS/PWSPage.aspx>). The token expires every 90 days. Pass the bare token, without `Bearer`. |

GLEIF, ClinicalTrials.gov, ChEMBL and Open Targets need no credentials.

Parquet → CSV conversion (Open Targets bulk `--format csv`) needs pyarrow:
`uv sync --extra structured`. Native Parquet downloads work without it.

## Common options

| Option | Meaning |
|---|---|
| `--dataset NAME` | Required whenever a mode offers more than one dataset. Bulk downloads always need an explicit choice. |
| `--format native\|csv` | `native` keeps provider files only. `csv` keeps them **and** adds normalized CSV under `normalized/`. Each source states what it supports, and unsupported combinations are rejected. |
| `--limit N`, `--max-pages N` | Bounded API samples. A capped run is always marked `truncated: true`, status `sample`. **For bulk archives `--limit` is not offered**: an archive downloads whole, and a row cap would not reduce its size. |
| `--dry-run` | Prints the plan (redacted URLs, expected sizes, release) and exits. It may make small metadata lookups, such as a HEAD request, a release listing or openFDA's `download.json`. |
| `--resume` | Reuses an existing snapshot folder. Files whose size and SHA-256 match its manifest are skipped, and `.part` transfers continue where the server supports byte ranges. |
| `--overwrite` | Replaces an existing snapshot folder of the same name. Without `--resume` or `--overwrite`, an existing snapshot is never touched. |
| `-o DIR` | Output root. The default is `source_documents/structured_<source>/`, which git ignores. |
| `--timeout`, `--max-retries` | Read timeout per request, and retries on 429/5xx/connection errors (exponential backoff, honours `Retry-After`). |

## Small runnable examples (API mode)

```bash
P="uv run python source_documents/structured_download.py"

$P sec api --dataset companyfacts --cik 320193 --format csv --limit 500     # needs SEC_USER_AGENT
$P fred api --series-id GDP,UNRATE --start 2020-01-01 --format csv          # needs FRED_API_KEY
$P fred api --series-id GDP --vintage-dates 2024-01-01,2025-01-01 --format csv
$P gleif api --name "Apple Inc" --limit 2 --relationships --format csv
$P ffiec api --dataset panel --period 2025-03-31 --format csv               # needs FFIEC_* vars
$P ffiec api --dataset facsimile --period 2025-03-31 --rssd 480228 --facsimile-format XBRL
$P clinicaltrials api --condition obesity --intervention semaglutide --limit 3 --format csv
$P drugsfda api --search 'openfda.brand_name:"OZEMPIC"' --limit 2 --format csv
$P orangebook api --search 'products.brand_name:"WEGOVY"' --limit 2 --format csv
$P chembl api --dataset activity --filter target_chembl_id=CHEMBL231 --filter standard_type=IC50 --limit 5 --format csv
$P opentargets api --dataset target --id ENSG00000169083 --limit 5 --format csv
```

## Bulk examples (whole files: check the `--dry-run` first)

```bash
$P sec bulk --dataset fsds --period 2024q1 --dry-run               # ~120 MB zip: sub/num/tag/pre.txt
$P sec bulk --dataset fsds --period 2024q1 --format csv
$P gleif bulk --dataset rr --dry-run                               # latest Golden Copy, ~25 MB
$P gleif bulk --dataset lei2 --delta LastDay                       # delta file, not a full snapshot
$P drugsfda bulk --dataset flatfiles --format csv                  # FDA zip, 12 tab-delimited tables
$P orangebook bulk --dataset flatfiles --format csv                # ~1 MB zip, tilde-delimited
$P drugsfda bulk --dataset openfda-json                            # partitions from api.fda.gov/download.json
$P chembl bulk --release 37 --asset chemreps --dry-run             # ~290 MB; sqlite asset is ~5.4 GB
$P opentargets bulk --dataset target --release 26.09 --max-files 1 --format csv
```

Manual-archive paths (no documented automated download exists):

```bash
# AACT: sign in at https://aact.ctti-clinicaltrials.org/, download the flat-file zip
$P clinicaltrials archive --dataset aact-flatfiles --from-archive ~/Downloads/20261108_pipe-delimited-export.zip --format csv
# FFIEC CDR: https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx -> "Call Reports -- Single Period",
#            a reporting date, "Tab Delimited"
$P ffiec archive --dataset cdr-call-bulk --from-archive "~/Downloads/FFIEC CDR Call Bulk All Schedules 03312025.zip" --format csv
# Federal Reserve NIC: https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload (browser; CAPTCHA)
$P ffiec archive --dataset nic --from-archive ~/Downloads/CSV_RELATIONSHIPS.zip
```

The archive is copied into the snapshot folder (your original is left where it is),
then safely extracted and documented.

## Output structure

```
<output>/<dataset>/<snapshot>/
  manifest.json
  native/        original provider files: API JSON pages, zips, Parquet parts, GraphQL query used
  extracted/     safely-extracted archive members (tab / tilde / pipe tables kept as-is)
  normalized/    optional CSV (UTF-8, comma), derived from native files only
```

`<snapshot>` is the provider release when one exists, e.g. SEC `2024q1`, GLEIF
`202611080800-full-csv`, ChEMBL `chembl_37`, Open Targets `26.09`, the FDA file date,
or openFDA `export_date`. Otherwise it is the UTC run time (`20261108T153554Z`), so API
samples never overwrite each other.

### Manifest fields

`source`, `dataset`, `mode`, `params` (redacted), `retrieved_at`, `finished_at`,
`source_release`, `snapshot_kind` (`query | full | delta | sample`), `status`
(`complete | sample | partial | failed`), `truncated` + `truncation_reasons`,
`pagination` (pages, records, provider total, complete), `files[]` (relative path,
role `native | extracted | normalized | dictionary`, format, delimiter, bytes,
sha256, rows, derived_from, redacted source_url), `transformations`,
`data_dictionaries`, `attribution`, `terms_urls`, `warnings`, `errors`,
`requests_made`. A failed run still writes its manifest, with `status: failed` and
the error.

## What normalization preserves

- **All values are text.** Leading zeros in CIKs, application and product numbers,
  ZIP codes and NCT ids survive. Delimiters are explicit (tab, `~`, `|`). A row with
  the wrong field count stops the conversion; a misaligned CSV is never written.
- **SEC:** one row per reported fact. Amended (`10-Q/A`) and repeated observations
  are all kept, along with accession numbers, form, filed date, fy/fp, frame and
  unit. Year-to-date durations are **not** turned into quarterly values; use
  `start`/`end`/`duration_days`/`fp`. Bulk SUB/NUM/TAG/PRE tables stay in their
  original tab-delimited form.
- **FRED:** `observations.csv` and `series.csv` are separate. The `.` missing marker
  becomes an empty cell only in `observations.csv`. Real-time and vintage fields are
  kept.
- **GLEIF:** `relationships.csv` (start_lei → end_lei, type, status, periods in
  `relationship_periods.csv`) and `reporting_exceptions.csv` are separate tables.
- **ClinicalTrials.gov:** `studies.csv` is a **selected-field** table. The native JSON
  pages are the complete record. Conditions, keywords, sponsors, arm groups,
  interventions, outcomes and locations are separate tables keyed by
  `nct_id + ordinal`, so arrays are never cross-joined. Results sections are
  aggregate, sponsor-reported data, never patient-level data, and remain in the
  native JSON only.
- **openFDA / ChEMBL / GLEIF entities / Open Targets GraphQL:** a generic flattener
  writes the parent table plus one child table per nested array (`<table>__<field>`).
  Each child table carries the parent key plus `<field>_ordinal`. Objects nested
  more than four levels deep are kept as JSON text and listed in `transformations`.
  The Orange Book API has no unique top-level key, so its rows carry
  `record_ordinal`, which is only meaningful within one snapshot.
- **ChEMBL activities:** `standard_type/relation/value/units` and the originally
  reported `type/relation/value/units` stay in separate columns. Different
  measurement types are never merged into one numeric column.
- **Open Targets Parquet → CSV:** streamed in 10,000-row batches. List and struct
  columns become JSON text and are named in the manifest.
- **FFIEC CDR bulk:** the MDRM code row becomes the CSV header, and the description
  row goes to `field_dictionary.csv`.

## Limitations and verification status (2026-11-08)

| Source / path | Status |
|---|---|
| ClinicalTrials.gov API, Drugs@FDA API, Orange Book API, GLEIF API (+ relationships), ChEMBL API, Open Targets GraphQL | **Live-tested** with bounded samples (`YEGI_LIVE_TESTS=1 uv run pytest -m live`), plus fixture tests. |
| GLEIF Golden Copy, ChEMBL release, Open Targets release, openFDA `download.json`, FDA Orange Book zip | Live `--dry-run` (listings, HEAD, manifest) only. Bulk transfers are **fixture-tested**; no bulk file was downloaded during development. |
| SEC EDGAR API and bulk | **Not live-verified.** `data.sec.gov` answered 403 to a generic development User-Agent. Set `SEC_USER_AGENT` to your organisation and contact email. URL patterns and the FSDS archive contents (`sub/num/tag/pre.txt`, `readme.htm`) were checked against sec.gov. |
| FRED / ALFRED | **Fixture-tested only** (no API key was available). Endpoints and parameters follow the current FRED docs. |
| FFIEC CDR REST API | **Fixture-tested only** (no PWS account). It follows spec CDR-PDD-SIS-611 v1.10. The facsimile response decoding accepts base64 or byte-array JSON, because the spec only says "byte stream array"; an unrecognised shape is saved raw with a warning. |
| FFIEC CDR bulk, Federal Reserve NIC, AACT | No documented automated download: the CDR bulk page is an ASP.NET form, NIC uses a CAPTCHA, and AACT requires sign-in. Supported as `archive` mode on a file you download; fixture-tested. AACT pipe files are parsed with standard quoting rules. If a table fails the field-count check, report the line it names. |
| Open Targets GraphQL | The schema changes between releases: 26.09 renamed drug fields to `maximumClinicalStage` / `maxClinicalStage`. Use `--query-file` for custom queries. The query used is saved in `native/query.graphql`. |
| GLEIF bulk | Only the **latest** publish can be selected. |

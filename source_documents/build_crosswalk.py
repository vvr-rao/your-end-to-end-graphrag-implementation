"""Build an entity crosswalk linking a document corpus to its structured data.

A crosswalk is a small table (companies, drugs, ...) with one row per entity:
hand-curated identifiers (SEC CIK, LEI, ChEMBL id, ...) from a JSON spec, plus
columns computed here:

  * how often the corpus documents mention the entity (regex aliases over the
    text the ingestion pipeline itself extracts -- backend document_io), and
  * coverage counts looked up in the sibling structured snapshots (SEC XBRL
    facts per CIK, GLEIF registration status, trials per drug, ...).

It writes one snapshot in the structured_download.py layout, so hf_dataset.py
uploads it with the rest of the bundle:

    <structured>/<output>/<UTC timestamp>/normalized/<table>.csv + manifest.json

Specs for the two test datasets live in source_documents/crosswalks/:

    uv run python source_documents/build_crosswalk.py source_documents/crosswalks/fin-longdoc.json
    uv run python source_documents/build_crosswalk.py source_documents/crosswalks/pharma30.json --dry-run

Spec format (JSON):

    {"corpus": "source_documents/<corpus>",           # repo-relative
     "structured": "source_documents/<bundle>",       # where snapshots live
     "output": "derived/crosswalk",                   # snapshot parent, under structured
     "dataset": "crosswalk", "description": "...", "warnings": ["..."],
     "tables": [{
        "name": "companies",                          # -> normalized/companies.csv
        "case_sensitive": true,                       # for alias matching
        "columns": ["company", "role", "sec_cik", "lei"],   # static, in order
        "enrich": ["sec_xbrl_facts", "lei_status"],   # computed lookups (see ENRICHERS)
        "rows": [{"company": "BHP Group Limited", "role": "subject",
                  "aliases": ["BHP"],                 # whole-word, regex-escaped
                  "pattern": "...",                   # OR a raw regex instead of aliases
                  "sec_cik": "811809", "lei": "WZE1..."}]}]}

An entity with neither aliases nor pattern gets 0 mentions (e.g. a second
listing of the same group). Mention counts are approximate: short aliases like
"Ford" or "GM" over-count.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOC_SUFFIXES = {".pdf", ".txt", ".docx", ".pptx", ".md"}
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class UsageError(Exception):
    pass


# ------------------------------------------------------------------ corpus


def extract_corpus(corpus: Path, loader: Callable[[Path], str] | None = None) -> dict[str, str]:
    """{file name: whitespace-normalised text} for every document in the corpus."""
    if loader is None:
        sys.path.insert(0, str(REPO))
        from backend.app.services import document_io

        def loader(p: Path) -> str:
            return document_io.load_document(p, warn_unreadable=False).text

    files = sorted(p for p in corpus.rglob("*")
                   if p.suffix.lower() in DOC_SUFFIXES and p.is_file())  # is_file follows symlinks
    if not files:
        raise UsageError(f"no documents (.pdf/.txt/.docx/.pptx/.md) under {corpus}")
    out = {}
    for i, p in enumerate(files, 1):
        print(f"  extracting {i}/{len(files)} {p.name[:70]}", file=sys.stderr, flush=True)
        out[p.relative_to(corpus).as_posix()] = re.sub(r"\s+", " ", loader(p) or "")
    return out


def mention_regex(row: dict, case_sensitive: bool) -> re.Pattern | None:
    if row.get("pattern"):
        rx = row["pattern"]
    else:
        alts = [re.escape(a) for a in row.get("aliases") or [] if a.strip()]
        if not alts:
            return None
        rx = r"\b(?:" + "|".join(alts) + r")\b"
    return re.compile(rx, 0 if case_sensitive else re.IGNORECASE)


def count_mentions(rx: re.Pattern | None, texts: dict[str, str]) -> dict[str, int]:
    if rx is None:
        return {}
    per = {name: len(rx.findall(t)) for name, t in texts.items()}
    return {k: v for k, v in sorted(per.items(), key=lambda kv: (-kv[1], kv[0])) if v}


# ------------------------------------------------------- structured lookups


def latest(structured: Path, pattern: str) -> Path | None:
    """Newest snapshot file matching <structured>/<pattern> (snapshot dirs sort by timestamp)."""
    hits = sorted(glob.glob(str(structured / pattern)))
    return Path(hits[-1]) if hits else None


def read_rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Lookups:
    """Lazy, cached views over the bundle's snapshots; missing sources -> None + warning."""

    def __init__(self, structured: Path):
        self.root = structured
        self.warnings: list[str] = []
        self._cache: dict[str, object] = {}
        self.used: dict[str, str] = {}

    def _get(self, key: str, pattern: str, build: Callable[[Path], object]):
        if key not in self._cache:
            p = latest(self.root, pattern)
            if p is None:
                self.warnings.append(f"{key}: no snapshot matching {pattern} under {self.root}; column left empty")
                self._cache[key] = None
            else:
                self.used[key] = p.relative_to(self.root).as_posix()
                self._cache[key] = build(p)
        return self._cache[key]

    def sec_facts(self):
        def build(p):
            c = collections.Counter()
            with p.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    c[r["cik"]] += 1
            return c
        return self._get("sec_xbrl_facts", "sec/companyfacts/*/normalized/facts.csv", build)

    def lei_status(self):
        def build(p):
            out = {}
            for r in read_rows(p):
                lei = r.get("lei") or r.get("id")
                st = next((v for k, v in r.items() if k.endswith("registration__status")), "")
                out[lei] = st
            return out
        return self._get("lei_status", "gleif/lei-records/*/normalized/entities.csv", build)

    def ctgov_interventions(self):
        return self._get("ctgov_trials", "clinicaltrials/studies/*/normalized/interventions.csv",
                         lambda p: [(r["nct_id"], (r.get("name") or "").lower()) for r in read_rows(p)])

    def cms(self):
        def build(p):
            rows = read_rows(p)
            g = next((c for c in (rows[0] if rows else {}) if c.lower().startswith("gnrc_name")), None)
            return [r[g].lower() for r in rows] if g else []
        return self._get("cms_partd_rows", "cms/partd-spending/*/normalized/study_drugs.csv", build)

    def faers(self):
        return self._get("faers_reports", "openfda/faers-reaction-counts/*/normalized/report_totals.csv",
                         lambda p: {r["drug"].lower(): r["reports"] for r in read_rows(p)})

    def rxnorm(self):
        return self._get("rxnorm_rxcui", "rxnorm/drug-crosswalk/*/normalized/ingredients.csv",
                         lambda p: {r["ingredient"].lower(): r["rxcui"] for r in read_rows(p)})

    def ot_indications(self):
        return self._get("opentargets_indications", "opentargets/drug/*/normalized/drug__indications__rows.csv",
                         lambda p: collections.Counter(r["id"] for r in read_rows(p)))


def _cik(row: dict) -> str:
    c = str(row.get("sec_cik") or "").strip()
    return c.zfill(10) if c.isdigit() else ""


def _drug(row: dict) -> str:
    return str(row.get("drug") or "").lower()


def _sec_facts(row, lk):
    c = lk.sec_facts()
    return "" if c is None or not _cik(row) else c.get(_cik(row), 0)


def _lei_status(row, lk):
    s = lk.lei_status()
    return "" if s is None else s.get(row.get("lei") or "", "")


def _ctgov(row, lk):
    rows = lk.ctgov_interventions()
    return "" if rows is None else len({n for n, name in rows if _drug(row) and _drug(row) in name})


def _cms(row, lk):
    rows = lk.cms()
    return "" if rows is None else sum(1 for g in rows if _drug(row) and _drug(row) in g)


def _faers(row, lk):
    m = lk.faers()
    return "" if m is None else m.get(_drug(row), "")


def _rxnorm(row, lk):
    m = lk.rxnorm()
    return "" if m is None else m.get(_drug(row), "")


def _ot(row, lk):
    c = lk.ot_indications()
    return "" if c is None else c.get(row.get("chembl_id") or "", 0)


ENRICHERS: dict[str, tuple[Callable, str]] = {
    "sec_xbrl_facts": (_sec_facts, "rows in sec/companyfacts facts.csv for sec_cik"),
    "lei_status": (_lei_status, "registration status of lei in the gleif/lei-records download"),
    "ctgov_trials": (_ctgov, "studies in clinicaltrials/studies whose intervention name contains the drug"),
    "cms_partd_rows": (_cms, "rows in cms/partd-spending study_drugs.csv whose generic name contains the drug"),
    "faers_reports": (_faers, "openFDA FAERS report total for the drug"),
    "rxnorm_rxcui": (_rxnorm, "RxNorm ingredient RxCUI(s) from rxnorm/drug-crosswalk"),
    "opentargets_indications": (_ot, "Open Targets indication rows for chembl_id"),
}


# ------------------------------------------------------------------- build


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for k in ("corpus", "structured", "tables"):
        if k not in spec:
            raise UsageError(f"{path}: spec needs '{k}'")
    for t in spec["tables"]:
        bad = [e for e in t.get("enrich", []) if e not in ENRICHERS]
        if bad:
            raise UsageError(f"table {t.get('name')}: unknown enrich {bad}; choose from {sorted(ENRICHERS)}")
        for r in t["rows"]:
            missing = [c for c in t["columns"] if c not in r]
            if missing:
                raise UsageError(f"table {t['name']}: row {r} lacks {missing}")
            if r.get("pattern"):
                re.compile(r["pattern"])  # fail early on a bad regex
    return spec


def build_tables(spec: dict, texts: dict[str, str], lk: Lookups) -> dict[str, list[dict]]:
    out = {}
    for t in spec["tables"]:
        rows = []
        for r in t["rows"]:
            per = count_mentions(mention_regex(r, t.get("case_sensitive", False)), texts)
            row = {c: (_cik(r) if c == "sec_cik" else r[c]) for c in t["columns"]}  # CIKs as SEC's 10 digits
            row["mentions_total"] = sum(per.values())
            row["documents_mentioning"] = len(per)
            row["mentions_by_document"] = "; ".join(f"{k}={v}" for k, v in per.items())
            for e in t.get("enrich", []):
                row[e] = ENRICHERS[e][0](r, lk)
            rows.append(row)
        out[t["name"]] = rows
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_snapshot(spec: dict, tables: dict[str, list[dict]], lk: Lookups, structured: Path,
                   ts: str, overwrite: bool) -> Path:
    snap = structured / spec.get("output", "derived/crosswalk") / ts
    if snap.exists() and any(snap.iterdir()) and not overwrite:
        raise UsageError(f"{snap} already exists (pass --overwrite)")
    (snap / "normalized").mkdir(parents=True, exist_ok=True)
    files = []
    for name, rows in tables.items():
        p = snap / "normalized" / f"{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
            w.writeheader()
            w.writerows(rows)
        files.append({"path": f"normalized/{name}.csv", "role": "normalized", "format": "csv",
                      "bytes": p.stat().st_size, "sha256": sha256(p), "rows": len(rows)})
    enrich = sorted({e for t in spec["tables"] for e in t.get("enrich", [])})
    m = {"source": "derived", "dataset": spec.get("dataset", "crosswalk"), "mode": "derived",
         "status": "complete", "snapshot_kind": "full",
         "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
         "description": spec.get("description", ""),
         "params": {"corpus": spec["corpus"], "inputs": lk.used},
         "files": files,
         "transformations": [
             "mentions_* = whole-word alias (or raw pattern) matches in text extracted with "
             "backend/app/services/document_io, whitespace-normalised; approximate",
             *[f"{e}: {ENRICHERS[e][1]}" for e in enrich]],
         "warnings": [*spec.get("warnings", []), *lk.warnings]}
    (snap / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return snap


def main(argv: list[str] | None = None, loader: Callable[[Path], str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("spec", type=Path, help="crosswalk spec (JSON)")
    ap.add_argument("--root", type=Path, default=REPO, help="resolve spec paths against this (default: repo root)")
    ap.add_argument("--snapshot", help="snapshot folder name (default: UTC timestamp)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the tables; write nothing")
    a = ap.parse_args(argv)
    try:
        spec = load_spec(a.spec)
        corpus, structured = a.root / spec["corpus"], a.root / spec["structured"]
        if not corpus.is_dir():
            raise UsageError(f"corpus not found: {corpus}")
        if not structured.is_dir():
            raise UsageError(f"structured bundle not found: {structured}")
        texts = extract_corpus(corpus, loader)
        lk = Lookups(structured)
        tables = build_tables(spec, texts, lk)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for name, rows in tables.items():
        print(f"== {name}: {len(rows)} rows")
        for r in rows:
            first = next(iter(r.values()))
            extra = "  ".join(f"{k}={r[k]}" for k in r if k not in ("mentions_by_document",) and k != next(iter(r)))
            print(f"  {str(first)[:36]:<36} {extra[:150]}")
    for w in lk.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    try:
        snap = write_snapshot(spec, tables, lk, structured, a.snapshot or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                              a.overwrite)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""source_documents/build_crosswalk.py: mention counting, lookups, snapshot (offline)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import build_crosswalk as bc


def write_csv(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_repo(tmp_path: Path) -> Path:
    corpus = tmp_path / "source_documents" / "mini"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("Ozempic (semaglutide) by Novo Nordisk. OZEMPIC again.\nBHP and bhp.")
    (corpus / "b.txt").write_text("Semaglutide vs  tirzepatide. Ford Fiesta, Bedford.")
    (corpus / "skip.json").write_text("{}")
    b = tmp_path / "source_documents" / "mini-structured"
    write_csv(b / "sec/companyfacts/20260101T000000Z/normalized/facts.csv",
              [{"cik": "0000811809", "val": "1"}, {"cik": "0000811809", "val": "2"}])
    write_csv(b / "sec/companyfacts/20250101T000000Z/normalized/facts.csv",   # older snapshot: ignored
              [{"cik": "0000811809", "val": "1"}])
    write_csv(b / "gleif/lei-records/20260101T000000Z/normalized/entities.csv",
              [{"lei": "WZE1", "attributes__registration__status": "ISSUED"}])
    write_csv(b / "clinicaltrials/studies/s/normalized/interventions.csv",
              [{"nct_id": "NCT1", "name": "Semaglutide"}, {"nct_id": "NCT1", "name": "semaglutide oral"},
               {"nct_id": "NCT2", "name": "Placebo"}])
    return tmp_path


def spec(tmp_path: Path) -> Path:
    s = {"corpus": "source_documents/mini", "structured": "source_documents/mini-structured",
         "output": "derived/crosswalk", "warnings": ["hand-curated"],
         "tables": [
             {"name": "drugs", "case_sensitive": False, "columns": ["drug"],
              "enrich": ["ctgov_trials", "faers_reports"],
              "rows": [{"drug": "semaglutide", "aliases": ["semaglutide", "Ozempic"]},
                       {"drug": "nothing", "aliases": []}]},
             {"name": "companies", "case_sensitive": True, "columns": ["company", "sec_cik", "lei"],
              "enrich": ["sec_xbrl_facts", "lei_status"],
              "rows": [{"company": "BHP", "pattern": r"\bBHP\b", "sec_cik": "811809", "lei": "WZE1"},
                       {"company": "Ford", "aliases": ["Ford"], "sec_cik": "", "lei": "NONE"}]}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    return p


def read(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(tmp_path, *extra):
    return bc.main([str(spec(tmp_path)), "--root", str(tmp_path), "--snapshot", "s1", *extra],
                   loader=lambda p: p.read_text())


def test_builds_tables_with_mentions_and_lookups(tmp_path):
    make_repo(tmp_path)
    assert run(tmp_path) == 0
    snap = tmp_path / "source_documents/mini-structured/derived/crosswalk/s1"
    drugs = {r["drug"]: r for r in read(snap / "normalized/drugs.csv")}
    sem = drugs["semaglutide"]
    assert sem["mentions_total"] == "4" and sem["documents_mentioning"] == "2"   # case-insensitive
    assert sem["mentions_by_document"] == "a.txt=3; b.txt=1"
    assert sem["ctgov_trials"] == "1"                 # NCT1 counted once
    assert sem["faers_reports"] == ""                 # no FAERS snapshot -> empty + warning
    assert drugs["nothing"]["mentions_total"] == "0"
    cos = {r["company"]: r for r in read(snap / "normalized/companies.csv")}
    assert cos["BHP"]["mentions_total"] == "1"        # case-sensitive: 'bhp' not counted
    assert cos["BHP"]["sec_cik"] == "0000811809"      # padded to SEC's 10 digits
    assert cos["BHP"]["sec_xbrl_facts"] == "2"        # newest snapshot only
    assert cos["BHP"]["lei_status"] == "ISSUED"
    assert cos["Ford"]["mentions_total"] == "1"       # whole word: 'Bedford' not counted
    assert cos["Ford"]["sec_xbrl_facts"] == "" and cos["Ford"]["lei_status"] == ""
    m = json.loads((snap / "manifest.json").read_text())
    assert m["source"] == "derived" and m["status"] == "complete"
    assert m["params"]["inputs"]["sec_xbrl_facts"].startswith("sec/companyfacts/20260101T000000Z/")
    assert "hand-curated" in m["warnings"] and any("faers_reports" in w for w in m["warnings"])
    assert {f["path"] for f in m["files"]} == {"normalized/drugs.csv", "normalized/companies.csv"}


def test_dry_run_writes_nothing(tmp_path, capsys):
    make_repo(tmp_path)
    assert run(tmp_path, "--dry-run") == 0
    assert not (tmp_path / "source_documents/mini-structured/derived").exists()
    assert "dry run" in capsys.readouterr().out


def test_existing_snapshot_needs_overwrite(tmp_path, capsys):
    make_repo(tmp_path)
    assert run(tmp_path) == 0
    assert run(tmp_path) == 2 and "--overwrite" in capsys.readouterr().err
    assert run(tmp_path, "--overwrite") == 0


def test_bad_spec_is_rejected(tmp_path, capsys):
    make_repo(tmp_path)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"corpus": "source_documents/mini", "structured": "source_documents/mini-structured",
                             "tables": [{"name": "t", "columns": ["x"], "enrich": ["nope"], "rows": []}]}))
    assert bc.main([str(p), "--root", str(tmp_path)], loader=lambda p: "") == 2
    assert "unknown enrich" in capsys.readouterr().err


def test_shipped_specs_are_valid():
    for p in sorted((bc.HERE / "crosswalks").glob("*.json")):
        s = bc.load_spec(p)
        assert s["tables"] and all(t["rows"] for t in s["tables"]), p.name

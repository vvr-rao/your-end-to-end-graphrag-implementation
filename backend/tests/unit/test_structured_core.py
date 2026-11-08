"""Shared structured-download behaviour: HTTP retries, atomic downloads,
safe extraction, manifests, delimited tables and JSON flattening."""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile

import pytest
import requests
from structured import _archive, _manifest, _tabular
from structured._http import (
    DownloadError,
    HttpClient,
    UnexpectedContentError,
    _parse_retry_after,
)

from backend.tests.unit.structured_fakes import FakeResponse, FakeSession


def make_client(handler, sleeps=None, **kw):
    sess = FakeSession(handler)
    client = HttpClient(contact="test", session=sess, log=lambda m: None,
                        sleep=(sleeps.append if sleeps is not None else (lambda s: None)), **kw)
    return client, sess


# ------------------------------------------------------------------ retries


def test_retry_honours_retry_after_then_succeeds():
    replies = [FakeResponse(429, "slow down", {"Retry-After": "7"}),
               FakeResponse(503, "busy"), FakeResponse(200, {"ok": True})]
    sleeps: list[float] = []
    client, sess = make_client(lambda *a: replies.pop(0), sleeps, max_retries=3)
    assert client.get_json("https://x.test/a") == {"ok": True}
    assert len(sess.calls) == 3
    assert sleeps[0] == 7.0                       # Retry-After used verbatim
    assert 0 < sleeps[1] <= 60                    # bounded exponential backoff


def test_retries_are_bounded_and_error_is_actionable():
    client, sess = make_client(lambda *a: FakeResponse(503, "down"), max_retries=2)
    with pytest.raises(DownloadError, match="HTTP 503"):
        client.get_json("https://x.test/a")
    assert len(sess.calls) == 3                   # 1 try + 2 retries


def test_excessive_retry_after_fails_fast():
    client, _ = make_client(lambda *a: FakeResponse(429, "", {"Retry-After": "3600"}),
                            max_retry_after=60)
    with pytest.raises(DownloadError, match="Retry-After"):
        client.get_json("https://x.test/a")


def test_timeout_retries_then_reports_hint():
    client, sess = make_client(lambda *a: requests.Timeout("t"), max_retries=1)
    with pytest.raises(DownloadError, match="--timeout"):
        client.get_json("https://x.test/a")
    assert len(sess.calls) == 2


def test_retry_after_http_date():
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:10 GMT", now=1445412480.0) == 10.0
    assert _parse_retry_after("garbage") is None


def test_auth_params_never_appear_in_error_messages():
    client, sess = make_client(lambda *a: FakeResponse(400, "bad"),
                               auth_params={"api_key": "SECRET123"}, max_retries=0)
    with pytest.raises(DownloadError) as exc:
        client.get_json("https://x.test/a", params={"q": "1"})
    assert "SECRET123" not in str(exc.value)
    assert sess.calls[0]["params"]["api_key"] == "SECRET123"   # but it was sent


# ---------------------------------------------------------------- downloads


ZIP_BYTES = b"PK\x03\x04" + b"x" * 5000


def test_download_atomic_with_checksum(tmp_path):
    client, _ = make_client(lambda *a: FakeResponse(200, ZIP_BYTES,
                                                    {"Content-Length": str(len(ZIP_BYTES))}))
    res = client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")
    import hashlib
    assert res.sha256 == hashlib.sha256(ZIP_BYTES).hexdigest()
    assert (tmp_path / "f.zip").read_bytes() == ZIP_BYTES
    assert not (tmp_path / "f.zip.part").exists()


def test_html_login_page_rejected_and_nothing_saved(tmp_path):
    html = b"<!DOCTYPE html><html><body>Please sign in</body></html>"
    client, _ = make_client(lambda *a: FakeResponse(200, html, {"Content-Type": "text/html"}))
    with pytest.raises(UnexpectedContentError, match="HTML page"):
        client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")
    assert list(tmp_path.iterdir()) == []


def test_wrong_magic_bytes_rejected(tmp_path):
    client, _ = make_client(lambda *a: FakeResponse(200, b"not a zip at all"))
    with pytest.raises(UnexpectedContentError, match="first bytes"):
        client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")


def test_interrupted_download_leaves_only_part_file(tmp_path):
    chunks = [ZIP_BYTES[:1000], ZIP_BYTES[1000:2000], ZIP_BYTES[2000:]]
    client, _ = make_client(lambda *a: FakeResponse(200, ZIP_BYTES, chunks=chunks,
                                                    fail_after=2))
    with pytest.raises(DownloadError, match="--resume"):
        client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")
    assert not (tmp_path / "f.zip").exists()
    assert (tmp_path / "f.zip.part").stat().st_size == 2000


def test_short_body_vs_content_length_is_an_error(tmp_path):
    client, _ = make_client(lambda *a: FakeResponse(200, ZIP_BYTES[:100],
                                                    {"Content-Length": "5000"}))
    with pytest.raises(DownloadError, match="announced"):
        client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")
    assert not (tmp_path / "f.zip").exists()


def test_resume_uses_range_only_when_advertised(tmp_path):
    (tmp_path / "f.zip.part").write_bytes(ZIP_BYTES[:2000])

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {"Accept-Ranges": "bytes"})
        assert headers.get("Range") == "bytes=2000-"
        rest = ZIP_BYTES[2000:]
        return FakeResponse(206, rest, {"Content-Length": str(len(rest))})

    client, _ = make_client(handler)
    res = client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip", resume=True)
    assert res.resumed and (tmp_path / "f.zip").read_bytes() == ZIP_BYTES
    import hashlib
    assert res.sha256 == hashlib.sha256(ZIP_BYTES).hexdigest()


def test_resume_restarts_when_ranges_not_supported(tmp_path):
    (tmp_path / "f.zip.part").write_bytes(b"stale-partial")

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {})
        assert "Range" not in headers
        return FakeResponse(200, ZIP_BYTES)

    client, _ = make_client(handler)
    res = client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip", resume=True)
    assert not res.resumed and (tmp_path / "f.zip").read_bytes() == ZIP_BYTES



def test_download_asks_for_identity_encoding(tmp_path):
    # GLEIF compresses on the fly when gzip is accepted and then ignores Range.
    seen = []
    client, _ = make_client(lambda m, u, p, headers, b: (seen.append(headers.copy()),
                                                         FakeResponse(200, ZIP_BYTES))[1])
    client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip")
    assert seen[-1]["Accept-Encoding"] == "identity"


def test_resume_logs_when_server_ignores_range(tmp_path):
    (tmp_path / "f.zip.part").write_bytes(ZIP_BYTES[:2000])
    logs = []

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {"Accept-Ranges": "bytes"})
        return FakeResponse(200, ZIP_BYTES)

    client = HttpClient(contact="test", session=FakeSession(handler), log=logs.append,
                        sleep=lambda s: None)
    res = client.download("https://x.test/f.zip", tmp_path / "f.zip", expect="zip", resume=True)
    assert not res.resumed and (tmp_path / "f.zip").read_bytes() == ZIP_BYTES
    assert any("ignored the byte range" in m for m in logs)

# ------------------------------------------------------------------ archives


def _zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt",
                                 "C:/evil.txt"])
def test_zip_traversal_rejected(tmp_path, bad):
    arc = _zip(tmp_path / "a.zip", {"ok.txt": b"1", bad: b"x"})
    with pytest.raises(_archive.UnsafeArchiveError):
        _archive.safe_extract(arc, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out" / "ok.txt").exists()     # nothing written before checks


def test_zip_symlink_rejected(tmp_path):
    arc = tmp_path / "s.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(_archive.UnsafeArchiveError, match="symlink"):
        _archive.safe_extract(arc, tmp_path / "out")


def test_tar_link_rejected(tmp_path):
    arc = tmp_path / "t.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(_archive.UnsafeArchiveError, match="link"):
        _archive.safe_extract(arc, tmp_path / "out")


def test_archive_size_cap(tmp_path):
    arc = _zip(tmp_path / "big.zip", {"a.txt": b"0" * 10_000})
    with pytest.raises(_archive.UnsafeArchiveError, match="expands"):
        _archive.safe_extract(arc, tmp_path / "out", max_bytes=1000)


def test_safe_extract_zip_and_tar(tmp_path):
    arc = _zip(tmp_path / "a.zip", {"dir/x.txt": b"hello"})
    out = _archive.safe_extract(arc, tmp_path / "out")
    assert [p.read_bytes() for p in out] == [b"hello"]
    tgz = tmp_path / "b.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        data = b"world"
        info = tarfile.TarInfo("y.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out = _archive.safe_extract(tgz, tmp_path / "out2")
    assert [p.read_bytes() for p in out] == [b"world"]


# ------------------------------------------------------------------ manifest


def test_redaction_of_params_and_urls():
    url = "https://user:pw@api.test/x?api_key=SECRET&series_id=GDP&token=T&pageToken=abc"
    red = _manifest.redact_url(url, {"access_token": "Z", "limit": 5})
    for secret in ("SECRET", "user:pw", "=T&", "Z"):
        assert secret not in red
    assert "series_id=GDP" in red and "pageToken=abc" in red and "limit=5" in red
    assert _manifest.redact_params({"api_key": "k", "lei": "X"}) == {"api_key": "***", "lei": "X"}


def test_truncated_run_is_never_complete(tmp_path):
    m = _manifest.Manifest(source="s", dataset="d", mode="api")
    m.truncate("--limit 5 reached")
    m.finish()
    assert m.status == "sample" and m.truncated and m.snapshot_kind == "sample"
    m2 = _manifest.Manifest(source="s", dataset="d", mode="api", errors=["boom"])
    f = tmp_path / "x.json"
    f.write_text("{}")
    m2.add_file(tmp_path, f, role="native", format="json")
    m2.finish()
    assert m2.status == "partial"


def test_manifest_entries_and_verification(tmp_path):
    f = tmp_path / "native" / "a.txt"
    f.parent.mkdir()
    f.write_text("a~b\n1~2\n")
    m = _manifest.Manifest(source="s", dataset="d", mode="bulk", params={"api_key": "x"})
    e = m.add_file(tmp_path, f, role="native", format="txt", delimiter="tilde", rows=1)
    n = tmp_path / "normalized" / "a.csv"
    n.parent.mkdir()
    n.write_text("a,b\n1,2\n")
    m.add_file(tmp_path, n, role="normalized", format="csv", derived_from=[f])
    m.finish()
    m.write(tmp_path)
    d = _manifest.load_manifest(tmp_path)
    assert d["params"]["api_key"] == "***"
    assert d["files"][0]["path"] == "native/a.txt" and d["files"][0]["sha256"] == e.sha256
    assert d["files"][1]["derived_from"] == ["native/a.txt"]
    for key in ("source", "dataset", "mode", "retrieved_at", "status", "truncated",
                "pagination", "transformations", "data_dictionaries", "terms_urls",
                "attribution", "warnings", "errors"):
        assert key in d
    assert _manifest.verified_entry(d, tmp_path, f)
    f.write_text("tampered")
    assert _manifest.verified_entry(d, tmp_path, f) is None


# ------------------------------------------------------------------ tabular


@pytest.mark.parametrize("delim,name", [("\t", "tab"), ("~", "tilde"), ("|", "pipe")])
def test_delimited_conversion_keeps_leading_zeros(tmp_path, delim, name):
    src = tmp_path / f"t.{name}.txt"
    src.write_text(delim.join(["Appl_No", "Product_No", "Name"]) + "\n"
                   + delim.join(["004567", "001", 'Brand "X", 5mg']) + "\n", encoding="utf-8")
    conv = _tabular.convert_delimited(src, tmp_path / "t.csv", delimiter=delim)
    assert conv.rows == 1 and conv.header == ["Appl_No", "Product_No", "Name"]
    assert conv.rejected == 0
    with (tmp_path / "t.csv").open(newline="", encoding="utf-8") as fh:
        data = list(csv.reader(fh))
    assert data[1] == ["004567", "001", 'Brand "X", 5mg']


def test_wrong_delimiter_fails_loudly(tmp_path):
    src = tmp_path / "t.txt"
    src.write_text("a~b~c\n1~2~3\n4~5\n")
    with pytest.raises(ValueError, match="line 3"):
        _tabular.convert_delimited(src, tmp_path / "t.csv", delimiter="~")
    assert not (tmp_path / "t.csv").exists()


def test_isolated_ragged_row_is_quarantined_not_repaired(tmp_path):
    # Real case (Drugs@FDA ApplicationDocs.txt, 2026-11-06): one row of 81k carries
    # an extra empty field between URL and date. Never shift it; quarantine it.
    good = [f"{i}\t2\t21949{i}\tORIG\thttp://x/{i}.pdf\t2026-01-05" for i in range(200)]
    bad = "84630\t2\t206627\tSUPPL\thttp://x/b.pdf\t\t2026-01-06"
    src = tmp_path / "ApplicationDocs.txt"
    src.write_text("ID\tType\tApplNo\tSub\tURL\tDate\r\n"
                   + "\r\n".join([*good[:100], bad, *good[100:]]) + "\r\n")
    rej = tmp_path / "ApplicationDocs.rejects.csv"
    conv = _tabular.convert_delimited(src, tmp_path / "out.csv", delimiter="\t", rejects=rej)
    assert (conv.rows, conv.rejected) == (200, 1)
    with rej.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["line_number"] == "102" and rows[0]["field_count"] == "7"
    assert json.loads(rows[0]["fields_json"])[0] == "84630"
    out = (tmp_path / "out.csv").read_text()
    assert "84630" not in out and out.count("\n") == 201


def test_widespread_ragged_rows_still_fail(tmp_path):
    src = tmp_path / "t.txt"
    src.write_text("a~b~c\n" + "1~2\n" * 50 + "1~2~3\n" * 50)
    with pytest.raises(ValueError, match="misaligned"):
        _tabular.convert_delimited(src, tmp_path / "t.csv", delimiter="~",
                                   rejects=tmp_path / "r.csv")
    assert not (tmp_path / "t.csv").exists() and not (tmp_path / "r.csv").exists()


def test_cp1252_input_detected(tmp_path):
    src = tmp_path / "t.txt"
    src.write_bytes("Name\tCity\nCaf\xe9\tS\xe3o\n".encode("cp1252"))
    conv = _tabular.convert_delimited(src, tmp_path / "t.csv", delimiter="\t")
    assert conv.encoding == "cp1252" and "Café" in (tmp_path / "t.csv").read_text(encoding="utf-8")


def test_null_markers_only_in_conversion(tmp_path):
    src = tmp_path / "t.txt"
    src.write_text("date\tvalue\n2020-01-01\t.\n2020-02-01\t1.5\n")
    _tabular.convert_delimited(src, tmp_path / "t.csv", delimiter="\t", null_markers=(".",))
    assert (tmp_path / "t.csv").read_text().splitlines()[1] == "2020-01-01,"
    assert "\t.\n" in src.read_text()                      # native untouched


def test_flattener_separates_independent_arrays():
    rec = {"id": "R1", "name": "n", "meta": {"a": 1},
           "conditions": ["c1", "c2"],
           "locations": [{"city": "x"}, {"city": "y"}, {"city": "z"}]}
    flat = _tabular.JsonFlattener("rec", lambda r, i: {"id": r["id"]}, skip_fields=("id",))
    rows = flat.rows(rec)
    tables = {}
    for t, r in rows:
        tables.setdefault(t, []).append(r)
    assert len(tables["rec"]) == 1 and tables["rec"][0]["meta__a"] == 1
    assert len(tables["rec__conditions"]) == 2
    assert len(tables["rec__locations"]) == 3                 # 2 + 3 rows, never 6
    assert tables["rec__locations"][2] == {"id": "R1", "locations_ordinal": 3, "city": "z"}


def test_flattener_writes_union_of_columns(tmp_path):
    recs = [{"k": "1", "a": "x"}, {"k": "2", "b": "y"}]
    flat = _tabular.JsonFlattener("t", lambda r, i: {"k": r["k"]}, skip_fields=("k",))
    tables = flat.write(lambda: iter(recs), tmp_path)
    text = tables["t"].path.read_text().splitlines()
    assert text == ["k,a,b", "1,x,", "2,,y"]

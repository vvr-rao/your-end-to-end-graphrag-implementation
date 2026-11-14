"""source_documents/hf_dataset.py against an in-memory fake Hugging Face Hub
(no network, no token)."""

from __future__ import annotations

import json
from pathlib import Path

import hf_dataset
import pytest

TOKEN = "hf_test_SECRET_do_not_print"


class NotFoundError(Exception):
    """Stands in for huggingface_hub's RepositoryNotFoundError / EntryNotFoundError."""

    class _R:
        status_code = 404

    response = _R()


class FakeInfo:
    def __init__(self, private: bool):
        self.private = private


class FakeHub:
    """Just enough of HfApi: repos are dicts of path -> bytes."""

    def __init__(self, tmp: Path, user: str = "alice"):
        self.user = user
        self.repos: dict[str, dict[str, bytes]] = {}
        self.private: dict[str, bool] = {}
        self.commits: list[tuple[str, list[str], list[str]]] = []
        self.tmp = tmp
        self.downloads = 0

    def __call__(self, token):  # api_factory
        self.token = token
        return self

    def whoami(self):
        return {"name": self.user}

    def repo_info(self, repo_id, repo_type=None):
        assert repo_type == "dataset"
        if repo_id not in self.repos:
            raise NotFoundError(repo_id)
        return FakeInfo(self.private[repo_id])

    def create_repo(self, repo_id, repo_type=None, private=None):
        assert repo_type == "dataset" and private is True
        self.repos[repo_id] = {}
        self.private[repo_id] = True

    def create_commit(self, repo_id, operations, repo_type=None, commit_message=""):
        added, deleted = [], []
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.repos[repo_id].pop(op.path_in_repo)
                deleted.append(op.path_in_repo)
                continue
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            self.repos[repo_id][op.path_in_repo] = data
            added.append(op.path_in_repo)
        self.commits.append((commit_message, added, deleted))

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
        files = self.repos.get(repo_id)
        if files is None or filename not in files:
            raise NotFoundError(filename)
        out = self.tmp / "hub-cache" / repo_id / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(files[filename])
        return str(out)

    def list_repo_files(self, repo_id, repo_type=None, revision=None):
        return sorted(self.repos[repo_id])

    def snapshot_download(self, repo_id, repo_type=None, revision=None, local_dir=None,
                          allow_patterns=None):
        for path, data in self.repos[repo_id].items():
            if allow_patterns is not None and path not in allow_patterns:
                continue
            out = Path(local_dir) / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            self.downloads += 1
        return local_dir


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.setattr(hf_dataset, "load_env", lambda: None)   # never read the real .env
    monkeypatch.setenv("HF_TOKEN", TOKEN)


@pytest.fixture
def data(tmp_path):
    corpus = tmp_path / "pharma-mini"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha")
    (corpus / "sub").mkdir()
    (corpus / "sub" / "b.pdf").write_bytes(b"%PDF-1.4 b")
    # things that must never be uploaded
    (corpus / ".env").write_text("OPENAI_API_KEY=sk-x")
    (corpus / "old.bak").write_text("x")
    (corpus / "__pycache__").mkdir()
    (corpus / "__pycache__" / "m.pyc").write_bytes(b"x")
    (corpus / "big.zip.part").write_bytes(b"x")

    structured = tmp_path / "structured_sec"
    snap = structured / "fsds" / "2024q1"
    (snap / "native").mkdir(parents=True)
    (snap / "native" / "sub.txt").write_text("adsh\tcik\n0001\t000320193\n")
    (snap / "manifest.json").write_text(json.dumps({
        "source": "sec", "dataset": "fsds", "status": "complete", "snapshot_kind": "full",
        "attribution": "U.S. SEC", "terms_urls": ["https://www.sec.gov/privacy"]}))
    bad = structured / "fsds" / "2024q2"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps(
        {"source": "sec", "dataset": "fsds", "status": "failed"}))

    evals = tmp_path / "testsets"
    evals.mkdir()
    (evals / "pharma-mini.json").write_text(json.dumps(
        {"format": "graphrag-testset/1", "corpus": "source_documents/pharma-mini",
         "questions": [{"id": "q1", "question": "?", "gold_documents": ["a.txt"]}]}))
    return {"corpus": corpus, "structured": structured, "eval": evals}


def push(hub, data, *extra):
    return hf_dataset.main(["push", "--repo", "yegi-test", "--corpus", str(data["corpus"]),
                            "--structured", str(data["structured"]), "--eval", str(data["eval"]),
                            *extra], api_factory=hub)


def test_push_creates_private_repo_with_manifest_and_card(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    files = hub.repos["alice/yegi-test"]
    assert hub.private["alice/yegi-test"] is True
    assert set(files) == {
        "unstructured/pharma-mini/a.txt", "unstructured/pharma-mini/sub/b.pdf",
        "structured/sec/fsds/2024q1/native/sub.txt", "structured/sec/fsds/2024q1/manifest.json",
        "eval/testsets/pharma-mini.json", "yegi_manifest.json", "README.md"}
    m = json.loads(files["yegi_manifest.json"])
    restore = {d["name"]: d["restore_to"] for d in m["items"]}
    assert restore["pharma-mini"] == "source_documents/pharma-mini"
    assert restore["sec/fsds/2024q1"] == "source_documents/structured_sec/fsds/2024q1"
    assert restore["testsets"] == "eval_questions/testsets"
    sec = next(d for d in m["items"] if d["kind"] == "structured")
    assert sec["meta"]["status"] == "complete" and sec["meta"]["attribution"] == "U.S. SEC"
    card = files["README.md"].decode()
    assert "viewer: false" in card and hf_dataset.CARD_MARKER in card and "U.S. SEC" in card
    out = capsys.readouterr()
    assert "2024q2 (manifest status failed" in out.out          # failed snapshot skipped
    assert ".env" in out.out and "may contain credentials" in out.out
    assert TOKEN not in out.out + out.err


def test_dry_run_needs_no_token_and_uploads_nothing(tmp_path, data, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN")
    hub = FakeHub(tmp_path)
    assert push(hub, data, "--dry-run") == 0
    assert hub.repos == {} and hub.commits == []
    out = capsys.readouterr().out
    assert "remote not checked" in out and "dry run: nothing uploaded" in out
    assert "upload: 5 files" in out


def test_push_without_token_is_a_clear_error(tmp_path, data, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN")
    assert push(FakeHub(tmp_path), data) == 2
    assert "HF_TOKEN is not set" in capsys.readouterr().err


def test_refuses_to_write_to_a_public_repo(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    hub.repos["alice/yegi-test"] = {}
    hub.private["alice/yegi-test"] = False
    assert push(hub, data) == 2
    assert "not private" in capsys.readouterr().err
    assert hub.commits == []


def test_repush_skips_unchanged_and_handles_removed_files(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    n = len(hub.commits)
    assert push(hub, data) == 0
    assert len(hub.commits) == n
    assert "remote already up to date" in capsys.readouterr().out

    (data["corpus"] / "a.txt").write_text("alpha v2")
    (data["corpus"] / "sub" / "b.pdf").unlink()
    assert push(hub, data) == 0
    uploaded = [p for _, added, _ in hub.commits[n:] for p in added]
    assert "unstructured/pharma-mini/a.txt" in uploaded
    assert "unstructured/pharma-mini/sub/b.pdf" in hub.repos["alice/yegi-test"]   # kept
    assert "no longer exist locally; kept" in capsys.readouterr().out

    assert push(hub, data, "--delete-missing") == 0
    assert "unstructured/pharma-mini/sub/b.pdf" not in hub.repos["alice/yegi-test"]
    m = json.loads(hub.repos["alice/yegi-test"]["yegi_manifest.json"])
    corpus = next(d for d in m["items"] if d["name"] == "pharma-mini")
    assert [f["path"] for f in corpus["files"]] == ["a.txt"]


def test_hand_edited_card_is_not_overwritten(tmp_path, data):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    hub.repos["alice/yegi-test"]["README.md"] = b"# my notes"
    (data["corpus"] / "a.txt").write_text("changed")
    assert push(hub, data) == 0
    assert hub.repos["alice/yegi-test"]["README.md"] == b"# my notes"


def test_pull_restores_to_original_paths_and_verifies(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    root = tmp_path / "checkout"
    root.mkdir()
    assert hf_dataset.main(["pull", "--repo", "alice/yegi-test", "--root", str(root)],
                           api_factory=hub) == 0
    assert (root / "source_documents/pharma-mini/a.txt").read_text() == "alpha"
    assert (root / "source_documents/pharma-mini/sub/b.pdf").read_bytes() == b"%PDF-1.4 b"
    assert (root / "source_documents/structured_sec/fsds/2024q1/native/sub.txt").exists()
    ts = json.loads((root / "eval_questions/testsets/pharma-mini.json").read_text())
    assert (root / ts["corpus"] / "a.txt").exists()          # test set still finds its corpus
    assert not (root / ".hf_staging").exists()
    assert not (root / "source_documents/pharma-mini/.env").exists()
    # second pull: everything already present
    assert hf_dataset.main(["pull", "--repo", "alice/yegi-test", "--root", str(root)],
                           api_factory=hub) == 0
    assert "everything already present" in capsys.readouterr().out


def test_pull_selects_by_kind_and_item(tmp_path, data):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    root = tmp_path / "c"
    root.mkdir()
    assert hf_dataset.main(["pull", "--repo", "alice/yegi-test", "--root", str(root),
                            "--kind", "eval"], api_factory=hub) == 0
    assert (root / "eval_questions/testsets/pharma-mini.json").exists()
    assert not (root / "source_documents").exists()
    assert hf_dataset.main(["pull", "--repo", "alice/yegi-test", "--root", str(root),
                            "--item", "nope"], api_factory=hub) == 2


def test_pull_never_clobbers_without_overwrite(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    root = tmp_path / "c"
    target = root / "source_documents/pharma-mini/a.txt"
    target.parent.mkdir(parents=True)
    target.write_text("my local edit")
    argv = ["pull", "--repo", "alice/yegi-test", "--root", str(root)]
    assert hf_dataset.main(argv, api_factory=hub) == 2
    assert "differ from the dataset" in capsys.readouterr().err
    assert target.read_text() == "my local edit" and hub.downloads == 0
    assert hf_dataset.main([*argv, "--overwrite"], api_factory=hub) == 0
    assert target.read_text() == "alpha"


def test_pull_rejects_path_traversal_in_remote_manifest(tmp_path, capsys):
    hub = FakeHub(tmp_path)
    hub.repos["bob/evil"] = {"yegi_manifest.json": json.dumps({
        "format": hf_dataset.FORMAT, "repo_id": "bob/evil", "items": [{
            "kind": "unstructured", "name": "x", "repo_prefix": "unstructured/x",
            "restore_to": "../../outside", "files": [{"path": "a", "bytes": 1, "sha256": "0"}]}]
    }).encode()}
    hub.private["bob/evil"] = True
    root = tmp_path / "c"
    root.mkdir()
    assert hf_dataset.main(["pull", "--repo", "bob/evil", "--root", str(root)],
                           api_factory=hub) == 2
    assert "unsafe path" in capsys.readouterr().err
    assert not (tmp_path / "outside").exists()


def test_pull_checksum_mismatch_installs_nothing(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    hub.repos["alice/yegi-test"]["unstructured/pharma-mini/a.txt"] = b"tampered"
    root = tmp_path / "c"
    root.mkdir()
    assert hf_dataset.main(["pull", "--repo", "alice/yegi-test", "--root", str(root)],
                           api_factory=hub) == 2
    assert "SHA-256 does not match" in capsys.readouterr().err
    assert not (root / "source_documents").exists() and not (root / ".hf_staging").exists()


def test_pull_dataset_not_written_by_this_tool(tmp_path):
    hub = FakeHub(tmp_path)
    hub.repos["bob/plain"] = {"data/train.csv": b"a,b\n1,2\n"}
    hub.private["bob/plain"] = False
    root = tmp_path / "c"
    root.mkdir()
    assert hf_dataset.main(["pull", "--repo", "bob/plain", "--root", str(root)],
                           api_factory=hub) == 0
    assert (root / "source_documents/hf_bob__plain/data/train.csv").exists()


def test_list_and_whoami(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert push(hub, data) == 0
    capsys.readouterr()
    assert hf_dataset.main(["list", "--repo", "yegi-test"], api_factory=hub) == 0
    out = capsys.readouterr().out
    assert "alice/yegi-test" in out and "sec/fsds/2024q1" in out
    assert hf_dataset.main(["whoami"], api_factory=hub) == 0
    out = capsys.readouterr()
    assert "user alice" in out.out and TOKEN not in out.out + out.err


def test_nothing_selected_and_duplicate_names(tmp_path, data, capsys):
    hub = FakeHub(tmp_path)
    assert hf_dataset.main(["push", "--repo", "x"], api_factory=hub) == 2
    other = tmp_path / "elsewhere" / "pharma-mini"
    other.mkdir(parents=True)
    (other / "a.txt").write_text("other")
    assert hf_dataset.main(["push", "--repo", "x", "--corpus", str(data["corpus"]),
                            "--corpus", str(other)], api_factory=hub) == 2
    assert "same repo path" in capsys.readouterr().err


@pytest.mark.parametrize("name,excluded", [
    ("notes.txt", False), (".env", True), (".env.local", True), ("aws_credentials.json", True),
    ("x.part", True), ("x.bak", True), ("server.pem", True), ("report.pdf", False)])
def test_exclusion_rules(name, excluded):
    from pathlib import PurePosixPath
    assert (hf_dataset.excluded_reason(PurePosixPath(name)) is not None) is excluded


def test_symlinks_inside_repo_are_followed_outside_are_not(tmp_path, monkeypatch, capsys):
    # Subset corpora are folders of links into a bigger corpus in the same repo.
    repo = tmp_path / "repo"
    big = repo / "source_documents" / "big"
    big.mkdir(parents=True)
    (big / "doc.txt").write_text("real content")
    secret = tmp_path / "home" / ".ssh_like"
    secret.mkdir(parents=True)
    (secret / "notes.txt").write_text("outside the repo")
    subset = repo / "source_documents" / "subset"
    subset.mkdir()
    try:
        (subset / "doc.txt").symlink_to(big / "doc.txt")
        (subset / "leak.txt").symlink_to(secret / "notes.txt")
        (subset / "gone.txt").symlink_to(big / "missing.txt")
    except OSError:
        pytest.skip("symlinks not permitted on this system")
    monkeypatch.setattr(hf_dataset, "REPO_ROOT", repo)
    hub = FakeHub(tmp_path)
    assert hf_dataset.main(["push", "--repo", "r", "--corpus", str(subset)],
                           api_factory=hub) == 0
    files = hub.repos["alice/r"]
    assert files["unstructured/subset/doc.txt"] == b"real content"
    assert not any("leak" in p or "gone" in p for p in files)
    out = capsys.readouterr().out
    assert "outside the repo (not followed)" in out and "broken symlink" in out
    m = json.loads(files["yegi_manifest.json"])
    assert m["items"][0]["restore_to"] == "source_documents/subset"


def test_structured_inside_repo_restores_to_its_own_path(tmp_path, data, monkeypatch):
    repo = tmp_path / "repo"
    snap = repo / "source_documents" / "bundle" / "sec" / "companyfacts" / "20260101T000000Z"
    snap.mkdir(parents=True)
    (snap / "facts.csv").write_text("cik,val\n0000320193,1\n")
    (snap / "manifest.json").write_text(json.dumps(
        {"source": "sec", "dataset": "companyfacts", "status": "complete"}))
    monkeypatch.setattr(hf_dataset, "REPO_ROOT", repo)
    hub = FakeHub(tmp_path)
    assert hf_dataset.main(["push", "--repo", "r", "--structured",
                            str(repo / "source_documents" / "bundle")], api_factory=hub) == 0
    m = json.loads(hub.repos["alice/r"]["yegi_manifest.json"])
    assert m["items"][0]["restore_to"] == (
        "source_documents/bundle/sec/companyfacts/20260101T000000Z")


def test_prefer_ipv4_orders_addresses_and_restores(monkeypatch):
    import socket
    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2600::1", 443, 0, 0))
    v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("3.1.1.1", 443))
    fake = lambda *a, **k: [v6, v4]  # noqa: E731
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with hf_dataset.prefer_ipv4():
        assert [r[0] for r in socket.getaddrinfo("huggingface.co", 443)] == [
            socket.AF_INET, socket.AF_INET6]                     # v6 kept as fallback
    assert socket.getaddrinfo is fake                            # restored afterwards
    monkeypatch.setenv("HF_PREFER_IPV6", "1")
    with hf_dataset.prefer_ipv4():
        assert socket.getaddrinfo("h", 443)[0][0] == socket.AF_INET6

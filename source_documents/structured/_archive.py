"""Safe extraction of provider archives (zip, tar[.gz], single-file .gz).

Every member path is checked before anything is written: absolute paths,
drive letters, ``..`` components, symlinks/hardlinks/devices are rejected,
and the total uncompressed size and member count are capped so a corrupt
or hostile archive cannot fill the disk. Members are streamed to disk, never
read whole into memory.
"""

from __future__ import annotations

import gzip
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_MAX_BYTES = 50 * 1024**3       # 50 GB uncompressed
DEFAULT_MAX_MEMBERS = 20_000


class UnsafeArchiveError(ValueError):
    """Archive contains a path or member type we refuse to extract."""


def _safe_member_path(dest: Path, name: str) -> Path:
    norm = name.replace("\\", "/")
    pure = PurePosixPath(norm)
    if norm.startswith("/") or pure.is_absolute() or (len(norm) > 1 and norm[1] == ":"):
        raise UnsafeArchiveError(f"absolute path in archive: {name!r}")
    if any(part == ".." for part in pure.parts):
        raise UnsafeArchiveError(f"path traversal in archive: {name!r}")
    target = (dest / Path(*pure.parts)).resolve()
    root = dest.resolve()
    if target != root and root not in target.parents:
        raise UnsafeArchiveError(f"archive member escapes destination: {name!r}")
    return target


def _check_totals(total: int, count: int, max_bytes: int, max_members: int) -> None:
    if count > max_members:
        raise UnsafeArchiveError(f"archive has {count} members (limit {max_members})")
    if total > max_bytes:
        raise UnsafeArchiveError(
            f"archive expands to {total:,} bytes (limit {max_bytes:,}); raise the "
            f"limit only if this size is expected")


def safe_extract(
    archive: Path,
    dest: Path,
    *,
    members: list[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> list[Path]:
    """Extract ``archive`` into ``dest``; return the written file paths.

    ``members`` optionally restricts extraction to those member names
    (case-insensitive basename or full-name match).
    """
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {m.lower() for m in members} if members else None

    def keep(name: str) -> bool:
        if wanted is None:
            return True
        return name.lower() in wanted or PurePosixPath(name).name.lower() in wanted

    if zipfile.is_zipfile(archive):
        return _extract_zip(archive, dest, keep, max_bytes, max_members)
    if tarfile.is_tarfile(archive):
        return _extract_tar(archive, dest, keep, max_bytes, max_members)
    with archive.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        out = dest / (archive.name[:-3] if archive.name.endswith(".gz") else archive.name + ".out")
        _safe_member_path(dest, out.name)
        written = 0
        part = out.with_name(out.name + ".part")
        with gzip.open(archive, "rb") as src, part.open("wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                written += len(block)
                _check_totals(written, 1, max_bytes, max_members)
                dst.write(block)
        os.replace(part, out)
        return [out]
    raise UnsafeArchiveError(f"{archive.name} is not a zip, tar or gzip archive")


def _extract_zip(archive, dest, keep, max_bytes, max_members) -> list[Path]:
    out: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        infos = [i for i in zf.infolist() if keep(i.filename)]
        _check_totals(sum(i.file_size for i in infos), len(infos), max_bytes, max_members)
        plan = []
        for info in infos:
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"symlink in archive: {info.filename!r}")
            target = _safe_member_path(dest, info.filename)
            plan.append((info, target))
        for info, target in plan:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with zf.open(info) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(part, target)
            out.append(target)
    return out


def _extract_tar(archive, dest, keep, max_bytes, max_members) -> list[Path]:
    out: list[Path] = []
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if keep(m.name)]
        _check_totals(sum(m.size for m in members if m.isfile()), len(members),
                      max_bytes, max_members)
        plan = []
        for m in members:
            if m.issym() or m.islnk():
                raise UnsafeArchiveError(f"link in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise UnsafeArchiveError(f"special file in archive: {m.name!r}")
            plan.append((m, _safe_member_path(dest, m.name)))
        for m, target in plan:
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            part = target.with_name(target.name + ".part")
            with src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(part, target)
            out.append(target)
    return out


def list_members(archive: Path) -> list[tuple[str, int]]:
    """(name, uncompressed size) for each file member, without extracting."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            return [(i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            return [(m.name, m.size) for m in tf.getmembers() if m.isfile()]
    return [(archive.name, archive.stat().st_size)]

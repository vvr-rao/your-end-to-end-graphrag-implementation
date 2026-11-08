"""HTTP client shared by every structured-data source.

Wraps the folder's existing ``polite_session()`` and adds what bulk / API
downloads need and the older document downloaders never did:

* connect/read timeouts on every call;
* bounded retries with exponential backoff that honour ``Retry-After``
  (seconds or HTTP-date) on 429/5xx and on connection errors;
* a per-source minimum interval between requests (rate limiting);
* streaming downloads into ``<name>.part`` followed by an atomic
  ``os.replace`` -- an interrupted transfer never leaves a final-looking file;
* byte-range resume, used only when the server advertised
  ``Accept-Ranges: bytes`` and answered ``206``;
* a content sniff that rejects HTML error / login pages returned where a
  zip, gzip, parquet or JSON payload was expected;
* SHA-256 computed while streaming.

Credentials are passed as ``auth_params`` / ``auth_headers`` and merged in at
request time only, so URLs printed in logs or written to manifests never
carry them (see ``_manifest.redact_url``).
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from _downloader_common import polite_session

from ._manifest import redact_url

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Leading bytes ("magic") that identify each expected payload kind.
_MAGIC: dict[str, tuple[bytes, ...]] = {
    "zip": (b"PK\x03\x04", b"PK\x05\x06"),
    "gzip": (b"\x1f\x8b",),
    "parquet": (b"PAR1",),
}


class DownloadError(RuntimeError):
    """A request failed in a way the user can act on (message says how)."""


class UnexpectedContentError(DownloadError):
    """The server returned something other than the expected payload
    (typically an HTML error or login page with a 200 status)."""


@dataclass
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    resumed: bool = False
    skipped: bool = False
    content_type: str = ""


def looks_like_html(head: bytes) -> bool:
    """True if the first bytes of a payload are an HTML document."""
    probe = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()[:512]
    return probe.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or (
        probe.startswith(b"<") and b"<html" in probe
    )


def check_payload(head: bytes, expect: str, url: str) -> None:
    """Raise UnexpectedContentError if ``head`` does not match ``expect``.

    ``expect`` is one of: zip, gzip, parquet, json, text, any. ``text`` and
    ``json`` only reject HTML; binary kinds also check magic bytes.
    """
    safe = redact_url(url)
    if expect == "any" or not head:
        return
    if expect != "html" and looks_like_html(head):
        raise UnexpectedContentError(
            f"expected {expect} from {safe} but received an HTML page "
            f"(often an error, maintenance or login page). Check the URL / "
            f"credentials, or download it manually and use --from-archive "
            f"where the source supports it."
        )
    magic = _MAGIC.get(expect)
    if magic and not head.startswith(magic):
        raise UnexpectedContentError(
            f"expected a {expect} file from {safe} but the first bytes were "
            f"{head[:8]!r}; refusing to save it as {expect}."
        )
    if expect == "json" and head.lstrip()[:1] not in (b"{", b"["):
        raise UnexpectedContentError(f"expected JSON from {safe}; got {head[:60]!r}")


def _parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - (now if now is not None else time.time()))


class RateLimiter:
    """Minimum interval between request starts (thread-safe)."""

    def __init__(self, per_second: float | None, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / per_second if per_second else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = self._clock()
            delay = self._next - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next = now + self.interval


class HttpClient:
    """Retrying, rate-limited HTTP client for one source."""

    def __init__(
        self,
        *,
        contact: str,
        rate_per_second: float | None = None,
        timeout: tuple[float, float] = (15.0, 120.0),
        max_retries: int = 4,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        max_retry_after: float = 300.0,
        user_agent: str | None = None,
        auth_params: Mapping[str, str] | None = None,
        auth_headers: Mapping[str, str] | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.session = session if session is not None else polite_session(contact=contact)
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept"] = "*/*"
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.max_retry_after = max_retry_after
        self.auth_params = dict(auth_params or {})
        self.auth_headers = dict(auth_headers or {})
        self.limiter = RateLimiter(rate_per_second, sleep=sleep)
        self._sleep = sleep
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self.requests_made = 0

    # ------------------------------------------------------------------ core

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        stream: bool = False,
        ok_statuses: tuple[int, ...] = (200,),
    ) -> requests.Response:
        """Send one request with retries. Returns a response whose status is
        in ``ok_statuses``; raises DownloadError otherwise."""
        merged_params = {**(params or {}), **self.auth_params} or None
        merged_headers = {**self.auth_headers, **(headers or {})}
        safe = redact_url(url, params)
        attempt = 0
        while True:
            self.limiter.wait()
            self.requests_made += 1
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=merged_params,
                    headers=merged_headers,
                    json=json_body,
                    stream=stream,
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise DownloadError(
                        f"{method} {safe} failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}. Check network access, or raise "
                        f"--timeout / --max-retries."
                    ) from None
                wait = self._backoff(attempt)
                self.log(f"  retry {attempt + 1}/{self.max_retries} in {wait:.1f}s "
                         f"({type(exc).__name__}): {safe}")
                self._sleep(wait)
                attempt += 1
                continue

            if resp.status_code in ok_statuses:
                return resp
            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                if retry_after is not None and retry_after > self.max_retry_after:
                    resp.close()
                    raise DownloadError(
                        f"{safe} asked us to wait {retry_after:.0f}s (Retry-After) -- "
                        f"longer than the {self.max_retry_after:.0f}s limit. The "
                        f"provider's rate limit is exhausted; try again later."
                    )
                wait = retry_after if retry_after is not None else self._backoff(attempt)
                self.log(f"  HTTP {resp.status_code}; retry {attempt + 1}/"
                         f"{self.max_retries} in {wait:.1f}s: {safe}")
                resp.close()
                self._sleep(wait)
                attempt += 1
                continue
            raise DownloadError(self._status_message(resp, safe))

    def _backoff(self, attempt: int) -> float:
        base = min(self.max_backoff, self.backoff_base * (2 ** attempt))
        return base * (0.75 + random.random() * 0.5)  # jitter, bounded

    @staticmethod
    def _status_message(resp: requests.Response, safe_url: str) -> str:
        code = resp.status_code
        hint = {
            400: "the provider rejected the parameters -- check identifiers / filters",
            401: "authentication failed -- check the credential environment variables",
            403: "access denied -- check credentials / User-Agent, or the provider "
                 "may require manual download",
            404: "not found -- check the identifier, period or release",
            429: "rate limited and retries exhausted -- wait and retry",
        }.get(code, "provider error")
        try:
            snippet = resp.text[:200].replace("\n", " ")
        except Exception:
            snippet = ""
        resp.close()
        return f"HTTP {code} from {safe_url}: {hint}. Body starts: {snippet!r}"

    # ----------------------------------------------------------- convenience

    def get_json(self, url: str, *, params=None, headers=None) -> Any:
        resp = self.request("GET", url, params=params,
                            headers={"Accept": "application/json", **(headers or {})})
        check_payload(resp.content[:512], "json", url)
        return resp.json()

    def post_json(self, url: str, body: Any, *, headers=None) -> Any:
        resp = self.request("POST", url, json_body=body,
                            headers={"Accept": "application/json", **(headers or {})})
        check_payload(resp.content[:512], "json", url)
        return resp.json()

    def head(self, url: str, *, params=None) -> requests.Response | None:
        """Best-effort HEAD (for sizes in --dry-run / range support). Never raises."""
        try:
            return self.request("HEAD", url, params=params, ok_statuses=(200, 204))
        except DownloadError:
            return None

    def download(
        self,
        url: str,
        dest: Path,
        *,
        params: Mapping[str, Any] | None = None,
        expect: str = "any",
        resume: bool = False,
        chunk_size: int = 1 << 20,
        progress_every: int = 64 << 20,
    ) -> DownloadResult:
        """Stream ``url`` into ``dest`` atomically and return its checksum.

        Writes ``dest.part`` and renames it on success. With ``resume`` and an
        existing ``.part``, a Range request is sent only if a HEAD shows
        ``Accept-Ranges: bytes``; a non-206 answer restarts from zero.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if (resume and part.exists()) else 0
        headers: dict[str, str] = {}
        if offset:
            head = self.head(url, params=params)
            if head is not None and head.headers.get("Accept-Ranges", "").lower() == "bytes":
                headers["Range"] = f"bytes={offset}-"
            else:
                self.log(f"  server does not advertise byte ranges; restarting {dest.name}")
                offset = 0
        resp = self.request("GET", url, params=params, headers=headers, stream=True,
                            ok_statuses=(200, 206))
        resumed = bool(offset) and resp.status_code == 206
        if offset and not resumed:
            offset = 0
        digest = hashlib.sha256()
        if resumed:
            with part.open("rb") as fh:
                for block in iter(lambda: fh.read(chunk_size), b""):
                    digest.update(block)
        total_hint = resp.headers.get("Content-Length")
        expected_len = int(total_hint) + offset if total_hint and total_hint.isdigit() else None
        written = offset
        next_report = written + progress_every
        first = not resumed
        try:
            with part.open("ab" if resumed else "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    if first:
                        check_payload(chunk[:512], expect, url)
                        first = False
                    fh.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        pct = f" ({100 * written / expected_len:.0f}%)" if expected_len else ""
                        self.log(f"    {dest.name}: {written / 1e6:,.0f} MB{pct}")
                        next_report += progress_every
        except UnexpectedContentError:
            resp.close()
            part.unlink(missing_ok=True)
            raise
        except (requests.RequestException, OSError) as exc:
            resp.close()
            raise DownloadError(
                f"transfer of {redact_url(url, params)} interrupted after "
                f"{written:,} bytes ({type(exc).__name__}); partial data kept in "
                f"{part.name} -- rerun with --resume to continue."
            ) from None
        finally:
            resp.close()
        if expected_len is not None and written != expected_len:
            raise DownloadError(
                f"{dest.name}: received {written:,} bytes but the server announced "
                f"{expected_len:,}; partial data kept in {part.name} -- rerun with --resume."
            )
        os.replace(part, dest)
        return DownloadResult(path=dest, bytes=written, sha256=digest.hexdigest(),
                              resumed=resumed,
                              content_type=resp.headers.get("Content-Type", ""))


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically (``.part`` then rename), pretty-printed UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(part, path)

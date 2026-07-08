"""The benign-asyncio-teardown filter must drop ONLY the 'Event loop is closed'
cleanup noise (httpx/anyio GC after a stage's asyncio.run loop closed), while
letting real asyncio errors through. Exercised via the real asyncio exception
handler -> 'asyncio' logger path that emits the message."""

from __future__ import annotations

import asyncio
import logging

from backend.app.cli.main import (
    _BenignAsyncTeardownFilter,
    _silence_benign_asyncio_teardown,
)


def _capture_asyncio(context: dict) -> list[logging.LogRecord]:
    """Route `context` through asyncio's default exception handler (which logs to
    the 'asyncio' logger) and return the records that survive the filter."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    lg = logging.getLogger("asyncio")
    handler = _Capture()
    lg.addHandler(handler)
    _silence_benign_asyncio_teardown()
    loop = asyncio.new_event_loop()
    try:
        loop.call_exception_handler(context)
    finally:
        loop.close()
        lg.removeHandler(handler)
    return records


def test_benign_event_loop_closed_is_suppressed() -> None:
    records = _capture_asyncio({
        "message": "Task exception was never retrieved",
        "exception": RuntimeError("Event loop is closed"),
    })
    assert records == []  # filtered out, no noise


def test_real_asyncio_error_still_logged() -> None:
    records = _capture_asyncio({
        "message": "Unhandled exception in event loop",
        "exception": ValueError("something actually broke"),
    })
    assert len(records) == 1  # real errors are NOT suppressed


def test_filter_unit() -> None:
    f = _BenignAsyncTeardownFilter()
    benign = logging.LogRecord("asyncio", logging.ERROR, "x", 0,
                               "Task exception was never retrieved", (),
                               (RuntimeError, RuntimeError("Event loop is closed"), None))
    real = logging.LogRecord("asyncio", logging.ERROR, "x", 0,
                             "boom", (),
                             (ValueError, ValueError("boom"), None))
    assert f.filter(benign) is False
    assert f.filter(real) is True

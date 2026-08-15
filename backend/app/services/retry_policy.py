"""Which API errors are worth retrying.

Retrying a deterministic 400 cannot succeed -- the same request produces the
same rejection -- but it does multiply the wall-clock cost of discovering that.
A `context_length_exceeded` observed in production burned all five attempts
before surfacing, and a client-side timeout nested inside SDK-level retries
turned one logical call into ~45 minutes.

Both the LLM router and the embeddings client classify errors here so the two
paths cannot drift apart.
"""
from __future__ import annotations

import httpx
import openai

# Status codes worth another attempt: transient server-side or rate-limit
# conditions. Mirrors `defaults.retries.retry_on_status` in config/models.yaml.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Deterministic client errors. The request is malformed, too large, or
# unauthorized -- an identical retry produces an identical failure.
_NON_RETRYABLE_EXC = (
    openai.BadRequestError,          # 400, incl. context_length_exceeded
    openai.AuthenticationError,      # 401
    openai.PermissionDeniedError,    # 403
    openai.NotFoundError,            # 404
    openai.UnprocessableEntityError, # 422
)


def is_retryable(exc: BaseException) -> bool:
    """True if retrying `exc` could plausibly succeed.

    Unknown exception types are treated as retryable: the pre-existing
    behaviour retried everything, and narrowing that to an allowlist risks
    turning a transient provider-specific error into a hard failure. The point
    here is to exclude the errors we can prove are permanent, not to enumerate
    every error that is transient.
    """
    if isinstance(exc, _NON_RETRYABLE_EXC):
        return False

    # Anthropic and Groq raise their own SDK types. Rather than importing and
    # matching each one, read the status code off any exception that carries
    # one -- all three SDKs expose `.status_code` on HTTP errors.
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in RETRYABLE_STATUS

    # Connection resets, DNS failures, read timeouts.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    return True


def describe(exc: BaseException) -> str:
    """Compact one-line description for logs: type, status, and message."""
    status = getattr(exc, "status_code", None)
    prefix = f"{type(exc).__name__}"
    if status is not None:
        prefix += f"[{status}]"
    return f"{prefix}: {exc}"

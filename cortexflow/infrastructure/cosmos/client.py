"""Cosmos DB client construction and error translation.

The SDK is imported lazily so that a unit test, or a local run on the memory
profile, never needs the Azure packages installed.
"""

from __future__ import annotations

from typing import Any

from cortexflow.config.settings import CosmosSettings
from cortexflow.shared.errors import (
    ConcurrencyConflictError,
    ConflictError,
    DependencyTimeoutError,
    DependencyUnavailableError,
    NotFoundError,
    RateLimitedError,
)

_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_HTTP_PRECONDITION_FAILED = 412
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_TIMEOUT = 408


def build_client(settings: CosmosSettings) -> Any:
    """Create an async Cosmos client, preferring managed identity over keys."""
    from azure.cosmos.aio import CosmosClient

    if settings.use_managed_identity and not settings.key:
        from azure.identity.aio import DefaultAzureCredential

        return CosmosClient(settings.endpoint, credential=DefaultAzureCredential())
    return CosmosClient(settings.endpoint, credential=settings.key)


def translate_error(exc: Exception, **context: Any) -> Exception:
    """Map Cosmos failures onto the platform's error taxonomy.

    412 is the important one: it is how optimistic concurrency surfaces, and it
    must become a *retryable* conflict rather than a generic 5xx.
    """
    status = getattr(exc, "status_code", None)
    match status:
        case c if c == _HTTP_PRECONDITION_FAILED:
            return ConcurrencyConflictError("Document was modified concurrently", **context)
        case c if c == _HTTP_NOT_FOUND:
            return NotFoundError("Document not found", **context)
        case c if c == _HTTP_CONFLICT:
            return ConflictError("Document already exists", **context)
        case c if c == _HTTP_TOO_MANY_REQUESTS:
            return RateLimitedError("Cosmos DB throttled the request", **context)
        case c if c == _HTTP_TIMEOUT:
            return DependencyTimeoutError("Cosmos DB request timed out", **context)
    if status is not None and 500 <= status < 600:
        return DependencyUnavailableError("Cosmos DB unavailable", status=status, **context)
    return exc

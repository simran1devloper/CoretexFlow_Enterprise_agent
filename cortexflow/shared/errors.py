"""Error taxonomy.

The retry machinery never inspects exception *types* from third-party SDKs.
Adapters translate their failures into this taxonomy, and the orchestrator
makes retry decisions purely from :class:`ErrorClass`.  "Classify failures"
is the whole point: a timeout is worth retrying, an invalid employee id is not.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorClass(StrEnum):
    TRANSIENT = "TRANSIENT"
    """Retry with backoff: timeouts, throttling, transient dependency outage."""

    PERMANENT = "PERMANENT"
    """Never retry: validation failures, authorization denials, bad input."""

    UNKNOWN = "UNKNOWN"
    """Retry cautiously (fewer attempts), then escalate to a human."""


class CortexFlowError(Exception):
    """Base class for every error the platform raises deliberately."""

    error_class: ErrorClass = ErrorClass.UNKNOWN
    code: str = "cortexflow_error"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    @property
    def retryable(self) -> bool:
        return self.error_class is not ErrorClass.PERMANENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "error_class": str(self.error_class),
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, details={self.details!r})"


# --------------------------------------------------------------------------
# Permanent
# --------------------------------------------------------------------------
class PermanentError(CortexFlowError):
    error_class = ErrorClass.PERMANENT
    code = "permanent_error"
    http_status = 400


class ValidationError(PermanentError):
    code = "validation_error"
    http_status = 422


class NotFoundError(PermanentError):
    code = "not_found"
    http_status = 404


class ConflictError(PermanentError):
    code = "conflict"
    http_status = 409


class AuthenticationError(PermanentError):
    code = "unauthenticated"
    http_status = 401


class AuthorizationError(PermanentError):
    code = "forbidden"
    http_status = 403


class PolicyViolationError(PermanentError):
    """A proposed action was refused by the deterministic policy engine."""

    code = "policy_violation"
    http_status = 403


class InvalidStateTransitionError(PermanentError):
    code = "invalid_state_transition"
    http_status = 409


class WorkflowDefinitionError(PermanentError):
    """A workflow definition is structurally invalid (cycle, unknown dep, ...)."""

    code = "invalid_workflow_definition"
    http_status = 422


class StructuredOutputError(PermanentError):
    """An LLM response could not be coerced into the declared schema."""

    code = "invalid_structured_output"
    http_status = 422


# --------------------------------------------------------------------------
# Transient
# --------------------------------------------------------------------------
class TransientError(CortexFlowError):
    error_class = ErrorClass.TRANSIENT
    code = "transient_error"
    http_status = 503


class DependencyTimeoutError(TransientError):
    code = "dependency_timeout"
    http_status = 504


class DependencyUnavailableError(TransientError):
    code = "dependency_unavailable"


class RateLimitedError(TransientError):
    code = "rate_limited"
    http_status = 429


class ConcurrencyConflictError(TransientError):
    """Optimistic concurrency check failed; reload and retry the mutation."""

    code = "concurrency_conflict"
    http_status = 409


class LeaseNotAcquiredError(TransientError):
    """Another worker currently holds the lease for this resource."""

    code = "lease_not_acquired"


# --------------------------------------------------------------------------
# Unknown
# --------------------------------------------------------------------------
class UnknownError(CortexFlowError):
    error_class = ErrorClass.UNKNOWN
    code = "unknown_error"


class ToolExecutionError(UnknownError):
    """An enterprise tool returned something we could not interpret."""

    code = "tool_execution_error"


def classify(exc: BaseException) -> ErrorClass:
    """Classify any exception, including ones raised outside our taxonomy."""
    if isinstance(exc, CortexFlowError):
        return exc.error_class
    if isinstance(exc, TimeoutError):
        return ErrorClass.TRANSIENT
    if isinstance(exc, ConnectionError):
        return ErrorClass.TRANSIENT
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ErrorClass.PERMANENT
    return ErrorClass.UNKNOWN

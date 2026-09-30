"""Retry policy.

Two decisions live here and nowhere else: *whether* to retry (driven by the
error class, never by the exception type) and *when* (exponential backoff with
jitter, so a recovering dependency is not hit by a synchronised retry storm).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cortexflow.config.settings import ReliabilitySettings
from cortexflow.modules.workflow.domain.definition import RetryPolicySpec
from cortexflow.shared.errors import ErrorClass

UNKNOWN_ERROR_ATTEMPT_CAP = 2
"""Unknown failures get fewer attempts than transient ones, then escalate."""


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: float = 0.0
    reason: str = ""


class RetryPolicy:
    """Resolves per-step retry settings against platform defaults."""

    def __init__(self, settings: ReliabilitySettings) -> None:
        self._settings = settings

    def max_attempts(self, spec: RetryPolicySpec) -> int:
        return spec.max_attempts or self._settings.max_attempts

    def decide(
        self,
        *,
        spec: RetryPolicySpec,
        error_class: ErrorClass,
        attempt: int,
        idempotent: bool = True,
    ) -> RetryDecision:
        """Decide whether attempt ``attempt`` should be followed by another."""
        if error_class is ErrorClass.PERMANENT:
            return RetryDecision(False, reason="Permanent failure; retrying cannot help")

        if not idempotent:
            # A non-idempotent step that may have partially applied is never
            # retried automatically -- a human decides.
            return RetryDecision(
                False, reason="Step is not idempotent; escalating instead of retrying"
            )

        limit = self.max_attempts(spec)
        if error_class is ErrorClass.UNKNOWN:
            if not spec.retry_on_unknown:
                return RetryDecision(False, reason="Unknown failures are not retried here")
            limit = min(limit, UNKNOWN_ERROR_ATTEMPT_CAP)

        if attempt >= limit:
            return RetryDecision(
                False, reason=f"Retry budget exhausted after {attempt} attempts"
            )

        return RetryDecision(
            True,
            delay_seconds=self.backoff(spec, attempt),
            reason=f"Retrying after a {error_class} failure",
        )

    def backoff(self, spec: RetryPolicySpec, attempt: int, *, seed: str = "") -> float:
        """Exponential backoff with deterministic jitter.

        Jitter is derived from a hash rather than ``random`` so that a retry
        schedule is reproducible in tests while still spreading concurrent
        retries of *different* steps apart.
        """
        base = spec.backoff_base_seconds or self._settings.backoff_base_seconds
        ceiling = spec.backoff_max_seconds or self._settings.backoff_max_seconds
        delay = min(base * (2 ** max(attempt - 1, 0)), ceiling)
        spread = self._settings.backoff_jitter
        if spread <= 0:
            return delay
        offset = _jitter_ratio(seed or f"{attempt}") * 2 - 1  # -1.0 .. 1.0
        return max(0.1, delay * (1 + spread * offset))


def _jitter_ratio(seed: str) -> float:
    return hashlib.sha256(seed.encode()).digest()[0] / 255

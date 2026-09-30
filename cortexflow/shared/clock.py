"""Time access.

Nothing in the domain calls ``datetime.now()`` directly: leases, timeouts and
backoff windows are all testable because time is a port.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Manually advanced clock for deterministic tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now


def utcnow() -> datetime:
    """Convenience for adapters and API layers that are not domain code."""
    return datetime.now(UTC)

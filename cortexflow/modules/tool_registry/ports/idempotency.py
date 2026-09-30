"""The tool registry's idempotency port.

Every method is tenant-scoped. Isolation is not a convention the callers must
remember -- it is in the signature.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IdempotencyStore(Protocol):
    """Exactly-once semantics for side-effecting operations.

    ``reserve`` is the atomic primitive: the first caller wins the right to
    execute, every later caller waits for or replays the recorded result.
    """

    async def reserve(self, key: str, *, ttl_seconds: int) -> bool:
        """True if this caller claimed the key; False if it was already claimed."""
        ...

    async def complete(self, key: str, result: dict[str, Any], *, ttl_seconds: int) -> None: ...

    async def get(self, key: str) -> dict[str, Any] | None:
        """The recorded result, or None if unclaimed or still in flight."""
        ...

    async def release(self, key: str) -> None:
        """Drop a reservation whose operation failed, so a retry may proceed."""
        ...

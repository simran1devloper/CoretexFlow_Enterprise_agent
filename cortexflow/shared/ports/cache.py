"""Cache and coordination ports.

Redis is an *accelerator*, never the source of truth.  The two interfaces are
separate because their failure modes differ: losing the cache costs latency,
while losing the lock manager costs safety -- so lock holders always re-verify
durable state under optimistic concurrency before committing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        """Atomic counter, used for rate limiting."""
        ...

    async def add(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> bool:
        """Set only if absent. The deduplication primitive. True when it was set."""
        ...


@runtime_checkable
class Lease(Protocol):
    """A time-bounded claim on a resource."""

    @property
    def resource(self) -> str: ...

    @property
    def owner(self) -> str: ...

    async def renew(self, *, ttl_seconds: int) -> bool: ...

    async def release(self) -> None: ...


@runtime_checkable
class LockManager(Protocol):
    """Distributed leases.

    A lease can expire, which is the point: a worker that crashes mid-step
    stops blocking the workflow once its lease lapses, and another worker
    reclaims it.
    """

    async def acquire(
        self, resource: str, *, owner: str, ttl_seconds: int
    ) -> Lease | None:
        """Claim ``resource``, or None if someone else currently holds it."""
        ...

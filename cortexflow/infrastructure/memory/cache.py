"""In-memory cache and lock manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from cortexflow.shared.ids import UuidIdGenerator


@dataclass
class _Entry:
    value: Any
    expires_at: float | None


def _now() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:  # pragma: no cover - only outside an event loop
        import time

        return time.monotonic()


class InMemoryCache:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    def _live(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= _now():
            del self._entries[key]
            return None
        return entry

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._live(key)
            return entry.value if entry else None

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        async with self._lock:
            self._entries[key] = _Entry(
                value=value,
                expires_at=_now() + ttl_seconds if ttl_seconds else None,
            )

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        async with self._lock:
            entry = self._live(key)
            value = int(entry.value) + 1 if entry else 1
            expires_at = (
                entry.expires_at
                if entry
                else (_now() + ttl_seconds if ttl_seconds else None)
            )
            self._entries[key] = _Entry(value=value, expires_at=expires_at)
            return value

    async def add(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> bool:
        async with self._lock:
            if self._live(key) is not None:
                return False
            self._entries[key] = _Entry(
                value=value,
                expires_at=_now() + ttl_seconds if ttl_seconds else None,
            )
            return True


class InMemoryLease:
    def __init__(self, manager: InMemoryLockManager, resource: str, owner: str) -> None:
        self._manager = manager
        self._resource = resource
        self._owner = owner

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def owner(self) -> str:
        return self._owner

    async def renew(self, *, ttl_seconds: int) -> bool:
        return await self._manager._renew(self._resource, self._owner, ttl_seconds)

    async def release(self) -> None:
        await self._manager._release(self._resource, self._owner)


class InMemoryLockManager:
    """Leases with expiry, so a crashed holder never blocks forever."""

    def __init__(self) -> None:
        self._holders: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()
        self._ids = UuidIdGenerator()

    async def acquire(
        self, resource: str, *, owner: str, ttl_seconds: int
    ) -> InMemoryLease | None:
        async with self._lock:
            held = self._holders.get(resource)
            if held is not None and held[1] > _now() and held[0] != owner:
                return None
            self._holders[resource] = (owner, _now() + ttl_seconds)
            return InMemoryLease(self, resource, owner)

    async def _renew(self, resource: str, owner: str, ttl_seconds: int) -> bool:
        async with self._lock:
            held = self._holders.get(resource)
            if held is None or held[0] != owner:
                return False
            self._holders[resource] = (owner, _now() + ttl_seconds)
            return True

    async def _release(self, resource: str, owner: str) -> None:
        async with self._lock:
            held = self._holders.get(resource)
            if held is not None and held[0] == owner:
                del self._holders[resource]

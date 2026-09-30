"""Redis cache and lease manager.

Redis is the acceleration layer, never the system of record.  Every read here
is optional: :class:`RedisCache` failures degrade to a miss so the caller falls
through to Cosmos DB, and the platform keeps running with higher latency
instead of stopping.
"""

from __future__ import annotations

import json
from typing import Any

from cortexflow.config.settings import RedisSettings
from cortexflow.shared.ids import UuidIdGenerator
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""

_RENEW_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
end
return 0
"""


def build_redis(settings: RedisSettings) -> Any:
    from redis.asyncio import Redis

    return Redis.from_url(settings.url, decode_responses=True)


class RedisCache:
    """Best-effort cache: an unavailable Redis is a cache miss, not an error."""

    def __init__(self, client: Any, settings: RedisSettings) -> None:
        self._client = client
        self._prefix = settings.key_prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(self._key(key))
        except Exception as exc:
            logger.warning("cache read failed; falling through", extra={"error": str(exc)})
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        try:
            await self._client.set(self._key(key), json.dumps(value, default=str), ex=ttl_seconds)
        except Exception as exc:
            logger.warning("cache write failed; ignoring", extra={"error": str(exc)})

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(self._key(key))
        except Exception as exc:
            logger.warning("cache delete failed; ignoring", extra={"error": str(exc)})

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        """Atomic counter. Unlike reads, a failure here must surface: rate
        limiting that silently returns 0 is a rate limiter that does nothing."""
        full = self._key(key)
        pipe = self._client.pipeline()
        pipe.incr(full)
        if ttl_seconds:
            pipe.expire(full, ttl_seconds, nx=True)
        result = await pipe.execute()
        return int(result[0])

    async def add(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> bool:
        return bool(
            await self._client.set(
                self._key(key), json.dumps(value, default=str), ex=ttl_seconds, nx=True
            )
        )


class RedisLease:
    def __init__(self, client: Any, resource: str, owner: str, token: str) -> None:
        self._client = client
        self._resource = resource
        self._owner = owner
        self._token = token

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def owner(self) -> str:
        return self._owner

    async def renew(self, *, ttl_seconds: int) -> bool:
        result = await self._client.eval(
            _RENEW_SCRIPT, 1, self._resource, self._token, ttl_seconds * 1000
        )
        return bool(result)

    async def release(self) -> None:
        """Release only if we still hold it -- never free someone else's lease."""
        await self._client.eval(_RELEASE_SCRIPT, 1, self._resource, self._token)


class RedisLockManager:
    def __init__(self, client: Any, settings: RedisSettings) -> None:
        self._client = client
        self._prefix = f"{settings.key_prefix}:lock"
        self._ids = UuidIdGenerator()

    async def acquire(self, resource: str, *, owner: str, ttl_seconds: int) -> RedisLease | None:
        key = f"{self._prefix}:{resource}"
        token = f"{owner}:{self._ids.new_id('LSE')}"
        acquired = await self._client.set(key, token, nx=True, ex=ttl_seconds)
        if not acquired:
            return None
        return RedisLease(self._client, key, owner, token)

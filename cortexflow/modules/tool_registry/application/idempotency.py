"""Idempotent execution of side-effecting tools.

The rule the platform enforces: *every write to an enterprise system carries a
key derived from the workflow step that caused it*.  A retry, a duplicate
message or an operator replay then converges on one effect rather than paying
an invoice twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from cortexflow.modules.tool_registry.domain.models import ToolResult
from cortexflow.modules.tool_registry.ports.idempotency import IdempotencyStore
from cortexflow.shared.errors import ToolExecutionError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 0.1
_MAX_WAIT_POLLS = 50


def build_key(
    *, workflow_id: str, step_id: str, tool: str, arguments: dict[str, Any]
) -> str:
    """Derive a stable key from the *cause* of the call, not the attempt.

    Arguments are folded in so that a step legitimately calling the same tool
    with different inputs gets distinct keys, while a retry of an identical
    call reuses one.
    """
    canonical = json.dumps(arguments, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{workflow_id}:{step_id}:{tool}:{digest}"


class IdempotentExecutor:
    """Wraps a tool call in reserve / execute / record."""

    def __init__(self, store: IdempotencyStore, *, ttl_seconds: int = 604_800) -> None:
        self._store = store
        self._ttl = ttl_seconds

    async def run(
        self,
        key: str,
        operation: Any,
        *,
        tool: str,
    ) -> ToolResult:
        """Execute ``operation`` at most once for ``key``."""
        recorded = await self._store.get(key)
        if recorded is not None:
            logger.info("tool call replayed from idempotency store", extra={"tool": tool})
            return ToolResult.model_validate({**recorded, "replayed": True})

        if not await self._store.reserve(key, ttl_seconds=self._ttl):
            # Someone else is mid-flight with the same key. Wait for their
            # result rather than racing them into a double side effect.
            return await self._await_result(key, tool=tool)

        try:
            result: ToolResult = await operation()
        except Exception:
            # Release so a retry can proceed; the effect did not land.
            await self._store.release(key)
            raise

        if result.succeeded:
            await self._store.complete(
                key, result.model_dump(mode="json"), ttl_seconds=self._ttl
            )
        else:
            await self._store.release(key)
        return result

    async def _await_result(self, key: str, *, tool: str) -> ToolResult:
        for _ in range(_MAX_WAIT_POLLS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            recorded = await self._store.get(key)
            if recorded is not None:
                return ToolResult.model_validate({**recorded, "replayed": True})
        raise ToolExecutionError(
            "Concurrent execution of the same idempotency key did not complete",
            tool=tool,
            idempotency_key=key,
        )

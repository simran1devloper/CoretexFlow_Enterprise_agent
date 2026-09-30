"""Messaging ports.

The bus contract is deliberately narrower than Service Bus: publish, schedule,
consume with explicit settlement, dead-letter.  Anything richer would leak a
vendor's semantics into the orchestrator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import Message


@runtime_checkable
class ReceivedMessage(Protocol):
    """A message checked out from a queue, awaiting settlement."""

    @property
    def message(self) -> Message: ...

    @property
    def delivery_count(self) -> int:
        """How many times the broker has delivered this message."""
        ...

    @property
    def queue(self) -> Queue: ...

    async def complete(self) -> None:
        """Acknowledge; the broker removes the message."""
        ...

    async def abandon(self, reason: str = "") -> None:
        """Release for redelivery, e.g. after a transient failure."""
        ...

    async def dead_letter(self, reason: str, description: str = "") -> None:
        """Move to the dead-letter queue after retries are exhausted."""
        ...

    async def renew_lock(self) -> None:
        """Extend the visibility timeout for genuinely long work."""
        ...


@runtime_checkable
class MessageBus(Protocol):
    async def publish(self, queue: Queue, message: Message) -> None: ...

    async def publish_many(self, queue: Queue, messages: list[Message]) -> None: ...

    async def schedule(
        self, queue: Queue, message: Message, *, delay_seconds: float
    ) -> None:
        """Deliver later -- how retry backoff is expressed without sleeping."""
        ...

    def consume(
        self, queue: Queue, *, max_messages: int = 1
    ) -> AsyncIterator[ReceivedMessage]:
        """Long-poll a queue, yielding unsettled messages."""
        ...

    async def dead_letters(self, queue: Queue, *, limit: int = 50) -> list[Message]: ...

    async def replay_dead_letter(self, queue: Queue, message: Message) -> None: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

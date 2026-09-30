"""In-process message bus with broker semantics.

It models the parts of Azure Service Bus the platform actually relies on:
at-least-once delivery, lock/visibility timeouts, delivery counts, scheduled
messages and dead-lettering.  Because a crashed consumer's message reappears
after its lock lapses, the crash-recovery tests are real tests rather than
simulations of one.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import Message
from cortexflow.shared.clock import Clock, SystemClock
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

_counter = itertools.count()


@dataclass(order=True)
class _Scheduled:
    due_at: float
    seq: int
    queue: Queue = field(compare=False)
    message: Message = field(compare=False)


@dataclass
class _Entry:
    message: Message
    delivery_count: int = 0
    locked_until: float = 0.0


class InMemoryReceivedMessage:
    """An unsettled message checked out from an in-memory queue."""

    def __init__(self, bus: InMemoryMessageBus, queue: Queue, entry: _Entry) -> None:
        self._bus = bus
        self._queue = queue
        self._entry = entry
        self._settled = False

    @property
    def message(self) -> Message:
        return self._entry.message

    @property
    def delivery_count(self) -> int:
        return self._entry.delivery_count

    @property
    def queue(self) -> Queue:
        return self._queue

    async def complete(self) -> None:
        self._settle()
        await self._bus._remove_inflight(self._queue, self._entry)

    async def abandon(self, reason: str = "") -> None:
        self._settle()
        await self._bus._requeue(self._queue, self._entry)

    async def dead_letter(self, reason: str, description: str = "") -> None:
        self._settle()
        await self._bus._dead_letter(self._queue, self._entry, reason, description)

    async def renew_lock(self) -> None:
        self._entry.locked_until = self._bus._monotonic() + self._bus.lock_seconds

    def _settle(self) -> None:
        if self._settled:
            raise RuntimeError("Message already settled")
        self._settled = True


class InMemoryMessageBus:
    """A broker good enough to test reliability against."""

    def __init__(
        self,
        *,
        lock_seconds: float = 30.0,
        max_delivery_count: int = 5,
        clock: Clock | None = None,
    ) -> None:
        self.lock_seconds = lock_seconds
        self.max_delivery_count = max_delivery_count
        self._clock = clock or SystemClock()
        self._queues: dict[Queue, list[_Entry]] = {q: [] for q in Queue}
        self._inflight: dict[Queue, list[_Entry]] = {q: [] for q in Queue}
        self._dlq: dict[Queue, list[tuple[Message, str, str]]] = {q: [] for q in Queue}
        self._scheduled: list[_Scheduled] = []
        self._lock = asyncio.Lock()
        self._arrival = asyncio.Event()
        self._scheduler_task: asyncio.Task[None] | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._scheduler_task is None:
            self._closed = False
            self._scheduler_task = asyncio.create_task(self._run_scheduler())

    async def close(self) -> None:
        self._closed = True
        self._arrival.set()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

    def _monotonic(self) -> float:
        return asyncio.get_running_loop().time()

    # -- producer ----------------------------------------------------------
    async def publish(self, queue: Queue, message: Message) -> None:
        async with self._lock:
            self._queues[queue].append(_Entry(message=message))
        self._arrival.set()
        logger.debug(
            "message published",
            extra={"queue": queue.value, "message_type": str(message.type)},
        )

    async def publish_many(self, queue: Queue, messages: list[Message]) -> None:
        async with self._lock:
            self._queues[queue].extend(_Entry(message=m) for m in messages)
        self._arrival.set()

    async def schedule(
        self, queue: Queue, message: Message, *, delay_seconds: float
    ) -> None:
        if delay_seconds <= 0:
            await self.publish(queue, message)
            return
        await self.start()
        async with self._lock:
            heapq.heappush(
                self._scheduled,
                _Scheduled(
                    due_at=self._monotonic() + delay_seconds,
                    seq=next(_counter),
                    queue=queue,
                    message=message,
                ),
            )

    async def _run_scheduler(self) -> None:
        """Release scheduled messages when their delay elapses."""
        while not self._closed:
            await asyncio.sleep(0.02)
            now = self._monotonic()
            ready: list[_Scheduled] = []
            async with self._lock:
                while self._scheduled and self._scheduled[0].due_at <= now:
                    ready.append(heapq.heappop(self._scheduled))
                for item in ready:
                    self._queues[item.queue].append(_Entry(message=item.message))
                self._reclaim_expired_locked()
            if ready:
                self._arrival.set()

    def _reclaim_expired_locked(self) -> None:
        """Return messages whose lock lapsed -- the consumer crashed or stalled."""
        now = self._monotonic()
        for queue, entries in self._inflight.items():
            expired = [e for e in entries if e.locked_until <= now]
            for entry in expired:
                entries.remove(entry)
                if entry.delivery_count >= self.max_delivery_count:
                    self._dlq[queue].append(
                        (entry.message, "max_delivery_count_exceeded", "")
                    )
                else:
                    self._queues[queue].append(entry)
                    self._arrival.set()

    # -- consumer ----------------------------------------------------------
    async def consume(
        self, queue: Queue, *, max_messages: int = 1
    ) -> AsyncIterator[InMemoryReceivedMessage]:
        """Yield messages as they arrive, honouring lock timeouts."""
        await self.start()
        while not self._closed:
            batch = await self._checkout(queue, max_messages)
            if not batch:
                self._arrival.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._arrival.wait(), timeout=0.05)
                continue
            for entry in batch:
                yield InMemoryReceivedMessage(self, queue, entry)

    async def _checkout(self, queue: Queue, limit: int) -> list[_Entry]:
        async with self._lock:
            self._reclaim_expired_locked()
            entries = self._queues[queue][:limit]
            del self._queues[queue][: len(entries)]
            now = self._monotonic()
            for entry in entries:
                entry.delivery_count += 1
                entry.locked_until = now + self.lock_seconds
                self._inflight[queue].append(entry)
            return entries

    async def _remove_inflight(self, queue: Queue, entry: _Entry) -> None:
        async with self._lock:
            if entry in self._inflight[queue]:
                self._inflight[queue].remove(entry)

    async def _requeue(self, queue: Queue, entry: _Entry) -> None:
        async with self._lock:
            if entry in self._inflight[queue]:
                self._inflight[queue].remove(entry)
            if entry.delivery_count >= self.max_delivery_count:
                self._dlq[queue].append((entry.message, "max_delivery_count_exceeded", ""))
            else:
                entry.locked_until = 0.0
                self._queues[queue].append(entry)
        self._arrival.set()

    async def _dead_letter(
        self, queue: Queue, entry: _Entry, reason: str, description: str
    ) -> None:
        async with self._lock:
            if entry in self._inflight[queue]:
                self._inflight[queue].remove(entry)
            self._dlq[queue].append((entry.message, reason, description))
        logger.warning(
            "message dead-lettered",
            extra={"queue": queue.value, "reason": reason},
        )

    # -- operations --------------------------------------------------------
    async def dead_letters(self, queue: Queue, *, limit: int = 50) -> list[Message]:
        return [m for m, _, _ in self._dlq[queue][:limit]]

    async def replay_dead_letter(self, queue: Queue, message: Message) -> None:
        async with self._lock:
            remaining = [
                item for item in self._dlq[queue] if item[0].headers.message_id
                != message.headers.message_id
            ]
            self._dlq[queue] = remaining
            self._queues[queue].append(_Entry(message=message))
        self._arrival.set()

    def depth(self, queue: Queue) -> int:
        """Queue depth, for backpressure metrics and tests."""
        return len(self._queues[queue])

    def inflight_count(self, queue: Queue) -> int:
        return len(self._inflight[queue])

    def dead_letter_count(self, queue: Queue) -> int:
        return len(self._dlq[queue])

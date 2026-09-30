"""The agent-worker loop.

One process consumes several queues concurrently, bounded by a semaphore so a
burst of work cannot exhaust memory or LLM quota.  Backpressure is the point:
when 10,000 workflows arrive, the queue grows and the workers keep their own
pace, rather than the platform trying to run 10,000 agents at once.
"""

from __future__ import annotations

import asyncio
import contextlib

from cortexflow.modules.workflow.application.executor import StepExecutor, result_queue
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import ExecuteStep, Message
from cortexflow.modules.workflow.ports.messaging import MessageBus, ReceivedMessage
from cortexflow.modules.workflow.ports.repositories import DeadLetterRepository
from cortexflow.shared.errors import ErrorClass, classify
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.ports.cache import Cache

logger = get_logger(__name__)

MAX_DELIVERY_ATTEMPTS = 5


class AgentWorker:
    """Consumes step commands from one or more queues."""

    def __init__(
        self,
        *,
        executor: StepExecutor,
        bus: MessageBus,
        cache: Cache,
        dead_letters: DeadLetterRepository,
        queues: tuple[Queue, ...],
        concurrency: int = 8,
        dedup_ttl_seconds: int = 86_400,
    ) -> None:
        self._executor = executor
        self._bus = bus
        self._cache = cache
        self._dead_letters = dead_letters
        self._queues = queues
        self._semaphore = asyncio.Semaphore(concurrency)
        self._dedup_ttl = dedup_ttl_seconds
        self._metrics = get_metrics()
        self._stopping = asyncio.Event()
        self._inflight: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        """Consume every configured queue until stopped."""
        logger.info(
            "agent worker started",
            extra={"queues": [q.value for q in self._queues]},
        )
        consumers = [asyncio.create_task(self._consume(q)) for q in self._queues]
        try:
            await asyncio.gather(*consumers)
        finally:
            await self._drain()
        logger.info("agent worker stopped")

    def stop(self) -> None:
        self._stopping.set()

    async def _drain(self) -> None:
        """Let in-flight steps finish so a shutdown is not a crash."""
        if self._inflight:
            logger.info("draining in-flight steps", extra={"count": len(self._inflight)})
            await asyncio.gather(*self._inflight, return_exceptions=True)

    async def _consume(self, queue: Queue) -> None:
        async for received in self._bus.consume(queue, max_messages=1):
            if self._stopping.is_set():
                await received.abandon("shutting down")
                break
            await self._semaphore.acquire()
            task = asyncio.create_task(self._handle(received, queue))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            task.add_done_callback(lambda _: self._semaphore.release())

    async def _handle(self, received: ReceivedMessage, queue: Queue) -> None:
        message = received.message
        body = message.body
        if not isinstance(body, ExecuteStep):
            logger.warning(
                "worker queue received an unexpected message type",
                extra={"queue": queue.value, "message_type": str(message.type)},
            )
            await received.complete()
            return

        try:
            if await self._is_duplicate(message):
                await received.complete()
                return

            result = await self._executor.execute(body, tenant_id=message.headers.tenant_id)
            # The result goes to the control plane *before* the command is
            # settled, so a crash in between redelivers the command rather
            # than losing the outcome.
            await self._bus.publish(result_queue(), _with_causation(result, message))
            await received.complete()

        except Exception as exc:
            await self._settle_failure(received, queue, exc)

    async def _is_duplicate(self, message: Message) -> bool:
        key = message.headers.dedup_key or message.headers.message_id
        if not key:
            return False
        first_time = await self._cache.add(
            f"dedup:{key}", message.headers.message_id, ttl_seconds=self._dedup_ttl
        )
        if first_time:
            return False
        self._metrics.count(self._metrics.duplicates, message_type=str(message.type))
        logger.info("duplicate step command dropped", extra={"dedup_key": key})
        return True

    async def _settle_failure(
        self, received: ReceivedMessage, queue: Queue, exc: Exception
    ) -> None:
        """Only *infrastructure* failures reach here.

        A step that failed on its own merits was already turned into a
        ``StepFailed`` report by the executor.  This path means the worker
        could not even do that -- the state store or the bus was unreachable.
        """
        error_class = classify(exc)
        exhausted = received.delivery_count >= MAX_DELIVERY_ATTEMPTS

        if error_class is not ErrorClass.PERMANENT and not exhausted:
            logger.warning(
                "worker could not process a step command; returning it",
                extra={
                    "queue": queue.value,
                    "error": str(exc),
                    "delivery_count": received.delivery_count,
                },
            )
            await received.abandon(str(exc))
            return

        reason = "permanent_failure" if exhausted is False else "max_delivery_attempts"
        logger.error(
            "step command dead-lettered",
            extra={"queue": queue.value, "reason": reason, "error": str(exc)},
        )
        self._metrics.count(self._metrics.dead_letters, queue=queue.value)
        with contextlib.suppress(Exception):
            await self._dead_letters.record(
                tenant_id=received.message.headers.tenant_id,
                queue=queue.value,
                message=received.message.to_transport(),
                reason=f"{reason}: {exc}",
                attempts=received.delivery_count,
            )
        await received.dead_letter(reason, str(exc)[:2000])


def _with_causation(result: Message, cause: Message) -> Message:
    """Link the result to the command that produced it, for trace continuity."""
    headers = result.headers.model_copy(
        update={
            "correlation_id": result.headers.correlation_id or cause.headers.correlation_id,
            "causation_id": cause.headers.message_id,
        }
    )
    return Message(headers=headers, body=result.body)

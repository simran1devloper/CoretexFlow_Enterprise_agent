"""Azure Service Bus adapter.

Service Bus provides the durability, competing-consumer scaling and
dead-lettering the execution plane relies on.  This adapter keeps the vendor
semantics behind the :class:`MessageBus` port: message locks become
``ReceivedMessage`` settlement, scheduled enqueue becomes ``schedule``, and
the built-in DLQ becomes ``dead_letters``/``replay_dead_letter``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from cortexflow.config.settings import ServiceBusSettings
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import Message
from cortexflow.modules.workflow.domain.topics import queue_name
from cortexflow.shared.errors import (
    DependencyTimeoutError,
    DependencyUnavailableError,
)
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


class ServiceBusReceivedMessage:
    """Wraps a Service Bus message with the platform's settlement contract."""

    def __init__(self, receiver: Any, raw: Any, queue: Queue) -> None:
        self._receiver = receiver
        self._raw = raw
        self._queue = queue
        self._message = Message.from_transport(json.loads(str(raw)))

    @property
    def message(self) -> Message:
        return self._message

    @property
    def delivery_count(self) -> int:
        return int(self._raw.delivery_count or 0) + 1

    @property
    def queue(self) -> Queue:
        return self._queue

    async def complete(self) -> None:
        await self._receiver.complete_message(self._raw)

    async def abandon(self, reason: str = "") -> None:
        await self._receiver.abandon_message(self._raw)

    async def dead_letter(self, reason: str, description: str = "") -> None:
        await self._receiver.dead_letter_message(
            self._raw, reason=reason[:128], error_description=description[:4096]
        )

    async def renew_lock(self) -> None:
        await self._receiver.renew_message_lock(self._raw)


class ServiceBusMessageBus:
    """Connection-pooling Service Bus client keyed by queue."""

    def __init__(self, settings: ServiceBusSettings) -> None:
        self._settings = settings
        self._client: Any = None
        self._senders: dict[str, Any] = {}

    async def start(self) -> None:
        if self._client is not None:
            return
        from azure.servicebus.aio import ServiceBusClient

        if self._settings.connection_string:
            self._client = ServiceBusClient.from_connection_string(
                self._settings.connection_string
            )
        else:
            from azure.identity.aio import DefaultAzureCredential

            self._client = ServiceBusClient(
                fully_qualified_namespace=self._settings.namespace,
                credential=DefaultAzureCredential(),
            )

    async def close(self) -> None:
        for sender in self._senders.values():
            await sender.close()
        self._senders.clear()
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _name(self, queue: Queue) -> str:
        return queue_name(queue, self._settings.queue_prefix)

    async def _sender(self, queue: Queue) -> Any:
        await self.start()
        name = self._name(queue)
        if name not in self._senders:
            self._senders[name] = self._client.get_queue_sender(queue_name=name)
        return self._senders[name]

    def _encode(self, message: Message) -> Any:
        from azure.servicebus import ServiceBusMessage

        return ServiceBusMessage(
            body=json.dumps(message.to_transport()),
            content_type="application/json",
            message_id=message.headers.message_id,
            correlation_id=message.headers.correlation_id or None,
            # Service Bus duplicate detection is a cheap first line of defence;
            # the application-level dedup guard remains the authoritative one.
            session_id=None,
            application_properties={
                "tenant_id": message.headers.tenant_id,
                "message_type": str(message.type),
                "dedup_key": message.headers.dedup_key,
            },
        )

    async def publish(self, queue: Queue, message: Message) -> None:
        sender = await self._sender(queue)
        try:
            await sender.send_messages(self._encode(message))
        except Exception as exc:  # pragma: no cover - requires a live namespace
            raise _translate(exc, queue=self._name(queue)) from exc

    async def publish_many(self, queue: Queue, messages: list[Message]) -> None:
        if not messages:
            return
        sender = await self._sender(queue)
        try:
            await sender.send_messages([self._encode(m) for m in messages])
        except Exception as exc:  # pragma: no cover
            raise _translate(exc, queue=self._name(queue)) from exc

    async def schedule(
        self, queue: Queue, message: Message, *, delay_seconds: float
    ) -> None:
        if delay_seconds <= 0:
            await self.publish(queue, message)
            return
        from datetime import timedelta

        from cortexflow.shared.clock import utcnow

        sender = await self._sender(queue)
        try:
            await sender.schedule_messages(
                self._encode(message), utcnow() + timedelta(seconds=delay_seconds)
            )
        except Exception as exc:  # pragma: no cover
            raise _translate(exc, queue=self._name(queue)) from exc

    async def consume(
        self, queue: Queue, *, max_messages: int = 1
    ) -> AsyncIterator[ServiceBusReceivedMessage]:
        await self.start()
        receiver = self._client.get_queue_receiver(
            queue_name=self._name(queue), prefetch_count=self._settings.prefetch
        )
        async with receiver:
            while True:
                batch = await receiver.receive_messages(
                    max_message_count=max_messages,
                    max_wait_time=self._settings.max_wait_seconds,
                )
                for raw in batch:
                    yield ServiceBusReceivedMessage(receiver, raw, queue)

    async def dead_letters(self, queue: Queue, *, limit: int = 50) -> list[Message]:
        from azure.servicebus import ServiceBusSubQueue

        await self.start()
        receiver = self._client.get_queue_receiver(
            queue_name=self._name(queue), sub_queue=ServiceBusSubQueue.DEAD_LETTER
        )
        async with receiver:
            batch = await receiver.peek_messages(max_message_count=limit)
            return [Message.from_transport(json.loads(str(raw))) for raw in batch]

    async def replay_dead_letter(self, queue: Queue, message: Message) -> None:
        """Resubmit a dead-lettered message to its original queue.

        Operators replay after fixing the underlying cause; the platform's
        idempotency keys make the replay safe even if the original attempt
        partially succeeded.
        """
        from azure.servicebus import ServiceBusSubQueue

        await self.start()
        receiver = self._client.get_queue_receiver(
            queue_name=self._name(queue), sub_queue=ServiceBusSubQueue.DEAD_LETTER
        )
        async with receiver:
            batch = await receiver.receive_messages(max_message_count=50, max_wait_time=5)
            for raw in batch:
                if raw.message_id == message.headers.message_id:
                    await self.publish(queue, message)
                    await receiver.complete_message(raw)
                    return
                await receiver.abandon_message(raw)


def _translate(exc: Exception, **context: Any) -> Exception:
    name = type(exc).__name__
    if "Timeout" in name:
        return DependencyTimeoutError("Service Bus timed out", **context)
    if "ServiceBus" in name or "Connection" in name:
        return DependencyUnavailableError("Service Bus unavailable", error=name, **context)
    return exc

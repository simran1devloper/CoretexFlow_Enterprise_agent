"""The notification service.

Consumes the notification queue and delivers messages.  Separate from the API
so that an email provider outage degrades notifications only -- it can never
block a payment or a workflow transition.
"""

from __future__ import annotations

import asyncio
import contextlib

from cortexflow.apps.notification_service.sender import LoggingNotificationSender
from cortexflow.apps.runtime import ServiceHost, run_service
from cortexflow.composition import Container
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.notification.ports.notifications import (
    NotificationRequest,
    NotificationSender,
)
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import Notify
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


class NotificationConsumer:
    def __init__(self, container: Container, sender: NotificationSender) -> None:
        self._container = container
        self._sender = sender
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        async for received in self._container.bus.consume(Queue.NOTIFICATION):
            if self._stopping.is_set():
                await received.abandon("shutting down")
                break

            body = received.message.body
            if not isinstance(body, Notify):
                await received.complete()
                continue

            try:
                await self._deliver(body, received.message.headers.tenant_id)
                await received.complete()
            except Exception as exc:
                # Notifications are best-effort. Retry a few times, then drop
                # them to the DLQ rather than blocking the queue.
                if received.delivery_count >= 3:
                    logger.error("giving up on a notification", extra={"error": str(exc)})
                    await received.dead_letter("delivery_failed", str(exc)[:1000])
                else:
                    await received.abandon(str(exc))

    async def _deliver(self, notify: Notify, tenant_id: str) -> None:
        context = notify.context
        request = NotificationRequest(
            channel=notify.channel,
            recipients=notify.recipients,
            subject=str(context.get("subject", "CortexFlow update")),
            body=str(context.get("body", "")),
            tenant_id=tenant_id,
            workflow_id=notify.workflow_id,
            metadata={"template": notify.template, "approval_id": notify.approval_id},
        )
        provider_id = await self._sender.send(request)
        with contextlib.suppress(Exception):
            await self._container.audit.record(
                tenant_id=tenant_id,
                action=AuditAction.NOTIFICATION_SENT,
                workflow_id=notify.workflow_id,
                summary=f"Notification sent via {notify.channel}",
                metadata={"provider_message_id": provider_id},
            )


async def serve(container: Container) -> None:
    host = ServiceHost("notification-service", container)
    consumer = NotificationConsumer(container, LoggingNotificationSender())

    host.supervise(consumer.run(), stop=consumer.stop)
    logger.info("notification service running")
    await host.serve()


def run() -> None:  # pragma: no cover - console entrypoint
    run_service("notification-service", serve)


if __name__ == "__main__":  # pragma: no cover
    run()

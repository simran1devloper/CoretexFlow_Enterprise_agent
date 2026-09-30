"""Notification delivery.

The local sender logs instead of sending, so the demo is safe to run.  In
Azure this is replaced by Communication Services or Graph.  Either way the
payload is redacted first: a notification leaves the trust boundary.
"""

from __future__ import annotations

from cortexflow.modules.notification.ports.notifications import NotificationRequest
from cortexflow.shared.ids import MESSAGE, UuidIdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.redaction import redact, scrub_text

logger = get_logger(__name__)


class LoggingNotificationSender:
    """Records what would be sent. Used locally and in tests."""

    def __init__(self) -> None:
        self._ids = UuidIdGenerator()
        self.sent: list[NotificationRequest] = []

    async def send(self, request: NotificationRequest) -> str:
        message_id = self._ids.new_id(MESSAGE)
        self.sent.append(request)
        logger.info(
            "notification sent",
            extra={
                "channel": request.channel,
                "recipients": len(request.recipients),
                "subject": scrub_text(request.subject),
                "workflow_id": request.workflow_id,
                "metadata": redact(request.metadata),
            },
        )
        return message_id

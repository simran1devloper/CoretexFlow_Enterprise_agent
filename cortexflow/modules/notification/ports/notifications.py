"""Notification port -- email, Teams, webhook."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class NotificationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel: str
    recipients: tuple[str, ...]
    subject: str
    body: str
    tenant_id: str = ""
    workflow_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class NotificationSender(Protocol):
    async def send(self, request: NotificationRequest) -> str:
        """Deliver, returning a provider message id."""
        ...

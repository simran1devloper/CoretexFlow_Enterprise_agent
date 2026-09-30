"""Message contracts carried by the execution backbone.

Messages are *pointers plus intent*, never business payloads.  A task message
says "run step X of workflow Y, attempt N"; the worker loads the authoritative
state from the durable store.  That keeps messages small, keeps sensitive data
out of the queue, and makes a duplicate delivery harmless.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.workflow.domain.definition import StepType
from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import ErrorClass


class MessageType(StrEnum):
    ADVANCE_WORKFLOW = "ADVANCE_WORKFLOW"
    """Ask the orchestrator to re-evaluate a workflow's DAG."""

    EXECUTE_STEP = "EXECUTE_STEP"
    """Ask a worker to execute one step."""

    STEP_COMPLETED = "STEP_COMPLETED"
    STEP_FAILED = "STEP_FAILED"
    APPROVAL_RESOLVED = "APPROVAL_RESOLVED"
    NOTIFY = "NOTIFY"


class MessageHeaders(BaseModel):
    """Envelope metadata common to every message."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    tenant_id: str
    correlation_id: str = ""
    causation_id: str = ""
    """The message that caused this one -- the audit chain through the bus."""

    published_at: datetime = Field(default_factory=utcnow)
    attempt: int = 1
    dedup_key: str = ""
    """Stable across redeliveries; the consumer's exactly-once guard."""

    scheduled_for: datetime | None = None


class AdvanceWorkflow(BaseModel):
    """Control-plane trigger. Idempotent by construction: it only re-reads state."""

    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.ADVANCE_WORKFLOW] = MessageType.ADVANCE_WORKFLOW
    workflow_id: str
    reason: str = ""


class ExecuteStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.EXECUTE_STEP] = MessageType.EXECUTE_STEP
    workflow_id: str
    step_id: str
    step_type: StepType
    run_id: str
    """Guards against a stale worker: a result whose run_id no longer matches
    the current attempt is discarded rather than applied."""

    attempt: int = 1
    idempotency_key: str = ""
    deadline: datetime | None = None


class StepCompleted(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.STEP_COMPLETED] = MessageType.STEP_COMPLETED
    workflow_id: str
    step_id: str
    run_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    duration_ms: int = 0
    cost: dict[str, Any] = Field(default_factory=dict)


class StepFailed(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.STEP_FAILED] = MessageType.STEP_FAILED
    workflow_id: str
    step_id: str
    run_id: str
    code: str
    message: str
    error_class: ErrorClass = ErrorClass.UNKNOWN
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class ApprovalResolved(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.APPROVAL_RESOLVED] = MessageType.APPROVAL_RESOLVED
    workflow_id: str
    step_id: str
    approval_id: str
    decision: str
    decided_by: str
    comment: str = ""


class Notify(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[MessageType.NOTIFY] = MessageType.NOTIFY
    channel: str
    recipients: tuple[str, ...] = ()
    template: str = ""
    workflow_id: str | None = None
    approval_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


MessageBody = (
    AdvanceWorkflow | ExecuteStep | StepCompleted | StepFailed | ApprovalResolved | Notify
)

_BODY_TYPES: dict[MessageType, type[BaseModel]] = {
    MessageType.ADVANCE_WORKFLOW: AdvanceWorkflow,
    MessageType.EXECUTE_STEP: ExecuteStep,
    MessageType.STEP_COMPLETED: StepCompleted,
    MessageType.STEP_FAILED: StepFailed,
    MessageType.APPROVAL_RESOLVED: ApprovalResolved,
    MessageType.NOTIFY: Notify,
}


class Message(BaseModel):
    """A typed envelope: headers for transport, body for intent."""

    model_config = ConfigDict(frozen=True)

    headers: MessageHeaders
    body: MessageBody = Field(discriminator="type")

    @property
    def type(self) -> MessageType:
        return self.body.type

    def to_transport(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_transport(cls, raw: dict[str, Any]) -> Message:
        """Rebuild an envelope from the wire, resolving the body union."""
        headers = MessageHeaders.model_validate(raw["headers"])
        body_raw = raw["body"]
        body_type = MessageType(body_raw["type"])
        body = _BODY_TYPES[body_type].model_validate(body_raw)
        return cls(headers=headers, body=body)  # type: ignore[arg-type]

"""The audit trail.

The question the audit store must answer is "why did this action happen?".
Every entry therefore names an actor, an action, a target and an outcome, and
carries the causal chain (workflow, step, run, correlation id) that links a
payment back to the email that triggered it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.shared.clock import utcnow


class AuditAction(StrEnum):
    """Every action the trail can record.

    Each member must have a call site. A declared-but-never-emitted action is
    worse than no action at all: it promises a record that does not exist, and
    anything reading the trail for it finds nothing and cannot tell absence
    from never-emitted. ``WORKFLOW_STATUS_CHANGED`` and ``STEP_STARTED`` were
    removed for exactly that reason, and a test now fails if another appears.

    Removing them was safe by construction -- nothing had ever written one, so
    no stored event carries the value.
    """

    WORKFLOW_CREATED = "WORKFLOW_CREATED"
    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
    WORKFLOW_REPLAYED = "WORKFLOW_REPLAYED"

    STEP_DISPATCHED = "STEP_DISPATCHED"
    STEP_SUCCEEDED = "STEP_SUCCEEDED"
    STEP_FAILED = "STEP_FAILED"
    STEP_SKIPPED = "STEP_SKIPPED"
    STEP_RETRY_SCHEDULED = "STEP_RETRY_SCHEDULED"
    STEP_DEAD_LETTERED = "STEP_DEAD_LETTERED"
    STEP_LEASE_RECLAIMED = "STEP_LEASE_RECLAIMED"

    AGENT_PROPOSED = "AGENT_PROPOSED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    POLICY_DENIED = "POLICY_DENIED"

    POLICY_PUBLISHED = "POLICY_PUBLISHED"
    """An administrator changed the rules a tenant is authorized against.

    The most consequential thing in this enum: policy is what authorizes, so
    this is the only record that a threshold moved without passing a code
    review. It carries the diff against whatever it replaced.
    """

    POLICY_WITHDRAWN = "POLICY_WITHDRAWN"
    """A tenant ruleset was removed, so the shipped one applies again."""

    TOOL_AUTHORIZED = "TOOL_AUTHORIZED"
    TOOL_DENIED = "TOOL_DENIED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    TOOL_REPLAYED = "TOOL_REPLAYED"
    """A duplicate call short-circuited by the idempotency store."""

    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    APPROVAL_INFO_REQUESTED = "APPROVAL_INFO_REQUESTED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"

    MESSAGE_DEDUPLICATED = "MESSAGE_DEDUPLICATED"
    NOTIFICATION_SENT = "NOTIFICATION_SENT"


class AuditOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"
    PENDING = "PENDING"


class AuditEvent(BaseModel):
    """An append-only audit record. Never mutated, never deleted in place."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    tenant_id: str
    action: AuditAction
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    actor: str
    """``user::alice@acme.com``, ``agent::decision_agent`` or ``system::orchestrator``."""

    actor_roles: tuple[str, ...] = ()
    workflow_id: str | None = None
    step_id: str | None = None
    run_id: str | None = None
    tool: str | None = None
    correlation_id: str = ""

    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Redacted, bounded detail. Business payloads live in the data plane, not here."""

    occurred_at: datetime = Field(default_factory=utcnow)

    @property
    def partition_key(self) -> str:
        return self.tenant_id

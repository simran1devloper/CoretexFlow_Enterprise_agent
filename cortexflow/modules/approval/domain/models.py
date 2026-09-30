"""Human-in-the-loop records.

A pending approval is a *document*, not a suspended coroutine.  No process
waits for a human: the workflow is checkpointed, the approval is durable, and
resolving it re-enters the orchestrator through the control queue.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.shared.clock import utcnow
from cortexflow.shared.identity import Role


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INFO_REQUESTED = "INFO_REQUESTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self is not ApprovalStatus.PENDING and self is not ApprovalStatus.INFO_REQUESTED


class ApprovalDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_INFO = "REQUEST_INFO"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _RISK_RANK[self]


_RISK_RANK = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class ApprovalResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: ApprovalDecision
    decided_by: str
    decided_by_name: str = ""
    comment: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class Approval(BaseModel):
    """A durable request for a human decision."""

    approval_id: str
    workflow_id: str
    step_id: str
    tenant_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING

    title: str
    description: str = ""
    risk: RiskLevel = RiskLevel.MEDIUM
    required_roles: tuple[Role, ...] = ()

    context: dict[str, Any] = Field(default_factory=dict)
    """A redacted snapshot: enough for a human to decide, no raw documents."""

    proposed_action: dict[str, Any] = Field(default_factory=dict)
    """What the agent recommended and why -- shown verbatim to the approver."""

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    on_timeout: str = "escalate"

    resolution: ApprovalResolution | None = None
    version: int = 0
    etag: str | None = None

    @property
    def partition_key(self) -> str:
        return self.tenant_id

    def can_be_decided_by_roles(self, roles: frozenset[Role]) -> bool:
        return not self.required_roles or bool(roles.intersection(self.required_roles))

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at

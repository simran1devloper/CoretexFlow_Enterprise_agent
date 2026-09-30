"""Tool metadata -- the security boundary between agents and enterprise systems.

Without a registry, an agent that can call twenty APIs has twenty implicit
permissions.  With one, every call passes a declared contract: which domain it
belongs to, how risky it is, who may authorize it, and whether it mutates
anything.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.shared.clock import utcnow
from cortexflow.shared.identity import Department, Role


class SideEffect(StrEnum):
    READ = "READ"
    """No state change; safe to retry freely."""

    WRITE = "WRITE"
    """Mutates enterprise state; requires an idempotency key."""

    IRREVERSIBLE = "IRREVERSIBLE"
    """Money movement, terminations. Never auto-retried without an explicit key."""


class ToolMetadata(BaseModel):
    """The declarative contract of one enterprise capability."""

    model_config = ConfigDict(frozen=True)

    name: str
    domain: Department
    description: str
    side_effect: SideEffect = SideEffect.READ
    risk: RiskLevel = RiskLevel.LOW
    requires_human_approval: bool = False
    allowed_roles: tuple[Role, ...] = ()
    """Empty means "any authenticated principal in the domain"."""

    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    idempotent: bool = True
    exposed_to_agents: bool = True
    """False keeps a tool orchestrator-only, out of every agent's tool list."""

    rate_limit_per_minute: int | None = None

    @property
    def requires_idempotency_key(self) -> bool:
        return self.side_effect is not SideEffect.READ


class ToolCall(BaseModel):
    """A fully-formed, authorized request to execute a tool."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    workflow_id: str = ""
    step_id: str = ""
    tenant_id: str = ""
    correlation_id: str = ""
    requested_at: datetime = Field(default_factory=utcnow)


class ToolResult(BaseModel):
    """The outcome of a tool execution."""

    model_config = ConfigDict(frozen=True)

    tool: str
    succeeded: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""
    duration_ms: int = 0
    replayed: bool = False
    """True when the idempotency store returned a previously-computed result."""

    idempotency_key: str = ""

    @classmethod
    def ok(cls, tool: str, data: dict[str, Any], **kwargs: Any) -> ToolResult:
        return cls(tool=tool, succeeded=True, data=data, **kwargs)

    @classmethod
    def failed(cls, tool: str, code: str, message: str, **kwargs: Any) -> ToolResult:
        return cls(
            tool=tool, succeeded=False, error_code=code, error_message=message, **kwargs
        )


class ToolAuthorization(BaseModel):
    """The registry's verdict on whether a call may proceed."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    tool: str
    reason: str = ""
    requires_approval: bool = False
    risk: RiskLevel = RiskLevel.LOW

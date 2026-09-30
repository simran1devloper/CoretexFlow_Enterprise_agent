"""The agent contract.

An agent's output is a *structured result*, never free text.  The orchestrator
reads `status`, `output`, `confidence` and `tool_requests` -- it never parses
prose.  That is what lets a probabilistic component sit inside a deterministic
workflow without contaminating it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import ErrorClass
from cortexflow.shared.identity import Principal


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    """The agent itself determined it should not decide this case."""


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merged_with(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            model=self.model or other.model,
        )


class ToolRequest(BaseModel):
    """An agent's *request* to act -- not the action itself.

    The registry authorizes it, the policy engine may veto it, and only then
    does anything reach an enterprise system.
    """

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolInvocation(BaseModel):
    """The record of a tool actually executed, for the audit trail."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    duration_ms: int = 0
    succeeded: bool = True
    error: str = ""


class AgentContext(BaseModel):
    """Everything an agent is allowed to see about the workflow it serves."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    tenant_id: str
    workflow_type: str
    step_id: str
    run_id: str
    attempt: int = 1
    principal: Principal
    inputs: dict[str, Any] = Field(default_factory=dict)
    """Bindings resolved from the step definition -- the agent's actual task input."""

    workflow_input: dict[str, Any] = Field(default_factory=dict)
    step_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    """Hard allowlist; the agent runtime exposes nothing beyond this."""

    deadline: datetime | None = None
    correlation_id: str = ""


class AgentError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    error_class: ErrorClass = ErrorClass.UNKNOWN
    details: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """The structured contract every agent returns."""

    status: AgentStatus
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    """The model's self-reported confidence. A signal, never an authorization."""

    reason: str = ""
    tool_requests: tuple[ToolRequest, ...] = ()
    tool_invocations: tuple[ToolInvocation, ...] = ()
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: AgentError | None = None
    completed_at: datetime = Field(default_factory=utcnow)
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is AgentStatus.SUCCESS

    @classmethod
    def success(cls, output: dict[str, Any], **kwargs: Any) -> AgentResult:
        return cls(status=AgentStatus.SUCCESS, output=output, **kwargs)

    @classmethod
    def needs_human(cls, reason: str, **kwargs: Any) -> AgentResult:
        return cls(status=AgentStatus.NEEDS_HUMAN, reason=reason, **kwargs)

    @classmethod
    def failure(cls, error: AgentError, **kwargs: Any) -> AgentResult:
        return cls(status=AgentStatus.FAILED, error=error, **kwargs)

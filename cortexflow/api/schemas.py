"""API request and response models.

Deliberately separate from the domain models: the wire contract should be able
to stay stable while internal state evolves, and nothing internal (lease
owners, etags, raw inputs) should leak to a client by accident.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.approval.domain.models import (
    Approval,
    ApprovalDecision,
    ApprovalStatus,
    RiskLevel,
)
from cortexflow.modules.audit.domain.models import AuditEvent
from cortexflow.modules.workflow.domain.dag import parallel_levels
from cortexflow.modules.workflow.domain.definition import StepType, WorkflowDefinition
from cortexflow.modules.workflow.domain.models import (
    ExecutionMode,
    StepStatus,
    Workflow,
    WorkflowStatus,
    WorkflowSummary,
)


class CreateWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_type: str = Field(description="Workflow definition name")
    version: int | None = Field(default=None, description="Pin a definition version")
    input: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.LIVE,
        description=(
            "SIMULATION runs the workflow for real -- same engine, agents and "
            "rules -- but holds back every write and raises no approvals, so "
            "nothing outside CortexFlow is touched. Defaults to LIVE: a caller "
            "that forgets the flag gets the safe surprise, not the other one."
        ),
    )


class StepView(BaseModel):
    step_id: str
    name: str
    type: StepType
    status: StepStatus
    depends_on: tuple[str, ...] = ()
    attempts: int = 0
    confidence: float | None = None
    duration_ms: int | None = None
    cost: dict[str, Any] = {}
    """Token and model accounting, for agent steps.

    Recorded by the executor since the engine was written; exposed here
    because a cost the platform measures and never reports is a cost nobody
    can manage.
    """

    approval_id: str | None = None
    error: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    next_attempt_at: datetime | None = None


class WorkflowView(BaseModel):
    workflow_id: str
    tenant_id: str
    workflow_type: str
    version: int
    status: WorkflowStatus
    execution_mode: ExecutionMode = ExecutionMode.LIVE
    created_at: datetime
    updated_at: datetime
    created_by: str
    correlation_id: str
    labels: dict[str, str] = Field(default_factory=dict)
    steps: list[StepView] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    cancellation_reason: str | None = None
    progress: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def build(cls, workflow: Workflow, definition: WorkflowDefinition) -> WorkflowView:
        steps = []
        for step_id in definition.topological_order():
            spec = definition.step(step_id)
            state = workflow.step(step_id)
            steps.append(
                StepView(
                    step_id=step_id,
                    name=spec.display_name,
                    type=spec.type,
                    status=state.status,
                    depends_on=spec.depends_on,
                    attempts=state.attempts,
                    confidence=state.confidence,
                    duration_ms=state.duration_ms,
                    cost=state.cost,
                    approval_id=state.approval_id,
                    error=state.error.model_dump(mode="json") if state.error else None,
                    output=state.output,
                    started_at=state.started_at or state.dispatched_at,
                    ended_at=state.ended_at,
                    next_attempt_at=state.next_attempt_at,
                )
            )
        completed = len(workflow.completed_step_ids())
        return cls(
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            workflow_type=workflow.definition_name,
            version=workflow.definition_version,
            status=workflow.status,
            execution_mode=workflow.execution_mode,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            correlation_id=workflow.correlation_id,
            labels=workflow.labels,
            steps=steps,
            error=workflow.error.model_dump(mode="json") if workflow.error else None,
            cancellation_reason=workflow.cancellation_reason,
            progress={
                # "settled" drives a progress bar: a skipped branch is done,
                # even though it did not succeed. "completed" is the narrower
                # count of steps that actually ran successfully.
                "settled": sum(1 for s in workflow.steps.values() if s.status.is_terminal),
                "completed": completed,
                "total": len(workflow.steps),
                "in_flight": len(workflow.in_flight_step_ids()),
            },
        )


class WorkflowListResponse(BaseModel):
    items: list[WorkflowSummary]
    continuation: str | None = None


class CancelWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="Cancelled by user", max_length=500)


class ApprovalView(BaseModel):
    approval_id: str
    workflow_id: str
    step_id: str
    status: ApprovalStatus
    title: str
    description: str
    risk: RiskLevel
    required_roles: tuple[str, ...] = ()
    context: dict[str, Any] = Field(default_factory=dict)
    proposed_action: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime | None = None
    resolution: dict[str, Any] | None = None

    @classmethod
    def build(cls, approval: Approval) -> ApprovalView:
        return cls(
            approval_id=approval.approval_id,
            workflow_id=approval.workflow_id,
            step_id=approval.step_id,
            status=approval.status,
            title=approval.title,
            description=approval.description,
            risk=approval.risk,
            required_roles=tuple(str(r) for r in approval.required_roles),
            context=approval.context,
            proposed_action=approval.proposed_action,
            created_at=approval.created_at,
            expires_at=approval.expires_at,
            resolution=(
                approval.resolution.model_dump(mode="json") if approval.resolution else None
            ),
        )


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: str = Field(default="", max_length=2000)


class AuditEventView(BaseModel):
    event_id: str
    action: str
    outcome: str
    actor: str
    workflow_id: str | None = None
    step_id: str | None = None
    tool: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime

    @classmethod
    def build(cls, event: AuditEvent) -> AuditEventView:
        return cls(
            event_id=event.event_id,
            action=str(event.action),
            outcome=str(event.outcome),
            actor=event.actor,
            workflow_id=event.workflow_id,
            step_id=event.step_id,
            tool=event.tool,
            summary=event.summary,
            metadata=event.metadata,
            occurred_at=event.occurred_at,
        )


class WorkflowDefinitionView(BaseModel):
    """A definition rendered for the dashboard's graph view."""

    name: str
    version: int
    department: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    parallel_levels: list[list[str]] = Field(default_factory=list)

    @classmethod
    def build(cls, definition: WorkflowDefinition) -> WorkflowDefinitionView:
        return cls(
            name=definition.name,
            version=definition.version,
            department=str(definition.department),
            description=definition.description,
            input_schema=definition.input_schema,
            steps=[
                {
                    "id": s.id,
                    "name": s.display_name,
                    "type": str(s.type),
                    "depends_on": list(s.depends_on),
                    "join": str(s.join),
                    "when": s.when,
                    "queue": s.effective_queue.value,
                    "critical": s.critical,
                    "agent": s.agent,
                    "tool": s.tool,
                    "ruleset": s.ruleset,
                }
                for s in definition.steps
            ],
            parallel_levels=[list(level) for level in parallel_levels(definition)],
        )


class ToolMetadataView(BaseModel):
    name: str
    domain: str
    description: str
    side_effect: str
    risk: str
    requires_human_approval: bool
    allowed_roles: list[str] = Field(default_factory=list)
    exposed_to_agents: bool = True


class ErrorResponse(BaseModel):
    code: str
    message: str
    error_class: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    profile: str
    workflows: list[str] = Field(default_factory=list)


__all__ = [
    "ApprovalDecision",
    "ApprovalDecisionRequest",
    "ApprovalView",
    "AuditEventView",
    "CancelWorkflowRequest",
    "CreateWorkflowRequest",
    "ErrorResponse",
    "HealthResponse",
    "StepView",
    "ToolMetadataView",
    "WorkflowDefinitionView",
    "WorkflowListResponse",
    "WorkflowView",
]

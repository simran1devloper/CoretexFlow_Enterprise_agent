"""Runtime workflow state -- the durable record the platform is built around.

This aggregate is the single source of truth.  Redis may cache it and Service
Bus may carry pointers to it, but recovery only ever reads *this* document.
Every mutation bumps ``version`` so the repository can enforce optimistic
concurrency and two workers can never silently overwrite each other.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.workflow.domain.definition import StepType
from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import ErrorClass, NotFoundError


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_RECOVERY = "WAITING_FOR_RECOVERY"
    """Exhausted retries; parked for an operator instead of lost."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_WORKFLOW_STATES

    @property
    def is_suspended(self) -> bool:
        return self in {WorkflowStatus.WAITING_FOR_APPROVAL, WorkflowStatus.WAITING_FOR_RECOVERY}


_TERMINAL_WORKFLOW_STATES = frozenset(
    {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.REJECTED,
        WorkflowStatus.CANCELLED,
    }
)


class ExecutionMode(StrEnum):
    """Whether a run may change anything outside CortexFlow.

    A workflow is only trustworthy if someone can find out what it will do
    before it does it. Reading the definition is not that: the definition says
    a payment tool is wired up, not that *these* eleven claims would be paid
    and *this* one would be refused for a reason nobody intended.

    So simulation is the same run -- same engine, same agents, same rules,
    same graph -- with the two things that touch the outside world held back.
    It is a mode of the real execution path rather than a separate predictor,
    because a predictor that drifts from the engine is worse than no predictor
    at all.
    """

    LIVE = "LIVE"
    """The real thing: tools run, approvals are raised, records change."""

    SIMULATION = "SIMULATION"
    """Nothing outside CortexFlow is touched.

    Reads still run, because a decision made on invented data predicts
    nothing. Writes are recorded as what *would* have been called, and an
    approval is recorded as a pause that *would* have happened -- then the run
    continues, so the steps behind the gate are exercised too.
    """

    @property
    def is_simulation(self) -> bool:
        return self is ExecutionMode.SIMULATION


class StepStatus(StrEnum):
    PENDING = "PENDING"
    """Dependencies not yet satisfied."""

    DISPATCHED = "DISPATCHED"
    """Handed to the bus; a worker has not yet claimed it."""

    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    """Guard evaluated false, or an upstream branch was not taken."""

    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STEP_STATES

    @property
    def is_in_flight(self) -> bool:
        return self in {StepStatus.DISPATCHED, StepStatus.RUNNING}


_TERMINAL_STEP_STATES = frozenset(
    {
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
        StepStatus.CANCELLED,
    }
)


class StepError(BaseModel):
    """A failure recorded on a step, in the platform's own error taxonomy."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    error_class: ErrorClass = ErrorClass.UNKNOWN
    details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utcnow)

    @property
    def retryable(self) -> bool:
        return self.error_class is not ErrorClass.PERMANENT


class StepState(BaseModel):
    """Mutable execution record for one node of the DAG."""

    step_id: str
    type: StepType
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    run_id: str | None = None
    """Identifies the current attempt; stale results from a previous attempt are dropped."""

    output: dict[str, Any] | None = None
    error: StepError | None = None
    approval_id: str | None = None
    idempotency_key: str | None = None

    dispatched_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    next_attempt_at: datetime | None = None
    lease_expires_at: datetime | None = None

    confidence: float | None = None
    duration_ms: int | None = None
    cost: dict[str, Any] = Field(default_factory=dict)
    """Token/latency accounting so cost-per-workflow is answerable."""

    spawned: tuple[str, ...] = ()
    """Fan-out: the child workflows this step started.

    Recorded so the step can be read without querying, but never trusted as
    the source of truth -- a crash between creating a child and saving the
    parent would leave this list short. What actually exists is whatever the
    repository returns for this parent and step, which is why the ids are
    derived rather than random: re-running the spawn cannot duplicate a child.
    """

    def reset_for_retry(self) -> None:
        self.status = StepStatus.PENDING
        self.run_id = None
        self.started_at = None
        self.ended_at = None
        self.lease_expires_at = None


class WorkflowSummary(BaseModel):
    """Projection used by list endpoints and the dashboard."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    tenant_id: str
    definition_name: str
    definition_version: int
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    completed_steps: int
    total_steps: int
    execution_mode: ExecutionMode = ExecutionMode.LIVE
    labels: dict[str, str] = Field(default_factory=dict)


class Workflow(BaseModel):
    """The durable workflow aggregate."""

    workflow_id: str
    tenant_id: str
    definition_name: str
    definition_version: int
    status: WorkflowStatus = WorkflowStatus.PENDING

    version: int = 0
    """Optimistic-concurrency token. Incremented by the repository on every write."""

    etag: str | None = None
    """Backend-native concurrency token (Cosmos ``_etag``), when available."""

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = ""
    correlation_id: str = ""
    labels: dict[str, str] = Field(default_factory=dict)

    input: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, StepState] = Field(default_factory=dict)
    error: StepError | None = None
    cancellation_reason: str | None = None

    scheduled_wake_at: datetime | None = None
    """When a timer (retry backoff, approval timeout) should revisit this workflow."""

    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    execution_mode: ExecutionMode = ExecutionMode.LIVE
    """Whether this run may change anything outside CortexFlow.

    Recorded on the aggregate rather than passed along with each command, so
    it survives a crash: a worker that picks the run up after a restart reads
    the mode from the same durable document it reads everything else from,
    and cannot resume a simulation as though it were live.
    """

    parent_workflow_id: str | None = None
    """Set on a child spawned by a fan-out step."""

    parent_step_id: str | None = None
    """Which step of the parent spawned it."""

    fan_out_index: int | None = None
    """Position in the parent's collection, from zero."""

    fan_out_key: str = ""
    """What the item was called -- an invoice number, an employee id."""

    @property
    def is_child(self) -> bool:
        return self.parent_workflow_id is not None

    @property
    def definition_key(self) -> str:
        return f"{self.definition_name}@{self.definition_version}"

    @property
    def partition_key(self) -> str:
        """Cosmos partition key: tenant-scoped so tenants never share throughput."""
        return self.tenant_id

    def step(self, step_id: str) -> StepState:
        try:
            return self.steps[step_id]
        except KeyError as exc:
            raise NotFoundError(
                "Unknown step", workflow_id=self.workflow_id, step_id=step_id
            ) from exc

    def completed_step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps.values() if s.status is StepStatus.SUCCEEDED]

    def pending_step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps.values() if not s.status.is_terminal]

    def in_flight_step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps.values() if s.status.is_in_flight]

    def has_in_flight_work(self) -> bool:
        return any(s.status.is_in_flight for s in self.steps.values())

    def context(self) -> dict[str, Any]:
        """Evaluation scope for guards and input bindings.

        Exposes only data, never the aggregate itself, so an expression cannot
        reach into workflow internals.
        """
        return {
            "input": self.input,
            "workflow": {
                "id": self.workflow_id,
                "tenant_id": self.tenant_id,
                "type": self.definition_name,
                "status": str(self.status),
                "mode": str(self.execution_mode),
                "labels": self.labels,
            },
            "steps": {
                sid: {
                    "status": str(s.status),
                    "output": s.output or {},
                    "confidence": s.confidence,
                    "attempts": s.attempts,
                }
                for sid, s in self.steps.items()
            },
        }

    def outputs(self) -> dict[str, dict[str, Any]]:
        return {sid: (s.output or {}) for sid, s in self.steps.items() if s.output}

    def touch(self) -> Self:
        self.updated_at = utcnow()
        return self

    def to_summary(self) -> WorkflowSummary:
        return WorkflowSummary(
            workflow_id=self.workflow_id,
            tenant_id=self.tenant_id,
            definition_name=self.definition_name,
            definition_version=self.definition_version,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            created_by=self.created_by,
            completed_steps=len(self.completed_step_ids()),
            total_steps=len(self.steps),
            execution_mode=self.execution_mode,
            labels=self.labels,
        )

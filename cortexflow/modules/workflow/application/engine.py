"""The workflow engine.

This is the platform's control centre.  It reads durable state, decides what
may run, dispatches work, records outcomes, and enforces every guarantee the
architecture promises:

* **Coordination, not execution.** The engine never calls an LLM or an
  enterprise API.  It dispatches to queues and applies the results.
* **Checkpointing.** Every decision is persisted before it is acted on, so a
  crash resumes from the last durable state rather than from the beginning.
* **Optimistic concurrency.** Two orchestrators may touch the same workflow;
  the loser reloads and retries instead of overwriting.
* **Leases.** A step claimed by a worker that dies is reclaimed once its lease
  lapses, so a crash costs latency and not a stuck workflow.
* **Save-then-publish.** State is committed before any message is sent.  A
  crash between the two costs a duplicate dispatch, which idempotency keys
  absorb -- whereas publishing first could execute work that state never
  recorded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from cortexflow.config.settings import ReliabilitySettings
from cortexflow.modules.approval.domain.models import (
    Approval,
    ApprovalDecision,
    ApprovalStatus,
    RiskLevel,
)
from cortexflow.modules.approval.ports.repositories import ApprovalRepository
from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction, AuditOutcome
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.policy.domain.models import PolicyDecision, PolicyEffect
from cortexflow.modules.tool_registry.application.idempotency import build_key
from cortexflow.modules.workflow.application.definitions import WorkflowDefinitionRegistry
from cortexflow.modules.workflow.application.retries import RetryPolicy
from cortexflow.modules.workflow.domain.dag import ExecutionPlan, build_plan
from cortexflow.modules.workflow.domain.definition import (
    ApprovalSpec,
    FanOutSpec,
    Queue,
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.modules.workflow.domain.envelope import (
    AdvanceWorkflow,
    ExecuteStep,
    Message,
    MessageHeaders,
)
from cortexflow.modules.workflow.domain.models import (
    StepError,
    StepState,
    StepStatus,
    Workflow,
    WorkflowStatus,
)
from cortexflow.modules.workflow.domain.state_machine import (
    assert_step_transition,
    assert_workflow_transition,
)
from cortexflow.modules.workflow.domain.topics import route_step
from cortexflow.modules.workflow.ports.messaging import MessageBus
from cortexflow.modules.workflow.ports.repositories import WorkflowRepository
from cortexflow.shared.clock import Clock
from cortexflow.shared.errors import (
    ConcurrencyConflictError,
    ConflictError,
    CortexFlowError,
    ErrorClass,
    NotFoundError,
    ValidationError,
)
from cortexflow.shared.expressions import evaluate, render
from cortexflow.shared.identity import Role
from cortexflow.shared.ids import APPROVAL, MESSAGE, STEP_RUN, IdGenerator
from cortexflow.shared.observability.logging import bind_context, get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.redaction import summarize
from cortexflow.shared.observability.telemetry import span
from cortexflow.shared.ports.cache import LockManager

logger = get_logger(__name__)

MAX_CONCURRENCY_RETRIES = 5

MAX_REPORTED_CHILDREN = 200
"""How many child workflows are itemised in a fan-out step's output.

The counts always cover every child; only the list is capped, because the
parent is one document that people read and a database stores.
"""
UPSTREAM_FAILURE_CODE = "upstream_failure"
"""Marks a step cancelled as collateral damage, so replay can re-arm it."""

APPROVAL_CONTEXT_KEYS = (
    "amount", "currency", "employee_id", "employee_name", "vendor",
    "category", "expense_date", "campaign_id", "grade", "recommendation",
    "reason", "risk_factors", "issues",
)


@dataclass
class AdvanceResult:
    """What one orchestration pass did -- returned for logging and tests."""

    workflow_id: str
    status: WorkflowStatus
    dispatched: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    cancelled: tuple[str, ...] = ()
    awaiting_approval: tuple[str, ...] = ()
    deferred: bool = False
    """True when another orchestrator held the lease; the trigger is redelivered."""

    changed: bool = False


@dataclass
class _Pass:
    """Mutable bookkeeping for a single advance pass."""

    outbox: list[tuple[Any, Message]] = field(default_factory=list)
    dispatched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    awaiting: list[str] = field(default_factory=list)
    changed: bool = False


class WorkflowEngine:
    """Coordinates workflow execution. One instance per orchestrator process."""

    def __init__(
        self,
        *,
        workflows: WorkflowRepository,
        approvals: ApprovalRepository,
        definitions: WorkflowDefinitionRegistry,
        bus: MessageBus,
        locks: LockManager,
        policy: PolicyEngine,
        audit: AuditTrail,
        retry: RetryPolicy,
        clock: Clock,
        ids: IdGenerator,
        settings: ReliabilitySettings,
        owner: str = "orchestrator",
    ) -> None:
        self._workflows = workflows
        self._approvals = approvals
        self._definitions = definitions
        self._bus = bus
        self._locks = locks
        self._policy = policy
        self._audit = audit
        self._retry = retry
        self._clock = clock
        self._ids = ids
        self._settings = settings
        self._owner = owner
        self._metrics = get_metrics()

    # ==================================================================
    # Entry points
    # ==================================================================
    async def advance(
        self, tenant_id: str, workflow_id: str, *, reason: str = ""
    ) -> AdvanceResult:
        """Re-evaluate a workflow and dispatch everything that may now run.

        Safe to call at any time, any number of times: it derives its actions
        from durable state, so a duplicate trigger is a no-op rather than a
        double dispatch.
        """
        lease = await self._locks.acquire(
            f"workflow:{tenant_id}:{workflow_id}",
            owner=self._owner,
            ttl_seconds=self._settings.workflow_lease_seconds,
        )
        if lease is None:
            logger.debug(
                "workflow is being advanced elsewhere; deferring",
                extra={"workflow_id": workflow_id},
            )
            return AdvanceResult(
                workflow_id=workflow_id, status=WorkflowStatus.RUNNING, deferred=True
            )

        try:
            return await self._advance_with_conflict_retry(tenant_id, workflow_id, reason)
        finally:
            await lease.release()

    async def _advance_with_conflict_retry(
        self, tenant_id: str, workflow_id: str, reason: str
    ) -> AdvanceResult:
        """Reload and retry when another writer wins the version race."""
        for attempt in range(MAX_CONCURRENCY_RETRIES):
            try:
                return await self._advance_once(tenant_id, workflow_id, reason)
            except ConcurrencyConflictError:
                self._metrics.count(self._metrics.concurrency_conflicts, stage="advance")
                logger.info(
                    "concurrency conflict while advancing; reloading",
                    extra={"workflow_id": workflow_id, "attempt": attempt + 1},
                )
        raise ConcurrencyConflictError(
            "Could not advance the workflow without conflicting",
            workflow_id=workflow_id,
            attempts=MAX_CONCURRENCY_RETRIES,
        )

    async def _advance_once(
        self, tenant_id: str, workflow_id: str, reason: str
    ) -> AdvanceResult:
        workflow = await self._workflows.get(tenant_id, workflow_id)
        definition = self._definitions.get(
            workflow.definition_name,
            workflow.definition_version,
            tenant_id=workflow.tenant_id,
        )

        with (
            bind_context(
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                correlation_id=workflow.correlation_id,
            ),
            span(
                "workflow.advance",
                **{
                    "workflow.id": workflow_id,
                    "workflow.type": workflow.definition_name,
                    "workflow.status": str(workflow.status),
                    "advance.reason": reason,
                },
            ),
        ):
            if workflow.status.is_terminal:
                logger.debug("advance ignored: workflow is terminal")
                return AdvanceResult(workflow_id, workflow.status)

            state = _Pass()
            self._reclaim_expired_leases(workflow, state)
            if workflow.status is WorkflowStatus.PENDING:
                self._set_status(workflow, WorkflowStatus.RUNNING, state)

            await self._run_to_fixed_point(workflow, definition, state)
            self._schedule_next_wake(workflow, definition)
            await self._recompute_status(workflow, definition, state)

            if not state.changed:
                return AdvanceResult(workflow_id, workflow.status)

            expected_version = workflow.version
            saved = await self._workflows.save(workflow, expected_version=expected_version)

            # Published only after the state that justifies them is durable.
            await self._flush(state.outbox)

            return AdvanceResult(
                workflow_id=workflow_id,
                status=saved.status,
                dispatched=tuple(state.dispatched),
                skipped=tuple(state.skipped),
                cancelled=tuple(state.cancelled),
                awaiting_approval=tuple(state.awaiting),
                changed=True,
            )

    # ==================================================================
    # Result handling
    # ==================================================================
    async def complete_step(
        self,
        tenant_id: str,
        workflow_id: str,
        step_id: str,
        *,
        run_id: str,
        output: dict[str, Any],
        confidence: float | None = None,
        duration_ms: int = 0,
        cost: dict[str, Any] | None = None,
    ) -> AdvanceResult:
        """Record a successful step, then advance."""
        applied = await self._apply_step_result(
            tenant_id,
            workflow_id,
            step_id,
            run_id=run_id,
            mutate=lambda wf, step: self._mark_succeeded(
                wf, step, output, confidence, duration_ms, cost or {}
            ),
        )
        if not applied:
            return AdvanceResult(workflow_id, WorkflowStatus.RUNNING)
        return await self.advance(tenant_id, workflow_id, reason=f"{step_id} succeeded")

    async def fail_step(
        self,
        tenant_id: str,
        workflow_id: str,
        step_id: str,
        *,
        run_id: str,
        error: StepError,
        duration_ms: int = 0,
    ) -> AdvanceResult:
        """Record a failed step, scheduling a retry when the policy allows one."""
        retry_message: list[Message] = []

        def mutate(workflow: Workflow, step: StepState) -> None:
            definition = self._definitions.get(
                workflow.definition_name,
                workflow.definition_version,
                tenant_id=workflow.tenant_id,
            )
            spec = definition.step(step_id)
            decision = self._retry.decide(
                spec=spec.retry,
                error_class=error.error_class,
                attempt=step.attempts,
                idempotent=spec.idempotent,
            )
            now = self._clock.now()
            step.error = error
            step.duration_ms = duration_ms
            step.lease_expires_at = None

            if decision.should_retry:
                # Stay PENDING with a future window: the sweeper will pick it
                # up, so no process holds a timer.
                step.status = StepStatus.PENDING
                step.run_id = None
                step.next_attempt_at = now + timedelta(seconds=decision.delay_seconds)
                workflow.scheduled_wake_at = _earliest(
                    workflow.scheduled_wake_at, step.next_attempt_at
                )
                self._metrics.count(
                    self._metrics.step_retries,
                    step=step_id,
                    error_class=str(error.error_class),
                )
                retry_message.append(
                    self._advance_message(
                        workflow, reason=f"retry {step_id} attempt {step.attempts + 1}"
                    )
                )
            else:
                assert_step_transition(
                    step.status, StepStatus.FAILED,
                    workflow_id=workflow_id, step_id=step_id,
                )
                step.status = StepStatus.FAILED
                step.ended_at = now
                self._metrics.count(
                    self._metrics.step_failures,
                    step=step_id,
                    error_class=str(error.error_class),
                )

        applied = await self._apply_step_result(
            tenant_id, workflow_id, step_id, run_id=run_id, mutate=mutate,
            audit_action=AuditAction.STEP_FAILED,
            audit_outcome=AuditOutcome.FAILURE,
            audit_summary=error.message,
        )
        if not applied:
            return AdvanceResult(workflow_id, WorkflowStatus.RUNNING)

        if retry_message:
            # Backoff is expressed as a *scheduled message*, not a sleep: no
            # process is held open waiting for a retry window to open.
            workflow = await self._workflows.get(tenant_id, workflow_id)
            next_attempt_at = workflow.step(step_id).next_attempt_at
            delay = (
                max((next_attempt_at - self._clock.now()).total_seconds(), 0.0)
                if next_attempt_at
                else 0.0
            )
            await self._bus.schedule(Queue.CONTROL, retry_message[0], delay_seconds=delay)
            await self._audit.record_for(
                workflow, AuditAction.STEP_RETRY_SCHEDULED,
                step_id=step_id,
                summary=f"Retry scheduled in {delay:.1f}s",
                metadata={"attempt": workflow.step(step_id).attempts, "delay_seconds": delay},
            )
            return AdvanceResult(workflow_id, WorkflowStatus.RUNNING)

        return await self.advance(tenant_id, workflow_id, reason=f"{step_id} failed")

    async def _apply_step_result(
        self,
        tenant_id: str,
        workflow_id: str,
        step_id: str,
        *,
        run_id: str,
        mutate: Any,
        audit_action: AuditAction = AuditAction.STEP_SUCCEEDED,
        audit_outcome: AuditOutcome = AuditOutcome.SUCCESS,
        audit_summary: str = "",
    ) -> bool:
        """Apply a worker's result under optimistic concurrency.

        Returns False when the result is stale -- a late reply from a previous
        attempt, or a duplicate for a step that already settled.  Dropping it
        is what makes at-least-once delivery safe.
        """
        for _ in range(MAX_CONCURRENCY_RETRIES):
            workflow = await self._workflows.get(tenant_id, workflow_id)
            try:
                step = workflow.step(step_id)
            except NotFoundError:
                logger.warning("result for unknown step ignored", extra={"step_id": step_id})
                return False

            if step.run_id != run_id:
                logger.info(
                    "discarding stale step result",
                    extra={
                        "step_id": step_id,
                        "expected_run_id": step.run_id,
                        "received_run_id": run_id,
                    },
                )
                return False
            if step.status.is_terminal:
                logger.info("step already settled; ignoring duplicate result",
                            extra={"step_id": step_id, "status": str(step.status)})
                return False

            mutate(workflow, step)
            try:
                await self._workflows.save(workflow, expected_version=workflow.version)
            except ConcurrencyConflictError:
                self._metrics.count(self._metrics.concurrency_conflicts, stage="step_result")
                continue

            await self._audit.record_for(
                workflow,
                audit_action,
                outcome=audit_outcome,
                step_id=step_id,
                run_id=run_id,
                summary=audit_summary,
                metadata={"status": str(step.status), "attempts": step.attempts},
            )
            await self._record_agent_proposal(workflow, step)
            return True

        raise ConcurrencyConflictError(
            "Could not record the step result", workflow_id=workflow_id, step_id=step_id
        )

    async def _record_agent_proposal(self, workflow: Workflow, step: StepState) -> None:
        """What the model advised, separately from the step having finished.

        ``STEP_SUCCEEDED`` says an agent step ran. It does not say what the
        agent *proposed*, how sure it was, or which model answered -- and
        those are the three things an audit of an AI decision is for. They
        were recorded on the step and nowhere in the trail, so "which model
        approved this claim?" could only be answered by reading workflow
        state, and only while that state was retained.

        Deliberately a second event rather than more metadata on the first:
        the recommendation is a different kind of claim from the step's
        outcome, and ADR 0001 turns on keeping advice and authority apart.
        A step that recommends nothing produces no event.
        """
        if step.type is not StepType.AGENT or step.status is not StepStatus.SUCCEEDED:
            return

        output = step.output or {}
        recommendation = output.get("recommendation")
        if not recommendation:
            return

        cost = step.cost or {}
        await self._audit.record_for(
            workflow,
            AuditAction.AGENT_PROPOSED,
            step_id=step.step_id,
            summary=str(output.get("reason", ""))[:200],
            metadata={
                "recommendation": str(recommendation),
                "confidence": step.confidence,
                "model": cost.get("model", ""),
                "prompt_tokens": cost.get("prompt_tokens"),
                "completion_tokens": cost.get("completion_tokens"),
            },
        )

    def _mark_succeeded(
        self,
        workflow: Workflow,
        step: StepState,
        output: dict[str, Any],
        confidence: float | None,
        duration_ms: int,
        cost: dict[str, Any],
    ) -> None:
        assert_step_transition(
            step.status, StepStatus.SUCCEEDED,
            workflow_id=workflow.workflow_id, step_id=step.step_id,
        )
        step.status = StepStatus.SUCCEEDED
        step.output = output
        step.confidence = confidence
        step.duration_ms = duration_ms
        step.cost = cost
        step.error = None
        step.ended_at = self._clock.now()
        step.lease_expires_at = None
        step.next_attempt_at = None
        self._metrics.record_duration(
            self._metrics.step_duration, duration_ms,
            step=step.step_id, workflow_type=workflow.definition_name,
        )

    # ==================================================================
    # Approvals
    # ==================================================================
    async def resolve_approval(
        self,
        tenant_id: str,
        workflow_id: str,
        step_id: str,
        *,
        approval_id: str,
        decision: ApprovalDecision,
        decided_by: str,
        decided_by_roles: tuple[str, ...] = (),
        comment: str = "",
    ) -> AdvanceResult:
        """Apply a human decision and resume the workflow.

        ``decided_by`` already carries its namespace -- ``user::priya.nair``
        for a person, ``system::timeout`` for an expiry -- and is written to
        the trail unchanged.
        """
        for _ in range(MAX_CONCURRENCY_RETRIES):
            workflow = await self._workflows.get(tenant_id, workflow_id)
            step = workflow.step(step_id)
            if step.status is not StepStatus.WAITING_FOR_APPROVAL:
                logger.info(
                    "approval resolution ignored; step is no longer waiting",
                    extra={"step_id": step_id, "status": str(step.status)},
                )
                return AdvanceResult(workflow_id, workflow.status)

            now = self._clock.now()
            if decision is ApprovalDecision.APPROVE:
                step.status = StepStatus.SUCCEEDED
                step.output = {
                    "decision": str(decision),
                    "decided_by": decided_by,
                    "comment": comment,
                    "approval_id": approval_id,
                }
                step.ended_at = now
            elif decision is ApprovalDecision.REJECT:
                step.status = StepStatus.SUCCEEDED
                step.output = {
                    "decision": str(decision),
                    "decided_by": decided_by,
                    "comment": comment,
                    "approval_id": approval_id,
                }
                step.ended_at = now
                # A rejection is a business outcome, not a failure: the
                # workflow ends REJECTED with a named decider and a reason.
                self._set_status(workflow, WorkflowStatus.REJECTED, _Pass())
                workflow.cancellation_reason = comment or "Rejected by approver"
                self._cancel_outstanding(workflow, reason="Approval rejected")
            else:  # REQUEST_INFO leaves the step waiting for a further decision
                logger.info("approver requested more information",
                            extra={"step_id": step_id, "approval_id": approval_id})

            if step.dispatched_at is not None:
                self._metrics.record_duration(
                    self._metrics.approval_wait,
                    (now - step.dispatched_at).total_seconds() * 1000,
                    workflow_type=workflow.definition_name,
                )

            try:
                await self._workflows.save(workflow, expected_version=workflow.version)
            except ConcurrencyConflictError:
                self._metrics.count(self._metrics.concurrency_conflicts, stage="approval")
                continue

            await self._audit.record_for(
                workflow,
                _APPROVAL_AUDIT[decision],
                # Not f"user::{decided_by}": the subject is already
                # namespaced, so prefixing it again filed every approval in
                # the trail under "user::user::priya.nair" -- a principal that
                # does not exist -- and attributed a timeout expiry to a user.
                actor=decided_by,
                actor_roles=decided_by_roles,
                step_id=step_id,
                summary=comment,
                metadata={"approval_id": approval_id, "decision": str(decision)},
            )
            if decision is ApprovalDecision.REQUEST_INFO:
                return AdvanceResult(workflow_id, workflow.status)
            return await self.advance(
                tenant_id, workflow_id, reason=f"approval {decision} on {step_id}"
            )

        raise ConcurrencyConflictError(
            "Could not apply the approval decision",
            workflow_id=workflow_id, approval_id=approval_id,
        )

    async def expire_approval(self, approval: Approval) -> None:
        """Apply the definition's timeout policy to an unanswered approval."""
        action = approval.on_timeout
        await self._audit.record(
            tenant_id=approval.tenant_id,
            action=AuditAction.APPROVAL_EXPIRED,
            workflow_id=approval.workflow_id,
            step_id=approval.step_id,
            summary=f"Approval timed out; applying '{action}'",
            metadata={"approval_id": approval.approval_id},
        )

        approval.status = ApprovalStatus.EXPIRED
        await self._approvals.save(approval, expected_version=approval.version)

        if action == "approve":
            await self.resolve_approval(
                approval.tenant_id, approval.workflow_id, approval.step_id,
                approval_id=approval.approval_id,
                decision=ApprovalDecision.APPROVE,
                decided_by="system::timeout",
                comment="Auto-approved on timeout by workflow policy",
            )
        elif action == "reject":
            await self.resolve_approval(
                approval.tenant_id, approval.workflow_id, approval.step_id,
                approval_id=approval.approval_id,
                decision=ApprovalDecision.REJECT,
                decided_by="system::timeout",
                comment="Auto-rejected on timeout by workflow policy",
            )
        else:
            # Escalate: keep the workflow suspended and raise a new approval
            # with a longer window, so nothing is silently decided by inaction.
            await self._reissue_approval(approval)

    async def _reissue_approval(self, approval: Approval) -> None:
        workflow = await self._workflows.get(approval.tenant_id, approval.workflow_id)
        definition = self._definitions.get(
            workflow.definition_name,
            workflow.definition_version,
            tenant_id=workflow.tenant_id,
        )
        spec = definition.step(approval.step_id).approval or ApprovalSpec()
        escalated = await self._create_approval(
            workflow,
            approval.step_id,
            spec,
            risk=RiskLevel.HIGH,
            proposed_action=approval.proposed_action,
            title=f"[Escalated] {approval.title}",
        )
        step = workflow.step(approval.step_id)
        step.approval_id = escalated.approval_id
        workflow.scheduled_wake_at = escalated.expires_at
        await self._workflows.save(workflow, expected_version=workflow.version)

    # ==================================================================
    # Recovery
    # ==================================================================
    def _reclaim_expired_leases(self, workflow: Workflow, state: _Pass) -> None:
        """Return steps abandoned by crashed workers to the runnable pool.

        This is the crash-recovery primitive: the worker is gone, its lease has
        lapsed, and the step becomes eligible again on the next pass.
        """
        now = self._clock.now()
        for step in workflow.steps.values():
            if not step.status.is_in_flight or step.lease_expires_at is None:
                continue
            if step.lease_expires_at > now:
                continue
            logger.warning(
                "reclaiming a step from an expired lease",
                extra={"step_id": step.step_id, "attempts": step.attempts},
            )
            step.reset_for_retry()
            state.changed = True
            self._metrics.count(self._metrics.leases_reclaimed, step=step.step_id)

    async def replay(self, tenant_id: str, workflow_id: str, *, actor: str) -> AdvanceResult:
        """Operator action: re-arm failed steps and resume a parked workflow."""
        workflow = await self._workflows.get(tenant_id, workflow_id)
        if workflow.status not in {
            WorkflowStatus.FAILED,
            WorkflowStatus.WAITING_FOR_RECOVERY,
        }:
            from cortexflow.shared.errors import ConflictError

            raise ConflictError(
                "Only failed or parked workflows can be replayed",
                workflow_id=workflow_id, status=str(workflow.status),
            )

        # Re-arm the failed steps *and* everything cancelled as a consequence
        # of them. A step cancelled by upstream failure is collateral, not an
        # operator decision, so leaving it cancelled would make recovery
        # impossible -- the workflow would resume and immediately stall again.
        rearmed: list[str] = []
        for step in workflow.steps.values():
            collateral = (
                step.status is StepStatus.CANCELLED
                and step.error is not None
                and step.error.code == UPSTREAM_FAILURE_CODE
            )
            if step.status is StepStatus.FAILED or collateral:
                step.reset_for_retry()
                step.attempts = 0 if collateral else step.attempts
                step.next_attempt_at = None
                step.error = None
                rearmed.append(step.step_id)
        workflow.error = None
        workflow.cancellation_reason = None
        assert_workflow_transition(
            workflow.status, WorkflowStatus.RUNNING, workflow_id=workflow_id
        )
        workflow.status = WorkflowStatus.RUNNING
        await self._workflows.save(workflow, expected_version=workflow.version)
        await self._audit.record_for(
            workflow, AuditAction.WORKFLOW_REPLAYED, actor=f"user::{actor}",
            summary="Workflow replayed by operator",
            metadata={"rearmed_steps": sorted(rearmed)},
        )
        return await self.advance(tenant_id, workflow_id, reason="operator replay")

    async def cancel(
        self, tenant_id: str, workflow_id: str, *, actor: str, reason: str
    ) -> Workflow:
        """Cancel a workflow and every approval still waiting on it."""
        for _ in range(MAX_CONCURRENCY_RETRIES):
            workflow = await self._workflows.get(tenant_id, workflow_id)
            if workflow.status.is_terminal:
                return workflow
            assert_workflow_transition(
                workflow.status, WorkflowStatus.CANCELLED, workflow_id=workflow_id
            )
            workflow.status = WorkflowStatus.CANCELLED
            workflow.cancellation_reason = reason
            workflow.scheduled_wake_at = None
            self._cancel_outstanding(workflow, reason=reason)
            try:
                saved = await self._workflows.save(
                    workflow, expected_version=workflow.version
                )
            except ConcurrencyConflictError:
                continue
            await self._approvals.cancel_for_workflow(tenant_id, workflow_id, reason=reason)
            await self._audit.record_for(
                saved, AuditAction.WORKFLOW_CANCELLED, actor=f"user::{actor}", summary=reason
            )
            return saved
        raise ConcurrencyConflictError("Could not cancel the workflow",
                                       workflow_id=workflow_id)

    @staticmethod
    def _cancel_outstanding(workflow: Workflow, *, reason: str) -> None:
        for step in workflow.steps.values():
            if not step.status.is_terminal:
                step.status = StepStatus.CANCELLED
                step.ended_at = workflow.updated_at
                step.error = step.error or StepError(
                    code="cancelled", message=reason, error_class=ErrorClass.PERMANENT
                )

    # ==================================================================
    # Plan application
    # ==================================================================
    async def _run_to_fixed_point(
        self, workflow: Workflow, definition: WorkflowDefinition, state: _Pass
    ) -> None:
        """Apply the plan repeatedly until nothing further can happen now.

        Control-plane steps (policy evaluation, skip and cancel propagation)
        resolve *within* a pass, which can unblock steps that were not runnable
        when the pass began -- a policy decision immediately determines which
        branch is live.  Iterating to a fixed point means one trigger fully
        settles the graph, instead of the workflow stalling until some
        unrelated message happens to wake it.

        Bounded by the step count: each iteration must terminate at least one
        step, so the graph being acyclic guarantees termination.
        """
        for _ in range(len(definition.steps) + 1):
            if workflow.status.is_terminal:
                return
            before = _progress_signature(workflow)

            # Children settle on their own schedule, so a waiting fan-out step
            # is collected before planning: finishing it can unblock the rest
            # of the graph within this same pass.
            await self._collect_fan_out(workflow, definition, state)

            plan = build_plan(definition, workflow, now=self._clock.now())
            await self._apply_terminations(workflow, plan, state)
            await self._process_runnable(workflow, definition, plan, state)

            if _progress_signature(workflow) == before:
                return

        logger.warning(
            "workflow did not settle within its step budget",
            extra={"workflow_id": workflow.workflow_id},
        )

    async def _apply_terminations(
        self, workflow: Workflow, plan: ExecutionPlan, state: _Pass
    ) -> None:
        now = self._clock.now()
        for decision in plan.to_skip:
            step = workflow.step(decision.step_id)
            step.status = StepStatus.SKIPPED
            step.ended_at = now
            state.skipped.append(decision.step_id)
            state.changed = True
            await self._audit.record_for(
                workflow, AuditAction.STEP_SKIPPED,
                step_id=decision.step_id, summary=decision.reason,
            )
        for decision in plan.to_cancel:
            step = workflow.step(decision.step_id)
            step.status = StepStatus.CANCELLED
            step.ended_at = now
            step.error = StepError(
                code=UPSTREAM_FAILURE_CODE,
                message=decision.reason,
                error_class=ErrorClass.PERMANENT,
            )
            state.cancelled.append(decision.step_id)
            state.changed = True

    async def _process_runnable(
        self,
        workflow: Workflow,
        definition: WorkflowDefinition,
        plan: ExecutionPlan,
        state: _Pass,
    ) -> None:
        """Handle every step that may run now.

        Control-plane steps (policy, approval) resolve inline because they are
        deterministic and fast.  Agent and tool steps go to the bus, where
        independent steps in the same wave are dispatched together and execute
        concurrently on separate workers.
        """
        for spec in plan.runnable:
            match spec.type:
                case StepType.POLICY:
                    await self._evaluate_policy_step(workflow, spec, state)
                case StepType.HUMAN_APPROVAL:
                    await self._request_approval(workflow, spec, state)
                case StepType.FAN_OUT:
                    await self._start_fan_out(workflow, spec, state)
                case _:
                    await self._dispatch(workflow, definition, spec, state)

    async def _dispatch(
        self,
        workflow: Workflow,
        definition: WorkflowDefinition,
        spec: StepDefinition,
        state: _Pass,
    ) -> None:
        step = workflow.step(spec.id)
        now = self._clock.now()

        step.attempts += 1
        step.run_id = self._ids.new_id(STEP_RUN)
        step.status = StepStatus.DISPATCHED
        step.dispatched_at = now
        step.next_attempt_at = None
        # The lease is what lets a crashed worker's step be reclaimed. It is
        # generous relative to the step timeout so slow work is not stolen.
        step.lease_expires_at = now + timedelta(
            seconds=max(spec.timeout_seconds, self._settings.task_visibility_seconds)
        )
        step.idempotency_key = build_key(
            workflow_id=workflow.workflow_id,
            step_id=spec.id,
            tool=spec.tool or spec.agent or spec.id,
            arguments={"attempt_scope": workflow.workflow_id},
        )

        message = Message(
            headers=MessageHeaders(
                message_id=self._ids.new_id(MESSAGE),
                tenant_id=workflow.tenant_id,
                correlation_id=workflow.correlation_id,
                dedup_key=f"exec:{workflow.workflow_id}:{spec.id}:{step.run_id}",
            ),
            body=ExecuteStep(
                workflow_id=workflow.workflow_id,
                step_id=spec.id,
                step_type=spec.type,
                run_id=step.run_id,
                attempt=step.attempts,
                idempotency_key=step.idempotency_key,
                deadline=now + timedelta(seconds=spec.timeout_seconds),
            ),
        )
        state.outbox.append((route_step(spec), message))
        state.dispatched.append(spec.id)
        state.changed = True

        await self._audit.record_for(
            workflow, AuditAction.STEP_DISPATCHED,
            step_id=spec.id, run_id=step.run_id,
            summary=f"Dispatched to the {route_step(spec).value} queue",
            metadata={"attempt": step.attempts, "type": str(spec.type)},
        )

    async def _evaluate_policy_step(
        self, workflow: Workflow, spec: StepDefinition, state: _Pass
    ) -> None:
        """Evaluate a policy ruleset inline.

        Deterministic and in-process: there is no benefit to a round trip
        through a queue, and keeping it here means the authorization decision
        is made by the orchestrator itself.
        """
        step = workflow.step(spec.id)
        step.attempts += 1
        facts = _resolve_inputs(spec, workflow).get("facts", {})

        try:
            # The tenant decides which ruleset applies: an authored one
            # shadows the shipped one for that tenant only.
            decision = self._policy.evaluate(
                str(spec.ruleset), {"facts": facts}, tenant_id=workflow.tenant_id
            )
        except CortexFlowError as exc:
            step.status = StepStatus.FAILED
            step.error = StepError(
                code=exc.code, message=exc.message, error_class=exc.error_class
            )
            step.ended_at = self._clock.now()
            state.changed = True
            return

        step.status = StepStatus.SUCCEEDED
        step.output = {
            "effect": str(decision.effect),
            "risk": str(decision.risk),
            "required_roles": [str(r) for r in decision.required_roles],
            "reasons": list(decision.reasons),
            "matched_rules": [m.rule_id for m in decision.matched_rules],
            "ruleset": decision.ruleset,
            "inputs_digest": decision.inputs_digest,
        }
        step.ended_at = self._clock.now()
        state.changed = True

        await self._audit.record_for(
            workflow,
            AuditAction.POLICY_DENIED
            if decision.denied
            else AuditAction.POLICY_EVALUATED,
            outcome=AuditOutcome.DENIED if decision.denied else AuditOutcome.SUCCESS,
            step_id=spec.id,
            summary="; ".join(decision.reasons)[:500],
            metadata={
                "effect": str(decision.effect),
                "risk": str(decision.risk),
                "ruleset": decision.ruleset,
                "matched_rules": [m.rule_id for m in decision.matched_rules],
                "inputs_digest": decision.inputs_digest,
                "facts": summarize(facts),
            },
        )

        if decision.denied:
            # A denial is an outcome, not an error: end the workflow REJECTED
            # with the reasons attached, and stop everything else.
            self._set_status(workflow, WorkflowStatus.REJECTED, state)
            workflow.cancellation_reason = "; ".join(decision.reasons)[:500]
            self._cancel_outstanding(workflow, reason="Denied by policy")

    async def _request_approval(
        self, workflow: Workflow, spec: StepDefinition, state: _Pass
    ) -> None:
        """Suspend the workflow on a durable approval record.

        No process blocks here. The workflow is checkpointed as
        WAITING_FOR_APPROVAL and will not be touched again until a human acts
        or the timeout sweeper fires.
        """
        step = workflow.step(spec.id)
        approval_spec = spec.approval or ApprovalSpec()
        inputs = _resolve_inputs(spec, workflow)
        policy_decision = _upstream_policy_decision(workflow, spec)

        risk = RiskLevel.MEDIUM
        roles = approval_spec.required_roles
        if policy_decision is not None:
            risk = policy_decision.risk
            roles = roles or policy_decision.required_roles

        if workflow.execution_mode.is_simulation:
            await self._record_simulated_approval(
                workflow, spec, approval_spec, risk=risk, roles=roles, state=state
            )
            return

        approval = await self._create_approval(
            workflow,
            spec.id,
            approval_spec.model_copy(update={"required_roles": roles}),
            risk=risk,
            proposed_action=inputs.get("proposed_action", {}),
            context=inputs.get("context", {}),
        )

        step.status = StepStatus.WAITING_FOR_APPROVAL
        step.approval_id = approval.approval_id
        step.dispatched_at = self._clock.now()
        state.awaiting.append(spec.id)
        state.changed = True
        workflow.scheduled_wake_at = _earliest(
            workflow.scheduled_wake_at, approval.expires_at
        )

    async def _record_simulated_approval(
        self,
        workflow: Workflow,
        spec: StepDefinition,
        approval_spec: ApprovalSpec,
        *,
        risk: RiskLevel,
        roles: tuple[Role, ...],
        state: _Pass,
    ) -> None:
        """Note the pause a live run would take here, and keep going.

        Two things could happen instead, and both are worse. Raising a real
        approval would put a rehearsal in a manager's queue. Stopping the run
        would mean the simulation never reaches the steps behind the gate --
        so the one question worth asking, "what does this workflow do to my
        systems once it is approved", would go unanswered.

        Continuing is not a prediction that the approval would be granted. It
        is a statement of what is downstream of it, and the step says so.
        """
        step = workflow.step(spec.id)
        step.status = StepStatus.SUCCEEDED
        step.ended_at = self._clock.now()
        step.output = {
            "simulated": True,
            "would_require_approval": True,
            "decision": "NOT_ASKED",
            "title": approval_spec.title,
            "risk": str(risk),
            "required_roles": [str(r) for r in roles],
            "note": (
                "Simulation: no approval was raised. A live run would pause "
                f"here for {', '.join(str(r) for r in roles) or 'an approver'}. "
                "The steps after this one ran so their effect could be seen."
            ),
        }
        state.changed = True

        await self._audit.record_for(
            workflow,
            AuditAction.APPROVAL_REQUESTED,
            outcome=AuditOutcome.PENDING,
            step_id=spec.id,
            summary=f"Simulated: would have requested '{approval_spec.title}'",
            metadata={
                "simulated": True,
                "risk": str(risk),
                "required_roles": [str(r) for r in roles],
            },
        )

    async def _create_approval(
        self,
        workflow: Workflow,
        step_id: str,
        spec: ApprovalSpec,
        *,
        risk: RiskLevel,
        proposed_action: dict[str, Any],
        context: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Approval:
        timeout = spec.timeout_seconds or self._settings.approval_timeout_seconds
        approval = Approval(
            approval_id=self._ids.new_id(APPROVAL),
            workflow_id=workflow.workflow_id,
            step_id=step_id,
            tenant_id=workflow.tenant_id,
            title=title or spec.title,
            description=spec.description,
            risk=risk,
            required_roles=spec.required_roles,
            # Approvers see a redacted projection, never the raw documents.
            context=summarize(context or workflow.input, keys=APPROVAL_CONTEXT_KEYS),
            proposed_action=summarize(proposed_action, keys=APPROVAL_CONTEXT_KEYS),
            expires_at=self._clock.now() + timedelta(seconds=timeout),
            on_timeout=spec.on_timeout,
        )
        created = await self._approvals.create(approval)
        await self._audit.record_for(
            workflow, AuditAction.APPROVAL_REQUESTED,
            outcome=AuditOutcome.PENDING,
            step_id=step_id,
            summary=created.title,
            metadata={
                "approval_id": created.approval_id,
                "risk": str(risk),
                "required_roles": [str(r) for r in created.required_roles],
            },
        )
        return created

    # ==================================================================
    # Fan-out
    # ==================================================================
    async def _start_fan_out(
        self, workflow: Workflow, spec: StepDefinition, state: _Pass
    ) -> None:
        """Start one child workflow per item, then wait.

        The step does not succeed here: it goes RUNNING and stays there until
        every child has settled. That is what makes the parent's own status
        honest -- a batch is not finished because it was dispatched.
        """
        fan_out = spec.fan_out
        assert fan_out is not None  # guaranteed by StepDefinition validation
        step = workflow.step(spec.id)
        step.attempts += 1

        try:
            items = self._fan_out_items(workflow, fan_out)
        except CortexFlowError as exc:
            step.status = StepStatus.FAILED
            step.error = StepError(
                code=exc.code, message=exc.message, error_class=exc.error_class
            )
            step.ended_at = self._clock.now()
            state.changed = True
            return

        # Whatever already exists wins. A retry after a crash mid-spawn must
        # adopt the children it already made rather than make them twice.
        existing = {
            child.fan_out_index: child
            for child in await self._workflows.list_children(
                workflow.tenant_id, workflow.workflow_id, spec.id
            )
        }

        spawned: list[str] = []
        for index, item in enumerate(items):
            child = existing.get(index)
            if child is not None:
                spawned.append(child.workflow_id)
                continue
            child_id = await self._spawn_child(workflow, spec, fan_out, index, item, state)
            spawned.append(child_id)

        step.spawned = tuple(spawned)
        step.status = StepStatus.RUNNING
        step.started_at = step.started_at or self._clock.now()
        state.changed = True

        await self._audit.record_for(
            workflow,
            AuditAction.STEP_DISPATCHED,
            step_id=spec.id,
            summary=f"Started {len(spawned)} child workflows of {fan_out.workflow}",
            metadata={"children": len(spawned), "definition": fan_out.workflow},
        )

    def _fan_out_items(self, workflow: Workflow, fan_out: FanOutSpec) -> list[Any]:
        """Resolve the collection, and refuse anything that is not a bounded list."""
        resolved = evaluate(fan_out.collection, workflow.context())
        if not isinstance(resolved, list):
            raise ValidationError(
                "A fan-out collection must resolve to a list",
                collection=fan_out.collection,
                resolved_type=type(resolved).__name__,
            )
        if not resolved:
            raise ValidationError(
                "The fan-out collection is empty", collection=fan_out.collection
            )
        if len(resolved) > fan_out.max_children:
            raise ValidationError(
                "The fan-out collection is larger than this step allows",
                collection=fan_out.collection,
                items=len(resolved),
                max_children=fan_out.max_children,
            )
        return resolved

    async def _spawn_child(
        self,
        workflow: Workflow,
        spec: StepDefinition,
        fan_out: FanOutSpec,
        index: int,
        item: Any,
        state: _Pass,
    ) -> str:
        """Create one child, or adopt the one a previous attempt created.

        The id is derived from the parent, the step and the position rather
        than generated, so this is safe to run again: the second attempt
        collides with its own first and keeps what is already there.
        """
        child_id = f"{workflow.workflow_id}:{spec.id}:{index:04d}"
        definition = self._definitions.get(
            fan_out.workflow, None, tenant_id=workflow.tenant_id
        )
        key = ""
        if fan_out.item_key and isinstance(item, dict):
            key = str(item.get(fan_out.item_key, ""))

        child = Workflow(
            workflow_id=child_id,
            tenant_id=workflow.tenant_id,
            definition_name=definition.name,
            definition_version=definition.version,
            status=WorkflowStatus.PENDING,
            created_by=workflow.created_by,
            # Shared with the parent, so one trace covers the whole batch.
            correlation_id=workflow.correlation_id,
            # Inherited, never defaulted: a simulated batch that spawned live
            # cases would be the one bug this whole feature exists to prevent.
            execution_mode=workflow.execution_mode,
            labels={**workflow.labels, "fan_out_parent": workflow.workflow_id},
            input=_child_input(workflow, fan_out, item),
            steps={
                step.id: StepState(step_id=step.id, type=step.type)
                for step in definition.steps
            },
            parent_workflow_id=workflow.workflow_id,
            parent_step_id=spec.id,
            fan_out_index=index,
            fan_out_key=key,
        )

        try:
            await self._workflows.create(child)
        except ConflictError:
            # Already created by an attempt that crashed before saving the
            # parent. Adopting it is the whole point of the derived id.
            logger.debug("fan-out child already exists", extra={"child_id": child_id})

        state.outbox.append(
            (
                Queue.CONTROL,
                Message(
                    headers=MessageHeaders(
                        message_id=self._ids.new_id(MESSAGE),
                        tenant_id=workflow.tenant_id,
                        correlation_id=workflow.correlation_id,
                        dedup_key=f"fanout:{child_id}",
                    ),
                    body=AdvanceWorkflow(
                        workflow_id=child_id, reason="fan-out child created"
                    ),
                ),
            )
        )
        return child_id

    async def _collect_fan_out(
        self, workflow: Workflow, definition: WorkflowDefinition, state: _Pass
    ) -> None:
        """Finish any fan-out step whose children have all settled."""
        for spec in definition.steps:
            if spec.type is not StepType.FAN_OUT:
                continue
            step = workflow.step(spec.id)
            if step.status is not StepStatus.RUNNING:
                continue

            assert spec.fan_out is not None
            children = await self._workflows.list_children(
                workflow.tenant_id, workflow.workflow_id, spec.id
            )
            if not children or not _children_settled(children, spec.fan_out):
                continue

            step.output = _summarise_children(children)
            step.ended_at = self._clock.now()
            step.duration_ms = (
                int((step.ended_at - step.started_at).total_seconds() * 1000)
                if step.started_at
                else None
            )

            failed = int(step.output["failed"])
            if failed and not spec.fan_out.tolerate_failures:
                step.status = StepStatus.FAILED
                step.error = StepError(
                    code="FAN_OUT_CHILD_FAILED",
                    message=f"{failed} of {len(children)} child workflows failed",
                    error_class=ErrorClass.PERMANENT,
                )
            else:
                step.status = StepStatus.SUCCEEDED
            state.changed = True

            await self._audit.record_for(
                workflow,
                AuditAction.STEP_SUCCEEDED
                if step.status is StepStatus.SUCCEEDED
                else AuditAction.STEP_FAILED,
                step_id=spec.id,
                summary=(
                    f"{step.output['completed']} completed, "
                    f"{step.output['rejected']} rejected, "
                    f"{step.output['awaiting_approval']} awaiting a person, "
                    f"{failed} failed"
                ),
                metadata={k: v for k, v in step.output.items() if k != "children"},
            )

    def _notify_parent(self, workflow: Workflow, state: _Pass) -> None:
        """Tell the parent that this child has settled.

        Belt and braces: the message is the fast path, and the parent's own
        timer is the fallback, because a batch that silently never finishes is
        far worse than one that finishes a sweep late.
        """
        if workflow.parent_workflow_id is None:
            return
        state.outbox.append(
            (
                Queue.CONTROL,
                Message(
                    headers=MessageHeaders(
                        message_id=self._ids.new_id(MESSAGE),
                        tenant_id=workflow.tenant_id,
                        correlation_id=workflow.correlation_id,
                        dedup_key=f"fanin:{workflow.workflow_id}:{workflow.status}",
                    ),
                    body=AdvanceWorkflow(
                        workflow_id=workflow.parent_workflow_id,
                        reason=f"child {workflow.workflow_id} settled",
                    ),
                ),
            )
        )

    # ==================================================================
    # Status
    # ==================================================================
    async def _recompute_status(
        self, workflow: Workflow, definition: WorkflowDefinition, state: _Pass
    ) -> None:
        """Derive the workflow's status from its steps."""
        if workflow.status.is_terminal:
            return

        steps = workflow.steps.values()
        if any(s.status is StepStatus.WAITING_FOR_APPROVAL for s in steps):
            self._set_status(workflow, WorkflowStatus.WAITING_FOR_APPROVAL, state)
            return

        failures = [
            s for s in steps
            if s.status is StepStatus.FAILED and definition.step(s.step_id).critical
        ]
        if failures:
            # Permanent failures are terminal; exhausted transient retries are
            # parked for recovery so an operator can fix and replay.
            permanent = any(
                f.error is not None and f.error.error_class is ErrorClass.PERMANENT
                for f in failures
            )
            target = WorkflowStatus.FAILED if permanent else WorkflowStatus.WAITING_FOR_RECOVERY
            workflow.error = failures[0].error
            self._set_status(workflow, target, state)
            return

        if all(s.status.is_terminal for s in steps):
            # A workflow whose critical work was cancelled did not complete,
            # even though every step has settled.
            cancelled = [
                s.step_id
                for s in steps
                if s.status is StepStatus.CANCELLED
                and definition.step(s.step_id).critical
            ]
            if cancelled:
                workflow.cancellation_reason = (
                    workflow.cancellation_reason
                    or f"Critical steps were cancelled: {sorted(cancelled)}"
                )
                self._set_status(workflow, WorkflowStatus.WAITING_FOR_RECOVERY, state)
                return
            self._set_status(workflow, WorkflowStatus.COMPLETED, state)
            self._metrics.record_duration(
                self._metrics.workflow_duration,
                (self._clock.now() - workflow.created_at).total_seconds() * 1000,
                workflow_type=workflow.definition_name,
            )
            return

        plan = build_plan(definition, workflow, now=self._clock.now())
        if plan.is_stalled and not workflow.has_in_flight_work():
            # Nothing runnable and nothing pending that could unblock: park it
            # rather than leaving a workflow that looks alive but is not.
            self._set_status(workflow, WorkflowStatus.WAITING_FOR_RECOVERY, state)
            return

        self._set_status(workflow, WorkflowStatus.RUNNING, state)

    def _set_status(
        self, workflow: Workflow, target: WorkflowStatus, state: _Pass
    ) -> None:
        if workflow.status is target:
            return
        assert_workflow_transition(
            workflow.status, target, workflow_id=workflow.workflow_id
        )
        previous = workflow.status
        workflow.status = target
        state.changed = True
        logger.info(
            "workflow status changed",
            extra={"from": str(previous), "to": str(target)},
        )
        # Every status change passes through here, so this is the one place a
        # child has to announce itself. Waiting for a person counts: a parent
        # that does not wait for approvals is finished the moment its last
        # child reaches that state, and nothing else would tell it.
        if target.is_terminal or target is WorkflowStatus.WAITING_FOR_APPROVAL:
            self._notify_parent(workflow, state)

    def _schedule_next_wake(
        self, workflow: Workflow, definition: WorkflowDefinition
    ) -> None:
        """Set the timer the sweeper uses: earliest retry, lease or approval."""
        candidates = [
            t
            for step in workflow.steps.values()
            for t in (step.next_attempt_at, step.lease_expires_at)
            if t is not None and not step.status.is_terminal
        ]
        workflow.scheduled_wake_at = min(candidates) if candidates else None

    # ==================================================================
    # Plumbing
    # ==================================================================
    def _advance_message(self, workflow: Workflow, *, reason: str) -> Message:
        return Message(
            headers=MessageHeaders(
                message_id=self._ids.new_id(MESSAGE),
                tenant_id=workflow.tenant_id,
                correlation_id=workflow.correlation_id,
                dedup_key=f"advance:{workflow.workflow_id}:{reason}",
            ),
            body=AdvanceWorkflow(workflow_id=workflow.workflow_id, reason=reason),
        )

    async def _flush(self, outbox: list[tuple[Any, Message]]) -> None:
        """Publish the pass's messages after its state is durable.

        A failure here leaves steps DISPATCHED with an unexpired lease; the
        lease sweeper reclaims them, so nothing is lost -- only delayed.
        """
        for queue, message in outbox:
            try:
                await self._bus.publish(queue, message)
            except Exception as exc:
                logger.error(
                    "failed to publish a dispatch message; the lease sweeper will recover",
                    extra={"queue": getattr(queue, "value", str(queue)), "error": str(exc)},
                )



def _progress_signature(workflow: Workflow) -> tuple[tuple[str, str, str | None], ...]:
    """A cheap fingerprint of step progress, used to detect a fixed point."""
    return tuple(
        (step_id, str(step.status), step.run_id)
        for step_id, step in sorted(workflow.steps.items())
    )


def _children_settled(children: list[Workflow], fan_out: FanOutSpec) -> bool:
    """Whether the parent has anything left to wait for.

    A child held for a person has stopped of its own accord: it will not move
    again until somebody acts. Whether that counts as settled is the author's
    call, because both answers are right for different batches.
    """
    for child in children:
        if child.status.is_terminal:
            continue
        if (
            not fan_out.await_approvals
            and child.status is WorkflowStatus.WAITING_FOR_APPROVAL
        ):
            continue
        return False
    return True


def _child_reason(child: Workflow) -> str:
    """Why this case ended where it did.

    A rejected or failed case says so itself. One waiting for a person does
    not -- and that is exactly the case somebody is about to open, so the
    reason comes from the decision that held it: the policy step's own words,
    which are the ones the approver will see too.
    """
    if child.cancellation_reason:
        return child.cancellation_reason
    if child.error is not None:
        return child.error.message
    for step in child.steps.values():
        output = step.output or {}
        reasons = output.get("reasons")
        if output.get("effect") and isinstance(reasons, list) and reasons:
            return "; ".join(str(r) for r in reasons)[:300]
    return ""


def _child_input(
    workflow: Workflow, fan_out: FanOutSpec, item: Any
) -> dict[str, Any]:
    """The input one child receives.

    The item *is* the input, rather than being nested under a key. That is
    what lets the same child definition serve both callers: one claim posted
    to the API and one row of a thousand-row batch arrive looking identical,
    so ``input.amount`` means the same thing either way.

    The parent's own input stays reachable under ``parent`` for the shared
    context a case sometimes needs -- the document it came from, the period it
    covers -- without it shadowing the item's own fields.
    """
    if isinstance(item, dict):
        return {**item, "parent": workflow.input}
    return {fan_out.item_as: item, "parent": workflow.input}


def _summarise_children(children: list[Workflow]) -> dict[str, Any]:
    """What a batch of child workflows came to.

    Counts first, because the question a person asks of a thousand invoices is
    how many need them. The per-child list follows, bounded, with the ones
    needing attention first -- a run of five hundred successes is not what
    anybody scrolls looking for.
    """
    by_status: Counter[str] = Counter(str(c.status) for c in children)
    rank = {
        WorkflowStatus.FAILED: 0,
        WorkflowStatus.REJECTED: 1,
        WorkflowStatus.WAITING_FOR_RECOVERY: 2,
        WorkflowStatus.CANCELLED: 3,
        WorkflowStatus.WAITING_FOR_APPROVAL: 4,
        WorkflowStatus.COMPLETED: 5,
    }
    ordered = sorted(
        children,
        key=lambda c: (rank.get(c.status, 9), c.fan_out_index or 0),
    )
    return {
        "total": len(children),
        "completed": by_status.get(str(WorkflowStatus.COMPLETED), 0),
        "rejected": by_status.get(str(WorkflowStatus.REJECTED), 0),
        "failed": by_status.get(str(WorkflowStatus.FAILED), 0),
        "cancelled": by_status.get(str(WorkflowStatus.CANCELLED), 0),
        "awaiting_approval": by_status.get(str(WorkflowStatus.WAITING_FOR_APPROVAL), 0),
        "needs_recovery": by_status.get(str(WorkflowStatus.WAITING_FOR_RECOVERY), 0),
        "children": [
            {
                "workflow_id": c.workflow_id,
                "index": c.fan_out_index,
                "key": c.fan_out_key,
                "status": str(c.status),
                "reason": _child_reason(c),
            }
            for c in ordered[:MAX_REPORTED_CHILDREN]
        ],
        "children_reported": min(len(children), MAX_REPORTED_CHILDREN),
    }


def _resolve_inputs(spec: StepDefinition, workflow: Workflow) -> dict[str, Any]:
    """Resolve a step's declared input bindings against the workflow context."""
    resolved = render(spec.inputs, workflow.context())
    return resolved if isinstance(resolved, dict) else {}


def _upstream_policy_decision(
    workflow: Workflow, spec: StepDefinition
) -> PolicyDecision | None:
    """Find the policy decision that led to this approval, if there was one."""
    for dep in spec.depends_on:
        try:
            step = workflow.step(dep)
        except NotFoundError:  # pragma: no cover - definition guarantees this
            continue
        output = step.output or {}
        if "effect" in output and "ruleset" in output:
            return PolicyDecision(
                effect=PolicyEffect(output["effect"]),
                risk=RiskLevel(output.get("risk", "MEDIUM")),
                ruleset=str(output.get("ruleset", "")),
                required_roles=tuple(output.get("required_roles", ())),
                reasons=tuple(output.get("reasons", ())),
            )
    return None


def _earliest(current: Any, candidate: Any) -> Any:
    if current is None:
        return candidate
    if candidate is None:
        return current
    return min(current, candidate)


_APPROVAL_AUDIT = {
    ApprovalDecision.APPROVE: AuditAction.APPROVAL_GRANTED,
    ApprovalDecision.REJECT: AuditAction.APPROVAL_REJECTED,
    ApprovalDecision.REQUEST_INFO: AuditAction.APPROVAL_INFO_REQUESTED,
}

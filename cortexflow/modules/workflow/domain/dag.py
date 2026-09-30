"""Dependency resolution: deciding what may run right now.

This is the scheduling core, and it is deliberately a *pure function* of
(definition, state, now).  No I/O, no clock reads, no randomness -- which is
why the concurrency and recovery behaviour is unit-testable without any
infrastructure at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from cortexflow.modules.workflow.domain.definition import (
    JoinPolicy,
    StepDefinition,
    WorkflowDefinition,
)
from cortexflow.modules.workflow.domain.models import StepStatus, Workflow
from cortexflow.shared.expressions import evaluate_condition


class Gate(StrEnum):
    """Why a step is not currently runnable."""

    READY = "READY"
    BLOCKED = "BLOCKED"
    """At least one dependency is still in flight."""

    SKIP = "SKIP"
    """Guard is false, or every contributing branch was skipped."""

    CANCEL = "CANCEL"
    """An upstream dependency failed or was cancelled."""

    BACKOFF = "BACKOFF"
    """Failed and awaiting its next retry window."""

    DONE = "DONE"


@dataclass(frozen=True, slots=True)
class StepDecision:
    step_id: str
    gate: Gate
    reason: str = ""

    @property
    def is_runnable(self) -> bool:
        return self.gate is Gate.READY


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """What the orchestrator should do on this pass."""

    runnable: tuple[StepDefinition, ...]
    to_skip: tuple[StepDecision, ...]
    to_cancel: tuple[StepDecision, ...]
    blocked: tuple[StepDecision, ...]
    waiting_backoff: tuple[StepDecision, ...]

    @property
    def has_work(self) -> bool:
        return bool(self.runnable or self.to_skip or self.to_cancel)

    @property
    def is_stalled(self) -> bool:
        """Nothing to do now, and nothing pending that could unblock later."""
        return not self.has_work and not self.blocked and not self.waiting_backoff


def decide_step(
    definition: WorkflowDefinition,
    workflow: Workflow,
    step_id: str,
    *,
    now: datetime,
) -> StepDecision:
    """Decide the fate of a single step given current state."""
    state = workflow.step(step_id)
    spec = definition.step(step_id)

    if state.status.is_terminal:
        return StepDecision(step_id, Gate.DONE, f"already {state.status}")
    if state.status.is_in_flight or state.status is StepStatus.WAITING_FOR_APPROVAL:
        return StepDecision(step_id, Gate.BLOCKED, f"in flight ({state.status})")

    dependency_gate = _evaluate_dependencies(definition, workflow, spec)
    if dependency_gate.gate is not Gate.READY:
        return dependency_gate

    if state.next_attempt_at is not None and state.next_attempt_at > now:
        return StepDecision(step_id, Gate.BACKOFF, f"retry at {state.next_attempt_at.isoformat()}")

    if spec.when and not evaluate_condition(spec.when, workflow.context()):
        return StepDecision(step_id, Gate.SKIP, f"guard false: {spec.when}")

    return StepDecision(step_id, Gate.READY)


def _evaluate_dependencies(
    definition: WorkflowDefinition, workflow: Workflow, spec: StepDefinition
) -> StepDecision:
    """Apply the join policy to a step's dependencies.

    Criticality matters here.  A *critical* dependency that failed makes the
    rest of the branch meaningless, so its dependents are cancelled.  A
    *non-critical* one -- an optional enrichment lookup, a best-effort report
    -- is allowed to fail without taking the workflow with it; downstream steps
    proceed with whatever data they do have.

    Cancellation is different from failure: a cancelled dependency means its
    branch was abandoned, so it propagates regardless of criticality.
    """
    if not spec.depends_on:
        return StepDecision(spec.id, Gate.READY)

    statuses = {dep: workflow.step(dep).status for dep in spec.depends_on}
    unfinished = [dep for dep, st in statuses.items() if not st.is_terminal]
    succeeded = [dep for dep, st in statuses.items() if st is StepStatus.SUCCEEDED]
    failed = [
        dep
        for dep, st in statuses.items()
        if st in {StepStatus.FAILED, StepStatus.CANCELLED}
    ]
    # Cancellation always propagates: it is a directive from above, so work
    # downstream of an abandoned branch must not run on partial data. Only a
    # *failed* non-critical step is tolerated.
    blocking = [
        dep
        for dep in failed
        if definition.step(dep).critical
        or statuses[dep] is StepStatus.CANCELLED
    ]

    if spec.join is JoinPolicy.ANY:
        # A merge point: as soon as one branch succeeds we may proceed, but we
        # wait for the others to settle so a late success cannot arrive after
        # the merge already ran.
        if unfinished:
            return StepDecision(spec.id, Gate.BLOCKED, f"waiting on {sorted(unfinished)}")
        if succeeded:
            return StepDecision(spec.id, Gate.READY)
        if blocking:
            return StepDecision(
                spec.id, Gate.CANCEL, f"all branches failed: {sorted(blocking)}"
            )
        return StepDecision(spec.id, Gate.SKIP, "no branch was taken")

    if blocking:
        return StepDecision(spec.id, Gate.CANCEL, f"dependency failed: {sorted(blocking)}")
    if unfinished:
        return StepDecision(spec.id, Gate.BLOCKED, f"waiting on {sorted(unfinished)}")
    # A skipped dependency means this branch of the graph was not taken.
    # A failed non-critical one is tolerated, so it does not propagate.
    skipped = [dep for dep, st in statuses.items() if st is StepStatus.SKIPPED]
    if skipped:
        return StepDecision(spec.id, Gate.SKIP, f"dependency skipped: {sorted(skipped)}")
    return StepDecision(spec.id, Gate.READY)


def build_plan(
    definition: WorkflowDefinition, workflow: Workflow, *, now: datetime
) -> ExecutionPlan:
    """Compute every action available on this orchestration pass.

    Independent steps land in ``runnable`` together and are dispatched
    concurrently; dependent steps stay ``blocked`` until their inputs settle.
    """
    runnable: list[StepDefinition] = []
    to_skip: list[StepDecision] = []
    to_cancel: list[StepDecision] = []
    blocked: list[StepDecision] = []
    backoff: list[StepDecision] = []

    for step_id in definition.topological_order():
        decision = decide_step(definition, workflow, step_id, now=now)
        match decision.gate:
            case Gate.READY:
                runnable.append(definition.step(step_id))
            case Gate.SKIP:
                to_skip.append(decision)
            case Gate.CANCEL:
                to_cancel.append(decision)
            case Gate.BLOCKED:
                blocked.append(decision)
            case Gate.BACKOFF:
                backoff.append(decision)
            case Gate.DONE:
                pass

    return ExecutionPlan(
        runnable=tuple(runnable),
        to_skip=tuple(to_skip),
        to_cancel=tuple(to_cancel),
        blocked=tuple(blocked),
        waiting_backoff=tuple(backoff),
    )


def critical_path(definition: WorkflowDefinition) -> tuple[str, ...]:
    """Longest dependency chain -- the floor on end-to-end latency."""
    best: dict[str, tuple[str, ...]] = {}
    for step_id in definition.topological_order():
        deps = definition.step(step_id).depends_on
        prefix = max((best[d] for d in deps), key=len, default=())
        best[step_id] = (*prefix, step_id)
    return max(best.values(), key=len, default=())


def parallel_levels(definition: WorkflowDefinition) -> tuple[tuple[str, ...], ...]:
    """Group steps into waves that *could* execute concurrently.

    Used by the dashboard to draw the graph and by tests to assert that
    independent validations really are in the same wave.
    """
    depth: dict[str, int] = {}
    for step_id in definition.topological_order():
        deps = definition.step(step_id).depends_on
        depth[step_id] = 1 + max((depth[d] for d in deps), default=-1)
    levels: dict[int, list[str]] = {}
    for step_id, level in depth.items():
        levels.setdefault(level, []).append(step_id)
    return tuple(tuple(sorted(levels[level])) for level in sorted(levels))

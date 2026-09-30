"""Explicit state machines for workflows and steps.

Illegal transitions are a bug, not a warning.  Making them raise turns whole
classes of race condition (a late worker completing a cancelled workflow, a
duplicate message re-running a finished step) into loud, testable failures.
"""

from __future__ import annotations

from cortexflow.modules.workflow.domain.models import StepStatus, WorkflowStatus
from cortexflow.shared.errors import InvalidStateTransitionError

_WORKFLOW_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.PENDING: frozenset(
        {
            WorkflowStatus.RUNNING,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.FAILED,
            WorkflowStatus.COMPLETED,
        }
    ),
    WorkflowStatus.RUNNING: frozenset(
        {
            WorkflowStatus.RUNNING,
            WorkflowStatus.WAITING_FOR_APPROVAL,
            WorkflowStatus.WAITING_FOR_RECOVERY,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.REJECTED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.WAITING_FOR_APPROVAL: frozenset(
        {
            WorkflowStatus.RUNNING,
            WorkflowStatus.WAITING_FOR_APPROVAL,
            # An approval can be the last thing a workflow does: "sign this
            # off and we are finished". There is nothing to resume into, so
            # the approval that completes it takes it straight to COMPLETED.
            WorkflowStatus.COMPLETED,
            WorkflowStatus.REJECTED,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.FAILED,
        }
    ),
    WorkflowStatus.WAITING_FOR_RECOVERY: frozenset(
        {
            WorkflowStatus.RUNNING,
            WorkflowStatus.WAITING_FOR_RECOVERY,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.FAILED,
        }
    ),
    WorkflowStatus.COMPLETED: frozenset(),
    WorkflowStatus.FAILED: frozenset({WorkflowStatus.RUNNING}),  # operator replay
    WorkflowStatus.REJECTED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
}

_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset(
        {
            StepStatus.DISPATCHED,
            StepStatus.WAITING_FOR_APPROVAL,
            StepStatus.SKIPPED,
            StepStatus.CANCELLED,
            StepStatus.SUCCEEDED,  # synchronous control-plane steps
            StepStatus.FAILED,
        }
    ),
    StepStatus.DISPATCHED: frozenset(
        {
            StepStatus.RUNNING,
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.CANCELLED,
            StepStatus.PENDING,  # lease expiry returns the step to the pool
        }
    ),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.WAITING_FOR_APPROVAL,
            StepStatus.CANCELLED,
            StepStatus.PENDING,
        }
    ),
    StepStatus.WAITING_FOR_APPROVAL: frozenset(
        {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CANCELLED}
    ),
    StepStatus.SUCCEEDED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
    StepStatus.FAILED: frozenset({StepStatus.PENDING}),  # retry or operator replay
}


def can_transition_workflow(current: WorkflowStatus, target: WorkflowStatus) -> bool:
    return target in _WORKFLOW_TRANSITIONS[current]


def assert_workflow_transition(
    current: WorkflowStatus, target: WorkflowStatus, *, workflow_id: str
) -> None:
    if not can_transition_workflow(current, target):
        raise InvalidStateTransitionError(
            f"Cannot move workflow from {current} to {target}",
            workflow_id=workflow_id,
            current=str(current),
            target=str(target),
        )


def can_transition_step(current: StepStatus, target: StepStatus) -> bool:
    return target in _STEP_TRANSITIONS[current]


def assert_step_transition(
    current: StepStatus, target: StepStatus, *, workflow_id: str, step_id: str
) -> None:
    if not can_transition_step(current, target):
        raise InvalidStateTransitionError(
            f"Cannot move step from {current} to {target}",
            workflow_id=workflow_id,
            step_id=step_id,
            current=str(current),
            target=str(target),
        )

"""Queue naming and routing.

Routing is a pure function of the step definition so that a workflow author
never hard-codes infrastructure, and so an operator can scale the extraction
queue independently of the decision queue.
"""

from __future__ import annotations

from cortexflow.modules.workflow.domain.definition import Queue, StepDefinition

AGENT_QUEUES: tuple[Queue, ...] = (
    Queue.EXTRACTION,
    Queue.VALIDATION,
    Queue.DECISION,
    Queue.REPORTING,
)
"""Queues consumed by agent workers."""

WORKER_QUEUES: tuple[Queue, ...] = (*AGENT_QUEUES, Queue.INTEGRATION)
"""Every queue that carries EXECUTE_STEP messages.

CONTROL and NOTIFICATION are deliberately absent: they carry orchestration
and delivery messages respectively, and are consumed by the orchestrator and
the notification service rather than by agent workers."""


def queue_name(queue: Queue, prefix: str = "cortexflow") -> str:
    return f"{prefix}-{queue.value}"


def dead_letter_name(queue: Queue, prefix: str = "cortexflow") -> str:
    return f"{queue_name(queue, prefix)}-dlq"


def route_step(step: StepDefinition) -> Queue:
    """Pick the execution queue for a step."""
    return step.effective_queue


def all_queues() -> tuple[Queue, ...]:
    return tuple(Queue)

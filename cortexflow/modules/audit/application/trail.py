"""Audit emission helper.

Centralised so that every orchestration decision produces a record with the
same shape, and so audit failures never take down the workflow that was being
recorded -- the event is logged and the workflow proceeds.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cortexflow.modules.audit.domain.models import AuditAction, AuditEvent, AuditOutcome
from cortexflow.modules.audit.ports.repositories import AuditRepository
from cortexflow.shared.ids import AUDIT, IdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.redaction import redact

logger = get_logger(__name__)

SYSTEM_ACTOR = "system::orchestrator"


@runtime_checkable
class Auditable(Protocol):
    """Anything with an identity and a correlation id can be audited."""

    tenant_id: str
    workflow_id: str
    correlation_id: str


class AuditTrail:
    def __init__(self, repository: AuditRepository, ids: IdGenerator) -> None:
        self._repository = repository
        self._ids = ids

    async def record(
        self,
        *,
        tenant_id: str,
        action: AuditAction,
        actor: str = SYSTEM_ACTOR,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        workflow_id: str | None = None,
        step_id: str | None = None,
        run_id: str | None = None,
        tool: str | None = None,
        correlation_id: str = "",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        actor_roles: tuple[str, ...] = (),
    ) -> None:
        event = AuditEvent(
            event_id=self._ids.new_id(AUDIT),
            tenant_id=tenant_id,
            action=action,
            outcome=outcome,
            actor=actor,
            actor_roles=actor_roles,
            workflow_id=workflow_id,
            step_id=step_id,
            run_id=run_id,
            tool=tool,
            correlation_id=correlation_id,
            summary=summary[:1000],
            metadata=redact(metadata or {}),
        )
        try:
            await self._repository.append(event)
        except Exception as exc:
            # Losing an audit record is serious, but failing the workflow to
            # protect the audit log would be worse. Log loudly and continue.
            logger.error(
                "failed to persist audit event",
                extra={
                    "action": str(action),
                    "workflow_id": workflow_id,
                    "error": str(exc),
                },
            )

    async def record_for(
        self, workflow: Auditable, action: AuditAction, **kwargs: Any
    ) -> None:
        """Record an event about something that has an identity and a trace.

        Typed structurally rather than as a ``Workflow``: the audit module
        needs three strings, and importing the workflow aggregate to read them
        would make the two modules depend on each other in a circle -- audit
        recording workflows, workflows recording audit. Declaring the shape
        here inverts that, and anything carrying these three fields can be
        audited without audit knowing what it is.
        """
        await self.record(
            tenant_id=workflow.tenant_id,
            action=action,
            workflow_id=workflow.workflow_id,
            correlation_id=workflow.correlation_id,
            **kwargs,
        )

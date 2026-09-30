"""The orchestrator's control-queue consumer.

Every control message converges on one behaviour: apply a fact, then advance
the workflow.  Because advancing is derived from durable state, a duplicate or
out-of-order delivery is harmless -- which is what makes at-least-once
messaging safe here.

Settlement rules:

* success or a permanent failure -> ``complete``; redelivery would not help.
* transient failure -> ``abandon``; the broker redelivers with backoff.
* repeated failure past the delivery limit -> ``dead_letter``; an operator
  investigates and replays from the dashboard.
"""

from __future__ import annotations

import asyncio
import contextlib

from cortexflow.modules.approval.domain.models import ApprovalDecision
from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.workflow.application.engine import WorkflowEngine
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import (
    AdvanceWorkflow,
    ApprovalResolved,
    Message,
    MessageType,
    StepCompleted,
    StepFailed,
)
from cortexflow.modules.workflow.domain.models import StepError
from cortexflow.modules.workflow.ports.messaging import MessageBus, ReceivedMessage
from cortexflow.modules.workflow.ports.repositories import DeadLetterRepository
from cortexflow.shared.errors import CortexFlowError, ErrorClass, classify
from cortexflow.shared.observability.logging import bind_context, get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.telemetry import span
from cortexflow.shared.ports.cache import Cache

logger = get_logger(__name__)

MAX_DELIVERY_ATTEMPTS = 5


class ControlPlaneDispatcher:
    """Consumes the control queue and drives the engine."""

    def __init__(
        self,
        *,
        engine: WorkflowEngine,
        bus: MessageBus,
        cache: Cache,
        audit: AuditTrail,
        dead_letters: DeadLetterRepository,
        dedup_ttl_seconds: int = 86_400,
    ) -> None:
        self._engine = engine
        self._bus = bus
        self._cache = cache
        self._audit = audit
        self._dead_letters = dead_letters
        self._dedup_ttl = dedup_ttl_seconds
        self._metrics = get_metrics()
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        """Consume until stopped."""
        logger.info("control-plane dispatcher started")
        async for received in self._bus.consume(Queue.CONTROL, max_messages=1):
            if self._stopping.is_set():
                await received.abandon("shutting down")
                break
            await self.handle(received)
        logger.info("control-plane dispatcher stopped")

    def stop(self) -> None:
        self._stopping.set()

    async def handle(self, received: ReceivedMessage) -> None:
        """Process one control message and settle it."""
        message = received.message
        headers = message.headers

        with (
            bind_context(
                tenant_id=headers.tenant_id,
                correlation_id=headers.correlation_id,
                message_id=headers.message_id,
            ),
            span(
                "control.handle",
                **{
                    "message.type": str(message.type),
                    "message.delivery_count": received.delivery_count,
                },
            ),
        ):
            try:
                if await self._is_duplicate(message):
                    await received.complete()
                    return
                await self._route(message)
                await received.complete()
            except CortexFlowError as exc:
                await self._settle_failure(received, exc, exc.error_class)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("unhandled error in the control dispatcher")
                await self._settle_failure(received, exc, classify(exc))

    async def _route(self, message: Message) -> None:
        body = message.body
        tenant_id = message.headers.tenant_id

        match body:
            case AdvanceWorkflow():
                await self._engine.advance(tenant_id, body.workflow_id, reason=body.reason)
            case StepCompleted():
                await self._engine.complete_step(
                    tenant_id,
                    body.workflow_id,
                    body.step_id,
                    run_id=body.run_id,
                    output=body.output,
                    confidence=body.confidence,
                    duration_ms=body.duration_ms,
                    cost=body.cost,
                )
            case StepFailed():
                await self._engine.fail_step(
                    tenant_id,
                    body.workflow_id,
                    body.step_id,
                    run_id=body.run_id,
                    error=StepError(
                        code=body.code,
                        message=body.message,
                        error_class=body.error_class,
                        details=body.details,
                    ),
                    duration_ms=body.duration_ms,
                )
            case ApprovalResolved():
                await self._engine.resolve_approval(
                    tenant_id,
                    body.workflow_id,
                    body.step_id,
                    approval_id=body.approval_id,
                    decision=ApprovalDecision(body.decision),
                    decided_by=body.decided_by,
                    comment=body.comment,
                )
            case _:
                logger.warning(
                    "control queue received an unsupported message type",
                    extra={"message_type": str(message.type)},
                )

    async def _is_duplicate(self, message: Message) -> bool:
        """Application-level deduplication.

        The broker's own duplicate detection has a bounded window; this guard
        covers the rest.  Advance messages are exempt because re-advancing is
        idempotent by construction and dropping one could strand a workflow.
        """
        if message.type is MessageType.ADVANCE_WORKFLOW:
            return False
        key = message.headers.dedup_key or message.headers.message_id
        if not key:
            return False

        first_time = await self._cache.add(
            f"dedup:{key}", message.headers.message_id, ttl_seconds=self._dedup_ttl
        )
        if first_time:
            return False

        self._metrics.count(self._metrics.duplicates, message_type=str(message.type))
        logger.info("duplicate message dropped", extra={"dedup_key": key})
        await self._audit.record(
            tenant_id=message.headers.tenant_id,
            action=AuditAction.MESSAGE_DEDUPLICATED,
            correlation_id=message.headers.correlation_id,
            summary=f"Duplicate {message.type} dropped",
            metadata={"dedup_key": key},
        )
        return True

    async def _settle_failure(
        self, received: ReceivedMessage, exc: Exception, error_class: ErrorClass
    ) -> None:
        message = received.message
        retryable = error_class is not ErrorClass.PERMANENT
        exhausted = received.delivery_count >= MAX_DELIVERY_ATTEMPTS

        if retryable and not exhausted:
            logger.warning(
                "control message failed; returning it for redelivery",
                extra={
                    "error": str(exc),
                    "error_class": str(error_class),
                    "delivery_count": received.delivery_count,
                },
            )
            await received.abandon(str(exc))
            return

        reason = (
            "permanent_failure" if not retryable else "max_delivery_attempts_exceeded"
        )
        logger.error(
            "control message dead-lettered",
            extra={"reason": reason, "error": str(exc), "message_type": str(message.type)},
        )
        self._metrics.count(
            self._metrics.dead_letters, queue="control", message_type=str(message.type)
        )
        with contextlib.suppress(Exception):
            await self._dead_letters.record(
                tenant_id=message.headers.tenant_id,
                queue=Queue.CONTROL.value,
                message=message.to_transport(),
                reason=f"{reason}: {exc}",
                attempts=received.delivery_count,
            )
        await self._audit.record(
            tenant_id=message.headers.tenant_id,
            action=AuditAction.STEP_DEAD_LETTERED,
            correlation_id=message.headers.correlation_id,
            summary=str(exc)[:500],
            metadata={"reason": reason, "message_type": str(message.type)},
        )
        await received.dead_letter(reason, str(exc)[:2000])

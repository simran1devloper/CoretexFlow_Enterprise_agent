"""The recovery sweeper.

Nothing in CortexFlow holds a timer in memory.  Retry windows, approval
timeouts and worker leases are all *stored*, and this periodic task is what
notices they have elapsed.  That is why a full restart of every process loses
no scheduled work.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta

from cortexflow.modules.approval.ports.repositories import ApprovalRepository
from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.workflow.application.engine import WorkflowEngine
from cortexflow.modules.workflow.ports.repositories import WorkflowRepository
from cortexflow.shared.clock import Clock
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import span
from cortexflow.shared.ports.cache import LockManager

logger = get_logger(__name__)

SWEEP_LOCK = "sweeper:global"


class RecoverySweeper:
    """Periodically reactivates workflows whose timers have elapsed."""

    def __init__(
        self,
        *,
        engine: WorkflowEngine,
        workflows: WorkflowRepository,
        approvals: ApprovalRepository,
        locks: LockManager,
        audit: AuditTrail,
        clock: Clock,
        interval_seconds: float = 10.0,
        batch_size: int = 100,
        owner: str = "sweeper",
    ) -> None:
        self._engine = engine
        self._workflows = workflows
        self._approvals = approvals
        self._locks = locks
        self._audit = audit
        self._clock = clock
        self._interval = interval_seconds
        self._batch_size = batch_size
        self._owner = owner
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        logger.info("recovery sweeper started", extra={"interval": self._interval})
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            if self._stopping.is_set():
                break
            try:
                await self.sweep_once()
            except Exception:  # pragma: no cover - the sweeper must never die
                logger.exception("sweep failed; continuing")
        logger.info("recovery sweeper stopped")

    def stop(self) -> None:
        self._stopping.set()

    async def sweep_once(self) -> dict[str, int]:
        """Run one sweep. Returns counts, which the tests assert on.

        A global lease keeps replicas from sweeping the same rows: several
        orchestrators may run, but only one sweeps at a time.
        """
        lease = await self._locks.acquire(
            SWEEP_LOCK, owner=self._owner, ttl_seconds=int(self._interval * 3)
        )
        if lease is None:
            return {"due": 0, "stalled": 0, "expired_approvals": 0}

        try:
            with span("sweeper.sweep"):
                due = await self._sweep_due()
                stalled = await self._sweep_stalled()
                expired = await self._sweep_expired_approvals()
        finally:
            await lease.release()

        if due or stalled or expired:
            logger.info(
                "sweep complete",
                extra={"due": due, "stalled": stalled, "expired_approvals": expired},
            )
        return {"due": due, "stalled": stalled, "expired_approvals": expired}

    async def _sweep_due(self) -> int:
        """Workflows whose retry backoff or approval timer has elapsed."""
        due = await self._workflows.find_due(
            now=self._clock.now(), limit=self._batch_size
        )
        for tenant_id, workflow_id in due:
            with contextlib.suppress(Exception):
                await self._engine.advance(tenant_id, workflow_id, reason="timer elapsed")
        return len(due)

    async def _sweep_stalled(self) -> int:
        """Workflows holding steps whose worker leases expired -- crashed workers.

        Advancing re-runs lease reclamation inside the engine, which returns
        the abandoned steps to the runnable pool.
        """
        cutoff = self._clock.now() - timedelta(seconds=1)
        stalled = await self._workflows.find_stalled(
            older_than=cutoff, limit=self._batch_size
        )
        for tenant_id, workflow_id in stalled:
            logger.warning(
                "recovering a workflow with expired step leases",
                extra={"workflow_id": workflow_id},
            )
            await self._audit.record(
                tenant_id=tenant_id,
                action=AuditAction.STEP_LEASE_RECLAIMED,
                workflow_id=workflow_id,
                summary="Step lease expired; the step was returned to the queue",
            )
            with contextlib.suppress(Exception):
                await self._engine.advance(
                    tenant_id, workflow_id, reason="lease recovery"
                )
        return len(stalled)

    async def _sweep_expired_approvals(self) -> int:
        """Approvals nobody answered in time."""
        expired = await self._approvals.find_expired(
            now=self._clock.now(), limit=self._batch_size
        )
        for approval in expired:
            with contextlib.suppress(Exception):
                await self._engine.expire_approval(approval)
        return len(expired)

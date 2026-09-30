"""Approval application service.

Enforces who may decide what, then hands the decision to the engine.  The
authorization check here is the one that makes human-in-the-loop meaningful:
a workflow cannot approve itself, and a principal cannot approve something
their role does not cover.
"""

from __future__ import annotations

from cortexflow.modules.approval.domain.models import Approval, ApprovalDecision, ApprovalStatus
from cortexflow.modules.approval.ports.repositories import ApprovalRepository
from cortexflow.shared.clock import Clock
from cortexflow.shared.errors import AuthorizationError, ConflictError
from cortexflow.shared.identity import Principal
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.security.rbac import Permission, require_permission, require_tenant

logger = get_logger(__name__)


class ApprovalService:
    def __init__(
        self,
        *,
        approvals: ApprovalRepository,
        engine: object,
        clock: Clock,
    ) -> None:
        self._approvals = approvals
        self._engine = engine
        self._clock = clock

    async def list_pending(
        self, principal: Principal, *, workflow_id: str | None = None, limit: int = 50
    ) -> list[Approval]:
        require_permission(principal, Permission.APPROVAL_READ)
        pending = await self._approvals.list_pending(
            principal.tenant_id, workflow_id=workflow_id, limit=limit
        )
        # Show only what this principal could actually act on, so the queue is
        # a work list rather than a wall of items they cannot touch.
        if principal.is_admin():
            return pending
        return [a for a in pending if a.can_be_decided_by_roles(principal.roles)]

    async def get(self, principal: Principal, approval_id: str) -> Approval:
        require_permission(principal, Permission.APPROVAL_READ)
        approval = await self._approvals.get(principal.tenant_id, approval_id)
        require_tenant(principal, approval.tenant_id)
        return approval

    async def decide(
        self,
        principal: Principal,
        approval_id: str,
        *,
        decision: ApprovalDecision,
        comment: str = "",
    ) -> Approval:
        """Record a human decision and resume the workflow."""
        require_permission(principal, Permission.APPROVAL_DECIDE)
        approval = await self._approvals.get(principal.tenant_id, approval_id)
        require_tenant(principal, approval.tenant_id)

        if approval.status.is_terminal:
            raise ConflictError(
                "Approval has already been resolved",
                approval_id=approval_id,
                status=str(approval.status),
            )
        if not approval.can_be_decided_by_roles(principal.roles) and not principal.is_admin():
            raise AuthorizationError(
                "Principal does not hold a role permitted to decide this approval",
                approval_id=approval_id,
                required_roles=[str(r) for r in approval.required_roles],
            )
        if (
            decision is ApprovalDecision.REQUEST_INFO
            and not approval.context.get("allow_request_info", True)
        ):
            raise ConflictError(
                "This approval does not allow information requests",
                approval_id=approval_id,
            )

        from cortexflow.modules.approval.domain.models import ApprovalResolution

        approval.status = _STATUS_FOR[decision]
        approval.resolution = ApprovalResolution(
            decision=decision,
            decided_by=principal.subject,
            decided_by_name=principal.display_name,
            comment=comment,
            decided_at=self._clock.now(),
        )
        saved = await self._approvals.save(approval, expected_version=approval.version)

        await self._engine.resolve_approval(  # type: ignore[attr-defined]
            saved.tenant_id,
            saved.workflow_id,
            saved.step_id,
            approval_id=saved.approval_id,
            decision=decision,
            decided_by=principal.subject,
            decided_by_roles=tuple(sorted(str(r) for r in principal.roles)),
            comment=comment,
        )
        logger.info(
            "approval decided",
            extra={
                "approval_id": approval_id,
                "decision": str(decision),
                "workflow_id": saved.workflow_id,
            },
        )
        return saved


_STATUS_FOR = {
    ApprovalDecision.APPROVE: ApprovalStatus.APPROVED,
    ApprovalDecision.REJECT: ApprovalStatus.REJECTED,
    ApprovalDecision.REQUEST_INFO: ApprovalStatus.INFO_REQUESTED,
}

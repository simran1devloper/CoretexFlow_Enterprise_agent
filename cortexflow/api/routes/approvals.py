"""Approval endpoints -- the human-in-the-loop surface."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from cortexflow.api.middleware.dependencies import ApprovalServiceDep, CurrentPrincipal
from cortexflow.api.schemas import ApprovalDecisionRequest, ApprovalView
from cortexflow.modules.approval.domain.models import ApprovalDecision

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("", response_model=list[ApprovalView])
async def list_pending_approvals(
    principal: CurrentPrincipal,
    service: ApprovalServiceDep,
    workflow_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ApprovalView]:
    """Approvals this principal can actually decide."""
    approvals = await service.list_pending(
        principal, workflow_id=workflow_id, limit=limit
    )
    return [ApprovalView.build(a) for a in approvals]


@router.get("/{approval_id}", response_model=ApprovalView)
async def get_approval(
    approval_id: str, principal: CurrentPrincipal, service: ApprovalServiceDep
) -> ApprovalView:
    return ApprovalView.build(await service.get(principal, approval_id))


@router.post("/{approval_id}/approve", response_model=ApprovalView)
async def approve(
    approval_id: str,
    request: ApprovalDecisionRequest,
    principal: CurrentPrincipal,
    service: ApprovalServiceDep,
) -> ApprovalView:
    """Approve and resume the workflow."""
    approval = await service.decide(
        principal, approval_id,
        decision=ApprovalDecision.APPROVE, comment=request.comment,
    )
    return ApprovalView.build(approval)


@router.post("/{approval_id}/reject", response_model=ApprovalView)
async def reject(
    approval_id: str,
    request: ApprovalDecisionRequest,
    principal: CurrentPrincipal,
    service: ApprovalServiceDep,
) -> ApprovalView:
    """Reject: the workflow ends REJECTED with this decision recorded."""
    approval = await service.decide(
        principal, approval_id,
        decision=ApprovalDecision.REJECT, comment=request.comment,
    )
    return ApprovalView.build(approval)


@router.post("/{approval_id}/request-info", response_model=ApprovalView)
async def request_info(
    approval_id: str,
    request: ApprovalDecisionRequest,
    principal: CurrentPrincipal,
    service: ApprovalServiceDep,
) -> ApprovalView:
    """Ask for more information; the workflow stays suspended."""
    approval = await service.decide(
        principal, approval_id,
        decision=ApprovalDecision.REQUEST_INFO, comment=request.comment,
    )
    return ApprovalView.build(approval)

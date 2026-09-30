"""Cases: one decision, read the way a business reads it.

A thin route over a read model. Nothing here decides anything; the engine
already did, and these endpoints report it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.modules.workflow.application.cases import CaseDetail, CaseSummary
from cortexflow.modules.workflow.domain.models import WorkflowStatus

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("")
async def list_cases(
    principal: CurrentPrincipal,
    container: ContainerDep,
    batch: Annotated[str | None, Query(description="Only the cases this run produced")] = None,
    status: Annotated[WorkflowStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[CaseSummary]:
    """Cases in this tenant, newest first.

    With ``batch``, the cases one fan-out run produced -- which is the read
    somebody makes after opening a batch of ten claims and wanting the claims.
    """
    return await container.workflow_service.list_cases(
        principal, batch_id=batch, status=status, limit=limit
    )


@router.get("/{workflow_id}")
async def get_case(
    workflow_id: str, principal: CurrentPrincipal, container: ContainerDep
) -> CaseDetail:
    """One case: its decision, what the model advised, and its timeline."""
    return await container.workflow_service.case(principal, workflow_id)

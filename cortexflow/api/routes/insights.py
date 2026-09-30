"""Insights: how the platform and its agents are behaving.

Read models over runs that already happened. Nothing here decides anything.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.modules.workflow.application.analytics import Analytics
from cortexflow.modules.workflow.application.evaluation import AiEvaluation

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/ai-evaluation")
async def ai_evaluation(
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    days: Annotated[int | None, Query(ge=1, le=365, description="Only runs this recent")] = None,
) -> AiEvaluation:
    """Agent performance, measured from what the engine recorded.

    The response carries an ``unmeasured`` list naming what this platform
    cannot assess yet, so a client showing these figures also has the caveat.
    """
    return await container.workflow_service.ai_evaluation(
        principal, limit=limit, days=days
    )


@router.get("/analytics")
async def analytics(
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    days: Annotated[int | None, Query(ge=1, le=365, description="Only runs this recent")] = None,
) -> Analytics:
    """Throughput, outcomes and cycle time, counted from finished runs.

    The response carries a ``caveats`` list naming what these figures do not
    account for, so a client showing them also has the qualification. What is
    deliberately absent is as considered as what is here: hours and cost saved
    would both need a baseline nothing records.
    """
    return await container.workflow_service.analytics(principal, limit=limit, days=days)

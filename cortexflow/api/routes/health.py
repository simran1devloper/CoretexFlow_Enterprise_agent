"""Health and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from cortexflow import __version__
from cortexflow.api.middleware.dependencies import ContainerDep
from cortexflow.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    """Liveness: the process is up and its configuration loaded."""
    return HealthResponse(
        status="ok",
        service="api",
        version=__version__,
        profile=str(container.settings.profile),
        workflows=list(container.definitions.names()),
    )


@router.get("/ready", response_model=HealthResponse)
async def ready(container: ContainerDep) -> HealthResponse:
    """Readiness: definitions loaded and the state store answers.

    Deliberately touches the durable store -- a replica that cannot read state
    should not receive traffic.
    """
    await container.workflows.list(container.settings.default_tenant_id, limit=1)
    return HealthResponse(
        status="ready",
        service="api",
        version=__version__,
        profile=str(container.settings.profile),
        workflows=list(container.definitions.names()),
    )

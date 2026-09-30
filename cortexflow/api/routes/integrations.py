"""What this deployment is connected to.

Read-only, and there is no POST. Adapters are chosen from settings when the
container is built, so adding one at runtime would mean constructing a client
and rebinding every repository while requests are in flight. The mechanism
that actually switches an integration is one line in ``config/<profile>.env``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from cortexflow.api.integrations import IntegrationsReport, build_report
from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.shared.security.rbac import Permission, require_permission

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("")
async def list_integrations(
    principal: CurrentPrincipal,
    container: ContainerDep,
    probe: Annotated[
        bool, Query(description="Run the live read-only probes")
    ] = True,
) -> IntegrationsReport:
    """Every outbound dependency: configured, reachable, and in use.

    Needs ``ops:read``. Endpoint hosts and bound adapters are operational
    detail, and the response deliberately carries no secret -- endpoints are
    reported host-only, so a key in a query string cannot leak through.

    ``probe=false`` returns configuration alone, for a caller that wants the
    shape without touching anything.
    """
    require_permission(principal, Permission.OPS_READ)
    return await build_report(container, probe=probe)

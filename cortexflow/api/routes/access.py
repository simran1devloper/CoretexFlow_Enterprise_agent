"""Who may do what here, and who has been doing it.

Two endpoints for two kinds of evidence. ``/access/model`` is the role matrix
the server enforces, served so the UI can stop mirroring it by hand.
``/access/directory`` is the audit trail folded by actor.

There is deliberately no POST: accounts and role assignments belong to
Microsoft Entra ID, and this platform validates tokens rather than issuing
identities.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.api.schemas import AuditEventView
from cortexflow.modules.audit.application.directory import (
    AccessDirectory,
    AccessModel,
    DirectoryEntry,
)
from cortexflow.modules.tool_registry.application.access import (
    ToolAccessReport,
    tool_access,
)
from cortexflow.shared.security.rbac import Permission, require_permission

router = APIRouter(prefix="/access", tags=["access"])


@router.get("/model")
async def access_model(
    principal: CurrentPrincipal, container: ContainerDep
) -> AccessModel:
    """Every role, and exactly which permissions it carries.

    The same table ``require_permission`` consults, so a client can explain a
    refusal instead of guessing at one.
    """
    return container.access_service.model(principal)


@router.get("/directory")
async def access_directory(
    principal: CurrentPrincipal,
    container: ContainerDep,
    days: Annotated[int | None, Query(ge=1, le=365)] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 2000,
) -> AccessDirectory:
    """Principals seen acting in this tenant, with what they did.

    A record of use, not a roster: anyone who has never acted is absent, and
    the response says so in ``unknown`` rather than leaving a client to infer
    that the list is complete.
    """
    return await container.access_service.people(principal, days=days, limit=limit)


@router.get("/tools")
async def access_tools(
    principal: CurrentPrincipal, container: ContainerDep
) -> ToolAccessReport:
    """Every tool, with what the registry would actually decide per role.

    Deliberately readable by anyone who can read a workflow, and deliberately
    not the same endpoint as ``/ops/tools``: that one needs ``definition:read``
    and answers "what is registered", while the person who most needs this
    answers "why can I not do that" -- and an Employee, who holds neither
    ``definition:read`` nor much else, is exactly who asks.
    """
    require_permission(principal, Permission.WORKFLOW_READ)
    return tool_access(container.tools)


@router.get("/activity/{subject:path}", response_model=list[AuditEventView])
async def access_activity(
    subject: str,
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AuditEventView]:
    """This principal's own slice of the audit trail, newest first.

    A separate path from ``/directory/{subject}`` rather than a flag on it:
    the aggregate is cheap and the page loads it for every row, while the feed
    is only wanted for the one row somebody opened.
    """
    events = await container.access_service.activity(principal, subject, limit=limit)
    return [AuditEventView.build(e) for e in events]


@router.get("/directory/{subject:path}")
async def access_person(
    subject: str,
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> DirectoryEntry:
    """One principal, folded from its own events.

    ``:path`` because a subject contains ``::`` and may be an email address.
    """
    return await container.access_service.person(principal, subject, limit=limit)

"""FastAPI dependencies.

The container is built once at startup and stored on ``app.state``; request
handlers receive services from here.  Authentication resolves a
:class:`Principal` on every request -- there is no anonymous path to anything
except ``/health``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from cortexflow.composition import Container
from cortexflow.modules.approval.application.service import ApprovalService
from cortexflow.modules.workflow.application.engine import WorkflowEngine
from cortexflow.modules.workflow.application.service import WorkflowService
from cortexflow.shared.errors import AuthenticationError
from cortexflow.shared.identity import Principal
from cortexflow.shared.security.auth import DEV_PRINCIPAL_HEADER


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


async def get_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_dev_principal: Annotated[str | None, Header(alias=DEV_PRINCIPAL_HEADER)] = None,
) -> Principal:
    """Authenticate the caller.

    Accepts a bearer token in Entra mode, or the development header when the
    dev authenticator is configured (which itself refuses to run outside a
    development environment).
    """
    container = get_container(request)
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_dev_principal:
        token = x_dev_principal

    if not token:
        raise AuthenticationError("Missing credentials")

    principal = await container.authenticator.authenticate(token)
    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
ContainerDep = Annotated[Container, Depends(get_container)]


def get_workflow_service(container: ContainerDep) -> WorkflowService:
    return container.workflow_service


def get_approval_service(container: ContainerDep) -> ApprovalService:
    return container.approval_service


def get_engine(container: ContainerDep) -> WorkflowEngine:
    return container.engine


WorkflowServiceDep = Annotated[WorkflowService, Depends(get_workflow_service)]
ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]
EngineDep = Annotated[WorkflowEngine, Depends(get_engine)]

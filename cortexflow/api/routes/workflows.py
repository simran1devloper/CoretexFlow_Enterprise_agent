"""Workflow endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from cortexflow.api.middleware.dependencies import (
    ContainerDep,
    CurrentPrincipal,
    EngineDep,
    WorkflowServiceDep,
)
from cortexflow.api.schemas import (
    AuditEventView,
    CancelWorkflowRequest,
    CreateWorkflowRequest,
    WorkflowDefinitionView,
    WorkflowListResponse,
    WorkflowView,
)
from cortexflow.modules.workflow.domain.models import WorkflowStatus
from cortexflow.shared.security.rbac import Permission, require_permission

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowView, status_code=status.HTTP_202_ACCEPTED)
async def create_workflow(
    request: CreateWorkflowRequest,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    container: ContainerDep,
) -> WorkflowView:
    """Create a workflow and return immediately.

    202, not 201: the workflow is accepted and queued. Execution happens
    server-side and continues regardless of what the caller does next.
    """
    workflow = await service.create(
        principal,
        definition_name=request.workflow_type,
        definition_version=request.version,
        input_data=request.input,
        correlation_id=request.correlation_id,
        labels=request.labels,
        execution_mode=request.execution_mode,
    )
    definition = container.definitions.get(
        workflow.definition_name,
        workflow.definition_version,
        tenant_id=workflow.tenant_id,
    )
    return WorkflowView.build(workflow, definition)


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    status_filter: Annotated[WorkflowStatus | None, Query(alias="status")] = None,
    workflow_type: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    continuation: Annotated[str | None, Query()] = None,
) -> WorkflowListResponse:
    items, token = await service.list(
        principal,
        status=status_filter,
        definition_name=workflow_type,
        limit=limit,
        continuation=continuation,
    )
    return WorkflowListResponse(items=items, continuation=token)


@router.get("/definitions", response_model=list[WorkflowDefinitionView])
async def list_definitions(
    principal: CurrentPrincipal, service: WorkflowServiceDep
) -> list[WorkflowDefinitionView]:
    """The workflow catalogue this principal may start, with its DAG shape."""
    return [WorkflowDefinitionView.build(d) for d in service.definitions(principal)]


@router.get("/{workflow_id}", response_model=WorkflowView)
async def get_workflow(
    workflow_id: str,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    container: ContainerDep,
) -> WorkflowView:
    """The current state of a workflow -- the answer to "what happened?"."""
    workflow = await service.get(principal, workflow_id)
    definition = container.definitions.get(
        workflow.definition_name,
        workflow.definition_version,
        tenant_id=workflow.tenant_id,
    )
    return WorkflowView.build(workflow, definition)


@router.get("/{workflow_id}/steps", response_model=list[dict])
async def get_workflow_steps(
    workflow_id: str,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    container: ContainerDep,
) -> list[dict]:
    workflow = await service.get(principal, workflow_id)
    definition = container.definitions.get(
        workflow.definition_name,
        workflow.definition_version,
        tenant_id=workflow.tenant_id,
    )
    return [s.model_dump(mode="json") for s in WorkflowView.build(workflow, definition).steps]


@router.get("/{workflow_id}/explanation")
async def explain_workflow(
    workflow_id: str,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
) -> dict[str, Any]:
    """Why this run came out the way it did.

    Including the awkward direction: what the workflow did *not* do and what
    stopped it, which is the question a step list answers worst and the one
    people actually ask.
    """
    explanation = await service.explain(principal, workflow_id)
    return explanation.model_dump(mode="json")


@router.get("/{workflow_id}/audit", response_model=list[AuditEventView])
async def get_workflow_audit(
    workflow_id: str,
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[AuditEventView]:
    """The causal chain: why every action on this workflow happened."""
    require_permission(principal, Permission.AUDIT_READ)
    events = await container.audit_repository.query(
        principal.tenant_id, workflow_id=workflow_id, limit=limit
    )
    return [AuditEventView.build(e) for e in events]


@router.post("/{workflow_id}/cancel", response_model=WorkflowView)
async def cancel_workflow(
    workflow_id: str,
    request: CancelWorkflowRequest,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    engine: EngineDep,
    container: ContainerDep,
) -> WorkflowView:
    await service.get(principal, workflow_id)  # authorization and tenancy check
    require_permission(principal, Permission.WORKFLOW_CANCEL)
    workflow = await engine.cancel(
        principal.tenant_id, workflow_id, actor=principal.subject, reason=request.reason
    )
    definition = container.definitions.get(
        workflow.definition_name,
        workflow.definition_version,
        tenant_id=workflow.tenant_id,
    )
    return WorkflowView.build(workflow, definition)


@router.post("/{workflow_id}/replay", response_model=WorkflowView)
async def replay_workflow(
    workflow_id: str,
    principal: CurrentPrincipal,
    service: WorkflowServiceDep,
    engine: EngineDep,
    container: ContainerDep,
) -> WorkflowView:
    """Operator recovery: re-arm failed steps and resume.

    Safe because every side-effecting step carries an idempotency key, so
    replaying a workflow that partially succeeded does not repeat its effects.
    """
    await service.get(principal, workflow_id)
    require_permission(principal, Permission.WORKFLOW_REPLAY)
    await engine.replay(principal.tenant_id, workflow_id, actor=principal.subject)
    workflow = await service.get(principal, workflow_id)
    definition = container.definitions.get(
        workflow.definition_name,
        workflow.definition_version,
        tenant_id=workflow.tenant_id,
    )
    return WorkflowView.build(workflow, definition)

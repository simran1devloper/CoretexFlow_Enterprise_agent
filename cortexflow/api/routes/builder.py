"""Workflow builder endpoints.

The builder assembles a *draft* in business vocabulary; the server compiles it
into an ordinary workflow definition and validates it exactly as it would one
loaded from disk. Validation is exposed separately from saving, so a person
sees problems while they are still editing.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, status

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.api.schemas import WorkflowDefinitionView
from cortexflow.modules.workflow.application.builder.compiler import (
    WorkflowDraft,
    parse_draft,
)

router = APIRouter(prefix="/builder", tags=["builder"])


@router.get("/catalogue")
async def get_catalogue(
    principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, Any]:
    """Everything the builder can offer this principal.

    Tools are filtered by the same authorization the registry applies at
    execution time, so the builder cannot offer something that would be
    refused when the workflow runs.
    """
    return container.builder_service.catalogue(principal)


@router.post("/validate")
async def validate_draft(
    principal: CurrentPrincipal,
    container: ContainerDep,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Check a draft without saving it.

    The body is parsed leniently, because the builder calls this on every edit
    and a draft mid-edit is routinely incomplete. Rejecting it at the parser
    would return a 422 whose whole content is "unprocessable", which is the
    one thing this endpoint exists not to say -- the missing ruleset comes
    back in ``problems`` instead, phrased for the person editing.

    Returns the execution waves too, so the builder can show which steps will
    run concurrently.
    """
    return container.builder_service.validate(principal, parse_draft(payload))


@router.post("/contract")
async def draft_contract(
    principal: CurrentPrincipal,
    container: ContainerDep,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """What this draft promises to do, for the reader about to publish it.

    Validation answers whether the workflow will execute. This answers whether
    it is the workflow they meant -- what one case is, where a model is
    consulted, which rules authorize the outcome, and what it will change in
    an enterprise system if nobody stops it.
    """
    return container.builder_service.contract(principal, parse_draft(payload))


@router.get("/workflows/{name}/contract")
async def published_contract(
    name: str,
    principal: CurrentPrincipal,
    container: ContainerDep,
    version: int | None = None,
) -> dict[str, Any]:
    """The same reading, for a workflow that is already published."""
    contract = container.builder_service.contract_for_definition(
        principal, name, version
    )
    return contract.model_dump(mode="json")


@router.post("/workflows", status_code=status.HTTP_201_CREATED)
async def publish_draft(
    draft: WorkflowDraft, principal: CurrentPrincipal, container: ContainerDep
) -> WorkflowDefinitionView:
    """Compile, validate and publish a definition for this tenant."""
    definition = await container.builder_service.save(principal, draft)
    return WorkflowDefinitionView.build(definition)


@router.get("/workflows", response_model=list[WorkflowDefinitionView])
async def list_published(
    principal: CurrentPrincipal, container: ContainerDep
) -> list[WorkflowDefinitionView]:
    """Definitions this tenant has authored, as opposed to the built-in ones."""
    definitions = await container.builder_service.list(principal)
    return [WorkflowDefinitionView.build(d) for d in definitions]


@router.get("/workflows/{name}", response_model=WorkflowDefinitionView)
async def get_published(
    name: str, principal: CurrentPrincipal, container: ContainerDep
) -> WorkflowDefinitionView:
    return WorkflowDefinitionView.build(
        await container.builder_service.get(principal, name)
    )


@router.get("/workflows/{name}/draft", response_model=WorkflowDraft)
async def published_draft(
    name: str, principal: CurrentPrincipal, container: ContainerDep
) -> WorkflowDraft:
    """Reopen what somebody assembled, so it can be edited and published again.

    The draft rather than the definition, because compiling is lossy: by the
    time a step is a ``StepDefinition`` the fields an extraction expects and
    the checks a validation runs have been folded into argument expressions.
    Reading that back would take a decompiler mirroring the compiler.

    A workflow shipped in the repository has no draft and says so -- those are
    reviewed and versioned with the code, and copying one under a new name is
    the way to change it here.
    """
    return await container.builder_service.draft(principal, name)


@router.delete("/workflows/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def withdraw(
    name: str, principal: CurrentPrincipal, container: ContainerDep
) -> None:
    """Withdraw a definition. Workflows already running are unaffected."""
    await container.builder_service.delete(principal, name)

"""Operations endpoints: dead letters, replay, platform inventory.

This is the surface behind the ops dashboard. Everything here requires the
operator permission, because it lets a human reach into the execution plane.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.api.schemas import ToolMetadataView
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.envelope import Message
from cortexflow.shared.errors import NotFoundError
from cortexflow.shared.security.rbac import Permission, require_permission

router = APIRouter(prefix="/ops", tags=["operations"])


@router.get("/dead-letters", response_model=list[dict])
async def list_dead_letters(
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    """Messages that exhausted their retries, with the context to diagnose them."""
    require_permission(principal, Permission.OPS_READ)
    return await container.dead_letters.list(principal.tenant_id, limit=limit)


@router.post("/dead-letters/{entry_id}/replay")
async def replay_dead_letter(
    entry_id: str, principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, str]:
    """Resubmit a dead-lettered message after the cause has been fixed.

    Safe to do: idempotency keys mean a replay of partially-applied work
    converges rather than duplicating.
    """
    require_permission(principal, Permission.OPS_MANAGE)
    entry = await container.dead_letters.get(principal.tenant_id, entry_id)
    if entry is None:
        raise NotFoundError("Dead letter entry not found", entry_id=entry_id)

    message = Message.from_transport(entry["message"])
    queue = Queue(entry["queue"])
    await container.bus.publish(queue, message)
    await container.dead_letters.discard(principal.tenant_id, entry_id)
    return {"status": "replayed", "entry_id": entry_id, "queue": queue.value}


@router.delete("/dead-letters/{entry_id}")
async def discard_dead_letter(
    entry_id: str, principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, str]:
    require_permission(principal, Permission.OPS_MANAGE)
    await container.dead_letters.discard(principal.tenant_id, entry_id)
    return {"status": "discarded", "entry_id": entry_id}


@router.get("/tools", response_model=list[ToolMetadataView])
async def list_tools(
    principal: CurrentPrincipal, container: ContainerDep
) -> list[ToolMetadataView]:
    """The tool registry: what exists, how risky it is, and who may call it."""
    require_permission(principal, Permission.DEFINITION_READ)
    return [
        ToolMetadataView(
            name=m.name,
            domain=str(m.domain),
            description=m.description,
            side_effect=str(m.side_effect),
            risk=str(m.risk),
            requires_human_approval=m.requires_human_approval,
            allowed_roles=[str(r) for r in m.allowed_roles],
            exposed_to_agents=m.exposed_to_agents,
        )
        for m in container.tools.list_metadata()
    ]


@router.get("/agents", response_model=list[dict])
async def list_agents(
    principal: CurrentPrincipal, container: ContainerDep
) -> list[dict[str, str]]:
    require_permission(principal, Permission.DEFINITION_READ)
    return container.agents.describe()


@router.get("/policies", response_model=list[dict])
async def list_policies(
    principal: CurrentPrincipal, container: ContainerDep
) -> list[dict[str, Any]]:
    """The rulesets that hold the platform's thresholds, readable by design."""
    require_permission(principal, Permission.DEFINITION_READ)
    return [
        {
            "name": ruleset.name,
            "version": ruleset.version,
            "description": ruleset.description,
            "default_effect": str(ruleset.default_effect),
            "rules": [
                {
                    "id": rule.id,
                    "description": rule.description,
                    "when": rule.when,
                    "effect": str(rule.effect),
                    "risk": str(rule.risk),
                    "required_roles": [str(r) for r in rule.required_roles],
                }
                for rule in ruleset.rules
            ],
        }
        for ruleset in container.policy.rulesets.values()
    ]

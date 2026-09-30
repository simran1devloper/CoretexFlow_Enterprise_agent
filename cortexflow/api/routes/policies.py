"""Policy ruleset endpoints.

The administrative counterpart to ``/builder``. A workflow says what happens;
a ruleset says what is allowed to happen, and these are the routes that let an
administrator change the second without shipping a release.

Shaped like the builder's on purpose -- a lenient ``/validate`` for live
feedback while editing, a strict ``POST`` to publish, a ``DELETE`` that
reverts -- because it is the same job done by the same person, and a second
vocabulary for it would be one more thing to learn.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, status

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("")
async def list_rulesets(
    principal: CurrentPrincipal, container: ContainerDep
) -> list[dict[str, Any]]:
    """Every ruleset in force for this tenant, shipped and authored alike.

    One list rather than two: the question is "what authorizes work here?",
    and answering it with two lists plus a precedence rule to apply in your
    head is how someone ends up editing the copy that is being shadowed.
    """
    return container.policy_service.list(principal)


@router.post("/validate")
async def validate_ruleset(
    principal: CurrentPrincipal,
    container: ContainerDep,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Check a draft without saving it.

    The body is deliberately untyped: the editor calls this on every edit and
    a draft mid-edit is routinely unfinished. A 422 whose whole content is
    "unprocessable" is the one thing this endpoint exists not to say -- the
    guard that does not parse comes back in ``problems`` instead, named by the
    rule it belongs to.
    """
    return container.policy_service.validate(principal, payload)


@router.post("/try")
async def try_ruleset(
    principal: CurrentPrincipal,
    container: ContainerDep,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """What this draft would decide about one case.

    Body: ``{"ruleset": {...}, "facts": {...}}``. Evaluated by the real
    engine against an unregistered copy -- nothing is saved, nothing is
    registered, and no concurrent workflow is judged differently because
    somebody was trying a rule out.
    """
    return container.policy_service.try_out(
        principal,
        payload.get("ruleset") or {},
        payload.get("facts") or {},
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def publish_ruleset(
    principal: CurrentPrincipal,
    container: ContainerDep,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Publish a ruleset for this tenant, and make it take effect now.

    Where it takes the name of a shipped ruleset it shadows it for this tenant
    only; the file on disk is untouched and every other tenant still gets it.
    The write is audited with who made it and what moved.
    """
    saved = await container.policy_service.save(principal, payload)
    return saved.model_dump(mode="json")


@router.get("/{name}")
async def get_ruleset(
    name: str, principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, Any]:
    """One ruleset in full, with whatever it shadows alongside it."""
    return container.policy_service.get(principal, name)


@router.delete("/{name}")
async def withdraw_ruleset(
    name: str, principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, Any]:
    """Drop this tenant's ruleset.

    Returns what is in force afterwards, because reverting to a shipped
    ruleset and deleting the only one that existed are very different things
    to have just done, and a 204 cannot tell them apart.
    """
    return await container.policy_service.delete(principal, name)

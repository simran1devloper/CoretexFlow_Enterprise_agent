"""Who may call each tool, answered by asking the enforcement path.

A page that lists tool permissions has two ways to be built. It can read
`allowed_roles` and render it, which is a description of one field. Or it can
ask `authorize()` the same question a real call asks, once per role, and
report what actually comes back.

This does the second. The difference is not pedantry -- `allowed_roles` is only
the middle of three gates:

1. **Department.** ``can_access_department(tool.domain)``. This is why the
   Employee role, which holds ``workflow:create``, cannot start any shipped
   workflow: it belongs to no department. A permission is necessary, not
   sufficient.
2. **Role.** ``allowed_roles``, with two twists a flat table hides -- a
   platform admin bypasses this gate (but not the first), and a service agent
   is added to the permitted set *for reads only*, which is the line that
   stops a worker escalating itself into a payment.
3. **Agent exposure.** ``exposed_to_agents: False`` is not "this role may
   not". It is "no model is ever offered this", so there is no prompt that
   talks an agent into calling it.

Rendering `allowed_roles` alone would have shown `finance.initiate_payment` as
available to a finance manager and said nothing about the department they must
also be in, nor that no agent can reach it at all.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.tool_registry.domain.models import ToolMetadata
from cortexflow.shared.identity import Department, Principal, Role
from cortexflow.shared.security.auth import departments_for_roles


class Authorizer(Protocol):
    """The part of the registry this read model needs.

    Declared structurally so the read model depends on the question, not on
    the registry class -- and so a test can answer it directly.
    """

    def list_metadata(self) -> list[ToolMetadata]: ...

    def authorize(self, name: str, principal: Principal) -> object: ...


class RoleVerdict(BaseModel):
    """What happens when this role tries this tool."""

    model_config = ConfigDict(frozen=True)

    role: str
    allowed: bool
    reason: str = ""
    """Empty when allowed. Otherwise the gate that stopped it, verbatim."""


class ToolAccess(BaseModel):
    """One tool, with the real verdict for every role."""

    model_config = ConfigDict(frozen=True)

    name: str
    domain: str
    description: str
    side_effect: str
    risk: str
    requires_human_approval: bool
    exposed_to_agents: bool

    declared_roles: tuple[str, ...] = ()
    """``allowed_roles`` as written. Empty means "any principal in the domain"."""

    verdicts: tuple[RoleVerdict, ...] = ()
    allowed_roles: tuple[str, ...] = ()
    """Roles that actually pass all the gates -- not the declared list."""

    idempotent: bool = True


class ToolAccessReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    tools: tuple[ToolAccess, ...] = ()

    total: int = 0
    blocked_from_agents: int = 0
    """Tools no model is ever offered. The platform's strongest guarantee."""

    high_risk: int = 0
    irreversible: int = 0
    role_restricted: int = 0

    notes: tuple[str, ...] = ()
    """What the numbers do and do not mean, carried with them."""


NOTES = (
    "A tool with no roles listed is open to any principal whose role reaches "
    "its department; the department gate still applies.",
    "Tools blocked from agents are not merely refused on call -- they are "
    "never described to a model, so no prompt can reach them.",
    "No tool sets requires_human_approval: approval is raised by a workflow's "
    "approval step, which is where the decision belongs, not per tool call.",
)


def tool_access(registry: Authorizer) -> ToolAccessReport:
    """Ask the registry, once per role, what it would actually decide."""
    rows: list[ToolAccess] = []

    for meta in sorted(registry.list_metadata(), key=lambda m: m.name):
        verdicts = tuple(
            _verdict(registry, meta.name, role) for role in Role
        )
        rows.append(
            ToolAccess(
                name=meta.name,
                domain=str(meta.domain),
                description=meta.description,
                side_effect=str(meta.side_effect),
                risk=str(meta.risk),
                requires_human_approval=meta.requires_human_approval,
                exposed_to_agents=meta.exposed_to_agents,
                declared_roles=tuple(str(r) for r in meta.allowed_roles),
                verdicts=verdicts,
                allowed_roles=tuple(v.role for v in verdicts if v.allowed),
                idempotent=meta.idempotent,
            )
        )

    return ToolAccessReport(
        tools=tuple(rows),
        total=len(rows),
        blocked_from_agents=sum(1 for t in rows if not t.exposed_to_agents),
        high_risk=sum(1 for t in rows if t.risk in {"HIGH", "CRITICAL"}),
        irreversible=sum(1 for t in rows if t.side_effect == "IRREVERSIBLE"),
        role_restricted=sum(1 for t in rows if t.declared_roles),
        notes=NOTES,
    )


# ----------------------------------------------------------------------
def _verdict(registry: Authorizer, tool: str, role: Role) -> RoleVerdict:
    decision = registry.authorize(tool, _principal_for(role))
    allowed = bool(getattr(decision, "allowed", False))
    return RoleVerdict(
        role=str(role),
        allowed=allowed,
        reason="" if allowed else str(getattr(decision, "reason", "")),
    )


def _principal_for(role: Role) -> Principal:
    """A principal carrying exactly this one role, and nothing else.

    Departments come from ``departments_for_roles`` rather than being granted
    wholesale, because the department gate is the first thing ``authorize``
    checks and handing this principal every department would quietly answer a
    different question than the one the page asks.
    """
    roles = frozenset({role})
    departments: frozenset[Department] = departments_for_roles(roles)
    if role is Role.SERVICE_AGENT:
        # A worker is built by the runtime with the workflow's reach, not from
        # a token, so deriving its departments from claims would understate it.
        departments = frozenset(Department)
    return Principal(
        subject=f"role::{role}",
        tenant_id="*",
        display_name=str(role),
        roles=roles,
        departments=departments,
    )

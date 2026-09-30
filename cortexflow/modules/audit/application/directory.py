"""Who has access here, and who has actually used it.

Two questions that look like one page and are answered from two very
different kinds of evidence:

**What a role may do** comes from ``shared/security/rbac.py``. That table is
not a description of access control -- it *is* the access control, consulted by
``require_permission`` on every request. Reporting it is therefore exact.

**Who exists** is not this platform's to answer. Accounts live in Microsoft
Entra ID; a ``Principal`` is derived from a validated token per request and
nothing is stored. There is no roster, no account state and no session, so
there is no "last login" to report -- nothing logs in.

What there is instead is the audit trail, which records every actor that ever
did anything, the roles it carried at the time, and when. That yields a
directory of principals *with evidence* rather than a roster: not when somebody
signed in, but what they did and when they last did it -- which is the question
an administrator reviewing access actually has.

What is therefore absent, and why, travels in ``unknown`` on the response, so a
client showing these figures also has the caveat.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.audit.domain.models import AuditAction, AuditEvent
from cortexflow.shared.identity import Department, Role
from cortexflow.shared.security.auth import departments_for_roles
from cortexflow.shared.security.rbac import ROLE_PERMISSIONS, Permission

# Business language for a permission, so the UI's checklist reads the way
# somebody asking "why is this greyed out for me?" would phrase it.
PERMISSION_LABELS: dict[Permission, str] = {
    Permission.WORKFLOW_CREATE: "Start a workflow",
    Permission.WORKFLOW_READ: "View workflows and runs",
    Permission.WORKFLOW_CANCEL: "Cancel a running workflow",
    Permission.WORKFLOW_REPLAY: "Replay a failed workflow",
    Permission.APPROVAL_READ: "See approval requests",
    Permission.APPROVAL_DECIDE: "Decide an approval",
    Permission.AUDIT_READ: "Read the audit trail",
    Permission.OPS_READ: "See dead letters and platform health",
    Permission.OPS_MANAGE: "Act on dead letters and force cancellation",
    Permission.DEFINITION_READ: "View workflow definitions",
    Permission.DEFINITION_MANAGE: "Publish a workflow",
}

# A role holding any of these can change what the platform does to the world,
# rather than only asking it for something.
_PRIVILEGED = frozenset(
    {Permission.DEFINITION_MANAGE, Permission.OPS_MANAGE, Permission.APPROVAL_DECIDE}
)

UNKNOWN = (
    "Account state (active, disabled, invited) lives in Microsoft Entra ID; "
    "this platform validates tokens and never stores an account.",
    "There is no 'last login' because there is no session -- authentication is "
    "per request. Last activity below is what the audit trail observed.",
    "Anyone who has never acted is absent: this is a record of use, not a "
    "roster. A roster needs the Microsoft Graph directory API.",
)


class PermissionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str


class RoleAccess(BaseModel):
    """One role, and exactly what it may do."""

    model_config = ConfigDict(frozen=True)

    role: str
    permissions: tuple[str, ...]
    departments: tuple[str, ...]
    """Departments this role can reach, derived rather than assigned."""

    can_publish: bool
    can_decide_approvals: bool
    is_privileged: bool
    is_human_role: bool = True
    """False for ``service_agent``: a worker, not a person.

    Its ``departments`` read empty because no token ever carries this role --
    a worker principal is constructed directly by the runtime
    (``Principal.service``) with access to every department. The empty list is
    therefore accurate about the claim path, not about what a worker can reach.
    """


class AccessModel(BaseModel):
    """The authoritative role/permission matrix, as the server enforces it."""

    model_config = ConfigDict(frozen=True)

    roles: tuple[RoleAccess, ...]
    permissions: tuple[PermissionInfo, ...]
    departments: tuple[str, ...]


class DirectoryEntry(BaseModel):
    """One principal the audit trail has seen act."""

    model_config = ConfigDict(frozen=True)

    subject: str
    display_name: str
    kind: str
    """``human``, ``agent`` or ``system``, from the actor namespace."""

    roles: tuple[str, ...]
    """The roles this actor carried, as recorded at the time it acted."""

    departments: tuple[str, ...]
    permissions: tuple[str, ...]

    actions: int
    workflows_started: int
    approvals_decided: int
    tools_denied: int
    """Times this principal asked for a tool it was not allowed to call.

    Counted from ``TOOL_DENIED`` only. A ``POLICY_DENIED`` event looks like a
    refusal and is not one -- it is the rules deciding a claim, which is the
    system working. Folding the two together would make a duplicate expense
    look like an access problem.
    """

    first_seen: str
    last_seen: str


class AccessDirectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_days: int | None = None
    events_examined: int
    entries: tuple[DirectoryEntry, ...] = ()
    humans: int = 0
    services: int = 0
    roles_in_use: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()


# ----------------------------------------------------------------------
def access_model() -> AccessModel:
    """Read ``ROLE_PERMISSIONS`` as a client-facing model.

    Served rather than mirrored in the frontend: a hand-copied table drifts,
    and a UI that disagrees with the server about what somebody may do is a UI
    that hides buttons that work and offers ones that do not.
    """
    roles = []
    for role in Role:
        granted = ROLE_PERMISSIONS.get(role, frozenset())
        roles.append(
            RoleAccess(
                role=str(role),
                permissions=tuple(sorted(str(p) for p in granted)),
                departments=tuple(sorted(str(d) for d in _departments_for(role))),
                can_publish=Permission.DEFINITION_MANAGE in granted,
                can_decide_approvals=Permission.APPROVAL_DECIDE in granted,
                is_privileged=bool(granted & _PRIVILEGED),
                is_human_role=role is not Role.SERVICE_AGENT,
            )
        )

    # Most-privileged first: the rows that matter for an access review.
    roles.sort(key=lambda r: (-len(r.permissions), r.role))
    return AccessModel(
        roles=tuple(roles),
        permissions=tuple(
            PermissionInfo(key=str(p), label=PERMISSION_LABELS.get(p, str(p)))
            for p in Permission
        ),
        departments=tuple(str(d) for d in Department),
    )


def directory(
    events: list[AuditEvent], *, window_days: int | None = None
) -> AccessDirectory:
    """Fold audit events into one entry per principal."""
    by_subject: dict[str, list[AuditEvent]] = defaultdict(list)
    for event in events:
        by_subject[event.actor].append(event)

    entries = [
        _entry(subject, subject_events)
        for subject, subject_events in by_subject.items()
    ]
    # Busiest first, then by name, so the list is stable between refreshes.
    entries.sort(key=lambda e: (-e.actions, e.subject))

    roles_in_use = sorted({role for e in entries for role in e.roles})
    return AccessDirectory(
        window_days=window_days,
        events_examined=len(events),
        entries=tuple(entries),
        humans=sum(1 for e in entries if e.kind == "human"),
        services=sum(1 for e in entries if e.kind != "human"),
        roles_in_use=tuple(roles_in_use),
        unknown=UNKNOWN,
    )


# ----------------------------------------------------------------------
def _entry(subject: str, events: list[AuditEvent]) -> DirectoryEntry:
    roles = _roles_seen(events)
    granted: set[Permission] = set()
    departments: set[Department] = set()
    for role in roles:
        granted |= ROLE_PERMISSIONS.get(role, frozenset())
        departments |= _departments_for(role)

    moments = sorted(e.occurred_at for e in events)
    return DirectoryEntry(
        subject=subject,
        display_name=_display_name(subject),
        kind=_kind(subject),
        roles=tuple(sorted(str(r) for r in roles)),
        departments=tuple(sorted(str(d) for d in departments)),
        permissions=tuple(sorted(str(p) for p in granted)),
        actions=len(events),
        workflows_started=sum(
            1 for e in events if e.action is AuditAction.WORKFLOW_CREATED
        ),
        approvals_decided=sum(
            1
            for e in events
            if e.action
            in {AuditAction.APPROVAL_GRANTED, AuditAction.APPROVAL_REJECTED}
        ),
        tools_denied=sum(1 for e in events if e.action is AuditAction.TOOL_DENIED),
        first_seen=_iso(moments[0]),
        last_seen=_iso(moments[-1]),
    )


def _roles_seen(events: list[AuditEvent]) -> set[Role]:
    """Every role this actor was recorded as carrying.

    Taken from the events rather than from a stored profile, because the
    trail is what actually happened; an unrecognised claim is dropped the same
    way the authenticator drops it.
    """
    roles: set[Role] = set()
    for event in events:
        for raw in event.actor_roles:
            try:
                roles.add(Role(str(raw)))
            except ValueError:
                continue
    return roles


def _departments_for(role: Role) -> set[Department]:
    """Delegate to the authenticator's own derivation.

    Written as a mirror of ``DEPARTMENT_ROLES`` first, and a test immediately
    caught the two disagreeing about ``service_agent``. That is the whole
    argument of this module in miniature: a page that reports access must read
    the rule that grants it, not a copy that looks the same today.
    """
    return set(departments_for_roles(frozenset({role})))


# Actor namespaces, as the trail writes them. ``service::step::validate_employee``
# is a worker executing one step; ``system::orchestrator`` is the engine itself.
_KINDS = {"user": "human", "agent": "agent", "service": "service", "system": "system"}


def _kind(subject: str) -> str:
    prefix, sep, _ = subject.partition("::")
    if not sep:
        return "human"
    return _KINDS.get(prefix, "unknown")


def _display_name(subject: str) -> str:
    """``user::priya.nair`` reads as "Priya Nair".

    The trail stores a subject, not a profile. Presenting it as a name is a
    formatting choice, not a claim about the directory -- the subject stays on
    the record beside it.
    """
    _, _, local = subject.rpartition("::")
    local = local or subject
    cleaned = local.replace(".", " ").replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return subject
    return " ".join(part[:1].upper() + part[1:] for part in cleaned.split())


def _iso(moment: datetime) -> str:
    return moment.isoformat()

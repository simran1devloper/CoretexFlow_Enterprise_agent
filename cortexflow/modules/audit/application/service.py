"""Access reporting: the role matrix, and who has used it.

Read-only. Nothing here grants, revokes or creates anything -- role assignment
is a directory operation in Entra ID, and a second store of identities here
would compete with it for the truth.
"""

from __future__ import annotations

from datetime import timedelta

from cortexflow.modules.audit.application.directory import (
    AccessDirectory,
    AccessModel,
    DirectoryEntry,
    access_model,
    directory,
)
from cortexflow.modules.audit.domain.models import AuditEvent
from cortexflow.modules.audit.ports.repositories import AuditRepository
from cortexflow.shared.clock import Clock
from cortexflow.shared.errors import NotFoundError
from cortexflow.shared.identity import Principal
from cortexflow.shared.security.rbac import Permission, require_permission

MAX_EVENTS = 2000


class AccessService:
    def __init__(self, *, audit: AuditRepository, clock: Clock) -> None:
        self._audit = audit
        self._clock = clock

    def model(self, principal: Principal) -> AccessModel:
        """The role/permission matrix as the server enforces it.

        Readable by anyone who may read a workflow, deliberately: a person
        needs to see why an action is refused them, and the matrix is not
        sensitive -- it is the published contract of the platform.
        """
        require_permission(principal, Permission.WORKFLOW_READ)
        return access_model()

    async def people(
        self, principal: Principal, *, days: int | None = None, limit: int = MAX_EVENTS
    ) -> AccessDirectory:
        """Principals the audit trail has seen act in this tenant.

        Needs ``audit:read``: this is the audit trail aggregated, and the same
        permission that governs reading an event governs reading a summary of
        every event. Nothing is aggregated into being less sensitive.
        """
        require_permission(principal, Permission.AUDIT_READ)
        since = (
            self._clock.now() - timedelta(days=days) if days is not None else None
        )
        events = await self._audit.query(
            principal.tenant_id, since=since, limit=limit
        )
        return directory(events, window_days=days)

    async def activity(
        self, principal: Principal, subject: str, *, limit: int = 100
    ) -> list[AuditEvent]:
        """What this principal actually did, newest first.

        The aggregate counts answer "how much"; this answers "what", which is
        the question an access review turns into as soon as a number looks
        wrong.
        """
        require_permission(principal, Permission.AUDIT_READ)
        events = await self._audit.query(
            principal.tenant_id, actor=subject, limit=limit
        )
        return sorted(events, key=lambda e: e.occurred_at, reverse=True)

    async def person(
        self, principal: Principal, subject: str, *, limit: int = 200
    ) -> DirectoryEntry:
        """One principal, from its own events."""
        require_permission(principal, Permission.AUDIT_READ)
        events = await self._audit.query(
            principal.tenant_id, actor=subject, limit=limit
        )
        if not events:
            raise NotFoundError(
                "No recorded activity for this principal",
                subject=subject,
            )
        return directory(events).entries[0]

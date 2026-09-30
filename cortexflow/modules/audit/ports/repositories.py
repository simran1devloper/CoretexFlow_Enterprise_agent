"""The audit module's persistence port.

Every method is tenant-scoped. Isolation is not a convention the callers must
remember -- it is in the signature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from cortexflow.modules.audit.domain.models import AuditEvent


@runtime_checkable
class AuditRepository(Protocol):
    """Append-only audit sink."""

    async def append(self, event: AuditEvent) -> None: ...

    async def append_many(self, events: list[AuditEvent]) -> None: ...

    async def query(
        self,
        tenant_id: str,
        *,
        workflow_id: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[AuditEvent]: ...

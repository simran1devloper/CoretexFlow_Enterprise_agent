"""The approval module's persistence port.

Every method is tenant-scoped. Isolation is not a convention the callers must
remember -- it is in the signature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from cortexflow.modules.approval.domain.models import Approval


@runtime_checkable
class ApprovalRepository(Protocol):
    async def create(self, approval: Approval) -> Approval: ...

    async def get(self, tenant_id: str, approval_id: str) -> Approval: ...

    async def save(self, approval: Approval, *, expected_version: int) -> Approval: ...

    async def list_pending(
        self,
        tenant_id: str,
        *,
        workflow_id: str | None = None,
        limit: int = 50,
    ) -> list[Approval]: ...

    async def find_expired(self, *, now: datetime, limit: int = 100) -> list[Approval]: ...

    async def cancel_for_workflow(
        self, tenant_id: str, workflow_id: str, *, reason: str
    ) -> int:
        """Terminate outstanding approvals when their workflow ends early."""
        ...

"""Blob storage port.

Documents (receipts, contracts, generated reports) never travel through
messages or land in workflow state -- only their references do.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# BlobRef is a domain type, re-exported here so adapters can import the port
# and its data type together. Defining it here instead would make the domain
# depend on the ports layer, which the dependency rule forbids -- and which
# produced a circular import the moment a domain model referenced it.
from cortexflow.modules.document.domain.storage import BlobRef

__all__ = ["BlobRef", "BlobStore"]


@runtime_checkable
class BlobStore(Protocol):
    async def put(
        self,
        *,
        tenant_id: str,
        path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> BlobRef: ...

    async def get(self, ref: BlobRef) -> bytes: ...

    async def delete(self, ref: BlobRef) -> None: ...

    async def signed_url(self, ref: BlobRef, *, expires_seconds: int = 900) -> str:
        """Short-lived read URL so the dashboard never proxies document bytes."""
        ...

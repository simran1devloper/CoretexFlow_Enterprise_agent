"""The document module's persistence port.

Every method is tenant-scoped. Isolation is not a convention the callers must
remember -- it is in the signature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cortexflow.modules.document.domain.models import Document

DocumentList = list[Document]


@runtime_checkable
class DocumentRepository(Protocol):
    """Uploaded documents and their parsed projections."""

    async def create(self, document: Document) -> Document: ...

    async def get(self, tenant_id: str, document_id: str) -> Document:
        """Raises NotFoundError when absent."""
        ...

    async def list(self, tenant_id: str, *, limit: int = 50) -> DocumentList: ...

    async def delete(self, tenant_id: str, document_id: str) -> None: ...

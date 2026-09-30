"""Document ingestion.

Uploading is a control-plane action: store the bytes, parse a bounded
projection, return something the builder can show. The original never enters
workflow state -- only a :class:`BlobRef` and the projection do.
"""

from __future__ import annotations

from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.document.application.parsers import parse
from cortexflow.modules.document.domain.models import Document
from cortexflow.modules.document.ports.repositories import DocumentRepository
from cortexflow.modules.document.ports.storage import BlobStore
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.identity import Principal
from cortexflow.shared.ids import DOCUMENT, IdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.security.rbac import Permission, require_permission, require_tenant

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
"""A hard ceiling. Parsing bounds what agents see; this bounds what we accept."""


class DocumentService:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        blobs: BlobStore,
        audit: AuditTrail,
        ids: IdGenerator,
    ) -> None:
        self._documents = documents
        self._blobs = blobs
        self._audit = audit
        self._ids = ids

    async def upload(
        self,
        principal: Principal,
        *,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> Document:
        """Store and parse an uploaded file."""
        require_permission(principal, Permission.WORKFLOW_CREATE)

        if not data:
            raise ValidationError("The file is empty", filename=filename)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValidationError(
                "The file is too large",
                filename=filename,
                size_bytes=len(data),
                limit_bytes=MAX_UPLOAD_BYTES,
            )

        document_id = self._ids.new_id(DOCUMENT)
        parsed = parse(data, filename=filename, content_type=content_type)

        blob = await self._blobs.put(
            tenant_id=principal.tenant_id,
            path=f"uploads/{document_id}/{_safe_name(filename)}",
            data=data,
            content_type=content_type or "application/octet-stream",
        )

        document = await self._documents.create(
            Document(
                document_id=document_id,
                tenant_id=principal.tenant_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(data),
                uploaded_by=principal.subject,
                blob=blob,
                parsed=parsed,
            )
        )

        logger.info(
            "document uploaded",
            extra={
                "document_id": document_id,
                "kind": str(parsed.kind),
                "rows": parsed.row_count,
                "size_bytes": len(data),
            },
        )
        await self._audit.record(
            tenant_id=principal.tenant_id,
            action=AuditAction.WORKFLOW_CREATED,
            actor=principal.subject,
            summary=f"Uploaded {filename}",
            metadata={
                "document_id": document_id,
                "kind": str(parsed.kind),
                "rows": parsed.row_count,
                # Filenames and column headers can carry personal data, so the
                # audit metadata records shape, not content.
                "columns": len(parsed.columns),
            },
        )
        return document

    async def get(self, principal: Principal, document_id: str) -> Document:
        require_permission(principal, Permission.WORKFLOW_READ)
        document = await self._documents.get(principal.tenant_id, document_id)
        require_tenant(principal, document.tenant_id)
        return document

    async def list(self, principal: Principal, *, limit: int = 50) -> list[Document]:
        require_permission(principal, Permission.WORKFLOW_READ)
        return await self._documents.list(principal.tenant_id, limit=limit)

    async def delete(self, principal: Principal, document_id: str) -> None:
        require_permission(principal, Permission.WORKFLOW_CREATE)
        document = await self._documents.get(principal.tenant_id, document_id)
        require_tenant(principal, document.tenant_id)
        if document.blob is not None:
            await self._blobs.delete(document.blob)
        await self._documents.delete(principal.tenant_id, document_id)

    async def workflow_input(self, principal: Principal, document_id: str) -> dict:
        """Build the workflow input for a document.

        The text projection is what agents read; the statistics are what the
        deterministic tools and the policy engine read.
        """
        document = await self.get(principal, document_id)
        parsed = document.parsed
        return {
            "document_id": document.document_id,
            "document_text": parsed.text,
            "filename": document.filename,
            "row_count": parsed.row_count,
            "columns": [c.name for c in parsed.columns],
        }


def _safe_name(filename: str) -> str:
    """Keep a readable name without letting it escape the storage prefix."""
    cleaned = "".join(c if c.isalnum() or c in "-._" else "_" for c in filename)
    return cleaned[-120:] or "upload"

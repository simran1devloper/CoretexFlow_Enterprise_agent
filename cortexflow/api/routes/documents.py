"""Document endpoints.

Uploading is a control-plane action: the bytes go to blob storage, a bounded
projection comes back for the builder to show. Workflow state never holds the
file, only a reference and the projection.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Query, UploadFile, status

from cortexflow.api.middleware.dependencies import ContainerDep, CurrentPrincipal
from cortexflow.modules.document.application.service import MAX_UPLOAD_BYTES
from cortexflow.modules.document.domain.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


def _view(document: Document, *, preview_rows: int = 10) -> dict[str, Any]:
    """What the builder shows: shape, quality and a small sample."""
    parsed = document.parsed
    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "uploaded_at": document.uploaded_at.isoformat(),
        "uploaded_by": document.uploaded_by,
        "kind": str(parsed.kind),
        "row_count": parsed.row_count,
        "truncated": parsed.truncated,
        "warnings": list(parsed.warnings),
        "columns": [
            {
                "name": c.name,
                "type": c.inferred_type,
                "missing": c.empty,
                "distinct": c.distinct,
                "completeness": c.completeness,
                "examples": list(c.examples),
            }
            for c in parsed.columns
        ],
        "preview_rows": [dict(row) for row in parsed.rows[:preview_rows]],
        "text_preview": parsed.text[:2000],
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    principal: CurrentPrincipal,
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="xlsx, csv, json, pdf or text")],
) -> dict[str, Any]:
    """Upload and parse a file.

    Parsing is deterministic: row counts, missing values and column types are
    computed in code, so what the builder displays is exact rather than a
    model's impression.
    """
    data = await file.read()
    document = await container.document_service.upload(
        principal,
        filename=file.filename or "upload",
        content_type=file.content_type or "",
        data=data,
    )
    return _view(document)


@router.get("", response_model=list[dict])
async def list_documents(
    principal: CurrentPrincipal,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    documents = await container.document_service.list(principal, limit=limit)
    return [_view(d, preview_rows=3) for d in documents]


@router.get("/limits")
async def upload_limits(principal: CurrentPrincipal) -> dict[str, Any]:
    """What the uploader will accept, so the UI can say so up front."""
    return {
        "max_bytes": MAX_UPLOAD_BYTES,
        "accepted": [".xlsx", ".xlsm", ".csv", ".tsv", ".json", ".pdf", ".txt", ".md"],
    }


@router.get("/{document_id}")
async def get_document(
    document_id: str, principal: CurrentPrincipal, container: ContainerDep
) -> dict[str, Any]:
    document = await container.document_service.get(principal, document_id)
    return _view(document, preview_rows=50)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str, principal: CurrentPrincipal, container: ContainerDep
) -> None:
    await container.document_service.delete(principal, document_id)

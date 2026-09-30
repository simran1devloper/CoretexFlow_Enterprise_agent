"""Filesystem blob store for local development.

Tenant-prefixed paths mirror the container layout used in Azure Blob Storage,
so the same ``BlobRef`` works against either backend.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from cortexflow.modules.document.ports.storage import BlobRef
from cortexflow.shared.errors import NotFoundError, ValidationError


class FilesystemBlobStore:
    def __init__(self, root: Path, container: str = "documents") -> None:
        self._root = Path(root)
        self._container = container
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root.resolve()):
            raise ValidationError("Blob path escapes the storage root", path=path)
        return target

    async def put(
        self,
        *,
        tenant_id: str,
        path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> BlobRef:
        relative = f"{tenant_id}/{path.lstrip('/')}"
        target = self._resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)
        return BlobRef(
            container=self._container,
            path=relative,
            content_type=content_type,
            size_bytes=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
        )

    async def get(self, ref: BlobRef) -> bytes:
        target = self._resolve(ref.path)
        if not target.exists():
            raise NotFoundError("Blob not found", path=ref.path)
        return await asyncio.to_thread(target.read_bytes)

    async def delete(self, ref: BlobRef) -> None:
        target = self._resolve(ref.path)
        if target.exists():
            await asyncio.to_thread(target.unlink)

    async def signed_url(self, ref: BlobRef, *, expires_seconds: int = 900) -> str:
        return self._resolve(ref.path).as_uri()

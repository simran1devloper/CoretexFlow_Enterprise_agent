"""Azure Blob Storage adapter."""

from __future__ import annotations

import hashlib
from typing import Any

from cortexflow.config.settings import BlobSettings
from cortexflow.modules.document.ports.storage import BlobRef
from cortexflow.shared.errors import DependencyUnavailableError, NotFoundError


def build_blob_client(settings: BlobSettings) -> Any:
    from azure.identity.aio import DefaultAzureCredential
    from azure.storage.blob.aio import BlobServiceClient

    return BlobServiceClient(
        account_url=settings.account_url, credential=DefaultAzureCredential()
    )


class AzureBlobStore:
    def __init__(self, client: Any, settings: BlobSettings) -> None:
        self._client = client
        self._container = settings.container

    def _blob(self, path: str) -> Any:
        return self._client.get_blob_client(container=self._container, blob=path)

    async def put(
        self,
        *,
        tenant_id: str,
        path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> BlobRef:
        from azure.storage.blob import ContentSettings

        # Tenant prefix keeps document paths isolated and makes lifecycle
        # policies (retention, legal hold) expressible per tenant.
        relative = f"{tenant_id}/{path.lstrip('/')}"
        try:
            await self._blob(relative).upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
        except Exception as exc:  # pragma: no cover - requires a live account
            raise DependencyUnavailableError("Blob upload failed", path=relative) from exc
        return BlobRef(
            container=self._container,
            path=relative,
            content_type=content_type,
            size_bytes=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
        )

    async def get(self, ref: BlobRef) -> bytes:
        try:
            stream = await self._blob(ref.path).download_blob()
            return await stream.readall()
        except Exception as exc:  # pragma: no cover
            if "BlobNotFound" in str(exc):
                raise NotFoundError("Blob not found", path=ref.path) from exc
            raise DependencyUnavailableError("Blob download failed", path=ref.path) from exc

    async def delete(self, ref: BlobRef) -> None:
        try:
            await self._blob(ref.path).delete_blob()
        except Exception as exc:  # pragma: no cover
            if "BlobNotFound" not in str(exc):
                raise

    async def signed_url(self, ref: BlobRef, *, expires_seconds: int = 900) -> str:
        """User-delegation SAS: short-lived, identity-scoped, no account key."""
        from datetime import timedelta

        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        from cortexflow.shared.clock import utcnow

        start = utcnow()
        expiry = start + timedelta(seconds=expires_seconds)
        delegation_key = await self._client.get_user_delegation_key(start, expiry)
        token = generate_blob_sas(
            account_name=self._client.account_name,
            container_name=ref.container,
            blob_name=ref.path,
            user_delegation_key=delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            start=start,
        )
        return f"{self._client.url}{ref.container}/{ref.path}?{token}"

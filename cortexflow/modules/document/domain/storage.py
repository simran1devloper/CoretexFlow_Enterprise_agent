"""A reference to stored content.

This lives in the domain, not behind the storage port, because it is a
*domain* concept: workflow state holds references to documents, and the
domain must be able to describe one without knowing whether the bytes are in
Azure Blob Storage or on a local disk.

Keeping it here also keeps the dependency rule intact -- the domain imports
nothing outside itself, and ``ports.storage`` re-exports this for adapters.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BlobRef(BaseModel):
    """A pointer to stored content, safe to embed in workflow state."""

    model_config = ConfigDict(frozen=True)

    container: str
    path: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    checksum: str = ""

    @property
    def uri(self) -> str:
        return f"blob://{self.container}/{self.path}"

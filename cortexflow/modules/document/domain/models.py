"""Uploaded documents.

A document is the *input* to a workflow: a spreadsheet of employees, a PDF
receipt, a JSON export. Two things are kept about it:

* a :class:`BlobRef` to the original bytes, which never enter workflow state;
* a parsed, bounded projection that agents and tools can actually read.

The projection matters. An agent handed a 50,000-row spreadsheet would blow
its context window and its budget, so parsing produces a sample plus computed
statistics, and the deterministic analysis runs over the whole file in code.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.document.domain.storage import BlobRef
from cortexflow.shared.clock import utcnow


class DocumentKind(StrEnum):
    SPREADSHEET = "spreadsheet"
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    TEXT = "text"

    @property
    def is_tabular(self) -> bool:
        return self in {DocumentKind.SPREADSHEET, DocumentKind.CSV}


class NumericSummary(BaseModel):
    """Aggregates over a numeric column, computed across every row."""

    model_config = ConfigDict(frozen=True)

    minimum: float = 0.0
    maximum: float = 0.0
    total: float = 0.0
    mean: float = 0.0


class ColumnProfile(BaseModel):
    """What a column contains, computed deterministically."""

    model_config = ConfigDict(frozen=True)

    name: str
    non_empty: int = 0
    empty: int = 0
    distinct: int = 0
    inferred_type: str = "string"
    examples: tuple[str, ...] = ()

    numeric: NumericSummary | None = None
    """Present for numeric columns. Exact, not sampled."""

    value_counts: dict[str, int] = Field(default_factory=dict)
    """How often each value occurs, for low-cardinality columns.

    This is what lets a policy gate read a flag column -- "how many rows are
    marked possible_duplicate" -- as an exact number rather than an agent's
    impression of a sample.
    """

    @property
    def completeness(self) -> float:
        total = self.non_empty + self.empty
        return round(self.non_empty / total, 4) if total else 0.0


class ParsedDocument(BaseModel):
    """The bounded projection of an uploaded file."""

    kind: DocumentKind
    text: str = ""
    """A textual rendering, for agents that read prose."""

    columns: tuple[ColumnProfile, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    """A *sample* of rows, not the whole file."""

    row_count: int = 0
    """The true total, even when only a sample is retained."""

    truncated: bool = False
    warnings: tuple[str, ...] = ()


class Document(BaseModel):
    """An uploaded document and its parsed projection."""

    document_id: str
    tenant_id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_by: str = ""
    uploaded_at: datetime = Field(default_factory=utcnow)

    blob: BlobRef | None = None
    parsed: ParsedDocument

    labels: dict[str, str] = Field(default_factory=dict)

    @property
    def partition_key(self) -> str:
        return self.tenant_id

    def summary(self) -> dict[str, Any]:
        """A compact description, safe to embed in workflow input."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "kind": str(self.parsed.kind),
            "row_count": self.parsed.row_count,
            "columns": [c.name for c in self.parsed.columns],
        }

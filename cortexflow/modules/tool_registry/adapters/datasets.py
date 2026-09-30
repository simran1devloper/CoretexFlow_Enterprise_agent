"""Dataset tools.

The same principle as ``finance.find_duplicate_expense``: counting rows,
finding blanks and spotting duplicates is arithmetic over recorded data, so it
is code. An agent's job is to interpret what the numbers *mean* for this case,
not to compute them.

That division is what makes the spreadsheet example honest. The issue counts
the policy engine sees are measured over every row; the agent never sees more
than a sample and could not count them if it tried.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.document.application.parsers import MAX_TRIAGE_ROWS, coerce, read_rows
from cortexflow.modules.document.domain.models import Document
from cortexflow.modules.document.ports.repositories import DocumentRepository
from cortexflow.modules.document.ports.storage import BlobStore
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.errors import NotFoundError, ValidationError
from cortexflow.shared.expressions import evaluate_condition, validate_expression
from cortexflow.shared.identity import Department, Principal

MAX_REPORTED_ISSUES = 50

MAX_CARRIED_ROWS = 500
"""The most rows a profile will carry when asked for them.

Matched to the fan-out ceiling: there is no point carrying more rows than a
step is allowed to act on.
"""


class AnalyseDatasetTool(Tool):
    """Profile an uploaded dataset and report its problems, deterministically."""

    class Args(BaseModel):
        document_id: str = Field(min_length=1, max_length=64)
        required_columns: list[str] = Field(default_factory=list)
        unique_columns: list[str] = Field(default_factory=list)

        include_rows: bool = False
        """Carry the rows themselves, not just the profile.

        Off by default: the profile is a summary that stays small whatever the
        file's size, and a workflow record is a document in a database. It is
        turned on only when a later step has to act on each row, and bounded
        even then.
        """

        max_rows: int = Field(default=MAX_CARRIED_ROWS, ge=1, le=MAX_CARRIED_ROWS)

    input_model = Args
    metadata = ToolMetadata(
        name="documents.analyse_dataset",
        domain=Department.PLATFORM,
        description=(
            "Profile an uploaded dataset: row count, missing values per column, "
            "duplicate keys, and which required columns are absent."
        ),
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
        allowed_roles=(),
    )

    def __init__(self, documents: DocumentRepository, blobs: BlobStore) -> None:
        self._documents = documents
        self._blobs = blobs

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        args = self.Args.model_validate(call.arguments)
        document = await self._documents.get(
            call.tenant_id or principal.tenant_id, args.document_id
        )
        analysis = analyse(
            document,
            required_columns=args.required_columns,
            unique_columns=args.unique_columns,
        )
        if args.include_rows:
            rows, total = await self._rows(document, args.max_rows)
            # Typed, unlike the profile's own sample. These rows exist to be
            # acted on -- compared against a threshold, matched to a rule --
            # and "18500" is not a number a rule can compare.
            analysis["rows"] = [
                {key: coerce(value) for key, value in row.items()} for row in rows
            ]
            analysis["rows_carried"] = len(rows)
            analysis["rows_complete"] = len(rows) >= total
        return ToolResult.ok(self.name, analysis)

    async def _rows(self, document: Document, limit: int) -> tuple[list[Any], int]:
        """Rows from the original file, not the stored sample."""
        parsed = document.parsed
        if not parsed.truncated or document.blob is None:
            return list(parsed.rows[:limit]), parsed.row_count
        data = await self._blobs.get(document.blob)
        rows, total = read_rows(
            data,
            filename=document.filename,
            content_type=document.content_type,
            limit=limit,
        )
        return rows, total or parsed.row_count


class ReadDocumentTool(Tool):
    """Return an uploaded document's text projection."""

    class Args(BaseModel):
        document_id: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="documents.read",
        domain=Department.PLATFORM,
        description="Read the text of an uploaded document.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, documents: DocumentRepository) -> None:
        self._documents = documents

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        try:
            document = await self._documents.get(
                call.tenant_id or principal.tenant_id, call.arguments["document_id"]
            )
        except NotFoundError:
            # A finding, not a failure: the workflow should reach the policy
            # gate and be denied there, with a reason a human can read.
            return ToolResult.ok(
                self.name,
                {"found": False, "reason": "No such document"},
            )
        return ToolResult.ok(
            self.name,
            {
                "found": True,
                "filename": document.filename,
                "kind": str(document.parsed.kind),
                "text": document.parsed.text,
                "row_count": document.parsed.row_count,
            },
        )


def analyse(
    document: Document,
    *,
    required_columns: list[str],
    unique_columns: list[str],
) -> dict[str, Any]:
    """Compute a dataset's quality facts over every row."""
    parsed = document.parsed
    present = {column.name for column in parsed.columns}

    missing_columns = [name for name in required_columns if name not in present]
    incomplete = {
        column.name: column.empty for column in parsed.columns if column.empty
    }

    duplicates: dict[str, list[str]] = {}
    for name in unique_columns:
        if name not in present:
            continue
        values = [
            str(row.get(name, "")).strip()
            for row in parsed.rows
            if str(row.get(name, "")).strip()
        ]
        repeated = [value for value, count in Counter(values).items() if count > 1]
        if repeated:
            duplicates[name] = repeated[:MAX_REPORTED_ISSUES]

    issues: list[str] = []
    issues.extend(f"Required column '{name}' is missing." for name in missing_columns)
    issues.extend(
        f"Column '{name}' has {count} missing value{'s' if count > 1 else ''}."
        for name, count in incomplete.items()
    )
    issues.extend(
        f"Column '{name}' has duplicate values: {', '.join(values[:5])}."
        for name, values in duplicates.items()
    )
    issues.extend(parsed.warnings)

    # Completeness is over the whole file; duplicate detection only covers the
    # retained sample, so say so rather than implying full coverage.
    completeness = (
        round(
            sum(c.completeness for c in parsed.columns) / len(parsed.columns), 4
        )
        if parsed.columns
        else 0.0
    )

    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "row_count": parsed.row_count,
        "column_count": len(parsed.columns),
        "columns": [
            {
                "name": c.name,
                "type": c.inferred_type,
                "missing": c.empty,
                "distinct": c.distinct,
                "completeness": c.completeness,
            }
            for c in parsed.columns
        ],
        # Exact aggregates over every row, so a policy gate can compare a
        # batch maximum against a threshold without a model in the loop.
        "numeric": {
            c.name: {
                "min": c.numeric.minimum,
                "max": c.numeric.maximum,
                "total": c.numeric.total,
                "mean": c.numeric.mean,
            }
            for c in parsed.columns
            if c.numeric is not None
        },
        "value_counts": {
            c.name: c.value_counts for c in parsed.columns if c.value_counts
        },
        "missing_columns": missing_columns,
        "incomplete_columns": incomplete,
        "duplicates": duplicates,
        "issues": issues[:MAX_REPORTED_ISSUES],
        "issue_count": len(issues),
        "clean": not issues,
        "completeness": completeness,
        "duplicate_scan_covered_rows": len(parsed.rows),
        "duplicate_scan_complete": not parsed.truncated,
    }


MAX_QUERIED_ROWS = 100
"""The most rows one query returns to an agent.

Bounded because the caller is a model with a context window and a per-token
price. An agent that needs more than a hundred examples to form a view does
not need more examples; it needs a different question.
"""


class QueryRowsTool(Tool):
    """Let an agent ask for the rows it needs, rather than be handed all of them.

    The alternative -- putting the dataset in the prompt -- is expensive,
    truncates arbitrarily, and invites the model to do arithmetic it is bad
    at. This keeps the division the platform already relies on: statistics
    come from code over every row, and the agent reads the handful of rows it
    asked about.

    The filter is the same whitelisted expression language used by workflow
    guards, evaluated per row. It cannot call anything, import anything or
    reach outside the row it is given.
    """

    class Args(BaseModel):
        document_id: str = Field(min_length=1, max_length=64)

        where: str = Field(default="", max_length=500)
        """A condition over ``row``, e.g. ``row.possible_duplicate == 'Yes'``."""

        columns: list[str] = Field(default_factory=list)
        """Only these columns, if given. Fewer columns, cheaper prompt."""

        limit: int = Field(default=20, ge=1, le=MAX_QUERIED_ROWS)

    input_model = Args
    metadata = ToolMetadata(
        name="documents.query_rows",
        domain=Department.PLATFORM,
        description=(
            "Fetch rows of an uploaded dataset matching a condition, e.g. "
            "\"row.possible_duplicate == 'Yes'\". Returns at most 100 rows. "
            "Use documents.analyse_dataset for counts and totals over every "
            "row -- this is for looking at specific cases, not for counting."
        ),
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
        timeout_seconds=60.0,
    )

    def __init__(self, documents: DocumentRepository, blobs: BlobStore) -> None:
        self._documents = documents
        self._blobs = blobs

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        args = self.Args.model_validate(call.arguments)
        document = await self._documents.get(
            call.tenant_id or principal.tenant_id, args.document_id
        )
        if not document.parsed.kind.is_tabular:
            raise ValidationError(
                "That document has no rows to query",
                document_id=args.document_id,
                kind=str(document.parsed.kind),
            )

        rows, total = await _all_rows(
            self._documents, self._blobs, document, MAX_TRIAGE_ROWS
        )
        typed = [{k: coerce(v) for k, v in row.items()} for row in rows]

        matched = _matching(typed, args.where)
        selected = matched[: args.limit]
        if args.columns:
            selected = [
                {c: row.get(c) for c in args.columns if c in row} for row in selected
            ]

        return ToolResult.ok(
            self.name,
            {
                "rows": selected,
                # Three numbers, because "20 of 340 matching, out of 5,000
                # scanned" and "20 of 20" are different answers and an agent
                # that cannot tell them apart will overstate what it saw.
                "returned": len(selected),
                "matched": len(matched),
                "scanned": total,
                "where": args.where,
                "truncated": len(matched) > len(selected),
            },
        )


def _matching(rows: list[dict[str, Any]], where: str) -> list[dict[str, Any]]:
    """Rows satisfying the condition, or all of them when there is none.

    A row whose values make the condition unevaluable is excluded rather than
    raising: a blank cell in one row of a thousand should narrow the answer,
    not fail the query.
    """
    if not where.strip():
        return rows
    validate_expression(where)
    return [row for row in rows if evaluate_condition(where, {"row": row})]


async def _all_rows(
    documents: DocumentRepository, blobs: BlobStore, document: Document, limit: int
) -> tuple[list[dict[str, Any]], int]:
    """Every row of a document, from the original file when the sample is short."""
    parsed = document.parsed
    if not parsed.truncated or document.blob is None:
        return list(parsed.rows[:limit]), parsed.row_count
    data = await blobs.get(document.blob)
    rows, total = read_rows(
        data,
        filename=document.filename,
        content_type=document.content_type,
        limit=limit,
    )
    return rows, total or parsed.row_count


def build_dataset_tools(
    documents: DocumentRepository, blobs: BlobStore
) -> list[Tool]:
    return [
        AnalyseDatasetTool(documents, blobs),
        ReadDocumentTool(documents),
        QueryRowsTool(documents, blobs),
    ]

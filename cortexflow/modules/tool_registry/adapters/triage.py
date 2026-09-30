"""Row-by-row policy triage.

A Policy Gate answers one question about one case. Pointed at a spreadsheet it
still answers one question -- about the whole file -- so a single flagged row
denies every row beside it, and nine clean claims are rejected with the tenth.

This tool asks the same ruleset once per row and reports what each row would
get. It classifies; it does not authorize. The Policy Gate remains the only
thing that can permit an action, and it can read these counts as facts:
"nothing is denied and fewer than five rows need a person" is a decision a
ruleset can make, where "the largest row in the file" was never really one.

Determinism matters here as much as anywhere: the same file and the same
thresholds must produce the same breakdown, so every value is read from the
recorded data and no model is consulted.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.document.application.parsers import MAX_TRIAGE_ROWS, coerce, read_rows
from cortexflow.modules.document.ports.repositories import DocumentRepository
from cortexflow.modules.document.ports.storage import BlobStore
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.policy.domain.models import PolicyEffect
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.identity import Department, Principal

MAX_REPORTED_ROWS = 500
"""How many per-row verdicts travel in the step output.

The counts always cover every row examined; only the itemised list is capped,
because a workflow record is read by people and stored in a document database.
"""

# Conservative combination, matching the policy engine's own ordering: one
# denial outranks any number of approvals.
_SEVERITY = {PolicyEffect.ALLOW: 0, PolicyEffect.REQUIRE_APPROVAL: 1, PolicyEffect.DENY: 2}
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class TriageRowsTool(Tool):
    """Evaluate a ruleset against every row of an uploaded dataset."""

    class Args(BaseModel):
        document_id: str = Field(min_length=1, max_length=64)
        ruleset: str = Field(min_length=1, max_length=128)

        row_facts: dict[str, str] = Field(default_factory=dict)
        """Fact name -> column name. Read afresh for each row."""

        facts: dict[str, Any] = Field(default_factory=dict)
        """Facts that are the same for every row: thresholds, budgets."""

        key_column: str = ""
        """Which column names a row in the report, e.g. ``expense_id``."""

        limit: int = Field(default=MAX_TRIAGE_ROWS, ge=1, le=MAX_TRIAGE_ROWS)

    input_model = Args
    metadata = ToolMetadata(
        name="policy.triage_rows",
        domain=Department.PLATFORM,
        description=(
            "Apply a policy ruleset to every row of an uploaded dataset and "
            "report which rows would be allowed, which need a person, and "
            "which would be denied."
        ),
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
        allowed_roles=(),
        timeout_seconds=120.0,
        # Orchestrator-only. An agent that could run this would be able to
        # probe the thresholds it is meant to be judged against.
        exposed_to_agents=False,
    )

    def __init__(
        self,
        documents: DocumentRepository,
        blobs: BlobStore,
        policy: PolicyEngine,
    ) -> None:
        self._documents = documents
        self._blobs = blobs
        self._policy = policy

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        args = self.Args.model_validate(call.arguments)
        tenant_id = call.tenant_id or principal.tenant_id
        document = await self._documents.get(tenant_id, args.document_id)

        if not document.parsed.kind.is_tabular:
            raise ValidationError(
                "Row triage needs a table",
                document_id=args.document_id,
                kind=str(document.parsed.kind),
                hint="Upload a CSV or spreadsheet, or use a Policy Gate instead.",
            )

        rows, total = await self._rows(document, args.limit)
        if not rows:
            raise ValidationError(
                "That file has no rows to judge", document_id=args.document_id
            )

        return ToolResult.ok(self.name, self._triage(args, rows, total, tenant_id))

    async def _rows(self, document: Any, limit: int) -> tuple[list[dict[str, Any]], int]:
        """All the rows, from the original file where the sample is short.

        The stored sample is enough when the file fit inside it; anything
        longer is re-read from the blob, because judging 50 of 2,000 rows and
        presenting it as the answer would be the very fault this tool exists
        to correct.
        """
        parsed = document.parsed
        if not parsed.truncated:
            return list(parsed.rows[:limit]), parsed.row_count

        if document.blob is None:  # pragma: no cover - blob is kept on upload
            return list(parsed.rows[:limit]), parsed.row_count

        data = await self._blobs.get(document.blob)
        rows, total = read_rows(
            data,
            filename=document.filename,
            content_type=document.content_type,
            limit=limit,
        )
        return rows, total or parsed.row_count

    def _triage(
        self, args: Args, rows: list[dict[str, Any]], total: int, tenant_id: str
    ) -> dict[str, Any]:
        shared = {name: coerce(value) for name, value in args.facts.items()}
        missing = sorted(
            {
                column
                for column in args.row_facts.values()
                if rows and column not in rows[0]
            }
        )

        verdicts: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        rule_counts: Counter[str] = Counter()
        reasons: dict[str, int] = {}
        worst = PolicyEffect.ALLOW
        worst_risk = RiskLevel.LOW

        for index, row in enumerate(rows):
            facts = dict(shared)
            for fact, column in args.row_facts.items():
                facts[fact] = coerce(row.get(column))

            decision = self._policy.evaluate(
                args.ruleset, {"facts": facts}, tenant_id=tenant_id
            )
            effect = decision.effect
            counts[str(effect)] += 1
            for match in decision.matched_rules:
                rule_counts[match.rule_id] += 1
            for reason in decision.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1

            if _SEVERITY[effect] > _SEVERITY[worst]:
                worst = effect
            if _RISK_ORDER[decision.risk] > _RISK_ORDER[worst_risk]:
                worst_risk = decision.risk

            if len(verdicts) < MAX_REPORTED_ROWS:
                verdicts.append(
                    {
                        "index": index + 1,
                        "key": str(row.get(args.key_column, "")) if args.key_column else "",
                        "effect": str(effect),
                        "risk": str(decision.risk),
                        "reasons": list(decision.reasons),
                        "matched_rules": [m.rule_id for m in decision.matched_rules],
                    }
                )

        examined = len(rows)
        return {
            "ruleset": args.ruleset,
            "total_rows": total,
            "examined_rows": examined,
            "complete": examined >= total,
            "allowed": counts.get(str(PolicyEffect.ALLOW), 0),
            "requires_approval": counts.get(str(PolicyEffect.REQUIRE_APPROVAL), 0),
            "denied": counts.get(str(PolicyEffect.DENY), 0),
            # The conservative roll-up, so a Policy Gate downstream can act on
            # the batch without re-deriving what "worst row" means.
            "effect": str(worst),
            "risk": str(worst_risk),
            "reasons": [
                f"{reason} ({count} row{'' if count == 1 else 's'})"
                for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
            ],
            "by_rule": dict(rule_counts.most_common()),
            "rows": verdicts,
            "rows_reported": len(verdicts),
            "unmapped_columns": missing,
        }


def build_triage_tools(
    documents: DocumentRepository, blobs: BlobStore, policy: PolicyEngine
) -> list[Tool]:
    return [TriageRowsTool(documents, blobs, policy)]

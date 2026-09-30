"""The step vocabulary the builder speaks.

Workflow definitions are precise: a step names an agent or a tool, declares
dependencies, wires inputs through an expression language. That is the right
contract for the engine and the wrong one for a person assembling a workflow
in a browser.

So the builder speaks in outcomes -- *Extract Data*, *Validate*, *AI
Decision*, *Approval* -- and this catalogue is the translation table. Each
entry says what the step means, what it needs upstream, and what downstream
steps can read from it.

Adding a step kind is an entry here plus a compiler clause; it is not a change
to the engine.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

TRIAGE_TOOL = "policy.triage_rows"
"""The registered tool a Row Triage step runs.

It goes through the tool registry like any other, but it authorizes
rather than acts, so several places have to tell it apart from a step
that changes an enterprise system.
"""


class StepKind(StrEnum):
    """What a builder step does, in the user's vocabulary."""

    EXTRACT = "extract"
    ANALYSE_DATASET = "analyse_dataset"
    VALIDATE = "validate"
    AI_DECISION = "ai_decision"
    POLICY = "policy"
    ROW_POLICY = "row_policy"
    FAN_OUT = "fan_out"
    APPROVAL = "approval"
    ACTION = "action"
    REPORT = "report"
    NOTIFY = "notify"


class TriggerKind(StrEnum):
    """How a workflow gets started."""

    MANUAL = "manual"
    """Started by a person, from the dashboard or the API."""

    API = "api"
    """Started by another system calling POST /workflows."""

    FILE = "file"
    """Started when a document is uploaded."""


class StepKindSpec(BaseModel):
    """What the frontend needs to render one palette entry."""

    model_config = ConfigDict(frozen=True)

    kind: StepKind
    label: str
    description: str
    category: str
    uses_ai: bool = False
    """True when the step calls a model, so cost and latency are visible."""

    needs_document: bool = False
    requires_ruleset: bool = False
    requires_tool: bool = False
    requires_child_workflow: bool = False
    """The step runs another workflow, which the author has to name."""
    produces: tuple[str, ...] = ()
    """Fields downstream steps can read from this step's output."""

    default_critical: bool = True
    """False where a failure should degrade the workflow, not fail it.

    A report that fails to generate must not undo a payment that already
    happened.
    """

    hint: str = ""


CATALOGUE: tuple[StepKindSpec, ...] = (
    StepKindSpec(
        kind=StepKind.EXTRACT,
        label="Extract Data",
        description="Read a document into structured fields, with a confidence score.",
        category="Understand",
        uses_ai=True,
        needs_document=True,
        produces=("fields", "missing_fields", "data_complete", "confidence"),
        hint="Works on spreadsheets, PDFs, CSV and free text.",
    ),
    StepKindSpec(
        kind=StepKind.ANALYSE_DATASET,
        label="Analyse Dataset",
        description=(
            "Count rows, find missing values and duplicate keys. Computed in "
            "code over every row, not by a model."
        ),
        category="Understand",
        uses_ai=False,
        needs_document=True,
        produces=("row_count", "issues", "issue_count", "clean", "completeness"),
        hint="Use this for spreadsheets. It is exact, fast and free.",
    ),
    StepKindSpec(
        kind=StepKind.VALIDATE,
        label="Validate",
        description="Inspect the document for anomalies that need a human's eye.",
        category="Understand",
        uses_ai=True,
        produces=("valid", "issues", "issue_count"),
        hint="Judgement only. Counting and exact matches belong in Analyse Dataset.",
    ),
    StepKindSpec(
        kind=StepKind.AI_DECISION,
        label="AI Decision",
        description="Recommend APPROVE, REJECT or MANUAL_REVIEW, with reasons.",
        category="Decide",
        uses_ai=True,
        produces=("recommendation", "reason", "risk_factors"),
        hint="A recommendation only. The policy step decides what actually happens.",
    ),
    StepKindSpec(
        kind=StepKind.POLICY,
        label="Policy Gate",
        description=(
            "Apply a deterministic ruleset. This is what authorizes the action "
            "-- an agent's recommendation cannot override it."
        ),
        category="Decide",
        uses_ai=False,
        requires_ruleset=True,
        produces=("effect", "risk", "reasons", "matched_rules"),
        hint="Put this before anything that changes an enterprise system.",
    ),
    StepKindSpec(
        kind=StepKind.ROW_POLICY,
        label="Row Triage",
        description=(
            "Apply the same ruleset to every row of an uploaded file and "
            "report which rows pass, which need a person, and which fail."
        ),
        category="Decide",
        uses_ai=False,
        needs_document=True,
        requires_ruleset=True,
        produces=(
            "allowed",
            "requires_approval",
            "denied",
            "effect",
            "risk",
            "total_rows",
            "examined_rows",
            "complete",
        ),
        hint=(
            "Use this for a file of many cases. It classifies each row; a "
            "Policy Gate after it can act on the counts."
        ),
    ),
    StepKindSpec(
        kind=StepKind.FAN_OUT,
        label="Run Per Row",
        description=(
            "Start a separate workflow for every row, each with its own "
            "approvals, retries and audit trail, then report the outcome."
        ),
        category="Decide",
        uses_ai=False,
        needs_document=True,
        requires_child_workflow=True,
        produces=(
            "total",
            "completed",
            "rejected",
            "failed",
            "awaiting_approval",
        ),
        hint=(
            "Use this when each row is its own case that can be approved or "
            "rejected separately. Row Triage only classifies; this acts."
        ),
    ),
    StepKindSpec(
        kind=StepKind.APPROVAL,
        label="Human Approval",
        description="Pause until a person decides. The workflow waits durably.",
        category="Decide",
        uses_ai=False,
        produces=("decision", "decided_by", "comment"),
        hint="No process is held open. It can wait for days.",
    ),
    StepKindSpec(
        kind=StepKind.ACTION,
        label="Action",
        description="Call an enterprise system through the tool registry.",
        category="Act",
        uses_ai=False,
        requires_tool=True,
        produces=("<the tool's response>",),
        hint="Authorized, validated, idempotent and audited on every call.",
    ),
    StepKindSpec(
        kind=StepKind.REPORT,
        label="Generate Report",
        description="Summarise the outcome for business readers.",
        category="Act",
        uses_ai=True,
        default_critical=False,
        produces=("title", "summary", "sections", "highlights"),
    ),
    StepKindSpec(
        kind=StepKind.NOTIFY,
        label="Notify",
        description="Draft a notification about the outcome.",
        category="Act",
        uses_ai=True,
        default_critical=False,
        produces=("subject", "body"),
    ),
)

BY_KIND: dict[StepKind, StepKindSpec] = {spec.kind: spec for spec in CATALOGUE}


class ToolChoice(BaseModel):
    """A tool the builder may offer for an Action step."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    domain: str
    description: str
    risk: str
    side_effect: str
    requires_approval: bool = False
    arguments: list[dict[str, object]] = Field(default_factory=list)


class RulesetChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    default_effect: str
    rule_count: int
    facts: list[str] = Field(default_factory=list)
    """The facts the ruleset reads, so the builder can show what to supply."""

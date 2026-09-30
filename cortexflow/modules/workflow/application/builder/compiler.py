"""Compiling a builder draft into a workflow definition.

The draft is what a person assembled; the definition is what the engine runs.
Compiling is where the two meet, and the important property is that the output
is an ordinary :class:`WorkflowDefinition` -- validated by the same rules,
executed by the same engine, with the same retries, approvals and audit trail
as a workflow shipped in the repository.

There is no second execution path for "workflows built in the UI".
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from cortexflow.modules.workflow.application.builder.catalogue import BY_KIND, StepKind, TriggerKind
from cortexflow.modules.workflow.domain.definition import (
    ApprovalSpec,
    FanOutSpec,
    JoinPolicy,
    Queue,
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.identity import Department, Role

DOCUMENT_TEXT = "${{ input.document_text }}"
DOCUMENT_ID = "${{ input.document_id }}"


class DraftStep(BaseModel):
    """One step as the builder describes it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    kind: StepKind
    name: str = ""
    depends_on: list[str] = Field(default_factory=list)
    when: str | None = None

    tool: str | None = None
    ruleset: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    facts: dict[str, Any] = Field(default_factory=dict)
    """Policy steps: which upstream value supplies each fact the ruleset reads.

    A ruleset asks for named facts -- ``amount``, ``employee_active`` -- and
    only the author knows which step produces each one. Auto-wiring can offer
    a sensible guess, but guessing wrong means the rule silently fails to
    evaluate and the case falls through to the ruleset default. So the mapping
    is explicit, and validation reports anything left unmapped.
    """

    expected_fields: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    unique_columns: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    guidance: str = ""

    approval_title: str = ""
    approval_roles: list[Role] = Field(default_factory=list)

    row_facts: dict[str, str] = Field(default_factory=dict)
    """Row Triage: which column of the file supplies each fact, per row.

    Distinct from ``facts``, which holds the values that are the same for
    every row -- a threshold, a budget, the agent's view of the batch. Keeping
    them apart is what makes "read this afresh for each row" unambiguous.
    """

    key_column: str = ""
    """Row Triage: the column that names a row in the report."""

    child_workflow: str = ""
    """Run Per Row: the workflow to start for each item."""

    collection: str = ""
    """Run Per Row: the expression yielding the items, defaulted from upstream."""

    await_approvals: bool = True
    """Run Per Row: whether the parent waits for cases held for a person."""

    critical: bool | None = None
    """Left unset, the step kind's default applies."""

    @model_validator(mode="after")
    def _check_requirements(self, info: ValidationInfo) -> DraftStep:
        if _is_lenient(info):
            # The draft is still being edited. What it is missing is reported
            # as a problem the author can read, not as a parse failure.
            return self
        problems = step_requirement_problems(self)
        if problems:
            raise ValueError(problems[0])
        return self


class WorkflowDraft(BaseModel):
    """A workflow as assembled in the builder."""

    model_config = ConfigDict(extra="forbid")

    # The name and step-count rules live in ``graph_problems`` rather than in
    # field constraints, so a half-finished draft can be *explained* instead
    # of rejected by the parser. Publishing still enforces them.
    name: str = Field(default="", max_length=64)
    title: str = ""
    description: str = ""
    department: Department = Department.PLATFORM
    trigger: TriggerKind = TriggerKind.MANUAL
    version: int = Field(default=1, ge=1)
    steps: list[DraftStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph(self, info: ValidationInfo) -> WorkflowDraft:
        if _is_lenient(info):
            return self
        problems = graph_problems(self)
        if problems:
            raise ValueError(problems[0])
        return self


# ---------------------------------------------------------------------------
# Draft problems
#
# The builder needs two answers about a draft, and they are not the same
# question: "is this safe to publish?" and "what is this author still
# missing?". Publishing raises; editing reports. Both read the rules from the
# functions below, so the two answers can never drift apart.
# ---------------------------------------------------------------------------
LENIENT = "lenient"
"""Validation-context key that turns parse failures into reportable problems."""

WORKFLOW_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_lenient(info: ValidationInfo) -> bool:
    context = info.context
    return bool(isinstance(context, dict) and context.get(LENIENT))


def parse_draft(payload: Any) -> WorkflowDraft:
    """Parse a draft that may still be incomplete.

    Used by the endpoints that *explain* a draft. They need the same object
    the compiler would receive, minus the refusal to build one -- a Policy
    Gate with no ruleset yet is the single most likely thing to be looking at
    while editing, and it is precisely what the strict model rejects.
    """
    return WorkflowDraft.model_validate(payload, context={LENIENT: True})


def step_requirement_problems(step: DraftStep) -> list[str]:
    """What one step still needs before it could run."""
    spec = BY_KIND[step.kind]
    problems: list[str] = []
    if spec.requires_tool and not step.tool:
        problems.append(f"step '{step.id}' is an Action and needs a tool")
    if spec.requires_ruleset and not step.ruleset:
        problems.append(f"step '{step.id}' is a {spec.label} and needs a ruleset")
    if spec.requires_child_workflow and not step.child_workflow:
        problems.append(
            f"step '{step.id}' is a {spec.label} and needs a workflow to run"
        )
    return problems


def graph_problems(draft: WorkflowDraft) -> list[str]:
    """What is wrong with the draft's shape, independent of any single step."""
    problems: list[str] = []
    if not draft.name:
        problems.append("the workflow needs a name")
    elif not WORKFLOW_NAME_PATTERN.match(draft.name):
        problems.append(
            f"workflow name '{draft.name}' must start with a letter and use "
            "only lowercase letters, digits and underscores"
        )
    if not draft.steps:
        problems.append("the workflow needs at least one step")

    ids = [step.id for step in draft.steps]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        problems.append(f"duplicate step ids: {sorted(duplicates)}")
    known = set(ids)
    for step in draft.steps:
        unknown = set(step.depends_on) - known
        if unknown:
            problems.append(
                f"step '{step.id}' depends on unknown steps: {sorted(unknown)}"
            )
    return problems


def draft_problems(draft: WorkflowDraft) -> list[str]:
    """Everything that would stop this draft compiling, all at once.

    All at once matters: raising on the first one makes an author fix a
    workflow one round trip per mistake.
    """
    problems = graph_problems(draft)
    for step in draft.steps:
        problems.extend(step_requirement_problems(step))
    return problems


def compile_draft(draft: WorkflowDraft) -> WorkflowDefinition:
    """Turn a draft into a definition the engine can run.

    Raises :class:`ValidationError` when the result would not be a valid
    workflow -- a cycle, a bad guard, an impossible dependency. The builder
    calls this before saving, so a person sees the problem while they are
    still editing rather than when a workflow fails.
    """
    by_id = {step.id: step for step in draft.steps}
    steps = tuple(_compile_step(step, by_id) for step in draft.steps)

    try:
        return WorkflowDefinition(
            name=draft.name,
            version=draft.version,
            department=draft.department,
            description=draft.description or draft.title,
            input_schema=_input_schema(draft),
            steps=steps,
            labels={
                "authored_in": "builder",
                "trigger": str(draft.trigger),
                **({"title": draft.title} if draft.title else {}),
            },
        )
    except Exception as exc:
        raise ValidationError(
            "The workflow is not valid", error=str(exc)[:400], workflow=draft.name
        ) from exc


def _input_schema(draft: WorkflowDraft) -> dict[str, Any]:
    """What a run of this workflow must supply."""
    needs_document = any(BY_KIND[s.kind].needs_document for s in draft.steps)
    properties: dict[str, Any] = {
        "document_id": {"type": "string"},
        "document_text": {"type": "string"},
        "filename": {"type": "string"},
    }
    return {
        "required": ["document_id"] if needs_document else [],
        "properties": properties,
    }


def _compile_step(step: DraftStep, by_id: dict[str, DraftStep]) -> StepDefinition:
    """Map one draft step onto a concrete engine step."""
    common: dict[str, Any] = {
        "id": step.id,
        "name": step.name or BY_KIND[step.kind].label,
        "depends_on": tuple(step.depends_on),
        "when": step.when or None,
        "critical": (
            step.critical
            if step.critical is not None
            else BY_KIND[step.kind].default_critical
        ),
        # A step that merges exclusive branches must not require all of them,
        # or the branch that was skipped would block it forever.
        "join": JoinPolicy.ANY if _merges_branches(step, by_id) else JoinPolicy.ALL,
    }

    match step.kind:
        case StepKind.EXTRACT:
            return StepDefinition(
                **common,
                type=StepType.AGENT,
                agent="extraction_agent",
                queue=Queue.EXTRACTION,
                timeout_seconds=180,
                inputs={
                    "text": DOCUMENT_TEXT,
                    "expected_fields": step.expected_fields,
                },
            )

        case StepKind.ANALYSE_DATASET:
            return StepDefinition(
                **common,
                type=StepType.TOOL,
                tool="documents.analyse_dataset",
                queue=Queue.VALIDATION,
                timeout_seconds=60,
                inputs={
                    "arguments": {
                        "document_id": DOCUMENT_ID,
                        "required_columns": step.required_columns,
                        "unique_columns": step.unique_columns,
                        # Only when a later step has to act on each row: the
                        # profile stays a summary for every other workflow.
                        "include_rows": _feeds_a_fan_out(step, by_id),
                    }
                },
            )

        case StepKind.VALIDATE:
            upstream = _first_of_kind(step, by_id, StepKind.EXTRACT)
            return StepDefinition(
                **common,
                type=StepType.AGENT,
                agent="validation_agent",
                queue=Queue.VALIDATION,
                timeout_seconds=120,
                inputs={
                    "data": _output_ref(upstream, "fields") if upstream else {},
                    "text": DOCUMENT_TEXT,
                    "checks": step.checks,
                },
            )

        case StepKind.AI_DECISION:
            return StepDefinition(
                **common,
                type=StepType.AGENT,
                agent="decision_agent",
                queue=Queue.DECISION,
                timeout_seconds=120,
                inputs={
                    "guidance": step.guidance
                    or (
                        "Review this case and recommend how it should be handled. "
                        "A policy engine holds the thresholds and will override you."
                    ),
                    "case": _case_from(step, by_id),
                },
            )

        case StepKind.POLICY:
            return StepDefinition(
                **common,
                type=StepType.POLICY,
                ruleset=step.ruleset,
                inputs={"facts": _facts_from(step, by_id)},
            )


        case StepKind.ROW_POLICY:
            return StepDefinition(
                **common,
                type=StepType.TOOL,
                tool="policy.triage_rows",
                queue=Queue.VALIDATION,
                timeout_seconds=120,
                inputs={
                    "arguments": {
                        "document_id": DOCUMENT_ID,
                        "ruleset": step.ruleset,
                        # Read per row, so they are column names and stay
                        # unrendered; everything else resolves from the
                        # workflow context exactly as a Policy Gate's does.
                        "row_facts": step.row_facts,
                        "facts": _facts_from(step, by_id),
                        "key_column": step.key_column,
                    }
                },
            )

        case StepKind.FAN_OUT:
            return StepDefinition(
                **common,
                type=StepType.FAN_OUT,
                fan_out=FanOutSpec(
                    workflow=step.child_workflow,
                    # Defaulted to the rows of the upstream analysis, which is
                    # where a batch's items come from nine times in ten.
                    collection=step.collection or _rows_from(step, by_id),
                    item_key=step.key_column,
                    await_approvals=step.await_approvals,
                ),
            )

        case StepKind.APPROVAL:
            return StepDefinition(
                **common,
                type=StepType.HUMAN_APPROVAL,
                approval=ApprovalSpec(
                    title=step.approval_title or step.name or "Approval required",
                    description="Raised by a workflow built in the CortexFlow builder.",
                    required_roles=tuple(step.approval_roles),
                    on_timeout="escalate",
                ),
                inputs={"proposed_action": _case_from(step, by_id)},
            )

        case StepKind.ACTION:
            return StepDefinition(
                **common,
                type=StepType.TOOL,
                tool=step.tool,
                queue=Queue.INTEGRATION,
                timeout_seconds=90,
                inputs={"arguments": step.arguments},
            )

        case StepKind.REPORT:
            return StepDefinition(
                **common,
                type=StepType.AGENT,
                agent="reporting_agent",
                queue=Queue.REPORTING,
                timeout_seconds=120,
                inputs={
                    "subject": {"filename": "${{ input.filename }}"},
                    "outcome": _case_from(step, by_id),
                },
            )

        case StepKind.NOTIFY:
            return StepDefinition(
                **common,
                type=StepType.AGENT,
                agent="communication_agent",
                queue=Queue.REPORTING,
                timeout_seconds=90,
                inputs={
                    "outcome": "processed",
                    "details": _case_from(step, by_id),
                },
            )

    raise ValidationError("Unsupported step kind", kind=str(step.kind))


# ---------------------------------------------------------------------------
# Input wiring
# ---------------------------------------------------------------------------
def _output_ref(step_id: str, field: str) -> str:
    return f"${{{{ steps.{step_id}.output.{field} }}}}"


def _first_of_kind(
    step: DraftStep, by_id: dict[str, DraftStep], kind: StepKind
) -> str | None:
    """The nearest upstream step of a given kind, searching breadth-first."""
    seen: set[str] = set()
    frontier = list(step.depends_on)
    while frontier:
        current = frontier.pop(0)
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        if by_id[current].kind is kind:
            return current
        frontier.extend(by_id[current].depends_on)
    return None


def _merges_branches(step: DraftStep, by_id: dict[str, DraftStep]) -> bool:
    """True when this step depends on two or more guarded (exclusive) steps."""
    guarded = [
        dep for dep in step.depends_on if dep in by_id and by_id[dep].when
    ]
    return len(guarded) >= 1 and len(step.depends_on) > 1


def _case_from(step: DraftStep, by_id: dict[str, DraftStep]) -> dict[str, Any]:
    """Assemble what upstream steps produced, for an agent or an approver.

    Wired from declared dependencies, so the case reflects the graph the
    person drew rather than a fixed assumption about workflow shape.
    """
    case: dict[str, Any] = {"filename": "${{ input.filename }}"}
    for dep in _upstream(step, by_id):
        upstream = by_id[dep]
        for field in BY_KIND[upstream.kind].produces:
            if field.startswith("<"):
                continue
            case[f"{dep}_{field}"] = _output_ref(dep, field)
    return case


def _feeds_a_fan_out(step: DraftStep, by_id: dict[str, DraftStep]) -> bool:
    """Whether any Run Per Row step draws its items from this one."""
    return any(
        other.kind is StepKind.FAN_OUT
        and not other.collection
        and step.id in _upstream(other, by_id)
        for other in by_id.values()
    )


def _rows_from(step: DraftStep, by_id: dict[str, DraftStep]) -> str:
    """Where a fan-out finds its items when the author has not said.

    The rows of the nearest upstream dataset step: the reading that makes
    "run this for every row of the file I uploaded" work without anyone having
    to learn the expression language first.
    """
    for dep in _upstream(step, by_id):
        if by_id[dep].kind is StepKind.ANALYSE_DATASET:
            return f"steps.{dep}.output.rows"
    return "input.rows"


def _facts_from(step: DraftStep, by_id: dict[str, DraftStep]) -> dict[str, Any]:
    """Assemble the facts a policy ruleset evaluates.

    The author's explicit mapping wins; the auto-derived upstream outputs are
    kept alongside it so a rule referring to something unmapped can still find
    it under its qualified name.

    Only upstream outputs and workflow input are included -- never raw
    document text, so a prompt injection in an uploaded file cannot reach the
    authorization decision.
    """
    facts = _case_from(step, by_id)
    facts.update(step.facts)
    return facts


def _upstream(step: DraftStep, by_id: dict[str, DraftStep]) -> list[str]:
    """Every transitive dependency, nearest first."""
    ordered: list[str] = []
    seen: set[str] = set()
    frontier = list(step.depends_on)
    while frontier:
        current = frontier.pop(0)
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        ordered.append(current)
        frontier.extend(by_id[current].depends_on)
    return ordered

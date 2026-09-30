"""Cases: the unit a business actually reasons about.

A finance manager does not think in workflow runs. They think in *claims* --
one expense, one decision, one person who may have to look at it. The engine
has always known this, because a fan-out already starts one workflow per row
and keys it by ``expense_id``. What was missing is a way to ask for it: the
only way to find EXP007 was to know which batch it came from and count.

This is a read model. It computes nothing the engine did not already decide,
holds no state, and is never consulted while a workflow runs. A case *is* a
workflow run, read with different questions in mind:

    a run that fanned out          -> a batch, containing cases
    a child of that fan-out        -> a case
    a run that decided one thing   -> a case in its own right

Which is why there is no Case table and no second engine. Introducing either
would mean two records of the same event that could disagree, and the one
that disagreed would be the one somebody trusted.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.workflow.application.explanation import (
    Disagreement,
    Moment,
    RunExplanation,
    explain,
)
from cortexflow.modules.workflow.domain.definition import WorkflowDefinition
from cortexflow.modules.workflow.domain.models import Workflow, WorkflowStatus

# Facts worth showing on a case card, in the order a reader wants them.
# Nothing is invented: a key appears only if the case's own input carries it.
HEADLINE_FIELDS = (
    "employee_name",
    "employee_id",
    "amount",
    "currency",
    "category",
    "department",
    "vendor",
    "candidate_name",
    "campaign_id",
)

MAX_HEADLINE_FACTS = 4


class CaseFact(BaseModel):
    """One labelled value from the case's own input."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    value: Any


class CaseSummary(BaseModel):
    """A case as it appears in a list."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    """What a person calls it -- ``EXP007``. Falls back to the run id."""

    workflow_id: str
    workflow: str
    title: str
    status: str
    created_at: str

    decision: str = ""
    """What the rules authorised: ALLOW, REQUIRE_APPROVAL, DENY."""

    recommendation: str = ""
    """What the model advised. Advisory, and labelled as such everywhere."""

    reason: str = ""
    disagreed: bool = False
    """True when the model and the rules reached different conclusions."""

    simulated: bool = False
    batch_id: str | None = None
    """The run that fanned out to produce this case, when there was one."""

    facts: tuple[CaseFact, ...] = ()


class CaseDetail(CaseSummary):
    """One case, with everything that happened to it."""

    model_config = ConfigDict(frozen=True)

    outcome: str = ""
    timeline: tuple[Moment, ...] = ()
    disagreements: tuple[Disagreement, ...] = ()
    awaiting_role: tuple[str, ...] = ()
    """Roles that can decide this case, when it is waiting for a person."""


def summarise(
    workflow: Workflow, definition: WorkflowDefinition
) -> CaseSummary:
    """Read one run as a case, without explaining it in full.

    Deliberately cheaper than :func:`detail`: a list of five hundred cases
    should not build five hundred timelines.
    """
    decision, reason = _decision(workflow)
    recommendation = _recommendation(workflow)
    return CaseSummary(
        case_id=workflow.fan_out_key or workflow.workflow_id,
        workflow_id=workflow.workflow_id,
        workflow=workflow.definition_name,
        title=definition.labels.get("title") or _humanise(definition.name),
        status=_status(workflow),
        created_at=workflow.created_at.isoformat(),
        decision=decision,
        recommendation=recommendation,
        reason=reason,
        disagreed=_disagrees(recommendation, decision),
        simulated=_is_simulated(workflow),
        batch_id=workflow.parent_workflow_id,
        facts=_facts(workflow),
    )


def detail(workflow: Workflow, definition: WorkflowDefinition) -> CaseDetail:
    """Read one run as a case, with its timeline and any disagreement.

    The timeline comes from the explanation read model rather than from a
    second traversal of the steps: two renderings of the same history are two
    chances to disagree about it.
    """
    reading: RunExplanation = explain(workflow, definition)
    base = summarise(workflow, definition)
    return CaseDetail(
        **base.model_dump(),
        outcome=reading.outcome,
        timeline=reading.timeline,
        disagreements=reading.disagreements,
        awaiting_role=_awaiting_role(workflow, definition),
    )


# ----------------------------------------------------------------------
def _status(workflow: Workflow) -> str:
    """The status a person reads, distinct from why it ended that way.

    REJECTED means the rules refused it; FAILED means the machinery broke.
    Collapsing the two is how a policy denial starts looking like an outage.
    """
    return str(workflow.status)


def _is_simulated(workflow: Workflow) -> bool:
    return workflow.labels.get("execution_mode", "").lower() == "simulation"


def _decision(workflow: Workflow) -> tuple[str, str]:
    """The authoritative decision and the reason given for it."""
    for step in workflow.steps.values():
        output = step.output or {}
        if "effect" in output and "matched_rules" in output:
            reasons = output.get("reasons") or []
            return str(output["effect"]), "; ".join(str(r) for r in reasons)[:300]
    if workflow.cancellation_reason:
        return "", workflow.cancellation_reason[:300]
    if workflow.error is not None:
        return "", workflow.error.message[:300]
    return "", ""


def _recommendation(workflow: Workflow) -> str:
    for step in workflow.steps.values():
        output = step.output or {}
        if "recommendation" in output:
            return str(output["recommendation"]).upper()
    return ""


def _disagrees(recommendation: str, decision: str) -> bool:
    """Whether the model and the rules reached different conclusions.

    Only meaningful when both spoke. An approval the rules allowed and the
    model also approved is agreement; silence from either is not disagreement.
    """
    if not recommendation or not decision:
        return False
    agreed = {
        ("APPROVE", "ALLOW"),
        ("REJECT", "DENY"),
        ("MANUAL_REVIEW", "REQUIRE_APPROVAL"),
    }
    return (recommendation, decision) not in agreed


def _facts(workflow: Workflow) -> tuple[CaseFact, ...]:
    """The few input values worth putting on a card.

    Read from the case's own input and nowhere else. A card that showed a
    value the case did not carry would be a card that invented one.
    """
    found: list[CaseFact] = []
    for name in HEADLINE_FIELDS:
        if name not in workflow.input:
            continue
        value = workflow.input[name]
        if value is None or value == "":
            continue
        found.append(CaseFact(name=name, label=_humanise(name), value=value))
        if len(found) >= MAX_HEADLINE_FACTS:
            break
    return tuple(found)


def _awaiting_role(
    workflow: Workflow, definition: WorkflowDefinition
) -> tuple[str, ...]:
    """Who can move this case on, when it is waiting for a person."""
    if workflow.status is not WorkflowStatus.WAITING_FOR_APPROVAL:
        return ()
    roles: list[str] = []
    for step in workflow.steps.values():
        spec = definition.step(step.step_id) if step.step_id in {
            s.id for s in definition.steps
        } else None
        if spec is not None and spec.approval is not None:
            roles.extend(str(r) for r in spec.approval.required_roles)
    return tuple(dict.fromkeys(roles))


def _humanise(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()

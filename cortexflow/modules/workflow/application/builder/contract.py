"""The workflow contract.

A definition can be perfectly valid and still be the wrong workflow. Every
check the builder already runs -- acyclic graph, references resolve, facts
mapped -- answers "will this execute?". None of them answers "is this what you
meant?", and that is the question someone is actually deciding when they press
publish.

So the contract restates a compiled definition as prose: what starts it, what
one case is, where a model is consulted and where it is not, which rules
authorize the outcome, who has to agree, what it will change in an enterprise
system, and what happens when part of it fails. It is derived entirely from
the definition -- nothing here is authored separately, so it cannot drift from
what will run.

Reading it is the last chance to notice that a workflow which passes every
check pays every invoice without asking anyone.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.domain.models import SideEffect
from cortexflow.modules.workflow.application.builder.catalogue import TRIAGE_TOOL
from cortexflow.modules.workflow.domain.definition import (
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.shared.errors import NotFoundError


class CaseModel(BaseModel):
    """What one case *is* for this workflow.

    The most consequential thing a reader can misunderstand about a workflow
    that starts from a spreadsheet, and the one the definition states only
    implicitly, in the choice of steps. A file of a hundred expenses is either
    one decision or a hundred, and the difference does not show up until
    someone asks why a single duplicate row rejected the whole batch.
    """

    model_config = ConfigDict(frozen=True)

    shape: str
    """``run`` | ``row`` | ``rows_classified``."""

    summary: str
    detail: str
    child_workflow: str = ""
    """The workflow each case runs, when cases are separate workflows."""


class AiUse(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: str
    agent: str
    purpose: str


class RuleStatement(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    condition: str
    outcome: str
    risk: str
    description: str = ""


class RuleSource(BaseModel):
    """One decision point, and the rules that authorize it."""

    model_config = ConfigDict(frozen=True)

    step: str
    ruleset: str
    applies_to: str
    """``case`` when it decides the whole run, ``row`` when it judges each row."""

    default_outcome: str
    rules: tuple[RuleStatement, ...] = ()
    unmapped_facts: tuple[str, ...] = ()


class HumanPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: str
    title: str
    roles: tuple[str, ...] = ()
    on_timeout: str = "escalate"


class ActionStatement(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: str
    tool: str
    description: str
    risk: str
    side_effect: str
    changes_enterprise_state: bool
    gated_by_policy: bool
    gated_by_approval: bool


class WorkflowContract(BaseModel):
    """What this workflow promises to do, in the reader's language."""

    model_config = ConfigDict(frozen=True)

    workflow: str
    version: int
    title: str
    purpose: str
    department: str
    trigger: str
    inputs: tuple[str, ...] = ()
    case_model: CaseModel
    ai: tuple[AiUse, ...] = ()
    rules: tuple[RuleSource, ...] = ()
    human: tuple[HumanPoint, ...] = ()
    actions: tuple[ActionStatement, ...] = ()
    failure_behaviour: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    """What a reader should look hard at before publishing."""

    @property
    def changes_enterprise_state(self) -> bool:
        return any(a.changes_enterprise_state for a in self.actions)

    @property
    def high_risk_actions(self) -> tuple[ActionStatement, ...]:
        return tuple(
            a for a in self.actions if a.risk in {str(RiskLevel.HIGH), str(RiskLevel.CRITICAL)}
        )


AGENT_PURPOSE: dict[str, str] = {
    "extraction_agent": "reads a document into structured fields",
    "validation_agent": "looks for anomalies a person would notice",
    "decision_agent": "recommends an outcome, which the rules may overrule",
    "reporting_agent": "writes the summary for business readers",
    "communication_agent": "drafts the notification",
    "triage_agent": "sorts cases by what they appear to need",
}


def build_contract(
    definition: WorkflowDefinition,
    *,
    tools: ToolRegistry,
    policy: PolicyEngine,
    tenant_id: str,
    unmapped_facts: dict[str, list[str]] | None = None,
) -> WorkflowContract:
    """Restate a compiled definition as the promise it makes."""
    gaps = unmapped_facts or {}
    steps = definition.steps

    actions = tuple(_action(s, tools, definition) for s in steps if _is_action(s))
    rules = tuple(
        _rule_source(s, policy, tenant_id, gaps)
        for s in steps
        if _decides_with_rules(s)
    )
    human = tuple(_human_point(s) for s in steps if s.type is StepType.HUMAN_APPROVAL)
    ai = tuple(
        AiUse(
            step=s.id,
            agent=str(s.agent),
            purpose=AGENT_PURPOSE.get(str(s.agent), "interprets this step's inputs"),
        )
        for s in steps
        if s.type is StepType.AGENT
    )

    return WorkflowContract(
        workflow=definition.name,
        version=definition.version,
        title=definition.labels.get("title") or definition.name.replace("_", " ").title(),
        purpose=definition.description or "No purpose was recorded for this workflow.",
        department=str(definition.department),
        trigger=definition.labels.get("trigger", "manual"),
        inputs=tuple(definition.input_schema.get("required", ())),
        case_model=_case_model(definition),
        ai=ai,
        rules=rules,
        human=human,
        actions=actions,
        failure_behaviour=_failure_behaviour(definition),
        concerns=_concerns(definition, actions=actions, human=human, rules=rules),
    )


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------
def _case_model(definition: WorkflowDefinition) -> CaseModel:
    fan_out = next((s for s in definition.steps if s.type is StepType.FAN_OUT), None)
    if fan_out is not None and fan_out.fan_out is not None:
        return CaseModel(
            shape="row",
            summary="One case per row",
            detail=(
                f"Each item becomes its own run of '{fan_out.fan_out.workflow}', "
                "with its own approvals, retries and audit trail. One case "
                f"failing does not stop the others (at most "
                f"{fan_out.fan_out.max_children} per run)."
            ),
            child_workflow=fan_out.fan_out.workflow,
        )

    triage = next(
        (s for s in definition.steps if s.type is StepType.TOOL and s.tool == TRIAGE_TOOL),
        None,
    )
    if triage is not None:
        return CaseModel(
            shape="rows_classified",
            summary="One case, with every row classified inside it",
            detail=(
                "Each row is judged against the ruleset and counted, but the "
                "run reaches a single outcome. Nothing acts on an individual "
                "row -- add a Run Per Row step if each one should be approved "
                "or actioned separately."
            ),
        )

    return CaseModel(
        shape="run",
        summary="One case per run",
        detail=(
            "Everything the run is given is decided together, as a single "
            "case with a single outcome."
        ),
    )


# ---------------------------------------------------------------------------
# Rules, people and actions
# ---------------------------------------------------------------------------
def _decides_with_rules(step: StepDefinition) -> bool:
    return step.type is StepType.POLICY or (
        step.type is StepType.TOOL and step.tool == TRIAGE_TOOL
    )


def _rule_source(
    step: StepDefinition,
    policy: PolicyEngine,
    tenant_id: str,
    gaps: dict[str, list[str]],
) -> RuleSource:
    per_row = step.type is StepType.TOOL
    name = (
        str(step.inputs.get("arguments", {}).get("ruleset", ""))
        if per_row
        else str(step.ruleset or "")
    )
    try:
        ruleset = policy.get(name, tenant_id)
    except NotFoundError:
        return RuleSource(
            step=step.id,
            ruleset=name,
            applies_to="row" if per_row else "case",
            default_outcome="unknown -- this ruleset does not exist",
            unmapped_facts=tuple(gaps.get(step.id, ())),
        )

    return RuleSource(
        step=step.id,
        ruleset=ruleset.name,
        applies_to="row" if per_row else "case",
        default_outcome=str(ruleset.default_effect),
        rules=tuple(
            RuleStatement(
                rule=rule.id,
                condition=rule.when,
                outcome=str(rule.effect),
                risk=str(rule.risk),
                description=rule.message or rule.description,
            )
            for rule in ruleset.rules
        ),
        unmapped_facts=tuple(gaps.get(step.id, ())),
    )


def _human_point(step: StepDefinition) -> HumanPoint:
    spec = step.approval
    return HumanPoint(
        step=step.id,
        title=spec.title if spec else step.display_name,
        roles=tuple(str(r) for r in (spec.required_roles if spec else ())),
        on_timeout=spec.on_timeout if spec else "escalate",
    )


def _is_action(step: StepDefinition) -> bool:
    """A tool step that is an action, not an inspection.

    Row Triage runs through the tool registry but authorizes rather than acts,
    so it is reported under the rules it applies, not as something the
    workflow does to an enterprise system.
    """
    return step.type is StepType.TOOL and step.tool != TRIAGE_TOOL


def _action(
    step: StepDefinition, tools: ToolRegistry, definition: WorkflowDefinition
) -> ActionStatement:
    name = str(step.tool)
    try:
        metadata = tools.metadata(name)
    except NotFoundError:
        return ActionStatement(
            step=step.id,
            tool=name,
            description="This tool is not registered.",
            risk=str(RiskLevel.HIGH),
            side_effect="unknown",
            changes_enterprise_state=True,
            gated_by_policy=False,
            gated_by_approval=False,
        )

    upstream = _all_upstream(step.id, definition)
    return ActionStatement(
        step=step.id,
        tool=name,
        description=metadata.description,
        risk=str(metadata.risk),
        side_effect=str(metadata.side_effect),
        changes_enterprise_state=metadata.side_effect is not SideEffect.READ,
        gated_by_policy=any(
            definition.step(sid).type is StepType.POLICY for sid in upstream
        ),
        gated_by_approval=any(
            definition.step(sid).type is StepType.HUMAN_APPROVAL for sid in upstream
        ),
    )


def _all_upstream(step_id: str, definition: WorkflowDefinition) -> set[str]:
    """Every step that must succeed before this one runs.

    Transitive, because "is this payment gated?" is not a question about the
    payment's immediate dependency -- a gate three steps back still gates it.
    """
    seen: set[str] = set()
    frontier = list(definition.step(step_id).depends_on)
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(definition.step(current).depends_on)
    return seen


# ---------------------------------------------------------------------------
# Failure and concerns
# ---------------------------------------------------------------------------
def _failure_behaviour(definition: WorkflowDefinition) -> tuple[str, ...]:
    lines: list[str] = []
    fan_out = next((s for s in definition.steps if s.type is StepType.FAN_OUT), None)
    if fan_out is not None and fan_out.fan_out is not None:
        lines.append(
            "A case that fails does not stop the others; the run reports a breakdown."
            if fan_out.fan_out.tolerate_failures
            else "A single failed case fails the whole run."
        )
        lines.append(
            "The run is complete only once every case is, including those waiting "
            "on a person."
            if fan_out.fan_out.await_approvals
            else "The run settles once nothing can move without a person, reporting "
            "how many are still outstanding."
        )

    degrading = [s.display_name for s in definition.steps if not s.critical]
    if degrading:
        lines.append(
            f"These steps degrade the run rather than fail it: {', '.join(degrading)}."
        )
    unsafe = [s.display_name for s in definition.steps if not s.idempotent]
    if unsafe:
        lines.append(
            f"These steps are never retried automatically: {', '.join(unsafe)}."
        )
    lines.append(
        "A step that exhausts its retries parks the run for an operator "
        "instead of losing it."
    )
    return tuple(lines)


def _concerns(
    definition: WorkflowDefinition,
    *,
    actions: tuple[ActionStatement, ...],
    human: tuple[HumanPoint, ...],
    rules: tuple[RuleSource, ...],
) -> tuple[str, ...]:
    """What to look hard at -- phrased as consequences, not rule violations.

    Distinct from the builder's validation warnings, which say what is
    unusual. These say what it would cost.
    """
    concerns: list[str] = []
    writes = [a for a in actions if a.changes_enterprise_state]

    for action in writes:
        if not action.gated_by_policy:
            concerns.append(
                f"'{action.step}' changes an enterprise system with no rule "
                "authorizing it. Whatever reaches this step runs."
            )
        elif not action.gated_by_approval and action.risk in {
            str(RiskLevel.HIGH),
            str(RiskLevel.CRITICAL),
        }:
            concerns.append(
                f"'{action.step}' is a {action.risk.lower()}-risk action that no "
                "person has to agree to. The rules decide it alone."
            )

    for source in rules:
        if source.unmapped_facts:
            concerns.append(
                f"'{source.step}' does not supply {', '.join(source.unmapped_facts)}, "
                f"so those rules cannot match and every case falls through to "
                f"{source.default_outcome}."
            )

    if writes and not rules:
        concerns.append(
            "Nothing in this workflow authorizes anything. A model's "
            "recommendation is the only thing standing between an input and "
            "an enterprise change."
        )
    if not writes and not human:
        concerns.append(
            "This workflow reads and reports but changes nothing, which is "
            "safe to publish."
        )
    return tuple(concerns)

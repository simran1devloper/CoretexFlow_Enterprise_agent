"""Why a run came out the way it did.

Every fact this module reports is already in the durable record. What is
missing is the reading: a step list says a policy step succeeded with effect
DENY and an agent step succeeded with recommendation APPROVE, and leaves the
one question anybody actually has -- *why was this refused when the AI said
yes?* -- for the reader to reconstruct.

Reconstructing it by hand is how a platform stops being trusted. So the
reading is assembled here, once, from the same state the engine wrote, and in
particular it answers the awkward direction: not what happened, but what did
*not* happen and what stopped it.

This is a read model. It computes nothing the engine did not already decide,
holds no state of its own, and is never consulted while a workflow runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.policy.domain.models import PolicyEffect
from cortexflow.modules.workflow.domain.definition import (
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.modules.workflow.domain.models import (
    ExecutionMode,
    StepState,
    StepStatus,
    Workflow,
    WorkflowStatus,
)

RECOMMENDATION_KEYS = ("recommendation", "decision", "verdict")
"""Where the different agents put the thing they are recommending."""


class Moment(BaseModel):
    """One thing that happened, in the order it happened."""

    model_config = ConfigDict(frozen=True)

    step: str
    name: str
    kind: str
    """``ai`` | ``rules`` | ``human`` | ``action`` | ``cases``."""

    status: str
    headline: str
    detail: str = ""
    at: str = ""
    authoritative: bool = False
    """True for the steps that decided the outcome, as opposed to informing it."""


class NotDone(BaseModel):
    """Something the workflow could have done and did not.

    The most-asked question in production, and the one a step list answers
    worst: a step that never ran leaves almost no trace, so "why was the
    payment not made?" is answered by a SKIPPED status and nothing else.
    """

    model_config = ConfigDict(frozen=True)

    step: str
    name: str
    what: str
    why: str


class Disagreement(BaseModel):
    """Where a model's recommendation and the rules parted company."""

    model_config = ConfigDict(frozen=True)

    ai_step: str
    ai_said: str
    rules_step: str
    rules_said: str
    resolution: str


class SimulatedEffect(BaseModel):
    """Something a live run would have done to the outside world."""

    model_config = ConfigDict(frozen=True)

    step: str
    kind: str
    """``call`` | ``approval``."""

    target: str
    detail: dict[str, Any] = {}


class RunExplanation(BaseModel):
    """The whole reading of one run."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    workflow: str
    version: int
    status: str
    execution_mode: str
    outcome: str
    """One sentence a person who has never seen this workflow can act on."""

    timeline: tuple[Moment, ...] = ()
    not_done: tuple[NotDone, ...] = ()
    disagreements: tuple[Disagreement, ...] = ()
    would_have: tuple[SimulatedEffect, ...] = ()
    """Populated only for a simulation: the effects that were held back."""

    cases: dict[str, Any] | None = None
    """Present when this run fanned out: the breakdown across its cases."""


def explain(
    workflow: Workflow,
    definition: WorkflowDefinition,
    *,
    cases: dict[str, Any] | None = None,
) -> RunExplanation:
    """Read one run's durable state back as an explanation."""
    order = definition.topological_order()
    steps = [(sid, definition.step(sid), workflow.steps.get(sid)) for sid in order]

    policy_moments = [
        (spec, state)
        for _, spec, state in steps
        if state is not None and spec.type is StepType.POLICY
    ]
    ai_moments = [
        (spec, state)
        for _, spec, state in steps
        if state is not None
        and spec.type is StepType.AGENT
        and _recommendation(state) is not None
    ]

    return RunExplanation(
        workflow_id=workflow.workflow_id,
        workflow=workflow.definition_name,
        version=workflow.definition_version,
        status=str(workflow.status),
        execution_mode=str(workflow.execution_mode),
        outcome=_outcome(workflow, policy_moments),
        timeline=tuple(
            m
            for _, spec, state in steps
            if state is not None
            for m in (_moment(spec, state),)
            if m is not None
        ),
        not_done=tuple(_not_done(workflow, definition, steps)),
        disagreements=tuple(_disagreements(ai_moments, policy_moments)),
        would_have=tuple(_would_have(steps))
        if workflow.execution_mode is ExecutionMode.SIMULATION
        else (),
        cases=cases,
    )


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------
def _outcome(
    workflow: Workflow, policy: list[tuple[StepDefinition, StepState]]
) -> str:
    denial = next(
        (
            (spec, state)
            for spec, state in policy
            if (state.output or {}).get("effect") == str(PolicyEffect.DENY)
        ),
        None,
    )
    simulated = workflow.execution_mode is ExecutionMode.SIMULATION
    prefix = "This was a simulation. " if simulated else ""

    match workflow.status:
        case WorkflowStatus.REJECTED:
            if denial is not None:
                spec, state = denial
                reasons = _sentence((state.output or {}).get("reasons", ()))
                return (
                    f"{prefix}Refused by the rules at '{spec.display_name}': {reasons}. "
                    "Nothing after that step ran."
                )
            return f"{prefix}Refused. {workflow.cancellation_reason or ''}".strip()
        case WorkflowStatus.COMPLETED:
            return (
                f"{prefix}Finished. Every step that was meant to run did."
                if not simulated
                else f"{prefix}Finished. Nothing outside CortexFlow was touched."
            )
        case WorkflowStatus.WAITING_FOR_APPROVAL:
            return f"{prefix}Waiting for a person to decide. Nothing is lost while it waits."
        case WorkflowStatus.WAITING_FOR_RECOVERY:
            return (
                f"{prefix}Stopped after exhausting its retries, and parked for an "
                "operator rather than dropped. It can be replayed once the cause is fixed."
            )
        case WorkflowStatus.FAILED:
            error = workflow.error
            return (
                f"{prefix}Failed: {error.message}" if error else f"{prefix}Failed."
            )
        case WorkflowStatus.CANCELLED:
            return f"{prefix}Cancelled. {workflow.cancellation_reason or ''}".strip()
        case _:
            return f"{prefix}Still running."


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
def _moment(spec: StepDefinition, state: StepState) -> Moment | None:
    if state.status is StepStatus.PENDING:
        return None

    output = state.output or {}
    kind, headline, detail, authoritative = _read_step(spec, state, output)
    at = state.ended_at or state.started_at or state.dispatched_at
    return Moment(
        step=spec.id,
        name=spec.display_name,
        kind=kind,
        status=str(state.status),
        headline=headline,
        detail=detail,
        at=at.isoformat() if at else "",
        authoritative=authoritative,
    )


def _read_step(
    spec: StepDefinition, state: StepState, output: dict[str, Any]
) -> tuple[str, str, str, bool]:
    """What one step amounted to: its kind, a headline, a detail, its authority.

    Every status that is not SUCCEEDED is answered here, before the per-type
    reading below. A step that was cancelled has whatever output a previous
    attempt left behind, or none at all, and describing it by that output
    would put "Called finance.initiate_payment" in the timeline of a run where
    no payment was made -- the single worst thing an explanation can say.
    """
    kind = _kind_of(spec)
    if state.status is StepStatus.FAILED:
        message = state.error.message if state.error else "no reason was recorded"
        return kind, f"Failed: {message}", "", False
    if state.status is StepStatus.SKIPPED:
        return kind, "Skipped", _skip_reason(spec), False
    if state.status is StepStatus.CANCELLED:
        return kind, "Cancelled before it ran", "", False
    if state.status is StepStatus.WAITING_FOR_APPROVAL:
        return kind, "Waiting for a person", str(output.get("title", "")), True
    if state.status.is_in_flight:
        return kind, "Running", "", False

    match spec.type:
        case StepType.POLICY:
            effect = str(output.get("effect", ""))
            matched = output.get("matched_rules") or []
            return (
                "rules",
                _effect_headline(effect),
                (
                    f"Matched {', '.join(matched)}."
                    if matched
                    else "No rule matched, so the ruleset default applied."
                )
                + (
                    " " + "; ".join(output.get("reasons", ()))
                    if output.get("reasons")
                    else ""
                ),
                True,
            )
        case StepType.AGENT:
            recommendation = _recommendation(output)
            if recommendation is None:
                return "ai", "Produced its output", str(output.get("reason", "")), False
            return (
                "ai",
                f"Recommended {recommendation}",
                str(output.get("reason", ""))
                + (
                    f" (confidence {state.confidence:.0%})"
                    if state.confidence is not None
                    else ""
                ),
                False,
            )
        case StepType.HUMAN_APPROVAL:
            if output.get("simulated"):
                return (
                    "human",
                    "Would have paused for approval",
                    str(output.get("note", "")),
                    True,
                )
            decision = str(output.get("decision", ""))
            return (
                "human",
                f"{decision.title()} by {output.get('decided_by', 'a person')}"
                if decision
                else "Decided",
                str(output.get("comment", "")),
                True,
            )
        case StepType.FAN_OUT:
            return (
                "cases",
                f"{output.get('total', 0)} cases: "
                f"{output.get('completed', 0)} completed, "
                f"{output.get('rejected', 0)} refused, "
                f"{output.get('awaiting_approval', 0)} awaiting a person, "
                f"{output.get('failed', 0)} failed",
                "Each case ran as its own workflow.",
                False,
            )
        case _:
            if output.get("simulated"):
                return "action", f"Held back: {output.get('would_call', spec.tool)}", str(
                    output.get("note", "")
                ), False
            return (
                "action",
                f"Called {spec.tool}",
                "Reused an earlier result rather than repeating the call."
                if output.get("replayed")
                else "",
                False,
            )


def _effect_headline(effect: str) -> str:
    match effect:
        case "":
            return "Evaluated, but recorded no effect"
        case "ALLOW":
            return "Authorized"
        case "DENY":
            return "Refused"
        case "REQUIRE_APPROVAL":
            return "Sent for human approval"
        case _:
            return f"Decided {effect}"


def _kind_of(spec: StepDefinition) -> str:
    match spec.type:
        case StepType.AGENT:
            return "ai"
        case StepType.POLICY:
            return "rules"
        case StepType.HUMAN_APPROVAL:
            return "human"
        case StepType.FAN_OUT:
            return "cases"
        case _:
            return "action"


def _sentence(reasons: Any) -> str:
    """Join policy reasons into something that can sit mid-sentence.

    Rule messages are authored as standalone sentences and end in a full
    stop, so embedding them raw produced "not active.. Nothing after".
    """
    joined = "; ".join(str(r).rstrip(". ") for r in reasons if str(r).strip())
    return joined or "no reason was recorded"


def _skip_reason(spec: StepDefinition) -> str:
    if spec.when:
        return f"Its condition was not met: {spec.when}"
    return "An upstream branch it depends on was not taken."


# ---------------------------------------------------------------------------
# What did not happen
# ---------------------------------------------------------------------------
def _not_done(
    workflow: Workflow,
    definition: WorkflowDefinition,
    steps: list[tuple[str, StepDefinition, StepState | None]],
) -> list[NotDone]:
    """Every action the workflow did not take, and what stopped it.

    Restricted to the steps that would have changed something or asked
    somebody. A skipped report is noise; a skipped payment is the question.
    """
    denial = next(
        (
            (spec, state)
            for _, spec, state in steps
            if state is not None
            and spec.type is StepType.POLICY
            and (state.output or {}).get("effect") == str(PolicyEffect.DENY)
        ),
        None,
    )

    results: list[NotDone] = []
    for _, spec, state in steps:
        if spec.type not in {StepType.TOOL, StepType.HUMAN_APPROVAL, StepType.FAN_OUT}:
            continue
        if state is None or state.status is StepStatus.SUCCEEDED:
            continue
        if state.status in {StepStatus.DISPATCHED, StepStatus.RUNNING}:
            continue

        what = (
            f"'{spec.tool}' was not called"
            if spec.type is StepType.TOOL
            else f"'{spec.display_name}' did not happen"
        )
        results.append(
            NotDone(step=spec.id, name=spec.display_name, what=what,
                    why=_why_not(spec, state, denial, workflow))
        )
    return results


def _why_not(
    spec: StepDefinition,
    state: StepState,
    denial: tuple[StepDefinition, StepState] | None,
    workflow: Workflow,
) -> str:
    if state.status is StepStatus.FAILED and state.error is not None:
        return f"It failed: {state.error.message}"
    if state.status is StepStatus.WAITING_FOR_APPROVAL:
        return "It is still waiting for a person to decide."
    if state.status is StepStatus.CANCELLED and denial is not None:
        gate, gate_state = denial
        reasons = _sentence((gate_state.output or {}).get("reasons", ()))
        return (
            f"The rules at '{gate.display_name}' refused the case: {reasons}. "
            "Everything downstream was cancelled."
        )
    if state.status is StepStatus.CANCELLED:
        return (
            f"The run was cancelled. {workflow.cancellation_reason or ''}".strip()
        )
    if state.status is StepStatus.SKIPPED:
        return _skip_reason(spec)
    return "It never became runnable."


# ---------------------------------------------------------------------------
# AI versus rules
# ---------------------------------------------------------------------------
def _recommendation(source: StepState | dict[str, Any]) -> str | None:
    output = source.output or {} if isinstance(source, StepState) else source
    for key in RECOMMENDATION_KEYS:
        value = output.get(key)
        if isinstance(value, str) and value:
            return value.upper()
    return None


def _disagreements(
    ai: list[tuple[StepDefinition, StepState]],
    policy: list[tuple[StepDefinition, StepState]],
) -> list[Disagreement]:
    """Where a model recommended one thing and the rules did another.

    Worth stating outright rather than leaving to be noticed, because it is
    the platform's central design decision showing up in a single case: a
    recommendation is evidence, and only the rules authorize.
    """
    equivalent = {
        "APPROVE": PolicyEffect.ALLOW,
        "REJECT": PolicyEffect.DENY,
        "MANUAL_REVIEW": PolicyEffect.REQUIRE_APPROVAL,
    }

    found: list[Disagreement] = []
    for gate, gate_state in policy:
        effect = (gate_state.output or {}).get("effect")
        if not effect:
            continue
        for agent, agent_state in ai:
            said = _recommendation(agent_state)
            if said is None:
                continue
            expected = equivalent.get(said)
            if expected is None or str(expected) == effect:
                continue
            found.append(
                Disagreement(
                    ai_step=agent.id,
                    ai_said=said,
                    rules_step=gate.id,
                    rules_said=str(effect),
                    resolution=(
                        f"The rules decided. '{agent.display_name}' recommends and "
                        f"'{gate.display_name}' authorizes, so a recommendation cannot "
                        "widen what the rules permit -- only inform it."
                    ),
                )
            )
    return found


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _would_have(
    steps: list[tuple[str, StepDefinition, StepState | None]],
) -> list[SimulatedEffect]:
    """The effects a live run would have had, gathered from the held-back steps."""
    effects: list[SimulatedEffect] = []
    for _, spec, state in steps:
        output = (state.output if state else None) or {}
        if not output.get("simulated"):
            continue
        if output.get("would_require_approval"):
            effects.append(
                SimulatedEffect(
                    step=spec.id,
                    kind="approval",
                    target=str(output.get("title", spec.display_name)),
                    detail={"required_roles": output.get("required_roles", [])},
                )
            )
        elif output.get("would_call"):
            effects.append(
                SimulatedEffect(
                    step=spec.id,
                    kind="call",
                    target=str(output["would_call"]),
                    detail={"arguments": output.get("would_send", {})},
                )
            )
    return effects

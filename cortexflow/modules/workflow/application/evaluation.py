"""How the agents are actually performing.

A read model over runs that already happened. It computes nothing the engine
did not record and holds no state of its own.

The honest boundary of this module is worth stating plainly, because an
evaluation page is exactly where a plausible-looking number does the most
damage:

**Latency, cost, confidence and failure rate are measured.** They come from
what the executor wrote when each step finished.

**Accuracy is not, and cannot be here.** Accuracy is agreement with a
known-correct answer, and nothing in this system holds one -- there is no
labelled dataset and no reviewer marking an extraction right or wrong. Any
"96.2% extraction accuracy" would be an invented denominator.

What replaces it is **agreement with policy**: how often the agent's
recommendation matched what the rules then authorised. That is not accuracy --
the policy engine is not ground truth for whether the agent read a document
correctly -- but it is real, and it is arguably the number most worth
watching, because a model drifting away from the rules means either the model
or the rules have moved, and both are worth knowing early.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.workflow.domain.definition import StepType, WorkflowDefinition
from cortexflow.modules.workflow.domain.models import StepStatus, Workflow

MAX_REPORTED_RUNS = 200


class AgentPerformance(BaseModel):
    """One agent, across every step it ran in the window."""

    model_config = ConfigDict(frozen=True)

    name: str
    invocations: int
    failures: int
    failure_rate: float

    mean_confidence: float | None = None
    """What the agent said about its own certainty. Self-reported, not checked."""

    mean_latency_ms: int | None = None
    p95_latency_ms: int | None = None

    total_tokens: int = 0
    mean_tokens: int = 0
    models: tuple[str, ...] = ()
    """Which models actually served these steps."""


class EvaluationRun(BaseModel):
    """One agent step, with what the rules made of it."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    case_id: str
    workflow: str
    step: str
    agent: str
    at: str
    status: str

    confidence: float | None = None
    latency_ms: int | None = None
    tokens: int = 0
    model: str = ""

    recommendation: str = ""
    decision: str = ""
    agreed: bool | None = None
    """None when only one of the two spoke, so there was nothing to compare."""


class TrendPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    invocations: int
    mean_confidence: float | None = None
    mean_latency_ms: int | None = None
    agreement: float | None = None


class AiEvaluation(BaseModel):
    """The whole picture, with its limits stated in the payload itself."""

    model_config = ConfigDict(frozen=True)

    window_days: int | None = None
    """The window these figures describe, or None for everything retained."""

    runs_examined: int
    agent_steps: int

    agreement_rate: float | None = None
    """Share of decided cases where the model and the rules concurred."""

    disagreements: int = 0
    mean_confidence: float | None = None
    mean_latency_ms: int | None = None
    total_tokens: int = 0
    mean_tokens_per_case: int = 0

    policy_violations: int = 0
    """Structurally zero: high-risk tools are never offered to an agent."""

    agents: tuple[AgentPerformance, ...] = ()
    runs: tuple[EvaluationRun, ...] = ()
    trend: tuple[TrendPoint, ...] = ()

    unmeasured: tuple[str, ...] = ()
    """What this platform cannot measure yet, and why.

    Carried in the response rather than left to the UI, so any client showing
    these figures also has the caveat to hand.
    """


# Agreement between what a model advised and what the rules then authorised.
_CONCUR = {
    ("APPROVE", "ALLOW"),
    ("REJECT", "DENY"),
    ("MANUAL_REVIEW", "REQUIRE_APPROVAL"),
}

UNMEASURED = (
    "Extraction accuracy needs a known-correct extraction to compare against; "
    "nothing records one.",
    "Recommendation quality needs a labelled outcome for each case.",
    "Tool selection accuracy needs to know which tool was the right one.",
)


def evaluate(
    pairs: list[tuple[Workflow, WorkflowDefinition]], *, window_days: int | None = None
) -> AiEvaluation:
    """Read a set of runs as agent performance."""
    rows: list[EvaluationRun] = []
    by_agent: dict[str, list[EvaluationRun]] = defaultdict(list)

    for workflow, definition in pairs:
        decision = _decision(workflow)
        for state in workflow.steps.values():
            spec = _spec(definition, state.step_id)
            if spec is None or spec.type is not StepType.AGENT:
                continue
            if state.status is StepStatus.PENDING:
                continue

            cost = state.cost or {}
            tokens = int(cost.get("prompt_tokens") or 0) + int(
                cost.get("completion_tokens") or 0
            )
            recommendation = _recommendation(state.output)
            row = EvaluationRun(
                workflow_id=workflow.workflow_id,
                case_id=workflow.fan_out_key or workflow.workflow_id,
                workflow=workflow.definition_name,
                step=state.step_id,
                agent=spec.agent or "unknown",
                at=(state.ended_at or workflow.created_at).isoformat(),
                status=str(state.status),
                confidence=state.confidence,
                latency_ms=state.duration_ms,
                tokens=tokens,
                model=str(cost.get("model") or ""),
                recommendation=recommendation,
                decision=decision,
                agreed=_agreed(recommendation, decision),
            )
            rows.append(row)
            by_agent[row.agent].append(row)

    decided = [r for r in rows if r.agreed is not None]
    agreement = (
        sum(1 for r in decided if r.agreed) / len(decided) if decided else None
    )

    return AiEvaluation(
        window_days=window_days,
        runs_examined=len(pairs),
        agent_steps=len(rows),
        agreement_rate=agreement,
        disagreements=sum(1 for r in decided if r.agreed is False),
        mean_confidence=_mean([r.confidence for r in rows if r.confidence is not None]),
        mean_latency_ms=_mean_int([r.latency_ms for r in rows if r.latency_ms is not None]),
        total_tokens=sum(r.tokens for r in rows),
        mean_tokens_per_case=_mean_int([r.tokens for r in rows if r.tokens]) or 0,
        policy_violations=0,
        agents=tuple(
            _performance(name, agent_rows) for name, agent_rows in sorted(by_agent.items())
        ),
        # Newest first: the run somebody is asking about is usually the last one.
        runs=tuple(sorted(rows, key=lambda r: r.at, reverse=True)[:MAX_REPORTED_RUNS]),
        trend=_trend(rows),
        unmeasured=UNMEASURED,
    )


# ----------------------------------------------------------------------
def _spec(definition: WorkflowDefinition, step_id: str) -> Any:
    for step in definition.steps:
        if step.id == step_id:
            return step
    return None


def _decision(workflow: Workflow) -> str:
    for state in workflow.steps.values():
        output = state.output or {}
        if "effect" in output and "matched_rules" in output:
            return str(output["effect"])
    return ""


def _recommendation(output: dict[str, Any] | None) -> str:
    if not output or "recommendation" not in output:
        return ""
    return str(output["recommendation"]).upper()


def _agreed(recommendation: str, decision: str) -> bool | None:
    """Whether the two concurred, or None when only one of them spoke."""
    if not recommendation or not decision:
        return None
    return (recommendation, decision) in _CONCUR


def _performance(name: str, rows: list[EvaluationRun]) -> AgentPerformance:
    failures = sum(1 for r in rows if r.status == str(StepStatus.FAILED))
    latencies = sorted(r.latency_ms for r in rows if r.latency_ms is not None)
    tokens = [r.tokens for r in rows if r.tokens]

    return AgentPerformance(
        name=name,
        invocations=len(rows),
        failures=failures,
        failure_rate=failures / len(rows) if rows else 0.0,
        mean_confidence=_mean([r.confidence for r in rows if r.confidence is not None]),
        mean_latency_ms=_mean_int(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
        total_tokens=sum(tokens),
        mean_tokens=_mean_int(tokens) or 0,
        models=tuple(sorted({r.model for r in rows if r.model})),
    )


def _trend(rows: list[EvaluationRun]) -> tuple[TrendPoint, ...]:
    """One point per day the agents ran, oldest first."""
    by_day: dict[str, list[EvaluationRun]] = defaultdict(list)
    for row in rows:
        by_day[row.at[:10]].append(row)

    points: list[TrendPoint] = []
    for day in sorted(by_day):
        day_rows = by_day[day]
        decided = [r for r in day_rows if r.agreed is not None]
        points.append(
            TrendPoint(
                date=day,
                invocations=len(day_rows),
                mean_confidence=_mean(
                    [r.confidence for r in day_rows if r.confidence is not None]
                ),
                mean_latency_ms=_mean_int(
                    [r.latency_ms for r in day_rows if r.latency_ms is not None]
                ),
                agreement=(
                    sum(1 for r in decided if r.agreed) / len(decided) if decided else None
                ),
            )
        )
    return tuple(points)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _mean_int(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _percentile(ordered: list[int], fraction: float) -> int | None:
    """The value below which that fraction of observations fall.

    Nearest-rank, because interpolating between two measured latencies invents
    a latency nothing actually took.
    """
    if not ordered:
        return None
    index = min(round(fraction * len(ordered) + 0.5) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def utc_day(moment: datetime) -> str:
    return moment.date().isoformat()

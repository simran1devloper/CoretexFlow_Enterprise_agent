"""What the platform actually did, counted.

A read model over runs that already happened, in the same spirit as
:mod:`~cortexflow.modules.workflow.application.evaluation`: it computes
nothing the engine did not record, holds no state, and is never consulted
while a workflow runs.

An analytics page is where a plausible-looking number does the most damage,
because it is the screen people quote in meetings. So the boundary is drawn
explicitly here rather than left to whoever renders it:

**Measured.** Volume, outcomes, how often a case reached the end without a
person, how long cases took, which policy rules sent work to a human, and
which workflows fail. All of it comes from timestamps and step output the
engine wrote.

**Not measured, and deliberately absent.** Hours saved, cost avoided,
headcount equivalents. Every one of those needs a baseline -- what the same
work would have cost done by hand -- and nothing in this system holds one.
A "saved 412 hours" tile is a number multiplied by a guess, and it is the
figure most likely to be repeated outside the room it was invented in.

Three counting decisions worth stating, because each one is a place the
figures could quietly lie:

* **Simulated runs are excluded from everything** and reported separately.
  A simulated batch of five hundred claims is not five hundred claims
  processed, and letting it into throughput would make the headline number
  something anyone could inflate on purpose.
* **A fan-out parent is a batch, not a case.** Counting the batch *and* its
  children double-counts the work, so parents are counted as batches and only
  the children are cases -- the same reading the Cases screen applies.
* **Cycle time is reported twice**: across everything concluded, and across
  cases decided without a person. Mixing a four-second automated run with a
  three-day wait for an approver produces a median that describes neither, and
  the gap between the two numbers is the more useful fact.

The automation rate deserves its own paragraph, because the obvious definition
is wrong in a way that only shows up once real data is in front of you. Taken
as "settled cases that finished without a person", it read **100% while four
cases sat in an approval queue** -- they had not settled, so they were in
neither half of the fraction, and a failed run counted as an automation success
because nobody had been asked about it either.

So the denominator is *cases the platform formed a view on*: those it decided
alone, plus every case where it raised an approval, answered or not. Asking for
a person is the platform declining to decide, and that counts against the rate
from the moment it happens rather than whenever the approver gets round to it.
Failures are in neither half -- a run that broke is an exception, reported as
one, and folding it in either direction would be a number about reliability
wearing the label of a number about autonomy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict

from cortexflow.modules.workflow.domain.definition import StepType, WorkflowDefinition
from cortexflow.modules.workflow.domain.models import Workflow, WorkflowStatus

MAX_REPORTED_REASONS = 12

CAVEATS = (
    "Hours and cost saved are not shown: both need a baseline for what this "
    "work would have cost done by hand, and nothing records one.",
    "Runs that failed are counted in neither half of the automation rate. A "
    "run that broke reached no verdict, and folding it in either direction "
    "would make a reliability figure wear an autonomy label.",
    "Cycle time for cases that needed a person includes the time that person "
    "took to respond, which is not the platform's to control.",
    "Figures describe the runs still retained, not all history.",
)


class OutcomeCounts(BaseModel):
    """Where cases ended up. Terminal states, plus the two waiting ones."""

    model_config = ConfigDict(frozen=True)

    completed: int = 0
    rejected: int = 0
    """Refused by policy. An outcome, not a failure."""

    failed: int = 0
    cancelled: int = 0
    awaiting_approval: int = 0
    awaiting_recovery: int = 0
    """Exhausted retries and parked for an operator."""

    in_flight: int = 0


class DayPoint(BaseModel):
    """One day of throughput, oldest first."""

    model_config = ConfigDict(frozen=True)

    date: str
    cases: int = 0
    """Cases created that day, whatever became of them."""

    concluded: int = 0
    """Reached a verdict: completed, or refused by policy."""

    decided_alone: int = 0
    needed_a_person: int = 0
    rejected: int = 0
    failed: int = 0


class WorkflowStats(BaseModel):
    """One workflow definition, across the window."""

    model_config = ConfigDict(frozen=True)

    workflow: str
    cases: int
    concluded: int
    decided_alone: int
    automation_rate: float | None = None
    needed_a_person: int = 0
    rejected: int = 0
    failed: int = 0
    failure_rate: float | None = None
    median_cycle_ms: int | None = None
    p95_cycle_ms: int | None = None


class EscalationReason(BaseModel):
    """One rule, and how much human attention it is responsible for.

    The actionable number on this page. "Sixty per cent of your manual
    reviews come from one confidence threshold" is a thing somebody can go and
    change; "automation rate is 82%" is not.
    """

    model_config = ConfigDict(frozen=True)

    ruleset: str
    rule_id: str
    cases: int
    share: float
    """Of all cases whose policy step asked for a person or refused."""

    effect: str
    """REQUIRE_APPROVAL or DENY -- whether this rule stops work or queues it."""


class Analytics(BaseModel):
    """The whole picture, with its limits carried in the payload."""

    model_config = ConfigDict(frozen=True)

    window_days: int | None = None
    """The window these figures describe, or None for everything retained."""

    runs_examined: int = 0
    """Runs read, before batches and simulations were set aside."""

    cases: int = 0
    batches: int = 0
    simulated: int = 0
    """Excluded from every figure above and below. Reported so the exclusion
    is visible rather than silent."""

    concluded: int = 0
    """Cases that reached a verdict: completed, or refused by policy."""

    decided_alone: int = 0
    """Concluded without anyone being asked."""

    needed_a_person: int = 0
    """An approval was raised, whether or not it has been answered."""

    automation_rate: float | None = None
    """``decided_alone / (decided_alone + needed_a_person)``.

    Failures are in neither half; see the module docstring for why the obvious
    denominator is wrong. None rather than zero when the platform has formed no
    views yet -- an empty platform has no automation rate, and 0% would read as
    a failing one.
    """

    median_cycle_ms: int | None = None
    p95_cycle_ms: int | None = None
    median_decided_alone_cycle_ms: int | None = None
    """The same figure for cases nobody was asked about -- the platform's own speed."""

    outcomes: OutcomeCounts = OutcomeCounts()
    trend: tuple[DayPoint, ...] = ()
    by_workflow: tuple[WorkflowStats, ...] = ()
    escalation_reasons: tuple[EscalationReason, ...] = ()

    caveats: tuple[str, ...] = ()


class _Case(BaseModel):
    """One run, reduced to what the counting needs."""

    model_config = ConfigDict(frozen=True)

    workflow: str
    day: str
    status: WorkflowStatus
    concluded: bool
    """Reached a verdict. A run that broke reached none."""

    touched_by_a_person: bool
    cycle_ms: int | None
    escalations: tuple[tuple[str, str, str], ...]
    """(ruleset, rule_id, effect) for each rule that asked for a person."""


def analyse(
    pairs: list[tuple[Workflow, WorkflowDefinition]], *, window_days: int | None = None
) -> Analytics:
    """Read a set of runs as throughput, outcomes and cycle time."""
    cases: list[_Case] = []
    batches = simulated = 0

    for workflow, definition in pairs:
        if workflow.execution_mode.is_simulation:
            simulated += 1
            continue
        if _fanned_out(workflow):
            batches += 1
            continue
        cases.append(_read(workflow, definition))

    concluded = [c for c in cases if c.concluded]
    alone = [c for c in concluded if not c.touched_by_a_person]
    asked = [c for c in cases if c.touched_by_a_person]
    cycles = sorted(c.cycle_ms for c in concluded if c.cycle_ms is not None)
    alone_cycles = sorted(c.cycle_ms for c in alone if c.cycle_ms is not None)

    return Analytics(
        window_days=window_days,
        runs_examined=len(pairs),
        cases=len(cases),
        batches=batches,
        simulated=simulated,
        concluded=len(concluded),
        decided_alone=len(alone),
        needed_a_person=len(asked),
        automation_rate=_rate(len(alone), len(asked)),
        median_cycle_ms=_percentile(cycles, 0.5),
        p95_cycle_ms=_percentile(cycles, 0.95),
        median_decided_alone_cycle_ms=_percentile(alone_cycles, 0.5),
        outcomes=_outcomes(cases),
        trend=_trend(cases),
        by_workflow=_by_workflow(cases),
        escalation_reasons=_escalations(cases),
        caveats=CAVEATS,
    )


# ----------------------------------------------------------------------
def _fanned_out(workflow: Workflow) -> bool:
    """Whether this run started children, making it a batch rather than a case."""
    return any(state.spawned for state in workflow.steps.values())


def _read(workflow: Workflow, definition: WorkflowDefinition) -> _Case:
    # Concluded, not merely terminal: a run that failed or was cancelled
    # stopped without the platform reaching a view, and averaging it in either
    # direction would make a reliability problem look like an autonomy figure.
    concluded = workflow.status in {WorkflowStatus.COMPLETED, WorkflowStatus.REJECTED}
    return _Case(
        workflow=workflow.definition_name,
        day=workflow.created_at.date().isoformat(),
        status=workflow.status,
        concluded=concluded,
        touched_by_a_person=_touched_by_a_person(workflow),
        # `updated_at` is the last write, which for a settled run is the write
        # that settled it. The alternative -- the latest step `ended_at` --
        # misses a run that ended by policy denial, where nothing ran after.
        cycle_ms=(
            int((workflow.updated_at - workflow.created_at).total_seconds() * 1000)
            if concluded
            else None
        ),
        escalations=_escalations_in(workflow, definition),
    )


def _touched_by_a_person(workflow: Workflow) -> bool:
    """Whether a human was actually asked to act on this case.

    Keyed on ``approval_id``, which the engine sets at the moment it creates
    an approval record -- so this is "somebody was asked", not "somebody could
    have been".

    Both halves of that matter, and each was a wrong answer first:

    * Reading the *workflow status* misses the case that waited for an
      approver on Tuesday and was approved on Wednesday. It reads COMPLETED
      now, and counting it as automated would be the most flattering mistake
      this page could make.
    * Reading the approval step's *status* over-counts in the other
      direction. When a run fails, outstanding steps are cancelled, so a
      CANCELLED approval step means nobody was ever asked -- and counting it
      made six failed runs report as six escalations.

    ``approval_id`` survives both: it is set when the request is raised and
    never cleared, whatever the step ends as.
    """
    return any(
        state.type is StepType.HUMAN_APPROVAL and state.approval_id is not None
        for state in workflow.steps.values()
    )


def _escalations_in(
    workflow: Workflow, definition: WorkflowDefinition
) -> tuple[tuple[str, str, str], ...]:
    """The rules that sent this case to a person, or refused it.

    Only the rules that actually matched, and only where the effect was not
    ALLOW: a rule that fired and allowed the case is not a reason anybody had
    to look at it.
    """
    by_id = {step.id: step for step in definition.steps}
    found: list[tuple[str, str, str]] = []

    for state in workflow.steps.values():
        spec = by_id.get(state.step_id)
        if spec is None or spec.type is not StepType.POLICY:
            continue
        output: dict[str, Any] = state.output or {}
        effect = str(output.get("effect") or "")
        if effect in {"", "ALLOW"}:
            continue
        ruleset = str(output.get("ruleset") or spec.ruleset or "")
        matched = output.get("matched_rules") or []
        if not matched:
            # No rule matched, so the ruleset default applied. That is a real
            # reason cases reach a person and it has no rule id of its own.
            found.append((ruleset, "(ruleset default)", effect))
            continue
        found.extend((ruleset, str(rule_id), effect) for rule_id in matched)
    return tuple(found)


def _outcomes(cases: list[_Case]) -> OutcomeCounts:
    counts = Counter(case.status for case in cases)
    return OutcomeCounts(
        completed=counts[WorkflowStatus.COMPLETED],
        rejected=counts[WorkflowStatus.REJECTED],
        failed=counts[WorkflowStatus.FAILED],
        cancelled=counts[WorkflowStatus.CANCELLED],
        awaiting_approval=counts[WorkflowStatus.WAITING_FOR_APPROVAL],
        awaiting_recovery=counts[WorkflowStatus.WAITING_FOR_RECOVERY],
        in_flight=counts[WorkflowStatus.PENDING] + counts[WorkflowStatus.RUNNING],
    )


def _trend(cases: list[_Case]) -> tuple[DayPoint, ...]:
    """One point per day that saw a case, oldest first.

    Days with no cases are absent rather than zero-filled: this reads runs
    that are still retained, so a gap means "nothing here to read", and
    drawing it as a zero would assert a quiet day the data cannot support.
    """
    by_day: dict[str, list[_Case]] = defaultdict(list)
    for case in cases:
        by_day[case.day].append(case)

    points: list[DayPoint] = []
    for day in sorted(by_day):
        rows = by_day[day]
        concluded = [c for c in rows if c.concluded]
        points.append(
            DayPoint(
                date=day,
                cases=len(rows),
                concluded=len(concluded),
                decided_alone=sum(1 for c in concluded if not c.touched_by_a_person),
                needed_a_person=sum(1 for c in rows if c.touched_by_a_person),
                rejected=sum(1 for c in rows if c.status is WorkflowStatus.REJECTED),
                failed=sum(1 for c in rows if c.status is WorkflowStatus.FAILED),
            )
        )
    return tuple(points)


def _by_workflow(cases: list[_Case]) -> tuple[WorkflowStats, ...]:
    by_name: dict[str, list[_Case]] = defaultdict(list)
    for case in cases:
        by_name[case.workflow].append(case)

    stats: list[WorkflowStats] = []
    for name, rows in sorted(by_name.items()):
        concluded = [c for c in rows if c.concluded]
        alone = [c for c in concluded if not c.touched_by_a_person]
        asked = [c for c in rows if c.touched_by_a_person]
        cycles = sorted(c.cycle_ms for c in concluded if c.cycle_ms is not None)
        failed = sum(1 for c in rows if c.status is WorkflowStatus.FAILED)
        stats.append(
            WorkflowStats(
                workflow=name,
                cases=len(rows),
                concluded=len(concluded),
                decided_alone=len(alone),
                automation_rate=_rate(len(alone), len(asked)),
                needed_a_person=len(asked),
                rejected=sum(1 for c in rows if c.status is WorkflowStatus.REJECTED),
                failed=failed,
                # Over every case, not just concluded ones -- a run that failed
                # is precisely what this rate is about.
                failure_rate=round(failed / len(rows), 4) if rows else None,
                median_cycle_ms=_percentile(cycles, 0.5),
                p95_cycle_ms=_percentile(cycles, 0.95),
            )
        )
    # Busiest first: the workflow carrying the most work is the one whose
    # automation rate is worth arguing about.
    return tuple(sorted(stats, key=lambda s: s.cases, reverse=True))


def _escalations(cases: list[_Case]) -> tuple[EscalationReason, ...]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for case in cases:
        # Per case, not per occurrence: a rule matching twice in one case is
        # still one case a person had to look at.
        counts.update(set(case.escalations))

    total = sum(1 for case in cases if case.escalations)
    if not total:
        return ()

    reasons = [
        EscalationReason(
            ruleset=ruleset,
            rule_id=rule_id,
            cases=count,
            share=round(count / total, 4),
            effect=effect,
        )
        for (ruleset, rule_id, effect), count in counts.items()
    ]
    reasons.sort(key=lambda r: (-r.cases, r.rule_id))
    return tuple(reasons[:MAX_REPORTED_REASONS])


def _rate(decided_alone: int, needed_a_person: int) -> float | None:
    """Of the cases the platform formed a view on, the share it settled itself.

    Asking for a person is the platform declining to decide, so it counts
    against the rate from the moment the approval is raised rather than
    whenever the approver gets round to answering. Otherwise a queue of
    untouched approvals reads as a perfect score.
    """
    views = decided_alone + needed_a_person
    return round(decided_alone / views, 4) if views else None


def _percentile(ordered: list[int], fraction: float) -> int | None:
    """The value below which that fraction of observations fall.

    Nearest-rank, because interpolating between two measured durations invents
    a duration nothing actually took. The same choice the evaluation read
    model makes, for the same reason.
    """
    if not ordered:
        return None
    index = min(round(fraction * len(ordered) + 0.5) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]

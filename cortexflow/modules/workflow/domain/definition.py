"""Declarative workflow definitions.

A definition is data, loaded from YAML and validated once at load time.  The
engine never executes author-supplied code; it only walks this graph.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortexflow.shared.errors import WorkflowDefinitionError
from cortexflow.shared.expressions import (
    iter_placeholder_expressions,
    validate_expression,
)
from cortexflow.shared.identity import Department, Role


class StepType(StrEnum):
    AGENT = "agent"
    """Delegates to an LLM-backed agent via the agent runtime."""

    TOOL = "tool"
    """Invokes a registered enterprise tool through the tool registry."""

    HUMAN_APPROVAL = "human_approval"
    """Suspends the workflow durably until a human decides."""

    POLICY = "policy"
    """Evaluates a deterministic policy ruleset -- no LLM involved."""

    FAN_OUT = "fan_out"
    """Starts one child workflow per item, then waits for all of them.

    The children are ordinary workflows: their own steps, retries, approvals,
    audit trail and replay. That is the point -- a batch of a thousand
    invoices is a thousand cases, each of which can fail, be approved or be
    retried on its own, rather than one case that succeeds or fails together.
    """


class JoinPolicy(StrEnum):
    ALL = "all"
    """Run when every dependency succeeded (skipped dependencies skip this step)."""

    ANY = "any"
    """Run when at least one dependency succeeded -- used to merge branches."""


class Queue(StrEnum):
    """Logical execution queues.

    Queues are per *workload shape*, not per step, so that expensive extraction
    work cannot starve cheap validation work and each can scale independently.
    """

    CONTROL = "control"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    DECISION = "decision"
    REPORTING = "reporting"
    INTEGRATION = "integration"
    NOTIFICATION = "notification"


class RetryPolicySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int | None = None
    backoff_base_seconds: float | None = None
    backoff_max_seconds: float | None = None
    retry_on_unknown: bool = True


class ApprovalSpec(BaseModel):
    """Who may decide a human-approval step, and what happens if nobody does."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = "Approval required"
    description: str = ""
    required_roles: tuple[Role, ...] = ()
    timeout_seconds: int | None = None
    on_timeout: str = "escalate"  # escalate | reject | approve
    allow_request_info: bool = True

    @model_validator(mode="after")
    def _check_timeout_action(self) -> ApprovalSpec:
        if self.on_timeout not in {"escalate", "reject", "approve"}:
            raise ValueError("on_timeout must be one of escalate|reject|approve")
        return self


_RESERVED_QUEUES = frozenset({Queue.CONTROL, Queue.NOTIFICATION})
"""Queues that carry platform messages, never step commands."""


class StepDefinition(BaseModel):
    """One node of the workflow DAG."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: StepType
    name: str = ""
    description: str = ""

    agent: str | None = None
    tool: str | None = None
    ruleset: str | None = None
    approval: ApprovalSpec | None = None

    depends_on: tuple[str, ...] = ()
    join: JoinPolicy = JoinPolicy.ALL
    when: str | None = None
    """Guard expression; a step whose guard is false is SKIPPED, not failed."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    """Bindings resolved from the workflow context via ``${{ ... }}``."""

    queue: Queue | None = None
    timeout_seconds: int = 300
    retry: RetryPolicySpec = Field(default_factory=RetryPolicySpec)
    critical: bool = True
    """A non-critical step's failure degrades the workflow instead of failing it."""

    idempotent: bool = True
    """False for steps whose retry must never be automatic (unsafe side effects)."""

    fan_out: FanOutSpec | None = None
    """Required for a FAN_OUT step, meaningless on any other."""

    @model_validator(mode="after")
    def _check_shape(self) -> StepDefinition:
        required = {
            StepType.AGENT: "agent",
            StepType.TOOL: "tool",
            StepType.POLICY: "ruleset",
            StepType.HUMAN_APPROVAL: None,
            StepType.FAN_OUT: "fan_out",
        }[self.type]
        if required and not getattr(self, required):
            raise ValueError(f"step '{self.id}' of type '{self.type}' requires '{required}'")
        if self.when:
            validate_expression(self.when)
        for expr in iter_placeholder_expressions(self.inputs):
            validate_expression(expr)
        if self.id in self.depends_on:
            raise ValueError(f"step '{self.id}' depends on itself")
        if self.fan_out is not None:
            validate_expression(self.fan_out.collection)
        if self.queue in _RESERVED_QUEUES and self.type in {StepType.AGENT, StepType.TOOL}:
            # CONTROL carries orchestration messages and NOTIFICATION carries
            # delivery requests. Neither carries step commands, so a step
            # routed there would be dispatched to a queue nobody polls.
            raise ValueError(
                f"step '{self.id}' cannot be routed to the reserved "
                f"'{self.queue.value}' queue"
            )
        return self

    @property
    def display_name(self) -> str:
        return self.name or self.id.replace("_", " ").title()

    @property
    def effective_queue(self) -> Queue:
        """Queue routing, defaulted from the step's shape."""
        if self.queue is not None:
            return self.queue
        match self.type:
            case StepType.TOOL:
                return Queue.INTEGRATION
            case StepType.POLICY | StepType.HUMAN_APPROVAL | StepType.FAN_OUT:
                return Queue.CONTROL
            case _:
                return Queue.DECISION


MAX_FAN_OUT_CHILDREN = 500
"""The most children one fan-out step may start.

A bound is not optional here: without one, a single uploaded file decides how
much work the platform creates, and a mis-mapped collection becomes an outage
rather than an error.
"""


class FanOutSpec(BaseModel):
    """How a fan-out step turns one collection into many workflows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow: str
    """The child definition to start for each item."""

    collection: str
    """An expression resolving to a list -- usually an upstream step's rows.

    A bare expression, as ``when`` is, not a ``${{ }}`` template: it yields one
    value rather than a rendered string, and writing it as a template would
    invite someone to interpolate it into text, which a list cannot be.
    """

    item_key: str = ""
    """Which field of an item names it, for the report and the child's label."""

    item_as: str = "item"
    """Where a non-dict item lands in the child's input.

    A dict item becomes the child's input directly, so this is only consulted
    for lists of plain values -- ids, filenames, strings.
    """

    max_children: int = Field(default=MAX_FAN_OUT_CHILDREN, ge=1, le=MAX_FAN_OUT_CHILDREN)

    await_approvals: bool = True
    """Whether the parent waits for children that are held for a person.

    True keeps "completed" honest: the batch is finished only when every case
    in it is, however many days that takes -- which is the point of durable
    workflows. False settles the parent once nothing can move without a human,
    reporting how many are outstanding, for the batch whose value is the
    summary rather than the completion.
    """

    tolerate_failures: bool = True
    """Whether a failed child fails the parent.

    True by default, because the useful answer to "did these thousand invoices
    go through" is a breakdown, not an exception. A parent that must be
    all-or-nothing sets this False.
    """


class WorkflowDefinition(BaseModel):
    """A versioned, immutable workflow template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int = 1
    department: Department
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    steps: tuple[StepDefinition, ...]
    labels: dict[str, str] = Field(default_factory=dict)

    requires_runtime: str | None = None
    """An optional agent runtime this definition depends on.

    Some workflows are written for a runtime that is not always configured --
    the Semantic Kernel runtime registers agents the native one does not. A
    definition declares that dependency here, and the loader skips it when
    that runtime is inactive, rather than failing the whole deployment over a
    workflow nobody asked for.
    """

    @model_validator(mode="after")
    def _validate_graph(self) -> WorkflowDefinition:
        if not self.steps:
            raise ValueError("workflow must declare at least one step")
        ids = [s.id for s in self.steps]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate step ids: {sorted(duplicates)}")
        known = set(ids)
        for step in self.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(f"step '{step.id}' depends on unknown steps: {sorted(unknown)}")
        _assert_acyclic({s.id: set(s.depends_on) for s in self.steps})
        return self

    @property
    def key(self) -> str:
        """Stable identity of this template version."""
        return f"{self.name}@{self.version}"

    @property
    def step_map(self) -> dict[str, StepDefinition]:
        return {step.id: step for step in self.steps}

    def step(self, step_id: str) -> StepDefinition:
        try:
            return self.step_map[step_id]
        except KeyError as exc:
            raise WorkflowDefinitionError(
                "Unknown step", workflow=self.key, step_id=step_id
            ) from exc

    def dependents_of(self, step_id: str) -> tuple[str, ...]:
        return tuple(s.id for s in self.steps if step_id in s.depends_on)

    def roots(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.steps if not s.depends_on)

    def topological_order(self) -> tuple[str, ...]:
        """Deterministic topological order, used for display and validation."""
        pending = {s.id: set(s.depends_on) for s in self.steps}
        order: list[str] = []
        while pending:
            ready = sorted(sid for sid, deps in pending.items() if not deps)
            if not ready:  # pragma: no cover - guarded by _assert_acyclic
                raise WorkflowDefinitionError("Cycle detected", workflow=self.key)
            for sid in ready:
                order.append(sid)
                del pending[sid]
            for deps in pending.values():
                deps.difference_update(ready)
        return tuple(order)


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Depth-first cycle detection reporting the offending path."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GREY
        for dep in sorted(graph[node]):
            if colour[dep] == GREY:
                cycle = [*path, node, dep]
                raise ValueError(f"workflow contains a cycle: {' -> '.join(cycle)}")
            if colour[dep] == WHITE:
                visit(dep, [*path, node])
        colour[node] = BLACK

    for node in sorted(graph):
        if colour[node] == WHITE:
            visit(node, [])

"""Workflow application service.

The API layer talks to this, never to the engine or repositories directly.
Creating a workflow is deliberately cheap: validate, persist, publish a single
control message, return.  All the real work happens asynchronously, which is
why closing the browser has no effect on execution.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.workflow.application.analytics import Analytics, analyse
from cortexflow.modules.workflow.application.cases import (
    CaseDetail,
    CaseSummary,
    detail,
    summarise,
)
from cortexflow.modules.workflow.application.definitions import WorkflowDefinitionRegistry
from cortexflow.modules.workflow.application.evaluation import AiEvaluation, evaluate
from cortexflow.modules.workflow.application.explanation import (
    RunExplanation,
    explain,
)
from cortexflow.modules.workflow.domain.definition import (
    Queue,
    StepType,
    WorkflowDefinition,
)
from cortexflow.modules.workflow.domain.envelope import AdvanceWorkflow, Message, MessageHeaders
from cortexflow.modules.workflow.domain.models import (
    ExecutionMode,
    StepState,
    Workflow,
    WorkflowStatus,
)
from cortexflow.modules.workflow.ports.messaging import MessageBus
from cortexflow.modules.workflow.ports.repositories import (
    WorkflowPage,
    WorkflowRepository,
)
from cortexflow.shared.clock import Clock
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.identity import Principal
from cortexflow.shared.ids import MESSAGE, WORKFLOW, IdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import span
from cortexflow.shared.security.rbac import Permission, require_permission, require_tenant

logger = get_logger(__name__)

# Declared at module scope: the service's own ``list`` method shadows the
# builtin inside the class body.
DefinitionList = list[WorkflowDefinition]


class WorkflowService:
    def __init__(
        self,
        *,
        workflows: WorkflowRepository,
        definitions: WorkflowDefinitionRegistry,
        bus: MessageBus,
        audit: AuditTrail,
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._workflows = workflows
        self._definitions = definitions
        self._bus = bus
        self._audit = audit
        self._clock = clock
        self._ids = ids

    async def create(
        self,
        principal: Principal,
        *,
        definition_name: str,
        input_data: dict[str, Any],
        definition_version: int | None = None,
        correlation_id: str = "",
        labels: dict[str, str] | None = None,
        execution_mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> Workflow:
        """Create and enqueue a workflow.

        ``execution_mode`` decides whether this run may change anything
        outside CortexFlow. It is fixed here, at creation, and never changed
        afterwards: a run that could be promoted from rehearsal to real
        mid-flight would have no honest answer to "did this happen?".
        """
        require_permission(principal, Permission.WORKFLOW_CREATE)
        definition = self._definitions.get(
            definition_name, definition_version, tenant_id=principal.tenant_id
        )
        _require_department(principal, definition)
        _validate_input(definition, input_data)

        workflow_id = self._ids.new_id(WORKFLOW)
        workflow = Workflow(
            workflow_id=workflow_id,
            tenant_id=principal.tenant_id,
            definition_name=definition.name,
            # Pinned at creation: publishing a new version never changes the
            # behaviour of a workflow that is already running.
            definition_version=definition.version,
            status=WorkflowStatus.PENDING,
            created_by=principal.subject,
            correlation_id=correlation_id or workflow_id,
            execution_mode=execution_mode,
            labels=labels or {},
            input=input_data,
            steps={
                step.id: StepState(step_id=step.id, type=step.type)
                for step in definition.steps
            },
        )

        with span(
            "workflow.create",
            **{"workflow.id": workflow_id, "workflow.type": definition.key},
        ):
            created = await self._workflows.create(workflow)
            await self._audit.record_for(
                created,
                AuditAction.WORKFLOW_CREATED,
                actor=principal.subject,
                actor_roles=tuple(sorted(str(r) for r in principal.roles)),
                summary=f"Created {definition.key}"
                + (" (simulation)" if execution_mode.is_simulation else ""),
                metadata={
                    "labels": created.labels,
                    "execution_mode": str(execution_mode),
                    "input_keys": sorted(input_data),
                },
            )
            await self._bus.publish(
                Queue.CONTROL,
                Message(
                    headers=MessageHeaders(
                        message_id=self._ids.new_id(MESSAGE),
                        tenant_id=created.tenant_id,
                        correlation_id=created.correlation_id,
                        dedup_key=f"advance:{workflow_id}:created",
                    ),
                    body=AdvanceWorkflow(workflow_id=workflow_id, reason="created"),
                ),
            )

        logger.info(
            "workflow created",
            extra={"workflow_id": workflow_id, "definition": definition.key},
        )
        return created

    async def get(self, principal: Principal, workflow_id: str) -> Workflow:
        require_permission(principal, Permission.WORKFLOW_READ)
        workflow = await self._workflows.get(principal.tenant_id, workflow_id)
        require_tenant(principal, workflow.tenant_id)
        return workflow

    async def explain(self, principal: Principal, workflow_id: str) -> RunExplanation:
        """Why this run came out the way it did.

        Assembled from the run's own durable state, so it is available for as
        long as the run is -- including long after the fact, which is when the
        question is usually asked.
        """
        workflow = await self.get(principal, workflow_id)
        definition = self._definitions.get(
            workflow.definition_name,
            workflow.definition_version,
            tenant_id=workflow.tenant_id,
        )
        return explain(workflow, definition, cases=await self._cases(workflow))

    async def list_cases(
        self,
        principal: Principal,
        *,
        batch_id: str | None = None,
        status: WorkflowStatus | None = None,
        limit: int = 100,
    ) -> list[CaseSummary]:
        """The cases this principal can see, newest first.

        ``batch_id`` narrows to the cases one fan-out run produced, which is
        the common read: somebody opened a batch and wants the ten claims in
        it rather than every claim in the tenant.
        """
        require_permission(principal, Permission.WORKFLOW_READ)

        if batch_id is not None:
            batch = await self.get(principal, batch_id)
            runs = await self._children_of(batch)
            # A run that did not fan out is itself the case, so opening it by
            # batch id should not come back empty.
            if not runs:
                runs = [batch]
        else:
            summaries, _ = await self._workflows.list(
                principal.tenant_id, status=status, limit=limit
            )
            runs = [
                await self._workflows.get(principal.tenant_id, s.workflow_id)
                for s in summaries
            ]

        seen: list[CaseSummary] = []
        for run in runs:
            if status is not None and run.status is not status:
                continue
            definition = self._definitions.get(
                run.definition_name, run.definition_version, tenant_id=run.tenant_id
            )
            seen.append(summarise(run, definition))
        return seen[:limit]

    async def ai_evaluation(
        self, principal: Principal, *, limit: int = 200, days: int | None = None
    ) -> AiEvaluation:
        """How the agents have been performing, read from finished runs.

        ``days`` narrows to a recent window. Filtering here rather than in the
        client means the headline figures describe the window a person chose,
        instead of describing everything and disagreeing with the table below
        them.
        """
        require_permission(principal, Permission.WORKFLOW_READ)

        pairs = await self._runs_in_window(principal, limit=limit, days=days)
        return evaluate(pairs, window_days=days)

    async def analytics(
        self, principal: Principal, *, limit: int = 500, days: int | None = None
    ) -> Analytics:
        """Throughput, outcomes and cycle time, read from finished runs.

        Same window handling as :meth:`ai_evaluation`, and a larger default
        limit: this one is counting cases rather than listing them, so the
        figures are worth more the further back they reach.
        """
        require_permission(principal, Permission.WORKFLOW_READ)

        pairs = await self._runs_in_window(principal, limit=limit, days=days)
        return analyse(pairs, window_days=days)

    async def _runs_in_window(
        self, principal: Principal, *, limit: int, days: int | None
    ) -> list[tuple[Workflow, WorkflowDefinition]]:
        """Full runs and their definitions, newest first, within the window.

        The list read gives summaries; both read models need the whole run,
        because what they count -- step output, approvals, timings -- lives in
        the steps and not in the summary.
        """
        cutoff = (
            self._clock.now() - timedelta(days=days) if days is not None else None
        )

        summaries, _ = await self._workflows.list(principal.tenant_id, limit=limit)
        pairs: list[tuple[Workflow, WorkflowDefinition]] = []
        for summary in summaries:
            if cutoff is not None and summary.created_at < cutoff:
                continue
            run = await self._workflows.get(principal.tenant_id, summary.workflow_id)
            definition = self._definitions.get(
                run.definition_name, run.definition_version, tenant_id=run.tenant_id
            )
            pairs.append((run, definition))
        return pairs

    async def case(self, principal: Principal, workflow_id: str) -> CaseDetail:
        """One case, with its timeline and any AI/rules disagreement."""
        workflow = await self.get(principal, workflow_id)
        definition = self._definitions.get(
            workflow.definition_name,
            workflow.definition_version,
            tenant_id=workflow.tenant_id,
        )
        return detail(workflow, definition)

    async def _children_of(self, workflow: Workflow) -> list[Workflow]:
        """Every child a fan-out run started, across all its fan-out steps."""
        children: list[Workflow] = []
        for state in workflow.steps.values():
            if state.type is not StepType.FAN_OUT:
                continue
            children.extend(
                await self._workflows.list_children(
                    workflow.tenant_id, workflow.workflow_id, state.step_id
                )
            )
        return children

    async def _cases(self, workflow: Workflow) -> dict[str, Any] | None:
        """The breakdown across a fan-out run's cases, read from the children.

        From the repository rather than the parent's recorded output, which is
        a snapshot: a case approved an hour after the batch settled has moved
        on, and the explanation should say where it got to, not where it was.
        """
        fan_out_steps = [
            state.step_id
            for state in workflow.steps.values()
            if state.type is StepType.FAN_OUT
        ]
        if not fan_out_steps:
            return None

        cases: list[dict[str, Any]] = []
        for step_id in fan_out_steps:
            children = await self._workflows.list_children(
                workflow.tenant_id, workflow.workflow_id, step_id
            )
            cases.extend(
                {
                    "workflow_id": child.workflow_id,
                    "key": child.fan_out_key or str(child.fan_out_index),
                    "index": child.fan_out_index,
                    "status": str(child.status),
                    "reason": child.cancellation_reason or "",
                }
                for child in children
            )

        by_status: dict[str, int] = {}
        for case in cases:
            by_status[case["status"]] = by_status.get(case["status"], 0) + 1
        return {"total": len(cases), "by_status": by_status, "cases": cases}

    async def list(
        self,
        principal: Principal,
        *,
        status: WorkflowStatus | None = None,
        definition_name: str | None = None,
        limit: int = 50,
        continuation: str | None = None,
    ) -> WorkflowPage:
        require_permission(principal, Permission.WORKFLOW_READ)
        return await self._workflows.list(
            principal.tenant_id,
            status=status,
            definition_name=definition_name,
            limit=limit,
            continuation=continuation,
        )

    def definitions(self, principal: Principal) -> DefinitionList:
        """The catalogue this principal may start.

        A tenant's own definitions shadow a built-in of the same name, so the
        catalogue shows each name once with the version that would actually run.
        """
        require_permission(principal, Permission.DEFINITION_READ)
        catalogue: dict[str, WorkflowDefinition] = {
            d.name: d for d in self._definitions.latest()
        }
        catalogue.update(
            {d.name: d for d in self._definitions.tenant_definitions(principal.tenant_id)}
        )
        return [
            definition
            for definition in sorted(catalogue.values(), key=lambda d: d.name)
            if principal.can_access_department(definition.department)
        ]


def _require_department(principal: Principal, definition: WorkflowDefinition) -> None:
    from cortexflow.shared.security.rbac import require_department

    require_department(principal, definition.department)


def _validate_input(definition: WorkflowDefinition, input_data: dict[str, Any]) -> None:
    """Check the workflow input against the definition's declared schema.

    A lightweight required/type check rather than full JSON Schema: it catches
    the mistakes that would otherwise surface as a confusing failure three
    steps into execution.
    """
    schema = definition.input_schema
    if not schema:
        return

    required: list[str] = schema.get("required", [])
    missing = [field for field in required if input_data.get(field) in (None, "")]
    if missing:
        raise ValidationError(
            "Workflow input is missing required fields",
            workflow=definition.key,
            missing=missing,
        )

    types: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for field, spec in (schema.get("properties") or {}).items():
        if field not in input_data or not isinstance(spec, dict):
            continue
        expected = types.get(spec.get("type", ""))
        if expected and not isinstance(input_data[field], expected):
            raise ValidationError(
                "Workflow input field has the wrong type",
                workflow=definition.key,
                field=field,
                expected=spec.get("type"),
            )

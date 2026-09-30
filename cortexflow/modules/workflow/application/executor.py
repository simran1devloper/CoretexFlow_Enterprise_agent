"""Step execution.

A worker is intentionally simple: take an ``EXECUTE_STEP`` pointer, load the
authoritative state, run one agent or one tool under a deadline, and report the
outcome back to the control queue.  It holds no workflow logic and makes no
scheduling decisions -- if a worker dies mid-step, the orchestrator's lease
recovery is what puts the work back, not anything the worker did on its way out.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from cortexflow.modules.agent.application.registry import AgentRegistry
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult, AgentStatus
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.domain.models import SideEffect, ToolCall
from cortexflow.modules.workflow.application.definitions import WorkflowDefinitionRegistry
from cortexflow.modules.workflow.domain.definition import Queue, StepDefinition, StepType
from cortexflow.modules.workflow.domain.envelope import (
    ExecuteStep,
    Message,
    MessageHeaders,
    StepCompleted,
    StepFailed,
)
from cortexflow.modules.workflow.domain.models import StepStatus, Workflow
from cortexflow.modules.workflow.ports.messaging import MessageBus
from cortexflow.modules.workflow.ports.repositories import WorkflowRepository
from cortexflow.shared.clock import Clock
from cortexflow.shared.errors import (
    CortexFlowError,
    DependencyTimeoutError,
    ErrorClass,
    NotFoundError,
    classify,
)
from cortexflow.shared.expressions import render
from cortexflow.shared.identity import Principal
from cortexflow.shared.ids import MESSAGE, IdGenerator
from cortexflow.shared.observability.logging import bind_context, get_logger
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)


class StepExecutor:
    """Executes one step and reports the result."""

    def __init__(
        self,
        *,
        workflows: WorkflowRepository,
        definitions: WorkflowDefinitionRegistry,
        agents: AgentRegistry,
        tools: ToolRegistry,
        bus: MessageBus,
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._workflows = workflows
        self._definitions = definitions
        self._agents = agents
        self._tools = tools
        self._bus = bus
        self._clock = clock
        self._ids = ids

    async def execute(self, command: ExecuteStep, *, tenant_id: str) -> Message:
        """Run the step and build the result message for the control queue."""
        started = time.perf_counter()

        with bind_context(
            tenant_id=tenant_id,
            workflow_id=command.workflow_id,
            step_id=command.step_id,
            run_id=command.run_id,
        ):
            try:
                workflow = await self._workflows.get(tenant_id, command.workflow_id)
            except NotFoundError as exc:
                return self._failure_message(command, tenant_id, exc, started)

            if not self._is_current(workflow, command):
                # A stale delivery: the attempt this message belongs to was
                # superseded. Report nothing; the orchestrator owns the state.
                logger.info("skipping a stale step command")
                return self._noop_message(command, tenant_id, workflow)

            definition = self._definitions.get(
                workflow.definition_name,
                workflow.definition_version,
                tenant_id=workflow.tenant_id,
            )
            spec = definition.step(command.step_id)

            with span(
                "step.execute",
                **{
                    "workflow.id": command.workflow_id,
                    "workflow.step": command.step_id,
                    "step.type": str(spec.type),
                    "step.attempt": command.attempt,
                },
            ):
                try:
                    output, confidence, cost = await self._run(
                        workflow, spec, command, tenant_id
                    )
                except CortexFlowError as exc:
                    return self._failure_message(command, tenant_id, exc, started)
                except TimeoutError:
                    return self._failure_message(
                        command,
                        tenant_id,
                        DependencyTimeoutError(
                            "Step exceeded its timeout",
                            timeout_seconds=spec.timeout_seconds,
                        ),
                        started,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("step raised an unhandled error")
                    return self._failure_message(command, tenant_id, exc, started)

        return self._success_message(
            command, tenant_id, workflow, output, confidence, cost, started
        )

    # -- execution paths ---------------------------------------------------
    async def _run(
        self,
        workflow: Workflow,
        spec: StepDefinition,
        command: ExecuteStep,
        tenant_id: str,
    ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        inputs = render(spec.inputs, workflow.context())
        principal = Principal.service(f"step::{spec.id}", tenant_id)

        # The deadline is enforced here, not inside the agent: a hung
        # dependency must not hold a worker slot indefinitely.
        coroutine = (
            self._run_agent(workflow, spec, command, inputs, principal)
            if spec.type is StepType.AGENT
            else self._run_tool(workflow, spec, command, inputs, principal)
        )
        return await asyncio.wait_for(coroutine, timeout=spec.timeout_seconds)

    async def _run_agent(
        self,
        workflow: Workflow,
        spec: StepDefinition,
        command: ExecuteStep,
        inputs: dict[str, Any],
        principal: Principal,
    ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        agent = self._agents.get(str(spec.agent))

        # Least privilege: a step grants tools explicitly or grants none.
        # Defaulting to "every tool this principal could use" would make the
        # allowlist decorative, which is the failure mode the tool registry
        # exists to prevent.
        declared = tuple(inputs.get("allowed_tools") or ())
        allowed = self._tools.tools_for(principal, names=declared) if declared else ()
        if workflow.execution_mode.is_simulation:
            # A rehearsal must not reach an enterprise system by the side
            # door. Holding back the step's own writes is not enough when the
            # step is an agent that has been handed tools of its own: a tool
            # the model cannot see is one it cannot call.
            allowed = tuple(
                name
                for name in allowed
                if self._tools.metadata(name).side_effect is SideEffect.READ
            )

        context = AgentContext(
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            workflow_type=workflow.definition_name,
            step_id=spec.id,
            run_id=command.run_id,
            attempt=command.attempt,
            principal=principal,
            inputs=inputs,
            workflow_input=workflow.input,
            step_outputs=workflow.outputs(),
            allowed_tools=allowed,
            deadline=command.deadline,
            correlation_id=workflow.correlation_id,
        )
        result = await agent.execute(context)
        return self._interpret_agent_result(result, spec)

    @staticmethod
    def _interpret_agent_result(
        result: AgentResult, spec: StepDefinition
    ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        """Translate an agent's structured result into step state."""
        cost = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "model": result.usage.model,
            "duration_ms": result.duration_ms,
        }

        if result.status is AgentStatus.FAILED:
            error = result.error
            raise _AgentFailure(
                code=error.code if error else "agent_failed",
                message=error.message if error else "Agent reported a failure",
                error_class=error.error_class if error else ErrorClass.UNKNOWN,
                details=error.details if error else {},
            )

        output = dict(result.output)
        if result.status is AgentStatus.NEEDS_HUMAN:
            # Not a failure: a fact the policy engine will act on downstream.
            output["needs_human"] = True
            output["needs_human_reason"] = result.reason
        output.setdefault("reason", result.reason)
        if result.tool_invocations:
            output["tool_invocations"] = [
                inv.model_dump() for inv in result.tool_invocations
            ]
        return output, result.confidence, cost

    async def _run_tool(
        self,
        workflow: Workflow,
        spec: StepDefinition,
        command: ExecuteStep,
        inputs: dict[str, Any],
        principal: Principal,
    ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        arguments = inputs.get("arguments", inputs)

        if self._would_change_the_world(workflow, spec):
            return self._simulated_call(spec, dict(arguments))

        call = ToolCall(
            tool=str(spec.tool),
            arguments=dict(arguments),
            # The key comes from the orchestrator, so every retry of this step
            # presents the same key and the side effect happens once.
            idempotency_key=command.idempotency_key,
            workflow_id=workflow.workflow_id,
            step_id=spec.id,
            tenant_id=workflow.tenant_id,
            correlation_id=workflow.correlation_id,
        )
        # The registry authorizes; a step reaching a tool behind an approval
        # gate is proof the approval already happened.
        result = await self._tools.execute(call, principal, approval_granted=True)
        if not result.succeeded:
            raise _AgentFailure(
                code=result.error_code or "tool_failed",
                message=result.error_message or "Tool reported a failure",
                error_class=ErrorClass.UNKNOWN,
            )
        return (
            {**result.data, "replayed": result.replayed},
            None,
            {"duration_ms": result.duration_ms},
        )

    def _would_change_the_world(
        self, workflow: Workflow, spec: StepDefinition
    ) -> bool:
        """Whether this call must be held back because the run is a rehearsal.

        Only writes are held back. A simulated run still reads -- it profiles
        the real file, looks up the real employee, and so reaches its
        decisions on the same data a live run would. Stubbing the reads too
        would make the simulation predict its own fixtures rather than the
        outcome, which is the failure mode that makes dry runs worthless.
        """
        if not workflow.execution_mode.is_simulation:
            return False
        try:
            return self._tools.metadata(str(spec.tool)).side_effect is not SideEffect.READ
        except NotFoundError:
            # An unregistered tool fails on its own in a moment. Until then,
            # assume the dangerous reading rather than calling it.
            return True

    @staticmethod
    def _simulated_call(
        spec: StepDefinition, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        """Record what would have been called, and let the run continue.

        The step succeeds, because the point of the rehearsal is to reach the
        steps *after* this one. What it returns is explicitly marked, so
        nothing downstream can mistake a rehearsal's output for a real
        system's response.
        """
        logger.info(
            "holding back a write in a simulated run",
            extra={"tool": spec.tool, "step_id": spec.id},
        )
        return (
            {
                "simulated": True,
                "would_call": str(spec.tool),
                "would_send": arguments,
                "note": (
                    f"Simulation: '{spec.tool}' was not called. In a live run "
                    "this would have changed an enterprise system."
                ),
            },
            None,
            {"duration_ms": 0},
        )

    # -- message construction ---------------------------------------------
    @staticmethod
    def _is_current(workflow: Workflow, command: ExecuteStep) -> bool:
        try:
            step = workflow.step(command.step_id)
        except NotFoundError:
            return False
        return step.run_id == command.run_id and step.status in {
            StepStatus.DISPATCHED,
            StepStatus.RUNNING,
        }

    def _headers(self, command: ExecuteStep, tenant_id: str, kind: str) -> MessageHeaders:
        return MessageHeaders(
            message_id=self._ids.new_id(MESSAGE),
            tenant_id=tenant_id,
            dedup_key=f"{kind}:{command.workflow_id}:{command.step_id}:{command.run_id}",
        )

    def _success_message(
        self,
        command: ExecuteStep,
        tenant_id: str,
        workflow: Workflow,
        output: dict[str, Any],
        confidence: float | None,
        cost: dict[str, Any],
        started: float,
    ) -> Message:
        headers = self._headers(command, tenant_id, "done")
        return Message(
            headers=headers.model_copy(update={"correlation_id": workflow.correlation_id}),
            body=StepCompleted(
                workflow_id=command.workflow_id,
                step_id=command.step_id,
                run_id=command.run_id,
                output=output,
                confidence=confidence,
                duration_ms=int((time.perf_counter() - started) * 1000),
                cost=cost,
            ),
        )

    def _failure_message(
        self, command: ExecuteStep, tenant_id: str, exc: Exception, started: float
    ) -> Message:
        code = getattr(exc, "code", type(exc).__name__)
        message = getattr(exc, "message", str(exc))
        details = getattr(exc, "details", {})
        logger.warning(
            "step failed",
            extra={
                "code": code,
                "error_class": str(classify(exc)),
                "step_id": command.step_id,
            },
        )
        return Message(
            headers=self._headers(command, tenant_id, "fail"),
            body=StepFailed(
                workflow_id=command.workflow_id,
                step_id=command.step_id,
                run_id=command.run_id,
                code=code,
                message=str(message)[:1000],
                error_class=classify(exc),
                details=details if isinstance(details, dict) else {},
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    def _noop_message(
        self, command: ExecuteStep, tenant_id: str, workflow: Workflow
    ) -> Message:
        """A stale command produces a harmless, already-settled report."""
        return Message(
            headers=self._headers(command, tenant_id, "stale"),
            body=StepCompleted(
                workflow_id=command.workflow_id,
                step_id=command.step_id,
                run_id=command.run_id,
                output={},
            ),
        )


class _AgentFailure(CortexFlowError):
    """Internal: carries a step failure with its classification intact."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        error_class: ErrorClass,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, **(details or {}))
        self.code = code
        self.error_class = error_class


def result_queue() -> Queue:
    """Step outcomes always return to the control plane."""
    return Queue.CONTROL

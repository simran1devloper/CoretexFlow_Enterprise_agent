"""The base agent.

Every agent shares one shape: take an :class:`AgentContext`, return an
:class:`AgentResult`.  The base class owns the cross-cutting concerns --
tracing, token accounting, tool mediation and error translation -- so a
concrete agent contains only its own reasoning.

Agents never touch repositories, the bus or enterprise APIs.  The only way out
of an agent is a tool call through the registry, restricted to the allowlist
in its context.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from cortexflow.modules.agent.adapters.structured import complete_structured
from cortexflow.modules.agent.domain.models import (
    AgentContext,
    AgentError,
    AgentResult,
    AgentStatus,
    TokenUsage,
    ToolInvocation,
)
from cortexflow.modules.agent.ports.llm import ChatMessage, LlmClient
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.domain.models import ToolCall
from cortexflow.shared.errors import (
    AuthorizationError,
    CortexFlowError,
    ErrorClass,
    classify,
)
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)

TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseAgent(ABC):
    """Common contract and machinery for all agents."""

    name: str = "agent"
    description: str = ""

    def __init__(self, llm: LlmClient, tools: ToolRegistry | None = None) -> None:
        self._llm = llm
        self._tools = tools

    # -- public entrypoint -------------------------------------------------
    async def execute(self, context: AgentContext) -> AgentResult:
        """Run the agent, translating any failure into a structured result.

        An agent never raises into the worker: a failure is data, carrying the
        error class the retry policy needs.
        """
        started = time.perf_counter()
        with span(
            "agent.execute",
            **{
                "agent.name": self.name,
                "workflow.id": context.workflow_id,
                "workflow.step": context.step_id,
                "agent.attempt": context.attempt,
            },
        ):
            try:
                result = await self.run(context)
            except CortexFlowError as exc:
                result = self._to_failure(exc, context)
            except Exception as exc:  # defensive: an agent bug is not a crash
                logger.exception(
                    "agent raised an unhandled error",
                    extra={"agent": self.name, "workflow_id": context.workflow_id},
                )
                result = AgentResult.failure(
                    AgentError(
                        code="agent_unhandled_error",
                        message=f"{type(exc).__name__}: {exc}"[:500],
                        error_class=classify(exc),
                    )
                )

        duration_ms = int((time.perf_counter() - started) * 1000)
        return result.model_copy(update={"duration_ms": duration_ms})

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """Agent-specific logic."""
        ...

    # -- helpers for subclasses -------------------------------------------
    async def think(
        self,
        context: AgentContext,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[TOutput],
        prompt_version: str = "",
        temperature: float = 0.0,
    ) -> tuple[TOutput, TokenUsage]:
        """One structured LLM turn, returning a validated object and its cost."""
        parsed, response = await complete_structured(
            self._llm,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ],
            model=output_model,
            prompt_version=prompt_version,
            temperature=temperature,
        )
        logger.info(
            "agent llm turn complete",
            extra={
                "agent": self.name,
                "schema": output_model.__name__,
                "prompt_version": prompt_version,
                "tokens": response.usage.total_tokens,
            },
        )
        return parsed, response.usage

    async def call_tool(
        self,
        context: AgentContext,
        tool: str,
        arguments: dict[str, Any],
    ) -> ToolInvocation:
        """Invoke a tool, enforcing this agent's allowlist first.

        The registry authorizes again independently -- this check exists so a
        model that hallucinates a tool name gets a clean, auditable refusal
        instead of reaching the registry at all.
        """
        if self._tools is None:
            raise AuthorizationError("This agent has no tool access", agent=self.name)
        if tool not in context.allowed_tools:
            raise AuthorizationError(
                "Tool is not in this step's allowlist",
                agent=self.name,
                tool=tool,
                allowed=list(context.allowed_tools),
            )

        started = time.perf_counter()
        call = ToolCall(
            tool=tool,
            arguments=arguments,
            workflow_id=context.workflow_id,
            step_id=context.step_id,
            tenant_id=context.tenant_id,
            correlation_id=context.correlation_id,
        )
        try:
            result = await self._tools.execute(call, context.principal)
        except CortexFlowError as exc:
            return ToolInvocation(
                tool=tool,
                arguments=arguments,
                duration_ms=int((time.perf_counter() - started) * 1000),
                succeeded=False,
                error=exc.message,
            )
        return ToolInvocation(
            tool=tool,
            arguments=arguments,
            result=result.data,
            idempotency_key=result.idempotency_key,
            duration_ms=result.duration_ms,
            succeeded=result.succeeded,
            error=result.error_message,
        )

    def _to_failure(self, exc: CortexFlowError, context: AgentContext) -> AgentResult:
        logger.warning(
            "agent failed",
            extra={
                "agent": self.name,
                "workflow_id": context.workflow_id,
                "step_id": context.step_id,
                "code": exc.code,
                "error_class": str(exc.error_class),
            },
        )
        return AgentResult.failure(
            AgentError(
                code=exc.code,
                message=exc.message,
                error_class=exc.error_class,
                details=exc.details,
            )
        )

    @staticmethod
    def _needs_human(reason: str, *, confidence: float | None = None) -> AgentResult:
        return AgentResult(
            status=AgentStatus.NEEDS_HUMAN, reason=reason, confidence=confidence
        )

    @staticmethod
    def _transient(message: str) -> AgentResult:
        return AgentResult.failure(
            AgentError(
                code="agent_transient_error",
                message=message,
                error_class=ErrorClass.TRANSIENT,
            )
        )

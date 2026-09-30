"""The Semantic Kernel agent runtime.

Where :class:`~cortexflow.modules.agent.application.base.BaseAgent` issues one structured
completion, this runtime hands control to Semantic Kernel's automatic function
calling: the model may call registry-backed tools across several turns before
producing its answer.

What does *not* change is the contract with the orchestrator. A ``KernelAgent``
still returns an :class:`AgentResult` -- structured output, confidence, token
usage, recorded tool invocations, a classified error. The orchestrator cannot
tell which runtime produced it, which is exactly the property that makes the
runtime a configuration choice rather than an architectural commitment.

The boundaries the native runtime enforces all still hold:

* tools come from the registry, so every call is authorized and audited;
* the step's allowlist decides which functions exist at all;
* the final answer is validated against a Pydantic schema before it can
  influence the workflow;
* automatic calling is capped, so a confused model cannot loop indefinitely.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from cortexflow.modules.agent.adapters.kernel.factory import KernelFactory, KernelSession
from cortexflow.modules.agent.adapters.sk_connectors import (
    translate_kernel_error,
    usage_from_response,
)
from cortexflow.modules.agent.adapters.sk_settings import build_execution_settings
from cortexflow.modules.agent.adapters.structured import extract_json, response_schema
from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult, TokenUsage
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.shared.errors import StructuredOutputError
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)

TOutput = TypeVar("TOutput", bound=BaseModel)


class KernelAgent(BaseAgent):
    """Base class for agents that run on the Semantic Kernel runtime."""

    def __init__(
        self,
        factory: KernelFactory,
        tools: ToolRegistry | None = None,
        *,
        max_function_calls: int = 5,
    ) -> None:
        # The LlmClient port is unused here: Semantic Kernel owns the model
        # interaction. It stays in the signature so both runtimes share one
        # agent contract.
        super().__init__(llm=None, tools=tools)  # type: ignore[arg-type]
        self._factory = factory
        self._max_function_calls = max_function_calls
        self._metrics = get_metrics()

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """Agent-specific logic, as in the native runtime."""
        ...

    async def reason(
        self,
        context: AgentContext,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[TOutput],
        prompt_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        use_tools: bool = True,
    ) -> tuple[TOutput, TokenUsage, KernelSession]:
        """Run a Kernel turn, letting the model call tools, then answer.

        This is deliberately **two phases**, because structured output and
        automatic function calling are mutually exclusive in a single turn.
        Constraining decoding to a response schema leaves the model no way to
        emit a tool call, so it stops calling tools and answers from memory --
        producing a confidently schema-valid hallucination, which is the worst
        possible failure for this platform.

        So:

        1. **Gather** -- tools offered, output unconstrained. The model calls
           what it needs and Semantic Kernel executes it through the registry.
           Results land in the chat history.
        2. **Answer** -- tools withdrawn, output constrained to the schema.
           The model now summarises real tool results rather than inventing
           them.

        A step with no tools skips straight to phase two.
        """
        session = self._factory.create(context)
        history = self._build_history(system_prompt, user_content)
        usage = TokenUsage(model=getattr(session.service, "ai_model_id", ""))

        with span(
            "kernel.reason",
            **{
                "agent.name": self.name,
                "agent.runtime": "semantic_kernel",
                "workflow.id": context.workflow_id,
                "workflow.step": context.step_id,
                "kernel.tools": len(session.function_names),
            },
        ):
            if use_tools and session.function_names:
                usage = usage.merged_with(
                    await self._gather(session, history, temperature, max_tokens)
                )

            response, answer_usage = await self._answer(
                session, history, output_model, temperature, max_tokens
            )
            usage = usage.merged_with(answer_usage)

        logger.info(
            "kernel turn complete",
            extra={
                "agent": self.name,
                "schema": output_model.__name__,
                "prompt_version": prompt_version,
                "tool_calls": len(session.plugin.invocations) if session.plugin else 0,
                "tokens": usage.total_tokens,
            },
        )
        return self._validate(response, output_model), usage, session

    async def _gather(
        self, session: KernelSession, history: object, temperature: float, max_tokens: int
    ) -> TokenUsage:
        """Phase one: let the model call tools, unconstrained by a schema."""
        settings = build_execution_settings(
            session.service,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=None,
            reasoning=session.dialect.reasoning,
            reasoning_effort=session.dialect.reasoning_effort,
            auto_invoke_tools=True,
            allowed_functions=session.function_names,
            max_auto_invoke_attempts=self._max_function_calls,
        )
        response, usage = await self._call(session, history, settings)

        # Keep the model's closing remark in context so phase two can use it.
        if response is not None and str(response).strip():
            history.add_assistant_message(str(response))  # type: ignore[attr-defined]
        return usage

    async def _answer(
        self,
        session: KernelSession,
        history: object,
        output_model: type[TOutput],
        temperature: float,
        max_tokens: int,
    ) -> tuple[object, TokenUsage]:
        """Phase two: constrain the answer to the schema, with tools withdrawn."""
        history.add_user_message(  # type: ignore[attr-defined]
            "Now report your result as a single JSON object matching the required "
            "schema. Use only information you actually obtained above. Do not "
            "invent values for anything you could not look up."
        )
        settings = build_execution_settings(
            session.service,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=response_schema(output_model),
            schema_name=output_model.__name__,
            reasoning=session.dialect.reasoning,
            reasoning_effort=session.dialect.reasoning_effort,
            auto_invoke_tools=False,
        )
        response, usage = await self._call(session, history, settings)
        if response is None or not str(response).strip():
            raise StructuredOutputError(
                "The model returned an empty response", agent=self.name
            )
        return response, usage

    async def _call(
        self, session: KernelSession, history: object, settings: object
    ) -> tuple[object, TokenUsage]:
        started = time.perf_counter()
        try:
            response = await session.service.get_chat_message_content(
                chat_history=history, settings=settings, kernel=session.kernel
            )
        except Exception as exc:
            raise translate_kernel_error(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = usage_from_response(
            response, model=getattr(session.service, "ai_model_id", "")
        )
        self._metrics.record_duration(
            self._metrics.llm_duration, elapsed_ms, model=usage.model, runtime="kernel"
        )
        if usage.total_tokens:
            self._metrics.count(
                self._metrics.llm_tokens, usage.total_tokens, model=usage.model
            )
        return response, usage

    # ------------------------------------------------------------------
    @staticmethod
    def _build_history(system_prompt: str, user_content: str) -> object:
        from semantic_kernel.contents import ChatHistory

        history = ChatHistory()
        history.add_system_message(system_prompt)
        history.add_user_message(user_content)
        return history

    def _validate(self, response: object, model: type[TOutput]) -> TOutput:
        """The structured-output boundary, unchanged from the native runtime.

        Semantic Kernel can constrain decoding to the schema, but constrained
        is not the same as valid -- so the answer is still parsed and
        validated before it can influence the workflow.
        """
        content = str(response)
        try:
            return model.model_validate(extract_json(content))
        except PydanticValidationError as exc:
            raise StructuredOutputError(
                "The model's response did not match the required schema",
                agent=self.name,
                schema=model.__name__,
                error=str(exc)[:500],
            ) from exc

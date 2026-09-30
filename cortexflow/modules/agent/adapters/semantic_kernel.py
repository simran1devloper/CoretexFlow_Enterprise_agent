"""The ``LlmClient`` port, implemented over a Semantic Kernel connector.

This is the smaller of the two Semantic Kernel integration points. It carries
no plugins and no function calling -- it is a single structured completion --
which is exactly what the five native agents need.

Its value is reach rather than capability: with this in place,
``CORTEXFLOW_LLM_BACKEND=ollama`` runs every existing agent against a locally
hosted model without touching a line of agent code. The richer integration,
where the model calls tools across several turns, lives in
``agents/kernel/``.
"""

from __future__ import annotations

import time
from typing import Any

from cortexflow.modules.agent.adapters.sk_connectors import (
    translate_kernel_error,
    usage_from_response,
)
from cortexflow.modules.agent.adapters.sk_settings import build_execution_settings
from cortexflow.modules.agent.domain.models import TokenUsage
from cortexflow.modules.agent.ports.llm import CompletionRequest, CompletionResponse
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)


class SemanticKernelLlmClient:
    """Adapts a Semantic Kernel chat service to the platform's LLM port."""

    def __init__(self, service: Any, *, model_name: str = "") -> None:
        self._service = service
        self._model_name = model_name or getattr(service, "ai_model_id", "unknown")
        self._metrics = get_metrics()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        from semantic_kernel.contents import ChatHistory

        history = ChatHistory()
        for message in request.messages:
            match message.role:
                case "system":
                    history.add_system_message(message.content)
                case "assistant":
                    history.add_assistant_message(message.content)
                case _:
                    history.add_user_message(message.content)

        settings = build_execution_settings(
            self._service,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_schema=request.json_schema or None,
            schema_name=request.schema_name,
            auto_invoke_tools=False,
        )

        started = time.perf_counter()
        with span(
            "llm.complete",
            **{
                "llm.model": self._model_name,
                "llm.schema": request.schema_name,
                "llm.prompt_version": request.prompt_version,
                "llm.runtime": "semantic_kernel",
            },
        ) as current:
            try:
                response = await self._service.get_chat_message_content(
                    chat_history=history, settings=settings
                )
            except Exception as exc:
                raise translate_kernel_error(exc) from exc

            elapsed_ms = (time.perf_counter() - started) * 1000
            content = str(response) if response is not None else ""

            usage: TokenUsage = usage_from_response(response, model=self._model_name)
            current.set_attribute("llm.tokens.total", usage.total_tokens)

        self._metrics.record_duration(
            self._metrics.llm_duration, elapsed_ms, model=self._model_name
        )
        if usage.total_tokens:
            self._metrics.count(
                self._metrics.llm_tokens, usage.total_tokens, model=self._model_name
            )

        return CompletionResponse(
            content=content,
            usage=usage,
            model=self._model_name,
            finish_reason=str(getattr(response, "finish_reason", "") or ""),
        )

"""LLM port.

The platform asks the model for *typed* answers.  Free-form completion is not
part of the contract, because free-form output cannot be a workflow contract.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cortexflow.modules.agent.domain.models import TokenUsage

TModel = TypeVar("TModel", bound=BaseModel)


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str  # system | user | assistant
    content: str


class CompletionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...]
    schema_name: str
    json_schema: dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 2048
    prompt_version: str = ""
    """Prompts are versioned so an output can be traced to the text that produced it."""


class CompletionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""


@runtime_checkable
class LlmClient(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Raises TransientError for throttling/timeouts, PermanentError otherwise."""
        ...

    @property
    def model_name(self) -> str: ...

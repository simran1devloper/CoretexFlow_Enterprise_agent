"""The tool contract.

A tool is metadata plus one async function.  Keeping the metadata declarative
is what lets the registry authorize, rate-limit, audit and de-duplicate every
call uniformly, without each tool re-implementing those concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from cortexflow.modules.tool_registry.domain.models import ToolCall, ToolMetadata, ToolResult
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.identity import Principal


class Tool(ABC):
    """Base class for every enterprise capability."""

    metadata: ToolMetadata
    input_model: type[BaseModel] | None = None

    @property
    def name(self) -> str:
        return self.metadata.name

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Reject bad input before it reaches an enterprise system.

        Agents produce these arguments, so this is a trust boundary: validate
        here rather than hoping the downstream API is strict.
        """
        if self.input_model is None:
            return arguments
        try:
            return self.input_model.model_validate(arguments).model_dump()
        except PydanticValidationError as exc:
            raise ValidationError(
                "Invalid tool arguments",
                tool=self.name,
                errors=exc.errors(include_url=False)[:5],
            ) from exc

    @abstractmethod
    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        """Perform the action. Implementations raise the platform's typed errors."""
        ...


def build_metadata(**kwargs: Any) -> ToolMetadata:
    return ToolMetadata(**kwargs)

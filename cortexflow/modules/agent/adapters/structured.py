"""Turning model output into typed objects.

An LLM's answer only becomes a workflow input after it survives this module:
JSON extraction, then Pydantic validation.  If it fails validation we retry
once with the validation error fed back, and then give up -- an unparseable
answer is a permanent failure, not something to hammer the model over.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from cortexflow.modules.agent.ports.llm import ChatMessage, CompletionRequest, LlmClient
from cortexflow.shared.errors import StructuredOutputError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)


def response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema to constrain a model's output with.

    Pydantic omits fields that have defaults from ``required``, which is
    correct for *validation* -- we are happy to accept a partial object and
    fill the rest in. It is wrong for *generation*: a schema that marks
    nothing required lets the model emit whichever properties it feels like,
    and a constrained decoder will happily oblige.

    Observed with llama3.1:8b on the extraction schema: it returned only
    ``missing_fields`` and ``reason``, omitting ``fields`` entirely. The
    result validated cleanly and was empty.

    So: strict in what we ask for, lenient in what we accept. Every property
    becomes required here, while the Pydantic model keeps its defaults for
    validation.
    """
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    if properties:
        schema["required"] = list(properties)
    return schema


def extract_json(content: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response, fenced or bare."""
    text = content.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group("body").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise StructuredOutputError(
                "Model response contained no JSON object", preview=text[:200]
            ) from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                "Model response was not valid JSON", preview=text[:200]
            ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError("Model response was not a JSON object")
    return parsed


async def complete_structured(
    client: LlmClient,
    *,
    messages: list[ChatMessage],
    model: type[TModel],
    prompt_version: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    repair_attempts: int = 1,
) -> tuple[TModel, Any]:
    """Request a completion and validate it against ``model``.

    Returns the parsed object and the raw response so callers can record token
    usage against the workflow's cost budget.
    """
    schema = response_schema(model)
    conversation = list(messages)
    last_error: Exception | None = None

    for attempt in range(repair_attempts + 1):
        response = await client.complete(
            CompletionRequest(
                messages=tuple(conversation),
                schema_name=model.__name__,
                json_schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_version=prompt_version,
            )
        )
        try:
            return model.model_validate(extract_json(response.content)), response
        except (StructuredOutputError, PydanticValidationError) as exc:
            last_error = exc
            logger.warning(
                "structured output rejected",
                extra={"attempt": attempt + 1, "schema": model.__name__, "error": str(exc)[:400]},
            )
            if attempt >= repair_attempts:
                break
            # One repair turn: show the model exactly what failed.
            conversation.extend(
                [
                    ChatMessage(role="assistant", content=response.content[:2000]),
                    ChatMessage(
                        role="user",
                        content=(
                            "That response did not match the required schema. "
                            f"Validation error:\n{str(exc)[:1000]}\n\n"
                            "Reply with a single JSON object matching the schema. "
                            "No prose, no code fences."
                        ),
                    ),
                ]
            )

    raise StructuredOutputError(
        "Model could not produce a valid structured response",
        schema=model.__name__,
        error=str(last_error)[:500],
    )

"""Prompt execution settings, built correctly per connector.

Three things differ between connectors and are easy to get wrong:

1. ``extension_data`` is only unpacked into real fields **at construction**.
   Assigning it afterwards leaves ``temperature`` as ``None``, silently
   discarding the setting.
2. Ollama has no ``temperature`` field at all -- it lives under ``options``.
   Azure OpenAI has it as a first-class field.
3. Structured output is ``format`` (a JSON schema) on Ollama and
   ``response_format`` on Azure OpenAI.
4. A reasoning deployment rejects ``max_tokens`` in favour of
   ``max_completion_tokens``, and rejects any ``temperature`` but its default.
   Unlike the three above this one fails loudly -- a 400 on the first call --
   which is at least honest, but it fails at run time on a workflow rather
   than at startup.

Getting these wrong fails quietly: the call succeeds, the setting is ignored.
Hence one place that knows the difference.
"""

from __future__ import annotations

from typing import Any

from cortexflow.modules.agent.adapters.sk_connectors import is_ollama
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


def build_execution_settings(
    service: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "Response",
    reasoning: bool = False,
    reasoning_effort: str = "",
    auto_invoke_tools: bool = False,
    allowed_functions: tuple[str, ...] = (),
    max_auto_invoke_attempts: int = 5,
) -> Any:
    """Build settings bound to ``service``.

    ``allowed_functions`` is a hard filter, not a hint: when tool calling is
    enabled, the model is offered exactly these functions and nothing else.
    """
    settings_class = service.get_prompt_execution_settings_class()
    service_id = getattr(service, "service_id", None)

    # Constructed, never assigned afterwards -- see the module docstring.
    settings = settings_class(service_id=service_id)

    if is_ollama(service):
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        settings.options = options
        if json_schema:
            # Ollama constrains decoding to a JSON schema through `format`.
            settings.format = json_schema
    elif reasoning:
        # A reasoning deployment thinks in tokens billed against this budget
        # before it emits a visible one, so the cap covers both. Temperature is
        # not lowered but omitted: these deployments accept only their default
        # and reject the request outright rather than clamping it.
        if max_tokens:
            settings.max_completion_tokens = max_tokens
        if reasoning_effort:
            settings.reasoning_effort = reasoning_effort
        if json_schema:
            settings.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": False,
                    "schema": json_schema,
                },
            }
    else:
        settings.temperature = temperature
        if max_tokens:
            settings.max_tokens = max_tokens
        if json_schema:
            settings.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": False,
                    "schema": json_schema,
                },
            }

    if auto_invoke_tools:
        settings.function_choice_behavior = _tool_behaviour(
            allowed_functions, max_auto_invoke_attempts
        )

    return settings


def _tool_behaviour(allowed_functions: tuple[str, ...], max_attempts: int) -> Any:
    """Automatic function calling, restricted to an explicit allowlist.

    ``max_attempts`` bounds the tool-calling *iterations*. Semantic Kernel
    then makes one final turn with function calling disabled, so the worst
    case is ``max_attempts + 1`` model calls -- budget latency and cost
    accordingly.

    With no allowlist we return a behaviour that offers *no* functions rather
    than every function in the kernel. A step grants tools explicitly or
    grants none -- the same least-privilege rule the native runtime follows.
    """
    from semantic_kernel.connectors.ai.function_choice_behavior import (
        FunctionChoiceBehavior,
    )

    if not allowed_functions:
        return FunctionChoiceBehavior.NoneInvoke()

    return FunctionChoiceBehavior.Auto(
        auto_invoke=True,
        filters={"included_functions": list(allowed_functions)},
        maximum_auto_invoke_attempts=max_attempts,
    )

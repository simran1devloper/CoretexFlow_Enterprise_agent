"""Bridging the tool registry into Semantic Kernel.

This is the security boundary for the Semantic Kernel runtime, and the reason
the runtime can be adopted without weakening anything.

Semantic Kernel's automatic function calling is genuinely useful: the model
decides which tool to call and the framework executes it. But a kernel plugin
that wrapped an enterprise API directly would hand the model an unmediated
path to that API, and the tool registry -- authorization, argument validation,
rate limiting, idempotency, audit -- would be bypassed entirely.

So plugins here are *generated from the registry*, and every generated
function is a thin shim that calls :meth:`ToolRegistry.execute`. The model
gains a convenient interface; it gains no new authority.

Two further restrictions:

* only the tools in the step's allowlist become functions at all -- a model
  cannot call what was never described to it;
* the generated parameter schema comes from each tool's Pydantic input model,
  so Semantic Kernel advertises the real contract rather than a free-form bag.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any

from pydantic import BaseModel

from cortexflow.modules.agent.domain.models import AgentContext, ToolInvocation
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.domain.models import ToolCall, ToolMetadata
from cortexflow.shared.errors import CortexFlowError
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.redaction import redact

logger = get_logger(__name__)

PLUGIN_NAME = "EnterpriseTools"
MAX_RESULT_CHARS = 4000
"""A tool result is fed back into the prompt, so it must stay bounded."""


def sk_function_name(tool_name: str) -> str:
    """``finance.get_budget`` -> ``finance_get_budget``.

    Semantic Kernel function names must be identifiers; dots are its own
    plugin/function separator.
    """
    return tool_name.replace(".", "_").replace("-", "_")


def qualified_name(tool_name: str) -> str:
    """The fully-qualified name used in function-choice filters."""
    return f"{PLUGIN_NAME}-{sk_function_name(tool_name)}"


class ToolRegistryPlugin:
    """A Semantic Kernel plugin backed by the tool registry.

    One instance per agent step, bound to that step's context and allowlist.
    It also records every invocation, so the structured ``AgentResult`` keeps
    the same audit fidelity as the native runtime.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        context: AgentContext,
        tools: tuple[str, ...],
    ) -> None:
        self._registry = registry
        self._context = context
        self._tools = tools
        self._invocations: list[ToolInvocation] = []

    @property
    def invocations(self) -> tuple[ToolInvocation, ...]:
        """Everything the model actually called, in order."""
        return tuple(self._invocations)

    @property
    def function_names(self) -> tuple[str, ...]:
        """Qualified names, for the function-choice filter."""
        return tuple(qualified_name(name) for name in self._tools)

    def build(self) -> object:
        """Create the object Semantic Kernel registers as a plugin.

        Functions are generated dynamically because the set of tools is a
        runtime decision -- it depends on the step's allowlist and on what the
        principal is authorized to use.
        """
        from semantic_kernel.functions import kernel_function

        namespace: dict[str, Any] = {}

        for tool_name in self._tools:
            metadata = self._registry.metadata(tool_name)
            method = self._build_method(metadata)
            namespace[sk_function_name(tool_name)] = kernel_function(
                name=sk_function_name(tool_name),
                description=_describe(metadata),
            )(method)

        plugin_type = type("_GeneratedToolPlugin", (), namespace)
        return plugin_type()

    # ------------------------------------------------------------------
    def _build_method(self, metadata: ToolMetadata) -> Any:
        """Generate one shim with the tool's real signature."""
        tool = self._registry.get(metadata.name)
        parameters = _signature_parameters(tool.input_model)

        # Semantic Kernel binds plugin functions as methods, so the generated
        # callable must accept the instance positionally. Semantic Kernel
        # strips `self` when it builds the parameter schema, so it never
        # reaches the model.
        async def invoke(_plugin: Any, **kwargs: Any) -> str:
            return await self._execute(metadata.name, kwargs)

        invoke.__name__ = sk_function_name(metadata.name)
        invoke.__doc__ = _describe(metadata)
        invoke.__annotations__ = {
            **{p.name: p.annotation for p in parameters},
            "return": str,
        }
        # Semantic Kernel reads the signature to build its parameter schema,
        # so the generated function must advertise the real one.
        invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD), *parameters],
            return_annotation=str,
        )
        return invoke

    async def _execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run one tool call through the registry and report it as text.

        Failures are returned to the model rather than raised. A model that is
        told "employee EMP-9 was not found" can correct itself or ask for
        review; an exception would only abort the step. The failure is still
        recorded in ``invocations`` and audited by the registry.
        """
        started = time.perf_counter()
        cleaned = {k: v for k, v in arguments.items() if v is not None}

        call = ToolCall(
            tool=tool_name,
            arguments=cleaned,
            workflow_id=self._context.workflow_id,
            step_id=self._context.step_id,
            tenant_id=self._context.tenant_id,
            correlation_id=self._context.correlation_id,
        )

        logger.info(
            "semantic kernel invoked a registry tool",
            extra={
                "tool": tool_name,
                "workflow_id": self._context.workflow_id,
                "step_id": self._context.step_id,
                "arguments": redact(cleaned),
            },
        )

        try:
            result = await self._registry.execute(call, self._context.principal)
        except CortexFlowError as exc:
            self._invocations.append(
                ToolInvocation(
                    tool=tool_name,
                    arguments=cleaned,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    succeeded=False,
                    error=exc.message,
                )
            )
            return json.dumps({"error": exc.code, "message": exc.message})

        self._invocations.append(
            ToolInvocation(
                tool=tool_name,
                arguments=cleaned,
                result=result.data,
                idempotency_key=result.idempotency_key,
                duration_ms=result.duration_ms,
                succeeded=result.succeeded,
                error=result.error_message,
            )
        )

        if not result.succeeded:
            return json.dumps(
                {"error": result.error_code, "message": result.error_message}
            )

        payload = json.dumps(result.data, default=str)
        if len(payload) > MAX_RESULT_CHARS:
            payload = payload[:MAX_RESULT_CHARS] + '..."}'
        return payload


def _describe(metadata: ToolMetadata) -> str:
    """Describe a tool to the model, including its risk posture.

    Stating the side effect is not decoration: it discourages a model from
    treating a write as a free lookup, and it makes the transcript legible to
    whoever reads the audit trail afterwards.
    """
    return (
        f"{metadata.description} "
        f"[domain={metadata.domain}, side_effect={metadata.side_effect}]"
    )


def _signature_parameters(model: type[BaseModel] | None) -> list[inspect.Parameter]:
    """Derive Semantic Kernel parameters from a tool's Pydantic input model."""
    if model is None:
        return []

    parameters: list[inspect.Parameter] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation if field.annotation is not None else str
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )

    # Required parameters first: Signature forbids a required one after a
    # defaulted one, even for keyword-only parameters.
    parameters.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return parameters

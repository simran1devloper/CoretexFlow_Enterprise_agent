"""Building a Kernel for one agent step.

A Kernel is constructed **per step**, not shared. That is the whole point:
the plugins it carries are exactly the tools that step is allowed to use, so
the allowlist is enforced by what exists rather than by a check someone has to
remember to write.

A long-lived Kernel shared across steps would accumulate every plugin the
platform has, and the least-privilege boundary would quietly disappear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortexflow.modules.agent.adapters.kernel.tool_plugin import PLUGIN_NAME, ToolRegistryPlugin
from cortexflow.modules.agent.adapters.sk_connectors import SERVICE_ID, is_ollama
from cortexflow.modules.agent.domain.models import AgentContext
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelDialect:
    """What this deployment accepts, as opposed to what it is called.

    A reasoning deployment rejects ``max_tokens`` and any ``temperature`` but
    the default, so the parameters a turn is built with depend on which model
    is behind the connector. Carried here rather than sniffed from the service
    at the point of use, so an explicitly configured value survives -- name
    inference is a fallback, not the rule.
    """

    reasoning: bool = False
    reasoning_effort: str = ""


@dataclass(frozen=True)
class KernelSession:
    """A Kernel plus the plugin bound to this step, for reading invocations back."""

    kernel: Any
    service: Any
    plugin: ToolRegistryPlugin | None
    dialect: ModelDialect = ModelDialect()

    @property
    def function_names(self) -> tuple[str, ...]:
        return self.plugin.function_names if self.plugin else ()


class KernelFactory:
    """Creates per-step Kernels wired to the tool registry."""

    def __init__(
        self,
        *,
        service: Any,
        registry: ToolRegistry,
        dialect: ModelDialect | None = None,
    ) -> None:
        self._service = service
        self._registry = registry
        self._dialect = dialect or ModelDialect()

    @property
    def service(self) -> Any:
        return self._service

    def create(self, context: AgentContext) -> KernelSession:
        """Build a Kernel carrying only this step's authorized tools."""
        from semantic_kernel import Kernel

        kernel = Kernel()
        kernel.add_service(self._service)

        # Least privilege, identical to the native runtime: a step grants
        # tools explicitly or grants none. Passing an empty allowlist to
        # `tools_for` would mean "everything this principal may use", which
        # would make the allowlist decorative.
        granted = (
            self._registry.tools_for(context.principal, names=context.allowed_tools)
            if context.allowed_tools
            else ()
        )

        plugin: ToolRegistryPlugin | None = None
        if granted:
            plugin = ToolRegistryPlugin(
                registry=self._registry, context=context, tools=granted
            )
            kernel.add_plugin(plugin.build(), plugin_name=PLUGIN_NAME)

        if granted and is_ollama(self._service):
            # Semantic Kernel names functions "<Plugin>-<function>", and that
            # hyphen stops several Ollama-hosted models emitting structured
            # tool calls -- they return the call as plain text instead, which
            # Semantic Kernel cannot execute. Measured on llama3.1:8b:
            # "get_weather" and "Weather_get_weather" produce structured calls,
            # "Weather-get_weather" does not.
            #
            # The degradation is safe rather than silent: no tool calls means
            # no findings, which means low confidence, which means the policy
            # engine escalates to a human. But it is worth saying out loud.
            logger.warning(
                "tool calling on Ollama is unreliable: Semantic Kernel's "
                "hyphenated function names stop some local models emitting "
                "structured tool calls. Prefer Azure OpenAI for the kernel "
                "runtime, and expect escalation to a human if lookups return "
                "nothing.",
                extra={"step_id": context.step_id, "tools": len(granted)},
            )

        logger.debug(
            "kernel built for step",
            extra={
                "workflow_id": context.workflow_id,
                "step_id": context.step_id,
                "service_id": SERVICE_ID,
                "tools": list(granted),
            },
        )
        return KernelSession(
            kernel=kernel,
            service=self._service,
            plugin=plugin,
            dialect=self._dialect,
        )

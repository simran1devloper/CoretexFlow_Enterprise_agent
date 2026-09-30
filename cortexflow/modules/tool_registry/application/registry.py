"""The tool registry: the single door between agents and enterprise systems.

Everything an agent can cause to happen passes through ``execute`` here, which
in order: resolves the tool, authorizes the principal, validates arguments,
applies rate limits, enforces idempotency, times the call, and writes an audit
event.  An agent never holds a client credential and never sees an endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from cortexflow.modules.audit.domain.models import AuditAction, AuditEvent, AuditOutcome
from cortexflow.modules.audit.ports.repositories import AuditRepository
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.application.idempotency import IdempotentExecutor, build_key
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolAuthorization,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.errors import (
    AuthorizationError,
    CortexFlowError,
    NotFoundError,
    RateLimitedError,
    ToolExecutionError,
    ValidationError,
    classify,
)
from cortexflow.shared.identity import Principal, Role
from cortexflow.shared.ids import AUDIT, IdGenerator, UuidIdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.redaction import redact
from cortexflow.shared.observability.telemetry import span
from cortexflow.shared.ports.cache import Cache

logger = get_logger(__name__)


class ToolRegistry:
    """Authorizing, auditing façade over the registered tools."""

    def __init__(
        self,
        *,
        audit: AuditRepository,
        idempotency: IdempotentExecutor,
        cache: Cache | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._audit = audit
        self._idempotency = idempotency
        self._cache = cache
        self._ids = ids or UuidIdGenerator()
        self._metrics = get_metrics()

    # -- registration ------------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info(
            "tool registered",
            extra={
                "tool": tool.name,
                "domain": str(tool.metadata.domain),
                "risk": str(tool.metadata.risk),
                "side_effect": str(tool.metadata.side_effect),
            },
        )

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise NotFoundError("Unknown tool", tool=name) from exc

    def metadata(self, name: str) -> ToolMetadata:
        return self.get(name).metadata

    def list_metadata(self) -> list[ToolMetadata]:
        return [tool.metadata for tool in self._tools.values()]

    def tools_for(self, principal: Principal, *, names: tuple[str, ...] = ()) -> tuple[str, ...]:
        """The tools an agent may see.

        This is the allowlist handed to the agent runtime.  A tool the
        principal cannot use is not merely rejected on call -- it is never
        described to the model in the first place.
        """
        candidates = names or tuple(self._tools)
        allowed: list[str] = []
        for name in candidates:
            tool = self._tools.get(name)
            if tool is None or not tool.metadata.exposed_to_agents:
                continue
            if self.authorize(name, principal).allowed:
                allowed.append(name)
        return tuple(allowed)

    # -- authorization -----------------------------------------------------
    def authorize(self, name: str, principal: Principal) -> ToolAuthorization:
        """Decide whether ``principal`` may invoke ``name``."""
        try:
            metadata = self.metadata(name)
        except NotFoundError:
            return ToolAuthorization(allowed=False, tool=name, reason="Tool is not registered")

        if not principal.can_access_department(metadata.domain):
            return ToolAuthorization(
                allowed=False,
                tool=name,
                reason=f"Principal has no access to the {metadata.domain} domain",
                risk=metadata.risk,
            )

        if metadata.allowed_roles:
            permitted = set(metadata.allowed_roles)
            # A service agent inherits the workflow's authority for reads, but
            # a role-restricted write still requires that explicit role.
            if metadata.side_effect is SideEffect.READ:
                permitted.add(Role.SERVICE_AGENT)
            if not principal.roles.intersection(permitted) and not principal.is_admin():
                return ToolAuthorization(
                    allowed=False,
                    tool=name,
                    reason="Principal lacks a role permitted for this tool",
                    risk=metadata.risk,
                )

        return ToolAuthorization(
            allowed=True,
            tool=name,
            requires_approval=metadata.requires_human_approval,
            risk=metadata.risk,
        )

    # -- execution ---------------------------------------------------------
    async def execute(
        self,
        call: ToolCall,
        principal: Principal,
        *,
        approval_granted: bool = False,
    ) -> ToolResult:
        """Authorize, validate and run a tool call exactly once."""
        tool = self.get(call.tool)
        metadata = tool.metadata

        authorization = self.authorize(call.tool, principal)
        if not authorization.allowed:
            await self._audit_event(
                call, principal, AuditAction.TOOL_DENIED, AuditOutcome.DENIED,
                summary=authorization.reason,
            )
            raise AuthorizationError(
                authorization.reason, tool=call.tool, subject=principal.subject
            )

        if metadata.requires_human_approval and not approval_granted:
            # Defence in depth: the workflow definition should already have
            # routed this through an approval step. If it did not, refuse.
            await self._audit_event(
                call, principal, AuditAction.TOOL_DENIED, AuditOutcome.DENIED,
                summary="Tool requires human approval that was not granted",
            )
            raise AuthorizationError(
                "Tool requires human approval", tool=call.tool
            )

        arguments = tool.validate_arguments(call.arguments)

        if metadata.rate_limit_per_minute and self._cache is not None:
            await self._enforce_rate_limit(call, metadata)

        key = call.idempotency_key or build_key(
            workflow_id=call.workflow_id,
            step_id=call.step_id,
            tool=call.tool,
            arguments=arguments,
        )
        if metadata.requires_idempotency_key and not key.strip():
            raise ValidationError("Side-effecting tool requires an idempotency key",
                                 tool=call.tool)

        authorized_call = call.model_copy(update={"arguments": arguments, "idempotency_key": key})
        await self._audit_event(
            authorized_call, principal, AuditAction.TOOL_AUTHORIZED, AuditOutcome.SUCCESS
        )

        started = time.perf_counter()
        with span(
            "tool.execute",
            **{
                "tool.name": call.tool,
                "tool.domain": str(metadata.domain),
                "tool.risk": str(metadata.risk),
                "tool.side_effect": str(metadata.side_effect),
                "workflow.id": call.workflow_id,
            },
        ):
            try:
                if metadata.side_effect is SideEffect.READ:
                    # Reads need no idempotency bookkeeping -- they are free to repeat.
                    result = await tool.execute(authorized_call, principal)
                else:
                    result = await self._idempotency.run(
                        key,
                        lambda: tool.execute(authorized_call, principal),
                        tool=call.tool,
                    )
            except CortexFlowError as exc:
                await self._record_failure(authorized_call, principal, exc, started)
                raise
            except Exception as exc:
                wrapped = ToolExecutionError(
                    "Tool raised an unexpected error",
                    tool=call.tool,
                    error=type(exc).__name__,
                )
                await self._record_failure(authorized_call, principal, wrapped, started)
                raise wrapped from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = result.model_copy(update={"duration_ms": elapsed_ms, "idempotency_key": key})
        self._metrics.record_duration(
            self._metrics.tool_duration, elapsed_ms, tool=call.tool,
            domain=str(metadata.domain),
        )
        await self._audit_event(
            authorized_call,
            principal,
            AuditAction.TOOL_REPLAYED if result.replayed else AuditAction.TOOL_EXECUTED,
            AuditOutcome.SUCCESS if result.succeeded else AuditOutcome.FAILURE,
            summary=result.error_message or "",
            metadata={"duration_ms": elapsed_ms, "replayed": result.replayed},
        )
        return result

    async def _enforce_rate_limit(self, call: ToolCall, metadata: ToolMetadata) -> None:
        limit = metadata.rate_limit_per_minute
        if self._cache is None or limit is None:
            return

        from cortexflow.shared.clock import utcnow

        # A fixed one-minute window keyed per tenant and tool: a noisy tenant
        # cannot consume another tenant's allowance.
        window = utcnow().strftime("%Y%m%d%H%M")
        key = f"ratelimit:{call.tenant_id}:{metadata.name}:{window}"
        count = await self._cache.incr(key, ttl_seconds=120)
        if count > limit:
            raise RateLimitedError(
                "Tool rate limit exceeded",
                tool=metadata.name,
                limit=limit,
            )

    async def _record_failure(
        self, call: ToolCall, principal: Principal, exc: CortexFlowError, started: float
    ) -> None:
        await self._audit_event(
            call,
            principal,
            AuditAction.TOOL_EXECUTED,
            AuditOutcome.FAILURE,
            summary=exc.message,
            metadata={
                "error_code": exc.code,
                "error_class": str(classify(exc)),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    async def _audit_event(
        self,
        call: ToolCall,
        principal: Principal,
        action: AuditAction,
        outcome: AuditOutcome,
        *,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._audit.append(
            AuditEvent(
                event_id=self._ids.new_id(AUDIT),
                tenant_id=call.tenant_id or principal.tenant_id,
                action=action,
                outcome=outcome,
                actor=principal.subject,
                actor_roles=tuple(sorted(str(r) for r in principal.roles)),
                workflow_id=call.workflow_id or None,
                step_id=call.step_id or None,
                tool=call.tool,
                correlation_id=call.correlation_id,
                summary=summary,
                metadata={
                    "arguments": redact(call.arguments),
                    **(metadata or {}),
                },
            )
        )

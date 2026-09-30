"""Platform metrics.

Backed by OpenTelemetry when the SDK is installed, and by a no-op recorder
otherwise, so importing this module never forces an observability dependency
on a unit test.
"""

from __future__ import annotations

from typing import Any, Protocol

try:  # pragma: no cover - exercised by deployment, not unit tests
    from opentelemetry import metrics as otel_metrics

    _OTEL = True
except ImportError:  # pragma: no cover
    _OTEL = False


class _Instrument(Protocol):
    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None: ...


class _NoopInstrument:
    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        return None

    def add(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        return None


class Metrics:
    """The metric surface described in the observability plan.

    Names follow OTel conventions (``cortexflow.<subject>.<unit-or-event>``) so
    dashboards and alerts can be built without reading this file.
    """

    def __init__(self, meter_name: str = "cortexflow") -> None:
        if _OTEL:
            meter = otel_metrics.get_meter(meter_name)
            self.workflow_duration = meter.create_histogram(
                "cortexflow.workflow.duration", unit="ms",
                description="End-to-end workflow latency",
            )
            self.step_duration = meter.create_histogram(
                "cortexflow.step.duration", unit="ms", description="Per-step latency",
            )
            self.llm_duration = meter.create_histogram(
                "cortexflow.llm.duration", unit="ms", description="LLM call latency",
            )
            self.llm_tokens = meter.create_counter(
                "cortexflow.llm.tokens", description="Tokens consumed",
            )
            self.tool_duration = meter.create_histogram(
                "cortexflow.tool.duration", unit="ms", description="Enterprise tool latency",
            )
            self.step_failures = meter.create_counter(
                "cortexflow.step.failures", description="Step failures by error class",
            )
            self.step_retries = meter.create_counter(
                "cortexflow.step.retries", description="Retries scheduled",
            )
            self.dead_letters = meter.create_counter(
                "cortexflow.messages.dead_lettered", description="Messages dead-lettered",
            )
            self.duplicates = meter.create_counter(
                "cortexflow.messages.duplicates", description="Duplicate deliveries dropped",
            )
            self.concurrency_conflicts = meter.create_counter(
                "cortexflow.state.conflicts", description="Optimistic concurrency conflicts",
            )
            self.leases_reclaimed = meter.create_counter(
                "cortexflow.leases.reclaimed", description="Leases reclaimed from crashed workers",
            )
            self.approval_wait = meter.create_histogram(
                "cortexflow.approval.wait", unit="ms", description="Human approval latency",
            )
            self.policy_decisions = meter.create_counter(
                "cortexflow.modules.policy.decisions", description="Policy decisions by effect",
            )
        else:  # pragma: no cover
            noop = _NoopInstrument()
            for name in (
                "workflow_duration", "step_duration", "llm_duration", "llm_tokens",
                "tool_duration", "step_failures", "step_retries", "dead_letters",
                "duplicates", "concurrency_conflicts", "leases_reclaimed",
                "approval_wait", "policy_decisions",
            ):
                setattr(self, name, noop)

    def record_duration(
        self, instrument: Any, milliseconds: float, **attributes: Any
    ) -> None:
        instrument.record(milliseconds, attributes)

    def count(self, instrument: Any, value: int = 1, **attributes: Any) -> None:
        instrument.add(value, attributes)


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics

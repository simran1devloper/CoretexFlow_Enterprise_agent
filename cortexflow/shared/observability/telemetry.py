"""OpenTelemetry wiring and tracing helpers.

One trace spans a whole workflow: the API request that created it, each
orchestration pass, every agent run, every LLM call and every enterprise API
call hang off the same correlation id.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cortexflow.config.settings import ObservabilitySettings
from cortexflow.shared.observability.logging import configure_logging
from cortexflow.shared.observability.redaction import redact

try:  # pragma: no cover
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL = True
except ImportError:  # pragma: no cover
    _OTEL = False

_configured = False


def configure_telemetry(settings: ObservabilitySettings, *, service: str) -> None:
    """Configure logging and tracing for a service entrypoint. Idempotent."""
    global _configured
    configure_logging(
        service=service,
        level=settings.log_level,
        json_output=settings.log_json,
    )
    if _configured or not _OTEL:
        return

    resource = Resource.create(
        {
            "service.name": f"{settings.service_name}-{service}",
            "service.namespace": settings.service_name,
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.exporter_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.exporter_endpoint))
            )
        except ImportError:  # pragma: no cover
            pass
    if settings.console_traces:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _configured = True


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        return None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, redacting attributes before they reach the exporter."""
    if not _OTEL:
        yield _NoopSpan()
        return

    tracer = trace.get_tracer("cortexflow")
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, _attr(value))
        try:
            yield current
        except BaseException as exc:
            current.record_exception(exc)
            current.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise


def _attr(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(redact(value))


def current_trace_id() -> str:
    """Hex trace id for correlating a log line or an API response to a trace."""
    if not _OTEL:
        return ""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""

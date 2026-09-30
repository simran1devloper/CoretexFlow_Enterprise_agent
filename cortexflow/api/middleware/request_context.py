"""Request middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from cortexflow.shared.ids import UuidIdGenerator
from cortexflow.shared.observability.logging import bind_context
from cortexflow.shared.observability.telemetry import current_trace_id

CORRELATION_HEADER = "X-Correlation-Id"
_ids = UuidIdGenerator()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Propagates a correlation id across the whole request.

    The id a caller supplies flows into the workflow it creates, into every
    message, agent run and tool call that follows, and back out on the
    response -- so a user-reported problem can be traced end to end.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER) or _ids.new_id("REQ")
        request.state.correlation_id = correlation_id

        with bind_context(correlation_id=correlation_id):
            response = await call_next(request)

        response.headers[CORRELATION_HEADER] = correlation_id
        trace_id = current_trace_id()
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response

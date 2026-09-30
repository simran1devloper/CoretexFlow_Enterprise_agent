"""Translating domain errors into HTTP responses.

Handlers never construct ``HTTPException``. They raise domain errors, and this
module maps them -- so the same error means the same thing whether it surfaces
through the API, a worker log or the audit trail.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from cortexflow.api.schemas import ErrorResponse
from cortexflow.shared.errors import CortexFlowError
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.redaction import redact
from cortexflow.shared.observability.telemetry import current_trace_id

logger = get_logger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(CortexFlowError)
    async def _domain_error(request: Request, exc: CortexFlowError) -> JSONResponse:
        log = logger.warning if exc.http_status < 500 else logger.error
        log(
            "request failed",
            extra={
                "code": exc.code,
                "path": request.url.path,
                "status": exc.http_status,
                "details": exc.details,
            },
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                error_class=str(exc.error_class),
                details=redact(exc.details),
                trace_id=current_trace_id(),
            ).model_dump(),
        )

    @app.exception_handler(ValidationError)
    async def _model_error(request: Request, exc: ValidationError) -> JSONResponse:
        """A body a model refused is the caller's problem, not a server fault.

        Endpoints that parse their own payload (the builder does, so an
        unfinished draft is not rejected at the parser) raise pydantic's error
        rather than one of ours. Without this it reached the catch-all and came
        back as 500 "An unexpected error occurred" -- which told a person
        choosing a department that does not exist nothing at all.
        """
        problems = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "problem": error["msg"],
            }
            for error in exc.errors()
        ]
        logger.warning(
            "request body rejected",
            extra={"path": request.url.path, "problems": problems},
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="validation_error",
                message="; ".join(f"{p['field']}: {p['problem']}" for p in problems),
                error_class="PERMANENT",
                details={"problems": problems},
                trace_id=current_trace_id(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak an internal message to a client; the trace id is the
        # bridge between what the caller sees and what we logged.
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="internal_error",
                message="An unexpected error occurred",
                trace_id=current_trace_id(),
            ).model_dump(),
        )

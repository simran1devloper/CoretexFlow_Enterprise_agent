"""The API service.

Thin by design.  It authenticates, validates, writes durable state and
publishes one message.  It never executes a step, so a slow agent cannot make
the dashboard slow, and the two scale independently.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cortexflow import __version__
from cortexflow.api.middleware.errors import install_error_handlers
from cortexflow.api.middleware.request_context import CorrelationIdMiddleware
from cortexflow.api.routes import ALL_ROUTERS
from cortexflow.composition import build_container
from cortexflow.config.settings import Settings, get_settings
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import configure_telemetry

logger = get_logger(__name__)

DESCRIPTION = """
Control plane for CortexFlow.

Agents interpret; deterministic code decides. Workflow execution is
server-side and durable: creating a workflow returns immediately, and closing
the browser has no effect on it.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_telemetry(settings.observability, service="api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = await build_container(settings)
        logger.info("api ready", extra={"profile": str(settings.profile)})
        try:
            yield
        finally:
            await app.state.container.close()

    app = FastAPI(
        title="CortexFlow",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)

    install_error_handlers(app)
    for router in ALL_ROUTERS:
        app.include_router(router)

    _instrument(app)
    return app


def _instrument(app: FastAPI) -> None:
    try:  # pragma: no cover - optional dependency
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready")
    except ImportError:
        logger.debug("FastAPI OpenTelemetry instrumentation not installed")


app = create_app


def run() -> None:  # pragma: no cover - console entrypoint
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cortexflow.api.main:build",
        host=settings.api_host,
        port=settings.api_port,
        factory=True,
        log_config=None,
    )


def build() -> FastAPI:  # pragma: no cover - uvicorn factory
    return create_app()


if __name__ == "__main__":  # pragma: no cover
    run()

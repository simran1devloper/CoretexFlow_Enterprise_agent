"""The modular monolith: every module in one process.

Runs the API, the orchestrator, the agent workers and the recovery sweeper
together, against whatever adapters the profile selects -- in-memory locally,
Cosmos DB and Service Bus on the azure profile. It is both how the platform is
understood by running it and how it is deployed by default.

That is a deliberate position rather than a simplification. The modules keep
their boundaries whichever way this is run: they talk through ports, the bus
carries the same messages, and a step dispatched here takes the same path it
would take between two containers. What changes is the number of deployable
units, and one is the right number until a workload has a reason to scale
apart from the rest.

The split runtimes -- ``cortexflow-api``, ``cortexflow-agent-worker`` and the
others -- run the same image in one role at a time, for when that reason
arrives. Agent work is slow and expensive while API traffic is fast and cheap,
so those two are usually the first to separate. Extracting one is then a
deployment change, not a rewrite, which is the payoff for keeping the
boundaries honest while everything shared a process.
"""

from __future__ import annotations

import asyncio
import contextlib

import uvicorn

from cortexflow.api.middleware.errors import install_error_handlers
from cortexflow.api.middleware.request_context import CorrelationIdMiddleware
from cortexflow.api.routes import ALL_ROUTERS
from cortexflow.composition import Container, build_container
from cortexflow.config.settings import Settings, get_settings
from cortexflow.modules.workflow.domain.topics import WORKER_QUEUES
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import configure_telemetry

logger = get_logger(__name__)


def create_app(container: Container):  # type: ignore[no-untyped-def]
    """Build a FastAPI app around an already-constructed container."""
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from cortexflow import __version__

    background: list[asyncio.Task[None]] = []
    dispatcher = container.build_dispatcher()
    sweeper = container.build_sweeper(interval_seconds=2.0)
    worker = container.build_worker(WORKER_QUEUES)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container
        background.extend(
            [
                asyncio.create_task(dispatcher.run()),
                asyncio.create_task(worker.run()),
                asyncio.create_task(sweeper.run()),
            ]
        )
        logger.info("all-in-one runtime started")
        try:
            yield
        finally:
            dispatcher.stop()
            worker.stop()
            sweeper.stop()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await container.close()

    app = FastAPI(
        title="CortexFlow (all-in-one)",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=container.settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)
    install_error_handlers(app)
    for router in ALL_ROUTERS:
        app.include_router(router)
    return app


async def _build(settings: Settings):  # type: ignore[no-untyped-def]
    configure_telemetry(settings.observability, service="allinone")
    container = await build_container(settings)
    return create_app(container)


def run() -> None:  # pragma: no cover - interactive entrypoint
    settings = get_settings()
    app = asyncio.run(_build(settings))
    logger.info(
        "serving on http://%s:%s/docs", settings.api_host, settings.api_port
    )
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_config=None)


if __name__ == "__main__":  # pragma: no cover
    run()

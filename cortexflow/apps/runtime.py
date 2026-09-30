"""Shared service-process plumbing.

Every entrypoint needs the same four things: configured telemetry, a built
container, supervised background tasks, and a shutdown that drains rather than
kills.  Keeping that here means each service's ``main`` is a few lines that
say only what that service does.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable

from cortexflow.composition import Container, build_container
from cortexflow.config.settings import Settings, get_settings
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.telemetry import configure_telemetry

logger = get_logger(__name__)


class ServiceHost:
    """Runs a set of long-lived tasks until a termination signal arrives."""

    def __init__(self, name: str, container: Container) -> None:
        self.name = name
        self.container = container
        self._tasks: list[asyncio.Task[None]] = []
        self._stoppers: list[Callable[[], None]] = []
        self._shutdown = asyncio.Event()

    def supervise(
        self, coro: Awaitable[None], *, stop: Callable[[], None] | None = None
    ) -> None:
        """Run ``coro`` for the life of the service."""
        task = asyncio.create_task(self._guard(coro))
        self._tasks.append(task)
        if stop is not None:
            self._stoppers.append(stop)

    async def _guard(self, coro: Awaitable[None]) -> None:
        """A component that dies unexpectedly takes the process down.

        Restarting a half-dead service is the orchestrator's job (Container
        Apps, Kubernetes); silently continuing without a consumer is worse
        than exiting.
        """
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("a supervised component failed; shutting down")
            self._shutdown.set()

    async def serve(self) -> None:
        self._install_signal_handlers()
        logger.info("service started", extra={"service": self.name})
        await self._shutdown.wait()
        await self.shutdown()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._shutdown.set)

    async def shutdown(self) -> None:
        """Ask components to stop, give them time to drain, then close."""
        logger.info("service stopping", extra={"service": self.name})
        for stop in self._stoppers:
            with contextlib.suppress(Exception):
                stop()

        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=15)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        await self.container.close()
        logger.info("service stopped", extra={"service": self.name})


async def bootstrap(service: str, settings: Settings | None = None) -> Container:
    """Configure telemetry and build the container for a service process."""
    settings = settings or get_settings()
    configure_telemetry(settings.observability, service=service)
    return await build_container(settings)


def run_service(service: str, main: Callable[[Container], Awaitable[None]]) -> None:
    """Synchronous entrypoint wrapper used by the console scripts."""

    async def _run() -> None:
        container = await bootstrap(service)
        try:
            await main(container)
        finally:
            await container.close()

    with contextlib.suppress(KeyboardInterrupt):  # pragma: no cover
        asyncio.run(_run())

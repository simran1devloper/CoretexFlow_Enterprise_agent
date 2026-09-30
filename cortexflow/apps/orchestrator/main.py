"""The orchestrator service.

Runs the control-plane dispatcher and the recovery sweeper.  It coordinates;
it does not execute.  Several replicas can run safely: workflow leases keep
them from colliding, and a global sweep lease keeps them from duplicating
recovery work.
"""

from __future__ import annotations

from cortexflow.apps.runtime import ServiceHost, run_service
from cortexflow.composition import Container
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

SWEEP_INTERVAL_SECONDS = 10.0


async def serve(container: Container) -> None:
    host = ServiceHost("orchestrator", container)

    dispatcher = container.build_dispatcher()
    sweeper = container.build_sweeper(interval_seconds=SWEEP_INTERVAL_SECONDS)

    host.supervise(dispatcher.run(), stop=dispatcher.stop)
    host.supervise(sweeper.run(), stop=sweeper.stop)

    logger.info(
        "orchestrator running",
        extra={"workflows": list(container.definitions.names())},
    )
    await host.serve()


def run() -> None:  # pragma: no cover - console entrypoint
    run_service("orchestrator", serve)


if __name__ == "__main__":  # pragma: no cover
    run()

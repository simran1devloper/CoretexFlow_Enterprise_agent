"""The integration service.

A dedicated worker for the integration queue -- the steps that actually touch
HR, ERP and CRM systems.  It is its own deployable for two reasons:

* enterprise APIs impose their own rate limits and connection pools, which are
  easier to respect from a bounded pool of processes;
* it is the only service that needs network paths into corporate systems, so
  it can sit behind stricter network policy than the agent workers.
"""

from __future__ import annotations

from cortexflow.apps.runtime import ServiceHost, run_service
from cortexflow.composition import Container
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


async def serve(container: Container) -> None:
    host = ServiceHost("integration-service", container)
    worker = container.build_worker((Queue.INTEGRATION,))

    host.supervise(worker.run(), stop=worker.stop)
    logger.info(
        "integration service running",
        extra={"tools": len(container.tools.list_metadata())},
    )
    await host.serve()


def run() -> None:  # pragma: no cover - console entrypoint
    run_service("integration-service", serve)


if __name__ == "__main__":  # pragma: no cover
    run()

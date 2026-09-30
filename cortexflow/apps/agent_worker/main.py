"""The agent-worker service.

Consumes step commands and runs agents and tools.  Scaled separately from the
API because its workload is the opposite shape: slow, expensive and bursty.

``CORTEXFLOW_WORKER_QUEUES`` narrows which queues an instance serves, so an
expensive extraction pool can scale independently of cheap validation work.
"""

from __future__ import annotations

from cortexflow.apps.runtime import ServiceHost, run_service
from cortexflow.composition import Container
from cortexflow.modules.workflow.domain.definition import Queue
from cortexflow.modules.workflow.domain.topics import WORKER_QUEUES
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


def resolve_queues(container: Container) -> tuple[Queue, ...]:
    """Pick the queues this instance consumes, defaulting to all of them."""
    configured = container.settings.worker_queues
    if not configured:
        return WORKER_QUEUES
    return tuple(Queue(name.strip().lower()) for name in configured if name.strip())


async def serve(container: Container) -> None:
    host = ServiceHost("agent-worker", container)
    queues = resolve_queues(container)
    worker = container.build_worker(queues)

    host.supervise(worker.run(), stop=worker.stop)
    logger.info(
        "agent worker running",
        extra={
            "queues": [q.value for q in queues],
            "concurrency": container.settings.worker_concurrency,
        },
    )
    await host.serve()


def run() -> None:  # pragma: no cover - console entrypoint
    run_service("agent-worker", serve)


if __name__ == "__main__":  # pragma: no cover
    run()

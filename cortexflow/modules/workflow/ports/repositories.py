"""Persistence ports owned by the workflow module.

Every method is tenant-scoped. Isolation is not a convention the callers must
remember -- it is in the signature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from cortexflow.modules.workflow.domain.definition import WorkflowDefinition
from cortexflow.modules.workflow.domain.models import (
    Workflow,
    WorkflowStatus,
    WorkflowSummary,
)

# Declared at module scope deliberately: the repository method named
# ``list`` shadows the builtin inside its own class body, so return
# annotations written there would not resolve.
WorkflowPage = tuple[list[WorkflowSummary], str | None]
"""A page of workflow summaries plus the continuation token, if any."""

WorkflowRef = tuple[str, str]
"""``(tenant_id, workflow_id)`` -- the minimum needed to reload a workflow."""

WorkflowRefs = list[WorkflowRef]

WorkflowList = list[Workflow]

DefinitionList = list[WorkflowDefinition]

DeadLetterEntries = list[dict[str, Any]]


@runtime_checkable
class WorkflowRepository(Protocol):
    """Durable workflow state with optimistic concurrency."""

    async def create(self, workflow: Workflow) -> Workflow:
        """Insert a new workflow. Raises ConflictError if the id already exists."""
        ...

    async def get(self, tenant_id: str, workflow_id: str) -> Workflow:
        """Load a workflow. Raises NotFoundError when absent."""
        ...

    async def try_get(self, tenant_id: str, workflow_id: str) -> Workflow | None: ...

    async def save(self, workflow: Workflow, *, expected_version: int) -> Workflow:
        """Persist a mutation.

        Raises :class:`ConcurrencyConflictError` when the stored version no
        longer matches ``expected_version``, so a losing writer reloads rather
        than clobbering the winner.
        """
        ...

    async def list(
        self,
        tenant_id: str,
        *,
        status: WorkflowStatus | None = None,
        definition_name: str | None = None,
        limit: int = 50,
        continuation: str | None = None,
    ) -> WorkflowPage: ...

    async def list_children(
        self, tenant_id: str, parent_workflow_id: str, parent_step_id: str
    ) -> WorkflowList:
        """Every child a fan-out step started, in the order it started them.

        The authoritative answer to "what did this step spawn": the parent's
        own record of it can be short if a crash landed between creating a
        child and saving the parent.
        """
        ...

    async def find_due(self, *, now: datetime, limit: int = 100) -> WorkflowRefs:
        """Workflows whose timer has elapsed, as ``(tenant_id, workflow_id)``.

        Drives retry backoff, approval timeouts and lease recovery: the sweeper
        asks for due work rather than any process holding a timer in memory.
        """
        ...

    async def find_stalled(
        self, *, older_than: datetime, limit: int = 100
    ) -> WorkflowRefs:
        """Running workflows whose step leases expired -- crashed workers."""
        ...


@runtime_checkable
class WorkflowDefinitionRepository(Protocol):
    """Workflow definitions authored at runtime, through the builder.

    Kept apart from the definitions shipped on disk: those are reviewed,
    versioned with the code and global, while these belong to one tenant and
    can change between two runs of the same workflow. Both are validated
    identically before they can execute.
    """

    async def save(self, tenant_id: str, definition: WorkflowDefinition) -> WorkflowDefinition:
        """Insert or replace a definition version for this tenant."""
        ...

    async def get(
        self, tenant_id: str, name: str, version: int | None = None
    ) -> WorkflowDefinition | None:
        """The named definition, defaulting to this tenant's latest version."""
        ...

    async def list(self, tenant_id: str) -> DefinitionList:
        """The latest version of each definition this tenant has authored."""
        ...

    async def delete(self, tenant_id: str, name: str) -> None: ...


@runtime_checkable
class WorkflowDraftRepository(Protocol):
    """The builder draft each definition was compiled from.

    Stored because compiling is lossy. A ``DraftStep`` carries things the
    definition has no field for -- the fields an extraction step expects, the
    checks a validation step runs, which column supplies each fact -- and by
    the time it is a ``StepDefinition`` they have been folded into ``inputs``
    and argument expressions. Reading a definition back well enough to edit it
    would take a decompiler that mirrors the compiler, and the two would drift
    the first time either changed.

    So the draft is kept as what it is: the source the definition was built
    from. The definition remains the only thing the engine reads, and nothing
    here is consulted while a workflow runs.

    A workflow shipped in the repository has no draft, and that is the honest
    answer rather than a gap -- those are reviewed and versioned with the code,
    and the builder says so instead of pretending it can edit one.
    """

    async def save(self, tenant_id: str, name: str, draft: dict[str, Any]) -> None:
        """Insert or replace the draft behind ``name`` for this tenant."""
        ...

    async def get(self, tenant_id: str, name: str) -> dict[str, Any] | None: ...

    async def delete(self, tenant_id: str, name: str) -> None: ...


@runtime_checkable
class DeadLetterRepository(Protocol):
    """Poison messages parked for human investigation."""

    async def record(
        self,
        *,
        tenant_id: str,
        queue: str,
        message: dict[str, Any],
        reason: str,
        attempts: int,
    ) -> str: ...

    async def list(self, tenant_id: str, *, limit: int = 50) -> DeadLetterEntries: ...

    async def get(self, tenant_id: str, entry_id: str) -> dict[str, Any] | None: ...

    async def discard(self, tenant_id: str, entry_id: str) -> None: ...

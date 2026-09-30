"""Cosmos DB repositories.

Design notes that matter in production:

* ``tenant_id`` is the partition key everywhere, so a tenant's throughput and
  blast radius are its own.
* Writes use ``etag`` + ``if_match``, which is how the lost-update protection
  described by the workflow ``version`` field is actually enforced.
* Audit events are append-only and carry a TTL, keeping retention a policy
  setting rather than a cleanup script.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cortexflow.config.settings import CosmosSettings
from cortexflow.infrastructure.cosmos.client import translate_error
from cortexflow.modules.approval.domain.models import Approval, ApprovalStatus
from cortexflow.modules.audit.domain.models import AuditEvent
from cortexflow.modules.workflow.domain.models import Workflow, WorkflowStatus
from cortexflow.modules.workflow.ports.repositories import (
    DeadLetterEntries,
    WorkflowList,
    WorkflowPage,
    WorkflowRefs,
)
from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import ConcurrencyConflictError, NotFoundError
from cortexflow.shared.ids import IdGenerator, UuidIdGenerator
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


class _CosmosRepositoryBase:
    def __init__(self, client: Any, settings: CosmosSettings, container_name: str) -> None:
        self._client = client
        self._settings = settings
        self._container_name = container_name

    @property
    def _container(self) -> Any:
        return self._client.get_database_client(self._settings.database).get_container_client(
            self._container_name
        )

    async def _query(
        self, query: str, parameters: list[dict[str, Any]], *, partition_key: str | None = None
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query": query, "parameters": parameters}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        else:
            kwargs["enable_cross_partition_query"] = True
        return [item async for item in self._container.query_items(**kwargs)]


class CosmosWorkflowRepository(_CosmosRepositoryBase):
    def __init__(self, client: Any, settings: CosmosSettings) -> None:
        super().__init__(client, settings, settings.workflow_container)

    @staticmethod
    def _to_document(workflow: Workflow) -> dict[str, Any]:
        doc = workflow.model_dump(mode="json")
        doc["id"] = workflow.workflow_id
        doc.pop("etag", None)
        # Denormalised fields so the sweeper queries stay single-index scans.
        doc["_due_at"] = (
            workflow.scheduled_wake_at.isoformat() if workflow.scheduled_wake_at else None
        )
        doc["_lease_expiry"] = max(
            (
                s.lease_expires_at.isoformat()
                for s in workflow.steps.values()
                if s.status.is_in_flight and s.lease_expires_at
            ),
            default=None,
        )
        return doc

    @staticmethod
    def _from_document(doc: dict[str, Any]) -> Workflow:
        etag = doc.get("_etag")
        payload = {k: v for k, v in doc.items() if not k.startswith("_")}
        workflow = Workflow.model_validate(payload)
        workflow.etag = etag
        return workflow

    async def create(self, workflow: Workflow) -> Workflow:
        workflow.version = 1
        try:
            doc = await self._container.create_item(self._to_document(workflow))
        except Exception as exc:
            raise translate_error(exc, workflow_id=workflow.workflow_id) from exc
        return self._from_document(doc)

    async def get(self, tenant_id: str, workflow_id: str) -> Workflow:
        workflow = await self.try_get(tenant_id, workflow_id)
        if workflow is None:
            raise NotFoundError("Workflow not found", workflow_id=workflow_id)
        return workflow

    async def try_get(self, tenant_id: str, workflow_id: str) -> Workflow | None:
        try:
            doc = await self._container.read_item(workflow_id, partition_key=tenant_id)
        except Exception as exc:
            translated = translate_error(exc, workflow_id=workflow_id)
            if isinstance(translated, NotFoundError):
                return None
            raise translated from exc
        return self._from_document(doc)

    async def save(self, workflow: Workflow, *, expected_version: int) -> Workflow:
        if workflow.version != expected_version:
            raise ConcurrencyConflictError(
                "Stale workflow instance",
                workflow_id=workflow.workflow_id,
                expected_version=expected_version,
                actual_version=workflow.version,
            )
        workflow.version = expected_version + 1
        workflow.updated_at = utcnow()
        options = {"if_match": workflow.etag} if workflow.etag else {}
        try:
            doc = await self._container.replace_item(
                workflow.workflow_id, self._to_document(workflow), **options
            )
        except Exception as exc:
            workflow.version = expected_version  # leave the caller's copy consistent
            raise translate_error(exc, workflow_id=workflow.workflow_id) from exc
        return self._from_document(doc)

    async def list(
        self,
        tenant_id: str,
        *,
        status: WorkflowStatus | None = None,
        definition_name: str | None = None,
        limit: int = 50,
        continuation: str | None = None,
    ) -> WorkflowPage:
        clauses = ["c.tenant_id = @tenant"]
        params: list[dict[str, Any]] = [{"name": "@tenant", "value": tenant_id}]
        if status is not None:
            clauses.append("c.status = @status")
            params.append({"name": "@status", "value": str(status)})
        if definition_name:
            clauses.append("c.definition_name = @name")
            params.append({"name": "@name", "value": definition_name})

        query = (
            f"SELECT * FROM c WHERE {' AND '.join(clauses)} "
            "ORDER BY c.created_at DESC"
        )
        iterator = self._container.query_items(
            query=query, parameters=params, partition_key=tenant_id
        ).by_page(continuation)

        try:
            page = await iterator.__anext__()
        except StopAsyncIteration:
            # An empty container yields no pages at all, not one empty page.
            # Uncaught, this escaped as a 500 from every endpoint that lists
            # workflows -- including /ready, which made a fresh deployment
            # unable to report itself healthy until somebody had created a
            # workflow, which required it to be healthy first.
            #
            # Every other query here goes through `_query`, which iterates
            # with `async for` and is therefore already safe. This is the one
            # that paginates.
            return [], None

        rows = [self._from_document(doc).to_summary() async for doc in page]
        return rows[:limit], iterator.continuation_token

    async def list_children(
        self, tenant_id: str, parent_workflow_id: str, parent_step_id: str
    ) -> WorkflowList:
        # Partitioned by tenant, so this stays a single-partition read however
        # many tenants share the container.
        docs = await self._query(
            "SELECT * FROM c WHERE c.tenant_id = @tenant "
            "AND c.parent_workflow_id = @parent AND c.parent_step_id = @step "
            "ORDER BY c.fan_out_index",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@parent", "value": parent_workflow_id},
                {"name": "@step", "value": parent_step_id},
            ],
            partition_key=tenant_id,
        )
        return [Workflow.model_validate(doc) for doc in docs]

    async def find_due(self, *, now: datetime, limit: int = 100) -> WorkflowRefs:
        docs = await self._query(
            "SELECT TOP @limit c.tenant_id, c.workflow_id FROM c "
            "WHERE c._due_at != null AND c._due_at <= @now "
            "AND c.status NOT IN ('COMPLETED','FAILED','CANCELLED','REJECTED')",
            [
                {"name": "@limit", "value": limit},
                {"name": "@now", "value": now.isoformat()},
            ],
        )
        return [(d["tenant_id"], d["workflow_id"]) for d in docs]

    async def find_stalled(
        self, *, older_than: datetime, limit: int = 100
    ) -> WorkflowRefs:
        docs = await self._query(
            "SELECT TOP @limit c.tenant_id, c.workflow_id FROM c "
            "WHERE c._lease_expiry != null AND c._lease_expiry <= @cutoff "
            "AND c.status = 'RUNNING'",
            [
                {"name": "@limit", "value": limit},
                {"name": "@cutoff", "value": older_than.isoformat()},
            ],
        )
        return [(d["tenant_id"], d["workflow_id"]) for d in docs]


class CosmosApprovalRepository(_CosmosRepositoryBase):
    def __init__(self, client: Any, settings: CosmosSettings) -> None:
        super().__init__(client, settings, settings.approval_container)

    @staticmethod
    def _to_document(approval: Approval) -> dict[str, Any]:
        doc = approval.model_dump(mode="json")
        doc["id"] = approval.approval_id
        doc.pop("etag", None)
        return doc

    @staticmethod
    def _from_document(doc: dict[str, Any]) -> Approval:
        approval = Approval.model_validate({k: v for k, v in doc.items() if not k.startswith("_")})
        approval.etag = doc.get("_etag")
        return approval

    async def create(self, approval: Approval) -> Approval:
        approval.version = 1
        try:
            doc = await self._container.create_item(self._to_document(approval))
        except Exception as exc:
            raise translate_error(exc, approval_id=approval.approval_id) from exc
        return self._from_document(doc)

    async def get(self, tenant_id: str, approval_id: str) -> Approval:
        try:
            doc = await self._container.read_item(approval_id, partition_key=tenant_id)
        except Exception as exc:
            raise translate_error(exc, approval_id=approval_id) from exc
        return self._from_document(doc)

    async def save(self, approval: Approval, *, expected_version: int) -> Approval:
        if approval.version != expected_version:
            raise ConcurrencyConflictError("Stale approval", approval_id=approval.approval_id)
        approval.version = expected_version + 1
        approval.updated_at = utcnow()
        options = {"if_match": approval.etag} if approval.etag else {}
        try:
            doc = await self._container.replace_item(
                approval.approval_id, self._to_document(approval), **options
            )
        except Exception as exc:
            approval.version = expected_version
            raise translate_error(exc, approval_id=approval.approval_id) from exc
        return self._from_document(doc)

    async def list_pending(
        self, tenant_id: str, *, workflow_id: str | None = None, limit: int = 50
    ) -> list[Approval]:
        clauses = ["c.tenant_id = @tenant", "c.status = 'PENDING'"]
        params: list[dict[str, Any]] = [{"name": "@tenant", "value": tenant_id}]
        if workflow_id:
            clauses.append("c.workflow_id = @wf")
            params.append({"name": "@wf", "value": workflow_id})
        params.append({"name": "@limit", "value": limit})
        docs = await self._query(
            f"SELECT TOP @limit * FROM c WHERE {' AND '.join(clauses)} "
            "ORDER BY c.created_at ASC",
            params,
            partition_key=tenant_id,
        )
        return [self._from_document(d) for d in docs]

    async def find_expired(self, *, now: datetime, limit: int = 100) -> list[Approval]:
        docs = await self._query(
            "SELECT TOP @limit * FROM c WHERE c.status = 'PENDING' "
            "AND c.expires_at != null AND c.expires_at <= @now",
            [
                {"name": "@limit", "value": limit},
                {"name": "@now", "value": now.isoformat()},
            ],
        )
        return [self._from_document(d) for d in docs]

    async def cancel_for_workflow(
        self, tenant_id: str, workflow_id: str, *, reason: str
    ) -> int:
        cancelled = 0
        for approval in await self.list_pending(tenant_id, workflow_id=workflow_id, limit=1000):
            approval.status = ApprovalStatus.CANCELLED
            approval.context = {**approval.context, "cancellation_reason": reason}
            await self.save(approval, expected_version=approval.version)
            cancelled += 1
        return cancelled


class CosmosAuditRepository(_CosmosRepositoryBase):
    """Append-only audit store with TTL-driven retention."""

    def __init__(
        self, client: Any, settings: CosmosSettings, *, retention_days: int = 2555
    ) -> None:
        super().__init__(client, settings, settings.audit_container)
        self._ttl_seconds = retention_days * 86_400

    async def append(self, event: AuditEvent) -> None:
        doc = event.model_dump(mode="json")
        doc["id"] = event.event_id
        doc["ttl"] = self._ttl_seconds
        try:
            await self._container.create_item(doc)
        except Exception as exc:
            raise translate_error(exc, event_id=event.event_id) from exc

    async def append_many(self, events: list[AuditEvent]) -> None:
        for event in events:
            await self.append(event)

    async def query(
        self,
        tenant_id: str,
        *,
        workflow_id: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[AuditEvent]:
        clauses = ["c.tenant_id = @tenant"]
        params: list[dict[str, Any]] = [{"name": "@tenant", "value": tenant_id}]
        if workflow_id:
            clauses.append("c.workflow_id = @wf")
            params.append({"name": "@wf", "value": workflow_id})
        if actor:
            clauses.append("c.actor = @actor")
            params.append({"name": "@actor", "value": actor})
        if since:
            clauses.append("c.occurred_at >= @since")
            params.append({"name": "@since", "value": since.isoformat()})
        params.append({"name": "@limit", "value": limit})
        docs = await self._query(
            f"SELECT TOP @limit * FROM c WHERE {' AND '.join(clauses)} "
            "ORDER BY c.occurred_at ASC",
            params,
            partition_key=tenant_id,
        )
        return [
            AuditEvent.model_validate({k: v for k, v in d.items() if not k.startswith("_")})
            for d in docs
        ]


class CosmosIdempotencyStore(_CosmosRepositoryBase):
    """Idempotency via conditional create -- the write itself is the lock."""

    def __init__(self, client: Any, settings: CosmosSettings) -> None:
        super().__init__(client, settings, settings.idempotency_container)

    @staticmethod
    def _doc_id(key: str) -> str:
        import hashlib

        return hashlib.sha256(key.encode()).hexdigest()

    async def reserve(self, key: str, *, ttl_seconds: int) -> bool:
        from cortexflow.shared.errors import ConflictError

        try:
            await self._container.create_item(
                {
                    "id": self._doc_id(key),
                    "key": key,
                    "state": "IN_FLIGHT",
                    "reserved_at": utcnow().isoformat(),
                    "ttl": ttl_seconds,
                }
            )
            return True
        except Exception as exc:
            if isinstance(translate_error(exc, key=key), ConflictError):
                return False
            raise translate_error(exc, key=key) from exc

    async def complete(self, key: str, result: dict[str, Any], *, ttl_seconds: int) -> None:
        await self._container.upsert_item(
            {
                "id": self._doc_id(key),
                "key": key,
                "state": "COMPLETED",
                "result": result,
                "completed_at": utcnow().isoformat(),
                "ttl": ttl_seconds,
            }
        )

    async def get(self, key: str) -> dict[str, Any] | None:
        doc_id = self._doc_id(key)
        try:
            doc = await self._container.read_item(doc_id, partition_key=doc_id)
        except Exception as exc:
            if isinstance(translate_error(exc, key=key), NotFoundError):
                return None
            raise translate_error(exc, key=key) from exc
        return doc.get("result") if doc.get("state") == "COMPLETED" else None

    async def release(self, key: str) -> None:
        doc_id = self._doc_id(key)
        try:
            await self._container.delete_item(doc_id, partition_key=doc_id)
        except Exception as exc:
            if not isinstance(translate_error(exc, key=key), NotFoundError):
                raise


class CosmosDeadLetterRepository(_CosmosRepositoryBase):
    def __init__(
        self, client: Any, settings: CosmosSettings, ids: IdGenerator | None = None
    ) -> None:
        super().__init__(client, settings, settings.audit_container)
        self._ids = ids or UuidIdGenerator()

    async def record(
        self,
        *,
        tenant_id: str,
        queue: str,
        message: dict[str, Any],
        reason: str,
        attempts: int,
    ) -> str:
        entry_id = self._ids.new_id("DLQ")
        await self._container.create_item(
            {
                "id": entry_id,
                "entry_id": entry_id,
                "doc_type": "dead_letter",
                "tenant_id": tenant_id,
                "queue": queue,
                "message": message,
                "reason": reason,
                "attempts": attempts,
                "recorded_at": utcnow().isoformat(),
            }
        )
        return entry_id

    async def list(self, tenant_id: str, *, limit: int = 50) -> DeadLetterEntries:
        return await self._query(
            "SELECT TOP @limit * FROM c WHERE c.tenant_id = @tenant "
            "AND c.doc_type = 'dead_letter' ORDER BY c.recorded_at DESC",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@limit", "value": limit},
            ],
            partition_key=tenant_id,
        )

    async def get(self, tenant_id: str, entry_id: str) -> dict[str, Any] | None:
        try:
            return await self._container.read_item(entry_id, partition_key=tenant_id)
        except Exception as exc:
            if isinstance(translate_error(exc), NotFoundError):
                return None
            raise

    async def discard(self, tenant_id: str, entry_id: str) -> None:
        try:
            await self._container.delete_item(entry_id, partition_key=tenant_id)
        except Exception as exc:
            if not isinstance(translate_error(exc), NotFoundError):
                raise

"""In-memory repositories.

These are not toys: they implement the same concurrency and tenancy semantics
as the Cosmos adapters, including version checks that raise
:class:`ConcurrencyConflictError`.  That is deliberate -- the chaos tests drive
lost-update and crash scenarios through these implementations, so the rules
they exercise are the rules production runs under.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from typing import Any

from cortexflow.modules.approval.domain.models import Approval, ApprovalStatus
from cortexflow.modules.audit.domain.models import AuditEvent
from cortexflow.modules.document.domain.models import Document
from cortexflow.modules.document.ports.repositories import DocumentList
from cortexflow.modules.policy.domain.models import PolicyRuleset
from cortexflow.modules.workflow.domain.definition import WorkflowDefinition
from cortexflow.modules.workflow.domain.models import Workflow, WorkflowStatus
from cortexflow.modules.workflow.ports.repositories import (
    DeadLetterEntries,
    DefinitionList,
    WorkflowList,
    WorkflowPage,
    WorkflowRefs,
)
from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import ConcurrencyConflictError, ConflictError, NotFoundError
from cortexflow.shared.ids import IdGenerator, UuidIdGenerator


def _key(tenant_id: str, entity_id: str) -> str:
    return f"{tenant_id}::{entity_id}"


class InMemoryWorkflowRepository:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(self, workflow: Workflow) -> Workflow:
        async with self._lock:
            key = _key(workflow.tenant_id, workflow.workflow_id)
            if key in self._items:
                raise ConflictError(
                    "Workflow already exists", workflow_id=workflow.workflow_id
                )
            workflow.version = 1
            workflow.updated_at = utcnow()
            self._items[key] = workflow.model_dump_json()
            return workflow.model_copy(deep=True)

    async def get(self, tenant_id: str, workflow_id: str) -> Workflow:
        workflow = await self.try_get(tenant_id, workflow_id)
        if workflow is None:
            raise NotFoundError(
                "Workflow not found", workflow_id=workflow_id, tenant_id=tenant_id
            )
        return workflow

    async def try_get(self, tenant_id: str, workflow_id: str) -> Workflow | None:
        raw = self._items.get(_key(tenant_id, workflow_id))
        return Workflow.model_validate_json(raw) if raw else None

    async def save(self, workflow: Workflow, *, expected_version: int) -> Workflow:
        async with self._lock:
            key = _key(workflow.tenant_id, workflow.workflow_id)
            raw = self._items.get(key)
            if raw is None:
                raise NotFoundError("Workflow not found", workflow_id=workflow.workflow_id)
            stored_version = json.loads(raw)["version"]
            if stored_version != expected_version:
                raise ConcurrencyConflictError(
                    "Workflow was modified concurrently",
                    workflow_id=workflow.workflow_id,
                    expected_version=expected_version,
                    actual_version=stored_version,
                )
            workflow.version = expected_version + 1
            workflow.updated_at = utcnow()
            self._items[key] = workflow.model_dump_json()
            return workflow.model_copy(deep=True)

    async def list(
        self,
        tenant_id: str,
        *,
        status: WorkflowStatus | None = None,
        definition_name: str | None = None,
        limit: int = 50,
        continuation: str | None = None,
    ) -> WorkflowPage:
        rows = [
            Workflow.model_validate_json(raw)
            for key, raw in self._items.items()
            if key.startswith(f"{tenant_id}::")
        ]
        if status is not None:
            rows = [w for w in rows if w.status is status]
        if definition_name:
            rows = [w for w in rows if w.definition_name == definition_name]
        rows.sort(key=lambda w: w.created_at, reverse=True)

        offset = _decode_continuation(continuation)
        page = rows[offset : offset + limit]
        next_token = (
            _encode_continuation(offset + limit) if offset + limit < len(rows) else None
        )
        return [w.to_summary() for w in page], next_token

    async def list_children(
        self, tenant_id: str, parent_workflow_id: str, parent_step_id: str
    ) -> WorkflowList:
        children = [
            workflow
            for raw in self._items.values()
            if (workflow := Workflow.model_validate_json(raw)).tenant_id == tenant_id
            and workflow.parent_workflow_id == parent_workflow_id
            and workflow.parent_step_id == parent_step_id
        ]
        return sorted(children, key=lambda w: w.fan_out_index or 0)

    async def find_due(self, *, now: datetime, limit: int = 100) -> WorkflowRefs:
        due: list[tuple[str, str]] = []
        for raw in self._items.values():
            workflow = Workflow.model_validate_json(raw)
            if workflow.status.is_terminal:
                continue
            if workflow.scheduled_wake_at is not None and workflow.scheduled_wake_at <= now:
                due.append((workflow.tenant_id, workflow.workflow_id))
            if len(due) >= limit:
                break
        return due

    async def find_stalled(
        self, *, older_than: datetime, limit: int = 100
    ) -> WorkflowRefs:
        stalled: list[tuple[str, str]] = []
        for raw in self._items.values():
            workflow = Workflow.model_validate_json(raw)
            if workflow.status is not WorkflowStatus.RUNNING:
                continue
            expiries = [
                s.lease_expires_at
                for s in workflow.steps.values()
                if s.status.is_in_flight and s.lease_expires_at is not None
            ]
            if expiries and max(expiries) <= older_than:
                stalled.append((workflow.tenant_id, workflow.workflow_id))
            if len(stalled) >= limit:
                break
        return stalled


class InMemoryApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(self, approval: Approval) -> Approval:
        async with self._lock:
            key = _key(approval.tenant_id, approval.approval_id)
            if key in self._items:
                raise ConflictError("Approval already exists", approval_id=approval.approval_id)
            approval.version = 1
            self._items[key] = approval.model_dump_json()
            return approval.model_copy(deep=True)

    async def get(self, tenant_id: str, approval_id: str) -> Approval:
        raw = self._items.get(_key(tenant_id, approval_id))
        if raw is None:
            raise NotFoundError("Approval not found", approval_id=approval_id)
        return Approval.model_validate_json(raw)

    async def save(self, approval: Approval, *, expected_version: int) -> Approval:
        async with self._lock:
            key = _key(approval.tenant_id, approval.approval_id)
            raw = self._items.get(key)
            if raw is None:
                raise NotFoundError("Approval not found", approval_id=approval.approval_id)
            if json.loads(raw)["version"] != expected_version:
                raise ConcurrencyConflictError(
                    "Approval was modified concurrently",
                    approval_id=approval.approval_id,
                )
            approval.version = expected_version + 1
            approval.updated_at = utcnow()
            self._items[key] = approval.model_dump_json()
            return approval.model_copy(deep=True)

    async def list_pending(
        self, tenant_id: str, *, workflow_id: str | None = None, limit: int = 50
    ) -> list[Approval]:
        rows = [
            Approval.model_validate_json(raw)
            for key, raw in self._items.items()
            if key.startswith(f"{tenant_id}::")
        ]
        rows = [a for a in rows if a.status is ApprovalStatus.PENDING]
        if workflow_id:
            rows = [a for a in rows if a.workflow_id == workflow_id]
        rows.sort(key=lambda a: a.created_at)
        return rows[:limit]

    async def find_expired(self, *, now: datetime, limit: int = 100) -> list[Approval]:
        rows = [Approval.model_validate_json(raw) for raw in self._items.values()]
        expired = [
            a for a in rows if a.status is ApprovalStatus.PENDING and a.is_expired(now)
        ]
        return expired[:limit]

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


class InMemoryAuditRepository:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = asyncio.Lock()

    async def append(self, event: AuditEvent) -> None:
        async with self._lock:
            self._events.append(event)

    async def append_many(self, events: list[AuditEvent]) -> None:
        async with self._lock:
            self._events.extend(events)

    async def query(
        self,
        tenant_id: str,
        *,
        workflow_id: str | None = None,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[AuditEvent]:
        rows = [e for e in self._events if e.tenant_id == tenant_id]
        if workflow_id:
            rows = [e for e in rows if e.workflow_id == workflow_id]
        if actor:
            rows = [e for e in rows if e.actor == actor]
        if since:
            rows = [e for e in rows if e.occurred_at >= since]
        rows.sort(key=lambda e: e.occurred_at)
        return rows[-limit:]


class InMemoryIdempotencyStore:
    """Reservation-based idempotency with TTL expiry."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[datetime, dict[str, Any] | None]] = {}
        self._lock = asyncio.Lock()

    def _purge(self) -> None:
        now = utcnow()
        for key, (expires_at, _) in list(self._entries.items()):
            if expires_at <= now:
                del self._entries[key]

    async def reserve(self, key: str, *, ttl_seconds: int) -> bool:
        from datetime import timedelta

        async with self._lock:
            self._purge()
            if key in self._entries:
                return False
            self._entries[key] = (utcnow() + timedelta(seconds=ttl_seconds), None)
            return True

    async def complete(self, key: str, result: dict[str, Any], *, ttl_seconds: int) -> None:
        from datetime import timedelta

        async with self._lock:
            self._entries[key] = (utcnow() + timedelta(seconds=ttl_seconds), result)

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            self._purge()
            entry = self._entries.get(key)
            return entry[1] if entry else None

    async def release(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)


class InMemoryDeadLetterRepository:
    def __init__(self, ids: IdGenerator | None = None) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
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
        self._entries[_key(tenant_id, entry_id)] = {
            "entry_id": entry_id,
            "tenant_id": tenant_id,
            "queue": queue,
            "message": message,
            "reason": reason,
            "attempts": attempts,
            "recorded_at": utcnow().isoformat(),
        }
        return entry_id

    async def list(self, tenant_id: str, *, limit: int = 50) -> DeadLetterEntries:
        rows = [
            entry
            for key, entry in self._entries.items()
            if key.startswith(f"{tenant_id}::")
        ]
        rows.sort(key=lambda e: e["recorded_at"], reverse=True)
        return rows[:limit]

    async def get(self, tenant_id: str, entry_id: str) -> dict[str, Any] | None:
        return self._entries.get(_key(tenant_id, entry_id))

    async def discard(self, tenant_id: str, entry_id: str) -> None:
        self._entries.pop(_key(tenant_id, entry_id), None)


def _encode_continuation(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def _decode_continuation(token: str | None) -> int:
    if not token:
        return 0
    try:
        return int(base64.urlsafe_b64decode(token.encode()).decode())
    except (ValueError, UnicodeDecodeError):
        return 0


class InMemoryDocumentRepository:
    """Uploaded documents, tenant-scoped like everything else."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    async def create(self, document: Document) -> Document:
        key = _key(document.tenant_id, document.document_id)
        if key in self._items:
            raise ConflictError("Document already exists", document_id=document.document_id)
        self._items[key] = document.model_dump_json()
        return document.model_copy(deep=True)

    async def get(self, tenant_id: str, document_id: str) -> Document:
        raw = self._items.get(_key(tenant_id, document_id))
        if raw is None:
            raise NotFoundError("Document not found", document_id=document_id)
        return Document.model_validate_json(raw)

    async def list(self, tenant_id: str, *, limit: int = 50) -> DocumentList:
        rows = [
            Document.model_validate_json(raw)
            for key, raw in self._items.items()
            if key.startswith(f"{tenant_id}::")
        ]
        rows.sort(key=lambda d: d.uploaded_at, reverse=True)
        return rows[:limit]

    async def delete(self, tenant_id: str, document_id: str) -> None:
        self._items.pop(_key(tenant_id, document_id), None)


class InMemoryWorkflowDefinitionRepository:
    """Runtime-authored workflow definitions, per tenant and per version."""

    def __init__(self) -> None:
        # tenant -> "name@version" -> definition JSON
        self._items: dict[str, dict[str, str]] = {}

    async def save(
        self, tenant_id: str, definition: WorkflowDefinition
    ) -> WorkflowDefinition:
        self._items.setdefault(tenant_id, {})[definition.key] = (
            definition.model_dump_json()
        )
        return definition

    async def get(
        self, tenant_id: str, name: str, version: int | None = None
    ) -> WorkflowDefinition | None:
        versions = self._versions(tenant_id, name)
        if not versions:
            return None
        if version is None:
            return max(versions, key=lambda d: d.version)
        return next((d for d in versions if d.version == version), None)

    async def list(self, tenant_id: str) -> DefinitionList:
        latest: dict[str, WorkflowDefinition] = {}
        for raw in self._items.get(tenant_id, {}).values():
            definition = WorkflowDefinition.model_validate_json(raw)
            current = latest.get(definition.name)
            if current is None or definition.version > current.version:
                latest[definition.name] = definition
        return sorted(latest.values(), key=lambda d: d.name)

    async def delete(self, tenant_id: str, name: str) -> None:
        store = self._items.get(tenant_id, {})
        for key in [k for k in store if k.rsplit("@", 1)[0] == name]:
            del store[key]

    def _versions(self, tenant_id: str, name: str) -> DefinitionList:
        return [
            WorkflowDefinition.model_validate_json(raw)
            for key, raw in self._items.get(tenant_id, {}).items()
            if key.rsplit("@", 1)[0] == name
        ]


class InMemoryPolicyRulesetRepository:
    """Runtime-authored policy rulesets, per tenant.

    One row per name, not per version: the engine only ever evaluates the
    current rules, and the history that matters -- who changed what, when, and
    how -- lives in the audit trail, where it cannot be deleted along with the
    ruleset it describes.
    """

    def __init__(self) -> None:
        # tenant -> name -> ruleset JSON
        self._items: dict[str, dict[str, str]] = {}

    async def save(self, tenant_id: str, ruleset: PolicyRuleset) -> PolicyRuleset:
        self._items.setdefault(tenant_id, {})[ruleset.name] = ruleset.model_dump_json()
        return ruleset

    async def get(self, tenant_id: str, name: str) -> PolicyRuleset | None:
        raw = self._items.get(tenant_id, {}).get(name)
        return PolicyRuleset.model_validate_json(raw) if raw else None

    async def list(self, tenant_id: str) -> list[PolicyRuleset]:
        return sorted(
            (
                PolicyRuleset.model_validate_json(raw)
                for raw in self._items.get(tenant_id, {}).values()
            ),
            key=lambda r: r.name,
        )

    async def delete(self, tenant_id: str, name: str) -> None:
        self._items.get(tenant_id, {}).pop(name, None)


class InMemoryWorkflowDraftRepository:
    """The builder draft behind each authored definition, per tenant."""

    def __init__(self) -> None:
        # tenant -> definition name -> draft JSON
        self._items: dict[str, dict[str, str]] = {}

    async def save(self, tenant_id: str, name: str, draft: dict[str, Any]) -> None:
        self._items.setdefault(tenant_id, {})[name] = json.dumps(draft)

    async def get(self, tenant_id: str, name: str) -> dict[str, Any] | None:
        raw = self._items.get(tenant_id, {}).get(name)
        return json.loads(raw) if raw else None

    async def delete(self, tenant_id: str, name: str) -> None:
        self._items.get(tenant_id, {}).pop(name, None)

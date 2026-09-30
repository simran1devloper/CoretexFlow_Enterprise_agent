"""Cosmos DB storage for what people author at runtime.

Uploaded documents, builder-authored workflow definitions and tenant policy
rulesets all live in the workflow container, tenant-partitioned like
everything else, distinguished by a ``doc_type`` discriminator. They are
small, read far more often than written, and always fetched by tenant -- so a
separate container each would buy nothing but two more things to provision.
"""

from __future__ import annotations

from typing import Any

from cortexflow.config.settings import CosmosSettings
from cortexflow.infrastructure.cosmos.client import translate_error
from cortexflow.modules.document.domain.models import Document
from cortexflow.modules.document.ports.repositories import DocumentList
from cortexflow.modules.policy.domain.models import PolicyRuleset
from cortexflow.modules.workflow.domain.definition import WorkflowDefinition
from cortexflow.modules.workflow.ports.repositories import DefinitionList
from cortexflow.shared.errors import NotFoundError

DOCUMENT_TYPE = "uploaded_document"
DEFINITION_TYPE = "workflow_definition"
DRAFT_TYPE = "workflow_draft"
RULESET_TYPE = "policy_ruleset"


class _Base:
    def __init__(self, client: Any, settings: CosmosSettings) -> None:
        self._client = client
        self._settings = settings

    @property
    def _container(self) -> Any:
        return self._client.get_database_client(
            self._settings.database
        ).get_container_client(self._settings.workflow_container)

    async def _query(
        self, query: str, parameters: list[dict[str, Any]], partition_key: str
    ) -> list[dict[str, Any]]:
        return [
            item
            async for item in self._container.query_items(
                query=query, parameters=parameters, partition_key=partition_key
            )
        ]


class CosmosDocumentRepository(_Base):
    async def create(self, document: Document) -> Document:
        payload = document.model_dump(mode="json")
        payload["id"] = document.document_id
        payload["doc_type"] = DOCUMENT_TYPE
        try:
            await self._container.create_item(payload)
        except Exception as exc:
            raise translate_error(exc, document_id=document.document_id) from exc
        return document

    async def get(self, tenant_id: str, document_id: str) -> Document:
        try:
            doc = await self._container.read_item(document_id, partition_key=tenant_id)
        except Exception as exc:
            raise translate_error(exc, document_id=document_id) from exc
        if doc.get("doc_type") != DOCUMENT_TYPE:
            raise NotFoundError("Document not found", document_id=document_id)
        return Document.model_validate(_clean(doc))

    async def list(self, tenant_id: str, *, limit: int = 50) -> DocumentList:
        rows = await self._query(
            "SELECT TOP @limit * FROM c WHERE c.tenant_id = @tenant "
            "AND c.doc_type = @kind ORDER BY c.uploaded_at DESC",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@kind", "value": DOCUMENT_TYPE},
                {"name": "@limit", "value": limit},
            ],
            tenant_id,
        )
        return [Document.model_validate(_clean(row)) for row in rows]

    async def delete(self, tenant_id: str, document_id: str) -> None:
        try:
            await self._container.delete_item(document_id, partition_key=tenant_id)
        except Exception as exc:
            if not isinstance(translate_error(exc), NotFoundError):
                raise


class CosmosWorkflowDefinitionRepository(_Base):
    @staticmethod
    def _doc_id(tenant_id: str, definition: WorkflowDefinition) -> str:
        return f"def::{tenant_id}::{definition.key}"

    async def save(
        self, tenant_id: str, definition: WorkflowDefinition
    ) -> WorkflowDefinition:
        payload = definition.model_dump(mode="json")
        payload["id"] = self._doc_id(tenant_id, definition)
        payload["doc_type"] = DEFINITION_TYPE
        payload["tenant_id"] = tenant_id
        payload["definition_name"] = definition.name
        try:
            await self._container.upsert_item(payload)
        except Exception as exc:
            raise translate_error(exc, workflow=definition.key) from exc
        return definition

    async def get(
        self, tenant_id: str, name: str, version: int | None = None
    ) -> WorkflowDefinition | None:
        clauses = [
            "c.tenant_id = @tenant",
            "c.doc_type = @kind",
            "c.definition_name = @name",
        ]
        parameters: list[dict[str, Any]] = [
            {"name": "@tenant", "value": tenant_id},
            {"name": "@kind", "value": DEFINITION_TYPE},
            {"name": "@name", "value": name},
        ]
        if version is not None:
            clauses.append("c.version = @version")
            parameters.append({"name": "@version", "value": version})

        rows = await self._query(
            f"SELECT * FROM c WHERE {' AND '.join(clauses)} "
            "ORDER BY c.version DESC",
            parameters,
            tenant_id,
        )
        return WorkflowDefinition.model_validate(_clean(rows[0])) if rows else None

    async def list(self, tenant_id: str) -> DefinitionList:
        rows = await self._query(
            "SELECT * FROM c WHERE c.tenant_id = @tenant AND c.doc_type = @kind",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@kind", "value": DEFINITION_TYPE},
            ],
            tenant_id,
        )
        latest: dict[str, WorkflowDefinition] = {}
        for row in rows:
            definition = WorkflowDefinition.model_validate(_clean(row))
            current = latest.get(definition.name)
            if current is None or definition.version > current.version:
                latest[definition.name] = definition
        return sorted(latest.values(), key=lambda d: d.name)

    async def delete(self, tenant_id: str, name: str) -> None:
        rows = await self._query(
            "SELECT c.id FROM c WHERE c.tenant_id = @tenant "
            "AND c.doc_type = @kind AND c.definition_name = @name",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@kind", "value": DEFINITION_TYPE},
                {"name": "@name", "value": name},
            ],
            tenant_id,
        )
        for row in rows:
            try:
                await self._container.delete_item(row["id"], partition_key=tenant_id)
            except Exception as exc:
                if not isinstance(translate_error(exc), NotFoundError):
                    raise


class CosmosPolicyRulesetRepository(_Base):
    """Policy rulesets a tenant authored, one document per name.

    Deliberately not versioned in storage, unlike workflow definitions. A
    running workflow pins the definition version it started with, so old
    versions have to stay readable; a policy decision does not replay, it is
    recorded whole -- effect, matched rules and inputs digest -- in the audit
    trail at the moment it was made. Keeping superseded rules here as well
    would be a second, deletable copy of a record that already exists
    somewhere it cannot be deleted from.
    """

    @staticmethod
    def _doc_id(tenant_id: str, name: str) -> str:
        return f"policy::{tenant_id}::{name}"

    async def save(self, tenant_id: str, ruleset: PolicyRuleset) -> PolicyRuleset:
        payload = ruleset.model_dump(mode="json")
        payload["id"] = self._doc_id(tenant_id, ruleset.name)
        payload["doc_type"] = RULESET_TYPE
        payload["tenant_id"] = tenant_id
        payload["ruleset_name"] = ruleset.name
        try:
            await self._container.upsert_item(payload)
        except Exception as exc:
            raise translate_error(exc, ruleset=ruleset.key) from exc
        return ruleset

    async def get(self, tenant_id: str, name: str) -> PolicyRuleset | None:
        try:
            doc = await self._container.read_item(
                self._doc_id(tenant_id, name), partition_key=tenant_id
            )
        except Exception as exc:
            if isinstance(translate_error(exc), NotFoundError):
                return None
            raise translate_error(exc, ruleset=name) from exc
        return PolicyRuleset.model_validate(_ruleset_fields(doc))

    async def list(self, tenant_id: str) -> list[PolicyRuleset]:
        rows = await self._query(
            "SELECT * FROM c WHERE c.tenant_id = @tenant AND c.doc_type = @kind",
            [
                {"name": "@tenant", "value": tenant_id},
                {"name": "@kind", "value": RULESET_TYPE},
            ],
            tenant_id,
        )
        rulesets = [PolicyRuleset.model_validate(_ruleset_fields(row)) for row in rows]
        return sorted(rulesets, key=lambda r: r.name)

    async def delete(self, tenant_id: str, name: str) -> None:
        try:
            await self._container.delete_item(
                self._doc_id(tenant_id, name), partition_key=tenant_id
            )
        except Exception as exc:
            if not isinstance(translate_error(exc), NotFoundError):
                raise


def _clean(document: dict[str, Any]) -> dict[str, Any]:
    """Strip Cosmos bookkeeping and our own discriminators.

    ``tenant_id`` stays: a ``Document`` carries one as a real field. The
    ruleset reader drops it separately, because ``PolicyRuleset`` forbids
    extras and knows nothing about tenancy -- it is the same rules whoever
    stored them.
    """
    return {
        k: v
        for k, v in document.items()
        if not k.startswith("_")
        and k not in {"id", "doc_type", "definition_name", "ruleset_name"}
    }


class CosmosWorkflowDraftRepository(_Base):
    """The builder draft behind each authored definition.

    One row per definition name, not per version: the draft is what someone
    opens to edit, and that is always the current one. Superseded versions
    stay readable as *definitions*, which is what a run in flight pins.
    """

    @staticmethod
    def _doc_id(tenant_id: str, name: str) -> str:
        return f"draft::{tenant_id}::{name}"

    async def save(self, tenant_id: str, name: str, draft: dict[str, Any]) -> None:
        try:
            await self._container.upsert_item(
                {
                    "id": self._doc_id(tenant_id, name),
                    "doc_type": DRAFT_TYPE,
                    "tenant_id": tenant_id,
                    "definition_name": name,
                    "draft": draft,
                }
            )
        except Exception as exc:
            raise translate_error(exc, workflow=name) from exc

    async def get(self, tenant_id: str, name: str) -> dict[str, Any] | None:
        try:
            row = await self._container.read_item(
                self._doc_id(tenant_id, name), partition_key=tenant_id
            )
        except Exception as exc:
            if isinstance(translate_error(exc), NotFoundError):
                return None
            raise translate_error(exc, workflow=name) from exc
        found = row.get("draft")
        return found if isinstance(found, dict) else None

    async def delete(self, tenant_id: str, name: str) -> None:
        try:
            await self._container.delete_item(
                self._doc_id(tenant_id, name), partition_key=tenant_id
            )
        except Exception as exc:
            if not isinstance(translate_error(exc), NotFoundError):
                raise


def _ruleset_fields(document: dict[str, Any]) -> dict[str, Any]:
    fields = _clean(document)
    fields.pop("tenant_id", None)
    return fields

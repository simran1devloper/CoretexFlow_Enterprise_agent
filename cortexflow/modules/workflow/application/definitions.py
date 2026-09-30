"""Workflow definition loading and registry.

Definitions are validated at startup: cycles, unknown dependencies, malformed
guards and unknown agents or tools all fail fast, before a single workflow is
accepted.  A broken definition should fail a deployment, not a workflow that
is already halfway through a payment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cortexflow.modules.workflow.domain.definition import (
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.shared.errors import NotFoundError, WorkflowDefinitionError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


def load_definition(path: Path) -> WorkflowDefinition:
    """Parse and validate one workflow YAML file."""
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise WorkflowDefinitionError("Malformed workflow YAML", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise WorkflowDefinitionError("Workflow file must contain a mapping", path=str(path))

    # Accept both a bare mapping and one nested under a `workflow:` key.
    body = raw.get("workflow", raw)
    try:
        return WorkflowDefinition.model_validate(body)
    except Exception as exc:
        raise WorkflowDefinitionError(
            "Invalid workflow definition", path=str(path), error=str(exc)
        ) from exc


class WorkflowDefinitionRegistry:
    """Holds every known definition version, keyed by name.

    Two populations live here, deliberately separated:

    * **built-in** definitions, loaded from disk at startup. They are reviewed,
      versioned with the code, and shared by every tenant.
    * **tenant** definitions, authored at runtime through the builder. They
      belong to one tenant and may change between two runs.

    Resolution checks the tenant's own definitions first, so a tenant can
    shadow a built-in name without affecting anyone else. Both populations are
    validated identically -- authoring a workflow in the UI does not buy a
    weaker check than shipping one in the repository.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, WorkflowDefinition] = {}
        self._latest: dict[str, int] = {}
        self._tenant: dict[str, dict[str, WorkflowDefinition]] = {}
        self._tenant_latest: dict[str, dict[str, int]] = {}

    def register(self, definition: WorkflowDefinition) -> None:
        if definition.key in self._by_key:
            raise WorkflowDefinitionError(
                "Duplicate workflow definition version", workflow=definition.key
            )
        self._by_key[definition.key] = definition
        if definition.version >= self._latest.get(definition.name, 0):
            self._latest[definition.name] = definition.version
        logger.info(
            "workflow definition registered",
            extra={
                "workflow": definition.key,
                "department": str(definition.department),
                "steps": len(definition.steps),
            },
        )

    def register_for_tenant(
        self, tenant_id: str, definition: WorkflowDefinition
    ) -> None:
        """Register (or replace) a tenant's own definition version."""
        self._tenant.setdefault(tenant_id, {})[definition.key] = definition
        latest = self._tenant_latest.setdefault(tenant_id, {})
        if definition.version >= latest.get(definition.name, 0):
            latest[definition.name] = definition.version

    def forget_for_tenant(self, tenant_id: str, name: str) -> None:
        for key in [
            k for k in self._tenant.get(tenant_id, {}) if k.rsplit("@", 1)[0] == name
        ]:
            del self._tenant[tenant_id][key]
        self._tenant_latest.get(tenant_id, {}).pop(name, None)

    def tenant_definitions(self, tenant_id: str) -> list[WorkflowDefinition]:
        """The latest version of each definition this tenant authored."""
        latest = self._tenant_latest.get(tenant_id, {})
        return [
            self._tenant[tenant_id][f"{name}@{version}"]
            for name, version in sorted(latest.items())
        ]

    def get(
        self, name: str, version: int | None = None, tenant_id: str | None = None
    ) -> WorkflowDefinition:
        """Resolve a definition, defaulting to the latest version.

        A running workflow always pins the version it started with, so
        publishing v2 never changes the behaviour of an in-flight v1 workflow.
        """
        if tenant_id is not None:
            found = self._resolve_tenant(tenant_id, name, version)
            if found is not None:
                return found

        resolved = version if version is not None else self._latest.get(name)
        if resolved is None:
            raise NotFoundError("Unknown workflow definition", workflow=name)
        try:
            return self._by_key[f"{name}@{resolved}"]
        except KeyError as exc:
            raise NotFoundError(
                "Unknown workflow definition version", workflow=name, version=resolved
            ) from exc

    def _resolve_tenant(
        self, tenant_id: str, name: str, version: int | None
    ) -> WorkflowDefinition | None:
        store = self._tenant.get(tenant_id, {})
        if not store:
            return None
        resolved = (
            version
            if version is not None
            else self._tenant_latest.get(tenant_id, {}).get(name)
        )
        return store.get(f"{name}@{resolved}") if resolved is not None else None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._latest))

    def all(self) -> list[WorkflowDefinition]:
        return list(self._by_key.values())

    def all_for_tenant(self, tenant_id: str) -> list[WorkflowDefinition]:
        """The latest version of everything this tenant can start.

        Shipped definitions plus the tenant's own, with the tenant's winning
        where both exist -- the same precedence ``get`` applies, so the
        builder offers exactly what would actually run.
        """
        latest: dict[str, WorkflowDefinition] = {}
        for definition in self._by_key.values():
            current = latest.get(definition.name)
            if current is None or definition.version > current.version:
                latest[definition.name] = definition
        for definition in self._tenant.get(tenant_id, {}).values():
            current = latest.get(definition.name)
            if (
                current is None
                or current.version <= definition.version
                or definition.name not in latest
            ):
                latest[definition.name] = definition
        return list(latest.values())

    def latest(self) -> list[WorkflowDefinition]:
        return [self.get(name) for name in self.names()]

    def validate_against(
        self, *, agent_names: tuple[str, ...], tool_names: tuple[str, ...],
        ruleset_names: tuple[str, ...],
    ) -> None:
        """Cross-check every definition against what is actually registered."""
        known_agents, known_tools = set(agent_names), set(tool_names)
        known_rulesets = set(ruleset_names)
        known = {d.name for d in self._by_key.values()}
        problems: list[str] = []

        for definition in self._by_key.values():
            for step in definition.steps:
                match step.type:
                    case StepType.AGENT if step.agent not in known_agents:
                        problems.append(
                            f"{definition.key}.{step.id}: unknown agent '{step.agent}'"
                        )
                    case StepType.TOOL if step.tool not in known_tools:
                        problems.append(
                            f"{definition.key}.{step.id}: unknown tool '{step.tool}'"
                        )
                    case StepType.POLICY if step.ruleset not in known_rulesets:
                        problems.append(
                            f"{definition.key}.{step.id}: unknown ruleset '{step.ruleset}'"
                        )
                    case StepType.FAN_OUT:
                        problems.extend(_fan_out_problems(definition, step, known))
                    case _:
                        pass

        if problems:
            raise WorkflowDefinitionError(
                "Workflow definitions reference unknown components", problems=problems
            )


def _fan_out_problems(
    definition: WorkflowDefinition, step: StepDefinition, known: set[str]
) -> list[str]:
    """Check a fan-out step's target before anything is allowed to run.

    Recursion is refused outright rather than bounded. A workflow that fans
    out to itself is almost always a mistake, and the failure mode -- work
    multiplying until something gives -- is one nobody should discover in
    production.
    """
    assert step.fan_out is not None  # StepDefinition validation guarantees it
    target = step.fan_out.workflow
    problems: list[str] = []
    if target not in known:
        problems.append(
            f"{definition.key}.{step.id}: unknown child workflow '{target}'"
        )
    if target == definition.name:
        problems.append(
            f"{definition.key}.{step.id}: fans out to itself, which would not terminate"
        )
    return problems


def load_registry(
    directory: Path, *, runtime: str | None = None
) -> WorkflowDefinitionRegistry:
    """Load every ``*.yaml`` workflow under ``directory``.

    A definition that declares ``requires_runtime`` is skipped unless that
    runtime is the configured one. Skipping is deliberate and logged: the
    alternative is a deployment that fails because of a workflow the
    deployment never intended to run.
    """
    registry = WorkflowDefinitionRegistry()
    if not directory.exists():
        logger.warning("workflow directory missing", extra={"path": str(directory)})
        return registry

    for path in sorted(directory.rglob("*.yaml")):
        definition = load_definition(path)
        required = definition.requires_runtime
        if required and runtime is not None and required != runtime:
            logger.info(
                "workflow skipped: it requires a runtime that is not configured",
                extra={
                    "workflow": definition.key,
                    "requires_runtime": required,
                    "configured_runtime": runtime,
                },
            )
            continue
        registry.register(definition)
    return registry

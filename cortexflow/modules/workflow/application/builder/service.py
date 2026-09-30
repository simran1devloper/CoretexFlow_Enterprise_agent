"""The workflow builder service.

Saving a workflow built in the browser runs the *same* checks as shipping one
in the repository: the graph must be acyclic, guards must parse, and every
agent, tool and ruleset it names must actually exist. A workflow authored in
the UI does not get a weaker contract than one authored in YAML -- it simply
gets the errors back while someone is still editing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.domain.models import SideEffect
from cortexflow.modules.workflow.application.builder.catalogue import (
    BY_KIND,
    CATALOGUE,
    TRIAGE_TOOL,
    RulesetChoice,
    ToolChoice,
)
from cortexflow.modules.workflow.application.builder.compiler import (
    WorkflowDraft,
    compile_draft,
    draft_problems,
    parse_draft,
)
from cortexflow.modules.workflow.application.builder.contract import (
    WorkflowContract,
    build_contract,
)
from cortexflow.modules.workflow.application.definitions import WorkflowDefinitionRegistry
from cortexflow.modules.workflow.domain.dag import critical_path, parallel_levels
from cortexflow.modules.workflow.domain.definition import (
    StepDefinition,
    StepType,
    WorkflowDefinition,
)
from cortexflow.modules.workflow.ports.repositories import (
    DefinitionList,
    WorkflowDefinitionRepository,
    WorkflowDraftRepository,
)
from cortexflow.shared.errors import NotFoundError, ValidationError
from cortexflow.shared.identity import Principal
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.security.rbac import Permission, require_department, require_permission

logger = get_logger(__name__)

# Declared at module scope: the service's own ``list`` method shadows the
# builtin inside the class body, so annotations written there cannot use it.
StringList = list[str]


class BuilderService:
    def __init__(
        self,
        *,
        definitions: WorkflowDefinitionRegistry,
        repository: WorkflowDefinitionRepository,
        drafts: WorkflowDraftRepository,
        tools: ToolRegistry,
        policy: PolicyEngine,
        agents: Any,
        audit: AuditTrail,
    ) -> None:
        self._definitions = definitions
        self._repository = repository
        self._drafts = drafts
        self._tools = tools
        self._policy = policy
        self._agents = agents
        self._audit = audit

    # -- what the builder can offer ---------------------------------------
    def catalogue(self, principal: Principal) -> dict[str, Any]:
        """Everything the builder needs to render its palette."""
        require_permission(principal, Permission.DEFINITION_READ)
        return {
            "step_kinds": [spec.model_dump(mode="json") for spec in CATALOGUE],
            "tools": [t.model_dump(mode="json") for t in self._tool_choices(principal)],
            "rulesets": [
                r.model_dump(mode="json")
                for r in self._ruleset_choices(principal.tenant_id)
            ],
            "agents": self._agents.describe(),
            "workflows": self._child_workflow_choices(principal),
        }

    def _child_workflow_choices(self, principal: Principal) -> list[dict[str, Any]]:
        """Workflows a Run Per Row step may start for each item.

        Only this principal's own department, because fanning out is starting
        work: a step that could run any workflow in the tenant would be a way
        around the departmental boundary rather than a batch feature.
        """
        choices: list[dict[str, Any]] = []
        for definition in self._definitions.all_for_tenant(principal.tenant_id):
            if not principal.can_access_department(definition.department):
                continue
            choices.append(
                {
                    "name": definition.name,
                    "version": definition.version,
                    "description": definition.description,
                    "department": str(definition.department),
                    "step_count": len(definition.steps),
                    "needs_approval": any(
                        s.type is StepType.HUMAN_APPROVAL for s in definition.steps
                    ),
                    # A workflow that starts from a file cannot be fed rows:
                    # every case would fail on fields it never extracted. The
                    # builder says so rather than letting someone find out a
                    # hundred failed runs later.
                    "needs_document": "document_id"
                    in definition.input_schema.get("required", [])
                    or "document_text"
                    in definition.input_schema.get("required", []),
                }
            )
        return sorted(choices, key=lambda c: c["name"])

    def _tool_choices(self, principal: Principal) -> list[ToolChoice]:
        """Tools this principal may wire into an Action step.

        Filtered by the same authorization the registry applies at execution
        time, so the builder cannot offer something that would be refused when
        the workflow runs.
        """
        choices: list[ToolChoice] = []
        for metadata in self._tools.list_metadata():
            if not self._tools.authorize(metadata.name, principal).allowed:
                continue
            tool = self._tools.get(metadata.name)
            arguments: list[dict[str, Any]] = []
            if tool.input_model is not None:
                for name, field in tool.input_model.model_fields.items():
                    arguments.append(
                        {
                            "name": name,
                            "type": getattr(field.annotation, "__name__", "string"),
                            "required": field.is_required(),
                            "description": field.description or "",
                        }
                    )
            choices.append(
                ToolChoice(
                    name=metadata.name,
                    label=metadata.name.replace(".", " · ").replace("_", " "),
                    domain=str(metadata.domain),
                    description=metadata.description,
                    risk=str(metadata.risk),
                    side_effect=str(metadata.side_effect),
                    requires_approval=metadata.requires_human_approval,
                    arguments=arguments,
                )
            )
        return sorted(choices, key=lambda c: c.name)

    def _ruleset_choices(self, tenant_id: str) -> list[RulesetChoice]:
        choices: list[RulesetChoice] = []
        for ruleset in self._policy.rulesets_for(tenant_id).values():
            facts: set[str] = set()
            for rule in ruleset.rules:
                facts.update(_facts_in(rule.when))
            choices.append(
                RulesetChoice(
                    name=ruleset.name,
                    description=ruleset.description,
                    default_effect=str(ruleset.default_effect),
                    rule_count=len(ruleset.rules),
                    facts=sorted(facts),
                )
            )
        return sorted(choices, key=lambda c: c.name)

    # -- validate and save -------------------------------------------------
    def validate(self, principal: Principal, draft: WorkflowDraft) -> dict[str, Any]:
        """Check a draft without saving it, for live feedback in the builder.

        An incomplete draft is the normal case here, not an error: this is
        what the builder calls on every edit. So a draft that could not yet
        compile comes back as a list of problems to read, and only a draft
        that *does* compile gets the graph analysis layered on top.
        """
        require_permission(principal, Permission.DEFINITION_READ)

        incomplete = draft_problems(draft)
        if incomplete:
            return _unbuildable(draft, incomplete)
        try:
            definition = compile_draft(draft)
        except ValidationError as exc:
            # A cycle, or a guard that does not parse: the draft is complete
            # enough to compile but the result would not be a valid workflow.
            return _unbuildable(draft, [str(exc.details.get("error") or exc.message)])

        problems = self._reference_problems(definition, principal.tenant_id)
        fact_gaps = self._unmapped_facts(definition, principal.tenant_id)

        return {
            "valid": not problems,
            "problems": problems,
            "unmapped_facts": fact_gaps,
            "workflow": definition.name,
            "version": definition.version,
            "step_count": len(definition.steps),
            "waves": [list(level) for level in parallel_levels(definition)],
            "critical_path": list(critical_path(definition)),
            "uses_ai_steps": [
                s.id for s in definition.steps if s.type is StepType.AGENT
            ],
            "warnings": _advisory_warnings(definition, self._is_write) + [
                f"Policy step '{step}' does not supply: {', '.join(missing)}. "
                "Those rules cannot evaluate and the case will fall through to "
                "the ruleset default."
                for step, missing in fact_gaps.items()
            ],
        }

    def contract(self, principal: Principal, draft: WorkflowDraft) -> dict[str, Any]:
        """What this draft promises to do, for the reader about to publish it.

        Separate from ``validate`` on purpose. Validation runs on every edit
        and answers whether the workflow will execute; this runs once, at the
        moment someone commits, and answers whether it is the workflow they
        meant. Conflating them would bury the second question under a hundred
        repetitions of the first.
        """
        require_permission(principal, Permission.DEFINITION_READ)

        incomplete = draft_problems(draft)
        if incomplete:
            return {"ready": False, "problems": incomplete, "contract": None}
        try:
            definition = compile_draft(draft)
        except ValidationError as exc:
            return {
                "ready": False,
                "problems": [str(exc.details.get("error") or exc.message)],
                "contract": None,
            }

        problems = self._reference_problems(definition, principal.tenant_id)
        return {
            "ready": not problems,
            "problems": problems,
            "contract": self._contract_for(
                definition, principal.tenant_id
            ).model_dump(mode="json"),
        }

    def contract_for_definition(
        self, principal: Principal, name: str, version: int | None = None
    ) -> WorkflowContract:
        """The same reading, for a workflow that is already published.

        The question "what does this actually do?" does not stop being worth
        answering once a workflow is live -- it is the first thing anyone asks
        about a workflow they inherited.
        """
        require_permission(principal, Permission.DEFINITION_READ)
        definition = self._definitions.get(
            name, version, tenant_id=principal.tenant_id
        )
        require_department(principal, definition.department)
        return self._contract_for(definition, principal.tenant_id)

    def _contract_for(
        self, definition: WorkflowDefinition, tenant_id: str
    ) -> WorkflowContract:
        return build_contract(
            definition,
            tools=self._tools,
            policy=self._policy,
            tenant_id=tenant_id,
            unmapped_facts=self._unmapped_facts(definition, tenant_id),
        )

    async def save(
        self, principal: Principal, draft: WorkflowDraft
    ) -> WorkflowDefinition:
        """Compile, validate and publish a definition for this tenant."""
        require_permission(principal, Permission.DEFINITION_MANAGE)
        require_department(principal, draft.department)

        # Publishing is the strict path, and it re-checks rather than trusting
        # how the draft was parsed. Only /validate loosens the model, and a
        # loosened draft must not reach a saved definition by another route.
        incomplete = draft_problems(draft)
        if incomplete:
            raise ValidationError(
                "The workflow is not finished", problems=incomplete, workflow=draft.name
            )

        definition = compile_draft(draft)
        problems = self._reference_problems(definition, principal.tenant_id)
        if problems:
            raise ValidationError(
                "The workflow references components that do not exist",
                problems=problems,
                workflow=definition.name,
            )

        # Publishing a new version never disturbs a running workflow: each run
        # pins the version it started with.
        existing = await self._repository.get(principal.tenant_id, definition.name)
        if existing is not None and existing.version >= definition.version:
            definition = definition.model_copy(
                update={"version": existing.version + 1}
            )

        saved = await self._repository.save(principal.tenant_id, definition)
        # The draft is kept as the source this was compiled from, so the
        # builder can reopen exactly what someone assembled rather than a
        # guess reconstructed from the output. Written after the definition
        # on purpose: a stored draft with no definition behind it would be a
        # workflow that appears editable and cannot run.
        await self._drafts.save(
            principal.tenant_id, saved.name, draft.model_dump(mode="json")
        )
        self._definitions.register_for_tenant(principal.tenant_id, saved)

        logger.info(
            "workflow definition published from the builder",
            extra={
                "workflow": saved.key,
                "tenant_id": principal.tenant_id,
                "steps": len(saved.steps),
            },
        )
        await self._audit.record(
            tenant_id=principal.tenant_id,
            action=AuditAction.WORKFLOW_CREATED,
            actor=principal.subject,
            summary=f"Published workflow definition {saved.key}",
            metadata={"workflow": saved.key, "steps": len(saved.steps)},
        )
        return saved

    async def list(self, principal: Principal) -> DefinitionList:
        require_permission(principal, Permission.DEFINITION_READ)
        return await self._repository.list(principal.tenant_id)

    async def get(self, principal: Principal, name: str) -> WorkflowDefinition:
        require_permission(principal, Permission.DEFINITION_READ)
        found = await self._repository.get(principal.tenant_id, name)
        if found is None:
            raise NotFoundError("No such workflow definition", workflow=name)
        return found

    async def draft(self, principal: Principal, name: str) -> WorkflowDraft:
        """Reopen what someone assembled, to edit and publish again.

        Only a workflow authored here has one. A workflow shipped in the
        repository is reviewed and versioned with the code, so the answer is
        that it cannot be edited here rather than a draft reconstructed from
        its definition -- which would be a decompiler mirroring the compiler,
        drifting the first time either changed, and quietly producing a
        workflow that is not the one on disk.
        """
        require_permission(principal, Permission.DEFINITION_READ)

        stored = await self._drafts.get(principal.tenant_id, name)
        if stored is None:
            # The registry raises rather than returning None, so "does it
            # exist at all" has to be asked by catching. The distinction is
            # worth the catch: "shipped, so not editable here" and "no such
            # workflow" send a reader to completely different places.
            try:
                self._definitions.get(name, tenant_id=principal.tenant_id)
            except NotFoundError:
                raise NotFoundError("No such workflow definition", workflow=name) from None
            raise ValidationError(
                "That workflow was shipped in the repository, so there is no "
                "draft to edit. Copy it under a new name to change it here.",
                workflow=name,
            )

        found = parse_draft(stored)
        require_department(principal, found.department)
        return found

    async def delete(self, principal: Principal, name: str) -> None:
        """Withdraw a definition.

        Workflows already running keep executing: they hold their own pinned
        version, and removing the template does not retract work in flight.
        """
        require_permission(principal, Permission.DEFINITION_MANAGE)
        await self._repository.delete(principal.tenant_id, name)
        await self._drafts.delete(principal.tenant_id, name)
        self._definitions.forget_for_tenant(principal.tenant_id, name)
        await self._audit.record(
            tenant_id=principal.tenant_id,
            action=AuditAction.WORKFLOW_CANCELLED,
            actor=principal.subject,
            summary=f"Withdrew workflow definition {name}",
            metadata={"workflow": name},
        )

    async def restore(self, tenant_ids: StringList) -> int:
        """Re-register saved definitions into the registry at startup."""
        restored = 0
        for tenant_id in tenant_ids:
            for definition in await self._repository.list(tenant_id):
                self._definitions.register_for_tenant(tenant_id, definition)
                restored += 1
        return restored

    def _unmapped_facts(
        self, definition: WorkflowDefinition, tenant_id: str
    ) -> dict[str, StringList]:
        """Facts a ruleset reads that the draft does not supply.

        Not an error -- the ruleset default still applies, which is the safe
        direction -- but almost always a mistake worth surfacing while the
        author is still editing.
        """
        gaps: dict[str, list[str]] = {}
        for step in definition.steps:
            ruleset_name, supplied = _ruleset_and_facts(step)
            if not ruleset_name:
                continue
            ruleset = self._policy.rulesets_for(tenant_id).get(ruleset_name)
            if ruleset is None:
                continue
            required: set[str] = set()
            for rule in ruleset.rules:
                required.update(_facts_in(rule.when))
            missing = sorted(required - supplied)
            if missing:
                gaps[step.id] = missing
        return gaps

    def _is_write(self, tool: str) -> bool:
        """Whether a tool changes enterprise state, per its declared metadata."""
        try:
            return self._tools.metadata(tool).side_effect is not SideEffect.READ
        except NotFoundError:
            # Unknown tools are already reported as problems; do not also
            # guess at what they might do.
            return False

    # -- reference checking ------------------------------------------------
    def _reference_problems(
        self, definition: WorkflowDefinition, tenant_id: str
    ) -> StringList:
        """Every agent, tool and ruleset the definition names must exist.

        Rulesets are resolved through the tenant, so a workflow may name one
        this tenant authored -- otherwise the policy editor would produce
        rules the builder refuses to wire up.
        """
        known_agents = set(self._agents.names())
        known_tools = {t.name for t in self._tools.list_metadata()}
        known_rulesets = set(self._policy.rulesets_for(tenant_id))

        problems: list[str] = []
        for step in definition.steps:
            match step.type:
                case StepType.AGENT if step.agent not in known_agents:
                    problems.append(f"{step.id}: unknown agent '{step.agent}'")
                case StepType.TOOL if step.tool not in known_tools:
                    problems.append(f"{step.id}: unknown tool '{step.tool}'")
                case StepType.POLICY if step.ruleset not in known_rulesets:
                    problems.append(f"{step.id}: unknown ruleset '{step.ruleset}'")
                case _:
                    pass

            # Row Triage names its ruleset in tool arguments rather than in
            # `ruleset`, and a typo there fails at run time instead of here.
            triage = _triage_arguments(step)
            if triage and triage.get("ruleset") not in known_rulesets:
                problems.append(
                    f"{step.id}: unknown ruleset '{triage.get('ruleset')}'"
                )
        return problems


def _unbuildable(draft: WorkflowDraft, problems: StringList) -> dict[str, Any]:
    """The answer for a draft that cannot be compiled yet.

    Deliberately the same shape as a successful validation, with the graph
    analysis empty rather than absent, so the builder renders one response
    type instead of branching on whether the draft happened to compile.
    """
    return {
        "valid": False,
        "problems": problems,
        "unmapped_facts": {},
        "workflow": draft.name,
        "version": draft.version,
        "step_count": len(draft.steps),
        "waves": [],
        "critical_path": [],
        "uses_ai_steps": [
            s.id for s in draft.steps if BY_KIND[s.kind].uses_ai
        ],
        "warnings": [],
    }


def _triage_arguments(step: StepDefinition) -> dict[str, Any] | None:
    """The arguments of a Row Triage step, or None for anything else."""
    if step.type is not StepType.TOOL or step.tool != TRIAGE_TOOL:
        return None
    arguments = step.inputs.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _ruleset_and_facts(step: StepDefinition) -> tuple[str, set[str]]:
    """Which ruleset a step applies, and which of its facts are supplied.

    A Policy Gate and a Row Triage step both evaluate a ruleset, so both
    deserve the same warning about facts left unmapped -- they just keep that
    mapping in different places, and Row Triage keeps it in two.
    """
    triage = _triage_arguments(step)
    if triage is not None:
        supplied = set(triage.get("facts") or {}) | set(triage.get("row_facts") or {})
        return str(triage.get("ruleset") or ""), supplied
    if step.type is StepType.POLICY and step.ruleset:
        return step.ruleset, set(step.inputs.get("facts", {}))
    return "", set()


def _facts_in(expression: str) -> list[str]:
    """The `facts.x` names a rule reads, so the builder can show what to supply."""
    import re

    return re.findall(r"facts\.([a-zA-Z_][a-zA-Z0-9_]*)", expression)


def _advisory_warnings(
    definition: WorkflowDefinition, is_write: Callable[[str], bool]
) -> StringList:
    """Things worth saying but not worth refusing to save over."""
    warnings: StringList = []
    kinds = {step.type for step in definition.steps}

    # Whether a tool changes state is declared in its metadata. An earlier
    # version guessed from the name -- "get" meant read -- which called
    # `documents.analyse_dataset` a write and told a read-only workflow it
    # needed a policy gate.
    writes = [
        step.id
        for step in definition.steps
        if step.type is StepType.TOOL and step.tool and is_write(step.tool)
    ]

    if writes and StepType.POLICY not in kinds:
        warnings.append(
            "This workflow changes an enterprise system but has no Policy Gate. "
            "Consider adding one before the action."
        )
    if writes and StepType.HUMAN_APPROVAL not in kinds:
        warnings.append(
            "This workflow changes an enterprise system and cannot pause for a "
            "human. Consider adding an approval step."
        )

    agent_steps = [s for s in definition.steps if s.type is StepType.AGENT]
    if len(agent_steps) > 4:
        warnings.append(
            f"{len(agent_steps)} steps call a model. Each one costs latency and "
            "tokens -- consider whether any could be done deterministically."
        )
    return warnings

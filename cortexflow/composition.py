"""The composition root.

Every service process builds its dependency graph here and nowhere else.  This
is the only module that knows which concrete adapter backs which port, which
is what keeps the rest of the codebase free of ``if azure:`` branches.

Construction is async and explicit rather than magical: ``build_container``
creates adapters, loads and cross-validates definitions and policies, then
registers the tools and agents.  A misconfiguration fails here, at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortexflow.config.settings import (
    AgentRuntime,
    BlobBackend,
    BusBackend,
    CacheBackend,
    LlmBackend,
    Settings,
    StateBackend,
)
from cortexflow.infrastructure.blob.filesystem import FilesystemBlobStore
from cortexflow.infrastructure.memory.bus import InMemoryMessageBus
from cortexflow.infrastructure.memory.cache import InMemoryCache, InMemoryLockManager
from cortexflow.infrastructure.memory.repositories import (
    InMemoryApprovalRepository,
    InMemoryAuditRepository,
    InMemoryDeadLetterRepository,
    InMemoryDocumentRepository,
    InMemoryIdempotencyStore,
    InMemoryPolicyRulesetRepository,
    InMemoryWorkflowDefinitionRepository,
    InMemoryWorkflowDraftRepository,
    InMemoryWorkflowRepository,
)
from cortexflow.modules.agent.adapters.deterministic import DeterministicLlmClient
from cortexflow.modules.agent.adapters.kernel.factory import KernelFactory, ModelDialect
from cortexflow.modules.agent.adapters.semantic_kernel import SemanticKernelLlmClient
from cortexflow.modules.agent.adapters.sk_connectors import build_chat_service, model_name_for
from cortexflow.modules.agent.application.communication import CommunicationAgent
from cortexflow.modules.agent.application.decision import DecisionAgent
from cortexflow.modules.agent.application.extraction import ExtractionAgent
from cortexflow.modules.agent.application.registry import AgentRegistry
from cortexflow.modules.agent.application.reporting import ReportingAgent
from cortexflow.modules.agent.application.triage import TriageAgent
from cortexflow.modules.agent.application.validation import ValidationAgent
from cortexflow.modules.agent.ports.llm import LlmClient
from cortexflow.modules.approval.application.service import ApprovalService
from cortexflow.modules.approval.ports.repositories import ApprovalRepository
from cortexflow.modules.audit.application.service import AccessService
from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.ports.repositories import AuditRepository
from cortexflow.modules.document.application.service import DocumentService
from cortexflow.modules.document.ports.repositories import DocumentRepository
from cortexflow.modules.document.ports.storage import BlobStore
from cortexflow.modules.policy.adapters.loader import load_rulesets
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.policy.application.service import PolicyService
from cortexflow.modules.policy.ports.repositories import PolicyRulesetRepository
from cortexflow.modules.tool_registry.adapters.datasets import build_dataset_tools
from cortexflow.modules.tool_registry.adapters.finance import build_finance_tools
from cortexflow.modules.tool_registry.adapters.hr import build_hr_tools
from cortexflow.modules.tool_registry.adapters.marketing import build_marketing_tools
from cortexflow.modules.tool_registry.adapters.systems.fake_backends import (
    FinanceSystem,
    HrSystem,
    MarketingSystem,
    seed_demo_data,
)
from cortexflow.modules.tool_registry.adapters.triage import build_triage_tools
from cortexflow.modules.tool_registry.application.idempotency import IdempotentExecutor
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.modules.tool_registry.ports.idempotency import IdempotencyStore
from cortexflow.modules.workflow.application.builder.service import BuilderService
from cortexflow.modules.workflow.application.definitions import (
    WorkflowDefinitionRegistry,
    load_registry,
)
from cortexflow.modules.workflow.application.dispatcher import ControlPlaneDispatcher
from cortexflow.modules.workflow.application.engine import WorkflowEngine
from cortexflow.modules.workflow.application.executor import StepExecutor
from cortexflow.modules.workflow.application.retries import RetryPolicy
from cortexflow.modules.workflow.application.runner import AgentWorker
from cortexflow.modules.workflow.application.service import WorkflowService
from cortexflow.modules.workflow.application.sweeper import RecoverySweeper
from cortexflow.modules.workflow.domain.topics import WORKER_QUEUES
from cortexflow.modules.workflow.ports.messaging import MessageBus
from cortexflow.modules.workflow.ports.repositories import (
    DeadLetterRepository,
    WorkflowDefinitionRepository,
    WorkflowDraftRepository,
    WorkflowRepository,
)
from cortexflow.shared.clock import Clock, SystemClock
from cortexflow.shared.ids import IdGenerator, UuidIdGenerator
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.ports.cache import Cache, LockManager
from cortexflow.shared.security.auth import build_authenticator

logger = get_logger(__name__)


@dataclass
class Container:
    """The wired application graph."""

    settings: Settings
    clock: Clock
    ids: IdGenerator

    workflows: WorkflowRepository
    approvals: ApprovalRepository
    audit_repository: AuditRepository
    idempotency: IdempotencyStore
    dead_letters: DeadLetterRepository
    documents: DocumentRepository
    definition_repository: WorkflowDefinitionRepository
    draft_repository: WorkflowDraftRepository
    ruleset_repository: PolicyRulesetRepository
    bus: MessageBus
    cache: Cache
    locks: LockManager
    blobs: BlobStore
    llm: LlmClient

    audit: AuditTrail
    policy: PolicyEngine
    definitions: WorkflowDefinitionRegistry
    tools: ToolRegistry
    agents: AgentRegistry
    kernel_factory: KernelFactory | None
    """Present only when the Semantic Kernel runtime is configured."""

    engine: WorkflowEngine
    workflow_service: WorkflowService
    approval_service: ApprovalService
    document_service: DocumentService
    builder_service: BuilderService
    policy_service: PolicyService
    access_service: AccessService
    authenticator: Any

    hr_system: HrSystem
    finance_system: FinanceSystem
    marketing_system: MarketingSystem

    _closers: list[Any] = field(default_factory=list)

    # -- runtime components -----------------------------------------------
    def build_dispatcher(self) -> ControlPlaneDispatcher:
        return ControlPlaneDispatcher(
            engine=self.engine,
            bus=self.bus,
            cache=self.cache,
            audit=self.audit,
            dead_letters=self.dead_letters,
            dedup_ttl_seconds=self.settings.reliability.dedup_ttl_seconds,
        )

    def build_sweeper(self, *, interval_seconds: float = 10.0) -> RecoverySweeper:
        return RecoverySweeper(
            engine=self.engine,
            workflows=self.workflows,
            approvals=self.approvals,
            locks=self.locks,
            audit=self.audit,
            clock=self.clock,
            interval_seconds=interval_seconds,
        )

    def build_executor(self) -> StepExecutor:
        return StepExecutor(
            workflows=self.workflows,
            definitions=self.definitions,
            agents=self.agents,
            tools=self.tools,
            bus=self.bus,
            clock=self.clock,
            ids=self.ids,
        )

    def build_worker(self, queues: tuple[Any, ...] = ()) -> AgentWorker:
        return AgentWorker(
            executor=self.build_executor(),
            bus=self.bus,
            cache=self.cache,
            dead_letters=self.dead_letters,
            queues=queues or WORKER_QUEUES,
            concurrency=self.settings.worker_concurrency,
            dedup_ttl_seconds=self.settings.reliability.dedup_ttl_seconds,
        )

    async def close(self) -> None:
        for closer in reversed(self._closers):
            try:
                await closer()
            except Exception:  # pragma: no cover - shutdown must not raise
                logger.exception("error while closing a resource")


async def build_container(
    settings: Settings,
    *,
    clock: Clock | None = None,
    ids: IdGenerator | None = None,
    seed_demo: bool | None = None,
    chat_service: Any | None = None,
) -> Container:
    """Construct the full application graph for ``settings``.

    ``chat_service`` overrides the Semantic Kernel connector that would
    otherwise be built from configuration. Tests supply a scripted service so
    the Kernel runtime can be exercised without a model server; production
    leaves it unset.
    """
    clock = clock or SystemClock()
    ids = ids or UuidIdGenerator()
    closers: list[Any] = []

    workflows, approvals, audit_repo, idempotency, dead_letters = await _build_state(
        settings, closers, ids
    )
    (
        documents_repo,
        definition_repo,
        draft_repo,
        ruleset_repo,
    ) = _build_authoring_state(settings)
    bus = await _build_bus(settings, closers)
    cache, locks = await _build_cache(settings, closers)
    blobs = _build_blobs(settings)
    llm = _build_llm(settings)

    audit = AuditTrail(audit_repo, ids)
    policy = PolicyEngine(load_rulesets(settings.policy_rulesets_path))
    definitions = load_registry(
        settings.workflow_definitions_path, runtime=str(settings.agent_runtime)
    )

    hr_system, finance_system, marketing_system = (
        seed_demo_data()
        if (seed_demo if seed_demo is not None else settings.profile.value == "local")
        else (HrSystem(), FinanceSystem(), MarketingSystem())
    )

    tools = ToolRegistry(
        audit=audit_repo,
        idempotency=IdempotentExecutor(
            idempotency, ttl_seconds=settings.reliability.idempotency_ttl_seconds
        ),
        cache=cache,
        ids=ids,
    )
    tools.register_all(
        [
            *build_hr_tools(hr_system),
            *build_finance_tools(finance_system),
            *build_marketing_tools(marketing_system),
            # Dataset profiling is deterministic, so it is a tool rather than
            # something an agent is asked to compute.
            *build_dataset_tools(documents_repo, blobs),
            # Row-by-row triage needs the ruleset and the original file, which
            # is why it lives here rather than inside the orchestrator.
            *build_triage_tools(documents_repo, blobs, policy),
        ]
    )

    agents = AgentRegistry()
    agents.register_all(
        [
            ExtractionAgent(llm, tools, blobs),
            ValidationAgent(llm, tools),
            DecisionAgent(llm, tools),
            ReportingAgent(llm, tools),
            CommunicationAgent(llm, tools),
        ]
    )

    # The Semantic Kernel runtime is additive: it registers agents that need
    # multi-turn tool calling alongside the native ones, rather than replacing
    # them. A workflow step picks a runtime by naming an agent.
    kernel_factory: KernelFactory | None = None
    if settings.agent_runtime is AgentRuntime.SEMANTIC_KERNEL:
        kernel_factory = KernelFactory(
            service=chat_service or build_chat_service(settings),
            registry=tools,
            dialect=_model_dialect(settings),
        )
        agents.register(
            TriageAgent(
                kernel_factory,
                tools,
                max_function_calls=settings.kernel_max_function_calls,
            )
        )
        logger.info(
            "semantic kernel runtime enabled",
            extra={
                "model": model_name_for(settings),
                "max_function_calls": settings.kernel_max_function_calls,
            },
        )

    # Fail fast: a definition referencing an agent, tool or ruleset that does
    # not exist is a deployment error, not a runtime surprise.
    definitions.validate_against(
        agent_names=agents.names(),
        tool_names=tuple(t.name for t in tools.list_metadata()),
        ruleset_names=tuple(policy.rulesets),
    )

    engine = WorkflowEngine(
        workflows=workflows,
        approvals=approvals,
        definitions=definitions,
        bus=bus,
        locks=locks,
        policy=policy,
        audit=audit,
        retry=RetryPolicy(settings.reliability),
        clock=clock,
        ids=ids,
        settings=settings.reliability,
    )
    workflow_service = WorkflowService(
        workflows=workflows,
        definitions=definitions,
        bus=bus,
        audit=audit,
        clock=clock,
        ids=ids,
    )
    access_service = AccessService(audit=audit_repo, clock=clock)
    approval_service = ApprovalService(
        approvals=approvals, engine=engine, clock=clock
    )
    document_service = DocumentService(
        documents=documents_repo, blobs=blobs, audit=audit, ids=ids
    )
    builder_service = BuilderService(
        definitions=definitions,
        repository=definition_repo,
        drafts=draft_repo,
        tools=tools,
        policy=policy,
        agents=agents,
        audit=audit,
    )
    # Definitions authored in previous runs are registered again at startup,
    # so a restart does not lose a workflow someone built.
    restored = await builder_service.restore([settings.default_tenant_id])
    if restored:
        logger.info("restored builder definitions", extra={"count": restored})

    policy_service = PolicyService(
        engine=policy, repository=ruleset_repo, audit=audit
    )
    # Authored rulesets are registered again at startup for the same reason
    # definitions are -- and for a sharper one: the engine keeps overrides in
    # memory because `evaluate` is synchronous and on the hot path, so without
    # this a restart would silently put the shipped rules back in force.
    re_registered = await policy_service.restore([settings.default_tenant_id])
    if re_registered:
        logger.info("restored policy rulesets", extra={"count": re_registered})
    authenticator = build_authenticator(
        settings.auth,
        default_tenant=settings.default_tenant_id,
        environment=settings.environment,
    )

    logger.info(
        "container built",
        extra={
            "profile": str(settings.profile),
            "state": str(settings.state_backend),
            "bus": str(settings.bus_backend),
            "cache": str(settings.cache_backend),
            "llm": str(settings.llm_backend),
            "workflows": list(definitions.names()),
            "tools": len(tools.list_metadata()),
            "agents": list(agents.names()),
        },
    )

    return Container(
        settings=settings,
        clock=clock,
        ids=ids,
        workflows=workflows,
        approvals=approvals,
        audit_repository=audit_repo,
        idempotency=idempotency,
        dead_letters=dead_letters,
        documents=documents_repo,
        definition_repository=definition_repo,
        draft_repository=draft_repo,
        ruleset_repository=ruleset_repo,
        bus=bus,
        cache=cache,
        locks=locks,
        blobs=blobs,
        llm=llm,
        audit=audit,
        policy=policy,
        definitions=definitions,
        tools=tools,
        agents=agents,
        kernel_factory=kernel_factory,
        engine=engine,
        workflow_service=workflow_service,
        approval_service=approval_service,
        document_service=document_service,
        builder_service=builder_service,
        policy_service=policy_service,
        access_service=access_service,
        authenticator=authenticator,
        hr_system=hr_system,
        finance_system=finance_system,
        marketing_system=marketing_system,
        _closers=closers,
    )


# ----------------------------------------------------------------------
# Adapter selection
# ----------------------------------------------------------------------
async def _build_state(
    settings: Settings, closers: list[Any], ids: IdGenerator
) -> tuple[
    WorkflowRepository,
    ApprovalRepository,
    AuditRepository,
    IdempotencyStore,
    DeadLetterRepository,
]:
    if settings.state_backend is StateBackend.COSMOS:
        from cortexflow.infrastructure.cosmos.client import build_client
        from cortexflow.infrastructure.cosmos.repositories import (
            CosmosApprovalRepository,
            CosmosAuditRepository,
            CosmosDeadLetterRepository,
            CosmosIdempotencyStore,
            CosmosWorkflowRepository,
        )

        client = build_client(settings.cosmos)
        closers.append(client.close)
        return (
            CosmosWorkflowRepository(client, settings.cosmos),
            CosmosApprovalRepository(client, settings.cosmos),
            CosmosAuditRepository(client, settings.cosmos),
            CosmosIdempotencyStore(client, settings.cosmos),
            CosmosDeadLetterRepository(client, settings.cosmos, ids),
        )

    return (
        InMemoryWorkflowRepository(),
        InMemoryApprovalRepository(),
        InMemoryAuditRepository(),
        InMemoryIdempotencyStore(),
        InMemoryDeadLetterRepository(ids),
    )


def _build_authoring_state(
    settings: Settings,
) -> tuple[
    DocumentRepository,
    WorkflowDefinitionRepository,
    WorkflowDraftRepository,
    PolicyRulesetRepository,
]:
    """Storage for everything people author at runtime.

    Uploaded documents, workflow definitions built in the browser, the drafts
    those were compiled from, and policy rulesets edited by an administrator.
    On the Azure profile all of them belong in Cosmos alongside workflow
    state; the in-memory implementations keep the
    same contract for local runs.
    """
    if settings.state_backend is StateBackend.COSMOS:
        from cortexflow.infrastructure.cosmos.authoring import (
            CosmosDocumentRepository,
            CosmosPolicyRulesetRepository,
            CosmosWorkflowDefinitionRepository,
            CosmosWorkflowDraftRepository,
        )
        from cortexflow.infrastructure.cosmos.client import build_client

        client = build_client(settings.cosmos)
        return (
            CosmosDocumentRepository(client, settings.cosmos),
            CosmosWorkflowDefinitionRepository(client, settings.cosmos),
            CosmosWorkflowDraftRepository(client, settings.cosmos),
            CosmosPolicyRulesetRepository(client, settings.cosmos),
        )
    return (
        InMemoryDocumentRepository(),
        InMemoryWorkflowDefinitionRepository(),
        InMemoryWorkflowDraftRepository(),
        InMemoryPolicyRulesetRepository(),
    )


async def _build_bus(settings: Settings, closers: list[Any]) -> MessageBus:
    if settings.bus_backend is BusBackend.SERVICE_BUS:
        from cortexflow.infrastructure.service_bus.bus import ServiceBusMessageBus

        bus: MessageBus = ServiceBusMessageBus(settings.servicebus)
    else:
        bus = InMemoryMessageBus(
            lock_seconds=settings.reliability.task_visibility_seconds,
        )
    await bus.start()
    closers.append(bus.close)
    return bus


async def _build_cache(
    settings: Settings, closers: list[Any]
) -> tuple[Cache, LockManager]:
    if settings.cache_backend is CacheBackend.REDIS:
        from cortexflow.infrastructure.redis.cache import (
            RedisCache,
            RedisLockManager,
            build_redis,
        )

        client = build_redis(settings.redis)
        closers.append(client.aclose)
        return RedisCache(client, settings.redis), RedisLockManager(client, settings.redis)

    return InMemoryCache(), InMemoryLockManager()


def _build_blobs(settings: Settings) -> BlobStore:
    if settings.blob_backend is BlobBackend.AZURE_BLOB:
        from cortexflow.infrastructure.blob.azure_blob import AzureBlobStore, build_blob_client

        return AzureBlobStore(build_blob_client(settings.blob), settings.blob)
    return FilesystemBlobStore(settings.blob.local_root, settings.blob.container)


def _model_dialect(settings: Settings) -> ModelDialect:
    """What the configured deployment accepts, for the kernel runtime.

    Only Azure OpenAI has reasoning deployments to worry about. A locally
    hosted model reached through Ollama takes the ordinary parameters, and the
    deterministic stub takes whatever it is given.
    """
    if settings.llm_backend is not LlmBackend.AZURE_OPENAI:
        return ModelDialect()
    return ModelDialect(
        reasoning=settings.openai.is_reasoning_model,
        reasoning_effort=settings.openai.reasoning_effort,
    )


def _build_llm(settings: Settings) -> LlmClient:
    match settings.llm_backend:
        case LlmBackend.AZURE_OPENAI:
            from cortexflow.infrastructure.azure_openai.client import AzureOpenAIClient

            return AzureOpenAIClient(settings.openai)
        case LlmBackend.OLLAMA:
            # Reached through Semantic Kernel's connector, so every existing
            # agent runs against a local model with no agent-code change.
            return SemanticKernelLlmClient(
                build_chat_service(settings), model_name=model_name_for(settings)
            )
        case _:
            return DeterministicLlmClient()

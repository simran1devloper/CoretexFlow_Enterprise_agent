"""Typed application configuration.

Every service in the platform reads the same ``Settings`` object.  Which
concrete adapter is wired for each port is a *configuration* decision, not a
code decision -- that is what keeps the domain free of Azure imports and what
makes the whole platform runnable on a laptop with ``CORTEXFLOW_PROFILE=local``.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Profile(StrEnum):
    """Deployment profile, selecting the default adapter set."""

    LOCAL = "local"
    AZURE = "azure"


class StateBackend(StrEnum):
    MEMORY = "memory"
    COSMOS = "cosmos"


class BusBackend(StrEnum):
    MEMORY = "memory"
    SERVICE_BUS = "servicebus"


class CacheBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"


class BlobBackend(StrEnum):
    FILESYSTEM = "filesystem"
    AZURE_BLOB = "azure_blob"


class LlmBackend(StrEnum):
    """Which model provider answers agent prompts."""

    DETERMINISTIC = "deterministic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    """A locally-hosted model, reached through Semantic Kernel's connector."""


class AgentRuntime(StrEnum):
    """How an agent step is executed.

    This is deliberately separate from :class:`LlmBackend`. The platform's
    position (see docs/adr/0007) is that Semantic Kernel is an *agent runtime*
    -- prompts, plugins, function calling, structured output -- and not the
    enterprise workflow engine. Which runtime executes a step and which model
    answers it are independent choices.
    """

    NATIVE = "native"
    """Single-turn structured completion through the ``LlmClient`` port."""

    SEMANTIC_KERNEL = "semantic_kernel"
    """Multi-turn Kernel loop with automatic function calling over registry tools."""


class AuthMode(StrEnum):
    """How incoming API calls are authenticated."""

    DEV = "dev"
    """Signed-nothing development principals via the ``X-Dev-Principal`` header."""

    ENTRA = "entra"
    """Microsoft Entra ID JWT bearer tokens."""


def _config_dir() -> Path:
    """Where the profile files live.

    Beside the source tree when running from a checkout. Installed into a
    container the package sits in site-packages, so the path is given
    explicitly -- the same treatment ``workflows`` and ``rulesets`` already
    get, for the same reason.
    """
    override = os.getenv("CORTEXFLOW_CONFIG_DIR")
    if override:
        return Path(override)
    beside_source = REPO_ROOT / "config"
    if beside_source.is_dir():
        return beside_source
    # An installed package with no explicit path: try the working directory,
    # which is where a container's config is copied.
    return Path.cwd() / "config"


CONFIG_DIR = _config_dir()


def _profile_env_files() -> tuple[Path, ...]:
    """Which files configure this process, in increasing precedence.

    ``config/<profile>.env`` is the deployment's own baseline, committed and
    reviewable. ``.env`` is the developer's private overrides and is not.
    Real environment variables beat both, which is what lets a container get
    its configuration from the platform without a file at all.

    The profile is read from the environment directly rather than from the
    settings being constructed, because the answer decides which file the
    construction reads -- a chicken-and-egg that is cleaner to cut here than
    to solve with a second pass.
    """
    profile = os.getenv("CORTEXFLOW_PROFILE", Profile.LOCAL.value).strip().lower()
    if profile not in set(Profile):
        profile = Profile.LOCAL.value

    files = [CONFIG_DIR / f"{profile}.env"]
    # `.env` is one developer's machine. Anything that needs to answer "what
    # does this committed profile actually configure?" -- CI, a container
    # image built from a checkout, the profile tests -- has to be able to
    # exclude it, or the answer depends on whose laptop asked.
    if os.getenv("CORTEXFLOW_IGNORE_DOTENV", "").strip().lower() not in {"1", "true", "yes"}:
        files.append(REPO_ROOT / ".env")
    return tuple(files)


class _ProfileSettings(BaseSettings):
    """Base for every settings group, so they all read the same files.

    Each group is its own ``BaseSettings`` with its own prefix, which means
    each also needs telling where to read from -- otherwise the root object
    honours ``config/azure.env`` and every nested group silently ignores it,
    leaving a Cosmos endpoint blank in a profile that names one.
    """

    model_config = SettingsConfigDict(
        env_file=_profile_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class CosmosSettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_COSMOS_")

    endpoint: str = ""
    key: str = ""
    database: str = "cortexflow"
    workflow_container: str = "workflows"
    approval_container: str = "approvals"
    audit_container: str = "audit"
    idempotency_container: str = "idempotency"
    use_managed_identity: bool = True


class ServiceBusSettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_SERVICEBUS_")

    namespace: str = ""
    connection_string: str = ""
    queue_prefix: str = "cortexflow"
    max_wait_seconds: int = 20
    prefetch: int = 10
    use_managed_identity: bool = True


class RedisSettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_REDIS_")

    url: str = "redis://localhost:6379/0"
    key_prefix: str = "cortexflow"


class BlobSettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_BLOB_")

    account_url: str = ""
    container: str = "documents"
    local_root: Path = REPO_ROOT / ".data" / "blobs"
    use_managed_identity: bool = True


class OllamaSettings(_ProfileSettings):
    """A locally-hosted model, for development against a real LLM."""

    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_OLLAMA_")

    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    """Smaller models do not emit structured tool calls reliably.

    Measured: llama3.2:1b returns tool calls as plain text and fabricates
    values it should have looked up. 8b is the smallest that behaves.
    """

    temperature: float = 0.1
    """Low by default: the platform wants reproducible structure, not prose."""

    timeout_seconds: float = 120.0
    num_ctx: int = 8192


class AzureOpenAISettings(_ProfileSettings):
    """Where the model lives, and which dialect it speaks.

    Two of these are derived rather than configured, because both are
    properties of the endpoint and the deployment that somebody would
    otherwise have to know to set correctly by hand -- and would get wrong
    once, quietly, in production.
    """

    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_OPENAI_")

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-06-01"
    chat_deployment: str = "gpt-4o"
    """The *deployment* name, which need not match the model name."""

    use_managed_identity: bool = True
    timeout_seconds: float = 60.0
    max_output_tokens: int = 2048
    temperature: float = 0.0

    entra_scope: str = "https://cognitiveservices.azure.com/.default"
    """Token audience when authenticating as a managed identity.

    Configurable because it is not the same everywhere: an Azure OpenAI
    resource takes the Cognitive Services audience, while some AI Foundry
    project endpoints are documented with ``https://ai.azure.com/.default``.
    Getting it wrong yields a 401 that says nothing about audiences.
    """

    reasoning: bool | None = None
    """Whether this deployment is a reasoning model. None infers from the name.

    It matters because reasoning deployments reject the two parameters every
    other deployment requires: ``max_tokens`` (they want
    ``max_completion_tokens``) and any ``temperature`` other than the default.
    Set it explicitly when the inference is wrong for your deployment name.
    """

    reasoning_effort: str = ""
    """``minimal`` | ``low`` | ``medium`` | ``high``; empty leaves the default.

    Only sent to reasoning deployments, where it trades latency and tokens
    against how long the model thinks before answering.
    """

    @property
    def uses_v1_api(self) -> bool:
        """Whether the endpoint is the OpenAI-compatible ``/openai/v1`` surface.

        AI Foundry projects expose the model twice. The classic surface takes
        ``azure_endpoint`` plus an ``api-version`` query parameter and builds
        ``/openai/deployments/{deployment}/...``; the v1 surface is
        OpenAI-compatible, takes a plain ``base_url`` and no api-version, and
        is what the portal hands you today.

        Inferred from the URL rather than configured separately: the path is
        unambiguous, and a second setting to keep in step with it is a second
        thing to get wrong.
        """
        return self.endpoint.rstrip("/").endswith("/openai/v1")

    @property
    def base_url(self) -> str:
        """The v1 base URL, normalised without a trailing slash."""
        return self.endpoint.rstrip("/")

    @property
    def is_reasoning_model(self) -> bool:
        if self.reasoning is not None:
            return self.reasoning
        name = self.chat_deployment.lower().lstrip("-_")
        return name.startswith(("o1", "o3", "o4", "gpt-5"))


class AuthSettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_AUTH_")

    mode: AuthMode = AuthMode.DEV
    tenant_id: str = ""
    audience: str = ""
    issuer: str = ""
    jwks_url: str = ""
    jwks_cache_seconds: int = 3600


class ReliabilitySettings(_ProfileSettings):
    """Knobs for the deterministic reliability machinery."""

    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_RELIABILITY_")

    max_attempts: int = 4
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    backoff_jitter: float = 0.2

    workflow_lease_seconds: int = 30
    task_visibility_seconds: int = 300
    dedup_ttl_seconds: int = 86_400
    idempotency_ttl_seconds: int = 604_800

    hot_state_cache_seconds: int = 30
    approval_timeout_seconds: int = 259_200  # 3 days


class ObservabilitySettings(_ProfileSettings):
    model_config = SettingsConfigDict(env_prefix="CORTEXFLOW_OTEL_")

    service_name: str = "cortexflow"
    exporter_endpoint: str = ""
    console_traces: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    redact_payloads: bool = True


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="CORTEXFLOW_",
        # Later files win, and real environment variables win over all of them.
        env_file=_profile_env_files(),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    profile: Profile = Profile.LOCAL
    environment: str = "dev"
    default_tenant_id: str = "ACME"

    workflow_definitions_path: Path = REPO_ROOT / "workflows"
    policy_rulesets_path: Path = (
        Path(__file__).resolve().parent.parent / "modules" / "policy" / "rulesets"
    )

    state_backend: StateBackend | None = None
    bus_backend: BusBackend | None = None
    cache_backend: CacheBackend | None = None
    blob_backend: BlobBackend | None = None
    llm_backend: LlmBackend | None = None

    agent_runtime: AgentRuntime = AgentRuntime.NATIVE
    """Native by default so the platform stays runnable with no model server."""

    kernel_max_function_calls: int = 5
    """Ceiling on automatic tool calls per agent step.

    An agent loop that can call tools without bound is a cost and latency
    incident waiting to happen; the orchestrator's step timeout is the outer
    guard, and this is the inner one.
    """

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    worker_concurrency: int = 8
    worker_queues: list[str] = Field(default_factory=list)
    """Queues an agent-worker instance consumes; empty means "all agent queues"."""

    cosmos: CosmosSettings = Field(default_factory=CosmosSettings)
    servicebus: ServiceBusSettings = Field(default_factory=ServiceBusSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    blob: BlobSettings = Field(default_factory=BlobSettings)
    openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> Settings:
        """Fill unset backends from the profile so deployments stay one-liners."""
        azure = self.profile is Profile.AZURE
        if self.state_backend is None:
            self.state_backend = StateBackend.COSMOS if azure else StateBackend.MEMORY
        if self.bus_backend is None:
            self.bus_backend = BusBackend.SERVICE_BUS if azure else BusBackend.MEMORY
        if self.cache_backend is None:
            self.cache_backend = CacheBackend.REDIS if azure else CacheBackend.MEMORY
        if self.blob_backend is None:
            self.blob_backend = BlobBackend.AZURE_BLOB if azure else BlobBackend.FILESYSTEM
        if self.llm_backend is None:
            self.llm_backend = LlmBackend.AZURE_OPENAI if azure else LlmBackend.DETERMINISTIC
        return self

    @property
    def is_distributed(self) -> bool:
        """True when services communicate through real infrastructure."""
        return self.bus_backend is not BusBackend.MEMORY


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()

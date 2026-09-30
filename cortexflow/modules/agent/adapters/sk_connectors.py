"""Semantic Kernel chat-completion connectors.

Semantic Kernel is the *agent runtime*: it owns LLM interaction, plugin
invocation, prompt management and structured output. It is deliberately not
the workflow engine -- CortexFlow keeps workflow state, dependencies,
security, retries, durability and human approval.

    Enterprise workflow
            │
    CortexFlow orchestrator     ← state · policy · retry · approval · audit
            │
      Semantic Kernel           ← prompts · plugins · function calling
            │
    Azure OpenAI / Ollama

This module lives in the adapter layer because it is the only place that
touches the Semantic Kernel SDK's connector classes. The agent runtime that
uses it sits in ``agents/kernel/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cortexflow.config.settings import AzureOpenAISettings, LlmBackend, OllamaSettings, Settings
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.observability.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semantic_kernel.connectors.ai.chat_completion_client_base import (
        ChatCompletionClientBase,
    )

logger = get_logger(__name__)

SERVICE_ID = "cortexflow-chat"
"""One service id everywhere, so execution settings always bind to the right service."""


def build_ollama_service(settings: OllamaSettings) -> ChatCompletionClientBase:
    """A locally-hosted model, for development against a real LLM."""
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion

    logger.info(
        "semantic kernel: using the Ollama connector",
        extra={"model": settings.model, "host": settings.host},
    )
    return OllamaChatCompletion(
        ai_model_id=settings.model,
        host=settings.host,
        service_id=SERVICE_ID,
    )


def build_azure_openai_service(
    settings: AzureOpenAISettings,
) -> ChatCompletionClientBase:
    """Azure OpenAI, on whichever surface the endpoint names.

    The connector is handed a pre-built SDK client rather than an endpoint and
    a key. That is the only way to reach the ``/openai/v1`` surface, whose URL
    the connector would otherwise rewrite into a classic deployment path.

    **Which connector follows from which client.** The v1 surface is the
    OpenAI-compatible API, so it gets an ``AsyncOpenAI`` and therefore
    ``OpenAIChatCompletion``; ``AzureChatCompletion`` validates that its client
    is specifically an ``AsyncAzureOpenAI`` and refuses anything else at
    construction. Both connectors implement the same interface, so nothing
    downstream -- function calling, execution settings, usage accounting --
    can tell them apart.

    This mirrors :func:`cortexflow.infrastructure.azure_openai.client.build_async_client`,
    which it cannot import: a module may not name infrastructure. Neither
    decides anything -- the URL form, the audience and whether the deployment
    reasons all come off the settings object -- and a test asserts the two
    resolve a given settings object identically.
    """
    logger.info(
        "semantic kernel: using the Azure OpenAI connector",
        extra={
            "deployment": settings.chat_deployment,
            "api": "v1" if settings.uses_v1_api else "azure",
            "auth": "key" if settings.api_key else "entra",
            "reasoning": settings.is_reasoning_model,
        },
    )
    client = build_openai_async_client(settings)

    if settings.uses_v1_api and not _is_azure_client(client):
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

        return OpenAIChatCompletion(
            service_id=SERVICE_ID,
            ai_model_id=settings.chat_deployment,
            async_client=client,
        )

    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    return AzureChatCompletion(
        service_id=SERVICE_ID,
        deployment_name=settings.chat_deployment,
        async_client=client,
    )


def _is_azure_client(client: Any) -> bool:
    """Whether this is an ``AsyncAzureOpenAI``, which is a subclass of the other.

    ``isinstance`` and not a name check, because the Entra path builds an
    ``AsyncAzureOpenAI`` even for a v1 URL -- a plain client cannot refresh a
    token -- and that one has to go back to ``AzureChatCompletion``.
    """
    from openai import AsyncAzureOpenAI

    return isinstance(client, AsyncAzureOpenAI)


def build_openai_async_client(settings: AzureOpenAISettings) -> Any:
    """The SDK client for this endpoint. See :func:`build_azure_openai_service`."""
    key = settings.api_key
    use_entra = settings.use_managed_identity and not key

    if not use_entra:
        if settings.uses_v1_api:
            from openai import AsyncOpenAI

            return AsyncOpenAI(
                base_url=settings.base_url,
                api_key=key,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=key,
            api_version=settings.api_version,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI

    common: dict[str, Any] = {
        "azure_ad_token_provider": get_bearer_token_provider(
            DefaultAzureCredential(), settings.entra_scope
        ),
        "api_version": settings.api_version,
        "timeout": settings.timeout_seconds,
        "max_retries": 0,
    }
    if settings.uses_v1_api:
        return AsyncAzureOpenAI(base_url=settings.base_url, **common)
    return AsyncAzureOpenAI(azure_endpoint=settings.endpoint, **common)


def build_chat_service(settings: Settings) -> ChatCompletionClientBase:
    """Build the connector for the configured model provider."""
    match settings.llm_backend:
        case LlmBackend.OLLAMA:
            return build_ollama_service(settings.ollama)
        case LlmBackend.AZURE_OPENAI:
            return build_azure_openai_service(settings.openai)
        case _:
            raise ValidationError(
                "The Semantic Kernel runtime needs a real model provider. "
                "Set CORTEXFLOW_LLM_BACKEND to 'ollama' or 'azure_openai'.",
                llm_backend=str(settings.llm_backend),
            )


def model_name_for(settings: Settings) -> str:
    match settings.llm_backend:
        case LlmBackend.OLLAMA:
            return settings.ollama.model
        case LlmBackend.AZURE_OPENAI:
            return settings.openai.chat_deployment
        case _:
            return str(settings.llm_backend)


def is_ollama(service: Any) -> bool:
    """Connector-specific settings differ; this keeps the check in one place."""
    return type(service).__name__.startswith("Ollama")


# ---------------------------------------------------------------------------
# Response shapes differ between connectors; these keep that knowledge here.
# ---------------------------------------------------------------------------
def _as_int(value: object | None) -> int:
    """Coerce a connector's token count, which arrives loosely typed."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def usage_from_response(response: object, *, model: str) -> Any:
    """Read token usage from whichever shape the connector reports.

    Azure OpenAI reports ``usage`` on the raw response; Ollama reports
    ``prompt_eval_count`` / ``eval_count``. Cost per workflow has to be
    answerable either way.
    """
    from cortexflow.modules.agent.domain.models import TokenUsage

    inner = getattr(response, "inner_content", None)
    metadata = getattr(response, "metadata", {}) or {}

    usage = getattr(inner, "usage", None) or metadata.get("usage")
    if usage is not None:
        return TokenUsage(
            prompt_tokens=_as_int(_read(usage, "prompt_tokens")),
            completion_tokens=_as_int(_read(usage, "completion_tokens")),
            model=model,
        )

    prompt_eval = _read(inner, "prompt_eval_count") or metadata.get("prompt_eval_count")
    eval_count = _read(inner, "eval_count") or metadata.get("eval_count")
    if prompt_eval or eval_count:
        return TokenUsage(
            prompt_tokens=_as_int(prompt_eval),
            completion_tokens=_as_int(eval_count),
            model=model,
        )

    return TokenUsage(model=model)


def _read(source: object, name: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def translate_kernel_error(exc: Exception) -> Exception:
    """Map connector failures onto the platform's retry taxonomy.

    Without this, a model server being briefly unreachable would classify as
    UNKNOWN and burn the smaller retry budget meant for genuinely ambiguous
    failures.
    """
    from cortexflow.shared.errors import (
        DependencyTimeoutError,
        DependencyUnavailableError,
        RateLimitedError,
        UnknownError,
    )

    name = type(exc).__name__
    text = str(exc).lower()

    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return DependencyTimeoutError("The model service timed out", error=name)
    if isinstance(exc, ConnectionError) or any(
        marker in text for marker in ("connection", "unreachable", "refused", "503")
    ):
        return DependencyUnavailableError("The model service is unavailable", error=name)
    if "rate" in text and "limit" in text:
        return RateLimitedError("The model service throttled the request")
    return UnknownError(f"{name}: {exc}"[:400])

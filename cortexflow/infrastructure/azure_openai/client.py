"""Azure OpenAI client.

Wraps the SDK with three things the platform needs and the SDK does not give
us: error translation into the retry taxonomy, JSON-schema-constrained output,
and token accounting emitted per call.

:func:`build_async_client` is deliberately a module-level function rather than
a method. Semantic Kernel's connector needs the same client built the same
way, and it lives in ``modules/agent/adapters`` where it cannot import
infrastructure -- so it constructs its own. What keeps the two from drifting
is that neither decides anything: which URL form, which audience and whether
the deployment reasons are all answered by
:class:`~cortexflow.config.settings.AzureOpenAISettings`, and a test asserts
both builders resolve a given settings object identically.
"""

from __future__ import annotations

import time
from typing import Any

from cortexflow.config.settings import AzureOpenAISettings
from cortexflow.modules.agent.domain.models import TokenUsage
from cortexflow.modules.agent.ports.llm import CompletionRequest, CompletionResponse
from cortexflow.shared.errors import (
    DependencyTimeoutError,
    DependencyUnavailableError,
    PermanentError,
    RateLimitedError,
    ValidationError,
)
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)


class AzureOpenAIClient:
    def __init__(self, settings: AzureOpenAISettings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or self._build(settings)
        self._metrics = get_metrics()

    @staticmethod
    def _build(settings: AzureOpenAISettings) -> Any:
        return build_async_client(settings)

    @property
    def model_name(self) -> str:
        return self._settings.chat_deployment

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        with span(
            "llm.complete",
            **{
                "llm.model": self.model_name,
                "llm.schema": request.schema_name,
                "llm.prompt_version": request.prompt_version,
            },
        ) as current:
            try:
                response_format = self._response_format(request)
                raw = await self._client.chat.completions.create(
                    model=self._settings.chat_deployment,
                    messages=[
                        {"role": m.role, "content": m.content} for m in request.messages
                    ],
                    **({"response_format": response_format} if response_format else {}),
                    **self._decoding(request),
                )
            except Exception as exc:
                raise _translate(exc) from exc

            elapsed_ms = (time.perf_counter() - started) * 1000
            choice = raw.choices[0]
            usage = TokenUsage(
                prompt_tokens=getattr(raw.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw.usage, "completion_tokens", 0) or 0,
                model=self.model_name,
            )
            current.set_attribute("llm.tokens.total", usage.total_tokens)
            self._metrics.record_duration(
                self._metrics.llm_duration, elapsed_ms, model=self.model_name
            )
            self._metrics.count(
                self._metrics.llm_tokens, usage.total_tokens, model=self.model_name
            )
            return CompletionResponse(
                content=choice.message.content or "",
                usage=usage,
                model=self.model_name,
                finish_reason=choice.finish_reason or "",
            )

    def _decoding(self, request: CompletionRequest) -> dict[str, Any]:
        """The parameters that differ between a reasoning deployment and the rest.

        A reasoning model refuses both of the ones every other deployment
        needs: ``max_tokens`` is rejected outright in favour of
        ``max_completion_tokens``, and any ``temperature`` but the default is
        an error rather than a rounding. Sending them anyway is a 400 on the
        first call, which is a poor way to find out.
        """
        budget = min(request.max_tokens, self._settings.max_output_tokens)
        if not self._settings.is_reasoning_model:
            return {"temperature": request.temperature, "max_tokens": budget}

        # The reasoning itself is billed against this budget before a single
        # visible token is produced, so it cannot be as tight as the other path.
        options: dict[str, Any] = {"max_completion_tokens": budget}
        if self._settings.reasoning_effort:
            options["reasoning_effort"] = self._settings.reasoning_effort
        return options

    @staticmethod
    def _response_format(request: CompletionRequest) -> dict[str, Any] | None:
        """Constrain decoding to the schema when one is supplied.

        With no schema this returns None, and no ``response_format`` is sent
        at all. It used to fall back to ``{"type": "json_object"}``, which an
        OpenAI-compatible endpoint rejects outright unless the prompt happens
        to contain the word "json" -- a 400 whose message is about the
        *messages*, for a header the caller never set. Asking for JSON without
        saying what shape it should be was buying nothing anyway:
        ``extract_json`` reads the object out of an ordinary reply.
        """
        if not request.json_schema:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "strict": False,
                "schema": request.json_schema,
            },
        }


def build_async_client(settings: AzureOpenAISettings) -> Any:
    """An SDK client for whichever surface the endpoint names.

    An AI Foundry project exposes the model twice. The ``/openai/v1`` surface
    is OpenAI-compatible -- a plain ``base_url``, no api-version, the same
    client you would point at api.openai.com -- and is what the portal hands
    you now. The classic surface takes ``azure_endpoint`` and stamps an
    ``api-version`` onto every request. Sending one's URL to the other's
    client produces a 404 on a path you never wrote.
    """
    key = settings.api_key
    use_entra = settings.use_managed_identity and not key

    if not use_entra:
        # A key is the same string on both surfaces; only the URL shape moves.
        if settings.uses_v1_api:
            from openai import AsyncOpenAI

            return AsyncOpenAI(
                base_url=settings.base_url,
                api_key=key,
                timeout=settings.timeout_seconds,
                # Retries are the platform's decision, not the SDK's: they have
                # to go through the step's retry policy so they are counted,
                # backed off and dead-lettered like every other attempt.
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

    # Entra, on either surface, goes through AsyncAzureOpenAI -- including for
    # the v1 URL, where `base_url` is passed instead of `azure_endpoint` so the
    # SDK stops composing a deployment path and uses the URL as given. The
    # plain AsyncOpenAI client cannot be used here: its `api_key` is a string
    # fixed at construction, so a token would be captured once and the process
    # would start failing an hour later when it expired.
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), settings.entra_scope
    )
    common: dict[str, Any] = {
        "azure_ad_token_provider": token_provider,
        "api_version": settings.api_version,
        "timeout": settings.timeout_seconds,
        "max_retries": 0,
    }
    if settings.uses_v1_api:
        return AsyncAzureOpenAI(base_url=settings.base_url, **common)
    return AsyncAzureOpenAI(azure_endpoint=settings.endpoint, **common)


def _translate(exc: Exception) -> Exception:
    """Map SDK failures onto the retry taxonomy.

    Content filtering and malformed requests are *permanent*: retrying an
    identical prompt that tripped the filter only burns quota.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "RateLimitError" or status == 429:
        return RateLimitedError("Azure OpenAI throttled the request")
    if name in {"APITimeoutError", "APIConnectionTimeoutError"} or status == 408:
        return DependencyTimeoutError("Azure OpenAI timed out")
    if name == "APIConnectionError" or (status is not None and status >= 500):
        return DependencyUnavailableError("Azure OpenAI unavailable", status=status)
    if name == "BadRequestError" and "content_filter" in str(exc).lower():
        return PermanentError("Request blocked by the content filter")

    # An auth failure and a malformed request are both permanent and both 4xx,
    # and lumping them together sends you reading your payload when the problem
    # is the key. The detail is carried through because the SDK's own message
    # names the parameter or the deployment that was refused.
    if status in {401, 403}:
        return PermanentError(
            "Azure OpenAI refused the credentials",
            status=status,
            detail=str(exc)[:300],
        )
    if status == 404:
        return PermanentError(
            "Azure OpenAI has no such deployment at that URL",
            status=status,
            detail=str(exc)[:300],
        )
    if status is not None and 400 <= status < 500:
        return ValidationError(
            "Azure OpenAI rejected the request", status=status, detail=str(exc)[:300]
        )
    return exc

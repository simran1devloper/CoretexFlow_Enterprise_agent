"""What this deployment is actually connected to.

Three different kinds of claim, kept apart because they are not equally
strong:

**Configured** comes from settings and is exact -- it is which adapter the
composition root bound at startup, and whether its settings are filled in.

**Reachable** is a live, read-only probe. Four dependencies offer a cheap read
that proves they answered; two deliberately do not get one. A language model
is not probed because a completion costs money and seconds, and a status page
that bills you to render is a bad status page. Identity is not probed because
validating a token requires a token.

**In use** is evidence from the audit trail: when a system was last actually
called, and which model actually served work. It is the strongest of the
three and the one a green dot cannot give you -- "it answered a synthetic
ping" is weaker than "it did this, at this time".

``reachable`` is therefore three-valued. ``None`` means *not probed*, and must
never be rendered as a failure.

Placement: this reads settings, the container's bound adapters and the audit
trail at once. That is composition-level knowledge rather than any one
module's, so it sits beside the health probe instead of being pushed into a
module that would then have to import three others.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from cortexflow.config.settings import (
    AuthMode,
    BlobBackend,
    BusBackend,
    CacheBackend,
    LlmBackend,
    Settings,
    StateBackend,
)
from cortexflow.modules.document.domain.storage import BlobRef
from cortexflow.modules.workflow.domain.definition import Queue

PROBE_KEY = "cortexflow::integration-probe"
"""A key nothing ever writes. A miss is a successful answer."""

ABSENT_BLOB = "__cortexflow_probe__/absent"


class EnvVar(BaseModel):
    """One setting an adapter needs, with an example value.

    ``example`` is always a placeholder. Nothing here is read from the running
    configuration, so a client copying this block cannot accidentally be
    handed another deployment's endpoint -- or a secret.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    example: str
    secret: bool = False
    """True when the value is a credential and belongs in a platform secret."""


class AdapterOption(BaseModel):
    """One adapter this integration point supports.

    Listed whether or not it is the one bound, because otherwise a deployment
    running locally gives no sign that Azure OpenAI or Cosmos DB are even
    options -- and "what can this connect to" is half of what an integrations
    page is for. Every entry here has an implementation behind it; nothing is
    listed that would have to be written first.
    """

    model_config = ConfigDict(frozen=True)

    value: str
    """The value the setting takes, e.g. ``azure_openai``."""

    name: str
    active: bool
    configured: bool
    """Whether the settings this adapter needs are actually filled in."""

    kind: str
    """``cloud``, ``self-hosted`` or ``in-process``."""

    requires: tuple[str, ...] = ()
    """Environment variables it needs. Names only, never values."""

    auth: str = "none"
    """``managed-identity``, ``platform-secret`` or ``none``."""

    auth_note: str = ""
    """Why no key is needed, or which secret is."""

    env: tuple[EnvVar, ...] = ()
    """The settings block to copy, with placeholder values."""


class IntegrationStatus(BaseModel):
    """One outbound dependency."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    category: str
    adapter: str
    """The implementation actually bound, not the one that could be."""

    configured: bool
    detail: str = ""
    """Where it points, host-only. Never a key, token or connection string."""

    reachable: bool | None = None
    """True, False, or None for "deliberately not probed"."""

    probe_ms: int | None = None
    error: str = ""

    simulated: bool = False
    """True when this is a stand-in, not a connection to a real system."""

    last_used: str = ""
    calls: int = 0
    failures: int = 0
    note: str = ""

    setting: str = ""
    """The environment variable that switches this integration point."""

    options: tuple[AdapterOption, ...] = ()
    """Every adapter this point supports, active one included."""


class IntegrationsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: str
    environment: str
    integrations: tuple[IntegrationStatus, ...] = ()

    total: int = 0
    configured: int = 0
    reachable: int = 0
    unreachable: int = 0
    not_probed: int = 0
    simulated: int = 0

    notes: tuple[str, ...] = ()


NOTES = (
    "Adapters are chosen from settings when the container is built, so an "
    "integration is switched in config/azure.env or config/local.env, not "
    "from this page.",
    "Credentials come from the environment, and on Azure from Key Vault via "
    "managed identity. Nothing here accepts or stores one.",
    "The language model and identity provider are not probed: a completion "
    "costs money, and validating a token needs a token. Their evidence is "
    "what they have actually served.",
    "Enterprise systems marked simulated are in-process stand-ins. They "
    "exercise the real tool, policy and approval path, but talk to no "
    "external system.",
)

# Tool name prefix -> the system it speaks to. The audit trail records the
# tool, so this is how "last used" is attributed.
_SYSTEMS = {
    "hr": ("hr_system", "HR system"),
    "finance": ("finance_system", "Finance / ERP"),
    "marketing": ("marketing_system", "Marketing / CRM"),
}


async def build_report(container: Any, *, probe: bool = True) -> IntegrationsReport:
    """Assemble every integration's status."""
    settings: Settings = container.settings
    usage = await _usage(container)

    rows: list[IntegrationStatus] = [
        await _state(container, settings, probe),
        await _bus(container, settings, probe),
        await _cache(container, settings, probe),
        await _blobs(container, settings, probe),
        _model(settings),
        _identity(settings),
    ]
    rows.extend(_system(key, label, usage) for key, (_, label) in _SYSTEMS.items())

    return IntegrationsReport(
        profile=str(settings.profile),
        environment=settings.environment,
        integrations=tuple(rows),
        total=len(rows),
        configured=sum(1 for r in rows if r.configured),
        reachable=sum(1 for r in rows if r.reachable is True),
        unreachable=sum(1 for r in rows if r.reachable is False),
        not_probed=sum(1 for r in rows if r.reachable is None),
        simulated=sum(1 for r in rows if r.simulated),
        notes=NOTES,
    )


# ----------------------------------------------------------------------
# The catalogue: every adapter each integration point supports.
#
# Kept as data next to the settings enums it mirrors, and asserted against
# them by a test -- adding a `LlmBackend` member without a row here fails,
# which is the only way this stays a catalogue rather than a stale list.
# ----------------------------------------------------------------------
_MI = (
    "managed-identity",
    "No key to enter. The app authenticates as its own managed identity, and "
    "this resource is provisioned with key authentication disabled -- a key "
    "would be rejected even if you had one.",
)
_NONE = ("none", "Needs no credential.")
_SECRET = (
    "platform-secret",
    "The one integration here that needs a credential. Put it in a platform "
    "secret -- a Container App secret, a Kubernetes secret or Key Vault -- "
    "and reference it, rather than in a committed file.",
)
_SELF = (
    "none",
    "Reached over the network with no credential, so keep it on a private "
    "network. There is nothing to authenticate.",
)


def _opt(
    value: str,
    name: str,
    kind: str,
    auth: str,
    auth_note: str,
    *,
    env: tuple[tuple[str, str, bool], ...] = (),
) -> AdapterOption:
    """One catalogue row. ``active`` and ``configured`` are filled in later."""
    variables = tuple(
        EnvVar(name=n, example=example, secret=secret) for n, example, secret in env
    )
    return AdapterOption(
        value=value,
        name=name,
        active=False,
        configured=False,
        kind=kind,
        # Only the settings that identify the target, not the selector itself:
        # "what must I supply" is a different question from "what do I paste".
        requires=tuple(v.name for v in variables if v.name not in _SELECTORS),
        auth=auth,
        auth_note=auth_note,
        env=variables,
    )


_SELECTORS = {
    "CORTEXFLOW_STATE_BACKEND",
    "CORTEXFLOW_BUS_BACKEND",
    "CORTEXFLOW_CACHE_BACKEND",
    "CORTEXFLOW_BLOB_BACKEND",
    "CORTEXFLOW_LLM_BACKEND",
    "CORTEXFLOW_AUTH_MODE",
}


_OPTIONS: dict[str, tuple[AdapterOption, ...]] = {
    "state": (
        _opt("memory", "In-memory state store", "in-process", *_NONE),
        _opt(
            "cosmos", "Azure Cosmos DB", "cloud", *_MI,
            env=(
                ("CORTEXFLOW_STATE_BACKEND", "cosmos", False),
                    (
                    "CORTEXFLOW_COSMOS_ENDPOINT",
                    "https://<account>.documents.azure.com:443/",
                    False,
                ),
                ("CORTEXFLOW_COSMOS_DATABASE", "cortexflow", False),
                ("CORTEXFLOW_COSMOS_USE_MANAGED_IDENTITY", "true", False),
            ),
        ),
    ),
    "bus": (
        _opt("memory", "In-memory message bus", "in-process", *_NONE),
        _opt(
            "servicebus", "Azure Service Bus", "cloud", *_MI,
            env=(
                ("CORTEXFLOW_BUS_BACKEND", "servicebus", False),
                ("CORTEXFLOW_SERVICEBUS_NAMESPACE", "<namespace>.servicebus.windows.net", False),
                ("CORTEXFLOW_SERVICEBUS_USE_MANAGED_IDENTITY", "true", False),
            ),
        ),
    ),
    "cache": (
        _opt("memory", "In-memory cache", "in-process", *_NONE),
        _opt(
            "redis", "Redis", "cloud", *_SECRET,
            env=(
                ("CORTEXFLOW_CACHE_BACKEND", "redis", False),
                ("CORTEXFLOW_REDIS_URL", "rediss://:<access-key>@<host>:6380/0", True),
            ),
        ),
    ),
    "blobs": (
        _opt(
            "filesystem", "Local filesystem", "in-process", *_NONE,
            env=(
                ("CORTEXFLOW_BLOB_BACKEND", "filesystem", False),
                ("CORTEXFLOW_BLOB_LOCAL_ROOT", ".data/blobs", False),
            ),
        ),
        _opt(
            "azure_blob", "Azure Blob Storage", "cloud", *_MI,
            env=(
                ("CORTEXFLOW_BLOB_BACKEND", "azure_blob", False),
                ("CORTEXFLOW_BLOB_ACCOUNT_URL", "https://<account>.blob.core.windows.net/", False),
                ("CORTEXFLOW_BLOB_CONTAINER", "documents", False),
                ("CORTEXFLOW_BLOB_USE_MANAGED_IDENTITY", "true", False),
            ),
        ),
    ),
    "llm": (
        _opt("deterministic", "Deterministic model", "in-process", *_NONE),
        _opt(
            "azure_openai", "Azure OpenAI", "cloud", *_MI,
            env=(
                ("CORTEXFLOW_LLM_BACKEND", "azure_openai", False),
                ("CORTEXFLOW_OPENAI_ENDPOINT", "https://<account>.openai.azure.com/", False),
                ("CORTEXFLOW_OPENAI_CHAT_DEPLOYMENT", "gpt-4o", False),
                ("CORTEXFLOW_OPENAI_USE_MANAGED_IDENTITY", "true", False),
            ),
        ),
        _opt(
            "ollama", "Ollama", "self-hosted", *_SELF,
            env=(
                ("CORTEXFLOW_LLM_BACKEND", "ollama", False),
                ("CORTEXFLOW_OLLAMA_HOST", "http://<host>:11434", False),
                ("CORTEXFLOW_OLLAMA_MODEL", "llama3.1:8b", False),
            ),
        ),
    ),
    "identity": (
        _opt("dev", "Development headers", "in-process", *_NONE),
        _opt(
            "entra", "Microsoft Entra ID", "cloud", *_NONE,
            env=(
                ("CORTEXFLOW_AUTH_MODE", "entra", False),
                ("CORTEXFLOW_AUTH_TENANT_ID", "<directory-tenant-id>", False),
                ("CORTEXFLOW_AUTH_AUDIENCE", "api://cortexflow", False),
                (
                    "CORTEXFLOW_AUTH_ISSUER",
                    "https://login.microsoftonline.com/<tenant>/v2.0",
                    False,
                ),
            ),
        ),
    ),
}


_SETTING = {
    "state": "CORTEXFLOW_STATE_BACKEND",
    "bus": "CORTEXFLOW_BUS_BACKEND",
    "cache": "CORTEXFLOW_CACHE_BACKEND",
    "blobs": "CORTEXFLOW_BLOB_BACKEND",
    "llm": "CORTEXFLOW_LLM_BACKEND",
    "identity": "CORTEXFLOW_AUTH_MODE",
}


def _options(key: str, active: str, settings: Settings) -> tuple[AdapterOption, ...]:
    """Every adapter this point supports, with the live flags filled in."""
    return tuple(
        option.model_copy(
            update={
                "active": option.value == active,
                "configured": _option_configured(option.value, settings),
            }
        )
        for option in _OPTIONS[key]
    )


def _option_configured(value: str, settings: Settings) -> bool:
    """Whether somebody actually pointed this adapter at something.

    Tested with ``model_fields_set`` -- the fields the environment or profile
    file supplied -- rather than by asking whether the value is non-empty.
    Several of these carry a localhost default: ``redis.url`` is
    ``redis://localhost:6379/0`` and ``ollama.host`` is
    ``http://localhost:11434`` whether or not either is running. Truthiness
    would report both as configured on a machine that has neither installed,
    which is exactly the reassuring-but-wrong answer an operations page must
    not give.

    Adapters needing nothing external are always configured: there is nothing
    to point at.
    """
    if value in {"memory", "filesystem", "deterministic", "dev"}:
        return True

    group, fields = {
        "cosmos": (settings.cosmos, ("endpoint",)),
        "servicebus": (settings.servicebus, ("namespace",)),
        "redis": (settings.redis, ("url",)),
        "azure_blob": (settings.blob, ("account_url",)),
        "azure_openai": (settings.openai, ("endpoint",)),
        "ollama": (settings.ollama, ("host",)),
        "entra": (settings.auth, ("tenant_id", "audience")),
    }.get(value, (None, ()))

    if group is None:
        return False
    supplied = group.model_fields_set
    return all(
        field in supplied and bool(getattr(group, field)) for field in fields
    )


# ----------------------------------------------------------------------
# Probes. Every one is read-only, and none may raise: a dependency being down
# is exactly when this page is read, so a probe that propagates its failure
# would take out the screen that explains it.
# ----------------------------------------------------------------------
async def _probe(run: Callable[[], Awaitable[Any]]) -> tuple[bool, int, str]:
    started = time.perf_counter()
    try:
        await run()
    except Exception as exc:
        return False, _elapsed(started), f"{type(exc).__name__}: {exc}"[:200]
    return True, _elapsed(started), ""


def _elapsed(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


async def _state(container: Any, settings: Settings, probe: bool) -> IntegrationStatus:
    cosmos = settings.state_backend is StateBackend.COSMOS
    adapter = "cosmos" if cosmos else "memory"
    reachable, ms, error = (None, None, "")
    if probe:
        reachable, ms, error = await _probe(
            lambda: container.workflows.list(settings.default_tenant_id, limit=1)
        )

    return IntegrationStatus(
        key="state",
        name="Azure Cosmos DB" if cosmos else "In-memory state store",
        category="State & storage",
        adapter=adapter,
        configured=bool(settings.cosmos.endpoint) if cosmos else True,
        detail=_host(settings.cosmos.endpoint) if cosmos else "process memory",
        reachable=reachable,
        probe_ms=ms,
        error=error,
        simulated=not cosmos,
        note=(
            "Workflow state, approvals and the audit trail. The same read "
            "/ready performs."
            if cosmos
            else "State lives in this process and is lost on restart. Real "
            "durability needs the azure profile."
        ),
        setting=_SETTING["state"],
        options=_options("state", adapter, settings),
    )


async def _bus(container: Any, settings: Settings, probe: bool) -> IntegrationStatus:
    real = settings.bus_backend is BusBackend.SERVICE_BUS
    adapter = "servicebus" if real else "memory"
    reachable, ms, error = (None, None, "")
    if probe:
        # A read of the dead-letter queue: it touches the broker and
        # publishes nothing.
        reachable, ms, error = await _probe(
            lambda: container.bus.dead_letters(Queue.CONTROL, limit=1)
        )

    return IntegrationStatus(
        key="bus",
        name="Azure Service Bus" if real else "In-memory message bus",
        category="Messaging",
        adapter=adapter,
        configured=bool(settings.servicebus.namespace) if real else True,
        detail=settings.servicebus.namespace if real else "process memory",
        reachable=reachable,
        probe_ms=ms,
        error=error,
        simulated=not real,
        note=(
            "Carries every step dispatch, retry and dead letter."
            if real
            else "Keeps the production semantics that matter -- locks, "
            "delivery counts, dead-lettering -- inside one process."
        ),
        setting=_SETTING["bus"],
        options=_options("bus", adapter, settings),
    )


async def _cache(container: Any, settings: Settings, probe: bool) -> IntegrationStatus:
    real = settings.cache_backend is CacheBackend.REDIS
    adapter = "redis" if real else "memory"
    reachable, ms, error = (None, None, "")
    if probe:
        # A miss is a successful answer, so this needs nothing to exist.
        reachable, ms, error = await _probe(lambda: container.cache.get(PROBE_KEY))

    return IntegrationStatus(
        key="cache",
        name="Redis" if real else "In-memory cache",
        category="State & storage",
        adapter=adapter,
        configured=bool(settings.redis.url) if real else True,
        detail=_host(settings.redis.url) if real else "process memory",
        reachable=reachable,
        probe_ms=ms,
        error=error,
        simulated=not real,
        note="Idempotency keys, distributed locks and rate limiting.",
        setting=_SETTING["cache"],
        options=_options("cache", adapter, settings),
    )


async def _blobs(container: Any, settings: Settings, probe: bool) -> IntegrationStatus:
    azure = settings.blob_backend is BlobBackend.AZURE_BLOB
    adapter = "azure_blob" if azure else "filesystem"
    reachable, ms, error = (None, None, "")
    if probe:
        reachable, ms, error = await _probe(
            lambda: container.blobs.get(
                BlobRef(container=settings.blob.container, path=ABSENT_BLOB)
            )
        )
        # "No such blob" is the store answering, which is what was asked.
        if not reachable and _is_not_found(error):
            reachable, error = True, ""

    return IntegrationStatus(
        key="blobs",
        name="Azure Blob Storage" if azure else "Local filesystem",
        category="State & storage",
        adapter=adapter,
        configured=bool(settings.blob.account_url) if azure else True,
        detail=(
            _host(settings.blob.account_url) if azure else str(settings.blob.local_root)
        ),
        reachable=reachable,
        probe_ms=ms,
        error=error,
        simulated=not azure,
        note=(
            "Uploaded documents and generated reports. Probed by asking for a "
            "blob that does not exist: being told so proves the store "
            "answered."
        ),
        setting=_SETTING["blobs"],
        options=_options("blobs", adapter, settings),
    )


def _model(settings: Settings) -> IntegrationStatus:
    # Optional in the model, but the profile validator always fills it.
    backend = settings.llm_backend or LlmBackend.DETERMINISTIC
    names = {
        LlmBackend.AZURE_OPENAI: "Azure OpenAI",
        LlmBackend.OLLAMA: "Ollama",
        LlmBackend.DETERMINISTIC: "Deterministic model",
    }
    configured = {
        LlmBackend.AZURE_OPENAI: bool(settings.openai.endpoint),
        LlmBackend.OLLAMA: bool(settings.ollama.host),
        LlmBackend.DETERMINISTIC: True,
    }
    detail = {
        LlmBackend.AZURE_OPENAI: _host(settings.openai.endpoint),
        LlmBackend.OLLAMA: _host(settings.ollama.host),
        LlmBackend.DETERMINISTIC: "in-process, seeded",
    }
    served = {
        LlmBackend.AZURE_OPENAI: settings.openai.chat_deployment,
        LlmBackend.OLLAMA: settings.ollama.model,
        LlmBackend.DETERMINISTIC: "seeded stub",
    }.get(backend, "")

    return IntegrationStatus(
        key="llm",
        name=names.get(backend, str(backend)),
        category="AI models",
        adapter=str(backend),
        configured=configured.get(backend, False),
        detail=detail.get(backend, ""),
        # Deliberately unprobed -- see the module docstring.
        reachable=None,
        simulated=backend is LlmBackend.DETERMINISTIC,
        note=(
            f"Not probed: a completion costs money and seconds, and a status "
            f"page should not bill you to render. Configured model: "
            f"{served or 'none'}. Which model actually served each step is "
            f"recorded per run and reported on AI Evaluation."
        ),
        setting=_SETTING["llm"],
        options=_options("llm", str(backend), settings),
    )


def _identity(settings: Settings) -> IntegrationStatus:
    entra = settings.auth.mode is AuthMode.ENTRA
    return IntegrationStatus(
        key="identity",
        name="Microsoft Entra ID" if entra else "Development headers",
        category="Identity",
        adapter=str(settings.auth.mode),
        configured=bool(settings.auth.tenant_id and settings.auth.audience)
        if entra
        else True,
        detail=f"tenant {settings.auth.tenant_id}" if entra else "unverified",
        reachable=None,
        simulated=not entra,
        note=(
            "Not probed -- validating a token needs a token. Accounts and app "
            "roles live here; this platform only reads claims."
            if entra
            else "Requests are not verified. Refused outside a dev "
            "environment by the authenticator itself."
        ),
        setting=_SETTING["identity"],
        options=_options("identity", str(settings.auth.mode), settings),
    )


def _system(key: str, label: str, usage: dict[str, Any]) -> IntegrationStatus:
    stats = usage.get("systems", {}).get(key, {})
    return IntegrationStatus(
        key=f"{key}_system",
        name=label,
        category="Enterprise systems",
        adapter="in-process stand-in",
        configured=True,
        detail="simulated",
        reachable=None,
        simulated=True,
        last_used=stats.get("last_used", ""),
        calls=stats.get("calls", 0),
        failures=stats.get("failures", 0),
        note=(
            "A stand-in, not a connection. It exercises the real tool, policy "
            "and approval path, which is what the workflow tests need, but no "
            "external system is contacted. A real deployment replaces it with "
            "the integration service's HTTP client."
        ),
    )


# ----------------------------------------------------------------------
async def _usage(container: Any) -> dict[str, Any]:
    """What the audit trail says has actually been called.

    Stronger than a probe: not "it answered a ping" but "it did this, then".
    """
    try:
        events = await container.audit_repository.query(
            container.settings.default_tenant_id, limit=1000
        )
    except Exception:
        return {}

    systems: dict[str, dict[str, Any]] = {}
    for event in events:
        if not event.tool:
            continue
        domain, _, _ = event.tool.partition(".")
        if domain not in _SYSTEMS:
            continue
        stats = systems.setdefault(domain, {"calls": 0, "failures": 0, "last_used": ""})
        stats["calls"] += 1
        if str(event.outcome) == "FAILURE":
            stats["failures"] += 1
        stats["last_used"] = _newest(stats["last_used"], event.occurred_at)

    return {"systems": systems}


def _newest(current: str, moment: datetime) -> str:
    candidate = moment.isoformat()
    return candidate if not current or candidate > current else current


def _host(url: str) -> str:
    """Host only. An endpoint may carry a key in its query string."""
    if not url:
        return ""
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0].split("?", 1)[0]


def _is_not_found(error: str) -> bool:
    lowered = error.lower()
    return "notfound" in lowered or "not found" in lowered or "no such" in lowered

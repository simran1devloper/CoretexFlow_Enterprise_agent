"""Authentication.

Two modes, chosen by configuration:

* ``entra`` validates Microsoft Entra ID JWTs against the tenant's JWKS, with
  signature, issuer, audience and expiry all enforced.
* ``dev`` accepts an unsigned header describing the caller, for local runs.
  It refuses to start outside a development environment, so it cannot be
  switched on in production by accident.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from cortexflow.config.settings import AuthMode, AuthSettings
from cortexflow.shared.errors import AuthenticationError
from cortexflow.shared.identity import Department, Principal, Role
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.security.rbac import DEPARTMENT_ROLES

logger = get_logger(__name__)

DEV_PRINCIPAL_HEADER = "X-Dev-Principal"
TENANT_HEADER = "X-Tenant-Id"


def departments_for_roles(roles: frozenset[Role]) -> frozenset[Department]:
    """Derive department access from roles so claims stay minimal."""
    if Role.PLATFORM_ADMIN in roles or Role.OPERATOR in roles:
        return frozenset(Department)
    return frozenset(
        dept for dept, dept_roles in DEPARTMENT_ROLES.items() if roles & dept_roles
    )


class _JwksCache:
    """Caches signing keys; Entra rotates them, so this must expire."""

    def __init__(self, url: str, ttl_seconds: int) -> None:
        self._url = url
        self._ttl = ttl_seconds
        self._keys: dict[str, Any] = {}
        self._fetched_at = 0.0

    async def get(self, kid: str) -> Any:
        if kid not in self._keys or time.monotonic() - self._fetched_at > self._ttl:
            await self._refresh()
        if kid not in self._keys:
            raise AuthenticationError("Token signed with an unknown key", kid=kid)
        return self._keys[kid]

    async def _refresh(self) -> None:
        from jwt import PyJWK

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self._url)
            response.raise_for_status()
            document = response.json()
        self._keys = {key["kid"]: PyJWK.from_dict(key).key for key in document["keys"]}
        self._fetched_at = time.monotonic()


class EntraAuthenticator:
    """Validates Entra ID bearer tokens."""

    def __init__(self, settings: AuthSettings) -> None:
        if not settings.tenant_id or not settings.audience:
            raise ValueError("Entra authentication requires a tenant id and audience")
        self._settings = settings
        jwks_url = (
            settings.jwks_url
            or f"https://login.microsoftonline.com/{settings.tenant_id}/discovery/v2.0/keys"
        )
        self._issuer = (
            settings.issuer
            or f"https://login.microsoftonline.com/{settings.tenant_id}/v2.0"
        )
        self._jwks = _JwksCache(jwks_url, settings.jwks_cache_seconds)

    async def authenticate(self, token: str) -> Principal:
        import jwt

        try:
            header = jwt.get_unverified_header(token)
            key = await self._jwks.get(header["kid"])
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self._settings.audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("Invalid bearer token", reason=str(exc)) from exc
        except KeyError as exc:
            raise AuthenticationError("Malformed token header") from exc
        return self._to_principal(claims)

    def _to_principal(self, claims: dict[str, Any]) -> Principal:
        roles = _parse_roles(claims.get("roles", []))
        # The Entra directory tenant is the platform tenant: multi-tenancy is
        # anchored to the identity provider, never to a client-supplied header.
        tenant_id = claims.get("tid") or self._settings.tenant_id
        return Principal(
            subject=claims["sub"],
            tenant_id=tenant_id,
            display_name=claims.get("name", ""),
            email=claims.get("preferred_username", claims.get("email", "")),
            roles=roles,
            departments=departments_for_roles(roles),
        )


class DevAuthenticator:
    """Header-based principals for local development only."""

    def __init__(self, settings: AuthSettings, *, default_tenant: str, environment: str) -> None:
        if environment.lower() in {"prod", "production", "staging"}:
            raise RuntimeError(
                "Development authentication cannot be enabled outside a dev environment"
            )
        self._settings = settings
        self._default_tenant = default_tenant
        logger.warning("development authentication enabled: requests are not verified")

    async def authenticate(self, token: str) -> Principal:
        """``token`` is a JSON object: ``{"sub": ..., "roles": [...]}``."""
        try:
            claims = json.loads(token)
        except json.JSONDecodeError as exc:
            raise AuthenticationError("Dev principal header must be JSON") from exc
        if not isinstance(claims, dict) or "sub" not in claims:
            raise AuthenticationError("Dev principal header must contain 'sub'")

        roles = _parse_roles(claims.get("roles", []))
        return Principal(
            subject=str(claims["sub"]),
            tenant_id=str(claims.get("tenant_id") or self._default_tenant),
            display_name=str(claims.get("name", claims["sub"])),
            email=str(claims.get("email", "")),
            roles=roles,
            departments=departments_for_roles(roles),
        )


def _parse_roles(raw: Any) -> frozenset[Role]:
    if isinstance(raw, str):
        raw = [raw]
    roles: set[Role] = set()
    for item in raw or []:
        try:
            roles.add(Role(str(item).strip().lower()))
        except ValueError:
            logger.debug("ignoring unknown role claim", extra={"role": str(item)})
    return frozenset(roles)


def build_authenticator(
    settings: AuthSettings, *, default_tenant: str, environment: str
) -> EntraAuthenticator | DevAuthenticator:
    if settings.mode is AuthMode.ENTRA:
        return EntraAuthenticator(settings)
    return DevAuthenticator(settings, default_tenant=default_tenant, environment=environment)

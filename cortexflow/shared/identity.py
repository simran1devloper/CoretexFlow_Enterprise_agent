"""Tenancy and caller identity.

Every persisted entity carries a ``tenant_id``; every repository method takes
one.  Isolation is enforced at the data-access boundary rather than remembered
at each call site.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Platform roles, mapped from Entra ID app roles / group claims."""

    PLATFORM_ADMIN = "platform_admin"
    OPERATOR = "operator"
    """Runs the ops dashboard: replay, cancel, inspect dead letters."""

    FINANCE_MANAGER = "finance_manager"
    FINANCE_ANALYST = "finance_analyst"
    HR_MANAGER = "hr_manager"
    HR_SPECIALIST = "hr_specialist"
    MARKETING_MANAGER = "marketing_manager"
    MARKETING_ANALYST = "marketing_analyst"
    EMPLOYEE = "employee"

    SERVICE_AGENT = "service_agent"
    """Non-human principal: a worker executing a step on a workflow's behalf."""


class Department(StrEnum):
    FINANCE = "finance"
    HR = "hr"
    MARKETING = "marketing"
    PLATFORM = "platform"


class Principal(BaseModel):
    """An authenticated caller -- human or service."""

    model_config = ConfigDict(frozen=True)

    subject: str
    tenant_id: str
    display_name: str = ""
    email: str = ""
    roles: frozenset[Role] = Field(default_factory=frozenset)
    departments: frozenset[Department] = Field(default_factory=frozenset)

    def has_role(self, *roles: Role) -> bool:
        return bool(self.roles.intersection(roles))

    def is_admin(self) -> bool:
        return Role.PLATFORM_ADMIN in self.roles

    def can_access_department(self, department: Department) -> bool:
        return self.is_admin() or department in self.departments

    @classmethod
    def service(cls, name: str, tenant_id: str) -> Principal:
        """Build the principal a worker uses when it executes a step."""
        return cls(
            subject=f"service::{name}",
            tenant_id=tenant_id,
            display_name=name,
            roles=frozenset({Role.SERVICE_AGENT}),
            departments=frozenset(Department),
        )

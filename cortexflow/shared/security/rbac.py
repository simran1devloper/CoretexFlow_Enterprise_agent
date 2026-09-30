"""Role-based access control.

Permissions are declared as data, so adding a role is a table edit rather than
a hunt through ``if`` statements scattered across routers.
"""

from __future__ import annotations

from enum import StrEnum

from cortexflow.shared.errors import AuthorizationError
from cortexflow.shared.identity import Department, Principal, Role


class Permission(StrEnum):
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_CANCEL = "workflow:cancel"
    WORKFLOW_REPLAY = "workflow:replay"

    APPROVAL_READ = "approval:read"
    APPROVAL_DECIDE = "approval:decide"

    AUDIT_READ = "audit:read"
    OPS_READ = "ops:read"
    OPS_MANAGE = "ops:manage"
    """Dead-letter inspection, replay, forced cancellation."""

    DEFINITION_READ = "definition:read"
    DEFINITION_MANAGE = "definition:manage"


_ALL: frozenset[Permission] = frozenset(Permission)

_MANAGER_PERMISSIONS = frozenset(
    {
        Permission.WORKFLOW_CREATE,
        Permission.WORKFLOW_READ,
        Permission.WORKFLOW_CANCEL,
        Permission.APPROVAL_READ,
        Permission.APPROVAL_DECIDE,
        Permission.AUDIT_READ,
        Permission.DEFINITION_READ,
    }
)

_ANALYST_PERMISSIONS = frozenset(
    {
        Permission.WORKFLOW_CREATE,
        Permission.WORKFLOW_READ,
        Permission.APPROVAL_READ,
        Permission.DEFINITION_READ,
    }
)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.PLATFORM_ADMIN: _ALL,
    Role.OPERATOR: frozenset(
        {
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_CANCEL,
            Permission.WORKFLOW_REPLAY,
            Permission.APPROVAL_READ,
            Permission.AUDIT_READ,
            Permission.OPS_READ,
            Permission.OPS_MANAGE,
            Permission.DEFINITION_READ,
        }
    ),
    Role.FINANCE_MANAGER: _MANAGER_PERMISSIONS,
    Role.HR_MANAGER: _MANAGER_PERMISSIONS,
    Role.MARKETING_MANAGER: _MANAGER_PERMISSIONS,
    Role.FINANCE_ANALYST: _ANALYST_PERMISSIONS,
    Role.HR_SPECIALIST: _ANALYST_PERMISSIONS,
    Role.MARKETING_ANALYST: _ANALYST_PERMISSIONS,
    Role.EMPLOYEE: frozenset({Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ}),
    # A worker may read and drive workflows, but may never decide an approval:
    # that separation is what keeps "human in the loop" meaningful.
    Role.SERVICE_AGENT: frozenset(
        {
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_CREATE,
            Permission.APPROVAL_READ,
            Permission.DEFINITION_READ,
        }
    ),
}

DEPARTMENT_ROLES: dict[Department, frozenset[Role]] = {
    Department.FINANCE: frozenset({Role.FINANCE_MANAGER, Role.FINANCE_ANALYST}),
    Department.HR: frozenset({Role.HR_MANAGER, Role.HR_SPECIALIST}),
    Department.MARKETING: frozenset({Role.MARKETING_MANAGER, Role.MARKETING_ANALYST}),
    Department.PLATFORM: frozenset({Role.PLATFORM_ADMIN, Role.OPERATOR}),
}


def permissions_for(principal: Principal) -> frozenset[Permission]:
    granted: set[Permission] = set()
    for role in principal.roles:
        granted |= ROLE_PERMISSIONS.get(role, frozenset())
    return frozenset(granted)


def has_permission(principal: Principal, permission: Permission) -> bool:
    return permission in permissions_for(principal)


def require_permission(principal: Principal, permission: Permission) -> None:
    if not has_permission(principal, permission):
        raise AuthorizationError(
            "Principal lacks the required permission",
            subject=principal.subject,
            permission=str(permission),
        )


def require_tenant(principal: Principal, tenant_id: str) -> None:
    """Cross-tenant access is refused even for platform admins.

    An admin who genuinely needs another tenant's data authenticates into that
    tenant; there is no ambient superuser that spans the boundary.
    """
    if principal.tenant_id != tenant_id:
        raise AuthorizationError(
            "Cross-tenant access denied",
            subject=principal.subject,
            principal_tenant=principal.tenant_id,
            requested_tenant=tenant_id,
        )


def require_department(principal: Principal, department: Department) -> None:
    if not principal.can_access_department(department):
        raise AuthorizationError(
            "Principal lacks access to this department",
            subject=principal.subject,
            department=str(department),
        )

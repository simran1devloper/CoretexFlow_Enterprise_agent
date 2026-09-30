"""HTTP routers.

``ALL_ROUTERS`` is the single list, and both entry points mount it. They kept
their own copies until a new router was added to one of them and silently did
not exist in the other -- which is the failure this list prevents: the app that
serves development traffic is not the one most changes are read against.
"""

from cortexflow.api.routes import (
    access,
    approvals,
    builder,
    cases,
    documents,
    health,
    insights,
    integrations,
    operations,
    policies,
    workflows,
)

ALL_ROUTERS = (
    health.router,
    workflows.router,
    cases.router,
    insights.router,
    access.router,
    integrations.router,
    approvals.router,
    operations.router,
    documents.router,
    builder.router,
    policies.router,
)

__all__ = [
    "ALL_ROUTERS",
    "access",
    "approvals",
    "builder",
    "cases",
    "documents",
    "health",
    "insights",
    "integrations",
    "operations",
    "policies",
    "workflows",
]

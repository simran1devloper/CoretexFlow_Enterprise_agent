"""The policy module's persistence port.

Rulesets authored at runtime, through the admin screens.

Kept apart from the rulesets shipped on disk for the same reason workflow
definitions are: those are reviewed, versioned with the code and global, while
these belong to one tenant and can change between two runs of the same
workflow. Both are validated identically before anything can evaluate against
them.

A tenant ruleset may take the name of a shipped one, and then shadows it for
that tenant only -- the file is untouched and every other tenant still sees
it. That is deliberate: the common reason to want an editor at all is to move
a real threshold, and requiring a new name would mean re-pointing every
workflow that referenced the old one.

It is also the sharp edge. Policy is what authorizes, so a change here can
weaken a rule without passing a code review. Hence: every write is audited
with who made it and what moved, and the shipped version is always one call
away as a revert.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cortexflow.modules.policy.domain.models import PolicyRuleset


@runtime_checkable
class PolicyRulesetRepository(Protocol):
    """Tenant-authored rulesets. Every method is tenant-scoped."""

    async def save(self, tenant_id: str, ruleset: PolicyRuleset) -> PolicyRuleset:
        """Insert or replace a ruleset for this tenant."""
        ...

    async def get(self, tenant_id: str, name: str) -> PolicyRuleset | None: ...

    # `Sequence`, not `list`: the method named `list` shadows the builtin
    # inside this class body, so `list[...]` here is not a type.
    async def list(self, tenant_id: str) -> Sequence[PolicyRuleset]: ...

    async def delete(self, tenant_id: str, name: str) -> None:
        """Remove the override, so the shipped ruleset applies again."""
        ...

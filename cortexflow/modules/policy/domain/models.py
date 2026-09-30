"""Policy engine contracts.

The separation this module encodes is the platform's central claim:

    the agent *recommends*, the policy engine *authorizes*.

Rules are deterministic data.  An LLM never computes a threshold and never
grants itself permission.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.shared.expressions import validate_expression
from cortexflow.shared.identity import Role

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class PolicyEffect(StrEnum):
    ALLOW = "ALLOW"
    """Proceed automatically."""

    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    """Proceed only after a human with the required role approves."""

    DENY = "DENY"
    """Refuse outright; the workflow cannot take this action at all."""


_EFFECT_SEVERITY = {
    PolicyEffect.ALLOW: 0,
    PolicyEffect.REQUIRE_APPROVAL: 1,
    PolicyEffect.DENY: 2,
}


class PolicyRule(BaseModel):
    """A single deterministic rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str = ""
    when: str
    """Guard expression over the policy evaluation scope."""

    effect: PolicyEffect
    risk: RiskLevel = RiskLevel.MEDIUM
    required_roles: tuple[Role, ...] = ()
    message: str = ""
    stop_on_match: bool = False
    """Short-circuit evaluation; used for hard denials."""

    @model_validator(mode="after")
    def _validate(self) -> PolicyRule:
        validate_expression(self.when)
        return self


class DerivedFact(BaseModel):
    """A fact computed from another before any rule is evaluated.

    Real data does not agree with itself about vocabulary. One HR export calls
    a grade ``L7`` and the next calls it ``Director``, and a rule written
    against either is wrong half the time -- silently, because a comparison
    that does not match simply does not fire. The onboarding policy had
    ``facts.grade in ['L7', 'L8', 'L9', 'EXEC']`` and quietly auto-approved
    every director in a file that used titles.

    Declaring the mapping here keeps the rule a clean comparison and puts the
    messy part in one reviewable place, next to the rules that depend on it
    rather than in whichever caller happened to assemble the facts. Matching
    is case-insensitive and whitespace-normalised, because the input is a
    spreadsheet cell.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact: str
    """The name the rules will use."""

    source: str
    """The fact to read."""

    values: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Derived value -> the source values that produce it."""

    default: str = "unknown"
    """What an unrecognised source value becomes.

    Deliberately *not* one of the known bands: an unmapped title must fail the
    standard-hire rule and escalate, rather than being waved through because
    nobody had added it to the list yet.
    """

    def apply(self, value: Any) -> str:
        if value is None:
            return self.default
        needle = " ".join(str(value).split()).casefold()
        for derived, sources in self.values.items():
            if any(" ".join(s.split()).casefold() == needle for s in sources):
                return derived
        return self.default


class PolicyRuleset(BaseModel):
    """A named, versioned collection of rules for one decision point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int = 1
    description: str = ""
    default_effect: PolicyEffect = PolicyEffect.REQUIRE_APPROVAL
    """Fail *closed*: an unmatched case asks a human rather than auto-approving."""

    default_risk: RiskLevel = RiskLevel.MEDIUM
    default_required_roles: tuple[Role, ...] = ()
    derive: tuple[DerivedFact, ...] = ()
    """Facts computed from other facts before the rules run."""

    rules: tuple[PolicyRule, ...] = ()

    @model_validator(mode="after")
    def _validate_name(self) -> PolicyRuleset:
        """A ruleset's name is a handle other things hold.

        Workflow steps reference it, decisions are recorded against
        ``name@version``, and a tenant ruleset shadows a shipped one *by name*.
        An empty or whitespace name makes all three meaningless -- it produced
        the key ``@1`` and would have shadowed nothing while looking saved.

        Checked in the domain rather than in the editor, because YAML on disk
        and a form in the browser both arrive here and neither should get the
        weaker contract.
        """
        if not _NAME.match(self.name):
            raise ValueError(
                "A ruleset name must start with a letter and contain only "
                "letters, digits, underscores, dots or hyphens"
            )
        return self

    def with_derived(self, facts: dict[str, Any]) -> dict[str, Any]:
        """``facts`` plus whatever this ruleset derives from them.

        A derived name never overwrites a fact the caller supplied: a caller
        that already knows the seniority is a better source than a lookup
        table, and silently discarding it would make the ruleset the authority
        on something it only guesses at.
        """
        if not self.derive:
            return facts
        enriched = dict(facts)
        for rule in self.derive:
            if rule.fact in facts:
                continue
            enriched[rule.fact] = rule.apply(facts.get(rule.source))
        return enriched

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


class RuleMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    effect: PolicyEffect
    risk: RiskLevel
    message: str = ""


class PolicyDecision(BaseModel):
    """The authoritative answer. This -- not the agent's opinion -- drives the DAG."""

    model_config = ConfigDict(frozen=True)

    effect: PolicyEffect
    risk: RiskLevel
    ruleset: str
    required_roles: tuple[Role, ...] = ()
    matched_rules: tuple[RuleMatch, ...] = ()
    reasons: tuple[str, ...] = ()
    inputs_digest: str = ""
    """Hash of the evaluated scope, so a decision can be reproduced exactly."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.effect is PolicyEffect.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.effect is PolicyEffect.REQUIRE_APPROVAL

    @property
    def denied(self) -> bool:
        return self.effect is PolicyEffect.DENY


def most_severe(effects: list[PolicyEffect]) -> PolicyEffect:
    """Combine effects conservatively: any DENY wins, then REQUIRE_APPROVAL."""
    return max(effects, key=lambda e: _EFFECT_SEVERITY[e], default=PolicyEffect.ALLOW)

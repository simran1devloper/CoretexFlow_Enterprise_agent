"""The policy engine.

This is where the platform's central claim is enforced: an agent's output is
an *input* here, never an authorization.  Every rule is deterministic, every
evaluation is reproducible from its inputs digest, and the default when
nothing matches is to ask a human rather than to proceed.

Rules never see raw documents -- only the derived facts assembled by the
caller -- so a prompt injection in a receipt cannot reach the decision.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.policy.domain.models import (
    PolicyDecision,
    PolicyEffect,
    PolicyRuleset,
    RuleMatch,
    most_severe,
)
from cortexflow.shared.expressions import evaluate_condition
from cortexflow.shared.identity import Role
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.observability.metrics import get_metrics
from cortexflow.shared.observability.telemetry import span

logger = get_logger(__name__)


class PolicyEngine:
    """Evaluates rulesets against a fact scope."""

    def __init__(self, rulesets: dict[str, PolicyRuleset]) -> None:
        self._rulesets = rulesets
        self._overrides: dict[tuple[str, str], PolicyRuleset] = {}
        self._metrics = get_metrics()

    @property
    def rulesets(self) -> dict[str, PolicyRuleset]:
        return dict(self._rulesets)

    # -- tenant overrides --------------------------------------------------
    # Held in memory rather than read per evaluation, because `evaluate` is
    # synchronous and called on the hot path of every workflow step. The
    # policy service writes through: it persists and registers in one act, and
    # warms this on start. The same arrangement the builder uses for
    # definitions authored at runtime.

    def register(self, tenant_id: str, ruleset: PolicyRuleset) -> None:
        """Make a tenant's ruleset take effect now."""
        self._overrides[(tenant_id, ruleset.name)] = ruleset

    def unregister(self, tenant_id: str, name: str) -> None:
        """Drop an override, so the shipped ruleset applies again."""
        self._overrides.pop((tenant_id, name), None)

    def overrides_for(self, tenant_id: str) -> dict[str, PolicyRuleset]:
        return {
            name: ruleset
            for (tid, name), ruleset in self._overrides.items()
            if tid == tenant_id
        }

    def rulesets_for(self, tenant_id: str) -> dict[str, PolicyRuleset]:
        """Everything this tenant would evaluate against: shipped, then its own.

        What the builder must offer and check against. Offering only the
        shipped ones would let someone author a ruleset and then be told, by
        the workflow builder, that it does not exist.
        """
        return {**self._rulesets, **self.overrides_for(tenant_id)}

    def get(self, name: str, tenant_id: str | None = None) -> PolicyRuleset:
        """A tenant's own ruleset if it has one, otherwise the shipped one."""
        if tenant_id is not None:
            override = self._overrides.get((tenant_id, name))
            if override is not None:
                return override
        try:
            return self._rulesets[name]
        except KeyError as exc:
            from cortexflow.shared.errors import NotFoundError

            raise NotFoundError("Unknown policy ruleset", ruleset=name) from exc

    def evaluate(
        self,
        ruleset_name: str,
        scope: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> PolicyDecision:
        """Evaluate every rule and combine the matches conservatively.

        ``scope`` is the evaluation namespace, not the facts: rules are written
        as ``facts.amount``, so callers pass ``{"facts": {...}}``. The
        parameter was named ``facts`` and cost an afternoon.
        """
        ruleset = self.get(ruleset_name, tenant_id)

        # Derived facts first, so a rule comparing `facts.seniority` sees it
        # however the caller assembled the rest -- a row mapping, a workflow
        # expression or a direct call.
        if ruleset.derive and isinstance(scope.get("facts"), dict):
            scope = {**scope, "facts": ruleset.with_derived(scope["facts"])}

        facts = scope
        with span("policy.evaluate", **{"policy.ruleset": ruleset.key}):
            matches: list[RuleMatch] = []
            reasons: list[str] = []
            required_roles: set[Role] = set()

            for rule in ruleset.rules:
                if not self._matches(rule.when, facts, rule_id=rule.id, ruleset=ruleset.key):
                    continue
                matches.append(
                    RuleMatch(
                        rule_id=rule.id,
                        effect=rule.effect,
                        risk=rule.risk,
                        message=rule.message or rule.description,
                    )
                )
                if rule.message or rule.description:
                    reasons.append(rule.message or rule.description)
                required_roles.update(rule.required_roles)
                if rule.stop_on_match:
                    break

            if matches:
                effect = most_severe([m.effect for m in matches])
                risk = max((m.risk for m in matches), key=lambda r: r.rank)
            else:
                # Fail closed. An unanticipated case is exactly the case a
                # human should look at.
                effect = ruleset.default_effect
                risk = ruleset.default_risk
                required_roles.update(ruleset.default_required_roles)
                reasons.append("No rule matched; applying the ruleset default.")

            if effect is not PolicyEffect.ALLOW and not required_roles:
                required_roles.update(ruleset.default_required_roles)

            decision = PolicyDecision(
                effect=effect,
                risk=risk,
                ruleset=ruleset.key,
                required_roles=tuple(sorted(required_roles)),
                matched_rules=tuple(matches),
                reasons=tuple(reasons),
                inputs_digest=digest(facts),
                metadata={"rules_evaluated": len(ruleset.rules)},
            )

        self._metrics.count(
            self._metrics.policy_decisions,
            effect=str(effect),
            ruleset=ruleset.name,
            risk=str(risk),
        )
        logger.info(
            "policy evaluated",
            extra={
                "ruleset": ruleset.key,
                "effect": str(effect),
                "risk": str(risk),
                "matched_rules": [m.rule_id for m in matches],
            },
        )
        return decision

    @staticmethod
    def _matches(expression: str, facts: dict[str, Any], **context: str) -> bool:
        """A rule that cannot be evaluated does not match -- and is loud about it.

        Silently treating an erroring rule as a match would make policy depend
        on data shape; treating it as a non-match plus an error log keeps the
        default-effect fallback in charge.
        """
        try:
            return evaluate_condition(expression, facts)
        except Exception as exc:
            logger.error(
                "policy rule failed to evaluate",
                extra={**context, "expression": expression, "error": str(exc)},
            )
            return False


def digest(facts: dict[str, Any]) -> str:
    """Stable hash of the evaluated facts, so a decision can be reproduced."""
    canonical = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def risk_from_signals(
    *,
    amount: float = 0.0,
    high_value_threshold: float = 50_000,
    confidence: float | None = None,
    confidence_floor: float = 0.85,
    data_complete: bool = True,
    validation_issues: int = 0,
    side_effect_irreversible: bool = False,
) -> RiskLevel:
    """Combine independent signals into a risk level.

    Deliberately *not* a function of model confidence alone.  A model reporting
    0.99 is one input among several; data completeness, deterministic
    validation results and the reversibility of the action all carry weight,
    because a confident model can still be confidently wrong.
    """
    score = 0
    if amount >= high_value_threshold:
        score += 2
    elif amount >= high_value_threshold / 5:
        score += 1
    if confidence is not None and confidence < confidence_floor:
        score += 1
    if confidence is not None and confidence < confidence_floor - 0.15:
        score += 1
    if not data_complete:
        score += 1
    score += min(validation_issues, 2)
    if side_effect_irreversible:
        score += 1

    if score >= 5:
        return RiskLevel.CRITICAL
    if score >= 3:
        return RiskLevel.HIGH
    if score >= 1:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW

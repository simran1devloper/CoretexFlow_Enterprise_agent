"""Authoring policy rulesets at runtime.

The counterpart to the workflow builder, for the other half of ADR 0001. A
workflow says what happens; a ruleset says what is *allowed* to happen, and
until now the second could only be changed by editing a YAML file in the
repository and shipping it. For a platform whose whole claim is that policy --
not the model -- authorizes, an administrator having no way to move a
threshold is a gap in the product, not a safety feature: it pushes the change
into the workflow, where an agent's output ends up standing in for a rule.

So this exists, and everything here is arranged around the fact that it is the
sharpest edit in the system:

* A tenant ruleset **shadows** a shipped one of the same name, for that tenant
  only. The file on disk is untouched and every other tenant still gets it, so
  a revert is "delete the override" rather than "remember what it used to say".
* **Every write is audited with the diff** -- which rules appeared, vanished
  or changed, and how the defaults moved. A rule that was weakened leaves a
  record naming who weakened it.
* Validation is the *same* validation a shipped ruleset gets, because
  ``PolicyRuleset`` validates its own guard expressions. Authoring in the
  browser buys no weaker contract than authoring in YAML.
* Lenient parsing stops at :meth:`validate`, exactly as it does in the
  builder. :meth:`save` re-checks rather than trusting how the body parsed.
"""

from __future__ import annotations

import re
from typing import Any

from cortexflow.modules.audit.application.trail import AuditTrail
from cortexflow.modules.audit.domain.models import AuditAction
from cortexflow.modules.policy.application.engine import PolicyEngine
from cortexflow.modules.policy.domain.models import PolicyEffect, PolicyRuleset
from cortexflow.modules.policy.ports.repositories import PolicyRulesetRepository
from cortexflow.shared.errors import NotFoundError, ValidationError
from cortexflow.shared.identity import Principal
from cortexflow.shared.observability.logging import get_logger
from cortexflow.shared.security.rbac import Permission, require_permission

logger = get_logger(__name__)

# Declared at module scope: this service's own ``list`` method shadows the
# builtin inside the class body, so annotations written there cannot use it.
StringList = list[str]
Json = dict[str, Any]

# How permissive each effect is. Named here rather than imported from the
# domain's private table, so this module's reading of "weaker" is its own and
# visible next to the code that uses it.
_PERMISSIVENESS = {
    PolicyEffect.DENY: 0,
    PolicyEffect.REQUIRE_APPROVAL: 1,
    PolicyEffect.ALLOW: 2,
}


class PolicyService:
    """Read, author, revert and try out the rules that authorize work."""

    def __init__(
        self,
        *,
        engine: PolicyEngine,
        repository: PolicyRulesetRepository,
        audit: AuditTrail,
    ) -> None:
        self._engine = engine
        self._repository = repository
        self._audit = audit

    # -- reading -----------------------------------------------------------
    def list(self, principal: Principal) -> list[Json]:
        """Every ruleset this tenant would actually evaluate against.

        Shipped and tenant-authored in one list rather than two, because the
        question an administrator has is "what is in force?", and answering it
        with two lists and a precedence rule to apply in their head is how
        someone edits the copy that is being shadowed.
        """
        require_permission(principal, Permission.DEFINITION_READ)
        overrides = self._engine.overrides_for(principal.tenant_id)
        shipped = self._engine.rulesets

        rows = [
            self._summarise(ruleset, shipped.get(name), source="tenant")
            for name, ruleset in overrides.items()
        ]
        rows += [
            self._summarise(ruleset, None, source="shipped")
            for name, ruleset in shipped.items()
            if name not in overrides
        ]
        return sorted(rows, key=lambda r: str(r["name"]))

    def get(self, principal: Principal, name: str) -> Json:
        """One ruleset, in full, with whatever it shadows alongside it.

        Both halves on purpose. The editor needs the effective rules to edit,
        and the reader needs the shipped ones to answer "what did we change?"
        without leaving the screen -- and the revert button needs to know
        whether there is anything to revert *to*.
        """
        require_permission(principal, Permission.DEFINITION_READ)
        effective = self._engine.get(name, principal.tenant_id)
        shipped = self._engine.rulesets.get(name)
        override = self._engine.overrides_for(principal.tenant_id).get(name)

        return {
            "source": "tenant" if override is not None else "shipped",
            "ruleset": effective.model_dump(mode="json"),
            "shipped": shipped.model_dump(mode="json") if shipped else None,
            "can_revert": override is not None and shipped is not None,
            "facts": facts_required(effective),
            "changes": diff(shipped, effective) if override is not None else {},
        }

    # -- authoring ---------------------------------------------------------
    def validate(self, principal: Principal, payload: Json) -> Json:
        """Check a draft without saving it, for live feedback in the editor.

        An incomplete draft is the normal case here, not an error: this is
        what the editor calls on every keystroke. So a draft that cannot yet
        be a ruleset comes back as a list of problems to read, in the shape a
        successful validation has, rather than as a 422 whose whole content is
        "unprocessable".
        """
        require_permission(principal, Permission.DEFINITION_READ)

        ruleset, problems = _parse(payload)
        if ruleset is None:
            return {
                "valid": False,
                "problems": problems,
                "name": str(payload.get("name") or ""),
                "rule_count": len(payload.get("rules") or []),
                "facts": [],
                "warnings": [],
                "changes": {},
            }

        shipped = self._engine.rulesets.get(ruleset.name)
        return {
            "valid": True,
            "problems": [],
            "name": ruleset.name,
            "rule_count": len(ruleset.rules),
            "facts": facts_required(ruleset),
            "warnings": warnings_for(ruleset, shipped),
            "changes": diff(shipped, ruleset),
        }

    def try_out(self, principal: Principal, payload: Json, facts: Json) -> Json:
        """What this draft would decide about one case.

        Evaluated by the *real* engine against an unregistered copy, never by
        a second implementation that predicts it. A predictor drifts from the
        engine and people act on the prediction -- the same reason workflow
        simulation is a mode of the executor rather than a model of it.

        Nothing is saved and nothing is registered, so trying a draft out
        cannot change what a concurrent workflow is being judged against.
        """
        require_permission(principal, Permission.DEFINITION_READ)

        ruleset, problems = _parse(payload)
        if ruleset is None:
            raise ValidationError("That is not a ruleset yet", problems=problems)

        decision = PolicyEngine({ruleset.name: ruleset}).evaluate(
            ruleset.name, {"facts": facts}
        )
        unsupplied = sorted(set(facts_required(ruleset)) - set(facts))
        return {
            "decision": decision.model_dump(mode="json"),
            "unsupplied_facts": unsupplied,
            "warnings": [
                f"No value given for {', '.join(unsupplied)}. Rules reading "
                "them cannot evaluate, so this case falls through to the "
                "ruleset default -- which is what a real case missing those "
                "fields would also do."
            ]
            if unsupplied
            else [],
        }

    async def save(self, principal: Principal, payload: Json) -> PolicyRuleset:
        """Publish a ruleset for this tenant, and make it take effect now.

        Requires ``definition:manage`` -- the permission that lets someone
        change what runs -- because changing what authorizes is the larger
        version of the same power, not a lesser one.
        """
        require_permission(principal, Permission.DEFINITION_MANAGE)

        # The strict path re-checks rather than trusting how the body was
        # parsed. Only `validate` loosens the model, and a loosened draft must
        # not reach a saved ruleset by another route.
        ruleset, problems = _parse(payload)
        if ruleset is None:
            raise ValidationError(
                "The ruleset is not valid",
                problems=problems,
                ruleset=str(payload.get("name") or ""),
            )

        previous = self._engine.overrides_for(principal.tenant_id).get(ruleset.name)
        shipped = self._engine.rulesets.get(ruleset.name)
        baseline = previous or shipped

        # Version up rather than in place, so the audit trail's `ruleset.key`
        # stays a stable handle on a specific set of rules: a decision that
        # recorded `expense_reimbursement@3` can still be read back as the
        # rules that actually produced it.
        if baseline is not None and ruleset.version <= baseline.version:
            ruleset = ruleset.model_copy(update={"version": baseline.version + 1})

        saved = await self._repository.save(principal.tenant_id, ruleset)
        self._engine.register(principal.tenant_id, saved)

        changes = diff(baseline, saved)
        logger.info(
            "policy ruleset published",
            extra={
                "ruleset": saved.key,
                "tenant_id": principal.tenant_id,
                "rules": len(saved.rules),
                "shadows_shipped": shipped is not None,
            },
        )
        await self._audit.record(
            tenant_id=principal.tenant_id,
            action=AuditAction.POLICY_PUBLISHED,
            actor=principal.subject,
            actor_roles=tuple(str(r) for r in principal.roles),
            summary=(
                f"Published policy ruleset {saved.key}"
                + (" (shadowing the shipped one)" if shipped is not None else "")
            ),
            metadata={
                "ruleset": saved.key,
                "rules": len(saved.rules),
                "shadows_shipped": shipped is not None,
                "replaces": baseline.key if baseline else None,
                "changes": changes,
            },
        )
        return saved

    async def delete(self, principal: Principal, name: str) -> Json:
        """Drop this tenant's ruleset.

        Where it shadowed a shipped one this is a revert and the shipped rules
        apply again from the next evaluation; where it did not, the ruleset is
        simply gone and any workflow naming it will fail to resolve it. The
        response says which of the two happened, because they are very
        different things to have just done.
        """
        require_permission(principal, Permission.DEFINITION_MANAGE)

        override = self._engine.overrides_for(principal.tenant_id).get(name)
        if override is None:
            raise NotFoundError("This tenant has no ruleset by that name", ruleset=name)
        shipped = self._engine.rulesets.get(name)

        await self._repository.delete(principal.tenant_id, name)
        self._engine.unregister(principal.tenant_id, name)

        await self._audit.record(
            tenant_id=principal.tenant_id,
            action=AuditAction.POLICY_WITHDRAWN,
            actor=principal.subject,
            actor_roles=tuple(str(r) for r in principal.roles),
            summary=(
                f"Reverted policy ruleset {name} to the shipped version"
                if shipped is not None
                else f"Withdrew policy ruleset {name}"
            ),
            metadata={
                "ruleset": override.key,
                "reverted_to": shipped.key if shipped else None,
                "changes": diff(override, shipped),
            },
        )
        return {
            "reverted": shipped is not None,
            "now_in_force": shipped.model_dump(mode="json") if shipped else None,
        }

    async def restore(self, tenant_ids: StringList) -> int:
        """Re-register saved rulesets into the engine at startup.

        The engine holds overrides in memory because ``evaluate`` is
        synchronous and on the hot path of every workflow step; this is what
        makes that safe across a restart. Same arrangement as the builder's.
        """
        restored = 0
        for tenant_id in tenant_ids:
            for ruleset in await self._repository.list(tenant_id):
                self._engine.register(tenant_id, ruleset)
                restored += 1
        return restored

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _summarise(
        ruleset: PolicyRuleset, shipped: PolicyRuleset | None, *, source: str
    ) -> Json:
        return {
            "name": ruleset.name,
            "version": ruleset.version,
            "description": ruleset.description,
            "source": source,
            "shadows_shipped": shipped is not None,
            "default_effect": str(ruleset.default_effect),
            "default_risk": str(ruleset.default_risk),
            "rule_count": len(ruleset.rules),
            "facts": facts_required(ruleset),
        }


def _parse(payload: Json) -> tuple[PolicyRuleset | None, StringList]:
    """A ruleset, or the reasons it is not one yet -- phrased for a person.

    Pydantic's own message is accurate and unreadable at the point of use
    (``rules.2.when: Value error, ...``), so each error is rewritten with the
    rule it belongs to named the way the editor names it.
    """
    try:
        return PolicyRuleset.model_validate(payload), []
    except Exception as exc:
        return None, _problems(payload, exc)


def _problems(payload: Json, exc: Exception) -> StringList:
    rules = payload.get("rules")
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        # A guard that does not parse is raised by `validate_expression` as one
        # of our own errors, which Pydantic passes straight through without a
        # location. So the rule is found by re-validating each one alone --
        # slower, and the only way to answer "which rule?" for the person
        # whose editor is showing twelve of them.
        return _attributed(rules, exc)
    problems: StringList = []
    for error in errors():
        location = list(error.get("loc", ()))
        message = str(error.get("msg", "is not valid"))
        where = "This ruleset"
        if len(location) >= 2 and location[0] == "rules":
            index = location[1]
            identifier = ""
            if isinstance(rules, list) and isinstance(index, int) and index < len(rules):
                entry = rules[index]
                identifier = str(entry.get("id") or "") if isinstance(entry, dict) else ""
            where = f"Rule '{identifier}'" if identifier else f"Rule {index}"
            if len(location) > 2:
                where += f" field '{location[-1]}'"
        elif location:
            where = f"Field '{location[-1]}'"
        problems.append(f"{where}: {message}")
    return problems or [str(exc)]


def _attributed(rules: Any, exc: Exception) -> StringList:
    """Name the rule that broke, by validating each one on its own."""
    from cortexflow.modules.policy.domain.models import PolicyRule

    message = getattr(exc, "message", None) or str(exc)
    detail = ""
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details.get("expression"):
        detail = f" ({details['expression']})"

    if isinstance(rules, list):
        culprits: StringList = []
        for index, entry in enumerate(rules):
            if not isinstance(entry, dict):
                continue
            try:
                PolicyRule.model_validate(entry)
            except Exception as rule_exc:
                identifier = str(entry.get("id") or "") or str(index)
                reason = getattr(rule_exc, "message", None) or "is not valid"
                culprits.append(f"Rule '{identifier}': {reason}{detail}")
        if culprits:
            return culprits

    return [f"This ruleset: {message}{detail}"]


def facts_required(ruleset: PolicyRuleset) -> StringList:
    """The ``facts.x`` names this ruleset reads.

    What a workflow must supply for the rules to mean anything, which is the
    one thing an editor cannot infer from the rules by looking at them.
    Derived facts are listed too: a caller may supply either the derived name
    or the source it is computed from.
    """
    names: set[str] = set()
    for rule in ruleset.rules:
        names.update(re.findall(r"facts\.([a-zA-Z_][a-zA-Z0-9_]*)", rule.when))
    for derived in ruleset.derive:
        names.add(derived.source)
        names.discard(derived.fact)
    return sorted(names)


def warnings_for(ruleset: PolicyRuleset, shipped: PolicyRuleset | None) -> StringList:
    """Things worth saying, none of them worth refusing to save over.

    All three are about a ruleset that would quietly authorize more than the
    author thinks -- the failure this whole module has to be careful about.
    """
    warnings: StringList = []
    if ruleset.default_effect is PolicyEffect.ALLOW:
        warnings.append(
            "The default is ALLOW, so any case no rule matches proceeds "
            "automatically. The shipped rulesets ask a human instead."
        )
    if not ruleset.rules:
        warnings.append(
            "This ruleset has no rules, so every case gets the default."
        )

    # Specifically a *permissive* short-circuit. Every shipped ruleset opens
    # with stop-on-match denials and then a dozen escalations, so warning
    # about anything after any stop-on-match rule fires on every ruleset there
    # is -- and a warning that is always on is one nobody reads. The hazard is
    # the rule that allows and then skips the checks below it.
    short_circuits = [
        rule.id
        for index, rule in enumerate(ruleset.rules[:-1])
        if rule.stop_on_match and rule.effect is PolicyEffect.ALLOW
    ]
    if short_circuits:
        warnings.append(
            f"{', '.join(short_circuits)} allow a case and stop evaluating, so "
            "no rule below can deny it or send it for approval."
        )

    needs_role = [
        rule.id
        for rule in ruleset.rules
        if rule.effect is PolicyEffect.REQUIRE_APPROVAL and not rule.required_roles
    ]
    if needs_role and not ruleset.default_required_roles:
        warnings.append(
            f"{', '.join(needs_role)} require approval but name no role, and "
            "the ruleset has no default -- anyone who can approve will do."
        )

    if shipped is not None:
        weakened = _weakened(shipped, ruleset)
        if weakened:
            warnings.append(
                "Compared with the shipped version this allows more: "
                + "; ".join(weakened)
            )
    return warnings


def _weakened(before: PolicyRuleset, after: PolicyRuleset) -> StringList:
    """Where the new version authorizes something the old one would not.

    Only in the permissive direction. A rule made *stricter* is a normal edit
    that needs no commentary; a rule made looser is the thing a reviewer of
    this diff is actually looking for.
    """
    notes: StringList = []
    old_rules = {rule.id: rule for rule in before.rules}
    new_rules = {rule.id: rule for rule in after.rules}

    for rule_id, rule in new_rules.items():
        previous = old_rules.get(rule_id)
        if previous is None:
            continue
        if _PERMISSIVENESS[rule.effect] > _PERMISSIVENESS[previous.effect]:
            notes.append(f"'{rule_id}' is {rule.effect} where it was {previous.effect}")

    # A deleted rule only matters here if it was holding something back. One
    # that said ALLOW was never the reason a case stopped.
    for rule_id in sorted(old_rules.keys() - new_rules.keys()):
        if old_rules[rule_id].effect is not PolicyEffect.ALLOW:
            notes.append(
                f"'{rule_id}' has been removed, and it said "
                f"{old_rules[rule_id].effect}"
            )

    if _PERMISSIVENESS[after.default_effect] > _PERMISSIVENESS[before.default_effect]:
        notes.append(
            f"the default is {after.default_effect} where it was "
            f"{before.default_effect}"
        )
    return notes


def diff(before: PolicyRuleset | None, after: PolicyRuleset | None) -> Json:
    """What moved between two versions, rule by rule.

    Recorded in the audit trail on every write, because a ruleset is what
    authorizes: "someone edited the expense policy" is not a useful record,
    and "someone changed auto_approve_small from 5000 to 50000" is.
    """
    if before is None or after is None:
        return {
            "added": [r.id for r in after.rules] if after else [],
            "removed": [r.id for r in before.rules] if before else [],
            "changed": {},
            "defaults": {},
        }

    old = {rule.id: rule for rule in before.rules}
    new = {rule.id: rule for rule in after.rules}

    changed: dict[str, Json] = {}
    for rule_id in old.keys() & new.keys():
        fields = {
            field: {"from": getattr(old[rule_id], field), "to": getattr(new[rule_id], field)}
            for field in ("when", "effect", "risk", "required_roles", "stop_on_match")
            if getattr(old[rule_id], field) != getattr(new[rule_id], field)
        }
        if fields:
            changed[rule_id] = _jsonable(fields)

    defaults = {
        field: {"from": str(getattr(before, field)), "to": str(getattr(after, field))}
        for field in ("default_effect", "default_risk", "default_required_roles")
        if getattr(before, field) != getattr(after, field)
    }
    return {
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "changed": changed,
        "defaults": defaults,
    }


def _jsonable(fields: dict[str, Json]) -> Json:
    """Audit metadata has to survive a JSON round-trip; enums and tuples do not."""
    return {
        name: {
            side: (
                [str(v) for v in value]
                if isinstance(value, tuple | list)
                else (str(value) if not isinstance(value, bool | int | float) else value)
            )
            for side, value in change.items()
        }
        for name, change in fields.items()
    }

"""Loading policy rulesets from YAML.

Rulesets are validated at startup, not first use: a malformed rule should fail
a deployment, not a payment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cortexflow.modules.policy.domain.models import PolicyRuleset
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


def load_ruleset(path: Path) -> PolicyRuleset:
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValidationError("Malformed policy YAML", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise ValidationError("Policy file must contain a mapping", path=str(path))
    try:
        return PolicyRuleset.model_validate(raw)
    except Exception as exc:
        raise ValidationError(
            "Invalid policy ruleset", path=str(path), error=str(exc)
        ) from exc


def load_rulesets(directory: Path) -> dict[str, PolicyRuleset]:
    """Load every ``*.yaml`` ruleset in ``directory``, keyed by name."""
    rulesets: dict[str, PolicyRuleset] = {}
    if not directory.exists():
        logger.warning("policy directory missing", extra={"path": str(directory)})
        return rulesets

    for path in sorted(directory.rglob("*.yaml")):
        ruleset = load_ruleset(path)
        if ruleset.name in rulesets:
            raise ValidationError("Duplicate policy ruleset name", name=ruleset.name)
        rulesets[ruleset.name] = ruleset
        logger.info(
            "policy ruleset loaded",
            extra={"ruleset": ruleset.key, "rules": len(ruleset.rules)},
        )
    return rulesets

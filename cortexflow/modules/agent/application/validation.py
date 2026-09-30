"""Validation agent: anomaly detection over a document.

Scoped narrowly on purpose. Arithmetic, exact matches and duplicate lookups
are done in code by tools; this agent only judges what genuinely requires
reading the document.
"""

from __future__ import annotations

import json

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.application.prompts import VALIDATION
from cortexflow.modules.agent.application.schemas import ValidationOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


class ValidationAgent(BaseAgent):
    name = "validation_agent"
    description = "Inspects a document for anomalies that warrant human attention."

    async def run(self, context: AgentContext) -> AgentResult:
        subject = context.inputs.get("data") or {}
        document_text = str(context.inputs.get("text", ""))[:8_000]
        checks = context.inputs.get("checks") or []

        parsed, usage = await self.think(
            context,
            system_prompt=VALIDATION.system,
            user_content=(
                f"Checks requested: {json.dumps(list(checks))}\n\n"
                f"<data>\n{json.dumps(subject, default=str)[:6000]}\n</data>\n\n"
                f"<document>\n{document_text}\n</document>"
            ),
            output_model=ValidationOutput,
            prompt_version=VALIDATION.version,
        )

        issues = [i.strip() for i in parsed.issues if i.strip()][:10]
        return AgentResult.success(
            output={
                "valid": parsed.valid and not issues,
                "issues": issues,
                "issue_count": len(issues),
            },
            confidence=round(parsed.confidence, 3),
            reason=parsed.reason,
            usage=usage,
        )

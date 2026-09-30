"""Decision agent: contextual recommendation, never authorization.

The agent sees the case and recommends; the policy engine then decides. This
class returns a recommendation plus its risk factors, and nothing it returns
can cause an action on its own.
"""

from __future__ import annotations

import json

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.application.prompts import DECISION
from cortexflow.modules.agent.application.schemas import DecisionOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

VALID_RECOMMENDATIONS = frozenset({"APPROVE", "REJECT", "MANUAL_REVIEW"})


class DecisionAgent(BaseAgent):
    name = "decision_agent"
    description = "Recommends how a case should be handled, given its full context."

    async def run(self, context: AgentContext) -> AgentResult:
        case = context.inputs.get("case") or {}
        guidance = context.inputs.get("guidance", "")
        history = context.inputs.get("history") or []

        parsed, usage = await self.think(
            context,
            system_prompt=DECISION.system,
            user_content=(
                f"{guidance}\n\n"
                f"<data name=\"case\">\n{json.dumps(case, default=str)[:8000]}\n</data>\n\n"
                f"<data name=\"history\">\n{json.dumps(history, default=str)[:3000]}\n</data>"
            ),
            output_model=DecisionOutput,
            prompt_version=DECISION.version,
        )

        recommendation = parsed.recommendation.strip().upper()
        if recommendation not in VALID_RECOMMENDATIONS:
            # An off-contract answer is not a failure -- it is a reason to
            # involve a human, which is the safe direction to fall.
            logger.warning(
                "decision agent returned an unrecognised recommendation",
                extra={"value": recommendation[:64], "workflow_id": context.workflow_id},
            )
            recommendation = "MANUAL_REVIEW"

        return AgentResult.success(
            output={
                "recommendation": recommendation,
                "risk_factors": parsed.risk_factors[:10],
                "reason": parsed.reason,
            },
            confidence=round(parsed.confidence, 3),
            reason=parsed.reason,
            usage=usage,
        )

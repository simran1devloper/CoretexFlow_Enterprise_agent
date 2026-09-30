"""Reporting agent: renders a completed workflow as a business summary."""

from __future__ import annotations

import json

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.application.prompts import REPORTING
from cortexflow.modules.agent.application.schemas import ReportOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult
from cortexflow.shared.observability.redaction import redact


class ReportingAgent(BaseAgent):
    name = "reporting_agent"
    description = "Summarises a completed workflow for business readers."

    async def run(self, context: AgentContext) -> AgentResult:
        # Reports circulate widely, so the agent is shown a redacted view even
        # though it is inside the trust boundary.
        payload = redact(
            {
                "workflow_type": context.workflow_type,
                "subject": context.inputs.get("subject", {}),
                "outcome": context.inputs.get("outcome", {}),
                "steps": context.inputs.get("steps", {}),
            }
        )

        parsed, usage = await self.think(
            context,
            system_prompt=REPORTING.system,
            user_content=(
                f"Produce a summary of this completed workflow.\n\n"
                f"<data>\n{json.dumps(payload, default=str)[:8000]}\n</data>"
            ),
            output_model=ReportOutput,
            prompt_version=REPORTING.version,
        )

        return AgentResult.success(
            output={
                "title": parsed.title or f"{context.workflow_type} summary",
                "summary": parsed.summary,
                "sections": [s.model_dump() for s in parsed.sections],
                "highlights": parsed.highlights[:8],
            },
            confidence=0.9,
            usage=usage,
        )

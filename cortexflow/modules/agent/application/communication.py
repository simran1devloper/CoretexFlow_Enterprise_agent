"""Communication agent: drafts employee-facing notifications.

It only drafts. Delivery is the notification service's job, and the draft
passes through redaction before it is sent.
"""

from __future__ import annotations

import json

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.application.prompts import COMMUNICATION
from cortexflow.modules.agent.application.schemas import CommunicationOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult
from cortexflow.shared.observability.redaction import redact


class CommunicationAgent(BaseAgent):
    name = "communication_agent"
    description = "Drafts notifications to employees about their requests."

    async def run(self, context: AgentContext) -> AgentResult:
        context_payload = redact(
            {
                "outcome": context.inputs.get("outcome", ""),
                "workflow_type": context.workflow_type,
                "details": context.inputs.get("details", {}),
            }
        )

        parsed, usage = await self.think(
            context,
            system_prompt=COMMUNICATION.system,
            user_content=(
                "Draft a notification for the employee.\n\n"
                f"<data>\n{json.dumps(context_payload, default=str)[:4000]}\n</data>"
            ),
            output_model=CommunicationOutput,
            prompt_version=COMMUNICATION.version,
        )

        return AgentResult.success(
            output={
                "subject": parsed.subject[:200],
                "body": parsed.body[:4000],
                "tone": parsed.tone,
            },
            confidence=0.9,
            usage=usage,
        )

"""The triage agent -- the case for the Semantic Kernel runtime.

Most agents in this platform take a fixed input and return a judgement, which
one structured completion handles perfectly well. Triage is different: *which*
facts a case needs is itself a contextual decision. A claim from an unfamiliar
cost centre needs a budget lookup; one from a known employee at a routine
amount may need nothing further.

Encoding that as fixed workflow steps means always paying for every lookup.
Encoding it as agent reasoning with tool access means paying for the ones that
matter -- which is what Semantic Kernel's automatic function calling provides.

The controls are unchanged. Tools come from the registry, so every call is
authorized, validated, rate-limited, idempotent and audited; the step's
allowlist decides what exists; and the findings feed the policy engine, which
is still the only thing that authorizes an action.
"""

from __future__ import annotations

import json

from cortexflow.modules.agent.adapters.kernel.agent import KernelAgent
from cortexflow.modules.agent.application.prompts import TRIAGE
from cortexflow.modules.agent.application.schemas import TriageOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult, AgentStatus
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

MAX_CASE_CHARS = 6000


class TriageAgent(KernelAgent):
    name = "triage_agent"
    description = (
        "Gathers the facts a case needs by calling enterprise tools, "
        "then reports structured findings."
    )

    async def run(self, context: AgentContext) -> AgentResult:
        case = context.inputs.get("case") or {}
        guidance = str(context.inputs.get("guidance", ""))

        if not context.allowed_tools:
            # Triage without tools is just paraphrasing. Say so rather than
            # returning confident-looking findings gathered from nothing.
            logger.warning(
                "triage agent has no tools; escalating",
                extra={"workflow_id": context.workflow_id, "step_id": context.step_id},
            )
            return self._needs_human(
                "No enterprise tools were granted to this step, so no facts "
                "could be gathered."
            )

        parsed, usage, session = await self.reason(
            context,
            system_prompt=TRIAGE.system,
            user_content=(
                f"{guidance}\n\n"
                f"<data name=\"case\">\n"
                f"{json.dumps(case, default=str)[:MAX_CASE_CHARS]}\n"
                f"</data>"
            ),
            output_model=TriageOutput,
            prompt_version=TRIAGE.version,
        )

        invocations = session.plugin.invocations if session.plugin else ()
        failed = [i.tool for i in invocations if not i.succeeded]

        # The model's own `needs_human` is one signal; a failed lookup is a
        # fact. Trust the fact.
        needs_human = parsed.needs_human or bool(failed)
        concerns = list(parsed.concerns)
        concerns.extend(f"Lookup failed: {tool}" for tool in failed)

        confidence = parsed.confidence
        if failed:
            confidence = min(confidence, 0.5)

        return AgentResult(
            status=AgentStatus.NEEDS_HUMAN if needs_human else AgentStatus.SUCCESS,
            output={
                "findings": parsed.findings,
                "assessment": parsed.assessment,
                "concerns": concerns,
                "concern_count": len(concerns),
                "needs_human": needs_human,
                # The policy ruleset compares a recommendation string, so the
                # agent speaks that vocabulary rather than leaving the mapping
                # to a fragile expression in the workflow YAML.
                "recommendation": "MANUAL_REVIEW" if needs_human else "APPROVE",
                "tools_called": [i.tool for i in invocations],
                "tool_call_count": len(invocations),
            },
            confidence=round(confidence, 3),
            reason=parsed.assessment,
            tool_invocations=tuple(invocations),
            usage=usage,
        )

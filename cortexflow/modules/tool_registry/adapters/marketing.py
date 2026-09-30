"""Marketing tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.tool_registry.adapters.systems.fake_backends import MarketingSystem
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.identity import Department, Principal, Role


class GetCampaignTool(Tool):
    class Args(BaseModel):
        campaign_id: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="marketing.get_campaign",
        domain=Department.MARKETING,
        description="Read a campaign record.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: MarketingSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        return ToolResult.ok(
            self.name, await self._system.get_campaign(call.arguments["campaign_id"])
        )


class GetCampaignMetricsTool(Tool):
    """Metrics are computed by the CRM, not inferred by a model."""

    class Args(BaseModel):
        campaign_id: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="marketing.get_campaign_metrics",
        domain=Department.MARKETING,
        description="Read computed performance metrics for a campaign.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: MarketingSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        return ToolResult.ok(
            self.name, await self._system.get_campaign_metrics(call.arguments["campaign_id"])
        )


class UpdateCampaignBudgetTool(Tool):
    class Args(BaseModel):
        campaign_id: str = Field(min_length=1)
        budget: float = Field(gt=0)
        reason: str = ""

    input_model = Args
    metadata = ToolMetadata(
        name="marketing.update_campaign_budget",
        domain=Department.MARKETING,
        description="Change a campaign's budget allocation.",
        side_effect=SideEffect.WRITE,
        risk=RiskLevel.HIGH,
        allowed_roles=(Role.MARKETING_MANAGER, Role.SERVICE_AGENT),
        exposed_to_agents=False,
    )

    def __init__(self, system: MarketingSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        updated = await self._system.update_campaign(
            call.arguments["campaign_id"],
            {"budget": call.arguments["budget"], "budget_change_reason":
             call.arguments.get("reason", "")},
        )
        return ToolResult.ok(self.name, updated)


def build_marketing_tools(system: MarketingSystem) -> list[Tool]:
    return [
        GetCampaignTool(system),
        GetCampaignMetricsTool(system),
        UpdateCampaignBudgetTool(system),
    ]

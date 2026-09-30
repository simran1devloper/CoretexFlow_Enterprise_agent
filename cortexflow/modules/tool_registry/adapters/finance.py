"""Finance tools.

Note the escalating metadata as the side effects get more serious: reads are
open to any finance principal, ``create_expense`` is a WRITE, and
``initiate_payment`` is IRREVERSIBLE, restricted to finance managers and
flagged ``requires_human_approval`` so it can never be reached without an
approval step having run.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.tool_registry.adapters.systems.fake_backends import FinanceSystem
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.identity import Department, Principal, Role


class _EmployeeIdArgs(BaseModel):
    employee_id: str = Field(min_length=1, max_length=64)


class GetBudgetTool(Tool):
    class Args(BaseModel):
        cost_centre: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="finance.get_budget",
        domain=Department.FINANCE,
        description="Read the remaining budget for a cost centre.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: FinanceSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        budget = await self._system.get_budget(call.arguments["cost_centre"])
        return ToolResult.ok(self.name, budget)


class GetExpensePolicyTool(Tool):
    class Args(BaseModel):
        grade: str = Field(min_length=1, max_length=16)
        category: str = "other"

    input_model = Args
    metadata = ToolMetadata(
        name="finance.get_expense_policy",
        domain=Department.FINANCE,
        description="Read the per-claim expense limit for a grade and category.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: FinanceSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        policy = await self._system.get_expense_policy(
            call.arguments["grade"], call.arguments["category"]
        )
        return ToolResult.ok(self.name, policy)


class FindDuplicateExpenseTool(Tool):
    """Deterministic duplicate detection.

    This is exactly the kind of check that must not be delegated to a model:
    it is an exact comparison against recorded history, so it is code.
    """

    class Args(BaseModel):
        employee_id: str = Field(min_length=1)
        amount: float = Field(ge=0)
        expense_date: str = ""

    input_model = Args
    metadata = ToolMetadata(
        name="finance.find_duplicate_expense",
        domain=Department.FINANCE,
        description="Check whether a matching expense has already been recorded.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: FinanceSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        match = await self._system.find_duplicate_expense(
            employee_id=call.arguments["employee_id"],
            amount=float(call.arguments["amount"]),
            expense_date=call.arguments.get("expense_date", ""),
        )
        return ToolResult.ok(
            self.name, {"duplicate_found": match is not None, "existing": match or {}}
        )


class CreateExpenseTool(Tool):
    class Args(BaseModel):
        expense_id: str = Field(min_length=1)
        employee_id: str = Field(min_length=1)
        amount: float = Field(gt=0)
        currency: str = "INR"
        category: str = "other"
        cost_centre: str = ""
        expense_date: str = ""
        description: str = ""

    input_model = Args
    metadata = ToolMetadata(
        name="finance.create_expense",
        domain=Department.FINANCE,
        description="Record an expense claim in the ERP.",
        side_effect=SideEffect.WRITE,
        risk=RiskLevel.MEDIUM,
        allowed_roles=(Role.FINANCE_MANAGER, Role.FINANCE_ANALYST, Role.SERVICE_AGENT),
    )

    def __init__(self, system: FinanceSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        record = await self._system.create_expense(call.arguments)
        return ToolResult.ok(self.name, record)


class InitiatePaymentTool(Tool):
    """Money movement: the highest-risk action the platform can take.

    ``requires_human_approval`` here is a backstop, not the primary control --
    the workflow definition routes this behind an approval step. Both exist so
    that a mis-authored workflow cannot quietly pay someone.
    """

    class Args(BaseModel):
        employee_id: str = Field(min_length=1)
        amount: float = Field(gt=0)
        currency: str = "INR"
        cost_centre: str = ""
        expense_id: str = ""
        reference: str = ""

    input_model = Args
    metadata = ToolMetadata(
        name="finance.initiate_payment",
        domain=Department.FINANCE,
        description="Initiate a reimbursement payment.",
        side_effect=SideEffect.IRREVERSIBLE,
        risk=RiskLevel.CRITICAL,
        requires_human_approval=False,
        allowed_roles=(Role.FINANCE_MANAGER, Role.SERVICE_AGENT),
        # Never appears in an agent's tool list: only a workflow step reaches it.
        exposed_to_agents=False,
        max_attempts=3,
        timeout_seconds=45.0,
    )

    def __init__(self, system: FinanceSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        payment = await self._system.initiate_payment(
            idempotency_key=call.idempotency_key, payload=call.arguments
        )
        return ToolResult.ok(self.name, payment)


def build_finance_tools(system: FinanceSystem) -> list[Tool]:
    return [
        GetBudgetTool(system),
        GetExpensePolicyTool(system),
        FindDuplicateExpenseTool(system),
        CreateExpenseTool(system),
        InitiatePaymentTool(system),
    ]

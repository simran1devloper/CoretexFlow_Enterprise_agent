"""HR tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortexflow.modules.approval.domain.models import RiskLevel
from cortexflow.modules.tool_registry.adapters.systems.fake_backends import HrSystem
from cortexflow.modules.tool_registry.application.base import Tool
from cortexflow.modules.tool_registry.domain.models import (
    SideEffect,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from cortexflow.shared.identity import Department, Principal, Role


class GetEmployeeTool(Tool):
    class Args(BaseModel):
        employee_id: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="hr.get_employee",
        domain=Department.HR,
        description="Read an employee's master record.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: HrSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        return ToolResult.ok(
            self.name, await self._system.get_employee(call.arguments["employee_id"])
        )


class ValidateEmployeeTool(Tool):
    """Deterministic employment checks.

    Returns structured facts, not a verdict: whether those facts *permit* an
    action is the policy engine's call, not this tool's.
    """

    class Args(BaseModel):
        employee_id: str = Field(min_length=1, max_length=64)

    input_model = Args
    metadata = ToolMetadata(
        name="hr.validate_employee",
        domain=Department.HR,
        description="Check that an employee exists and is active, returning their facts.",
        side_effect=SideEffect.READ,
        risk=RiskLevel.LOW,
    )

    def __init__(self, system: HrSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        from cortexflow.shared.errors import NotFoundError

        employee_id = call.arguments["employee_id"]
        try:
            employee = await self._system.get_employee(employee_id)
        except NotFoundError:
            # A missing employee is a *finding*, not a step failure: the
            # workflow should continue to the policy gate and be denied there,
            # with a reason a human can read.
            return ToolResult.ok(
                self.name,
                {
                    "exists": False,
                    "active": False,
                    "employee_id": employee_id,
                    "reason": "No employee record found",
                },
            )
        return ToolResult.ok(
            self.name,
            {
                "exists": True,
                "active": employee.get("status") == "ACTIVE",
                "employee_id": employee_id,
                "name": employee.get("name", ""),
                "grade": employee.get("grade", ""),
                "department": employee.get("department", ""),
                "cost_centre": employee.get("cost_centre", ""),
                "manager_id": employee.get("manager_id", ""),
                "email": employee.get("email", ""),
                "reason": "" if employee.get("status") == "ACTIVE" else "Employee is not active",
            },
        )


class CreateEmployeeTool(Tool):
    class Args(BaseModel):
        employee_id: str = Field(min_length=1)
        name: str = Field(min_length=1)
        email: str = ""
        grade: str = "L1"
        department: str = ""
        cost_centre: str = ""
        manager_id: str = ""
        start_date: str = ""

    input_model = Args
    metadata = ToolMetadata(
        name="hr.create_employee",
        domain=Department.HR,
        description="Create an employee record in the HR system.",
        side_effect=SideEffect.WRITE,
        risk=RiskLevel.HIGH,
        allowed_roles=(Role.HR_MANAGER, Role.HR_SPECIALIST, Role.SERVICE_AGENT),
        exposed_to_agents=False,
    )

    def __init__(self, system: HrSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        return ToolResult.ok(self.name, await self._system.create_employee(call.arguments))


class CreateLeaveTool(Tool):
    class Args(BaseModel):
        leave_id: str = Field(min_length=1)
        employee_id: str = Field(min_length=1)
        leave_type: str = "annual"
        start_date: str = ""
        end_date: str = ""
        days: float = Field(gt=0, default=1)

    input_model = Args
    metadata = ToolMetadata(
        name="hr.create_leave",
        domain=Department.HR,
        description="Submit a leave request.",
        side_effect=SideEffect.WRITE,
        risk=RiskLevel.MEDIUM,
        allowed_roles=(Role.HR_MANAGER, Role.HR_SPECIALIST, Role.SERVICE_AGENT),
    )

    def __init__(self, system: HrSystem) -> None:
        self._system = system

    async def execute(self, call: ToolCall, principal: Principal) -> ToolResult:
        return ToolResult.ok(self.name, await self._system.create_leave(call.arguments))


def build_hr_tools(system: HrSystem) -> list[Tool]:
    return [
        GetEmployeeTool(system),
        ValidateEmployeeTool(system),
        CreateEmployeeTool(system),
        CreateLeaveTool(system),
    ]

"""Stand-in enterprise systems (HR, ERP/Finance, CRM/Marketing).

In a real deployment these are replaced by the integration service's HTTP
clients.  They exist here so the platform is runnable end-to-end, and because
the reliability tests need a dependency whose failures they can *choose*:
``fail_next`` and ``latency_ms`` let a test make the payment API time out at
exactly the wrong moment.

They deliberately implement server-side idempotency too, mirroring how a real
payments API behaves -- so the platform's key is checked on both sides.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from cortexflow.shared.clock import utcnow
from cortexflow.shared.errors import (
    DependencyTimeoutError,
    DependencyUnavailableError,
    NotFoundError,
)


@dataclass
class FaultInjector:
    """Controls simulated dependency failure, for chaos testing.

    Faults are targetable: ``only_operations`` narrows injection to specific
    API calls, so a test can make *the payment* time out without also breaking
    the budget lookup three steps earlier.
    """

    fail_next: int = 0
    """Number of upcoming matching calls that should fail."""

    failure: type[Exception] = DependencyTimeoutError
    only_operations: frozenset[str] = frozenset()
    """When non-empty, only these operations are eligible to fail."""

    latency_ms: int = 0
    unavailable: bool = False

    def _targets(self, operation: str) -> bool:
        return not self.only_operations or operation in self.only_operations

    async def apply(self, operation: str) -> None:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if not self._targets(operation):
            return
        if self.unavailable:
            raise DependencyUnavailableError("Backend unavailable", operation=operation)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise self.failure(f"Injected failure in {operation}")


@dataclass
class HrSystem:
    """Employee master data."""

    employees: dict[str, dict[str, Any]] = field(default_factory=dict)
    leave_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: FaultInjector = field(default_factory=FaultInjector)

    async def get_employee(self, employee_id: str) -> dict[str, Any]:
        await self.faults.apply("get_employee")
        try:
            return dict(self.employees[employee_id])
        except KeyError as exc:
            raise NotFoundError("Employee not found", employee_id=employee_id) from exc

    async def create_employee(self, record: dict[str, Any]) -> dict[str, Any]:
        await self.faults.apply("create_employee")
        employee_id = record["employee_id"]
        existing = self.employees.get(employee_id)
        if existing:
            return dict(existing)  # server-side idempotency
        stored = {**record, "status": "ACTIVE", "created_at": utcnow().isoformat()}
        self.employees[employee_id] = stored
        return dict(stored)

    async def update_employee(self, employee_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        await self.faults.apply("update_employee")
        record = await self.get_employee(employee_id)
        record.update(changes)
        self.employees[employee_id] = record
        return dict(record)

    async def create_leave(self, request: dict[str, Any]) -> dict[str, Any]:
        await self.faults.apply("create_leave")
        leave_id = request["leave_id"]
        if leave_id not in self.leave_requests:
            self.leave_requests[leave_id] = {**request, "status": "SUBMITTED"}
        return dict(self.leave_requests[leave_id])


@dataclass
class FinanceSystem:
    """ERP: budgets, expenses and payments."""

    budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    expenses: dict[str, dict[str, Any]] = field(default_factory=dict)
    payments: dict[str, dict[str, Any]] = field(default_factory=dict)
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: FaultInjector = field(default_factory=FaultInjector)

    async def get_budget(self, cost_centre: str) -> dict[str, Any]:
        await self.faults.apply("get_budget")
        budget = self.budgets.get(cost_centre)
        if budget is None:
            raise NotFoundError("Cost centre not found", cost_centre=cost_centre)
        return dict(budget)

    async def get_expense_policy(self, grade: str, category: str) -> dict[str, Any]:
        await self.faults.apply("get_expense_policy")
        return dict(
            self.policies.get(
                f"{grade}:{category}",
                self.policies.get(grade, {"per_claim_limit": 10_000, "requires_receipt": True}),
            )
        )

    async def find_duplicate_expense(
        self, *, employee_id: str, amount: float, expense_date: str
    ) -> dict[str, Any] | None:
        """Deterministic duplicate detection -- code, not a model."""
        await self.faults.apply("find_duplicate_expense")
        for expense in self.expenses.values():
            if (
                expense["employee_id"] == employee_id
                and abs(float(expense["amount"]) - amount) < 0.01
                and expense.get("expense_date") == expense_date
                and expense.get("status") != "REJECTED"
            ):
                return dict(expense)
        return None

    async def create_expense(self, record: dict[str, Any]) -> dict[str, Any]:
        await self.faults.apply("create_expense")
        expense_id = record["expense_id"]
        if expense_id not in self.expenses:
            self.expenses[expense_id] = {
                **record,
                "status": "RECORDED",
                "created_at": utcnow().isoformat(),
            }
        return dict(self.expenses[expense_id])

    async def initiate_payment(
        self, *, idempotency_key: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Money movement. Keyed server-side as well as client-side.

        Double protection is intentional: the platform's key prevents a second
        request, and this check makes a second request harmless if one escapes.
        """
        await self.faults.apply("initiate_payment")
        existing = self.payments.get(idempotency_key)
        if existing:
            return {**existing, "duplicate": True}

        cost_centre = payload.get("cost_centre", "")
        amount = float(payload["amount"])
        budget = self.budgets.get(cost_centre)
        if budget is not None:
            if budget["available"] < amount:
                from cortexflow.shared.errors import ValidationError

                raise ValidationError(
                    "Insufficient budget", cost_centre=cost_centre, amount=amount
                )
            budget["available"] -= amount

        payment = {
            "payment_id": f"PAY-{len(self.payments) + 1:06d}",
            "status": "SETTLED",
            "amount": amount,
            "currency": payload.get("currency", "INR"),
            "beneficiary": payload.get("employee_id", ""),
            "settled_at": utcnow().isoformat(),
            "duplicate": False,
        }
        self.payments[idempotency_key] = payment
        return dict(payment)


@dataclass
class MarketingSystem:
    """CRM / campaign management."""

    campaigns: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: FaultInjector = field(default_factory=FaultInjector)

    async def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        await self.faults.apply("get_campaign")
        campaign = self.campaigns.get(campaign_id)
        if campaign is None:
            raise NotFoundError("Campaign not found", campaign_id=campaign_id)
        return dict(campaign)

    async def update_campaign(
        self, campaign_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        await self.faults.apply("update_campaign")
        campaign = await self.get_campaign(campaign_id)
        campaign.update(changes)
        campaign["updated_at"] = utcnow().isoformat()
        self.campaigns[campaign_id] = campaign
        return dict(campaign)

    async def get_campaign_metrics(self, campaign_id: str) -> dict[str, Any]:
        await self.faults.apply("get_campaign_metrics")
        campaign = await self.get_campaign(campaign_id)
        spend = float(campaign.get("spend", 0)) or 1.0
        revenue = float(campaign.get("revenue", 0))
        return {
            "campaign_id": campaign_id,
            "impressions": campaign.get("impressions", 0),
            "clicks": campaign.get("clicks", 0),
            "conversions": campaign.get("conversions", 0),
            "spend": spend,
            "revenue": revenue,
            "roi": round(revenue / spend, 3),
            "ctr": round(
                campaign.get("clicks", 0) / max(campaign.get("impressions", 1), 1), 4
            ),
        }


def seed_demo_data() -> tuple[HrSystem, FinanceSystem, MarketingSystem]:
    """Populate the simulated backends for the local demo profile."""
    today = date.today()
    hr = HrSystem(
        employees={
            "EMP-1001": {
                "employee_id": "EMP-1001",
                "name": "Asha Menon",
                "email": "asha.menon@acme.example",
                "status": "ACTIVE",
                "grade": "L4",
                "department": "engineering",
                "cost_centre": "CC-ENG-01",
                "manager_id": "EMP-2001",
            },
            "EMP-1002": {
                "employee_id": "EMP-1002",
                "name": "Ravi Iyer",
                "email": "ravi.iyer@acme.example",
                "status": "INACTIVE",
                "grade": "L3",
                "department": "sales",
                "cost_centre": "CC-SAL-01",
                "manager_id": "EMP-2002",
            },
            "EMP-2001": {
                "employee_id": "EMP-2001",
                "name": "Priya Nair",
                "email": "priya.nair@acme.example",
                "status": "ACTIVE",
                "grade": "L7",
                "department": "engineering",
                "cost_centre": "CC-ENG-01",
                "manager_id": "",
            },
        }
    )
    finance = FinanceSystem(
        budgets={
            "CC-ENG-01": {"cost_centre": "CC-ENG-01", "allocated": 5_000_000,
                          "available": 1_250_000, "fiscal_year": today.year},
            "CC-SAL-01": {"cost_centre": "CC-SAL-01", "allocated": 2_000_000,
                          "available": 40_000, "fiscal_year": today.year},
            "CC-MKT-01": {"cost_centre": "CC-MKT-01", "allocated": 8_000_000,
                          "available": 3_400_000, "fiscal_year": today.year},
        },
        policies={
            "L3": {"per_claim_limit": 15_000, "requires_receipt": True},
            "L4": {"per_claim_limit": 25_000, "requires_receipt": True},
            "L7": {"per_claim_limit": 100_000, "requires_receipt": True},
            "L4:travel": {"per_claim_limit": 40_000, "requires_receipt": True},
        },
    )
    marketing = MarketingSystem(
        campaigns={
            "CAM-2024-01": {
                "campaign_id": "CAM-2024-01",
                "name": "Q3 Product Launch",
                "status": "ACTIVE",
                "budget": 1_200_000,
                "spend": 840_000,
                "revenue": 1_512_000,
                "impressions": 2_400_000,
                "clicks": 61_000,
                "conversions": 3_050,
                "cost_centre": "CC-MKT-01",
                "ends_at": (today + timedelta(days=45)).isoformat(),
            },
            "CAM-2024-02": {
                "campaign_id": "CAM-2024-02",
                "name": "Retention Winback",
                "status": "ACTIVE",
                "budget": 400_000,
                "spend": 390_000,
                "revenue": 280_000,
                "impressions": 900_000,
                "clicks": 12_400,
                "conversions": 410,
                "cost_centre": "CC-MKT-01",
                "ends_at": (today + timedelta(days=12)).isoformat(),
            },
        }
    )
    return hr, finance, marketing

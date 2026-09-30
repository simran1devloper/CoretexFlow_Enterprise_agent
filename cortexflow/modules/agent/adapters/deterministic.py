"""A deterministic stand-in for Azure OpenAI.

Its purpose is not to imitate a model but to make the *platform* testable:
workflow orchestration, retries, approvals and recovery must be verifiable
without network access, token spend or run-to-run variance.  It also backs the
local demo profile so the system is runnable straight from a clone.

Handlers are registered per schema name, so an agent under test gets a
predictable, schema-valid answer derived from its own prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from cortexflow.modules.agent.domain.models import TokenUsage
from cortexflow.modules.agent.ports.llm import CompletionRequest, CompletionResponse
from cortexflow.shared.errors import DependencyUnavailableError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[[CompletionRequest], dict[str, Any]]

_NUMBER = re.compile(r"(?:amount|total|value)\D{0,12}?([0-9][0-9,]*\.?[0-9]*)", re.IGNORECASE)
_CURRENCY = re.compile(r"\b(INR|USD|EUR|GBP)\b")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_EMPLOYEE = re.compile(r"\b(EMP[-_]?\d{3,})\b", re.IGNORECASE)
_VENDOR = re.compile(r"(?:vendor|merchant|from)\s*[:\-]?\s*([A-Za-z][\w &.'-]{2,40})", re.I)
_CATEGORY_HINTS = {
    "travel": ("flight", "taxi", "cab", "uber", "train", "hotel", "airfare", "travel"),
    "meals": ("lunch", "dinner", "meal", "restaurant", "food", "catering"),
    "software": ("license", "subscription", "saas", "software"),
    "office": ("stationery", "printer", "desk", "office"),
}


class DeterministicLlmClient:
    """Pure-function LLM: identical input always yields identical output."""

    def __init__(
        self,
        *,
        handlers: dict[str, Handler] | None = None,
        failure_rate: float = 0.0,
        model_name: str = "deterministic-stub",
    ) -> None:
        self._handlers: dict[str, Handler] = {
            "ExtractionOutput": _extraction,
            "ValidationOutput": _validation,
            "DecisionOutput": _decision,
            "ReportOutput": _report,
            "CommunicationOutput": _communication,
            **(handlers or {}),
        }
        self._failure_rate = failure_rate
        self._model_name = model_name
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def register(self, schema_name: str, handler: Handler) -> None:
        self._handlers[schema_name] = handler

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        if self._failure_rate > 0 and _hash_ratio(request) < self._failure_rate:
            # Deterministic "outage" for chaos tests: same input, same failure.
            raise DependencyUnavailableError("Simulated Azure OpenAI outage")

        handler = self._handlers.get(request.schema_name, _fallback)
        payload = handler(request)
        content = json.dumps(payload)
        prompt_chars = sum(len(m.content) for m in request.messages)
        return CompletionResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=len(content) // 4,
                model=self._model_name,
            ),
            model=self._model_name,
            finish_reason="stop",
        )


def _hash_ratio(request: CompletionRequest) -> float:
    digest = hashlib.sha256(
        "".join(m.content for m in request.messages).encode()
    ).digest()
    return digest[0] / 255


_TAG = "<{name}[^>]*>(?P<body>.*?)</{name}>"


def _user_text(request: CompletionRequest) -> str:
    return "\n".join(m.content for m in request.messages if m.role == "user")


def _payload(request: CompletionRequest, *tags: str) -> str:
    """Return only the tagged payload sections of the prompt.

    The agents wrap untrusted content in ``<document>`` / ``<data>`` tags and
    tell the model to treat only that as input.  This stub honours the same
    boundary, so it cannot "read" instructions, field lists or check
    descriptions from the surrounding prompt -- which is exactly the mistake a
    careless real prompt would make.
    """
    text = _user_text(request)
    found: list[str] = []
    for tag in tags or ("document", "data"):
        found.extend(
            m.group("body").strip()
            for m in re.finditer(_TAG.format(name=tag), text, re.DOTALL)
        )
    return "\n\n".join(part for part in found if part)


def _number(text: str) -> float:
    match = _NUMBER.search(text)
    if not match:
        loose = re.search(r"\b([0-9][0-9,]{2,}\.?[0-9]*)\b", text)
        if not loose:
            return 0.0
        return float(loose.group(1).replace(",", ""))
    return float(match.group(1).replace(",", ""))


def _category(text: str) -> str:
    lowered = text.lower()
    for category, hints in _CATEGORY_HINTS.items():
        if any(hint in lowered for hint in hints):
            return category
    return "other"


_FIELD_LIST = re.compile(r"Fields to extract:\n(?P<body>(?:\s*-\s*\w+\n?)+)", re.I)
_LABELLED = re.compile(r"^\s*(?P<label>[A-Za-z][\w /-]{1,40}?)\s*:\s*(?P<value>.+?)\s*$", re.M)


def _requested_fields(request: CompletionRequest) -> list[str]:
    """The field list from the prompt scaffolding.

    Read from the instructions, not from the document: the field list is
    trusted prompt content, while the document is untrusted data.
    """
    match = _FIELD_LIST.search(_user_text(request))
    if not match:
        return []
    return [line.strip(" -\t") for line in match.group("body").splitlines() if line.strip()]


def _labelled_values(text: str) -> dict[str, str]:
    """Parse ``Label: value`` lines out of a document."""
    return {
        m.group("label").strip().lower().replace(" ", "_").replace("-", "_"):
            m.group("value").strip()
        for m in _LABELLED.finditer(text)
    }


NUMERIC_FIELDS = frozenset(
    {"amount", "total", "value", "budget", "proposed_budget", "days", "quantity"}
)
"""Fields whose value must reach the policy engine as a number.

A currency-suffixed string like "2450.00 INR" would compare as text against a
numeric threshold, which is exactly the class of bug this coercion prevents.
"""

_LEADING_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce(value: str, *, field: str = "") -> Any:
    cleaned = value.replace(",", "").strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return float(cleaned) if "." in cleaned else int(cleaned)
    if field in NUMERIC_FIELDS:
        match = _LEADING_NUMBER.search(cleaned)
        return float(match.group()) if match else 0.0
    return value.strip()


def _extraction(request: CompletionRequest) -> dict[str, Any]:
    """Department-agnostic extraction.

    Resolves each requested field from the document's labelled lines, falling
    back to pattern matching for the values that are rarely labelled cleanly
    (amounts, dates, employee ids). Fields it genuinely cannot find are
    reported as missing -- never invented.
    """
    text = _payload(request, "document")
    labelled = _labelled_values(text)
    requested = _requested_fields(request) or [
        "amount", "currency", "employee_id", "vendor", "expense_date", "category"
    ]

    patterns: dict[str, Any] = {}
    if (employee := _EMPLOYEE.search(text)) is not None:
        patterns["employee_id"] = employee.group(1).upper().replace("_", "-")
    if (date := _DATE.search(text)) is not None:
        patterns["expense_date"] = patterns["start_date"] = date.group(1)
    if (currency := _CURRENCY.search(text)) is not None:
        patterns["currency"] = currency.group(1)
    if (amount := _number(text)) :
        patterns["amount"] = amount
    if (vendor := _VENDOR.search(text)) is not None:
        patterns["vendor"] = vendor.group(1).strip()
    patterns["category"] = _category(text)

    aliases = {
        "expense_date": ("date", "expense_date", "invoice_date"),
        "employee_id": ("employee_id", "employee", "emp_id"),
        "candidate_name": ("candidate_name", "name", "candidate"),
        "vendor": ("vendor", "merchant", "supplier", "from"),
        "amount": ("amount", "total", "value"),
        "start_date": ("start_date", "start", "joining_date"),
        "identity_document": ("identity_document", "identity", "id_document"),
        "right_to_work_document": ("right_to_work_document", "right_to_work"),
    }

    fields: dict[str, Any] = {}
    for field in requested:
        keys = aliases.get(field, (field,))
        value = next(
            (labelled[key] for key in keys if labelled.get(key)),
            patterns.get(field),
        )
        if value not in (None, ""):
            fields[field] = (
                _coerce(value, field=field) if isinstance(value, str) else value
            )

    missing = [field for field in requested if field not in fields]
    # Confidence tracks how much of the requested data was actually found --
    # the signal the policy engine uses to decide whether to involve a human.
    found_ratio = (len(requested) - len(missing)) / max(len(requested), 1)
    confidence = round(min(0.6 + 0.39 * found_ratio, 0.99), 2)

    return {
        "fields": fields,
        "confidence": confidence,
        "missing_fields": missing,
        "reason": (
            "Extracted deterministically from the supplied document."
            if not missing
            else f"Could not locate: {', '.join(missing)}."
        ),
    }


def _validation(request: CompletionRequest) -> dict[str, Any]:
    text = _payload(request, "document", "data").lower()
    issues: list[str] = []
    if "duplicate" in text:
        issues.append("Possible duplicate submission detected")
    if "altered" in text or "edited" in text:
        issues.append("Document appears modified")
    if "missing" in text:
        issues.append("Required field missing from the document")
    return {
        "valid": not issues,
        "issues": issues,
        "confidence": 0.93 if not issues else 0.7,
        "reason": "; ".join(issues) or "No anomalies found in the document.",
    }


def _decision(request: CompletionRequest) -> dict[str, Any]:
    text = _payload(request, "data")
    amount = _number(text)
    lowered = text.lower()
    # Keyed off the structured case facts, not loose substring matching.
    flagged = (
        '"receipt_valid": false' in lowered
        or '"duplicate_found": true' in lowered
        or '"employee_active": false' in lowered
        or bool(re.search(r'"receipt_issues":\s*\[[^\]]+\]', lowered))
    )
    if flagged:
        recommendation, confidence = "REJECT", 0.82
        reason = "Upstream validation reported unresolved issues."
    elif amount > 50_000:
        recommendation, confidence = "MANUAL_REVIEW", 0.88
        reason = "Amount is materially above routine spend for this category."
    elif amount > 10_000:
        recommendation, confidence = "MANUAL_REVIEW", 0.9
        reason = "Amount exceeds the routine auto-approval band."
    else:
        recommendation, confidence = "APPROVE", 0.94
        reason = "Routine expense, consistent with prior submissions."
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
        "risk_factors": ["validation_issue"] if flagged else [],
    }


def _report(request: CompletionRequest) -> dict[str, Any]:
    text = _payload(request, "data")
    return {
        "title": "Workflow summary",
        "summary": (
            "Workflow completed. "
            f"Reference data length {len(text)} characters processed across all steps."
        )[:500],
        "sections": [
            {"heading": "Outcome", "body": "All steps completed and recorded."},
            {"heading": "Controls", "body": "Policy evaluation and approvals applied."},
        ],
        "highlights": ["Deterministic execution", "Full audit trail recorded"],
    }


def _communication(request: CompletionRequest) -> dict[str, Any]:
    return {
        "subject": "Update on your request",
        "body": "Your request has been processed. See the attached summary for details.",
        "tone": "professional",
    }


def _fallback(request: CompletionRequest) -> dict[str, Any]:
    """Best-effort: emit the schema's required keys with typed zero values."""
    schema = request.json_schema or {}
    properties: dict[str, Any] = schema.get("properties", {})
    result: dict[str, Any] = {}
    for key in schema.get("required", list(properties)):
        spec = properties.get(key, {})
        match spec.get("type"):
            case "number" | "integer":
                result[key] = 0
            case "boolean":
                result[key] = False
            case "array":
                result[key] = []
            case "object":
                result[key] = {}
            case _:
                result[key] = ""
    return result

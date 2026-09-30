"""Payload redaction.

Agent state routinely contains salary figures, bank details and personal data.
Logs, traces and audit metadata fan out to places with weaker access controls
than the data plane, so everything on its way out passes through here first.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password", "secret", "token", "api_key", "apikey", "authorization",
        "connection_string", "client_secret", "private_key", "credential",
        "ssn", "social_security_number", "national_id", "pan", "aadhaar",
        "account_number", "bank_account", "iban", "routing_number", "card_number",
        "cvv", "salary", "compensation", "date_of_birth", "dob",
        "raw_text", "raw_content", "document_content", "email_body",
        "prompt", "messages", "attachment",
    }
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
)

MAX_STRING_LENGTH = 512
MAX_COLLECTION_ITEMS = 25
MAX_DEPTH = 6


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_KEYS)


def scrub_text(text: str) -> str:
    """Mask identifiers that leak even when the key name looks innocuous."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > MAX_STRING_LENGTH:
        text = f"{text[:MAX_STRING_LENGTH]}...[truncated {len(text)} chars]"
    return text


def redact(value: Any, *, depth: int = 0) -> Any:
    """Return a copy of ``value`` safe to log.

    Bounded in depth and width as well as content: an unbounded log record is
    its own kind of incident.
    """
    if depth >= MAX_DEPTH:
        return "[MAX_DEPTH]"
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_COLLECTION_ITEMS]:
            out[str(key)] = REDACTED if _is_sensitive(str(key)) else redact(item, depth=depth + 1)
        if len(value) > MAX_COLLECTION_ITEMS:
            out["__truncated__"] = len(value) - MAX_COLLECTION_ITEMS
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:MAX_COLLECTION_ITEMS]
        return [redact(item, depth=depth + 1) for item in items]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return scrub_text(str(value))


def summarize(value: Any, *, keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Project a payload down to a named allowlist -- the safest default.

    Used for approval context and audit metadata, where we want *some*
    business detail but must not copy the whole document.
    """
    if not isinstance(value, dict):
        return {}
    selected = {k: value[k] for k in keys if k in value} if keys else value
    redacted = redact(selected)
    return redacted if isinstance(redacted, dict) else {}

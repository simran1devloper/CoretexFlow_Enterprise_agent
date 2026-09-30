"""Identifier generation.

Identifiers are generated through an injectable provider so tests can produce
deterministic documents and so every id carries a readable prefix in logs.
"""

from __future__ import annotations

import uuid
from typing import Protocol

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32, no I/L/O/U


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        value, rem = divmod(value, len(_ALPHABET))
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars))


class IdGenerator(Protocol):
    """Port for identifier generation."""

    def new_id(self, prefix: str) -> str: ...


class UuidIdGenerator:
    """Default generator: ``WF-3K7P2QF9XB4C``."""

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{_encode(uuid.uuid4().int, 12)}"


class SequentialIdGenerator:
    """Deterministic generator for tests: ``WF-000001``."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def new_id(self, prefix: str) -> str:
        nxt = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = nxt
        return f"{prefix}-{nxt:06d}"


WORKFLOW = "WF"
STEP_RUN = "RUN"
APPROVAL = "APR"
AUDIT = "AUD"
MESSAGE = "MSG"
DOCUMENT = "DOC"

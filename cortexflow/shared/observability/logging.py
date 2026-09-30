"""Structured logging with automatic context propagation.

Every log line carries the workflow, step and correlation ids of the work in
flight, set once by the entrypoint via :func:`bind_context` and picked up from
a context variable everywhere else.  That is what makes "why did this workflow
take four minutes?" answerable by query rather than by guesswork.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from cortexflow.shared.observability.redaction import redact

# An empty tuple is immutable, so no ContextVar can be mutated in place by
# one task and observed by another.
_EMPTY: tuple[tuple[str, Any], ...] = ()
_log_context: ContextVar[tuple[tuple[str, Any], ...]] = ContextVar(
    "cortexflow_log_context", default=_EMPTY
)

CONTEXT_FIELDS = (
    "tenant_id",
    "workflow_id",
    "step_id",
    "run_id",
    "correlation_id",
    "message_id",
    "service",
)


def current_context() -> dict[str, Any]:
    return dict(_log_context.get())


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every log record emitted inside the block."""
    merged = {
        **dict(_log_context.get()),
        **{k: v for k, v in fields.items() if v is not None},
    }
    token = _log_context.set(tuple(merged.items()))
    try:
        yield
    finally:
        _log_context.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _log_context.get():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter with no hard dependency on an external library."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_") and key != "taskName":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str)


def configure_logging(
    *, service: str, level: str = "INFO", json_output: bool = True
) -> None:
    """Install the platform's logging configuration. Idempotent."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("azure", "uamqp", "urllib3", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _log_context.set((("service", service),))


def get_logger(name: str) -> logging.LoggerAdapter[logging.Logger]:
    """Logger that redacts structured ``extra`` payloads automatically."""
    return _RedactingAdapter(logging.getLogger(name), {})


class _RedactingAdapter(logging.LoggerAdapter):
    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {k: redact(v) for k, v in extra.items()}
        return msg, kwargs

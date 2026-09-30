"""Extraction agent: unstructured document in, structured fields out.

The pipeline is OCR first, LLM second.  Document Intelligence is deterministic
and good at layout; the model is good at meaning.  Using each where it is
strongest is both cheaper and more reliable than asking the model to do both.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.modules.agent.application.prompts import EXTRACTION
from cortexflow.modules.agent.application.schemas import ExtractionOutput
from cortexflow.modules.agent.domain.models import AgentContext, AgentResult
from cortexflow.modules.agent.ports.llm import LlmClient
from cortexflow.modules.document.ports.storage import BlobRef, BlobStore
from cortexflow.modules.tool_registry.application.registry import ToolRegistry
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

MAX_DOCUMENT_CHARS = 20_000

NUMERIC_FIELDS: frozenset[str] = frozenset(
    {"amount", "total", "value", "budget", "proposed_budget", "days", "quantity"}
)
"""Fields whose value must reach the policy engine as a number.

A language model returns ``"2450.00 INR"`` or ``"2450.00"`` for an amount --
both strings. The policy engine compares amounts against numeric thresholds,
and ``"2450.00" > 10000`` is not a comparison Python will make. Without this
coercion the rule raises, the engine logs it and treats it as unmatched, and
the claim falls through to the ruleset default.

The workflow can extend the list per step via a ``numeric_fields`` input.
"""

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class ExtractionAgent(BaseAgent):
    name = "extraction_agent"
    description = "Extracts structured fields from documents, email bodies and forms."

    def __init__(
        self,
        llm: LlmClient,
        tools: ToolRegistry | None = None,
        blobs: BlobStore | None = None,
    ) -> None:
        super().__init__(llm, tools)
        self._blobs = blobs

    async def run(self, context: AgentContext) -> AgentResult:
        document_text = await self._resolve_text(context)
        if not document_text.strip():
            # Nothing to read is a case for a human, not a retry.
            return self._needs_human("No document content was available to extract from")

        expected = context.inputs.get("expected_fields") or []
        user_content = _build_prompt(
            document_text=document_text[:MAX_DOCUMENT_CHARS],
            expected_fields=list(expected),
            hints=context.inputs.get("hints", {}),
        )

        parsed, usage = await self.think(
            context,
            system_prompt=EXTRACTION.system,
            user_content=user_content,
            output_model=ExtractionOutput,
            prompt_version=EXTRACTION.version,
        )

        numeric = NUMERIC_FIELDS | {
            str(f) for f in (context.inputs.get("numeric_fields") or ())
        }
        fields = _coerce_numeric(parsed.fields, numeric)

        # Trust the deterministic check over the model's self-assessment: if a
        # required field is absent, it is missing regardless of what the model
        # reported.
        missing = sorted(
            {*parsed.missing_fields} | {f for f in expected if fields.get(f) in (None, "")}
        )
        confidence = parsed.confidence
        if missing:
            confidence = min(confidence, 0.8 - 0.05 * len(missing))

        return AgentResult.success(
            output={
                "fields": fields,
                "missing_fields": missing,
                "missing_count": len(missing),
                "data_complete": not missing,
                "document_chars": len(document_text),
            },
            confidence=round(max(confidence, 0.0), 3),
            reason=parsed.reason,
            usage=usage,
        )

    async def _resolve_text(self, context: AgentContext) -> str:
        """Assemble document text from inline content and/or a blob reference."""
        inline = context.inputs.get("text") or context.inputs.get("content") or ""
        blob_ref = context.inputs.get("document_ref")
        if not blob_ref or self._blobs is None:
            return str(inline)

        try:
            ref = BlobRef.model_validate(blob_ref)
            raw = await self._blobs.get(ref)
        except Exception as exc:
            logger.warning(
                "document fetch failed; continuing with inline content",
                extra={"error": str(exc), "workflow_id": context.workflow_id},
            )
            return str(inline)

        extracted = await self._ocr(ref, raw)
        return "\n\n".join(part for part in (str(inline), extracted) if part.strip())

    async def _ocr(self, ref: BlobRef, raw: bytes) -> str:
        """OCR hook.

        In Azure this delegates to Document Intelligence. Locally it decodes
        text-like content; a binary document without OCR yields an empty string,
        which surfaces as low confidence rather than as invented fields.
        """
        if ref.content_type.startswith("text/") or ref.content_type == "application/json":
            return raw.decode("utf-8", errors="replace")
        logger.info(
            "no OCR backend configured for this content type",
            extra={"content_type": ref.content_type},
        )
        return ""


def _build_prompt(
    *, document_text: str, expected_fields: list[str], hints: dict[str, Any]
) -> str:
    sections = []
    if expected_fields:
        sections.append("Fields to extract:\n" + "\n".join(f"- {f}" for f in expected_fields))
    if hints:
        sections.append(f"Context hints (trusted):\n{json.dumps(hints, default=str)}")
    sections.append(f"<document>\n{document_text}\n</document>")
    return "\n\n".join(sections)


def _coerce_numeric(
    fields: dict[str, Any], numeric_fields: frozenset[str] | set[str]
) -> dict[str, Any]:
    """Turn numeric-looking strings into numbers for the fields that need it.

    A value the model could not parse cleanly is left untouched rather than
    forced to zero: a wrong number is worse than a value the policy engine
    can see is not a number.
    """
    coerced: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in numeric_fields or not isinstance(value, str):
            coerced[name] = value
            continue
        match = _NUMBER.search(value.replace(",", ""))
        if match is None:
            logger.warning(
                "could not read a numeric field as a number; leaving it as text",
                extra={"field": name},
            )
            coerced[name] = value
            continue
        number = float(match.group())
        coerced[name] = int(number) if number.is_integer() else number
    return coerced

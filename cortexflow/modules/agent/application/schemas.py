"""Typed agent outputs.

These schemas *are* the contract between the probabilistic layer and the
deterministic one.  Every field here is something the orchestrator or the
policy engine can act on; nothing is free prose that a downstream step would
have to interpret.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fields: dict[str, object] = Field(
        default_factory=dict, description="Structured fields extracted from the document"
    )
    confidence: float = Field(ge=0, le=1, default=0.0)
    missing_fields: list[str] = Field(default_factory=list)
    reason: str = ""


class ValidationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    valid: bool = False
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.0)
    reason: str = ""


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recommendation: str = Field(
        description="APPROVE | REJECT | MANUAL_REVIEW -- a recommendation, not a decision"
    )
    confidence: float = Field(ge=0, le=1, default=0.0)
    reason: str = ""
    risk_factors: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    heading: str = ""
    body: str = ""


class ReportOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class CommunicationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: str = ""
    body: str = ""
    tone: str = "professional"


class TriageOutput(BaseModel):
    """Output of the Semantic Kernel triage agent.

    ``findings`` is the point: deterministic facts the agent gathered by
    calling registry tools, which the policy engine then evaluates. The
    ``assessment`` is commentary; it authorizes nothing.
    """

    model_config = ConfigDict(extra="ignore")

    findings: dict[str, object] = Field(
        default_factory=dict,
        description="Facts gathered from enterprise systems via tool calls",
    )
    assessment: str = Field(default="", description="Short summary of the case")
    concerns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.0)
    needs_human: bool = Field(
        default=False,
        description="Set when the agent could not gather what it needed",
    )

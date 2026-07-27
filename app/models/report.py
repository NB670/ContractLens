"""Structured model for the Checkpoint 4 executive report deliverable.

The original project plan's chatbot deliverable includes the ability to
"generate executive review reports summarizing key findings and risks" --
this is that report's shape. It reuses RiskFinding as-is so a report's
findings link back to the same clause evidence app/risk/analyzer.py already
produces.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.analysis import RiskFinding


class ExecutiveReport(BaseModel):
    """A one-page executive summary of a single reviewed contract."""

    contract_id: str
    filename: str
    contract_type: str = "Unknown"
    parties: list[str] = Field(default_factory=list)
    categories: dict[str, int] = Field(
        default_factory=dict, description="Clause count per identified category"
    )
    risk_level: str = "low"
    overall_score: float = Field(0.0, ge=0.0, le=100.0)
    top_findings: list[RiskFinding] = Field(default_factory=list)
    narrative: Optional[str] = Field(
        None, description="1-4 sentence plain-English summary; LLM-written when "
        "available, otherwise a deterministic template sentence"
    )

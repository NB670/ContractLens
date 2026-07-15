"""Structured models for the Checkpoint 3 downstream tasks.

These sit alongside `app/models/contract.py`. Everything here references the
same clause identity used in CP2 (`Clause.index`) so a hit, a change, or a
finding can always be linked back to the exact clause it came from. This file
grows across Checkpoint 3: retrieval's `RetrievalHit` first, then comparison's
`ClauseChange`/`ContractDiff`, then risk's `RiskFinding`/`RiskReport`.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """One ranked result from a semantic clause search."""

    contract_id: str
    clause_index: int = Field(..., description="Clause.index within its contract")
    category: str
    heading: Optional[str] = None
    text: str
    score: float = Field(..., description="Cosine similarity in [-1, 1]")


# --------------------------------------------------------------------------- #
# Contract comparison
# --------------------------------------------------------------------------- #

CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_MODIFIED = "modified"
CHANGE_UNCHANGED = "unchanged"


class ClauseChange(BaseModel):
    """A single aligned (or unmatched) clause between two contract versions."""

    change_type: str = Field(
        ..., description=f"one of {CHANGE_ADDED!r}/{CHANGE_REMOVED!r}/"
        f"{CHANGE_MODIFIED!r}/{CHANGE_UNCHANGED!r}"
    )
    category: str = "Unclassified"
    base_index: Optional[int] = Field(
        None, description="Clause.index in the base contract (None if added)"
    )
    revised_index: Optional[int] = Field(
        None, description="Clause.index in the revised contract (None if removed)"
    )
    similarity: float = Field(0.0, description="Cosine similarity of the aligned pair in [-1, 1]")
    base_text: Optional[str] = None
    revised_text: Optional[str] = None


class ContractDiff(BaseModel):
    """Structured diff between a base contract and a revised contract."""

    base_contract_id: str
    revised_contract_id: str
    changes: list[ClauseChange] = Field(default_factory=list)
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by change_type (added/removed/modified/unchanged)",
    )


# --------------------------------------------------------------------------- #
# Risk analysis
# --------------------------------------------------------------------------- #

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

# Numeric weight per severity, used to roll individual findings up into an
# overall 0-100 risk score.
SEVERITY_WEIGHT: dict[str, int] = {
    SEVERITY_LOW: 1,
    SEVERITY_MEDIUM: 3,
    SEVERITY_HIGH: 6,
}


class RiskFinding(BaseModel):
    """One risk flagged against a clause, carrying its supporting evidence."""

    rule_id: str = Field(..., description="Stable id of the rule that fired")
    category: str = Field("Unclassified", description="Clause category the rule is scoped to")
    severity: str = Field(..., description="low | medium | high")
    rationale: str = Field(..., description="Why this is a risk, in plain language")
    clause_index: Optional[int] = Field(None, description="Clause.index; None for whole-contract findings")
    evidence_text: str = Field("", description="The clause text (or excerpt) that triggered the rule")
    start_offset: int = 0
    end_offset: int = 0


class RiskReport(BaseModel):
    """The full risk assessment for a single contract."""

    contract_id: str
    overall_score: float = Field(0.0, ge=0.0, le=100.0, description="Aggregate risk score in [0, 100]")
    risk_level: str = Field(SEVERITY_LOW, description="low | medium | high")
    findings: list[RiskFinding] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict, description="Counts keyed by severity")

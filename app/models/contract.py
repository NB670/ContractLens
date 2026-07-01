"""Structured representations of a parsed contract.

These models are the "structured contract representation" the project plan refers
to: the bridge between the raw uploaded document and every downstream task
(classification now; retrieval, comparison, and risk analysis in later
checkpoints).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Clause(BaseModel):
    """A single segmented clause / section of a contract."""

    index: int = Field(..., description="0-based position of the clause in the document")
    heading: Optional[str] = Field(
        None, description="Detected heading/title for the clause, if any"
    )
    text: str = Field(..., description="Raw clause text")
    category: str = Field(
        "Unclassified", description="CUAD-style category assigned by the classifier"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Classifier confidence in [0, 1]"
    )

    # Character offsets into the original document text so the UI can link a
    # clause back to its source (the 'highlight relevant section' deliverable).
    start_offset: int = 0
    end_offset: int = 0


class ContractMetadata(BaseModel):
    """Lightweight metadata extracted during ingestion."""

    contract_type: str = "Unknown"
    parties: list[str] = Field(default_factory=list)
    num_clauses: int = 0
    num_chars: int = 0


class Contract(BaseModel):
    """A fully ingested + structured contract."""

    id: str
    filename: str
    source_format: str  # "pdf" | "docx" | "txt"
    metadata: ContractMetadata = Field(default_factory=ContractMetadata)
    clauses: list[Clause] = Field(default_factory=list)

    def categories_present(self) -> dict[str, int]:
        """Return a {category: count} map for the clause-visualization view."""
        counts: dict[str, int] = {}
        for clause in self.clauses:
            counts[clause.category] = counts.get(clause.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

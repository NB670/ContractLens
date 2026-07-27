"""Structured models for the Checkpoint 4 chatbot / QA task.

Every answer the assistant returns is grounded in retrieved clauses, so a
ChatResponse always carries the Citation list it was built from -- each
citation points back to the exact clause (contract_id + Clause.index) it
came from, reusing the same clause identity app/models/contract.py and
app/models/analysis.py use. This is the "cited answers" deliverable from the
project plan.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A natural-language question, optionally scoped to one stored contract."""

    question: str = Field(..., min_length=1, description="The user's question")
    contract_id: Optional[str] = Field(
        None, description="Restrict retrieval to this contract; None searches all"
    )
    k: int = Field(5, ge=1, le=20, description="How many clauses to retrieve as grounding")


class Citation(BaseModel):
    """A single retrieved clause used as grounding for an answer."""

    contract_id: str
    clause_index: int = Field(..., description="Clause.index within its contract")
    category: str = "Unclassified"
    heading: Optional[str] = None
    excerpt: str = Field("", description="Clause text (possibly truncated) shown to the user")
    score: float = Field(0.0, description="Cosine similarity of the clause to the question")


class ChatResponse(BaseModel):
    """A grounded answer to a natural-language question about contract(s)."""

    question: str
    answer: str
    backend: str = Field(..., description="'local-llm' or 'extractive'")
    contract_id: Optional[str] = Field(
        None, description="Contract the question was scoped to, if any"
    )
    citations: list[Citation] = Field(default_factory=list)
    tools_used: list[str] = Field(
        default_factory=list,
        description="MCP tools the assistant invoked to build this answer",
    )

"""Tests for the Checkpoint 4 RAG chatbot assistant.

All tests run offline against the dependency-free `extractive` backend and a
FakeMCPClient (no real subprocess, no model download) -- app/mcp/client.py's
real subprocess round trip is already covered by tests/test_mcp_client.py.
"""

import pytest

from app.chat.assistant import ChatAssistant


class FakeMCPClient:
    """Records calls and returns canned tool results, shaped like MCPToolClient."""

    def __init__(self, search_hits=None, risk_report=None):
        self._search_hits = search_hits if search_hits is not None else []
        self._risk_report = risk_report or {"found": False}
        self.calls: list[str] = []

    async def call_tool(self, name: str, **kwargs) -> dict:
        self.calls.append(name)
        if name == "search_clauses":
            hits = self._search_hits
            if kwargs.get("contract_id") is not None:
                hits = [h for h in hits if h["contract_id"] == kwargs["contract_id"]]
            return {"hits": hits[: kwargs.get("k", 5)]}
        if name == "assess_risk":
            return self._risk_report
        raise AssertionError(f"unexpected tool call: {name}")


def _hit(contract_id, clause_index, category, text, score=0.9):
    return {
        "contract_id": contract_id, "clause_index": clause_index, "category": category,
        "heading": None, "text": text, "score": score,
    }


@pytest.mark.asyncio
async def test_extractive_answer_is_grounded_and_cited():
    mcp = FakeMCPClient(search_hits=[
        _hit("c1", 0, "Confidentiality", "The receiving party shall keep all confidential information secret."),
    ])
    assistant = ChatAssistant(mcp, backend="extractive")

    resp = await assistant.answer("What are the confidentiality obligations?", k=3)

    assert resp.backend == "extractive"
    assert resp.citations, "answer should be grounded in retrieved clauses"
    assert "search_clauses" in resp.tools_used
    assert "confidential" in resp.answer.lower()


@pytest.mark.asyncio
async def test_answer_scopes_to_a_single_contract():
    mcp = FakeMCPClient(search_hits=[
        _hit("c1", 0, "Confidentiality", "Confidential information must be protected."),
        _hit("c2", 0, "Confidentiality", "Confidential data stays private."),
    ])
    assistant = ChatAssistant(mcp, backend="extractive")

    resp = await assistant.answer("confidentiality", contract_id="c2", k=5)

    assert resp.contract_id == "c2"
    assert resp.citations
    assert all(c.contract_id == "c2" for c in resp.citations)


@pytest.mark.asyncio
async def test_no_relevant_clauses_returns_graceful_answer():
    mcp = FakeMCPClient(search_hits=[])
    assistant = ChatAssistant(mcp, backend="extractive")

    resp = await assistant.answer("Is there a termination clause?", k=3)

    assert resp.citations == []
    assert "could not find" in resp.answer.lower()


@pytest.mark.asyncio
async def test_risk_intent_invokes_risk_tool_and_summarizes():
    mcp = FakeMCPClient(
        search_hits=[_hit("risk-1", 0, "Liability", "Liability shall be unlimited and uncapped.")],
        risk_report={
            "found": True, "contract_id": "risk-1", "overall_score": 97.8, "risk_level": "high",
            "findings": [{"severity": "high", "rationale": "Uncapped liability exposure."}],
            "severity_counts": {"high": 1},
        },
    )
    assistant = ChatAssistant(mcp, backend="extractive")

    resp = await assistant.answer("What are the biggest risks here?", contract_id="risk-1", k=3)

    assert "assess_risk" in resp.tools_used
    assert "risk" in resp.answer.lower()


@pytest.mark.asyncio
async def test_risk_intent_without_contract_does_not_call_risk_tool():
    mcp = FakeMCPClient(search_hits=[_hit("c1", 0, "Unclassified", "Some ordinary scheduling clause.")])
    assistant = ChatAssistant(mcp, backend="extractive")

    resp = await assistant.answer("what are the risks?", k=3)  # no contract_id

    assert "assess_risk" not in resp.tools_used
    assert "assess_risk" not in mcp.calls

"""Tests for executive report assembly (app/reports/generator.py).

Report *assembly* is fully deterministic and tested without any model; the
LLM-narrative path is tested with a fake backend so no model download is
required here either.
"""

import pytest

from app.reports.generator import build_report


class FakeMCPClient:
    def __init__(self, report_data: dict):
        self._report_data = report_data

    async def call_tool(self, name: str, **kwargs) -> dict:
        assert name == "build_report_data"
        return self._report_data


class FakeLLMBackend:
    def __init__(self, reply: str | None):
        self._reply = reply

    def generate_raw(self, messages, max_new_tokens=180):
        return self._reply


_SAMPLE_DATA = {
    "found": True,
    "contract_id": "r1",
    "filename": "r1.txt",
    "metadata": {"contract_type": "NDA", "parties": ["Acme", "Globex"], "num_clauses": 2, "num_chars": 500},
    "categories": {"Liability": 1, "Governing Law": 1},
    "risk_level": "high",
    "overall_score": 97.8,
    "top_findings": [
        {"rule_id": "liability_uncapped", "category": "Liability", "severity": "high",
         "rationale": "Uncapped liability exposure.", "clause_index": 0,
         "evidence_text": "unlimited and uncapped", "start_offset": 0, "end_offset": 10},
    ],
}


@pytest.mark.asyncio
async def test_build_report_returns_none_for_unknown_contract():
    mcp = FakeMCPClient({"found": False})
    report = await build_report("nope", mcp)
    assert report is None


@pytest.mark.asyncio
async def test_build_report_uses_template_narrative_without_llm_backend():
    mcp = FakeMCPClient(_SAMPLE_DATA)
    report = await build_report("r1", mcp, llm_backend=None)

    assert report is not None
    assert report.contract_id == "r1"
    assert report.contract_type == "NDA"
    assert report.risk_level == "high"
    assert len(report.top_findings) == 1
    assert report.narrative is not None
    assert "NDA" in report.narrative
    assert "high" in report.narrative


@pytest.mark.asyncio
async def test_build_report_uses_llm_narrative_when_available():
    mcp = FakeMCPClient(_SAMPLE_DATA)
    llm = FakeLLMBackend(reply="This NDA carries significant uncapped liability risk.")

    report = await build_report("r1", mcp, llm_backend=llm)

    assert report.narrative == "This NDA carries significant uncapped liability risk."


@pytest.mark.asyncio
async def test_build_report_falls_back_to_template_when_llm_returns_none():
    mcp = FakeMCPClient(_SAMPLE_DATA)
    llm = FakeLLMBackend(reply=None)  # simulates load/generation failure

    report = await build_report("r1", mcp, llm_backend=llm)

    assert report.narrative is not None
    assert "NDA" in report.narrative

"""Executive report generation (Checkpoint 4).

Assembles an ExecutiveReport from the MCP `build_report_data` tool's
structured output, then attaches a short narrative: an LLM-written summary
when a working LocalLLMBackend is supplied, otherwise a deterministic
template sentence built from the same data. The structured fields
(categories, risk_level, overall_score, top_findings) are always present and
are what's actually tested; the narrative is presentational only.
"""

from __future__ import annotations

import anyio

from app.models.analysis import RiskFinding
from app.models.report import ExecutiveReport


def _template_narrative(data: dict) -> str:
    n_clauses = sum(data["categories"].values())
    n_categories = len(data["categories"])
    top = data["top_findings"]
    driver = top[0]["rationale"] if top else "no significant findings"
    return (
        f"This {data['metadata']['contract_type']} contract has {n_clauses} "
        f"clauses across {n_categories} categories; overall risk is "
        f"{data['risk_level']} ({data['overall_score']:.0f}/100), driven "
        f"primarily by {driver}"
    )


def _llm_narrative(data: dict, llm_backend) -> str | None:
    prompt = (
        "Write a 2-4 sentence executive summary of this contract review, in "
        "plain English, using only the facts given below. Do not invent "
        "details.\n\n"
        f"Contract type: {data['metadata']['contract_type']}\n"
        f"Clause categories: {data['categories']}\n"
        f"Overall risk: {data['risk_level']} ({data['overall_score']:.0f}/100)\n"
        f"Top findings: {[f['rationale'] for f in data['top_findings']]}\n"
    )
    messages = [{"role": "user", "content": prompt}]
    return llm_backend.generate_raw(messages, max_new_tokens=180)


async def build_report(contract_id: str, mcp_client, llm_backend=None) -> ExecutiveReport | None:
    """Build an ExecutiveReport for `contract_id`, or None if it doesn't exist."""
    data = await mcp_client.call_tool("build_report_data", contract_id=contract_id, top_findings=5)
    if not data.get("found"):
        return None

    narrative = None
    if llm_backend is not None:
        # _llm_narrative ultimately calls the synchronous transformers
        # pipeline (LocalLLMBackend.generate_raw) -- run it in a worker
        # thread so it doesn't block this coroutine's event loop (and the
        # MCP subprocess's own stdio I/O) for the duration of generation.
        narrative = await anyio.to_thread.run_sync(_llm_narrative, data, llm_backend)
    if not narrative:
        narrative = _template_narrative(data)

    return ExecutiveReport(
        contract_id=data["contract_id"],
        filename=data["filename"],
        contract_type=data["metadata"]["contract_type"],
        parties=data["metadata"]["parties"],
        categories=data["categories"],
        risk_level=data["risk_level"],
        overall_score=data["overall_score"],
        top_findings=[RiskFinding(**f) for f in data["top_findings"]],
        narrative=narrative,
    )

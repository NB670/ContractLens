"""MCP server exposing ContractLens services as tools (Checkpoint 4).

The project plan calls for "MCP-style tool integrations ... to expose these
services to the LLM interface." This is the literal realization of that: a
real MCP server (built with FastMCP, running over stdio as its own process),
not an in-process function registry. Both the chat assistant
(``app/chat/assistant.py``) and the report generator
(``app/reports/generator.py``) reach these services exclusively through the
MCP client (``app/mcp/client.py``) -- there is no direct-call path left that
bypasses the MCP boundary.

Because this module runs as a separate OS process from the main FastAPI app,
it cannot share the main process's in-memory ``ClauseIndex`` -- separate
processes don't share memory. Instead this module keeps its own
``ClauseIndex``, lazily synced from the persistent SQLite ``store`` (which
*is* shared, as a file both processes open): before serving a search, any
contract id present in the store that this process hasn't embedded yet gets
added once. Already-synced contracts are never re-embedded. At this
project's corpus size the sync cost is negligible; this is a documented
tradeoff of the subprocess architecture, not an oversight.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.models.analysis import RiskReport
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.index import ClauseIndex
from app.risk.analyzer import analyze_risk
from app.store import store

_index = ClauseIndex(embedder=HashingEmbedder())
_indexed_ids: set[str] = set()

mcp_app = FastMCP("contractlens")


def _sync_index_with_store() -> None:
    """Add any not-yet-indexed contract from the store to ``_index``."""
    for contract_id in store.list_ids():
        if contract_id in _indexed_ids:
            continue
        contract = store.get(contract_id)
        if contract is not None:
            _index.add_contract(contract)
        _indexed_ids.add(contract_id)


@mcp_app.tool()
def search_clauses(
    query: str, k: int = 5, contract_id: str | None = None, category: str | None = None
) -> dict:
    """Find clauses most semantically relevant to a natural-language query.

    Scoped to one contract when ``contract_id`` is given, otherwise searches
    every contract in the store.
    """
    _sync_index_with_store()
    if contract_id is None:
        hits = _index.search(query, k=k, category=category)
    else:
        overfetch = max(k * 4, 20)
        hits = [
            h for h in _index.search(query, k=overfetch, category=category)
            if h.contract_id == contract_id
        ][:k]
    return {"hits": [h.model_dump() for h in hits]}


@mcp_app.tool()
def assess_risk(contract_id: str) -> dict:
    """Run the evidence-backed risk engine over a stored contract."""
    contract = store.get(contract_id)
    if contract is None:
        return {"found": False}
    report: RiskReport = analyze_risk(contract)
    return {"found": True, **report.model_dump()}


@mcp_app.tool()
def build_report_data(contract_id: str, top_findings: int = 5) -> dict:
    """Bundle the structured inputs an executive report needs, in one call."""
    contract = store.get(contract_id)
    if contract is None:
        return {"found": False}
    report = analyze_risk(contract)
    return {
        "found": True,
        "contract_id": contract.id,
        "filename": contract.filename,
        "metadata": contract.metadata.model_dump(),
        "categories": contract.categories_present(),
        "risk_level": report.risk_level,
        "overall_score": report.overall_score,
        "top_findings": [f.model_dump() for f in report.findings[:top_findings]],
    }


if __name__ == "__main__":  # pragma: no cover - process entry point
    mcp_app.run(transport="stdio")

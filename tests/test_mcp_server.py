"""Tests for MCP server tool functions (app/mcp/server.py).

@mcp_app.tool() registers a function but returns the original callable
unchanged, so these call the tool functions directly -- no subprocess or MCP
transport involved, same fast/offline posture as the rest of the suite.
"""

import pytest

from app.mcp import server as mcp_server
from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.index import ClauseIndex
from app.store import ContractStore


@pytest.fixture(autouse=True)
def _isolated_server_state(monkeypatch):
    """Isolate every test in this module from the real contractlens.db.

    app/mcp/server.py's tool functions read the module-level `store` name
    directly. Monkeypatching it to a fresh in-memory ContractStore per test
    (the same isolation pattern tests/test_store.py already uses) keeps
    these tests deterministic regardless of whatever's already in the local
    contractlens.db from running the app manually -- without this, tests
    that assert on *which* contract ranks first can pass or fail depending
    on unrelated local data.
    """
    monkeypatch.setattr(mcp_server, "store", ContractStore(database_url="sqlite://"))
    mcp_server._index = ClauseIndex(embedder=HashingEmbedder())
    mcp_server._indexed_ids.clear()


def _clause(index: int, text: str, category: str = "Unclassified") -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def _contract(cid: str, texts: list[str], categories: list[str] | None = None) -> Contract:
    cats = categories or ["Unclassified"] * len(texts)
    return Contract(
        id=cid, filename=f"{cid}.txt", source_format="txt",
        clauses=[_clause(i, t, c) for i, (t, c) in enumerate(zip(texts, cats))],
    )


def test_search_clauses_syncs_new_contracts_from_store():
    mcp_server.store.add(_contract("mcp-search-1", ["Confidential information must be protected."], ["Confidentiality"]))

    result = mcp_server.search_clauses(query="confidentiality", k=5)

    assert result["hits"]
    assert result["hits"][0]["contract_id"] == "mcp-search-1"
    assert "mcp-search-1" in mcp_server._indexed_ids


def test_search_clauses_does_not_reembed_already_synced_contract(monkeypatch):
    mcp_server.store.add(_contract("mcp-search-2", ["Termination for convenience is allowed."], ["Termination"]))
    mcp_server.search_clauses(query="termination", k=5)  # first call indexes it

    calls = []
    original_add = mcp_server._index.add_contract

    def spy_add(contract):
        calls.append(contract.id)
        return original_add(contract)

    monkeypatch.setattr(mcp_server._index, "add_contract", spy_add)
    mcp_server.search_clauses(query="termination", k=5)  # second call: nothing new to sync

    assert calls == []


def test_search_clauses_filters_by_contract_id():
    mcp_server.store.add(_contract("mcp-a", ["termination for convenience"], ["Termination"]))
    mcp_server.store.add(_contract("mcp-b", ["termination requires written cause"], ["Termination"]))

    result = mcp_server.search_clauses(query="termination", k=5, contract_id="mcp-b")

    assert result["hits"]
    assert all(h["contract_id"] == "mcp-b" for h in result["hits"])


def test_assess_risk_returns_report_for_known_contract():
    mcp_server.store.add(_contract(
        "mcp-risk-1",
        ["Each party's liability shall be unlimited and uncapped for any breach."],
        ["Liability"],
    ))

    result = mcp_server.assess_risk(contract_id="mcp-risk-1")

    assert result["found"] is True
    assert result["risk_level"] in ("low", "medium", "high")
    assert result["findings"]


def test_assess_risk_not_found_for_unknown_contract():
    assert mcp_server.assess_risk(contract_id="mcp-does-not-exist") == {"found": False}


def test_build_report_data_bundles_metadata_categories_and_top_findings():
    mcp_server.store.add(_contract(
        "mcp-report-1",
        [
            "Each party's liability shall be unlimited and uncapped for any breach.",
            "This agreement is governed by the laws of the State of Georgia.",
        ],
        ["Liability", "Governing Law"],
    ))

    data = mcp_server.build_report_data(contract_id="mcp-report-1", top_findings=3)

    assert data["found"] is True
    assert data["filename"] == "mcp-report-1.txt"
    assert "Liability" in data["categories"]
    assert len(data["top_findings"]) <= 3


def test_build_report_data_not_found_for_unknown_contract():
    assert mcp_server.build_report_data(contract_id="mcp-nope") == {"found": False}

"""End-to-end test of MCPToolClient against the real MCP server subprocess.

Unlike the rest of the suite this genuinely launches a subprocess (python -m
app.mcp.server) and talks MCP over stdio -- no network, no model download,
just a local process, so it stays offline. This is the one test that proves
the whole MCP boundary (not just the tool functions in isolation) actually
works end-to-end.

Isolation note: the client process and the server subprocess are different
OS processes, so a plain `monkeypatch.setattr` in this test cannot redirect
the server subprocess's `store` singleton the way tests/test_mcp_server.py
redirects it in-process. Instead each test points CONTRACTLENS_DB_PATH (via
MCPToolClient's `env` passthrough) at its own temp-file SQLite database, and
writes to that same database directly (not the shared `app.store.store`
singleton) before calling the tool -- so both this test process and the
server subprocess open the identical isolated file, deterministic
regardless of whatever's in the real local contractlens.db.
"""

import pytest

from app.mcp.client import MCPToolClient, MCPToolError
from app.models.contract import Clause, Contract
from app.store import ContractStore


def _contract(cid: str, text: str, category: str) -> Contract:
    clause = Clause(
        index=0, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )
    return Contract(id=cid, filename=f"{cid}.txt", source_format="txt", clauses=[clause])


def _isolated_client(tmp_path) -> MCPToolClient:
    db_path = tmp_path / "test.db"
    return MCPToolClient(env={"CONTRACTLENS_DB_PATH": str(db_path)}), db_path


@pytest.mark.asyncio
async def test_call_tool_round_trips_search_clauses(tmp_path):
    client, db_path = _isolated_client(tmp_path)
    ContractStore(database_url=f"sqlite:///{db_path}").add(
        _contract("mcp-client-1", "Confidential information must be protected.", "Confidentiality")
    )
    try:
        result = await client.call_tool("search_clauses", query="confidentiality", k=5)
        assert result["hits"]
        assert result["hits"][0]["contract_id"] == "mcp-client-1"
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_call_tool_round_trips_not_found(tmp_path):
    client, _ = _isolated_client(tmp_path)
    try:
        result = await client.call_tool("assess_risk", contract_id="does-not-exist")
        assert result == {"found": False}
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_call_tool_raises_mcp_tool_error_for_unknown_tool(tmp_path):
    client, _ = _isolated_client(tmp_path)
    try:
        with pytest.raises(MCPToolError):
            await client.call_tool("not_a_real_tool")
    finally:
        await client.stop()

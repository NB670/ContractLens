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


@pytest.mark.asyncio
async def test_start_failure_closes_the_partial_stack(monkeypatch):
    # This command exits immediately without ever speaking the MCP
    # handshake, so stdio_client/ClientSession entry succeeds but
    # session.initialize() never gets a response and fails. start() must
    # close the partially-entered AsyncExitStack in that case instead of
    # leaking the spawned subprocess.
    #
    # Just asserting client._stack/_session are None afterward is NOT
    # sufficient: those attributes are only assigned after the try block
    # succeeds in both the buggy and the fixed version of start(), so that
    # assertion alone passes identically whether or not stack.aclose() was
    # ever called. Spy on AsyncExitStack.aclose to actually observe that
    # teardown of the partially-entered stack happened.
    from contextlib import AsyncExitStack

    aclose_calls = []
    original_aclose = AsyncExitStack.aclose

    async def spy_aclose(self):
        aclose_calls.append(self)
        return await original_aclose(self)

    monkeypatch.setattr(AsyncExitStack, "aclose", spy_aclose)

    client = MCPToolClient(command=["python3", "-c", "import sys; sys.exit(1)"])
    with pytest.raises(Exception):
        await client.start()

    # In this mcp version, closing our stack cascades into the ClientSession's
    # own internal AsyncExitStack (also patched, since the spy is installed on
    # the class), so the fixed code observably closes >=1 stacks; the buggy
    # code (verified locally by temporarily reverting the try/except in
    # app/mcp/client.py) closes exactly 0, since nothing ever calls
    # stack.aclose() on the path where initialize() raises.
    assert len(aclose_calls) >= 1
    assert client._stack is None
    assert client._session is None

    # stop() after a failed start() must still be a safe no-op.
    await client.stop()

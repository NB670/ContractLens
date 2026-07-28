# Checkpoint 4 Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Checkpoint 4 "Platform and evaluation" milestone: a ContractLens dashboard UI, an MCP-backed retrieval-grounded chatbot with a small local instruct LLM, an executive-report generator, and a committed/reproducible benchmark artifact.

**Architecture:** A real MCP server (`app/mcp/server.py`, FastMCP over stdio) is the single service boundary for search/risk/report-data, run as a subprocess and called from the main FastAPI process via `MCPToolClient`. `ChatAssistant` and `app/reports/generator.py` both consume it — neither talks to `ClauseIndex`/`analyze_risk` directly. The FastAPI app (`app/main.py`) gains a dashboard, chat routes, and report routes on top of the unchanged CP2/CP3 routes.

**Tech Stack:** FastAPI, `mcp` (Model Context Protocol Python SDK, FastMCP), `transformers` (Qwen2.5-0.5B-Instruct, optional/soft dependency), existing `app.retrieval`/`app.risk`/`app.store`.

## Global Constraints

- Python 3.13, existing venv already has `mcp` (1.28.1), `transformers` (5.7.0), `sentence-transformers` (5.6.0) installed.
- Every new heavy/optional dependency (here: the LLM backend) must degrade gracefully — lazy `_ensure_*`-style load with a fallback, matching the existing pattern in `app/clauses/classifier.py`'s `LegalBertClassifier` and `app/retrieval/embedder.py`'s `SentenceTransformerEmbedder`. Nothing may hard-fail if `transformers`/the model is unavailable.
- Tests stay offline and fast by default: no network calls, no real model downloads required to pass `python -m pytest -q`. Any test that genuinely needs the real MCP subprocess or the real LLM is allowed (this project already accepts real end-to-end tests, e.g. `test_pipeline.py`), but the LLM backend itself is never required — those tests use `backend="extractive"` or a fake, exactly like the existing `test_chat.py` scaffold does.
- Commit messages: plain, descriptive, no AI/assistant attribution or mentions — matches every prior commit in this repo's history.
- Follow existing code style: `from __future__ import annotations`, dataclasses for internal state, Pydantic `BaseModel` for API-facing shapes, module-level docstrings explaining the *why*.

---

### Task 1: Dependencies and config

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py`

**Interfaces:**
- Produces: `settings.chat_backend: str` ("local" | "extractive"), `settings.chat_model: str` (HF model id), both already exist in the scaffold's `app/config.py` diff — this task adds them for real, pointed at the actual model this plan uses.

- [ ] **Step 1: Add the `mcp` dependency and update the chat model comment**

Edit `requirements.txt`, adding after the `scipy`/`numpy` comparison block:

```
# MCP server for the CP4 chatbot / report tool integrations. Imported in
# app/mcp/server.py (FastMCP) and app/mcp/client.py (ClientSession).
mcp>=1.28
```

- [ ] **Step 2: Add chat settings to `app/config.py`**

Edit `app/config.py`, adding after the `retrieval_model` field, before the closing of the `Settings` dataclass:

```python
    # Chatbot answer backend (Checkpoint 4): "local" (default; a local
    # transformers text-generation model, with an extractive fallback if the
    # dependency/model is unavailable) or "extractive" (dependency-free,
    # deterministic answers composed from retrieved clauses).
    chat_backend: str = os.environ.get("CONTRACTLENS_CHAT_BACKEND", "local")

    # Local HuggingFace instruction-tuned model id used when
    # chat_backend == "local". Small enough to run on CPU; instruction-tuned
    # so it actually follows the "answer only from these clauses" prompt,
    # unlike a raw completion model.
    chat_model: str = os.environ.get(
        "CONTRACTLENS_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
    )
```

- [ ] **Step 3: Verify config imports cleanly**

Run: `python3 -c "from app.config import settings; print(settings.chat_backend, settings.chat_model)"`
Expected: `local Qwen/Qwen2.5-0.5B-Instruct`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt app/config.py
git commit -m "Add mcp dependency and chat backend/model config for Checkpoint 4"
```

---

### Task 2: MCP server tool functions

**Files:**
- Create: `app/mcp/__init__.py`
- Create: `app/mcp/server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `app.retrieval.index.ClauseIndex` (`add_contract`, `search`), `app.retrieval.embedder.HashingEmbedder`, `app.risk.analyzer.analyze_risk`, `app.store.store` (`list_ids`, `get`) — all existing, unchanged.
- Produces (consumed by Task 3's FastMCP wiring and Task 4/5's tool callers via the client): three plain functions, each returning a JSON-serializable `dict`:
  - `search_clauses(query: str, k: int = 5, contract_id: str | None = None, category: str | None = None) -> dict` → `{"hits": [<RetrievalHit.model_dump() shape>, ...]}`
  - `assess_risk(contract_id: str) -> dict` → `{"found": False}` or `{"found": True, **RiskReport.model_dump()}`
  - `build_report_data(contract_id: str, top_findings: int = 5) -> dict` → `{"found": False}` or `{"found": True, "contract_id", "filename", "metadata": ContractMetadata.model_dump(), "categories": dict[str,int], "risk_level", "overall_score", "top_findings": [RiskFinding.model_dump(), ...]}`

- [ ] **Step 1: Write the failing tests**

Create `app/mcp/__init__.py` (empty file, just a package marker) first so the test import resolves:

```python
"""MCP server + client for the Checkpoint 4 chatbot and report tools."""
```

Create `tests/test_mcp_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_server.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'app.mcp.server'` (the module doesn't exist yet).

- [ ] **Step 3: Implement `app/mcp/server.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp_server.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add app/mcp/__init__.py app/mcp/server.py tests/test_mcp_server.py
git commit -m "Add MCP server exposing search/risk/report-data as real MCP tools"
```

---

### Task 3: MCP client wrapper + real subprocess round-trip test

**Files:**
- Create: `app/mcp/client.py`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Consumes: `app.mcp.server` (run as `python -m app.mcp.server` subprocess), `mcp.ClientSession`, `mcp.StdioServerParameters`, `mcp.client.stdio.stdio_client`.
- Produces (consumed by Task 4's `ChatAssistant` and Task 5's report generator):
  - `class MCPToolClient` with `def __init__(self, command: list[str] | None = None, env: dict[str, str] | None = None)`, `async def start(self) -> None`, `async def stop(self) -> None`, `async def call_tool(self, name: str, **kwargs) -> dict`.
  - `class MCPToolError(RuntimeError)` — raised by `call_tool` when the tool call errors.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp_client.py`:

```python
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
```

- [ ] **Step 2: Add `pytest-asyncio` and run to verify the test fails**

`pytest-asyncio` isn't yet a dependency. Add to `requirements.txt` under the existing `# Testing` block:

```
pytest-asyncio>=0.24
```

Add a `pytest.ini` (repo has none yet) so `@pytest.mark.asyncio` tests run automatically without individually marking every async def:

```ini
[pytest]
asyncio_mode = auto
```

Run: `pip install pytest-asyncio && python -m pytest tests/test_mcp_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.mcp.client'`

- [ ] **Step 3: Implement `app/mcp/client.py`**

```python
"""MCP client wrapper: launches app/mcp/server.py as a subprocess and speaks
MCP over stdio, exposing a single async `call_tool` used by the chat
assistant and the report generator.

The subprocess + session are started lazily on first use and kept open for
the lifetime of this client instance -- restarting the subprocess per call
would also throw away the server's synced ClauseIndex (see server.py's
module docstring), so the session is a long-lived resource, not a
per-request one.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call returns an error result."""


class MCPToolClient:
    def __init__(
        self, command: list[str] | None = None, env: dict[str, str] | None = None
    ) -> None:
        self._command = command or [sys.executable, "-m", "app.mcp.server"]
        self._env = env
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self._command[0], args=self._command[1:], env=self._env
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._stack = stack
        self._session = session

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call_tool(self, name: str, **kwargs) -> dict:
        if self._session is None:
            await self.start()
        assert self._session is not None  # for type-checkers; start() sets it

        result = await self._session.call_tool(name, arguments=kwargs)
        text = result.content[0].text if result.content else "{}"
        if result.isError:
            raise MCPToolError(text)
        if result.structuredContent is not None:
            return result.structuredContent
        import json

        return json.loads(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp_client.py -v`
Expected: PASS (3 tests). Note: this spawns a real Python subprocess per test, so it's slower than the rest of the suite (roughly 1-2s per test) — acceptable for 3 tests.

- [ ] **Step 5: Commit**

```bash
git add app/mcp/client.py tests/test_mcp_client.py requirements.txt pytest.ini
git commit -m "Add MCPToolClient with a real stdio subprocess round-trip test"
```

---

### Task 4: Chat models + chat assistant (MCP-backed, instruct-model LLM backend)

**Files:**
- Create: `app/models/chat.py`
- Create: `app/chat/__init__.py`
- Create: `app/chat/assistant.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `app.mcp.client.MCPToolClient` (Task 3), `app.config.settings.chat_backend`/`chat_model` (Task 1).
- Produces (consumed by Task 6's `main.py` wiring): `ChatRequest`, `Citation`, `ChatResponse` (Pydantic models); `class ChatAssistant` with `async def answer(self, question: str, contract_id: str | None = None, k: int = 5) -> ChatResponse`; `def build_default_assistant(mcp_client: MCPToolClient) -> ChatAssistant`; `class LocalLLMBackend` with a public `def generate_raw(self, messages: list[dict], max_new_tokens: int) -> str | None` (reused by Task 5's report narrative).

- [ ] **Step 1: Write the failing tests**

Create `app/models/chat.py`:

```python
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
```

Create `app/chat/__init__.py`:

```python
"""Retrieval-grounded chatbot for Checkpoint 4 (RAG over the MCP tool layer)."""
```

Create `tests/test_chat.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chat.py tests/test_mcp_server.py -v` (also re-running `test_mcp_server.py` as a quick regression check)
Expected: `test_chat.py` FAILs with `ModuleNotFoundError: No module named 'app.chat.assistant'`; `test_mcp_server.py` still passes.

- [ ] **Step 3: Implement `app/chat/assistant.py`**

```python
"""Retrieval-grounded contract chatbot (Checkpoint 4).

A user asks a natural-language question about their contract(s) and gets a
*cited* answer, where every claim is grounded in clauses retrieved via the
MCP tool layer (app/mcp/), never in the model's own parametric memory. Local-
first, in keeping with the project's privacy-preserving goal.

Two backends:

  * ExtractiveBackend (dependency-free) -- composes the answer directly from
    the retrieved clauses. No model download, deterministic, always
    available.

  * LocalLLMBackend (default) -- lazily loads a small local, instruction-
    tuned transformers text-generation model (Qwen2.5-0.5B-Instruct by
    default) and asks it to answer *using only* the retrieved clauses, citing
    them by number, via the model's chat template. Falls back to the
    extractive backend if the dependency or model is unavailable.

All grounding (search_clauses, assess_risk) comes from the MCP tool layer via
an injected client -- this module never imports app.retrieval or app.risk
directly.
"""

from __future__ import annotations

import re

from app.config import settings
from app.models.chat import ChatResponse, Citation

_EXCERPT_CHARS = 320
_RISK_INTENT = re.compile(r"\brisk|risky|liabilit|indemnif|dangerous|red flag|concern", re.IGNORECASE)


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _hit_to_citation(hit: dict) -> Citation:
    return Citation(
        contract_id=hit["contract_id"],
        clause_index=hit["clause_index"],
        category=hit.get("category", "Unclassified"),
        heading=hit.get("heading"),
        excerpt=_excerpt(hit.get("text", "")),
        score=hit.get("score", 0.0),
    )


class ExtractiveBackend:
    """Compose a grounded answer straight from the retrieved clauses."""

    name = "extractive"

    def generate(self, question: str, citations: list[Citation], risk_summary: str | None) -> str:
        if not citations:
            return (
                "I could not find any clauses relevant to that question in the "
                "uploaded contract(s). Try uploading the contract first, or "
                "rephrasing the question."
            )
        lines = [f"Based on {len(citations)} relevant clause(s) in the reviewed contract(s):", ""]
        for i, cit in enumerate(citations, start=1):
            label = cit.heading or cit.category
            lines.append(
                f"[{i}] ({label}) {cit.excerpt} "
                f"— {cit.contract_id}, clause {cit.clause_index} "
                f"(similarity {cit.score:.2f})"
            )
        if risk_summary:
            lines += ["", risk_summary]
        return "\n".join(lines)


class LocalLLMBackend:
    """Local instruction-tuned transformers backend with extractive fallback."""

    name = "local-llm"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.chat_model
        self._pipe = None
        self._load_failed = False
        self._fallback = ExtractiveBackend()

    def _ensure_pipe(self) -> bool:
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False
        try:  # pragma: no cover - depends on optional heavy dep
            from transformers import pipeline

            self._pipe = pipeline("text-generation", model=self.model_name)
            return True
        except Exception:
            self._load_failed = True
            return False

    def generate_raw(self, messages: list[dict], max_new_tokens: int = 256) -> str | None:
        """Run the model on a chat-style message list; None on any failure.

        Public (not underscore-prefixed) because app/reports/generator.py
        reuses this for the executive-report narrative, sharing one model
        load instead of duplicating the pipeline-loading logic.
        """
        if not self._ensure_pipe():
            return None
        try:  # pragma: no cover - depends on optional heavy dep
            out = self._pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
            generated = out[0]["generated_text"]
            text = generated[-1]["content"].strip() if isinstance(generated, list) else str(generated).strip()
            return text or None
        except Exception:
            return None

    @staticmethod
    def _build_messages(question: str, citations: list[Citation], risk_summary: str | None) -> list[dict]:
        context_lines = [f"[{i}] ({c.category}) {c.excerpt}" for i, c in enumerate(citations, start=1)]
        context = "\n".join(context_lines) if context_lines else "(no relevant clauses found)"
        risk_block = f"\n\nRisk analysis:\n{risk_summary}" if risk_summary else ""
        system = (
            "You are ContractLens, a contract-review assistant. Answer the "
            "question using ONLY the numbered contract clauses provided. Cite "
            "the clauses you rely on by their number in square brackets, e.g. "
            "[1]. If the clauses do not contain the answer, say so plainly."
        )
        user = f"Clauses:\n{context}{risk_block}\n\nQuestion: {question}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, question: str, citations: list[Citation], risk_summary: str | None) -> str:
        text = self.generate_raw(self._build_messages(question, citations, risk_summary))
        return text or self._fallback.generate(question, citations, risk_summary)


def _make_backend(backend: str):
    if backend == "extractive":
        return ExtractiveBackend()
    return LocalLLMBackend()


class ChatAssistant:
    """RAG assistant: retrieve grounding clauses via MCP, then answer with citations.

    ``backend`` is public (not underscore-prefixed) because
    app/reports/generator.py reuses it, when it's a LocalLLMBackend, to write
    the executive-report narrative -- sharing the one loaded model instead of
    loading a second copy just for reports.
    """

    def __init__(self, mcp_client, backend: str | None = None) -> None:
        self.mcp_client = mcp_client
        self.backend = _make_backend(backend or settings.chat_backend)

    async def _risk_summary(self, contract_id: str | None) -> tuple[str | None, bool]:
        if contract_id is None:
            return None, False
        result = await self.mcp_client.call_tool("assess_risk", contract_id=contract_id)
        if not result.get("found"):
            return None, False
        top = result["findings"][:3]
        if not top:
            return (
                f"Risk analysis: overall risk {result['risk_level']} "
                f"(score {result['overall_score']:.0f}/100); no specific findings.",
                True,
            )
        bullet = "; ".join(f"{f['severity']} — {f['rationale']}" for f in top)
        return (
            f"Risk analysis: overall risk {result['risk_level']} "
            f"(score {result['overall_score']:.0f}/100). Top findings: {bullet}",
            True,
        )

    async def answer(self, question: str, contract_id: str | None = None, k: int = 5) -> ChatResponse:
        tools_used: list[str] = []

        hits: list[dict] = []
        if question and question.strip():
            result = await self.mcp_client.call_tool(
                "search_clauses", query=question, k=k, contract_id=contract_id
            )
            hits = result["hits"]
            tools_used.append("search_clauses")

        citations = [_hit_to_citation(h) for h in hits]

        risk_summary = None
        if _RISK_INTENT.search(question or ""):
            risk_summary, used = await self._risk_summary(contract_id)
            if used:
                tools_used.append("assess_risk")

        answer_text = self.backend.generate(question, citations, risk_summary)

        return ChatResponse(
            question=question, answer=answer_text, backend=self.backend.name,
            contract_id=contract_id, citations=citations, tools_used=tools_used,
        )


def build_default_assistant(mcp_client) -> ChatAssistant:
    """Construct the assistant over a given MCPToolClient."""
    return ChatAssistant(mcp_client)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chat.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/models/chat.py app/chat/__init__.py app/chat/assistant.py tests/test_chat.py
git commit -m "Add MCP-backed chat assistant with an instruct-model local LLM backend"
```

---

### Task 5: Executive report model + generator

**Files:**
- Create: `app/models/report.py`
- Create: `app/reports/__init__.py`
- Create: `app/reports/generator.py`
- Test: `tests/test_reports.py`

**Interfaces:**
- Consumes: `app.mcp.client.MCPToolClient` (via injected fake in tests), `app.chat.assistant.LocalLLMBackend.generate_raw` (Task 4, optional).
- Produces (consumed by Task 6's `main.py` wiring): `ExecutiveReport` (Pydantic model); `async def build_report(contract_id: str, mcp_client, llm_backend=None) -> ExecutiveReport | None`.

- [ ] **Step 1: Write the failing tests**

Create `app/models/report.py`:

```python
"""Structured model for the Checkpoint 4 executive report deliverable.

The original project plan's chatbot deliverable includes the ability to
"generate executive review reports summarizing key findings and risks" --
this is that report's shape. It reuses RiskFinding as-is so a report's
findings link back to the same clause evidence app/risk/analyzer.py already
produces.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.analysis import RiskFinding


class ExecutiveReport(BaseModel):
    """A one-page executive summary of a single reviewed contract."""

    contract_id: str
    filename: str
    contract_type: str = "Unknown"
    parties: list[str] = Field(default_factory=list)
    categories: dict[str, int] = Field(
        default_factory=dict, description="Clause count per identified category"
    )
    risk_level: str = "low"
    overall_score: float = Field(0.0, ge=0.0, le=100.0)
    top_findings: list[RiskFinding] = Field(default_factory=list)
    narrative: Optional[str] = Field(
        None, description="1-4 sentence plain-English summary; LLM-written when "
        "available, otherwise a deterministic template sentence"
    )
```

Create `app/reports/__init__.py`:

```python
"""Executive report generation for Checkpoint 4."""
```

Create `tests/test_reports.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.reports.generator'`

- [ ] **Step 3: Implement `app/reports/generator.py`**

```python
"""Executive report generation (Checkpoint 4).

Assembles an ExecutiveReport from the MCP `build_report_data` tool's
structured output, then attaches a short narrative: an LLM-written summary
when a working LocalLLMBackend is supplied, otherwise a deterministic
template sentence built from the same data. The structured fields
(categories, risk_level, overall_score, top_findings) are always present and
are what's actually tested; the narrative is presentational only.
"""

from __future__ import annotations

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
        narrative = _llm_narrative(data, llm_backend)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reports.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/models/report.py app/reports/__init__.py app/reports/generator.py tests/test_reports.py
git commit -m "Add executive report generator with LLM narrative and template fallback"
```

---

### Task 6: Wire dashboard, chat, and report routes into `app/main.py`

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.mcp.client.MCPToolClient` (Task 3), `app.chat.assistant.{build_default_assistant, LocalLLMBackend}` and `ChatAssistant.backend` (public attribute, Task 4), `app.reports.generator.build_report` (Task 5).
- Produces: `GET /` (dashboard HTML), `GET /health` (the old root JSON health check, moved), `POST /chat`, `GET /contracts/{id}/chat`, `GET /contracts/{id}/report`, `GET /contracts/{id}/report.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:

```python
"""HTTP-level tests for the Checkpoint 4 routes in app/main.py.

Uses FastAPI's TestClient. The dashboard/health routes never touch the MCP
client, so those stay fast. The report routes do go through the real
MCPToolClient (lazily spawning the app/mcp/server.py subprocess, same as
tests/test_mcp_client.py) since build_report's assembly logic is exactly what
these tests check end-to-end -- slower than a pure unit test (~1-2s each) but
still fully offline. /chat is deliberately NOT exercised here: its logic is
already covered by tests/test_chat.py against a FakeMCPClient, and hitting it
here would additionally load (or attempt to load) the LocalLLMBackend model,
which is unnecessary given the extractive-backend path is what's actually
under test.
"""

import io

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.chat.assistant import ExtractiveBackend
from app.main import app


@pytest.fixture(autouse=True)
def _force_extractive_backend():
    """Keep this test module offline.

    app/config.py defaults chat_backend to "local", so app.main._assistant is
    built with a LocalLLMBackend by default; the report routes reuse that
    backend for their narrative (see app/main.py's _report_llm_backend()),
    which would otherwise try to download/load an LLM the first time a report
    test runs here. The narrative logic itself (LLM vs. template) is already
    covered against a fake in tests/test_reports.py -- these tests only need
    the real MCP subprocess + build_report wiring, not a real model.
    """
    original_backend = main_module._assistant.backend
    main_module._assistant.backend = ExtractiveBackend()
    yield
    main_module._assistant.backend = original_backend


def _upload_txt(client: TestClient, filename: str, text: str) -> str:
    resp = client.post(
        "/upload", files={"file": (filename, io.BytesIO(text.encode()), "text/plain")}
    )
    assert resp.status_code == 200
    return resp.json()["id"]


def test_health_endpoint_reports_checkpoint_4():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint"] == 4
    assert body["service"] == "ContractLens"


def test_dashboard_lists_uploaded_contracts():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "dashboard-test.txt",
            "1. Confidentiality\nEach party shall keep information confidential.\n",
        )
        resp = client.get("/")

    assert resp.status_code == 200
    assert "dashboard-test.txt" in resp.text
    assert contract_id in resp.text


def test_dashboard_has_upload_form():
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "<form" in resp.text
    assert '/upload' in resp.text


def test_report_html_route_renders_for_known_contract():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "report-test.txt",
            "1. Liability\nEach party's liability shall be unlimited and uncapped for any breach.\n",
        )
        resp = client.get(f"/contracts/{contract_id}/report")

    assert resp.status_code == 200
    assert "report-test.txt" in resp.text


def test_report_json_route_matches_html_data():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "report-json-test.txt",
            "1. Liability\nEach party's liability shall be unlimited and uncapped for any breach.\n",
        )
        resp = client.get(f"/contracts/{contract_id}/report.json")

    assert resp.status_code == 200
    body = resp.json()
    assert body["contract_id"] == contract_id
    assert body["filename"] == "report-json-test.txt"


def test_report_route_404s_for_unknown_contract():
    with TestClient(app) as client:
        resp = client.get("/contracts/does-not-exist/report.json")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL — `test_health_endpoint_reports_checkpoint_4` gets 404 (no `/health` route yet); dashboard/report tests fail similarly (`/` still returns the old JSON, `/contracts/{id}/report[.json]` don't exist).

- [ ] **Step 3: Implement the `app/main.py` changes**

Modify the imports and module-level singletons near the top of `app/main.py` (replacing the current lines 21-30):

```python
from app.chat.assistant import LocalLLMBackend, build_default_assistant
from app.comparison.comparator import compare_contracts
from app.config import settings
from app.ingestion.parsers import UnsupportedFormatError, detect_format
from app.mcp.client import MCPToolClient
from app.models.chat import ChatRequest
from app.models.contract import Contract
from app.pipeline import ingest
from app.reports.generator import build_report
from app.retrieval.index import ClauseIndex
from app.risk.analyzer import analyze_risk
from app.store import store

_clause_index = ClauseIndex()
_mcp_client = MCPToolClient()
_assistant = build_default_assistant(_mcp_client)
```

Note: the report routes below pass `_assistant.backend` to `build_report` when it's a `LocalLLMBackend`, so the executive-report narrative reuses the exact same loaded model as chat instead of loading a second copy — when `chat_backend` is `"extractive"`, reports simply fall back to the template narrative (no LLM to reuse).

Replace the `FastAPI(...)` construction and the existing `root()` function (currently the app metadata + `GET /`):

```python
app = FastAPI(
    title="ContractLens",
    description=(
        "Privacy-preserving contract intelligence platform "
        "(CP4: platform UI, MCP-backed chatbot, and evaluation)"
    ),
    version="0.4.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "service": "ContractLens",
        "checkpoint": 4,
        "supported_formats": list(settings.supported_extensions),
        "classifier_backend": settings.classifier_backend,
        "chat_backend": settings.chat_backend,
    }


@app.on_event("shutdown")
async def _stop_mcp_client() -> None:
    """Terminate the MCP server subprocess, if `/chat` or `/report` ever started it.

    MCPToolClient.start() is called lazily on first use (see app/mcp/client.py),
    not eagerly at app startup, so routes that never touch chat/report pay no
    subprocess cost. stop() is a no-op if it was never started, so this is
    always safe to call unconditionally on shutdown -- without it the
    subprocess would leak past the app's lifetime (e.g. once per test that
    uses `with TestClient(app) as client:` in tests/test_main.py).
    """
    await _mcp_client.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    """Upload form + list of stored contracts -- the CP4 platform UI's home page."""
    rows = []
    for contract_id in store.list_ids():
        contract = store.get(contract_id)
        if contract is None:
            continue
        cid = html.escape(contract_id)
        fname = html.escape(contract.filename)
        rows.append(
            f"<tr><td><code>{cid}</code></td><td>{fname}</td>"
            f"<td>{html.escape(contract.metadata.contract_type)}</td>"
            f"<td>{len(contract.clauses)}</td>"
            f"<td><a href='/contracts/{cid}/view'>view</a> &middot; "
            f"<a href='/contracts/{cid}/chat'>chat</a> &middot; "
            f"<a href='/contracts/{cid}/report'>report</a></td></tr>"
        )
    table_rows = "".join(rows) or "<tr><td colspan='5'>No contracts uploaded yet.</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ContractLens</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ text-align: left; border-bottom: 1px solid #ddd; padding: .5rem; }}
form {{ margin-top: 1rem; }}
</style></head><body>
<h1>ContractLens</h1>
<p>Privacy-preserving contract intelligence platform.</p>
<form action='/upload' method='post' enctype='multipart/form-data'>
  <input type='file' name='file' required>
  <button type='submit'>Upload contract</button>
</form>
<h2>Contracts</h2>
<table>
<tr><th>ID</th><th>Filename</th><th>Type</th><th>Clauses</th><th>Actions</th></tr>
{table_rows}
</table>
</body></html>"""
```

Add the chat and report routes at the end of `app/main.py`, after the existing `contract_risk` route:

```python
# --------------------------------------------------------------------------- #
# Checkpoint 4 -- chatbot / question answering (RAG over the MCP tool layer)
# --------------------------------------------------------------------------- #
@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """Answer a natural-language question with clauses cited as grounding."""
    if request.contract_id is not None and store.get(request.contract_id) is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    response = await _assistant.answer(request.question, contract_id=request.contract_id, k=request.k)
    return response.model_dump()


@app.get("/contracts/{contract_id}/chat", response_class=HTMLResponse)
def chat_ui(contract_id: str) -> str:
    """Minimal chat UI for asking questions about one contract."""
    contract = store.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="Contract not found.")

    cid = html.escape(contract_id)
    fname = html.escape(contract.filename)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>ContractLens — Chat · {fname}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
#log {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; min-height: 8rem; }}
.msg {{ margin: .75rem 0; }}
.you {{ font-weight: 600; }}
.bot pre {{ white-space: pre-wrap; background: #f7f7f7; padding: .6rem; border-radius: 6px; }}
.cites {{ font-size: .82em; color: #555; }}
form {{ display: flex; gap: .5rem; margin-top: 1rem; }}
input[type=text] {{ flex: 1; padding: .55rem; }}
button {{ padding: .55rem 1rem; }}
.nav a {{ font-size: .85em; }}
</style></head><body>
<h1>ContractLens Chat</h1>
<p class='nav'><a href='/'>&larr; dashboard</a> &middot; Contract <code>{cid}</code> — {fname} ·
   <a href='/contracts/{cid}/view'>clause view</a> &middot;
   <a href='/contracts/{cid}/report'>report</a></p>
<div id='log'></div>
<form id='f'>
  <input type='text' id='q' placeholder='Ask about this contract…' autocomplete='off' required>
  <button type='submit'>Ask</button>
</form>
<script>
const CID = {contract_id!r};
const log = document.getElementById('log');
function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}
document.getElementById('f').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  log.innerHTML += `<div class='msg you'>You: ${{esc(q)}}</div>`;
  document.getElementById('q').value = '';
  const r = await fetch('/chat', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ question: q, contract_id: CID }})
  }});
  const data = await r.json();
  let cites = (data.citations || []).map((c, i) =>
    `[${{i + 1}}] ${{esc(c.category)}} · clause ${{c.clause_index}} (sim ${{c.score.toFixed(2)}})`
  ).join('<br>');
  log.innerHTML += `<div class='msg bot'><pre>${{esc(data.answer)}}</pre>` +
    (cites ? `<div class='cites'>${{cites}}</div>` : '') + `</div>`;
  log.scrollTop = log.scrollHeight;
}});
</script>
</body></html>"""


# --------------------------------------------------------------------------- #
# Checkpoint 4 -- executive report
# --------------------------------------------------------------------------- #
def _report_llm_backend() -> LocalLLMBackend | None:
    """Reuse the chat assistant's LocalLLMBackend for report narratives, if any.

    Returns None when chat_backend is "extractive" (no LLM loaded at all) so
    build_report falls back to its deterministic template narrative.
    """
    return _assistant.backend if isinstance(_assistant.backend, LocalLLMBackend) else None


@app.get("/contracts/{contract_id}/report.json")
async def contract_report_json(contract_id: str) -> dict:
    report = await build_report(contract_id, _mcp_client, llm_backend=_report_llm_backend())
    if report is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    return report.model_dump()


@app.get("/contracts/{contract_id}/report", response_class=HTMLResponse)
async def contract_report_html(contract_id: str) -> str:
    report = await build_report(contract_id, _mcp_client, llm_backend=_report_llm_backend())
    if report is None:
        raise HTTPException(status_code=404, detail="Contract not found.")

    findings_rows = "".join(
        f"<li><strong>{html.escape(f.severity)}</strong> ({html.escape(f.category)}): "
        f"{html.escape(f.rationale)}</li>"
        for f in report.top_findings
    ) or "<li>No significant findings.</li>"
    categories_rows = "".join(
        f"<li>{html.escape(cat)}: {count}</li>" for cat, count in report.categories.items()
    )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ContractLens — Report · {html.escape(report.filename)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; }}
.narrative {{ background: #f7f7f7; padding: 1rem; border-radius: 6px; }}
.nav a {{ font-size: .85em; }}
</style></head><body>
<p class='nav'><a href='/'>&larr; dashboard</a> &middot;
   <a href='/contracts/{html.escape(report.contract_id)}/chat'>chat</a></p>
<h1>Executive Report — {html.escape(report.filename)}</h1>
<p><strong>Type:</strong> {html.escape(report.contract_type)} &middot;
   <strong>Risk:</strong> {html.escape(report.risk_level)} ({report.overall_score:.0f}/100)</p>
<div class='narrative'>{html.escape(report.narrative or "")}</div>
<h2>Clause categories</h2>
<ul>{categories_rows or '<li>None identified.</li>'}</ul>
<h2>Top risk findings</h2>
<ul>{findings_rows}</ul>
</body></html>"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS (6 tests)

Then run the full suite to check nothing else regressed:

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "Add ContractLens dashboard, chat routes, and executive report routes"
```

---

### Task 7: Benchmark runner reproducibility (`--out` results artifact)

**Files:**
- Create: `scripts/run_benchmarks.py`
- Test: `tests/test_run_benchmarks.py`

**Interfaces:**
- Consumes: `app.clauses.classifier.{RuleBasedClassifier, LegalBertClassifier}`, `app.retrieval.embedder.{HashingEmbedder, SentenceTransformerEmbedder}`, `scripts.evaluate_clauses.{load_sample, score}`, `scripts.evaluate_retrieval.evaluate` — all existing, unchanged.
- Produces: `def run(classifier_name="rule", retrieval_backend="hashing", k=5, limit=None) -> dict` (combined summary); CLI writes `results/benchmark_<UTC-ISO-timestamp>.json` by default.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_benchmarks.py`:

```python
"""Tests for the consolidated Checkpoint 4 benchmark runner.

Runs against the committed CUAD sample with the offline default backends
(rule classifier + hashing embedder), on a small `limit` slice for speed, and
checks the --out artifact is written as valid JSON.
"""

import json

from scripts.run_benchmarks import run, write_results


def test_run_benchmarks_smoke_hashing_rule():
    summary = run(classifier_name="rule", retrieval_backend="hashing", k=5, limit=40)

    assert summary["dataset"] == "cuad_sample.json"
    assert summary["records_scored"] == 40

    macro = summary["classification"]["macro_avg"]
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= macro[key] <= 1.0

    ret = summary["retrieval"]
    assert ret["backend"] == "hashing"
    assert ret["k"] == 5
    for key in ("recall_at_k", "success_at_k", "mrr"):
        assert 0.0 <= ret[key] <= 1.0
    assert ret["queries"] >= 0


def test_run_benchmarks_reports_per_category_without_macro_row():
    summary = run(limit=60)
    per_category = summary["classification"]["per_category"]
    assert per_category, "expected at least one scored category"
    assert "macro_avg" not in per_category


def test_write_results_produces_valid_json_file(tmp_path):
    summary = run(limit=20)
    out_path = tmp_path / "benchmark.json"

    write_results(summary, out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded == summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_benchmarks'` (this is a new file in this repo; the scaffold's version was never committed here).

- [ ] **Step 3: Implement `scripts/run_benchmarks.py`**

```python
"""Consolidated Checkpoint 4 benchmark runner.

Run: python -m scripts.run_benchmarks [--classifier rule|legalbert]
                                      [--retrieval-backend hashing|sentence]
                                      [--k 5] [--limit N] [--json]
                                      [--out results/benchmark_<timestamp>.json]

The CP4 milestone is to "evaluate classification and retrieval performance
with benchmark datasets." Rather than duplicate any scoring logic, this
driver reuses the harnesses already committed in Checkpoint 2/3 --
scripts/evaluate_clauses.py (classification P/R/F1) and
scripts/evaluate_retrieval.py (Recall@K / Success@K / MRR) -- over the same
committed CUAD sample (data/cuad_sample.json), and writes one combined
report. By default every run is also written to results/ as a timestamped
JSON file (--out), so headline numbers are reproducible directly from the
repository rather than living only in report screenshots -- the concrete
suggestion from the Checkpoint 3 TA feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.clauses.classifier import LegalBertClassifier, RuleBasedClassifier
from app.retrieval.embedder import HashingEmbedder, SentenceTransformerEmbedder
from scripts.evaluate_clauses import load_sample
from scripts.evaluate_clauses import score as score_classification
from scripts.evaluate_retrieval import evaluate as evaluate_retrieval

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _make_classifier(name: str):
    if name == "legalbert":
        clf = LegalBertClassifier()
        if not clf._ensure_embed_fn():
            print(
                "WARNING: LegalBERT embedding backend unavailable — classification "
                "numbers are the RuleBasedClassifier fallback.",
                file=sys.stderr,
            )
        return clf
    return RuleBasedClassifier()


def run(
    classifier_name: str = "rule",
    retrieval_backend: str = "hashing",
    k: int = 5,
    limit: int | None = None,
) -> dict:
    """Run both benchmarks over the CUAD sample and return a combined summary."""
    records = load_sample()
    if limit is not None:
        records = records[:limit]

    classification = score_classification(_make_classifier(classifier_name), records)

    embedder = (
        SentenceTransformerEmbedder() if retrieval_backend == "sentence" else HashingEmbedder()
    )
    retrieval = evaluate_retrieval(records, embedder, k=k)

    return {
        "dataset": "cuad_sample.json",
        "records_scored": len(records),
        "classification": {
            "backend": classifier_name,
            "macro_avg": classification["macro_avg"],
            "per_category": {c: m for c, m in classification.items() if c != "macro_avg"},
        },
        "retrieval": {"backend": retrieval_backend, **retrieval},
    }


def print_report(summary: dict) -> None:
    print("=" * 60)
    print("ContractLens benchmark report")
    print(f"dataset          : {summary['dataset']} ({summary['records_scored']} clauses)")
    print("-" * 60)
    clf = summary["classification"]
    macro = clf["macro_avg"]
    print(f"Clause classification (backend={clf['backend']})")
    print(f"  macro precision: {macro['precision']:.3f}")
    print(f"  macro recall   : {macro['recall']:.3f}")
    print(f"  macro F1       : {macro['f1']:.3f}")
    print("-" * 60)
    ret = summary["retrieval"]
    print(f"Semantic retrieval (backend={ret['backend']})")
    print(f"  queries        : {ret['queries']}")
    print(f"  Recall@{ret['k']}      : {ret['recall_at_k']:.4f}")
    print(f"  Success@{ret['k']}     : {ret['success_at_k']:.4f}")
    print(f"  MRR            : {ret['mrr']:.4f}")
    print("=" * 60)


def write_results(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def _default_out_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"benchmark_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", choices=["rule", "legalbert"], default="rule")
    parser.add_argument("--retrieval-backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N clauses (smoke run).")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON instead of the report.")
    parser.add_argument(
        "--out", type=str, default="__default__",
        help="Write the summary JSON to this path (default: results/benchmark_<timestamp>.json). "
             "Pass an empty string to skip writing.",
    )
    args = parser.parse_args()

    summary = run(
        classifier_name=args.classifier,
        retrieval_backend=args.retrieval_backend,
        k=args.k,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)

    if args.out != "":
        out_path = Path(args.out) if args.out != "__default__" else _default_out_path()
        write_results(summary, out_path)
        print(f"\nWrote results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_benchmarks.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_benchmarks.py tests/test_run_benchmarks.py
git commit -m "Add consolidated benchmark runner with a committed results artifact"
```

---

### Task 8: README update and final full-suite verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update "What's implemented"**

In `README.md`, after the existing "Evidence-backed risk analysis" bullet (the last bullet in the "What's implemented" list), add:

```markdown
- A **ContractLens dashboard** (`GET /`) — upload form plus a list of stored
  contracts linking to their view/chat/report pages.
- A **retrieval-grounded chatbot** (`app/chat/`) backed by a real **MCP
  server** (`app/mcp/`) — `POST /chat` / `GET /contracts/{id}/chat`. Two
  answer backends: a small local instruction-tuned LLM
  (`Qwen/Qwen2.5-0.5B-Instruct`, `app/chat/assistant.py`) with a
  dependency-free extractive fallback.
- **Executive report generation** (`app/reports/`) — `GET
  /contracts/{id}/report` (HTML) / `GET /contracts/{id}/report.json` —
  contract metadata, clause-category breakdown, and top risk findings, plus
  an LLM-written (or template) narrative summary.
- A **consolidated benchmark runner** (`scripts/run_benchmarks.py`) combining
  the classification and retrieval eval harnesses into one report, writing a
  timestamped, committed JSON artifact under `results/`.
```

- [ ] **Step 2: Update the Project layout tree**

In `README.md`'s "Project layout" code block, change the `app/main.py` line and add the new packages/files. Replace:

```
  main.py                  FastAPI app + routes (upload/view, search/similar, compare, risk)
```

with:

```
  main.py                  FastAPI app + routes (dashboard, upload/view, search/similar,
                             compare, risk, chat, report)
```

Add these blocks after the existing `risk/` block (before `scripts/`):

```
  mcp/
    server.py                    real MCP server (FastMCP/stdio): search_clauses,
                                   assess_risk, build_report_data tools
    client.py                    MCPToolClient — subprocess + ClientSession wrapper
  chat/
    assistant.py                 RAG ChatAssistant; ExtractiveBackend + LocalLLMBackend
  reports/
    generator.py                 executive report assembly + narrative (LLM or template)
```

Add to `models/`:

```
    chat.py                      ChatRequest / Citation / ChatResponse models
    report.py                     ExecutiveReport model
```

Add to `scripts/` (after `evaluate_risk.py`):

```
  run_benchmarks.py           combined classification + retrieval report; writes results/
```

Add a new top-level block after `data/`:

```
results/
  benchmark_<timestamp>.json  committed benchmark runs (scripts/run_benchmarks.py)
```

Add to `tests/` (after `test_evaluate_risk.py`):

```
  test_mcp_server.py
  test_mcp_client.py
  test_chat.py
  test_reports.py
  test_main.py
  test_run_benchmarks.py
```

- [ ] **Step 3: Update the Configuration table**

In the Configuration table, add two rows after the `CONTRACTLENS_DB_PATH` row:

```markdown
| `CONTRACTLENS_CHAT_BACKEND` | `local` | chat backend: `local` or `extractive` |
| `CONTRACTLENS_CHAT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HF model id for the local chat backend |
```

- [ ] **Step 4: Add a "Checkpoint 4" section**

Insert a new section after the existing "## Checkpoint 3 — semantic retrieval, comparison, and risk analysis" section (after its "Trying the Checkpoint 3 endpoints" subsection, before "## Datasets"):

```markdown
## Checkpoint 4 — platform, chatbot, and evaluation

Building on the CP3 services, this checkpoint adds a UI, a chatbot, and
report generation, all going through one real MCP server rather than direct
function calls:

- **MCP server** (`app/mcp/server.py`) — a `FastMCP` server run as a
  subprocess over stdio, exposing `search_clauses`, `assess_risk`, and
  `build_report_data` as real MCP tools. `app/mcp/client.py`'s
  `MCPToolClient` is the only way the rest of the app reaches these
  services — the chat assistant and the report generator both go through it.
- **Chatbot** (`app/chat/assistant.py`) — retrieval-grounded (RAG) question
  answering with cited clauses. `LocalLLMBackend` uses a small
  instruction-tuned model (`Qwen/Qwen2.5-0.5B-Instruct` by default) via its
  chat template, constrained to answer only from retrieved clauses; falls
  back to a dependency-free `ExtractiveBackend` if the model is unavailable.
  - `POST /chat` — `{"question": ..., "contract_id": ..., "k": 5}` → a cited
    answer.
  - `GET /contracts/{id}/chat` — minimal browser chat UI.
- **Executive reports** (`app/reports/generator.py`) — contract metadata,
  clause-category breakdown, and top risk findings, with a short narrative
  (LLM-written when available, otherwise a deterministic template sentence).
  - `GET /contracts/{id}/report` — HTML view.
  - `GET /contracts/{id}/report.json` — the underlying structured data.
- **Dashboard** (`GET /`) — upload form + list of stored contracts, linking
  to each contract's view/chat/report pages.

### Trying the Checkpoint 4 endpoints

With the server running (`uvicorn app.main:app --reload`) and at least one
contract uploaded:

```bash
# Ask a grounded, cited question about one contract (substitute a real id)
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the confidentiality obligations?", "contract_id": "<id>"}' \
  | python3 -m json.tool

# Executive report for a stored contract
curl -s "http://127.0.0.1:8000/contracts/<id>/report.json" | python3 -m json.tool
```

Both are also usable from the browser: `/` for the dashboard,
`/contracts/<id>/chat` for the chat UI, `/contracts/<id>/report` for the HTML
report. Note: the local LLM backend downloads `Qwen/Qwen2.5-0.5B-Instruct`
(~1GB) on first use; set `CONTRACTLENS_CHAT_BACKEND=extractive` to skip that
entirely and get deterministic, clause-composed answers instead.

### Benchmarking classification and retrieval

```bash
python -m scripts.run_benchmarks                       # rule + hashing, writes results/
python -m scripts.run_benchmarks --classifier legalbert --retrieval-backend sentence
python -m scripts.run_benchmarks --out ""               # skip writing a results file
```

Prints a combined classification + retrieval report and writes a timestamped
JSON summary to `results/` by default, so the headline numbers are
reproducible directly from the repository rather than living only in report
screenshots.
```

- [ ] **Step 5: Replace the Roadmap section**

Replace:

```markdown
## Roadmap

- ContractLens UI and local-LLM chatbot integration (question answering over
  uploaded contracts).
```

with:

```markdown
## Roadmap

- Fine-tuning a small local model on CUAD (e.g. via Georgia Tech PACE) —
  stretch goal beyond Checkpoint 4.
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (existing 58 + this plan's ~28 new tests).

- [ ] **Step 7: Manual smoke check of the dashboard and chat UI**

Run: `uvicorn app.main:app --reload` in one terminal, then in another:
```bash
curl -s -F "file=@data/sample_contract.txt" http://127.0.0.1:8000/upload | python3 -m json.tool
```
Note the returned `id`, then open `http://127.0.0.1:8000/` in a browser and confirm the uploaded contract appears with working view/chat/report links, and `http://127.0.0.1:8000/contracts/<id>/chat` answers a question end-to-end (first call will be slow — it downloads Qwen2.5-0.5B-Instruct on first use).

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "Document Checkpoint 4 endpoints and update project layout in README"
```

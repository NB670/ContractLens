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

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

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


def test_chat_raises_clear_error_when_mcp_client_fails_to_start(monkeypatch):
    """A start() failure must surface as an error, not hang forever.

    Regression test for a bug caught in review: app.main's _mcp_supervisor
    previously only called `ready.set()` on the *success* path of
    `_mcp_client.start()`, so if start() ever raised (subprocess spawn
    failure, bad path, resource exhaustion, ...) every caller of
    `_ensure_mcp_started()` -- both /chat and the /report routes -- would
    hang forever on `await _mcp_ready.wait()` instead of getting a clear
    failure. The fix stores the exception in `_mcp_start_error`, always sets
    `ready` in a `finally`, and has `_ensure_mcp_started()` re-raise it.

    This patches `MCPToolClient.start` on the shared `_mcp_client` singleton
    to raise, then hits /chat (no contract upload needed, and the autouse
    fixture above already keeps it on ExtractiveBackend so no LLM load is
    involved) inside a worker thread with a hard join timeout -- so if this
    regresses to a hang, this test fails fast instead of freezing the whole
    suite.
    """

    async def _broken_start() -> None:
        raise RuntimeError("simulated MCP subprocess spawn failure")

    monkeypatch.setattr(main_module._mcp_client, "start", _broken_start)

    def _make_request():
        with TestClient(app) as client:
            return client.post("/chat", json={"question": "What is the notice period?"})

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_make_request)
        try:
            resp = future.result(timeout=10)
        except FutureTimeoutError:
            pytest.fail("chat route hung instead of surfacing the start() failure")
        except RuntimeError as exc:
            assert "simulated MCP subprocess spawn failure" in str(exc)
            return

    assert resp.status_code == 500


async def test_ensure_mcp_started_raises_for_every_concurrent_caller_on_failure(monkeypatch):
    """Two concurrent callers racing a start() failure must both see it.

    Regression test for a second bug caught in review, on top of the
    original hang fix above: the first version of that fix read
    `_mcp_start_error` via a pop-and-clear (`error, _mcp_start_error =
    _mcp_start_error, None`), so as soon as *one* concurrent caller of
    `_ensure_mcp_started()` observed the failure it wiped the shared
    variable out from under any other caller waking on the same `ready`
    Event -- that second caller would then read back `None` and silently
    return as if the MCP client had started, instead of raising. The fix
    reads `_mcp_start_error` into a local per call without clearing the
    shared variable, so every waiter on a given failed attempt sees the
    same error object.

    Drives `_ensure_mcp_started()` directly (rather than through two real
    HTTP requests) via `asyncio.gather` so both calls are genuinely
    in-flight -- both past their `await _mcp_ready.wait()` -- before the
    patched start() fails and wakes both at once; wraps the whole thing in
    `asyncio.wait_for(..., timeout=5)` so a regression to a hang fails this
    test fast instead of freezing the suite.
    """

    async def _broken_start() -> None:
        raise RuntimeError("simulated concurrent MCP subprocess spawn failure")

    monkeypatch.setattr(main_module._mcp_client, "start", _broken_start)

    async def _gather_both():
        return await asyncio.gather(
            main_module._ensure_mcp_started(),
            main_module._ensure_mcp_started(),
            return_exceptions=True,
        )

    results = await asyncio.wait_for(_gather_both(), timeout=5)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, RuntimeError), f"expected both callers to raise, got {result!r}"
        assert "simulated concurrent MCP subprocess spawn failure" in str(result)

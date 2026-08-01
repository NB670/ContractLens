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
from app.mcp.client import MCPToolClient
from app.models.contract import Clause, Contract
from app.store import ContractStore, resolve_db_path


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


def test_dashboard_upload_form_posts_to_redirecting_route_not_raw_json():
    """Regression test: the dashboard's form must not POST straight to the
    plain JSON /upload endpoint, or a plain HTML form submission makes the
    browser navigate to and display raw JSON instead of returning to the
    dashboard -- confirmed happening live before this fix.
    """
    with TestClient(app) as client:
        resp = client.get("/")
    assert "/upload/dashboard" in resp.text


def test_upload_from_dashboard_redirects_to_dashboard_with_success_banner():
    with TestClient(app) as client:
        resp = client.post(
            "/upload/dashboard",
            files={"file": ("dash-upload-test.txt", io.BytesIO(
                b"1. Confidentiality\nEach party shall keep information confidential.\n"
            ), "text/plain")},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/?uploaded=")

        dashboard_resp = client.get(resp.headers["location"])

    assert dashboard_resp.status_code == 200
    assert "dash-upload-test.txt" in dashboard_resp.text
    assert "Uploaded" in dashboard_resp.text


def test_upload_from_dashboard_redirects_with_error_banner_on_bad_file():
    with TestClient(app) as client:
        resp = client.post(
            "/upload/dashboard",
            files={"file": ("bad.xyz", io.BytesIO(b"not a real contract"), "application/octet-stream")},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "upload_error=" in resp.headers["location"]

        dashboard_resp = client.get(resp.headers["location"])

    assert dashboard_resp.status_code == 200
    assert "Upload failed" in dashboard_resp.text


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


def test_view_contract_links_to_similar_clauses_page():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "similar-link-test.txt",
            "1. Confidentiality\nEach party shall keep information confidential.\n",
        )
        resp = client.get(f"/contracts/{contract_id}/view")

    assert resp.status_code == 200
    assert f"/contracts/{contract_id}/similar/0/view" in resp.text


def test_similar_clauses_view_shows_source_and_results():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "similar-view-test.txt",
            "1. Confidentiality\nEach party shall keep information confidential.\n"
            "2. Termination\nEither party may terminate this agreement upon notice.\n",
        )
        resp = client.get(f"/contracts/{contract_id}/similar/0/view")

    assert resp.status_code == 200
    assert "Source clause" in resp.text
    assert "confidential" in resp.text.lower()


def test_similar_clauses_view_404s_for_unknown_contract():
    with TestClient(app) as client:
        resp = client.get("/contracts/does-not-exist/similar/0/view")
    assert resp.status_code == 404


def test_similar_clauses_view_404s_for_out_of_range_clause_index():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "similar-oob-test.txt", "1. Confidentiality\nKeep this secret.\n",
        )
        resp = client.get(f"/contracts/{contract_id}/similar/99/view")
    assert resp.status_code == 404


def test_compare_view_renders_upload_form():
    with TestClient(app) as client:
        resp = client.get("/compare")
    assert resp.status_code == 200
    assert "<form" in resp.text
    assert "/compare/html" in resp.text
    assert 'name=\'base\'' in resp.text or 'name="base"' in resp.text
    assert 'name=\'revised\'' in resp.text or 'name="revised"' in resp.text


def test_compare_html_renders_diff_result():
    text = "1. Confidentiality\nEach party shall keep information confidential.\n"
    with TestClient(app) as client:
        resp = client.post(
            "/compare/html",
            files={
                "base": ("base.txt", io.BytesIO(text.encode()), "text/plain"),
                "revised": ("revised.txt", io.BytesIO(text.encode()), "text/plain"),
            },
        )

    assert resp.status_code == 200
    assert "base.txt" in resp.text
    assert "revised.txt" in resp.text
    assert "unchanged" in resp.text


def test_nav_bar_shows_compare_link_on_every_page():
    with TestClient(app) as client:
        contract_id = _upload_txt(
            client, "nav-compare-test.txt", "1. Confidentiality\nKeep this secret.\n",
        )
        dashboard_resp = client.get("/")
        view_resp = client.get(f"/contracts/{contract_id}/view")

    assert "/compare" in dashboard_resp.text
    assert "/compare" in view_resp.text


def test_chat_raises_clear_error_when_mcp_client_fails_to_start(monkeypatch):
    """A start() failure must surface as a clear 502, not hang or 500.

    Regression test for a bug caught in review: app.main's _mcp_supervisor
    previously only called `ready.set()` on the *success* path of
    `_mcp_client.start()`, so if start() ever raised (subprocess spawn
    failure, bad path, resource exhaustion, ...) every caller of
    `_ensure_mcp_started()` -- both /chat and the /report routes -- would
    hang forever on `await _mcp_ready.wait()` instead of getting a clear
    failure. The fix stores the exception in `_mcp_start_error`, always sets
    `ready` in a `finally`, and has `_ensure_mcp_started()` re-raise it.

    A second review pass then found that the re-raised exception was left to
    propagate unhandled out of the route, producing a raw 500 with no useful
    detail. Per the design spec ("a real dependency the app now needs... a
    clear error beats a silent partial answer"), /chat now wraps its
    MCP-dependent calls and turns any such failure into a 502 with a
    descriptive detail message -- asserted below instead of the old 500.

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

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "simulated MCP subprocess spawn failure" in detail
    assert "MCP service unavailable" in detail


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


def _contract_with_clause(cid: str, text: str, category: str) -> Contract:
    clause = Clause(
        index=0, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )
    return Contract(id=cid, filename=f"{cid}.txt", source_format="txt", clauses=[clause])


@pytest.mark.asyncio
async def test_mcp_subprocess_inherits_custom_contractlens_db_path(monkeypatch, tmp_path):
    """Regression test: the MCP server subprocess must see the same custom
    CONTRACTLENS_DB_PATH as the main process, not silently fall back to the
    store.py default.

    Bug being guarded against: mcp's stdio_client does NOT fully inherit the
    parent process's environment when MCPToolClient is constructed with
    env=None (the mcp SDK only forwards a small safe allowlist like PATH/
    HOME). app/main.py used to do exactly `MCPToolClient()` with no env, so
    if CONTRACTLENS_DB_PATH were set to a custom path for the main process,
    the subprocess's own ContractStore singleton would read that env var
    fresh in its own process, get nothing, and silently open the *default*
    "contractlens.db" instead -- a different, effectively-empty file. /chat
    and /report would then see zero clauses even though a contract was
    successfully uploaded and persisted at the custom path.

    This test proves the fix (app/main.py now explicitly passes
    env={"CONTRACTLENS_DB_PATH": resolve_db_path()} when constructing
    _mcp_client) actually works, by exercising the same construction here:
      1. Set CONTRACTLENS_DB_PATH to a custom temp-file path.
      2. Build an MCPToolClient exactly as app/main.py's module-level code
         does: env={"CONTRACTLENS_DB_PATH": resolve_db_path()} -- importing
         resolve_db_path() from app.store (rather than re-inlining its
         default literal here) so this test can't silently drift out of
         sync with app/store.py's own default the way the original bug did.
      3. Write a contract directly into a ContractStore pointed at that same
         custom path (bypassing the shared app.store.store singleton, since
         this test process and the server subprocess are different OS
         processes -- same isolation technique as tests/test_mcp_client.py).
      4. Call search_clauses through the client and assert the contract is
         found.

    This is not tautological because the test also drives the contrast case
    (below) side by side, within the same custom-env process: a second
    MCPToolClient constructed the old buggy way -- MCPToolClient(env=None),
    i.e. exactly `MCPToolClient()` as app/main.py used to call it -- talking
    to the SAME custom CONTRACTLENS_DB_PATH the test process has set. If the
    fix's env-passthrough weren't actually doing anything, both clients
    would behave identically; instead the env=None client's subprocess falls
    back to the default "contractlens.db" and finds nothing, while the
    env=<passthrough> client's subprocess opens the custom file and finds
    the clause. That divergence is the actual proof the fix works, not just
    that search_clauses works in general.
    """
    custom_db_path = tmp_path / "custom-cp4-followup.db"
    monkeypatch.setenv("CONTRACTLENS_DB_PATH", str(custom_db_path))

    contract_id = "cp4-followup-1"
    ContractStore(database_url=f"sqlite:///{custom_db_path}").add(
        _contract_with_clause(
            contract_id,
            "The parties agree to a governing law of Delaware for this agreement.",
            "Governing Law",
        )
    )

    # Contrast case: the OLD buggy construction app/main.py used to use (no
    # env passed at all). Its subprocess does NOT inherit
    # CONTRACTLENS_DB_PATH from this process and falls back to whatever
    # store.py's default "contractlens.db" resolves to in the subprocess's
    # cwd (which may be a real, pre-existing dev database with unrelated
    # contracts in it -- so this asserts our specific contract is absent
    # from the results, not that there are zero hits, keeping the check
    # deterministic regardless of what else is in that default file).
    buggy_client = MCPToolClient()
    try:
        buggy_result = await buggy_client.call_tool("search_clauses", query="governing law", k=5)
        found_ids = {hit["contract_id"] for hit in buggy_result["hits"]}
        assert contract_id not in found_ids, (
            "the env=None client unexpectedly found our contract -- this test's "
            "contrast case is broken, or CONTRACTLENS_DB_PATH is leaking through "
            "some other channel"
        )
    finally:
        await buggy_client.stop()

    # Fixed case: mirrors app/main.py's current module-level construction
    # exactly.
    client = MCPToolClient(env={"CONTRACTLENS_DB_PATH": resolve_db_path()})
    try:
        result = await client.call_tool("search_clauses", query="governing law", k=5)
        assert result["hits"], "subprocess did not find the clause -- it likely opened the wrong db file"
        assert result["hits"][0]["contract_id"] == "cp4-followup-1"
    finally:
        await client.stop()

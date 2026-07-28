"""ContractLens FastAPI application (Checkpoint 4).

Routes:
  GET  /                            dashboard: upload form + list of stored contracts
  GET  /health                      health / liveness
  POST /upload                      upload a PDF/DOCX/TXT contract, returns structured JSON
  GET  /contracts/{id}              structured JSON for a previously uploaded contract
  GET  /contracts/{id}/view         minimal HTML clause-visualization view
  GET  /search                      semantic search for clauses across all contracts
  GET  /contracts/{id}/similar/{i}  clauses similar to one clause of a contract
  POST /compare                     upload two contracts, get a clause-level semantic diff
  GET  /contracts/{id}/risk         evidence-backed risk report
  POST /chat                        MCP-grounded, cited question answering
  GET  /contracts/{id}/chat         minimal chat UI
  GET  /contracts/{id}/report       executive report (HTML)
  GET  /contracts/{id}/report.json  executive report (JSON)

The view groups clauses by their identified CUAD category and links each clause
back to its source offsets — the "browse automatically identified clauses /
highlight relevant section" deliverable for Checkpoint 2. /chat and /report
are backed by the MCP tool layer (app/mcp/) and return 502 with a clear
detail message if the MCP subprocess fails to start or a tool call raises,
rather than surfacing a raw, uninformative 500.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from app.chat.assistant import LocalLLMBackend, build_default_assistant
from app.comparison.comparator import compare_contracts
from app.config import settings
from app.ingestion.parsers import UnsupportedFormatError, detect_format
from app.mcp.client import MCPToolClient, MCPToolError
from app.models.chat import ChatRequest
from app.models.contract import Contract
from app.pipeline import ingest
from app.reports.generator import build_report
from app.retrieval.index import ClauseIndex
from app.risk.analyzer import analyze_risk
from app.store import resolve_db_path, store

_REPO_ROOT = Path(__file__).resolve().parent.parent

_clause_index = ClauseIndex()
# env is passed explicitly (rather than left as MCPToolClient's default None)
# because the mcp SDK's stdio_client does NOT fully inherit the parent
# process's environment when env=None -- it only forwards a small safe
# allowlist (PATH, HOME, etc.), silently dropping app-specific vars like
# CONTRACTLENS_DB_PATH. Without this, a custom CONTRACTLENS_DB_PATH set for
# this (main) process would not reach the app/mcp/server.py subprocess,
# which would then fall back to store.py's default "contractlens.db" --  a
# different file than the one this process's `store` singleton actually
# opened -- so /chat and /report would silently see an empty store even
# though contracts were uploaded successfully. Resolving the value here (via
# app.store.resolve_db_path(), the single source of truth store.py's own
# default is also built from) guarantees both processes agree on the same
# SQLite file regardless of whether the env var was customized. PYTHONPATH
# and VIRTUAL_ENV are forwarded too, for the same reason -- the mcp SDK's
# safe allowlist doesn't include them, so a deployment relying on either
# (rather than a plain "run uvicorn from the repo root with its venv already
# on PATH") would otherwise have the subprocess silently fail to `import
# app...`. cwd is pinned to the repo root explicitly rather than relying on
# inheriting the parent's cwd, which only happens to work today because
# `python -m` puts the inherited cwd on sys.path.
_mcp_client = MCPToolClient(
    env={
        "CONTRACTLENS_DB_PATH": resolve_db_path(),
        **{k: os.environ[k] for k in ("PYTHONPATH", "VIRTUAL_ENV") if k in os.environ},
    },
    cwd=_REPO_ROOT,
)
_assistant = build_default_assistant(_mcp_client)

# MCPToolClient.start()/stop() lifecycle: see _ensure_mcp_started()/_stop_mcp_client()
# below for why this can't just be a bare lazy `await _mcp_client.start()` call.
_mcp_supervisor_task: asyncio.Task | None = None
_mcp_ready: asyncio.Event | None = None
_mcp_stop_signal: asyncio.Event | None = None
_mcp_start_error: BaseException | None = None

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


async def _mcp_supervisor(ready: asyncio.Event, stop_signal: asyncio.Event) -> None:
    """Own the MCPToolClient's entire start/stop lifecycle in one asyncio Task.

    MCPToolClient.start() enters an anyio task group (inside mcp's
    stdio_client) whose cancel scope may only be exited from the same
    asyncio Task that entered it -- otherwise anyio raises "Attempted to
    exit cancel scope in a different task than it was entered in". Under
    FastAPI, each request is handled in its own Task, and the app's
    shutdown hook runs in yet another (the lifespan Task); if `start()` were
    awaited lazily inside a request handler and `stop()` awaited from the
    shutdown hook, those would be two different Tasks and stop() would
    reliably crash (verified while writing this: Starlette's TestClient
    schedules every request through `portal.call(...)`, a fresh Task each
    time, distinct from the one persistent Task it uses for the whole
    lifespan). Routing both calls through this single supervisor Task --
    created lazily on first use, parked on `stop_signal` in between --
    keeps start() and stop() in the same Task no matter which request
    Task kicked things off.

    If `_mcp_client.start()` itself raises (subprocess spawn failure, bad
    path, resource exhaustion, ...), that exception is stashed in the
    module-level `_mcp_start_error` and `ready` is still set (in `finally`)
    so every caller parked on `_ensure_mcp_started()`'s `await
    _mcp_ready.wait()` wakes up instead of hanging forever -- `stop_signal`
    is never waited on in that case since there is nothing running to stop.
    """
    global _mcp_start_error
    try:
        await _mcp_client.start()
    except BaseException as exc:  # noqa: BLE001 -- must unblock waiters on *any* failure
        _mcp_start_error = exc
    finally:
        ready.set()
    if _mcp_start_error is None:
        await stop_signal.wait()
        await _mcp_client.stop()


async def _ensure_mcp_started() -> None:
    """Lazily start the MCP subprocess (via the supervisor Task) before first use.

    Routes that never touch chat/report never call this, so they pay no
    subprocess cost -- the laziness the original design intended is
    preserved; only *how* the start happens changes (see _mcp_supervisor).

    Re-raises the supervisor's start() failure (if any) to the caller
    instead of leaving it to hang, and clears `_mcp_supervisor_task` so a
    *later* call gets a fresh attempt rather than being permanently wedged
    against a dead supervisor.

    Concurrency note: if two callers are both parked on `_mcp_ready.wait()`
    when start() fails, both wake up once `ready` is set, and both must see
    the failure -- not just whichever runs first. So `_mcp_start_error` is
    read into a local (`error`) rather than popped/cleared, and only the
    caller that still finds `_mcp_supervisor_task` pointing at *this*
    attempt's Task resets it to `None` -- avoiding a race where the first
    caller's reset makes a second, concurrent caller (or a call from within
    the same failed attempt) read back `None` and silently proceed as if
    the client had started.
    """
    global _mcp_supervisor_task, _mcp_ready, _mcp_stop_signal, _mcp_start_error
    if _mcp_supervisor_task is None:
        _mcp_ready = asyncio.Event()
        _mcp_stop_signal = asyncio.Event()
        _mcp_start_error = None
        _mcp_supervisor_task = asyncio.create_task(_mcp_supervisor(_mcp_ready, _mcp_stop_signal))
    supervisor_task = _mcp_supervisor_task
    await _mcp_ready.wait()
    error = _mcp_start_error
    if error is not None:
        if _mcp_supervisor_task is supervisor_task:
            _mcp_supervisor_task = None
        raise error


@app.on_event("shutdown")
async def _stop_mcp_client() -> None:
    """Terminate the MCP server subprocess, if `/chat` or `/report` ever started it.

    A no-op if the supervisor Task was never created (chat/report never
    used) -- but if it was, this signals it to run `_mcp_client.stop()`
    itself (same Task as its `start()`, see _mcp_supervisor) and waits for
    that to finish, then resets the module globals so a later app restart
    (e.g. the next `with TestClient(app) as client:` block in
    tests/test_main.py) starts a fresh supervisor instead of reusing a
    torn-down one. The reset happens in `finally` so a supervisor Task that
    ended via an exception (e.g. a start() failure that was already
    re-raised to its caller) still leaves clean state behind, rather than
    wedging every later app lifespan against stale globals.
    """
    global _mcp_supervisor_task, _mcp_ready, _mcp_stop_signal
    if _mcp_supervisor_task is not None:
        _mcp_stop_signal.set()
        try:
            await _mcp_supervisor_task
        finally:
            _mcp_supervisor_task = None
            _mcp_ready = None
            _mcp_stop_signal = None


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


async def _ingest_upload(file: UploadFile) -> Contract:
    """Validate an uploaded file and run it through the ingestion pipeline.

    Shared by ``/upload`` and ``/compare`` so both enforce identical format,
    empty-file, and size checks and surface the same HTTP error codes. Also
    feeds the shared retrieval index so any ingested contract -- via either
    endpoint -- becomes searchable immediately.

    Note: documents uploaded via ``/compare`` are persisted and indexed just
    like ``/upload`` uploads -- they are not treated as ephemeral. That is an
    intentional simplicity choice at this project's scale, not an oversight.
    """
    filename = file.filename or "upload"
    try:
        detect_format(filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        contract = ingest(filename, data)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Missing optional parser dependency (pypdf / python-docx).
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    _clause_index.add_contract(contract)
    return contract


@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)) -> dict:
    """Upload and structure a contract."""
    contract = await _ingest_upload(file)
    return {
        "id": contract.id,
        "filename": contract.filename,
        "source_format": contract.source_format,
        "metadata": contract.metadata.model_dump(),
        "categories": contract.categories_present(),
        "clauses": [c.model_dump() for c in contract.clauses],
    }


@app.on_event("startup")
def _load_existing_contracts_into_index() -> None:
    """Populate the in-memory retrieval index from persisted contracts.

    The SQLite store survives a restart; the in-memory ``ClauseIndex`` does
    not, so every already-stored contract is re-embedded once at startup.

    This synchronous re-embedding cost grows with the store's size, which is
    acceptable at this project's scale but would need to move to lazy or
    background indexing before running against a production-sized store.
    """
    contracts = [c for c in (store.get(cid) for cid in store.list_ids()) if c is not None]
    for contract in contracts:
        _clause_index.add_contract(contract)


@app.get("/contracts/{contract_id}")
def get_contract(contract_id: str) -> dict:
    contract = store.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    return contract.model_dump()


@app.get("/contracts/{contract_id}/view", response_class=HTMLResponse)
def view_contract(contract_id: str) -> str:
    """Minimal clause-visualization page grouped by identified category."""
    contract = store.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="Contract not found.")

    rows = []
    for category, count in contract.categories_present().items():
        rows.append(f"<li><strong>{html.escape(category)}</strong>: {count}</li>")

    clause_blocks = []
    for clause in contract.clauses:
        heading = html.escape(clause.heading or f"Clause {clause.index + 1}")
        snippet = html.escape(clause.text[:400])
        clause_blocks.append(
            f"<div class='clause'>"
            f"<h3>{heading} "
            f"<span class='cat'>[{html.escape(clause.category)} "
            f"&middot; {clause.confidence:.2f}]</span></h3>"
            f"<p class='offsets'>chars {clause.start_offset}-{clause.end_offset}</p>"
            f"<pre>{snippet}</pre>"
            f"</div>"
        )

    meta = contract.metadata
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ContractLens — {html.escape(contract.filename)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; }}
.cat {{ color: #555; font-weight: normal; font-size: .85em; }}
.offsets {{ color: #999; font-size: .8em; margin: 0; }}
.clause {{ border-left: 3px solid #ddd; padding-left: 1rem; margin: 1rem 0; }}
pre {{ white-space: pre-wrap; background: #f7f7f7; padding: .5rem; }}
</style></head><body>
<h1>{html.escape(contract.filename)}</h1>
<p><strong>Type:</strong> {html.escape(meta.contract_type)} &middot;
   <strong>Parties:</strong> {html.escape(', '.join(meta.parties) or 'n/a')} &middot;
   <strong>Clauses:</strong> {meta.num_clauses}</p>
<h2>Identified categories</h2>
<ul>{''.join(rows) or '<li>None</li>'}</ul>
<h2>Clauses</h2>
{''.join(clause_blocks) or '<p>No clauses detected.</p>'}
</body></html>"""


# --------------------------------------------------------------------------- #
# Checkpoint 3 -- semantic retrieval
# --------------------------------------------------------------------------- #
@app.get("/search")
def search_clauses(
    q: str = Query(..., description="Natural-language / clause-text query"),
    k: int = Query(5, ge=1, le=50),
    category: str | None = Query(None, description="Restrict to one category"),
) -> dict:
    """Semantic search for clauses across all uploaded contracts."""
    hits = _clause_index.search(q, k=k, category=category)
    return {"query": q, "k": k, "count": len(hits), "hits": [h.model_dump() for h in hits]}


@app.get("/contracts/{contract_id}/similar/{clause_index}")
def similar_clauses(
    contract_id: str, clause_index: int, k: int = Query(5, ge=1, le=50)
) -> dict:
    """Find clauses (in any contract) similar to one clause of a contract."""
    if store.get(contract_id) is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    hits = _clause_index.most_similar_to(contract_id, clause_index, k=k)
    return {
        "contract_id": contract_id,
        "clause_index": clause_index,
        "count": len(hits),
        "hits": [h.model_dump() for h in hits],
    }


# --------------------------------------------------------------------------- #
# Checkpoint 3 -- contract comparison
# --------------------------------------------------------------------------- #
@app.post("/compare")
async def compare_uploaded_contracts(
    base: UploadFile = File(..., description="Base contract / template"),
    revised: UploadFile = File(..., description="Revised / counterparty contract"),
) -> dict:
    """Upload two contracts and return a clause-level semantic diff."""
    base_contract = await _ingest_upload(base)
    revised_contract = await _ingest_upload(revised)
    diff = compare_contracts(base_contract, revised_contract)
    return diff.model_dump()


# --------------------------------------------------------------------------- #
# Checkpoint 3 -- risk analysis
# --------------------------------------------------------------------------- #
@app.get("/contracts/{contract_id}/risk")
def contract_risk(contract_id: str) -> dict:
    """Return an evidence-backed risk report for a stored contract."""
    contract = store.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    return analyze_risk(contract).model_dump()


# --------------------------------------------------------------------------- #
# Checkpoint 4 -- chatbot / question answering (RAG over the MCP tool layer)
# --------------------------------------------------------------------------- #
@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """Answer a natural-language question with clauses cited as grounding."""
    if request.contract_id is not None and store.get(request.contract_id) is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    try:
        await _ensure_mcp_started()
        response = await _assistant.answer(
            request.question, contract_id=request.contract_id, k=request.k
        )
    except (MCPToolError, Exception) as exc:
        # MCPToolError is itself an Exception, but it's named explicitly here
        # to document intent: this catches both a raising tool call and an
        # MCP subprocess start failure (_ensure_mcp_started() re-raises
        # whatever _mcp_client.start() raised) alike -- a real dependency the
        # app now needs is unavailable, and a clear 502 beats a silent
        # partial answer or an opaque 500.
        raise HTTPException(status_code=502, detail=f"MCP service unavailable: {exc}") from exc
    return response.model_dump()


@app.get("/contracts/{contract_id}/chat", response_class=HTMLResponse)
def chat_ui(contract_id: str) -> str:
    """Minimal chat UI for asking questions about one contract."""
    contract = store.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="Contract not found.")

    cid = html.escape(contract_id)
    fname = html.escape(contract.filename)
    # json.dumps (not Python repr) because this value is embedded directly
    # into a <script> block: repr() escapes quotes/backslashes but not
    # "</script>", which would prematurely close the script tag if it ever
    # appeared in the value. Not currently exploitable (contract ids are
    # server-generated uuid4 hex and unknown ids 404 before reaching this
    # template), but this is the safe idiom regardless, at no cost.
    cid_js = json.dumps(contract_id).replace("</", "<\\/")
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
const CID = {cid_js};
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
  if (!r.ok) {{
    log.innerHTML += `<div class='msg bot'><pre>Error: ${{esc(data.detail || ('request failed with status ' + r.status))}}</pre></div>`;
    log.scrollTop = log.scrollHeight;
    return;
  }}
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
    try:
        await _ensure_mcp_started()
        report = await build_report(contract_id, _mcp_client, llm_backend=_report_llm_backend())
    except (MCPToolError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"MCP service unavailable: {exc}") from exc
    if report is None:
        raise HTTPException(status_code=404, detail="Contract not found.")
    return report.model_dump()


@app.get("/contracts/{contract_id}/report", response_class=HTMLResponse)
async def contract_report_html(contract_id: str) -> str:
    try:
        await _ensure_mcp_started()
        report = await build_report(contract_id, _mcp_client, llm_backend=_report_llm_backend())
    except (MCPToolError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"MCP service unavailable: {exc}") from exc
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

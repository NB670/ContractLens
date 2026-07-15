"""ContractLens FastAPI application (Checkpoint 2).

Routes:
  GET  /                    health / liveness
  POST /upload              upload a PDF/DOCX/TXT contract, returns structured JSON
  GET  /contracts/{id}      structured JSON for a previously uploaded contract
  GET  /contracts/{id}/view minimal HTML clause-visualization view

The view groups clauses by their identified CUAD category and links each clause
back to its source offsets — the "browse automatically identified clauses /
highlight relevant section" deliverable for Checkpoint 2.
"""

from __future__ import annotations

import html

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from app.comparison.comparator import compare_contracts
from app.config import settings
from app.ingestion.parsers import UnsupportedFormatError, detect_format
from app.models.contract import Contract
from app.pipeline import ingest
from app.retrieval.index import ClauseIndex
from app.risk.analyzer import analyze_risk
from app.store import store

_clause_index = ClauseIndex()

app = FastAPI(
    title="ContractLens",
    description="Privacy-preserving contract intelligence platform (CP2: ingestion + clause intelligence)",
    version="0.2.0",
)


@app.get("/")
def root() -> dict:
    return {
        "service": "ContractLens",
        "checkpoint": 2,
        "supported_formats": list(settings.supported_extensions),
        "classifier_backend": settings.classifier_backend,
    }


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

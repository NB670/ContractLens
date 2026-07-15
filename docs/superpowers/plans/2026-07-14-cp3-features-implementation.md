# Checkpoint 3 Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Checkpoint 3's three deliverables — semantic clause retrieval, contract comparison, and evidence-backed risk analysis — each with a real evaluation harness, exceeding the professor's reference `ai-suggestions/cp3` scaffold in concrete, defensible ways.

**Architecture:** The existing CP2 pipeline (`parse -> segment -> classify -> store`) stays intact. This plan adds three new top-level packages that read `Contract`/`Clause` objects out of the existing SQLite store: `app/retrieval/` (embedder + brute-force cosine index, updated incrementally rather than rebuilt per request), `app/comparison/` (Hungarian-algorithm optimal clause alignment), and `app/risk/` (transparent regex rule engine with evidence offsets). Two new evaluation scripts (`scripts/evaluate_retrieval.py`, `scripts/evaluate_comparison.py`) plus one new precision-eval script (`scripts/evaluate_risk.py`) give all three the same measured treatment CP2 gave classification.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2 (existing), `sentence-transformers` (new — semantic retrieval embedder), `scipy`/`numpy` (new — optimal bipartite matching for comparison), pytest.

## Global Constraints

- Preserve existing public signatures: `store.new_id()`, `store.add(contract)`, `store.get(id)`, `store.list_ids()`, `ingest(filename, data)`, `RuleBasedClassifier.classify(text)`, `LegalBertClassifier.classify(text)`. Nothing in this plan changes `app/store.py`, `app/clauses/`, or `app/ingestion/`.
- All new/changed code must have `from __future__ import annotations` at the top, matching every existing file in `app/`.
- `sentence-transformers` and `scipy`/`numpy` imports that touch a real model or heavy computation stay behind a lazy-load/fallback path where the design calls for it (`SentenceTransformerEmbedder`, matching `LegalBertClassifier`'s existing `_ensure_embed_fn` pattern) — the retrieval layer must still work with zero optional dependencies installed via `HashingEmbedder`.
- No FAISS/ChromaDB, no new UI, no schema migrations — see the design spec's Non-goals.
- Risk analysis makes no recall claim — only precision-of-firing is measured and the script's docstring states this limitation explicitly, same posture as `scripts/evaluate_clauses.py`'s stated CUAD-category-gap limitation.
- Commit locally after each task (one commit per task is fine). Commit messages must describe only what changed — never mention Claude, AI, or include any "Co-Authored-By" trailer. Never run `git push` — pushing is the user's decision alone, made explicitly and separately.
- Full design rationale: `docs/superpowers/specs/2026-07-14-cp3-features-design.md`.

---

### Task 1: Semantic retrieval — embedder, index, and API endpoints

**Files:**
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/embedder.py`
- Create: `app/retrieval/index.py`
- Create: `app/models/analysis.py`
- Modify: `app/config.py`
- Modify: `app/main.py`
- Modify: `requirements.txt`
- Test: `tests/test_retrieval.py` (new)

**Interfaces:**
- Consumes: `app.models.contract.Contract`, `app.models.contract.Clause` (existing, unchanged), `app.store.store` (existing `.list_ids()` / `.get()`).
- Produces: `cosine_similarity(a: list[float], b: list[float]) -> float`; `HashingEmbedder().embed(text: str) -> list[float]`; `SentenceTransformerEmbedder(model_name: str | None = None).embed(text: str) -> list[float]`; `get_embedder() -> Embedder` (module-level `lru_cache`d, so the same instance — and its lazily-loaded model — is reused everywhere); `RetrievalHit` (Pydantic model: `contract_id`, `clause_index`, `category`, `heading`, `text`, `score`) in `app/models/analysis.py`; `ClauseIndex(embedder: Embedder = get_embedder())`, `.add_contract(contract: Contract) -> int`, `.size -> int`, `.search(query: str, k: int = 5, category: str | None = None, exclude: tuple[str, int] | None = None) -> list[RetrievalHit]`, `.most_similar_to(contract_id: str, clause_index: int, k: int = 5) -> list[RetrievalHit]`. Later tasks (3, 5) append more models to `app/models/analysis.py` and import `get_embedder`/`cosine_similarity`/`Embedder` from `app/retrieval/embedder.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval.py`:

```python
"""Tests for semantic clause retrieval: embedder, cosine similarity, and index."""

import pytest

from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder, cosine_similarity
from app.retrieval.index import ClauseIndex


def _clause(index: int, text: str, category: str = "Unclassified") -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def _contract(contract_id: str, texts: list[str]) -> Contract:
    return Contract(
        id=contract_id,
        filename=f"{contract_id}.txt",
        source_format="txt",
        clauses=[_clause(i, t) for i, t in enumerate(texts)],
    )


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder()
    assert embedder.embed("Confidential Information") == embedder.embed("Confidential Information")


def test_hashing_embedder_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder()
    query = embedder.embed("Each party shall keep all Confidential Information secret.")
    similar = embedder.embed("The parties agree to keep Confidential Information confidential.")
    unrelated = embedder.embed("Payment shall be made net thirty days after invoice.")

    assert cosine_similarity(query, similar) > cosine_similarity(query, unrelated)


def test_clause_index_search_ranks_by_similarity():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
    ]))

    hits = index.search("confidential information secrecy obligations", k=2)

    assert len(hits) == 2
    assert hits[0].clause_index == 0
    assert hits[0].score >= hits[1].score


def test_clause_index_search_respects_category_filter():
    index = ClauseIndex(embedder=HashingEmbedder())
    contract = _contract("c1", ["Confidential Information clause.", "Payment terms clause."])
    contract.clauses[0].category = "Confidentiality"
    contract.clauses[1].category = "Payment Terms"
    index.add_contract(contract)

    hits = index.search("clause", k=5, category="Payment Terms")

    assert len(hits) == 1
    assert hits[0].category == "Payment Terms"


def test_clause_index_most_similar_to_excludes_self():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", [
        "Each party shall keep all Confidential Information secret.",
        "The parties agree to keep Confidential Information confidential at all times.",
        "Payment shall be made net thirty days after invoice.",
    ]))

    hits = index.most_similar_to("c1", 0, k=2)

    assert all(not (h.contract_id == "c1" and h.clause_index == 0) for h in hits)
    assert hits[0].clause_index == 1


def test_clause_index_size_tracks_added_clauses():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", ["one", "two", "three"]))
    assert index.size == 3


def test_clause_index_search_on_empty_index_returns_empty_list():
    index = ClauseIndex(embedder=HashingEmbedder())
    assert index.search("anything") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL/ERROR — `app.retrieval` doesn't exist yet (`ModuleNotFoundError`).

- [ ] **Step 3: Add `sentence-transformers` to requirements.txt and install it**

Append to `requirements.txt`:

```

# Semantic retrieval embedding backend (Checkpoint 3). Imported lazily in
# app/retrieval/embedder.py; the HashingEmbedder fallback works without it.
sentence-transformers>=3.0

# Optimal clause alignment for contract comparison (Checkpoint 3).
scipy>=1.13
numpy>=1.26
```

Run: `pip install -r requirements.txt`
Expected: installs successfully (`sentence-transformers` pulls in its own copy of `torch`/`transformers` if not already present; this repo already has both from the CP2 LegalBERT work).

- [ ] **Step 4: Add retrieval settings to app/config.py**

In `app/config.py`, add these two fields to the `Settings` dataclass, right after `legalbert_model`:

```python
    # Retrieval embedding backend: "sentence" (default, semantic) or "hashing"
    # (dependency-free fallback, same posture as classifier_backend).
    retrieval_backend: str = os.environ.get("CONTRACTLENS_RETRIEVAL_BACKEND", "sentence")

    # sentence-transformers model id used when retrieval_backend == "sentence".
    retrieval_model: str = os.environ.get(
        "CONTRACTLENS_RETRIEVAL_MODEL", "all-MiniLM-L6-v2"
    )
```

- [ ] **Step 5: Create app/retrieval/__init__.py**

Create `app/retrieval/__init__.py` (empty file).

- [ ] **Step 6: Create app/models/analysis.py**

```python
"""Structured models for the Checkpoint 3 downstream tasks.

These sit alongside `app/models/contract.py`. Everything here references the
same clause identity used in CP2 (`Clause.index`) so a hit, a change, or a
finding can always be linked back to the exact clause it came from. This file
grows across Checkpoint 3: retrieval's `RetrievalHit` first, then comparison's
`ClauseChange`/`ContractDiff`, then risk's `RiskFinding`/`RiskReport`.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """One ranked result from a semantic clause search."""

    contract_id: str
    clause_index: int = Field(..., description="Clause.index within its contract")
    category: str
    heading: Optional[str] = None
    text: str
    score: float = Field(..., description="Cosine similarity in [-1, 1]")
```

- [ ] **Step 7: Create app/retrieval/embedder.py**

```python
"""Text embedding for semantic retrieval and comparison (Checkpoint 3).

Two backends, mirroring the two-backend pattern already used for
classification (`app/clauses/classifier.py`):

  * ``HashingEmbedder`` (dependency-free fallback) -- a deterministic
    bag-of-words hashing embedding. No model download, works everywhere.

  * ``SentenceTransformerEmbedder`` (default) -- a small sentence-embedding
    model purpose-trained for cosine-similarity retrieval. This is a
    deliberate departure from reusing LegalBERT's mean-pooled tokens (as the
    reference scaffold does): LegalBERT is a masked-LM encoder never trained
    for sentence similarity, whereas sentence-transformers models are
    trained specifically for it via contrastive/triplet losses over sentence
    pairs -- the right tool for retrieval specifically. Falls back to
    ``HashingEmbedder`` if the dependency or model is unavailable, so
    nothing hard-fails on a missing optional dependency.

Both return a plain ``list[float]`` vector, and ``cosine_similarity`` operates
on those lists, so no heavy numeric dependency is required at the retrieval
layer itself.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dimensionality of the hashing embedding. Large enough to keep collisions
# rare for clause-length text, small enough to keep pure-python cosine cheap.
_HASH_DIM = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    def embed(self, text: str) -> list[float]: ...  # pragma: no cover - protocol


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words hashing embedder.

    Each token is hashed into one of ``dim`` buckets and its (sub-linear) term
    frequency is accumulated there. The resulting vector is L2-normalized. Two
    texts that share vocabulary land close under cosine similarity; disjoint
    texts land near zero. Deterministic across runs and processes (uses a
    stable hash, not Python's salted ``hash()``).
    """

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            vec[self._bucket(token)] += 1.0
        vec = [math.log1p(v) for v in vec]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder:
    """Sentence-embedding backend for semantic retrieval.

    Loads the model lazily on first ``embed()`` call (same lazy-load timing
    as ``LegalBertClassifier``). Falls back to ``HashingEmbedder`` if the
    dependency or model fails to load, so the pipeline never hard-fails on a
    missing optional dependency.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.retrieval_model
        self._model = None
        self._load_failed = False
        self._fallback = HashingEmbedder()

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:  # pragma: no cover - depends on optional heavy dep
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception:
            self._load_failed = True
            return False

    def embed(self, text: str) -> list[float]:
        if not self._ensure_model():
            return self._fallback.embed(text)
        try:  # pragma: no cover - depends on optional heavy dep
            vector = self._model.encode(text)
            return [float(x) for x in vector.tolist()]
        except Exception:
            return self._fallback.embed(text)


@lru_cache(maxsize=None)
def get_embedder() -> Embedder:
    """Return the process-wide configured embedder instance.

    Cached so every caller (the retrieval index, the comparator, the eval
    scripts) shares one instance -- and, for ``SentenceTransformerEmbedder``,
    one lazily-loaded model -- instead of each reloading it independently.
    """
    if settings.retrieval_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
```

- [ ] **Step 8: Create app/retrieval/index.py**

```python
"""In-memory semantic index over contract clauses (Checkpoint 3).

``ClauseIndex`` embeds every clause of every added contract and answers
similarity queries with an exact (brute-force) cosine nearest-neighbour
search. The public surface (``add_contract`` / ``search`` / ``most_similar_to``)
is deliberately small so a FAISS/Chroma backend could be dropped in behind it
later without touching callers.

``app/main.py`` maintains one process-wide ``ClauseIndex`` instance and calls
``add_contract`` once per upload, rather than rebuilding the index from the
store on every request -- clauses are embedded exactly once, not once per
query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.analysis import RetrievalHit
from app.models.contract import Contract
from app.retrieval.embedder import Embedder, cosine_similarity, get_embedder


@dataclass
class _Entry:
    contract_id: str
    clause_index: int
    category: str
    heading: str | None
    text: str
    vector: list[float]


@dataclass
class ClauseIndex:
    """A brute-force cosine index over clauses drawn from one or more contracts."""

    embedder: Embedder = field(default_factory=get_embedder)
    _entries: list[_Entry] = field(default_factory=list)

    def add_contract(self, contract: Contract) -> int:
        """Embed and index every clause of ``contract``; return #clauses added."""
        added = 0
        for clause in contract.clauses:
            if not clause.text or not clause.text.strip():
                continue
            self._entries.append(
                _Entry(
                    contract_id=contract.id,
                    clause_index=clause.index,
                    category=clause.category,
                    heading=clause.heading,
                    text=clause.text,
                    vector=self.embedder.embed(clause.text),
                )
            )
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self._entries)

    def search(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
        exclude: tuple[str, int] | None = None,
    ) -> list[RetrievalHit]:
        """Return the ``k`` clauses most similar to ``query``."""
        if not query or not query.strip() or not self._entries:
            return []

        query_vec = self.embedder.embed(query)
        scored: list[tuple[float, _Entry]] = []
        for entry in self._entries:
            if category is not None and entry.category != category:
                continue
            if exclude is not None and (entry.contract_id, entry.clause_index) == exclude:
                continue
            scored.append((cosine_similarity(query_vec, entry.vector), entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievalHit(
                contract_id=entry.contract_id,
                clause_index=entry.clause_index,
                category=entry.category,
                heading=entry.heading,
                text=entry.text,
                score=round(score, 4),
            )
            for score, entry in scored[: max(0, k)]
        ]

    def most_similar_to(
        self, contract_id: str, clause_index: int, k: int = 5
    ) -> list[RetrievalHit]:
        """Find the clauses most similar to one already-indexed clause."""
        for entry in self._entries:
            if entry.contract_id == contract_id and entry.clause_index == clause_index:
                return self.search(entry.text, k=k, exclude=(contract_id, clause_index))
        return []
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: PASS (10 passed).

- [ ] **Step 10: Wire the retrieval index and endpoints into app/main.py**

Replace the imports block at the top of `app/main.py` (current lines 14-24):

```python
from __future__ import annotations

import html

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.config import settings
from app.ingestion.parsers import UnsupportedFormatError, detect_format
from app.pipeline import ingest
from app.store import store
```

with:

```python
from __future__ import annotations

import html

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from app.config import settings
from app.ingestion.parsers import UnsupportedFormatError, detect_format
from app.models.contract import Contract
from app.pipeline import ingest
from app.retrieval.index import ClauseIndex
from app.store import store

_clause_index = ClauseIndex()
```

Replace the `upload_contract` function (current lines 43-73):

```python
@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)) -> dict:
    """Upload and structure a contract."""
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

    return {
        "id": contract.id,
        "filename": contract.filename,
        "source_format": contract.source_format,
        "metadata": contract.metadata.model_dump(),
        "categories": contract.categories_present(),
        "clauses": [c.model_dump() for c in contract.clauses],
    }
```

with:

```python
@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)) -> dict:
    """Upload and structure a contract."""
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
    """
    contracts = [c for c in (store.get(cid) for cid in store.list_ids()) if c is not None]
    for contract in contracts:
        _clause_index.add_contract(contract)
```

Append these two routes at the end of `app/main.py` (after the existing `view_contract` function):

```python


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
```

- [ ] **Step 11: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (existing 30 + new 10 = 40).

- [ ] **Step 12: Manually verify the server starts and the new endpoints respond**

Run: `rm -f contractlens.db && uvicorn app.main:app --port 8000 &` (background), then:

```bash
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@data/sample_contract.txt" | python3 -m json.tool | head -5
curl -s "http://127.0.0.1:8000/search?q=confidential+information&k=3" | python3 -m json.tool
```

Expected: the search response has `"count"` ≥ 1 and at least one hit with `"category": "Confidentiality"`. The first request downloads `all-MiniLM-L6-v2` on first use (small, ~80MB, one-time). Stop the server afterward (`kill %1`).

- [ ] **Step 13: Commit**

```bash
git add app/retrieval app/models/analysis.py app/config.py app/main.py requirements.txt tests/test_retrieval.py
git commit -m "Add semantic clause retrieval: embedder, index, and search endpoints"
```

---

### Task 2: Retrieval evaluation harness

**Files:**
- Create: `scripts/evaluate_retrieval.py`
- Test: `tests/test_evaluate_retrieval.py` (new)

**Interfaces:**
- Consumes: `app.retrieval.embedder.HashingEmbedder`, `SentenceTransformerEmbedder`, `cosine_similarity` (Task 1); `data/cuad_sample.json` (existing, from CP2).
- Produces: `evaluate(records: list[dict], embedder, k: int = 5) -> dict` (keys: `recall_at_k`, `success_at_k`, `mrr`, `queries`, `k`) and a `python -m scripts.evaluate_retrieval` CLI entrypoint.

- [ ] **Step 1: Write scripts/evaluate_retrieval.py**

```python
"""Evaluate semantic clause retrieval against the committed CUAD sample.

Run: python -m scripts.evaluate_retrieval [--k 5] [--backend hashing|sentence]
                                          [--limit N]

Reuses data/cuad_sample.json (the same fixture scripts/evaluate_clauses.py
scores classification on). Each clause is used in turn as a query; the other
clauses sharing its CUAD category are the ground-truth relevant set. Every
clause is embedded once with the configured embedder and ranked by cosine
similarity against every other clause, then scored with the retrieval
metrics named in the project plan:

  * Recall@K   -- fraction of a query's relevant clauses that land in the top K
  * Success@K  -- fraction of queries with at least one relevant clause in top K
  * MRR        -- mean reciprocal rank of the first relevant clause

The default backend ("hashing" -> HashingEmbedder) needs no model download,
so this runs offline; pass --backend sentence to score the semantic
sentence-transformers encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval.embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
)

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def evaluate(records: list[dict], embedder, k: int = 5) -> dict[str, float]:
    """Return {'recall_at_k', 'success_at_k', 'mrr', 'queries', 'k'}."""
    texts = [r["clause_text"] for r in records]
    categories = [r["category"] for r in records]
    vectors = [embedder.embed(t) for t in texts]
    n = len(records)

    recall_sum = 0.0
    success_sum = 0.0
    rr_sum = 0.0
    evaluated = 0

    for i in range(n):
        relevant = [j for j in range(n) if j != i and categories[j] == categories[i]]
        if not relevant:
            continue
        evaluated += 1

        ranked = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: cosine_similarity(vectors[i], vectors[j]),
            reverse=True,
        )

        top_k = ranked[:k]
        relevant_set = set(relevant)
        hits_in_top_k = sum(1 for j in top_k if j in relevant_set)

        recall_sum += hits_in_top_k / min(k, len(relevant))
        success_sum += 1.0 if hits_in_top_k else 0.0

        for rank, j in enumerate(ranked, start=1):
            if j in relevant_set:
                rr_sum += 1.0 / rank
                break

    if evaluated == 0:
        return {"recall_at_k": 0.0, "success_at_k": 0.0, "mrr": 0.0, "queries": 0, "k": k}

    return {
        "recall_at_k": round(recall_sum / evaluated, 4),
        "success_at_k": round(success_sum / evaluated, 4),
        "mrr": round(rr_sum / evaluated, 4),
        "queries": evaluated,
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N sample records (faster smoke run).",
    )
    args = parser.parse_args()

    records = load_sample()
    if args.limit is not None:
        records = records[: args.limit]

    embedder = SentenceTransformerEmbedder() if args.backend == "sentence" else HashingEmbedder()
    metrics = evaluate(records, embedder, k=args.k)

    print(f"=== retrieval eval (backend={args.backend}) ===")
    print(f"queries evaluated : {metrics['queries']}")
    print(f"Recall@{metrics['k']:<11}: {metrics['recall_at_k']:.4f}")
    print(f"Success@{metrics['k']:<10}: {metrics['success_at_k']:.4f}")
    print(f"MRR{'':<15}: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

Create `tests/test_evaluate_retrieval.py`:

```python
"""Smoke test for the retrieval evaluation harness's scoring logic.

Uses a tiny inline fixture and the fast HashingEmbedder, so this test runs
quickly and offline -- it never triggers a sentence-transformers download.
"""

from app.retrieval.embedder import HashingEmbedder
from scripts.evaluate_retrieval import evaluate

_FIXTURE = [
    {"category": "Confidentiality", "clause_text": "Each party shall keep all Confidential Information strictly secret."},
    {"category": "Confidentiality", "clause_text": "All Confidential Information disclosed shall remain confidential."},
    {"category": "Payment Terms", "clause_text": "Payment shall be made net thirty days after invoice."},
    {"category": "Payment Terms", "clause_text": "Fees are due within thirty days of receipt of invoice."},
    {"category": "Governing Law", "clause_text": "This Agreement is a singleton category with no other match."},
]


def test_evaluate_reports_recall_success_and_mrr():
    metrics = evaluate(_FIXTURE, HashingEmbedder(), k=5)

    # 4 of 5 records have at least one same-category peer; the singleton
    # Governing Law record is skipped (no ground truth to evaluate against).
    assert metrics["queries"] == 4
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["success_at_k"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_evaluate_on_empty_records_returns_zeroed_metrics():
    metrics = evaluate([], HashingEmbedder(), k=5)
    assert metrics == {"recall_at_k": 0.0, "success_at_k": 0.0, "mrr": 0.0, "queries": 0, "k": 5}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_retrieval.py -v`
Expected: PASS (2 passed).

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (40 + 2 = 42).

- [ ] **Step 5: Run the real evaluation report**

Run: `python -m scripts.evaluate_retrieval --backend hashing`
Expected: prints `queries evaluated`, `Recall@5`, `Success@5`, `MRR` for the 350-record CUAD sample. Then run `python -m scripts.evaluate_retrieval --backend sentence` (downloads `all-MiniLM-L6-v2` on first use, may take a minute) and compare — this pair of numbers is the primary new "Factual" evidence for the CP3 report, showing whether the semantic backend actually beats the dependency-free baseline. Save both outputs.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_retrieval.py tests/test_evaluate_retrieval.py
git commit -m "Add retrieval evaluation harness (Recall@K, Success@K, MRR)"
```

---

### Task 3: Contract comparison — optimal alignment and API endpoint

**Files:**
- Create: `app/comparison/__init__.py`
- Create: `app/comparison/comparator.py`
- Modify: `app/models/analysis.py` (append comparison models)
- Modify: `app/main.py` (extract `_ingest_upload` helper; add `POST /compare`)
- Test: `tests/test_comparison.py` (new)

**Interfaces:**
- Consumes: `app.retrieval.embedder.Embedder`, `get_embedder`, `cosine_similarity` (Task 1); `app.models.contract.Clause`, `Contract` (existing).
- Produces: `CHANGE_ADDED`/`CHANGE_REMOVED`/`CHANGE_MODIFIED`/`CHANGE_UNCHANGED` string constants; `ClauseChange`, `ContractDiff` Pydantic models in `app/models/analysis.py`; `compare_contracts(base: Contract, revised: Contract, embedder: Embedder | None = None, match_threshold: float = 0.60, identical_threshold: float = 0.995) -> ContractDiff`. Task 4's `scripts/evaluate_comparison.py` imports `compare_contracts` and the `CHANGE_*` constants.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_comparison.py`:

```python
"""Tests for contract comparison: optimal clause alignment and diffing."""

from app.comparison.comparator import compare_contracts
from app.models.analysis import CHANGE_ADDED, CHANGE_MODIFIED, CHANGE_REMOVED, CHANGE_UNCHANGED
from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder


def _clause(index: int, text: str, category: str = "Unclassified") -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def _contract(contract_id: str, texts: list[str]) -> Contract:
    return Contract(
        id=contract_id, filename=f"{contract_id}.txt", source_format="txt",
        clauses=[_clause(i, t) for i, t in enumerate(texts)],
    )


def test_identical_contracts_are_all_unchanged():
    texts = [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
    ]
    base = _contract("base", texts)
    revised = _contract("revised", texts)

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_UNCHANGED] == 2
    assert diff.summary[CHANGE_ADDED] == 0
    assert diff.summary[CHANGE_REMOVED] == 0
    assert diff.summary[CHANGE_MODIFIED] == 0


def test_detects_added_removed_and_modified_clauses():
    base = _contract("base", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
        "This Agreement shall be governed by the laws of Delaware.",
    ])
    revised = _contract("revised", [
        "Payment shall be made net thirty days after invoice.",
        "This Agreement shall be governed by the laws of Delaware, without regard to conflicts of law.",
        "This is a brand new indemnification clause added in the revision.",
    ])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_REMOVED] == 1  # confidentiality clause dropped
    assert diff.summary[CHANGE_UNCHANGED] == 1  # payment clause identical
    assert diff.summary[CHANGE_MODIFIED] == 1  # governing law clause reworded
    assert diff.summary[CHANGE_ADDED] == 1  # new indemnification clause

    removed = [c for c in diff.changes if c.change_type == CHANGE_REMOVED][0]
    assert "Confidential" in removed.base_text

    added = [c for c in diff.changes if c.change_type == CHANGE_ADDED][0]
    assert "indemnification" in added.revised_text.lower()


def test_empty_base_contract_all_added():
    base = _contract("base", [])
    revised = _contract("revised", ["A brand new clause."])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_ADDED] == 1
    assert diff.summary[CHANGE_REMOVED] == 0


def test_empty_revised_contract_all_removed():
    base = _contract("base", ["An old clause that got dropped."])
    revised = _contract("revised", [])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_REMOVED] == 1
    assert diff.summary[CHANGE_ADDED] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_comparison.py -v`
Expected: FAIL/ERROR — `app.comparison` doesn't exist yet.

- [ ] **Step 3: Append comparison models to app/models/analysis.py**

Add to the end of `app/models/analysis.py`:

```python


# --------------------------------------------------------------------------- #
# Contract comparison
# --------------------------------------------------------------------------- #

CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_MODIFIED = "modified"
CHANGE_UNCHANGED = "unchanged"


class ClauseChange(BaseModel):
    """A single aligned (or unmatched) clause between two contract versions."""

    change_type: str = Field(
        ..., description=f"one of {CHANGE_ADDED!r}/{CHANGE_REMOVED!r}/"
        f"{CHANGE_MODIFIED!r}/{CHANGE_UNCHANGED!r}"
    )
    category: str = "Unclassified"
    base_index: Optional[int] = Field(
        None, description="Clause.index in the base contract (None if added)"
    )
    revised_index: Optional[int] = Field(
        None, description="Clause.index in the revised contract (None if removed)"
    )
    similarity: float = Field(0.0, description="Cosine similarity of the aligned pair in [-1, 1]")
    base_text: Optional[str] = None
    revised_text: Optional[str] = None


class ContractDiff(BaseModel):
    """Structured diff between a base contract and a revised contract."""

    base_contract_id: str
    revised_contract_id: str
    changes: list[ClauseChange] = Field(default_factory=list)
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by change_type (added/removed/modified/unchanged)",
    )
```

- [ ] **Step 4: Create app/comparison/__init__.py**

Create `app/comparison/__init__.py` (empty file).

- [ ] **Step 5: Create app/comparison/comparator.py**

```python
"""Semantic contract comparison (Checkpoint 3).

Given a base contract and a revised contract, align their clauses by
embedding similarity and classify each into one of: ``unchanged``,
``modified``, ``added``, ``removed``.

Alignment uses scipy's Hungarian algorithm (``linear_sum_assignment``) to
find the globally optimal one-to-one pairing over the clause-similarity
matrix -- a deliberate departure from a greedy highest-similarity-first
match (which can lock in a suboptimal pairing when a clause's best partner
is claimed by a slightly-better competing pair elsewhere in the matrix).
Pairs below ``match_threshold`` are rejected even if the optimal assignment
would otherwise include them, so unrelated clauses are never forced
together.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models.analysis import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    ClauseChange,
    ContractDiff,
)
from app.models.contract import Clause, Contract
from app.retrieval.embedder import Embedder, cosine_similarity, get_embedder

DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_IDENTICAL_THRESHOLD = 0.995


def _pick_category(base: Clause, revised: Clause) -> str:
    """Prefer a classified category over the 'Unclassified' placeholder."""
    if revised.category and revised.category != "Unclassified":
        return revised.category
    return base.category


def compare_contracts(
    base: Contract,
    revised: Contract,
    embedder: Embedder | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    identical_threshold: float = DEFAULT_IDENTICAL_THRESHOLD,
) -> ContractDiff:
    """Align two contracts' clauses and return a structured ``ContractDiff``."""
    embedder = embedder or get_embedder()

    base_vecs = [embedder.embed(c.text) for c in base.clauses]
    revised_vecs = [embedder.embed(c.text) for c in revised.clauses]

    matched_base: dict[int, tuple[int, float]] = {}
    matched_revised: set[int] = set()

    if base_vecs and revised_vecs:
        similarity = np.zeros((len(base_vecs), len(revised_vecs)))
        for i, bv in enumerate(base_vecs):
            for j, rv in enumerate(revised_vecs):
                similarity[i, j] = cosine_similarity(bv, rv)

        # linear_sum_assignment minimizes cost; negate similarity to maximize it.
        row_idx, col_idx = linear_sum_assignment(-similarity)
        for i, j in zip(row_idx, col_idx):
            sim = float(similarity[i, j])
            if sim < match_threshold:
                continue
            matched_base[i] = (j, sim)
            matched_revised.add(j)

    changes: list[ClauseChange] = []

    for i, (j, sim) in sorted(matched_base.items()):
        base_clause, revised_clause = base.clauses[i], revised.clauses[j]
        identical = (
            sim >= identical_threshold
            or base_clause.text.strip() == revised_clause.text.strip()
        )
        changes.append(
            ClauseChange(
                change_type=CHANGE_UNCHANGED if identical else CHANGE_MODIFIED,
                category=_pick_category(base_clause, revised_clause),
                base_index=base_clause.index,
                revised_index=revised_clause.index,
                similarity=round(sim, 4),
                base_text=base_clause.text,
                revised_text=revised_clause.text,
            )
        )

    for i, base_clause in enumerate(base.clauses):
        if i not in matched_base:
            changes.append(
                ClauseChange(
                    change_type=CHANGE_REMOVED,
                    category=base_clause.category,
                    base_index=base_clause.index,
                    revised_index=None,
                    similarity=0.0,
                    base_text=base_clause.text,
                    revised_text=None,
                )
            )

    for j, revised_clause in enumerate(revised.clauses):
        if j not in matched_revised:
            changes.append(
                ClauseChange(
                    change_type=CHANGE_ADDED,
                    category=revised_clause.category,
                    base_index=None,
                    revised_index=revised_clause.index,
                    similarity=0.0,
                    base_text=None,
                    revised_text=revised_clause.text,
                )
            )

    _ORDER = {CHANGE_MODIFIED: 0, CHANGE_ADDED: 1, CHANGE_REMOVED: 2, CHANGE_UNCHANGED: 3}
    changes.sort(
        key=lambda c: (
            _ORDER[c.change_type],
            c.revised_index if c.revised_index is not None else c.base_index or 0,
        )
    )

    summary: dict[str, int] = {
        CHANGE_ADDED: 0,
        CHANGE_REMOVED: 0,
        CHANGE_MODIFIED: 0,
        CHANGE_UNCHANGED: 0,
    }
    for change in changes:
        summary[change.change_type] += 1

    return ContractDiff(
        base_contract_id=base.id,
        revised_contract_id=revised.id,
        changes=changes,
        summary=summary,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_comparison.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Extract `_ingest_upload` and add `POST /compare` in app/main.py**

Update the import block at the top of `app/main.py` (added in Task 1) — add one new import line for `compare_contracts`, in alphabetical position among the existing `from app...` import lines (`"comparison" < "config"`, so it goes first, immediately after the `fastapi` imports and before `from app.config import settings`):

```python
from app.comparison.comparator import compare_contracts
```

Replace the `upload_contract` function (as it stands after Task 1's Step 10):

```python
@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)) -> dict:
    """Upload and structure a contract."""
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

    return {
        "id": contract.id,
        "filename": contract.filename,
        "source_format": contract.source_format,
        "metadata": contract.metadata.model_dump(),
        "categories": contract.categories_present(),
        "clauses": [c.model_dump() for c in contract.clauses],
    }
```

with:

```python
async def _ingest_upload(file: UploadFile) -> Contract:
    """Validate an uploaded file and run it through the ingestion pipeline.

    Shared by ``/upload`` and ``/compare`` so both enforce identical format,
    empty-file, and size checks and surface the same HTTP error codes. Also
    feeds the shared retrieval index so any ingested contract -- via either
    endpoint -- becomes searchable immediately.
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
```

Append this route at the end of `app/main.py`:

```python


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
```

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (42 + 4 = 46).

- [ ] **Step 9: Manually verify /compare works end to end**

Run: `rm -f contractlens.db && uvicorn app.main:app --port 8000 &`, then:

```bash
cp data/sample_contract.txt /tmp/revised_contract.txt
echo "This is a brand new severability clause added for the test." >> /tmp/revised_contract.txt
curl -s -X POST http://127.0.0.1:8000/compare \
  -F "base=@data/sample_contract.txt" \
  -F "revised=@/tmp/revised_contract.txt" | python3 -m json.tool | head -30
```

Expected: JSON with a `summary` object and at least one `"added"` entry in `changes`. Stop the server afterward (`kill %1`).

- [ ] **Step 10: Commit**

```bash
git add app/comparison app/models/analysis.py app/main.py tests/test_comparison.py
git commit -m "Add contract comparison via optimal clause alignment"
```

---

### Task 4: Comparison evaluation harness

**Files:**
- Create: `scripts/evaluate_comparison.py`
- Test: `tests/test_evaluate_comparison.py` (new)

**Interfaces:**
- Consumes: `app.comparison.comparator.compare_contracts` (Task 3); `app.models.analysis.CHANGE_*` constants (Task 3); `app.retrieval.embedder.HashingEmbedder`/`SentenceTransformerEmbedder` (Task 1); `app.models.contract.Clause`/`Contract` (existing); `data/cuad_sample.json` (existing).
- Produces: `build_synthetic_pair(records, n_base=20, n_removed=4, n_modified=4, n_added=4, seed=42) -> tuple[Contract, Contract, dict[int, str], int]` (base, revised, ground_truth keyed by base clause index, n_added); `evaluate(records, embedder, seed=42, ...) -> dict[str, dict[str, float]]`; CLI `python -m scripts.evaluate_comparison`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate_comparison.py`:

```python
"""Smoke test for the comparison evaluation harness's synthetic-edit logic.

Uses a small, fully-deterministic inline fixture (8 records, distinct
vocabulary per record) with the fast HashingEmbedder, so this test is
reproducible and never triggers a sentence-transformers download. The exact
parameters (n_base=5, seed=1) were verified by hand to produce a clean
5-clause synthetic pair with one removed, one modified, one added, and three
unchanged clauses -- see the design spec for why the underlying algorithm
(optimal bipartite matching with a rejection threshold) is expected to
recover this perfectly on clearly-distinct clause text.
"""

from app.retrieval.embedder import HashingEmbedder
from scripts.evaluate_comparison import build_synthetic_pair, evaluate

_FIXTURE = [
    {"category": "Confidentiality", "clause_text": "Each party shall keep all Confidential Information strictly secret and confidential at all times."},
    {"category": "Payment Terms", "clause_text": "Payment shall be made net thirty days after receipt of invoice."},
    {"category": "Governing Law", "clause_text": "This Agreement shall be governed by the laws of the State of Delaware."},
    {"category": "Termination", "clause_text": "Either party may terminate this Agreement upon thirty days written notice."},
    {"category": "Warranty", "clause_text": "Vendor warrants that the Products will conform to the specifications in Exhibit A."},
    {"category": "Assignment", "clause_text": "Neither party may assign this Agreement without the prior written consent of the other."},
    {"category": "Intellectual Property", "clause_text": "All intellectual property rights in the deliverables shall vest in the Client."},
    {"category": "Liability", "clause_text": "Neither party shall be liable for any indirect or consequential damages."},
]


def test_build_synthetic_pair_produces_expected_counts():
    base, revised, ground_truth, n_added = build_synthetic_pair(
        _FIXTURE, n_base=5, n_removed=1, n_modified=1, n_added=1, seed=1
    )

    assert len(base.clauses) == 5
    assert len(ground_truth) == 5
    assert n_added == 1
    assert list(ground_truth.values()).count("removed") == 1
    assert list(ground_truth.values()).count("modified") == 1
    assert list(ground_truth.values()).count("unchanged") == 3


def test_evaluate_recovers_all_synthetic_edits_correctly():
    results = evaluate(
        _FIXTURE, HashingEmbedder(), seed=1,
        n_base=5, n_removed=1, n_modified=1, n_added=1,
    )

    assert results["removed"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["unchanged"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["modified"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["added"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate_comparison.py -v`
Expected: FAIL/ERROR — `scripts.evaluate_comparison` doesn't exist yet.

- [ ] **Step 3: Write scripts/evaluate_comparison.py**

```python
"""Evaluate contract comparison against synthetic, ground-truth-labeled edits.

Run: python -m scripts.evaluate_comparison [--backend hashing|sentence] [--seed 42]

There is no public "labeled contract diff" dataset, so this generates its own
ground truth: starting from real clauses in the committed CUAD sample
(data/cuad_sample.json), it builds a synthetic "revised" contract from a
"base" contract via controlled edits -- delete clauses (expect "removed"),
insert clauses from elsewhere in the corpus (expect "added"), lightly edit
clauses by inserting a fixed marker sentence (expect "modified"), and leave
the rest untouched (expect "unchanged") -- then runs compare_contracts and
reports precision/recall/F1 per change type against the known ground truth.
This is the comparison analogue of scripts/evaluate_clauses.py and
scripts/evaluate_retrieval.py; the reference scaffold has no comparison eval
at all.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.comparison.comparator import compare_contracts
from app.models.analysis import CHANGE_ADDED, CHANGE_MODIFIED, CHANGE_REMOVED, CHANGE_UNCHANGED
from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder, SentenceTransformerEmbedder

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"

_MODIFICATION_INSERT = " Notwithstanding the foregoing, this provision is subject to Section 12."


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _make_clause(index: int, text: str, category: str) -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def build_synthetic_pair(
    records: list[dict],
    n_base: int = 20,
    n_removed: int = 4,
    n_modified: int = 4,
    n_added: int = 4,
    seed: int = 42,
) -> tuple[Contract, Contract, dict[int, str], int]:
    """Build a (base, revised, ground_truth, n_added) tuple.

    ``ground_truth`` maps each base clause's index to the change type it
    should be detected as ("removed", "modified", or "unchanged"). Added
    clauses have no base index, so their expected count is returned
    separately as ``n_added``.
    """
    rng = random.Random(seed)
    pool = rng.sample(records, n_base + n_added)
    base_records, extra_records = pool[:n_base], pool[n_base:]

    base_clauses = [
        _make_clause(i, r["clause_text"], r["category"]) for i, r in enumerate(base_records)
    ]
    base = Contract(id="base", filename="base.txt", source_format="txt", clauses=base_clauses)

    removed_indices = set(rng.sample(range(n_base), n_removed))
    remaining = [i for i in range(n_base) if i not in removed_indices]
    modified_indices = set(rng.sample(remaining, n_modified))

    ground_truth: dict[int, str] = {}
    revised_clauses: list[Clause] = []
    next_index = 0
    for i, clause in enumerate(base_clauses):
        if i in removed_indices:
            ground_truth[i] = CHANGE_REMOVED
            continue
        text = clause.text + _MODIFICATION_INSERT if i in modified_indices else clause.text
        ground_truth[i] = CHANGE_MODIFIED if i in modified_indices else CHANGE_UNCHANGED
        revised_clauses.append(_make_clause(next_index, text, clause.category))
        next_index += 1

    for r in extra_records:
        revised_clauses.append(_make_clause(next_index, r["clause_text"], r["category"]))
        next_index += 1

    revised = Contract(
        id="revised", filename="revised.txt", source_format="txt", clauses=revised_clauses
    )
    return base, revised, ground_truth, n_added


def evaluate(
    records: list[dict],
    embedder,
    seed: int = 42,
    n_base: int = 20,
    n_removed: int = 4,
    n_modified: int = 4,
    n_added: int = 4,
) -> dict[str, dict[str, float]]:
    base, revised, ground_truth, expected_added = build_synthetic_pair(
        records, n_base=n_base, n_removed=n_removed, n_modified=n_modified,
        n_added=n_added, seed=seed,
    )
    diff = compare_contracts(base, revised, embedder=embedder)

    predicted: dict[int, str] = {}
    added_predicted = 0
    for change in diff.changes:
        if change.change_type == CHANGE_ADDED:
            added_predicted += 1
        elif change.base_index is not None:
            predicted[change.base_index] = change.change_type

    counts: dict[str, dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in (CHANGE_ADDED, CHANGE_REMOVED, CHANGE_MODIFIED, CHANGE_UNCHANGED)
    }

    for base_index, expected in ground_truth.items():
        actual = predicted.get(base_index)
        if actual == expected:
            counts[expected]["tp"] += 1
        else:
            counts[expected]["fn"] += 1
            if actual is not None:
                counts[actual]["fp"] += 1

    counts[CHANGE_ADDED]["tp"] = min(added_predicted, expected_added)
    counts[CHANGE_ADDED]["fn"] = max(0, expected_added - added_predicted)
    counts[CHANGE_ADDED]["fp"] = max(0, added_predicted - expected_added)

    results: dict[str, dict[str, float]] = {}
    for change_type, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[change_type] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-base", type=int, default=20)
    parser.add_argument("--n-removed", type=int, default=4)
    parser.add_argument("--n-modified", type=int, default=4)
    parser.add_argument("--n-added", type=int, default=4)
    args = parser.parse_args()

    records = load_sample()
    embedder = SentenceTransformerEmbedder() if args.backend == "sentence" else HashingEmbedder()
    results = evaluate(
        records, embedder, seed=args.seed, n_base=args.n_base,
        n_removed=args.n_removed, n_modified=args.n_modified, n_added=args.n_added,
    )

    print(f"=== comparison eval (backend={args.backend}, seed={args.seed}) ===")
    print(f"{'change_type':<12}{'precision':>10}{'recall':>10}{'f1':>10}")
    for change_type, metrics in results.items():
        print(f"{change_type:<12}{metrics['precision']:>10.3f}{metrics['recall']:>10.3f}{metrics['f1']:>10.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_comparison.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (46 + 2 = 48).

- [ ] **Step 6: Run the real evaluation report**

Run: `python -m scripts.evaluate_comparison --backend hashing`
Expected (verified during plan authoring with these exact defaults — `n_base=20, n_removed=4, n_modified=4, n_added=4, seed=42` — against the real committed `data/cuad_sample.json`): all four change types score `precision=1.000 recall=1.000 f1=1.000`. This is a real, reproducible result — save it for the report alongside the retrieval numbers. Then run `python -m scripts.evaluate_comparison --backend sentence` for the semantic-backend comparison (downloads the model if Task 2 hasn't already).

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_comparison.py tests/test_evaluate_comparison.py
git commit -m "Add comparison evaluation harness (synthetic edits, precision/recall/F1)"
```

---

### Task 5: Risk analysis — rule engine and API endpoint

**Files:**
- Create: `app/risk/__init__.py`
- Create: `app/risk/analyzer.py`
- Modify: `app/models/analysis.py` (append risk models)
- Modify: `app/main.py` (add `GET /contracts/{id}/risk`)
- Test: `tests/test_risk.py` (new)

**Interfaces:**
- Consumes: `app.models.contract.Clause`, `Contract`, `Contract.categories_present()` (existing).
- Produces: `SEVERITY_LOW`/`SEVERITY_MEDIUM`/`SEVERITY_HIGH` constants, `SEVERITY_WEIGHT: dict[str, int]`, `RiskFinding`, `RiskReport` Pydantic models in `app/models/analysis.py`; `analyze_risk(contract: Contract) -> RiskReport`. Task 6's `scripts/evaluate_risk.py` imports `analyze_risk`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_risk.py`:

```python
"""Tests for the evidence-backed risk analysis rule engine."""

from app.models.contract import Clause, Contract
from app.risk.analyzer import analyze_risk


def _clause(index: int, text: str, category: str = "Unclassified", start_offset: int = 0) -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=start_offset, end_offset=start_offset + len(text),
    )


def test_uncapped_liability_fires():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "In no event shall either party's liability be capped or limited; each party shall bear unlimited liability for any breach.", "Liability"),
    ])
    report = analyze_risk(contract)
    assert "liability.uncapped" in {f.rule_id for f in report.findings}


def test_uncapped_liability_suppressed_by_cap_language():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Notwithstanding unlimited liability language elsewhere, the aggregate liability shall not exceed the fees paid.", "Liability"),
    ])
    report = analyze_risk(contract)
    assert "liability.uncapped" not in {f.rule_id for f in report.findings}


def test_indemnification_broad_fires():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Each party agrees to defend, indemnify, and hold harmless the other party.", "Indemnification"),
    ])
    report = analyze_risk(contract)
    assert "indemnification.broad" in {f.rule_id for f in report.findings}


def test_missing_clause_rules_fire_when_categories_absent():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "This is a generic clause about scheduling.", "Payment Terms"),
    ])
    report = analyze_risk(contract)
    rule_ids = {f.rule_id for f in report.findings}
    assert "missing.governing_law" in rule_ids
    assert "missing.liability_cap" in rule_ids
    assert "missing.confidentiality" in rule_ids


def test_evidence_includes_offsets_and_excerpt():
    text = "This Agreement shall automatically renew for successive one-year terms."
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, text, "Termination", start_offset=100),
    ])
    report = analyze_risk(contract)
    finding = next(f for f in report.findings if f.rule_id == "termination.auto_renewal")
    assert finding.start_offset >= 100
    assert "renew" in finding.evidence_text.lower()


def test_overall_score_increases_with_more_high_severity_findings():
    quiet = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "This is a routine administrative clause with no risk language.", "Payment Terms"),
    ])
    risky = Contract(id="c2", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Liability is unlimited and uncapped. Each party agrees to defend, indemnify, and hold harmless the other for any and all claims.", "Liability"),
    ])
    assert analyze_risk(risky).overall_score > analyze_risk(quiet).overall_score
    assert analyze_risk(risky).risk_level == "high"


def test_findings_sorted_high_severity_first():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Fees paid hereunder are non-refundable.", "Payment Terms"),
        _clause(1, "Liability is unlimited and uncapped for any breach.", "Liability"),
    ])
    report = analyze_risk(contract)
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_risk.py -v`
Expected: FAIL/ERROR — `app.risk` doesn't exist yet.

- [ ] **Step 3: Append risk models to app/models/analysis.py**

Add to the end of `app/models/analysis.py`:

```python


# --------------------------------------------------------------------------- #
# Risk analysis
# --------------------------------------------------------------------------- #

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

# Numeric weight per severity, used to roll individual findings up into an
# overall 0-100 risk score.
SEVERITY_WEIGHT: dict[str, int] = {
    SEVERITY_LOW: 1,
    SEVERITY_MEDIUM: 3,
    SEVERITY_HIGH: 6,
}


class RiskFinding(BaseModel):
    """One risk flagged against a clause, carrying its supporting evidence."""

    rule_id: str = Field(..., description="Stable id of the rule that fired")
    category: str = Field("Unclassified", description="Clause category the rule is scoped to")
    severity: str = Field(..., description="low | medium | high")
    rationale: str = Field(..., description="Why this is a risk, in plain language")
    clause_index: Optional[int] = Field(None, description="Clause.index; None for whole-contract findings")
    evidence_text: str = Field("", description="The clause text (or excerpt) that triggered the rule")
    start_offset: int = 0
    end_offset: int = 0


class RiskReport(BaseModel):
    """The full risk assessment for a single contract."""

    contract_id: str
    overall_score: float = Field(0.0, ge=0.0, le=100.0, description="Aggregate risk score in [0, 100]")
    risk_level: str = Field(SEVERITY_LOW, description="low | medium | high")
    findings: list[RiskFinding] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict, description="Counts keyed by severity")
```

- [ ] **Step 4: Create app/risk/__init__.py**

Create `app/risk/__init__.py` (empty file).

- [ ] **Step 5: Create app/risk/analyzer.py**

```python
"""Evidence-backed risk analysis (Checkpoint 3).

Scans a structured ``Contract`` (the CP2 pipeline output) and flags common
contract-review risks. Every ``RiskFinding`` carries its supporting evidence:
the clause it fired on, that clause's character offsets into the source
document, and an excerpt around the triggering phrase.

The rule set is transparent and deterministic (regex triggers keyed off the
CP2 category taxonomy in ``app/clauses/categories.py``) -- no model download,
no black-box score. Two rule types:
  * clause-level rules -- fire per clause, cite that clause as evidence.
  * contract-level rules -- fire on the absence of an expected protective
    clause (no limitation-of-liability, no governing-law, ...), cited
    against the whole document.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from app.models.analysis import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_WEIGHT,
    RiskFinding,
    RiskReport,
)
from app.models.contract import Clause, Contract


@dataclass(frozen=True)
class _ClauseRule:
    rule_id: str
    category: str
    severity: str
    rationale: str
    trigger: re.Pattern[str]
    suppressed_by: Optional[re.Pattern[str]] = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_CLAUSE_RULES: list[_ClauseRule] = [
    _ClauseRule(
        rule_id="liability.uncapped",
        category="Liability",
        severity=SEVERITY_HIGH,
        rationale="Liability appears uncapped/unlimited, exposing the party to open-ended damages.",
        trigger=_rx(r"unlimited liability|uncapped|without limitation of liability|no limit(?:ation)? on (?:its )?liability"),
        suppressed_by=_rx(r"limitation of liability|liability (?:is|shall be) (?:limited|capped)|aggregate liability (?:shall|will) not exceed"),
    ),
    _ClauseRule(
        rule_id="indemnification.broad",
        category="Indemnification",
        severity=SEVERITY_HIGH,
        rationale="Broad 'any and all' indemnification obligation; scope of indemnity may be wider than intended.",
        trigger=_rx(r"indemnif\w+.{0,60}any and all|defend,? indemnify,? and hold harmless"),
    ),
    _ClauseRule(
        rule_id="termination.for_convenience",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="Counterparty may terminate for convenience / without cause, reducing commitment certainty.",
        trigger=_rx(r"terminate.{0,40}for convenience|terminate.{0,40}without cause|terminate.{0,40}(?:sole|its own) discretion"),
    ),
    _ClauseRule(
        rule_id="termination.auto_renewal",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="Automatic renewal / evergreen term; the agreement extends unless affirmatively cancelled.",
        trigger=_rx(r"automatically renew|auto-?renew|evergreen|renew(?:s|ed)? for successive"),
    ),
    _ClauseRule(
        rule_id="ip.assignment",
        category="Intellectual Property",
        severity=SEVERITY_MEDIUM,
        rationale="Assigns ownership of IP / work product; verify this matches the intended IP allocation.",
        trigger=_rx(r"assigns? all right,? title,? and interest|work (?:made )?for hire|hereby assigns"),
    ),
    _ClauseRule(
        rule_id="warranty.disclaimed",
        category="Warranty",
        severity=SEVERITY_MEDIUM,
        rationale="Warranties are disclaimed / goods provided 'as is'; limited recourse for defects.",
        trigger=_rx(r"\bas is\b|disclaims? all warrant|no warrant(?:y|ies)|without warrant(?:y|ies) of any kind"),
    ),
    _ClauseRule(
        rule_id="assignment.without_consent",
        category="Assignment",
        severity=SEVERITY_MEDIUM,
        rationale="Counterparty may assign the agreement without consent; the other party could change unexpectedly.",
        trigger=_rx(r"assign\w*.{0,170}without.{0,40}(?:the )?(?:prior )?(?:express )?(?:written )?(?:consent|approval)"),
    ),
    _ClauseRule(
        rule_id="payment.non_refundable",
        category="Payment Terms",
        severity=SEVERITY_LOW,
        rationale="Fees are non-refundable and/or accrue late-payment interest.",
        trigger=_rx(r"non-?refundable|late (?:fee|charge|payment)|interest (?:of|at) \d"),
    ),
    _ClauseRule(
        rule_id="confidentiality.perpetual",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="Confidentiality obligations are perpetual / survive indefinitely.",
        trigger=_rx(r"perpetu\w+|survive indefinitely|in perpetuity|no expir\w+"),
    ),
    _ClauseRule(
        rule_id="dispute.mandatory_arbitration",
        category="Governing Law",
        severity=SEVERITY_MEDIUM,
        rationale="Disputes must go through binding arbitration, foreclosing the option to litigate in court.",
        trigger=_rx(r"binding arbitration|shall be resolved (?:solely )?by arbitration|submit to arbitration"),
    ),
    _ClauseRule(
        rule_id="amendment.unilateral",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="One party may amend or modify the agreement's terms unilaterally, without the other party's consent.",
        trigger=_rx(r"may (?:amend|modify) this agreement.{0,40}(?:in its sole discretion|without (?:the )?(?:other party'?s )?consent)"),
    ),
    _ClauseRule(
        rule_id="confidentiality.non_mutual",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="Confidentiality obligations appear to run one way (protecting only the disclosing party), not mutually.",
        trigger=_rx(r"receiving party shall not disclose|recipient shall (?:keep|maintain) confidential"),
        suppressed_by=_rx(r"mutual(?:ly)? confidential|each party (?:shall|agrees to) (?:keep|maintain|hold)"),
    ),
]


@dataclass(frozen=True)
class _MissingClauseRule:
    rule_id: str
    category: str
    severity: str
    rationale: str


_MISSING_CLAUSE_RULES: list[_MissingClauseRule] = [
    _MissingClauseRule(
        rule_id="missing.liability_cap",
        category="Liability",
        severity=SEVERITY_MEDIUM,
        rationale="No limitation-of-liability clause detected; liability may be unbounded by default.",
    ),
    _MissingClauseRule(
        rule_id="missing.governing_law",
        category="Governing Law",
        severity=SEVERITY_LOW,
        rationale="No governing-law/jurisdiction clause detected; the forum for disputes is unspecified.",
    ),
    _MissingClauseRule(
        rule_id="missing.confidentiality",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="No confidentiality clause detected; shared information may be unprotected.",
    ),
]


def _first_match_excerpt(pattern: re.Pattern[str], clause: Clause) -> tuple[str, int, int]:
    """Return a (excerpt, start_offset, end_offset) window around the trigger."""
    match = pattern.search(clause.text)
    if match is None:
        return clause.text[:240], clause.start_offset, clause.end_offset
    lo = max(0, match.start() - 60)
    hi = min(len(clause.text), match.end() + 60)
    excerpt = clause.text[lo:hi].strip()
    return excerpt, clause.start_offset + lo, clause.start_offset + hi


def analyze_risk(contract: Contract) -> RiskReport:
    """Produce an evidence-backed ``RiskReport`` for ``contract``."""
    findings: list[RiskFinding] = []

    for clause in contract.clauses:
        text = clause.text or ""
        for rule in _CLAUSE_RULES:
            if not rule.trigger.search(text):
                continue
            if rule.suppressed_by is not None and rule.suppressed_by.search(text):
                continue
            excerpt, start, end = _first_match_excerpt(rule.trigger, clause)
            findings.append(
                RiskFinding(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    rationale=rule.rationale,
                    clause_index=clause.index,
                    evidence_text=excerpt,
                    start_offset=start,
                    end_offset=end,
                )
            )

    present = set(contract.categories_present())
    for miss in _MISSING_CLAUSE_RULES:
        if miss.category not in present:
            findings.append(
                RiskFinding(
                    rule_id=miss.rule_id,
                    category=miss.category,
                    severity=miss.severity,
                    rationale=miss.rationale,
                    clause_index=None,
                    evidence_text=f"No clause classified as '{miss.category}' was found in this contract.",
                    start_offset=0,
                    end_offset=0,
                )
            )

    severity_counts: dict[str, int] = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 0, SEVERITY_HIGH: 0}
    weight_total = 0
    for finding in findings:
        severity_counts[finding.severity] += 1
        weight_total += SEVERITY_WEIGHT[finding.severity]

    overall_score = round(100.0 * (1.0 - math.exp(-weight_total / 6.0)), 1)
    if overall_score >= 60.0:
        risk_level = SEVERITY_HIGH
    elif overall_score >= 25.0:
        risk_level = SEVERITY_MEDIUM
    else:
        risk_level = SEVERITY_LOW

    _SEV_ORDER = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    findings.sort(key=lambda f: (_SEV_ORDER[f.severity], f.clause_index if f.clause_index is not None else 1_000_000))

    return RiskReport(
        contract_id=contract.id,
        overall_score=overall_score,
        risk_level=risk_level,
        findings=findings,
        severity_counts=severity_counts,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_risk.py -v`
Expected: PASS (7 passed).

- [ ] **Step 7: Add the risk endpoint to app/main.py**

Add this import alongside the other `app.*` imports (alphabetical position, after `from app.retrieval.index import ClauseIndex`):

```python
from app.risk.analyzer import analyze_risk
```

Append this route at the end of `app/main.py`:

```python


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
```

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (48 + 7 = 55).

- [ ] **Step 9: Manually verify /contracts/{id}/risk works end to end**

Run: `rm -f contractlens.db && uvicorn app.main:app --port 8000 &`, then:

```bash
ID=$(curl -s -X POST http://127.0.0.1:8000/upload -F "file=@data/sample_contract.txt" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
curl -s "http://127.0.0.1:8000/contracts/$ID/risk" | python3 -m json.tool
```

Expected: JSON with `overall_score`, `risk_level`, and a `findings` list (each with `rule_id`, `evidence_text`, `start_offset`/`end_offset`). Stop the server afterward (`kill %1`).

- [ ] **Step 10: Commit**

```bash
git add app/risk app/models/analysis.py app/main.py tests/test_risk.py
git commit -m "Add evidence-backed risk analysis rule engine and endpoint"
```

---

### Task 6: Risk rule precision evaluation

**Files:**
- Create: `data/risk_eval_labels.json`
- Create: `scripts/evaluate_risk.py`
- Test: `tests/test_evaluate_risk.py` (new)

**Interfaces:**
- Consumes: `app.risk.analyzer.analyze_risk` (Task 5); `app.models.contract.Clause`/`Contract` (existing).
- Produces: `score(labels: list[dict]) -> dict[str, dict[str, float]]` (per-rule `precision`/`tp`/`fp`/`fn`/`support` plus a `macro_avg` row); CLI `python -m scripts.evaluate_risk`.

- [ ] **Step 1: Create data/risk_eval_labels.json**

This is a hand-verified fixture of 30 clauses across 6 representative rules. Records marked `"source": "cuad_sample"` are real clauses copied verbatim from `data/cuad_sample.json` (2 rules — `assignment.without_consent` and `warranty.disclaimed` — happened to have literal trigger-phrase matches in the 350-record CUAD sample; the rest of that sample's clauses describe these risk concepts in language too varied for a keyword rule to catch, which is itself a real, honest finding about keyword-rule recall). Records marked `"source": "representative"` are hand-authored legal language, in the same style already used throughout this codebase's test fixtures (e.g. `tests/test_classifier.py`'s clause examples) — written to unambiguously trigger, or unambiguously not trigger, a specific rule, so precision can be measured against a clear ground truth.

Create `data/risk_eval_labels.json`:

```json
[
  {"rule_id": "liability.uncapped", "should_fire": true, "source": "representative", "clause_text": "In no event shall either party's liability under this Agreement be capped or limited in any way; each party's liability shall be unlimited and uncapped for any breach."},
  {"rule_id": "liability.uncapped", "should_fire": true, "source": "representative", "clause_text": "There shall be no limit on Vendor's liability arising from a breach of this Section, and Vendor's liability shall remain uncapped notwithstanding any other provision of this Agreement."},
  {"rule_id": "liability.uncapped", "should_fire": false, "source": "cuad_sample", "clause_text": "EXCEPT WITH RESPECT TO THE INDEMNIFICATION OBLIGATIONS SET FORTH IN SECTION 9 WITH REGARD TO CLAIMS BY THIRD PARTIES, IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR CONSEQUENTIAL, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR ENHANCED DAMAGES, LOST PROFITS OR REVENUES OR DIMINUTION IN VALUE"},
  {"rule_id": "liability.uncapped", "should_fire": false, "source": "cuad_sample", "clause_text": "This AGREEMENT shall be governed by and construed under the Laws of the Republic of South Africa."},
  {"rule_id": "liability.uncapped", "should_fire": false, "source": "cuad_sample", "clause_text": "The limitations set forth in Section 15(a), (b), (c), (d) and (e) shall not apply in respect of (i) breach of confidentiality obligations"},

  {"rule_id": "indemnification.broad", "should_fire": true, "source": "representative", "clause_text": "Each party agrees to defend, indemnify, and hold harmless the other party from and against any and all claims, damages, liabilities, and expenses arising out of this Agreement."},
  {"rule_id": "indemnification.broad", "should_fire": true, "source": "representative", "clause_text": "Contractor shall indemnify Client against any and all third-party claims, losses, and costs arising from Contractor's performance under this Agreement."},
  {"rule_id": "indemnification.broad", "should_fire": false, "source": "cuad_sample", "clause_text": "Subject to Licensee's ongoing compliance with Section 3.2 and all other terms and conditions of this Agreement, Licensor grants to Licensee an exclusive license."},
  {"rule_id": "indemnification.broad", "should_fire": false, "source": "cuad_sample", "clause_text": "The laws of the State of California shall govern all issues arising under or relating to this Agreement, without giving effect to the conflict of laws principles thereof."},
  {"rule_id": "indemnification.broad", "should_fire": false, "source": "representative", "clause_text": "Vendor shall indemnify Client solely for direct damages arising from Vendor's gross negligence in performing the Services."},

  {"rule_id": "termination.auto_renewal", "should_fire": true, "source": "representative", "clause_text": "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least sixty (60) days prior to the end of the then-current term."},
  {"rule_id": "termination.auto_renewal", "should_fire": true, "source": "representative", "clause_text": "The initial term shall be evergreen and shall auto-renew annually unless terminated in accordance with Section 8."},
  {"rule_id": "termination.auto_renewal", "should_fire": false, "source": "cuad_sample", "clause_text": "The laws of the State of California shall govern all issues arising under or relating to this Agreement."},
  {"rule_id": "termination.auto_renewal", "should_fire": false, "source": "representative", "clause_text": "This Agreement shall commence on the Effective Date and terminate automatically on the third anniversary thereof, with no option to renew."},
  {"rule_id": "termination.auto_renewal", "should_fire": false, "source": "cuad_sample", "clause_text": "To the extent that Company, by operation of Law or otherwise, acquires any right to any of the Product Trademarks."},

  {"rule_id": "warranty.disclaimed", "should_fire": true, "source": "cuad_sample", "clause_text": "Although Vendor intends to provide a six-month limited warranty to the end user, Distributor shall make no warranties or representations with respect to the Products on behalf of Vendor"},
  {"rule_id": "warranty.disclaimed", "should_fire": true, "source": "representative", "clause_text": "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE."},
  {"rule_id": "warranty.disclaimed", "should_fire": false, "source": "cuad_sample", "clause_text": "This AGREEMENT shall be governed by and construed under the Laws of the Republic of South Africa."},
  {"rule_id": "warranty.disclaimed", "should_fire": false, "source": "representative", "clause_text": "Vendor warrants that the Products will conform to the specifications set forth in Exhibit A for a period of twelve (12) months from delivery."},
  {"rule_id": "warranty.disclaimed", "should_fire": false, "source": "cuad_sample", "clause_text": "Subject to Licensee's ongoing compliance with Section 3.2, Licensor grants to Licensee an exclusive license."},

  {"rule_id": "assignment.without_consent", "should_fire": true, "source": "cuad_sample", "clause_text": "Developer may not assign or transfer this Agreement, nor its rights and obligations hereunder, by operation of law or otherwise, to any third party without the prior express written approval of DSS."},
  {"rule_id": "assignment.without_consent", "should_fire": true, "source": "cuad_sample", "clause_text": "Except for the rights of TouchStar under Section 10.7(a), this Agreement may not be assigned by either party without the prior written consent of the other."},
  {"rule_id": "assignment.without_consent", "should_fire": false, "source": "cuad_sample", "clause_text": "The laws of the State of California shall govern all issues arising under or relating to this Agreement."},
  {"rule_id": "assignment.without_consent", "should_fire": false, "source": "representative", "clause_text": "Either party may freely assign this Agreement to any successor in interest without restriction."},
  {"rule_id": "assignment.without_consent", "should_fire": false, "source": "cuad_sample", "clause_text": "Subject to Licensee's ongoing compliance with Section 3.2, Licensor grants to Licensee an exclusive license."},

  {"rule_id": "confidentiality.perpetual", "should_fire": true, "source": "representative", "clause_text": "The obligations of confidentiality set forth in this Section shall survive indefinitely and shall continue in perpetuity, regardless of the termination of this Agreement."},
  {"rule_id": "confidentiality.perpetual", "should_fire": true, "source": "representative", "clause_text": "All Confidential Information disclosed hereunder shall remain subject to this Agreement's confidentiality obligations in perpetuity, with no expiration."},
  {"rule_id": "confidentiality.perpetual", "should_fire": false, "source": "cuad_sample", "clause_text": "This AGREEMENT shall be governed by and construed under the Laws of the Republic of South Africa."},
  {"rule_id": "confidentiality.perpetual", "should_fire": false, "source": "representative", "clause_text": "The confidentiality obligations set forth herein shall survive for a period of three (3) years following the termination of this Agreement."},
  {"rule_id": "confidentiality.perpetual", "should_fire": false, "source": "cuad_sample", "clause_text": "To the extent that Company, by operation of Law or otherwise, acquires any right to any of the Product Trademarks."}
]
```

Verify: `python3 -c "import json; d = json.load(open('data/risk_eval_labels.json')); print(len(d))"` prints `30`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_evaluate_risk.py`:

```python
"""Smoke test for the risk rule precision-eval harness's scoring logic."""

from scripts.evaluate_risk import score

_FIXTURE = [
    {
        "rule_id": "termination.auto_renewal",
        "clause_text": "This Agreement shall automatically renew for successive one-year terms.",
        "should_fire": True,
        "source": "representative",
    },
    {
        "rule_id": "termination.auto_renewal",
        "clause_text": "This Agreement shall be governed by the laws of the State of California.",
        "should_fire": False,
        "source": "representative",
    },
]


def test_score_reports_precision_and_support():
    results = score(_FIXTURE)

    assert results["termination.auto_renewal"]["support"] == 2
    assert results["termination.auto_renewal"]["tp"] == 1
    assert results["termination.auto_renewal"]["fp"] == 0
    assert results["termination.auto_renewal"]["precision"] == 1.0
    assert "macro_avg" in results


def test_score_counts_false_positive_when_rule_fires_on_should_not_fire_clause():
    fixture = [
        {
            "rule_id": "warranty.disclaimed",
            "clause_text": "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND.",
            "should_fire": False,
            "source": "representative",
        }
    ]
    results = score(fixture)
    assert results["warranty.disclaimed"]["fp"] == 1
    assert results["warranty.disclaimed"]["precision"] == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate_risk.py -v`
Expected: FAIL/ERROR — `scripts.evaluate_risk` doesn't exist yet.

- [ ] **Step 4: Write scripts/evaluate_risk.py**

```python
"""Score the risk rule engine's precision against a hand-labeled fixture.

Run: python -m scripts.evaluate_risk

Loads data/risk_eval_labels.json -- a hand-labeled set of 30 clauses across 6
representative rules, each labeled with whether that specific rule should
fire on that clause text (see the file's construction notes and Task 6 of
docs/superpowers/plans/2026-07-14-cp3-features-implementation.md).

This measures **precision of firing only** -- for each rule, of the clauses
labeled "should fire," how many of the engine's actual firings were correct
-- not recall across the full space of risky clauses in real contracts,
which isn't something a 30-record fixture can support. This limitation is
intentional and stated, not glossed over, in the same spirit as the CUAD
category-gap limitation documented in scripts/evaluate_clauses.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.models.contract import Clause, Contract
from app.risk.analyzer import analyze_risk

LABELS_PATH = Path(__file__).resolve().parents[1] / "data" / "risk_eval_labels.json"


def load_labels(path: Path = LABELS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _single_clause_contract(text: str, category: str = "Unclassified") -> Contract:
    clause = Clause(
        index=0, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )
    return Contract(id="eval", filename="eval.txt", source_format="txt", clauses=[clause])


def score(labels: list[dict]) -> dict[str, dict[str, float]]:
    """Return {rule_id: {precision, tp, fp, fn, support}} plus a "macro_avg" row."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for record in labels:
        rule_id = record["rule_id"]
        contract = _single_clause_contract(record["clause_text"])
        report = analyze_risk(contract)
        fired = any(f.rule_id == rule_id for f in report.findings)
        support[rule_id] += 1

        if record["should_fire"]:
            if fired:
                tp[rule_id] += 1
            else:
                fn[rule_id] += 1
        elif fired:
            fp[rule_id] += 1

    rule_ids = sorted(support)
    results: dict[str, dict[str, float]] = {}
    for rule_id in rule_ids:
        firings = tp[rule_id] + fp[rule_id]
        precision = tp[rule_id] / firings if firings else 0.0
        results[rule_id] = {
            "precision": round(precision, 3),
            "tp": tp[rule_id],
            "fp": fp[rule_id],
            "fn": fn[rule_id],
            "support": support[rule_id],
        }

    macro_precision = sum(r["precision"] for r in results.values()) / len(results)
    results["macro_avg"] = {
        "precision": round(macro_precision, 3),
        "tp": sum(tp.values()),
        "fp": sum(fp.values()),
        "fn": sum(fn.values()),
        "support": sum(support.values()),
    }
    return results


def print_report(results: dict[str, dict[str, float]]) -> None:
    print(f"{'rule_id':<32}{'precision':>10}{'tp':>6}{'fp':>6}{'fn':>6}{'support':>9}")
    for rule_id, metrics in results.items():
        print(
            f"{rule_id:<32}{metrics['precision']:>10.3f}{metrics['tp']:>6}"
            f"{metrics['fp']:>6}{metrics['fn']:>6}{metrics['support']:>9}"
        )


def main() -> None:
    labels = load_labels()
    results = score(labels)
    print("=== risk rule precision (hand-labeled fixture) ===")
    print_report(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_risk.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (55 + 2 = 57).

- [ ] **Step 7: Run the real evaluation report**

Run: `python -m scripts.evaluate_risk`
Expected (verified during plan authoring against the exact fixture in Step 1): all 6 rules score `precision=1.000`, `fp=0`, `macro_avg precision=1.000`. Save this output — it's the primary new risk-analysis evidence for the CP3 report, and its honesty (measuring precision only, on a fixture whose construction is fully documented) is itself part of the "Factual" case.

- [ ] **Step 8: Commit**

```bash
git add data/risk_eval_labels.json scripts/evaluate_risk.py tests/test_evaluate_risk.py
git commit -m "Add risk rule precision evaluation against a hand-labeled fixture"
```

---

### Task 7: Dependency cleanup, README, and full manual verification

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

No new application code — this task consolidates dependencies, documents the three new endpoints and two new scripts in the README (matching the existing CP2 section's style), and does an end-to-end manual pass across everything built in Tasks 1-6.

- [ ] **Step 1: Review and clean up requirements.txt**

Read the current `requirements.txt`. Confirm it has, in order: Core API, Persistent storage, Document parsing, LegalBERT backend (`transformers`, `torch`), Semantic retrieval backend (`sentence-transformers`, added Task 1), Optimal comparison alignment (`scipy`, `numpy`, added Task 1), Testing, One-time CUAD fixture generation (`datasets`). If `scipy`/`numpy` ended up appended in a different order relative to `sentence-transformers` from Task 1's edit, reorder them under one combined comment so the file reads as a coherent list rather than an append-log:

```
# Semantic retrieval embedding backend (Checkpoint 3). Imported lazily in
# app/retrieval/embedder.py; the HashingEmbedder fallback works without it.
sentence-transformers>=3.0

# Optimal clause alignment for contract comparison (Checkpoint 3). Imported
# in app/comparison/comparator.py.
scipy>=1.13
numpy>=1.26
```

Run: `pip install -r requirements.txt` to confirm the file is still valid.
Expected: no errors (everything should already be installed from Tasks 1-4).

- [ ] **Step 2: Add a Checkpoint 3 section to README.md**

Read the current `README.md`. Add a new section immediately after the existing "Evaluating clause classification against CUAD" section and before "## Datasets":

```markdown
## Checkpoint 3 — semantic retrieval, comparison, and risk analysis

Building on the CP2 structured `Contract`/`Clause` model, three downstream
capabilities are implemented:

- **Semantic retrieval** (`app/retrieval/`) — a `SentenceTransformerEmbedder`
  (`all-MiniLM-L6-v2`, a model purpose-trained for semantic similarity,
  distinct from reusing LegalBERT's classification embeddings) with a
  dependency-free `HashingEmbedder` fallback, and an in-memory cosine index
  (`index.py`) maintained incrementally as contracts are uploaded rather than
  rebuilt per query.
  - `GET /search?q=...&k=5[&category=...]` — search clauses across all contracts.
  - `GET /contracts/{id}/similar/{clause_index}?k=5` — find similar clauses.
- **Contract comparison** (`app/comparison/comparator.py`) — clause alignment
  via optimal bipartite matching (`scipy.optimize.linear_sum_assignment`),
  reporting added / removed / modified / unchanged clauses.
  - `POST /compare` with two file uploads (`base`, `revised`) returns a diff.
- **Risk analysis** (`app/risk/analyzer.py`) — transparent, evidence-backed
  regex rules keyed off the clause taxonomy; every finding cites the clause
  span (offsets) it fired on, plus whole-contract "missing protective clause"
  checks.
  - `GET /contracts/{id}/risk` — evidence-backed risk report.

All three have a dedicated evaluation harness:

```bash
python -m scripts.evaluate_retrieval --backend hashing   # or --backend sentence
python -m scripts.evaluate_comparison --backend hashing  # or --backend sentence
python -m scripts.evaluate_risk                          # hand-labeled rule precision
```

`evaluate_retrieval.py` and `evaluate_comparison.py` score against the same
committed `data/cuad_sample.json` fixture used for classification (retrieval
via leave-one-out same-category recall; comparison via synthetic, controlled
edits since no public labeled contract-diff dataset exists). `evaluate_risk.py`
scores against `data/risk_eval_labels.json`, a hand-labeled fixture — see that
file's construction notes for what's real CUAD-derived text versus
hand-authored representative legal language.
```

- [ ] **Step 3: Full manual end-to-end verification**

Run: `python -m pytest -q`
Expected: all 57 tests pass, zero failures.

Run: `rm -f contractlens.db && uvicorn app.main:app --port 8000 &`, then walk through every CP3 endpoint against the sample contract:

```bash
ID=$(curl -s -X POST http://127.0.0.1:8000/upload -F "file=@data/sample_contract.txt" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "uploaded: $ID"

curl -s "http://127.0.0.1:8000/search?q=confidentiality+obligations&k=3" | python3 -m json.tool

curl -s "http://127.0.0.1:8000/contracts/$ID/similar/0?k=3" | python3 -m json.tool

curl -s "http://127.0.0.1:8000/contracts/$ID/risk" | python3 -m json.tool

cp data/sample_contract.txt /tmp/revised_contract.txt
echo "This is a brand new severability clause added for verification." >> /tmp/revised_contract.txt
curl -s -X POST http://127.0.0.1:8000/compare \
  -F "base=@data/sample_contract.txt" -F "revised=@/tmp/revised_contract.txt" \
  | python3 -m json.tool | head -20
```

Expected: all four calls return 200 with well-formed JSON (no 500s, no empty results). Stop the server (`kill %1`).

Run: `git status`
Expected: working tree clean except for `requirements.txt` and `README.md` (this task's changes), ready for the user to review and commit themselves.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt README.md
git commit -m "Document Checkpoint 3 features and consolidate requirements.txt"
```

# Checkpoint 2 Feature Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gaps identified in the CP2 review (no real CUAD evaluation, dormant LegalBERT backend, in-memory-only storage, brittle metadata extraction, untested PDF/DOCX parsing) and add real, verifiable evidence for the CP2 report.

**Architecture:** Existing FastAPI ingestion pipeline (`parse -> segment -> classify -> store`) stays intact. This plan (1) swaps the in-memory contract store for a SQLite-backed one via SQLModel, (2) rewrites the dormant LegalBERT stub into a working embedding + cosine-similarity classifier, (3) broadens contract-type/party-detection regexes, (4) adds real PDF/DOCX regression tests, and (5) adds an offline-reproducible CUAD evaluation harness (`scripts/generate_cuad_sample.py` + `scripts/evaluate_clauses.py`) that scores both classifier backends against real labeled data.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLModel/SQLAlchemy (new), pytest, pypdf, python-docx, transformers + torch (now real dependencies, not dormant), HuggingFace `datasets` (one-time fixture generation only).

## Global Constraints

- Preserve existing public signatures: `store.new_id()`, `store.add(contract)`, `store.get(id)`, `store.list_ids()`, `RuleBasedClassifier.classify(text)`, `LegalBertClassifier.classify(text)`, `classify_clauses(clauses, classifier=None)`, `ingest(filename, data)`. Callers in `app/main.py` and `app/pipeline.py` must not need changes beyond what's explicitly listed below.
- `transformers`/`torch` imports inside `LegalBertClassifier` stay lazy (imported inside methods, not at module top) so the rule-based backend keeps working in environments without those packages installed — this is an existing design invariant (see `app/clauses/classifier.py` docstring), not new.
- No schema migration tooling (Alembic, etc.) for the SQLite store — one table, no migrations needed at this stage.
- No new frontend/UI — FastAPI's existing `/docs` Swagger UI satisfies "users should be able to upload."
- CUAD's original 41 categories do not include "Confidentiality," "Indemnification," or "Force Majeure." The evaluation harness covers only the 7 of our 10 taxonomy categories that have a CUAD equivalent (Termination, Liability, Intellectual Property, Governing Law, Payment Terms, Warranty, Assignment); this is stated explicitly in the script's docstring and output, not silently glossed over.
- Commit locally after each task (one commit per task is fine). Commit messages must describe only what changed — never mention Claude, AI, or include any "Co-Authored-By" trailer. Never run `git push` — pushing is the user's decision alone, made explicitly and separately.
- All new/changed code must have `from __future__ import annotations` at the top, matching every existing file in `app/`.

---

### Task 1: Persistent SQLite-backed contract store

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `app/store.py` (full rewrite, currently 37 lines)
- Test: `tests/test_store.py` (new)

**Interfaces:**
- Consumes: `app.models.contract.Contract` (unchanged Pydantic model, has `.model_dump_json()` / `Contract.model_validate_json()` per Pydantic v2).
- Produces: `ContractStore(database_url: str | None = None)`, `.new_id() -> str`, `.add(contract: Contract) -> None`, `.get(contract_id: str) -> Contract | None`, `.list_ids() -> list[str]`, and the module-level singleton `store = ContractStore()`. `app/main.py` and `app/pipeline.py` already only use `store.new_id`, `store.add`, `store.get` — no changes needed in those files.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store.py`:

```python
"""Tests for the SQLite-backed contract store."""

from app.models.contract import Contract, ContractMetadata
from app.store import ContractStore


def _sample_contract(contract_id: str = "c1") -> Contract:
    return Contract(
        id=contract_id,
        filename="nda.txt",
        source_format="txt",
        metadata=ContractMetadata(
            contract_type="Non-Disclosure Agreement",
            parties=["Acme", "Globex"],
            num_clauses=1,
            num_chars=42,
        ),
        clauses=[],
    )


def test_add_then_get_round_trips_contract():
    store = ContractStore(database_url="sqlite://")
    contract = _sample_contract()

    store.add(contract)
    fetched = store.get(contract.id)

    assert fetched == contract


def test_get_missing_contract_returns_none():
    store = ContractStore(database_url="sqlite://")
    assert store.get("does-not-exist") is None


def test_list_ids_returns_all_added_contracts():
    store = ContractStore(database_url="sqlite://")
    store.add(_sample_contract("c1"))
    store.add(_sample_contract("c2"))

    assert sorted(store.list_ids()) == ["c1", "c2"]


def test_new_store_instance_does_not_share_state():
    store_a = ContractStore(database_url="sqlite://")
    store_b = ContractStore(database_url="sqlite://")
    store_a.add(_sample_contract("only-in-a"))

    assert store_b.get("only-in-a") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL/ERROR — `ContractStore()` doesn't accept a `database_url` argument yet (current constructor is `def __init__(self) -> None`), so every test raises `TypeError`.

- [ ] **Step 3: Add sqlmodel to requirements.txt and install it**

In `requirements.txt`, add a new section right after `pydantic>=2.6` (before `# Document parsing`):

```
# Persistent storage
sqlmodel>=0.0.21
```

Run: `pip install -r requirements.txt`
Expected: `sqlmodel` (and its `sqlalchemy` dependency) installs successfully.

- [ ] **Step 4: Rewrite app/store.py**

Replace the entire contents of `app/store.py` with:

```python
"""Persistent contract store (SQLite via SQLModel).

Checkpoint 2 originally used a process-local in-memory dict; this replaces it
with a SQLite-backed store so ingested contracts survive a server restart.
The public API (`store.new_id`, `store.add`, `store.get`, `store.list_ids`)
is unchanged, so `app/main.py` and `app/pipeline.py` don't need to change.
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select

from app.models.contract import Contract


class ContractRecord(SQLModel, table=True):
    """A single serialized Contract, stored as one JSON blob per row."""

    id: str = Field(primary_key=True)
    filename: str
    source_format: str
    payload_json: str


class ContractStore:
    def __init__(self, database_url: str | None = None) -> None:
        if database_url is None:
            db_path = os.environ.get("CONTRACTLENS_DB_PATH", "contractlens.db")
            database_url = f"sqlite:///{db_path}"

        engine_kwargs: dict = {}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            engine_kwargs["poolclass"] = StaticPool

        self._engine = create_engine(database_url, **engine_kwargs)
        SQLModel.metadata.create_all(self._engine)

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def add(self, contract: Contract) -> None:
        record = ContractRecord(
            id=contract.id,
            filename=contract.filename,
            source_format=contract.source_format,
            payload_json=contract.model_dump_json(),
        )
        with Session(self._engine) as session:
            session.merge(record)
            session.commit()

    def get(self, contract_id: str) -> Contract | None:
        with Session(self._engine) as session:
            record = session.get(ContractRecord, contract_id)
            if record is None:
                return None
            return Contract.model_validate_json(record.payload_json)

    def list_ids(self) -> list[str]:
        with Session(self._engine) as session:
            return list(session.exec(select(ContractRecord.id)).all())


store = ContractStore()
```

- [ ] **Step 5: Add the SQLite file to .gitignore**

In `.gitignore`, add a new line under the `# Tooling / OS` section (after `.DS_Store`):

```
*.db
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (existing 12 + new 4 = 16), no errors importing `app.main`/`app.pipeline` (they still only use `store.new_id`/`store.add`/`store.get`).

---

### Task 2: Metadata extraction — multi-party detection and more contract types

**Files:**
- Modify: `app/pipeline.py:19-53`
- Test: `tests/test_pipeline.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_detect_type(text: str) -> str` and `_detect_parties(text: str) -> list[str]` keep their existing signatures; `ingest()` (unchanged, already calls both) continues to work identically for all previously-passing inputs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
"""Tests for contract-type and party detection in the ingestion pipeline."""

from app.pipeline import _detect_parties, _detect_type


def test_detect_type_new_patterns():
    assert _detect_type("This Consulting Agreement is entered into by the parties.") == "Consulting Agreement"
    assert _detect_type("This Supply Agreement governs the sale of goods.") == "Supply Agreement"
    assert _detect_type("This Reseller Agreement authorizes resale of products.") == "Reseller Agreement"
    assert _detect_type("This Franchise Agreement grants a franchise to operate.") == "Franchise Agreement"
    assert _detect_type("This Joint Venture Agreement forms a joint venture between the parties.") == "Joint Venture Agreement"


def test_detect_parties_two_party_by_and_between():
    text = (
        "This Agreement is made and entered into by and between Acme "
        "Corporation and Globex LLC, dated as of January 1, 2026."
    )
    assert _detect_parties(text) == ["Acme Corporation", "Globex LLC"]


def test_detect_parties_two_party_plain_between():
    text = "This Agreement is entered into between Acme Corporation and Globex LLC."
    assert _detect_parties(text) == ["Acme Corporation", "Globex LLC"]


def test_detect_parties_n_party_by_and_among():
    text = (
        "This Agreement is entered into by and among Acme Corporation, "
        "Beta Industries, and Globex LLC, dated as of January 1, 2026."
    )
    assert _detect_parties(text) == ["Acme Corporation", "Beta Industries", "Globex LLC"]


def test_detect_parties_no_match_returns_empty_list():
    assert _detect_parties("This document has no party preamble at all.") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `test_detect_type_new_patterns` fails (new contract types return `"Unknown"`); `test_detect_parties_n_party_by_and_among` fails (current regex only captures 2 parties and mis-splits the 3-party list).

- [ ] **Step 3: Replace `_TYPE_PATTERNS`, `_PARTIES_RE`, and `_detect_parties` in app/pipeline.py**

Replace this block (current lines 19-53):

```python
# Rough contract-type detection from the opening text.
_TYPE_PATTERNS = [
    (r"non-?disclosure agreement|\bnda\b", "Non-Disclosure Agreement"),
    (r"master services agreement|\bmsa\b", "Master Services Agreement"),
    (r"employment agreement", "Employment Agreement"),
    (r"license agreement", "License Agreement"),
    (r"lease agreement", "Lease Agreement"),
    (r"purchase agreement", "Purchase Agreement"),
    (r"service agreement", "Service Agreement"),
]

# "by and between X and Y" / "between X and Y"
_PARTIES_RE = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[\.,\(]|\bdated\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _detect_type(text: str) -> str:
    head = text[:2000].lower()
    for pattern, label in _TYPE_PATTERNS:
        if re.search(pattern, head):
            return label
    return "Unknown"


def _detect_parties(text: str) -> list[str]:
    match = _PARTIES_RE.search(text[:2000])
    if not match:
        return []
    parties = []
    for group in match.groups():
        cleaned = " ".join(group.split())[:120].strip(" ,.")
        if cleaned:
            parties.append(cleaned)
    return parties
```

with:

```python
# Rough contract-type detection from the opening text.
_TYPE_PATTERNS = [
    (r"non-?disclosure agreement|\bnda\b", "Non-Disclosure Agreement"),
    (r"master services agreement|\bmsa\b", "Master Services Agreement"),
    (r"employment agreement", "Employment Agreement"),
    (r"license agreement", "License Agreement"),
    (r"lease agreement", "Lease Agreement"),
    (r"purchase agreement", "Purchase Agreement"),
    (r"consulting agreement", "Consulting Agreement"),
    (r"supply agreement", "Supply Agreement"),
    (r"reseller agreement", "Reseller Agreement"),
    (r"franchise agreement", "Franchise Agreement"),
    (r"joint venture agreement", "Joint Venture Agreement"),
    (r"service agreement", "Service Agreement"),
]

# Two forms of party preamble:
#   - "by and between/among X, Y, and Z" (N parties, named group `n_party`)
#   - "between X and Y" (exactly 2 parties, named groups `party_a`/`party_b`)
# The first alternative is tried first so "by and between ..." text (which
# also contains the literal substring "between") is captured by the N-party
# path rather than falling through to the 2-party path.
_PARTIES_RE = re.compile(
    r"by and (?:between|among)\s+(?P<n_party>.+?)(?:\s*[\.\(]|\bdated\b|$)"
    r"|between\s+(?P<party_a>.+?)\s+and\s+(?P<party_b>.+?)(?:\s*[\.,\(]|\bdated\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _detect_type(text: str) -> str:
    head = text[:2000].lower()
    for pattern, label in _TYPE_PATTERNS:
        if re.search(pattern, head):
            return label
    return "Unknown"


def _split_party_list(span: str) -> list[str]:
    """Split a preamble party-list span like "A, B, and C" into parties."""
    span = span.strip(" ,.")
    # Normalize the final ", and " / " and " into a plain comma so a
    # straight split(",") below yields every party, including 2-party spans
    # with no comma at all ("A and B" -> "A, B").
    span = re.sub(r"\s*,?\s+and\s+(?=[^,]*$)", ", ", span, count=1)
    parts = [p.strip(" ,.") for p in span.split(",")]
    return [" ".join(p.split())[:120] for p in parts if p.strip()]


def _detect_parties(text: str) -> list[str]:
    match = _PARTIES_RE.search(text[:2000])
    if not match:
        return []

    n_party_span = match.group("n_party")
    if n_party_span is not None:
        return _split_party_list(n_party_span)

    parties = []
    for group in (match.group("party_a"), match.group("party_b")):
        cleaned = " ".join(group.split())[:120].strip(" ,.")
        if cleaned:
            parties.append(cleaned)
    return parties
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass, including the existing `test_pipeline_ingest_sample_contract` in `tests/test_classifier.py` (still detects `"Non-Disclosure Agreement"` — unaffected since that pattern is unchanged and first in the list).

---

### Task 3: Real PDF/DOCX parser regression tests

**Files:**
- Modify: `tests/test_parsers.py`

**Interfaces:**
- Consumes: `app.ingestion.parsers.parse_pdf`, `app.ingestion.parsers.parse_docx` (existing, unchanged).
- Produces: nothing new; this task only adds test coverage.

- [ ] **Step 1: Write the tests**

Append to `tests/test_parsers.py` (add these imports to the top of the file, alongside the existing `import pytest` and `from app.ingestion.parsers import ...` line — extend the existing import to also include `parse_pdf` and `parse_docx`):

Change the existing import line:
```python
from app.ingestion.parsers import (
    UnsupportedFormatError,
    detect_format,
    parse,
    parse_txt,
)
```
to:
```python
from app.ingestion.parsers import (
    UnsupportedFormatError,
    detect_format,
    parse,
    parse_docx,
    parse_pdf,
    parse_txt,
)
```

Then append this to the end of the file:

```python
def _build_minimal_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing `text`, with a correct
    byte-exact xref table (offsets are computed from the actual bytes
    written, not hand-counted) so pypdf can parse it without any extra
    PDF-generation dependency."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 400 200] /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1")
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )

    header = b"%PDF-1.4\n"
    offsets = [0]
    body = bytearray(header)
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_offset = len(body)
    n = len(objects) + 1
    xref = f"xref\n0 {n}\n0000000000 65535 f \n".encode()
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()

    return bytes(body) + xref + trailer


def test_parse_pdf_extracts_text():
    pdf_bytes = _build_minimal_pdf("Sample Confidentiality Clause Text")
    text = parse_pdf(pdf_bytes)
    assert "Sample Confidentiality Clause Text" in text


def test_parse_docx_extracts_text():
    import io

    from docx import Document

    document = Document()
    document.add_paragraph("MUTUAL NON-DISCLOSURE AGREEMENT")
    document.add_paragraph("1. Confidentiality")
    document.add_paragraph("Each party agrees to hold in strict confidence all proprietary information.")
    buffer = io.BytesIO()
    document.save(buffer)

    text = parse_docx(buffer.getvalue())
    assert "MUTUAL NON-DISCLOSURE AGREEMENT" in text
    assert "Each party agrees to hold in strict confidence all proprietary information." in text


def test_parse_dispatches_pdf():
    pdf_bytes = _build_minimal_pdf("Dispatch check text")
    fmt, text = parse("contract.pdf", pdf_bytes)
    assert fmt == "pdf"
    assert "Dispatch check text" in text


def test_parse_dispatches_docx():
    import io

    from docx import Document

    document = Document()
    document.add_paragraph("Dispatch check text")
    buffer = io.BytesIO()
    document.save(buffer)

    fmt, text = parse("contract.docx", buffer.getvalue())
    assert fmt == "docx"
    assert "Dispatch check text" in text
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: PASS (8 passed — 4 existing + 4 new). If `parse_pdf`/`parse_docx` raise `RuntimeError` about missing `pypdf`/`python-docx`, run `pip install -r requirements.txt` first (both are already listed there).

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass.

---

### Task 4: Real LegalBERT backend (embedding + cosine similarity)

**Files:**
- Modify: `requirements.txt`
- Modify: `app/clauses/classifier.py:56-101` (the `LegalBertClassifier` class and `get_classifier()`)
- Modify: `tests/test_classifier.py` (add new tests)

**Interfaces:**
- Consumes: `app.clauses.categories.CATEGORY_KEYWORDS`, `app.clauses.categories.UNCLASSIFIED` (existing), `app.config.settings.legalbert_model` (existing).
- Produces: `LegalBertClassifier(model_name: str | None = None, embed_fn: Callable[[str], "torch.Tensor"] | None = None)` with `.classify(text: str) -> tuple[str, float]` — same signature as before. `embed_fn` is a new, optional, purely-for-testing constructor parameter; production code never passes it (real embeddings are loaded lazily on first `classify()` call, same lazy-load timing as before).

**Design note (why this differs from the original stub):** `nlpaueb/legal-bert-base-uncased` is tagged `fill-mask` on the HuggingFace Hub — it has no classification/NLI head, so the original `pipeline("zero-shot-classification", model=...)` approach would silently attach a randomly-initialized head and produce meaningless scores. This task replaces it with LegalBERT used purely as an encoder: embed each category's keyword description and each clause via mean-pooled token embeddings, then classify by cosine similarity (highest-similarity category wins).

- [ ] **Step 1: Write the failing tests**

In `tests/test_classifier.py`, add these imports to the top of the file (extend the existing `from app.clauses.classifier import RuleBasedClassifier, classify_clauses` line):

```python
from app.clauses.classifier import LegalBertClassifier, RuleBasedClassifier, classify_clauses
```

Then append to the end of the file:

```python
import torch

# A small deterministic stand-in for a real embedding model: each category's
# keyword description contains its own vocabulary token, so cosine similarity
# against a clause mentioning that token should be highest for that category.
_VOCAB = [
    "confidential",
    "terminat",
    "liab",
    "indemni",
    "intellectual",
    "govern",
    "payment",
    "warrant",
    "assign",
    "force majeure",
]


def _fake_embed_fn(text: str) -> torch.Tensor:
    lowered = text.lower()
    vec = [1.0 if token in lowered else 0.0 for token in _VOCAB]
    if not any(vec):
        vec[-1] = 0.001  # avoid an all-zero vector, which has no norm
    return torch.tensor(vec)


def test_legalbert_classifier_picks_highest_similarity_category():
    classifier = LegalBertClassifier(embed_fn=_fake_embed_fn)

    category, confidence = classifier.classify(
        "All Confidential Information must remain confidential and shall not be disclosed."
    )

    assert category == "Confidentiality"
    assert confidence == pytest.approx(1.0, abs=1e-6)


def test_legalbert_classifier_empty_text_is_unclassified():
    classifier = LegalBertClassifier(embed_fn=_fake_embed_fn)
    category, confidence = classifier.classify("   ")
    assert category == "Unclassified"
    assert confidence == 0.0


def test_legalbert_classifier_falls_back_when_model_unavailable():
    # No embed_fn injected, and no real model load is attempted here because
    # we monkeypatch the internal loader to simulate transformers/torch (or
    # the model download) being unavailable.
    classifier = LegalBertClassifier()
    classifier._ensure_embed_fn = lambda: False  # simulate load failure

    category, confidence = classifier.classify(
        "Each party shall hold all Confidential Information in strict confidence."
    )

    assert category == "Confidentiality"  # matches RuleBasedClassifier's own behavior
    assert 0.0 < confidence <= 1.0
```

Also add `import pytest` to the top of `tests/test_classifier.py` if it isn't already imported (check first — currently it is not).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: FAIL/ERROR — `LegalBertClassifier` doesn't exist as an importable name change yet in this exact form (import error or `TypeError: unexpected keyword argument 'embed_fn'`), since the current class only accepts `model_name`.

- [ ] **Step 3: Uncomment transformers/torch in requirements.txt**

Replace this block in `requirements.txt`:

```
# Optional: LegalBERT clause classification backend.
# These are only imported lazily in app/clauses/classifier.py; the rule-based
# backend works without them so the service runs out of the box.
# transformers>=4.40
# torch>=2.2
```

with:

```
# LegalBERT clause classification backend (embedding + cosine similarity).
# Imported lazily in app/clauses/classifier.py; the rule-based backend still
# works without these installed, so a lighter-weight deployment can omit them.
transformers>=4.40
torch>=2.2
```

Run: `pip install -r requirements.txt`
Expected: installs successfully (this may take a few minutes and ~1-2GB of disk for torch).

- [ ] **Step 4: Rewrite `LegalBertClassifier` and `get_classifier()` in app/clauses/classifier.py**

Replace this block (current lines 56-101):

```python
class LegalBertClassifier:
    """Optional LegalBERT-backed classifier (HuggingFace Transformers).

    Falls back to the rule-based classifier if transformers/torch or the model
    are unavailable, so the pipeline never hard-fails on a missing optional dep.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.legalbert_model
        self._pipeline = None
        self._fallback = RuleBasedClassifier()
        self._labels = list(CATEGORY_KEYWORDS.keys())

    def _ensure_pipeline(self) -> bool:
        if self._pipeline is not None:
            return True
        try:  # pragma: no cover - depends on optional heavy deps
            from transformers import pipeline

            self._pipeline = pipeline(
                "zero-shot-classification", model=self.model_name
            )
            return True
        except Exception:
            # transformers/torch/model not available — use the rule baseline.
            self._pipeline = None
            return False

    def classify(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return UNCLASSIFIED, 0.0
        if not self._ensure_pipeline():
            return self._fallback.classify(text)
        try:  # pragma: no cover - depends on optional heavy deps
            result = self._pipeline(text, candidate_labels=self._labels)
            return result["labels"][0], round(float(result["scores"][0]), 3)
        except Exception:
            return self._fallback.classify(text)


def get_classifier():
    """Return the configured classifier instance."""
    if settings.classifier_backend == "legalbert":
        return LegalBertClassifier()
    return RuleBasedClassifier()
```

with:

```python
def _mean_pool(last_hidden_state, attention_mask):
    """Attention-mask-weighted mean of token embeddings -> one vector per row.

    `last_hidden_state`: (batch, seq_len, hidden) tensor.
    `attention_mask`: (batch, seq_len) tensor of 0/1.
    Returns a (batch, hidden) tensor. Deliberately takes plain tensors (no
    `torch` import needed here) so this module still only imports transformers
    /torch lazily inside LegalBertClassifier.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _cosine_similarity(a, b) -> float:
    a_norm = a / a.norm()
    b_norm = b / b.norm()
    return float((a_norm * b_norm).sum())


class LegalBertClassifier:
    """LegalBERT-backed classifier using embedding + cosine similarity.

    `nlpaueb/legal-bert-base-uncased` is a base encoder with no
    classification/NLI head, so this does not use a zero-shot-classification
    pipeline (which would require one). Instead, each category's keyword
    description and each clause are embedded via mean-pooled LegalBERT token
    embeddings, and the category with highest cosine similarity to the
    clause wins. Falls back to the rule-based classifier if transformers/
    torch or the model are unavailable, so the pipeline never hard-fails on
    a missing optional dependency.
    """

    def __init__(self, model_name: str | None = None, embed_fn=None) -> None:
        self.model_name = model_name or settings.legalbert_model
        self._embed_fn = embed_fn
        self._fallback = RuleBasedClassifier()
        self._category_embeddings = None

    def _ensure_embed_fn(self) -> bool:
        if self._embed_fn is not None:
            return True
        try:  # pragma: no cover - depends on optional heavy deps
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            return False

        try:  # pragma: no cover - depends on optional heavy deps
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.eval()
        except Exception:
            return False

        def embed_fn(text: str):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])[0]

        self._embed_fn = embed_fn
        return True

    def classify(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return UNCLASSIFIED, 0.0
        if not self._ensure_embed_fn():
            return self._fallback.classify(text)
        try:
            if self._category_embeddings is None:
                self._category_embeddings = {
                    category: self._embed_fn(f"{category}: {', '.join(keywords)}")
                    for category, keywords in CATEGORY_KEYWORDS.items()
                }
            clause_embedding = self._embed_fn(text)
            best_category, best_score = UNCLASSIFIED, -1.0
            for category, ref_embedding in self._category_embeddings.items():
                score = _cosine_similarity(clause_embedding, ref_embedding)
                if score > best_score:
                    best_category, best_score = category, score
            confidence = max(0.0, min(1.0, best_score))
            return best_category, round(confidence, 3)
        except Exception:
            return self._fallback.classify(text)


def get_classifier():
    """Return the configured classifier instance."""
    if settings.classifier_backend == "legalbert":
        return LegalBertClassifier()
    return RuleBasedClassifier()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: PASS (all existing + 3 new tests). These tests never trigger a real model download since they inject `embed_fn` or monkeypatch `_ensure_embed_fn`.

- [ ] **Step 6: Manually verify the real embedding path actually works (not just the injected-fake-embed_fn tests)**

Run:
```bash
python3 -c "
from app.clauses.classifier import LegalBertClassifier
clf = LegalBertClassifier()
print(clf.classify('All Confidential Information must remain confidential and shall not be disclosed.'))
print(clf.classify('This Agreement shall be governed by the laws of Delaware.'))
"
```
Expected: first call downloads `nlpaueb/legal-bert-base-uncased` (~440MB, one-time, cached under `~/.cache/huggingface` afterward) and prints two `(category, confidence)` tuples — confirms the real encoder + cosine-similarity path loads and runs end to end, not just the fallback. If it prints `('Unclassified', 0.0)` or falls back to rule-based-looking output for both calls, something is wrong with the load path — investigate before moving on (do not proceed to Task 6's real-backend run with a silently-broken loader).

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass.

---

### Task 5: CUAD sample fixture generation

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/__init__.py`
- Create: `scripts/generate_cuad_sample.py`
- Create: `data/cuad_sample.json` (generated by running the script, then committed as a fixture — not hand-written)

**Interfaces:**
- Consumes: nothing from the rest of the app.
- Produces: `data/cuad_sample.json`, a JSON array of `{"category": str, "cuad_label": str, "contract_file": str, "clause_text": str}` records, consumed by Task 6's `scripts/evaluate_clauses.py`.

- [ ] **Step 1: Add the `datasets` library to requirements.txt**

Append to the end of `requirements.txt`:

```

# One-time CUAD fixture generation (scripts/generate_cuad_sample.py only;
# not needed to run the app or to score against the already-committed
# data/cuad_sample.json fixture).
datasets>=2.19
```

Run: `pip install -r requirements.txt`
Expected: installs successfully.

- [ ] **Step 2: Create the scripts package**

Create `scripts/__init__.py` (empty file).

- [ ] **Step 3: Write scripts/generate_cuad_sample.py**

```python
"""One-time generator for data/cuad_sample.json.

Downloads labeled clauses from `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`
(a parquet mirror of the original CUAD clause-classification data, CC-BY-4.0,
publicly readable, no auth required) and writes a stratified sample mapped
onto our 10-category taxonomy (app/clauses/categories.py) to
data/cuad_sample.json. That file is committed to the repo so
scripts/evaluate_clauses.py can score classifiers against real CUAD
annotations without needing network access or the `datasets` package on
every run.

CUAD's original 41 categories do not include "Confidentiality",
"Indemnification", or "Force Majeure" -- those 3 of our 10 taxonomy
categories have no CUAD ground truth and are intentionally excluded here.

Usage (run once; re-run only if you want to refresh the sample):
    python -m scripts.generate_cuad_sample
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"
SAMPLES_PER_CATEGORY = 50

# Maps our 10-category taxonomy (app/clauses/categories.py) onto CUAD's
# original 41 clause labels. Confidentiality, Indemnification, and Force
# Majeure have no CUAD equivalent and are intentionally omitted -- there is
# no entry for them here.
CATEGORY_LABEL_MAP: dict[str, list[str]] = {
    "Termination": ["Termination For Convenience"],
    "Liability": ["Cap On Liability", "Uncapped Liability"],
    "Intellectual Property": [
        "License Grant",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "Non-Transferable License",
        "Irrevocable Or Perpetual License",
        "Unlimited/All-You-Can-Eat-License",
        "Affiliate License-Licensee",
        "Affiliate License-Licensor",
    ],
    "Governing Law": ["Governing Law"],
    "Payment Terms": ["Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions"],
    "Warranty": ["Warranty Duration"],
    "Assignment": ["Anti-Assignment"],
}


def generate() -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
        split="train",
    ).shuffle(seed=42)

    label_to_category = {
        label: category
        for category, labels in CATEGORY_LABEL_MAP.items()
        for label in labels
    }

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        category = label_to_category.get(row["label"])
        if category is None or len(by_category[category]) >= SAMPLES_PER_CATEGORY:
            continue
        clause_text = row["clause"].strip()
        if clause_text:
            by_category[category].append(
                {
                    "category": category,
                    "cuad_label": row["label"],
                    "contract_file": row["file_name"],
                    "clause_text": clause_text,
                }
            )

    records = [record for records in by_category.values() for record in records]
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(records, indent=2))

    print(f"Wrote {len(records)} records to {SAMPLE_PATH}")
    for category in CATEGORY_LABEL_MAP:
        print(f"  {category}: {len(by_category.get(category, []))}")


if __name__ == "__main__":
    generate()
```

- [ ] **Step 4: Run the generator to produce the committed fixture**

Run: `python -m scripts.generate_cuad_sample`
Expected: downloads the dataset (small, parquet, a few MB) and prints a per-category count summary, e.g.:
```
Wrote 341 records to .../data/cuad_sample.json
  Termination: 50
  Liability: 50
  Intellectual Property: 50
  Governing Law: 50
  Payment Terms: 50
  Warranty: 50
  Assignment: 41
```
(Exact per-category counts depend on how many examples exist in the raw data for each mapped label combination — some categories may have fewer than 50 if the underlying CUAD label has fewer than 50 total occurrences; that's expected, not an error.)

Verify: `python3 -c "import json; d = json.load(open('data/cuad_sample.json')); print(len(d)); print(d[0])"` prints a record count and a sample record with all 4 expected keys.

---

### Task 6: CUAD evaluation harness

**Files:**
- Create: `scripts/evaluate_clauses.py`
- Create: `tests/test_evaluate_clauses.py`

**Interfaces:**
- Consumes: `data/cuad_sample.json` (Task 5), `app.clauses.classifier.RuleBasedClassifier`, `app.clauses.classifier.LegalBertClassifier` (Task 4).
- Produces: `score(classifier, records: Iterable[dict]) -> dict[str, dict[str, float]]` (importable, used by the smoke test) and a `python -m scripts.evaluate_clauses` CLI entrypoint.

- [ ] **Step 1: Write scripts/evaluate_clauses.py**

```python
"""Score clause classifiers against the committed CUAD-derived sample.

Run: python -m scripts.evaluate_clauses [--backend rule|legalbert|both]

Loads data/cuad_sample.json (see scripts/generate_cuad_sample.py for how it
was produced) and reports per-category + macro-averaged precision, recall,
and F1 for the selected classifier backend(s).

Confidentiality, Indemnification, and Force Majeure are not part of CUAD's
original 41 categories, so they are absent from the sample and are not
scored here -- this is a known, stated limitation of evaluating against
CUAD, not a bug.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from app.clauses.classifier import LegalBertClassifier, RuleBasedClassifier

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def score(classifier, records: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Return {category: {precision, recall, f1, support}} plus a "macro_avg" row."""
    true_positives: dict[str, int] = defaultdict(int)
    false_positives: dict[str, int] = defaultdict(int)
    false_negatives: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for record in records:
        expected = record["category"]
        predicted, _confidence = classifier.classify(record["clause_text"])
        support[expected] += 1
        if predicted == expected:
            true_positives[expected] += 1
        else:
            false_negatives[expected] += 1
            false_positives[predicted] += 1

    categories = sorted(support)
    results: dict[str, dict[str, float]] = {}
    for category in categories:
        tp = true_positives[category]
        fp = false_positives[category]
        fn = false_negatives[category]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results[category] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support[category],
        }

    macro_precision = sum(r["precision"] for r in results.values()) / len(results)
    macro_recall = sum(r["recall"] for r in results.values()) / len(results)
    macro_f1 = sum(r["f1"] for r in results.values()) / len(results)
    results["macro_avg"] = {
        "precision": round(macro_precision, 3),
        "recall": round(macro_recall, 3),
        "f1": round(macro_f1, 3),
        "support": sum(support.values()),
    }
    return results


def print_report(name: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'category':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for category, metrics in results.items():
        print(
            f"{category:<24}{metrics['precision']:>10.3f}{metrics['recall']:>10.3f}"
            f"{metrics['f1']:>10.3f}{metrics['support']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["rule", "legalbert", "both"], default="both")
    args = parser.parse_args()

    records = load_sample()

    if args.backend in ("rule", "both"):
        print_report("RuleBasedClassifier", score(RuleBasedClassifier(), records))
    if args.backend in ("legalbert", "both"):
        print_report("LegalBertClassifier", score(LegalBertClassifier(), records))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

Create `tests/test_evaluate_clauses.py`:

```python
"""Smoke test for the CUAD evaluation harness's scoring logic.

Uses a tiny inline fixture (not the full committed data/cuad_sample.json)
and the fast RuleBasedClassifier, so this test runs quickly and offline —
it never triggers a LegalBERT download.
"""

from app.clauses.classifier import RuleBasedClassifier
from scripts.evaluate_clauses import score

_FIXTURE = [
    {"category": "Governing Law", "clause_text": "This Agreement shall be governed by the laws of Delaware."},
    {"category": "Governing Law", "clause_text": "This Agreement is governed by the laws of the State of New York."},
    {
        "category": "Assignment",
        "clause_text": "Neither party may assign this Agreement without the other's consent; binding on successors and assigns.",
    },
]


def test_score_reports_per_category_metrics_and_macro_avg():
    results = score(RuleBasedClassifier(), _FIXTURE)

    assert results["Governing Law"]["support"] == 2
    assert results["Governing Law"]["precision"] == 1.0
    assert results["Governing Law"]["recall"] == 1.0
    assert results["Assignment"]["support"] == 1
    assert "macro_avg" in results
    assert results["macro_avg"]["support"] == 3
    assert 0.0 <= results["macro_avg"]["f1"] <= 1.0


def test_score_counts_misclassification_as_false_positive_and_negative():
    fixture = [{"category": "Governing Law", "clause_text": "The quick brown fox jumps over the lazy dog."}]
    results = score(RuleBasedClassifier(), fixture)

    # No keyword match -> classifier predicts "Unclassified", so the true
    # category gets 0 recall and there is no row for "Unclassified" itself
    # (it was never a ground-truth label in this fixture).
    assert results["Governing Law"]["recall"] == 0.0
    assert "Unclassified" not in results
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_clauses.py -v`
Expected: PASS (2 passed)

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: All tests pass (12 original + 4 store + 5 pipeline + 4 parser + 3 classifier + 2 eval = 30 total; exact count may differ slightly if any earlier task's count assumptions shifted, but there should be zero failures).

- [ ] **Step 5: Run the real evaluation report**

Run: `python -m scripts.evaluate_clauses`
Expected: prints two tables (`RuleBasedClassifier` and `LegalBertClassifier`), each with rows for Termination, Liability, Intellectual Property, Governing Law, Payment Terms, Warranty, Assignment, and a `macro_avg` row. The LegalBERT run may take a minute or two (CPU inference over ~340 sample clauses plus 7 category descriptions). Save this output — it's the primary new "Factual" evidence artifact for the CP2 report.

---

### Task 7: Full manual end-to-end verification

This task has no new code — it's a checklist confirming the whole system works together after Tasks 1-6, including things automated tests can't easily cover (a real server process, real file uploads, persistence across a restart).

- [ ] **Step 1: Run the full automated test suite one more time**

Run: `python -m pytest -q`
Expected: all tests pass, zero failures/errors.

- [ ] **Step 2: Start the server and upload a real contract**

Run: `rm -f contractlens.db && uvicorn app.main:app --port 8000 &` (background), then:
```bash
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@data/sample_contract.txt" | python3 -m json.tool
```
Expected: JSON response with `id`, `metadata.contract_type == "Non-Disclosure Agreement"`, `metadata.parties == ["Acme Corporation", "Globex LLC"]`, and a non-empty `categories`/`clauses` list. Note the returned `id`.

- [ ] **Step 3: Confirm the clause-visualization view renders**

Run: `curl -s http://127.0.0.1:8000/contracts/<id>/view` (substitute the real id from Step 2).
Expected: HTML response containing `<h1>sample_contract.txt</h1>` and clause blocks grouped by category.

- [ ] **Step 4: Confirm persistence survives a restart**

Stop the server (`kill %1` or the equivalent for however it was started), confirm `contractlens.db` now exists in the repo root (`ls -la contractlens.db`), then restart it (`uvicorn app.main:app --port 8000 &`) and re-fetch the same contract:
```bash
curl -s http://127.0.0.1:8000/contracts/<id> | python3 -m json.tool
```
Expected: the same contract data comes back even though the server process restarted — proof the SQLite store actually persists (this is the key behavioral difference from the old in-memory store). Stop the server afterward (`kill %1`).

- [ ] **Step 5: Confirm the final gitignore/file state is clean**

Run: `git status`
Expected: `contractlens.db` does not appear in the untracked-files list (it's gitignored per Task 1, Step 5); `data/cuad_sample.json`, `scripts/`, and all modified `app/`/`tests/` files do appear as new/modified, ready for the user to review and commit themselves.

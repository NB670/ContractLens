# ContractLens

**A Privacy-Preserving Contract Intelligence Platform**

ContractLens transforms unstructured contracts (PDF / DOCX / TXT) into structured
representations that support clause classification, semantic retrieval, contract
comparison, risk analysis, and question answering. The platform is designed to run
locally so that privacy-sensitive contracts never have to leave the user's machine.

## What's implemented

- A **FastAPI** service exposing `/upload`, `/contracts/{id}`, and
  `/contracts/{id}/view` endpoints for uploading and reviewing contracts.
- A **document parsing layer** that extracts raw text from PDF, DOCX, and TXT
  contracts (`app/ingestion/parsers.py`).
- A **clause segmenter** that splits a contract into individual clauses/sections
  using heading and numbering heuristics (`app/ingestion/segmenter.py`).
- A **clause classifier** that labels each clause with one of 10 CUAD-style
  categories (confidentiality, termination, liability, indemnification,
  intellectual property, governing law, payment terms, warranty, assignment,
  force majeure). Two backends are available (`app/clauses/classifier.py`):
  - `RuleBasedClassifier` (default) — transparent keyword scoring, no model
    download required.
  - `LegalBertClassifier` — embeds each clause and each category's keyword
    description with LegalBERT (HuggingFace Transformers) and classifies by
    cosine similarity.
- **Metadata extraction** that detects contract type (12 common types) and
  party names, including multi-party ("by and among A, B, and C") preambles
  (`app/pipeline.py`).
- A **structured contract model** capturing contract type, parties, key
  sections, identified clauses, and metadata (`app/models/contract.py`).
- A **persistent SQLite-backed contract store**, so uploaded contracts survive
  a server restart (`app/store.py`).
- A **clause-visualization** view (`/contracts/{id}/view`) that lists the
  identified categories and shows each clause's text and category tag.
- A **CUAD-based evaluation harness** that scores both classifier backends
  against real labeled CUAD data with precision/recall/F1
  (`scripts/evaluate_clauses.py`, `data/cuad_sample.json`).

## Architecture

```
contract file ──▶ parsers ──▶ raw text
                                  │
                                  ▼
                            segmenter (clauses)
                                  │
                                  ▼
                        classifier (CUAD categories)
                                  │
                                  ▼
                     structured Contract model
                                  │
                                  ▼
              SQLite-backed store ──▶ FastAPI JSON / HTML view
```

## Project layout

```
app/
  main.py                  FastAPI app + routes (/upload, /contracts/{id}, /contracts/{id}/view)
  config.py                 runtime configuration (env vars below)
  pipeline.py                parse -> segment -> classify -> store orchestration;
                              contract-type and party detection
  store.py                   SQLite-backed contract store (SQLModel)
  ingestion/
    parsers.py                PDF / DOCX / TXT text extraction
    segmenter.py               split text into clauses/sections
  clauses/
    categories.py              canonical CUAD-style clause categories
    classifier.py               rule-based + LegalBERT (embedding + cosine similarity) classifiers
  models/
    contract.py                 Contract / Clause data models
scripts/
  generate_cuad_sample.py    one-time generator for data/cuad_sample.json
  evaluate_clauses.py         scores both classifiers against data/cuad_sample.json
data/
  sample_contract.txt        tiny sample for local smoke testing
  cuad_sample.json            350 labeled clauses sampled from CUAD (7 categories)
tests/
  test_segmenter.py
  test_parsers.py
  test_classifier.py
  test_pipeline.py
  test_store.py
  test_evaluate_clauses.py
requirements.txt
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive API. Upload a contract
via `POST /upload` (try `data/sample_contract.txt`), note the returned `id`, then:

- `GET /contracts/{id}` — structured JSON (contract type, parties, clauses, categories)
- `GET /contracts/{id}/view` — HTML clause-visualization page

Uploaded contracts persist in `contractlens.db` (SQLite) across server restarts.

### Configuration

All settings are optional environment variables (see `app/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `CONTRACTLENS_MAX_UPLOAD` | `26214400` (25 MB) | max upload size in bytes |
| `CONTRACTLENS_CLASSIFIER` | `rule` | classifier backend: `rule` or `legalbert` |
| `CONTRACTLENS_LEGALBERT_MODEL` | `nlpaueb/legal-bert-base-uncased` | HF model id for the LegalBERT backend |
| `CONTRACTLENS_DB_PATH` | `contractlens.db` | SQLite file path |

## Running the test suite

```bash
python -m pytest -q
```

## Evaluating clause classification against CUAD

`data/cuad_sample.json` is a committed, offline-reproducible sample of 350 real
labeled clauses (from `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`,
CC-BY-4.0), covering 7 of the 10 taxonomy categories — CUAD's original 41 categories
don't include Confidentiality, Indemnification, or Force Majeure, so those 3 are
excluded from this evaluation.

```bash
python -m scripts.evaluate_clauses                  # both backends
python -m scripts.evaluate_clauses --backend rule
python -m scripts.evaluate_clauses --backend legalbert
```

Prints per-category and macro-averaged precision/recall/F1 for each backend. To
regenerate the sample from the source dataset (requires the `datasets` package
and network access):

```bash
python -m scripts.generate_cuad_sample
```

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

## Datasets

- **CUAD** (Contract Understanding Atticus Dataset) — clause categories and
  evaluation for classification. The canonical category list in
  `app/clauses/categories.py` follows the CUAD taxonomy; `data/cuad_sample.json`
  is a real labeled sample used by the evaluation harness above.
- **ACORD** — planned for clause-retrieval evaluation.

## Roadmap

- Semantic retrieval and contract comparison.
- Risk reporting backed by supporting evidence.
- ContractLens UI and local-LLM chatbot integration, with benchmark evaluation
  (Recall@K, MRR) for retrieval.

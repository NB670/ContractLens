# ContractLens

**A Privacy-Preserving Contract Intelligence Platform**

ContractLens transforms unstructured contracts (PDF / DOCX / TXT) into structured
representations that support clause classification, semantic retrieval, contract
comparison, risk analysis, and question answering. The platform is designed to run
locally so that privacy-sensitive contracts never have to leave the user's machine.

This repository currently contains the **Checkpoint 2** deliverable: the contract
ingestion and clause-intelligence pipeline.

## Checkpoint 2 scope (this deliverable)

Per the project milestone chart, Checkpoint 2 covers:

> Contract ingestion and clause intelligence — Implement the contract upload and
> parsing pipeline, structured extraction, and clause visualization. Users should
> be able to upload and view identified categories.

What is implemented here:

- A **FastAPI** service exposing a `/upload` endpoint that accepts PDF, DOCX, or
  TXT contracts.
- A **document parsing layer** that extracts raw text from each supported format
  (`app/ingestion/parsers.py`).
- A **clause segmenter** that splits a contract into individual clauses/sections
  using heading and numbering heuristics (`app/ingestion/segmenter.py`).
- A **clause classifier** that labels each clause with one of the key CUAD-style
  categories (confidentiality, termination, liability, indemnification,
  intellectual property, ...). It ships with a transparent keyword/rule-based
  backend and an optional LegalBERT (HuggingFace Transformers) backend that is
  loaded only if `transformers` and a model are available
  (`app/clauses/classifier.py`).
- A **structured contract model** capturing contract type, parties, key sections,
  identified clauses, and metadata (`app/models/contract.py`).
- A minimal **clause-visualization** view (`/contracts/{id}`) that lists the
  identified categories and links each one back to its source text.

> NOTE: This is autonomously generated starter code matching the team's own
> Checkpoint 2 plan. It is a head start, not a benchmark or a ceiling.

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
                     structured Contract model ──▶ FastAPI JSON / HTML view
```

## Project layout

```
app/
  main.py                 FastAPI app + routes (/upload, /contracts/{id})
  config.py               runtime configuration
  ingestion/
    parsers.py            PDF / DOCX / TXT text extraction
    segmenter.py          split text into clauses/sections
  clauses/
    categories.py         canonical CUAD-style clause categories
    classifier.py         rule-based + optional LegalBERT classifier
  models/
    contract.py           Contract / Clause data models
  store.py                in-memory contract store (CP2 placeholder)
data/
  sample_contract.txt     tiny sample for local smoke testing
tests/
  test_segmenter.py
  test_classifier.py
  test_parsers.py
requirements.txt
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive API, upload a contract
via `POST /upload`, and view the structured breakdown at `/contracts/{id}`.

## Datasets

- **CUAD** (Contract Understanding Atticus Dataset) — clause categories and
  evaluation for classification. The canonical category list in
  `app/clauses/categories.py` follows the CUAD taxonomy.
- **ACORD** — used in later checkpoints for clause-retrieval evaluation.

## Roadmap (subsequent checkpoints)

- **CP3:** semantic retrieval, contract comparison, and risk reporting.
- **CP4:** ContractLens UI, local-LLM chatbot integration, and benchmark
  evaluation (precision / recall / F1, Recall@K, MRR).

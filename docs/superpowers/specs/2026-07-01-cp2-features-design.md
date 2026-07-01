# Checkpoint 2 feature completion + stretch goals — design

Date: 2026-07-01
Status: Approved

## Context

CP2's milestone requirement (per the CS6365 project plan) is: "Implement the
contract upload and parsing pipeline, structured extraction, and clause
visualization. Users should be able to upload and view identified categories."

The repo currently has a working baseline (merged from the professor's
`ai-suggestions/cp2` branch, to be reworked/replaced with original engineering
per the CP1 feedback instructions): FastAPI upload/view routes, PDF/DOCX/TXT
parsing, a heuristic clause segmenter, a rule-based CUAD-category classifier
with a dormant LegalBERT fallback stub, an in-memory contract store, and 12
passing unit tests.

CP1 feedback (Satvik Agrawal) specifically dinged the **Match** and
**Factual** sections for having no verifiable engineering trace, and named
one concrete, actionable gap: *"Establish baseline evaluation metrics using
the CUAD dataset early in development."* The user wants CP2 to close known
gaps and go beyond the plan for full marks, while staying inside CP2's own
lane (ingestion + clause intelligence) rather than encroaching on CP3
(retrieval/comparison/risk analysis).

Gaps identified by manual review + test execution before this design:
1. No real CUAD dataset used anywhere — classifier only borrows category names.
2. `LegalBertClassifier` is implemented but dormant (`transformers`/`torch`
   commented out in `requirements.txt`), never exercised.
3. PDF/DOCX parsing has zero test coverage (only TXT path is tested).
4. Metadata extraction (`_detect_type`, `_detect_parties` in `app/pipeline.py`)
   only handles two-party "between X and Y" contracts and a fixed list of 7
   contract types.
5. Contract store is in-memory only — data lost on process restart.

## Goals

- Wire up the real LegalBERT backend so it actually runs (not a dormant stub).
- Build a CUAD-based evaluation harness producing real precision/recall/F1
  numbers for both classifier backends — directly answering the TA's named
  suggestion and giving the CP2 report a runnable "execution is truth" artifact.
- Add persistent storage (SQLite via SQLModel) so contracts survive restarts.
- Improve metadata extraction (multi-party detection, more contract types).
- Add regression test coverage for PDF/DOCX parsing and the new store/eval code.

## Non-goals

- No semantic retrieval, contract comparison, or risk scoring (CP3 territory).
- No fine-tuning of LegalBERT on CUAD (stretch goal for later checkpoints;
  this build only wires up zero-shot classification with the pretrained model).
- No schema migration tooling (Alembic, etc.) — one table, no migrations needed yet.
- No new frontend/upload form — FastAPI's existing `/docs` Swagger UI already
  lets a user upload a file interactively, which satisfies the CP2 milestone's
  "users should be able to upload" wording without new UI surface.

## Architecture overview

```
contract file ──▶ parsers ──▶ raw text
                                  │
                                  ▼
                            segmenter (clauses)
                                  │
                                  ▼
                 classifier (rule-based | legalbert, CUAD categories)
                                  │
                                  ▼
                     structured Contract model
                                  │
                                  ▼
              SQLModel-backed SQLite store (persists across restarts)
                                  │
                                  ▼
                     FastAPI JSON / HTML view

scripts/evaluate_clauses.py (offline, dev-facing):
  data/cuad_sample.json (committed fixture)
        │
        ▼
  score RuleBasedClassifier + LegalBertClassifier against labeled spans
        │
        ▼
  per-category + overall precision/recall/F1 report (stdout)
```

## Components

### 1. Persistent storage (`app/store.py`)

- Add `sqlmodel` to `requirements.txt`.
- Replace the in-memory dict in `ContractStore` with a SQLModel table
  `ContractRecord` holding `id`, `filename`, `source_format`, `metadata_json`,
  `clauses_json`. The existing Pydantic `Contract`/`Clause`/`ContractMetadata`
  models in `app/models/contract.py` remain unchanged and are the
  serialization boundary — `ContractStore.add()` serializes a `Contract` to a
  `ContractRecord` row; `ContractStore.get()` deserializes back.
- `ContractStore.__init__` accepts an optional SQLAlchemy engine URL
  (defaults to `sqlite:///{CONTRACTLENS_DB_PATH or "contractlens.db"}`);
  creates the table if missing via `SQLModel.metadata.create_all(engine)`.
- `main.py`/`pipeline.py` call sites (`store.add`, `store.get`, `store.new_id`)
  keep identical signatures — no changes needed outside `store.py`.
- Tests use `sqlite://` (in-memory) engines so they're isolated and don't
  touch disk.
- `contractlens.db` (or whatever `CONTRACTLENS_DB_PATH` resolves to) is
  gitignored.

### 2. CUAD evaluation harness (`scripts/evaluate_clauses.py`, `data/cuad_sample.json`)

- One-time fixture generation (documented in the script's docstring, run
  manually by the developer, not part of CI/tests): use the `datasets`
  library to load `theatticusproject/cuad` (public, ungated, CC-BY-4.0, no
  auth required — confirmed reachable), filter to the ~10 categories in
  `app/clauses/categories.py`, sample ~40-60 positive (non-empty-answer)
  clause spans per category, and write `{category, contract_title,
  clause_text}` records to `data/cuad_sample.json`. This file is committed
  (~1-2MB) so later runs and grading don't need network access or the
  `datasets` package.
- Scoring entrypoint (`python -m scripts.evaluate_clauses`, also importable
  for the smoke test): loads the committed fixture, classifies each
  `clause_text` with both `RuleBasedClassifier` and `LegalBertClassifier`,
  and prints a per-category + macro-averaged precision/recall/F1 table for
  each backend — using plain stdlib computation (no new sklearn dependency
  needed for single-label multi-class scoring at this size).
- This is the primary new "Factual" evidence artifact for the CP2 report:
  a script that runs and prints real numbers against a real (if sampled)
  slice of CUAD.

### 3. Real LegalBERT backend (`app/clauses/classifier.py`, `requirements.txt`)

- Uncomment `transformers`/`torch` in `requirements.txt`.
- **Design correction from the original brainstorm:** `nlpaueb/legal-bert-base-uncased`
  is tagged `fill-mask` on the HuggingFace Hub — it is a base encoder with no
  classification/NLI head. The originally-scaffolded approach
  (`pipeline("zero-shot-classification", model=self.model_name)`) would
  silently attach a randomly-initialized classification head and produce
  meaningless confidence scores, which would undermine rather than
  strengthen the eval report. Confirmed via the HuggingFace models API
  before finalizing this plan.
- **Corrected approach:** rewrite `LegalBertClassifier` to use LegalBERT
  purely as an encoder, with embedding + cosine-similarity classification:
  1. Load `AutoTokenizer`/`AutoModel` (not a `pipeline`) for
     `nlpaueb/legal-bert-base-uncased`.
  2. Build one reference embedding per category by mean-pooling LegalBERT's
     last-hidden-state token embeddings (attention-mask-weighted mean) over
     that category's keyword/description text (reuse
     `CATEGORY_KEYWORDS` from `app/clauses/categories.py`, joined into one
     string per category).
  3. For each clause, mean-pool its own token embeddings the same way, then
     compute cosine similarity against every category's reference embedding;
     predicted category = argmax similarity, confidence = that similarity
     score (clamped to `[0, 1]`).
  4. Still lazily imported (`transformers`/`torch` only imported inside the
     class, not at module load) and still falls back to `RuleBasedClassifier`
     if the imports or model load fail, preserving the existing
     "never hard-fails on missing optional dep" behavior.
- Known, explicitly-stated limitation for the report: this is embedding
  similarity against the pretrained (not fine-tuned) LegalBERT encoder —
  fine-tuning on CUAD is named as a stretch goal for later checkpoints, not
  attempted here.

### 4. Metadata extraction improvements (`app/pipeline.py`)

- `_TYPE_PATTERNS`: add Consulting Agreement, Supply Agreement, Reseller
  Agreement, Franchise Agreement, Joint Venture Agreement (common CUAD
  contract types) to the existing 7 patterns.
- `_detect_parties`: extend beyond the current "between X and Y" two-party
  regex to also match "by and between/among X, Y, and Z" style N-party
  preambles, splitting the matched span on commas/"and" into a party list.
  Stays regex-based (no NER/model dependency), consistent with the
  segmenter's existing transparent-heuristics design.

### 5. Test coverage additions

- `tests/test_parsers.py`: add fixture-based tests that actually exercise
  `parse_pdf`/`parse_docx` (generate a tiny real PDF/DOCX in a pytest
  fixture, assert extracted text matches expected content) — closing the
  gap where these paths were previously only manually verified.
- `tests/test_store.py` (new): round-trip a `Contract` through the SQLModel
  store (`add` → `get` → equals original) against an in-memory SQLite engine.
- Extend `tests/test_classifier.py` or add `tests/test_pipeline.py`: cases
  for the new multi-party detection and new contract-type patterns.
- `scripts/evaluate_clauses.py` gets a smoke test asserting the scoring
  function runs without error against a tiny (2-3 category) inline fixture —
  not the full committed sample, to keep test runtime fast.

## Error handling

No new user-facing error paths. `scripts/evaluate_clauses.py` is a
developer/reporting tool run out-of-band, not an API route — failures there
surface as a script traceback/non-zero exit, not an HTTP error response.

## Testing strategy

Existing `pytest` suite remains the source of truth; all additions above
plug into it. Full manual end-to-end verification (upload a real PDF/DOCX/TXT
through the FastAPI app, confirm persistence survives a process restart, run
the eval script and inspect output) happens once implementation is complete,
before writing the CP2 report.

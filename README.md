# ContractLens

**A Privacy-Preserving Contract Intelligence Platform**

ContractLens transforms unstructured contracts (PDF / DOCX / TXT) into structured
representations that support clause classification, semantic retrieval, contract
comparison, risk analysis, and question answering. The platform is designed to run
locally so that privacy-sensitive contracts never have to leave the user's machine.

## What's implemented

- A **FastAPI** service with a browser dashboard and a full JSON API (13 routes —
  see `app/main.py`'s module docstring for the complete list).
- A **document parsing layer** that extracts raw text from PDF, DOCX, and TXT
  contracts (`app/ingestion/parsers.py`).
- A **clause segmenter** that splits a contract into individual clauses/sections
  using heading and numbering heuristics (`app/ingestion/segmenter.py`).
- A **clause classifier** that labels each clause with one of 10 CUAD-style
  categories (confidentiality, termination, liability, indemnification,
  intellectual property, governing law, payment terms, warranty, assignment,
  force majeure). Three backends are available (`app/clauses/classifier.py`):
  - `RuleBasedClassifier` (default) — transparent keyword scoring, no model
    download required.
  - `LegalBertClassifier` — embeds each clause and each category's keyword
    description with LegalBERT (HuggingFace Transformers) and classifies by
    cosine similarity. No training involved.
  - `FineTunedLegalBertClassifier` — a real fine-tuned LegalBERT classification
    head, trained on a GPU (via PACE). Covers 7 of the 10 categories (see
    "Fine-tuned LegalBERT classifier" below for results and a real limitation
    worth knowing before you rely on it).
- **Metadata extraction** that detects contract type (12 common types) and
  party names, including multi-party ("by and among A, B, and C") preambles
  (`app/pipeline.py`).
- A **structured contract model** capturing contract type, parties, key
  sections, identified clauses, and metadata (`app/models/contract.py`).
- A **persistent SQLite-backed contract store**, so uploaded contracts survive
  a server restart (`app/store.py`).
- A **clause-visualization** view (`/contracts/{id}/view`) that lists the
  identified categories and shows each clause's text and category tag.
- **Semantic clause retrieval** (`app/retrieval/`) — search across every
  uploaded contract, or find clauses similar to one you already have.
- **Contract comparison** (`app/comparison/`) via optimal clause alignment —
  added / removed / modified / unchanged, clause by clause.
- **Evidence-backed risk analysis** (`app/risk/`) — every finding cites the
  exact clause text and character offsets it fired on.
- A **retrieval-grounded chatbot** (`app/chat/`) backed by a real **MCP
  server** (`app/mcp/`) — ask a natural-language question, get a cited answer.
  Two answer backends: a small local instruction-tuned LLM
  (`Qwen/Qwen2.5-0.5B-Instruct`) or a dependency-free extractive fallback.
- **Executive report generation** (`app/reports/`) — contract metadata,
  clause-category breakdown, and top risk findings, plus an LLM-written (or
  template) narrative summary.
- A **ContractLens dashboard** (`GET /`) — upload form plus a list of stored
  contracts linking to their view/chat/report pages.
- **Evaluation harnesses** for every one of the above: classification,
  retrieval, comparison, and risk-rule precision, plus a consolidated
  benchmark runner that writes committed, reproducible results.

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
                    ┌─────────────┼─────────────────────┐
                    ▼             ▼                      ▼
        SQLite-backed store   retrieval index      risk analyzer
                    │             │                      │
                    ▼             ▼                      ▼
             FastAPI JSON / HTML view      MCP server (search_clauses,
                    │                       assess_risk, build_report_data)
                    │                              │
                    ▼                              ▼
              /compare (diff)           chat assistant ──▶ /chat (cited answer)
                                         report generator ──▶ /report (executive summary)
```

## Project layout

```
app/
  main.py                  FastAPI app + routes (dashboard, upload/view, search/similar,
                             compare, risk, chat, report)
  config.py                 runtime configuration (env vars below)
  pipeline.py                parse -> segment -> classify -> store orchestration;
                              contract-type and party detection
  store.py                   SQLite-backed contract store (SQLModel)
  ingestion/
    parsers.py                PDF / DOCX / TXT text extraction
    segmenter.py               split text into clauses/sections
  clauses/
    categories.py              canonical CUAD-style clause categories
    classifier.py               rule-based, zero-shot LegalBERT, and fine-tuned
                                 LegalBERT classifiers
  models/
    contract.py                 Contract / Clause data models
    analysis.py                  RetrievalHit / ClauseChange / ContractDiff / RiskFinding / RiskReport models
    chat.py                      ChatRequest / Citation / ChatResponse models
    report.py                     ExecutiveReport model
  retrieval/
    embedder.py                 hashing + sentence-transformers embedding backends
    index.py                     in-memory cosine ClauseIndex
  comparison/
    comparator.py                optimal (Hungarian-algorithm) clause alignment
  risk/
    analyzer.py                  regex-based risk rule engine with evidence offsets
  mcp/
    server.py                    real MCP server (FastMCP/stdio): search_clauses,
                                   assess_risk, build_report_data tools
    client.py                    MCPToolClient — subprocess + ClientSession wrapper
  chat/
    assistant.py                 RAG ChatAssistant; ExtractiveBackend + LocalLLMBackend
  reports/
    generator.py                 executive report assembly + narrative (LLM or template)
  web/
    layout.py                    shared page shell / design tokens for the HTML UI
scripts/
  generate_cuad_sample.py    one-time generator for data/cuad_sample.json
  evaluate_clauses.py         scores all three classifiers against data/cuad_sample.json
  evaluate_retrieval.py       Recall@K / Success@K / MRR against data/cuad_sample.json
  evaluate_comparison.py      precision/recall/F1 against synthetic, controlled edits
  evaluate_risk.py            rule precision against data/risk_eval_labels.json
  run_benchmarks.py           combined classification + retrieval report; writes results/
  generate_finetune_dataset.py  builds the LegalBERT fine-tuning train/val split
  finetune_legalbert.py       LegalBERT fine-tuning script (run on a GPU, e.g. PACE)
  finetune_legalbert.sbatch   SLURM job script for PACE Phoenix
  make_presentation_charts.py  regenerates results/presentation/*.png from eval numbers
data/
  sample_contract.txt        tiny sample for local smoke testing
  cuad_sample.json            350 labeled clauses sampled from CUAD (7 categories)
  risk_eval_labels.json       30 hand-labeled clauses for risk-rule precision eval
  legalbert_finetune_train.json  560 clauses for fine-tuning (disjoint from cuad_sample.json)
  legalbert_finetune_val.json    140 held-out clauses for fine-tuning validation
models/                      gitignored -- fine-tuned classifier weights live here
  legalbert-finetuned/final/   produced by scripts/finetune_legalbert.py
results/
  benchmark_<timestamp>.json  committed benchmark runs (scripts/run_benchmarks.py)
  presentation/                charts + summary.json for the fine-tune results
docs/
  PACE_FINETUNE.md            runbook + results for the LegalBERT fine-tune
tests/
  test_segmenter.py, test_parsers.py, test_classifier.py, test_pipeline.py,
  test_store.py, test_evaluate_clauses.py, test_retrieval.py,
  test_evaluate_retrieval.py, test_comparison.py, test_evaluate_comparison.py,
  test_risk.py, test_evaluate_risk.py, test_mcp_server.py, test_mcp_client.py,
  test_chat.py, test_reports.py, test_main.py, test_run_benchmarks.py
requirements.txt
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/> for the dashboard, or
<http://127.0.0.1:8000/docs> for the interactive Swagger API. Uploaded
contracts persist in `contractlens.db` (SQLite) across server restarts.

The default configuration needs no model downloads to start up (rule-based
classifier, hashing-fallback-free semantic retrieval via
`sentence-transformers`). The chat backend defaults to the local LLM, which
downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1GB) on its first use — set
`CONTRACTLENS_CHAT_BACKEND=extractive` if you'd rather skip that.

### Configuration

All settings are optional environment variables (see `app/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `CONTRACTLENS_MAX_UPLOAD` | `26214400` (25 MB) | max upload size in bytes |
| `CONTRACTLENS_CLASSIFIER` | `rule` | classifier backend: `rule`, `legalbert`, or `legalbert-finetuned` |
| `CONTRACTLENS_LEGALBERT_MODEL` | `nlpaueb/legal-bert-base-uncased` | HF model id for the zero-shot LegalBERT backend |
| `CONTRACTLENS_LEGALBERT_FINETUNED_DIR` | `models/legalbert-finetuned/final` | local directory for the fine-tuned classifier's weights |
| `CONTRACTLENS_DB_PATH` | `contractlens.db` | SQLite file path |
| `CONTRACTLENS_RETRIEVAL_BACKEND` | `sentence` | retrieval embedding backend: `sentence` or `hashing` |
| `CONTRACTLENS_RETRIEVAL_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model id for the `sentence` backend |
| `CONTRACTLENS_CHAT_BACKEND` | `local` | chat backend: `local` or `extractive` |
| `CONTRACTLENS_CHAT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HF model id for the local chat backend |

## Running the test suite

```bash
python -m pytest -q
```

93 tests, fully offline by default (no model downloads or network access
required) — external dependencies are faked or injected where the real thing
would need a model download (`tests/test_chat.py`, `tests/test_reports.py`)
or spin up a subprocess (`tests/test_mcp_client.py`, `tests/test_main.py`,
which do genuinely launch the real MCP server as a subprocess, offline).

---

## Trying every feature

A single walkthrough hitting every endpoint, in the order you'd naturally use
them. Start the server first:

```bash
uvicorn app.main:app --reload
```

### 1. Upload a contract

```bash
curl -s -F "file=@data/sample_contract.txt" http://127.0.0.1:8000/upload | python3 -m json.tool
```

Returns structured JSON: contract type, parties, and every identified clause
with its category and confidence. Also works with real `.pdf` and `.docx`
files. Note the returned `"id"` — every example below substitutes `<id>` for it.

```json
{
  "id": "e7ccdecbc55a",
  "filename": "sample_contract.txt",
  "source_format": "txt",
  "metadata": {
    "contract_type": "Non-Disclosure Agreement",
    "parties": ["Acme Corporation", "Globex LLC"],
    "num_clauses": 7,
    "num_chars": 1476
  },
  "categories": {"Confidentiality": 2, "Governing Law": 1, "..." : "..."},
  "clauses": ["..."]
}
```

### 2. Structured JSON and the clause-visualization view

```bash
curl -s http://127.0.0.1:8000/contracts/<id> | python3 -m json.tool
```

Or in a browser: `http://127.0.0.1:8000/contracts/<id>/view` — every clause,
grouped by category, with a confidence badge and the exact character offsets
it was found at.

(`GET /health` is also available — a plain liveness/service-info endpoint,
useful for confirming the server is up and which backends it's configured
with, without touching contract data.)

### 3. Semantic search across every uploaded contract

```bash
curl -s "http://127.0.0.1:8000/search?q=confidentiality+obligations&k=3" | python3 -m json.tool
```

### 4. Clauses similar to one you already have

```bash
curl -s "http://127.0.0.1:8000/contracts/<id>/similar/0?k=3" | python3 -m json.tool
```

### 5. Compare two contracts

```bash
curl -s -X POST http://127.0.0.1:8000/compare \
  -F "base=@data/sample_contract.txt" \
  -F "revised=@data/sample_contract.txt" \
  | python3 -m json.tool
```

Returns a clause-level diff (`added` / `removed` / `modified` / `unchanged`)
via optimal bipartite matching, not a naive line diff. Both uploaded files
are persisted and indexed for search too, exactly like `/upload`.

### 6. Evidence-backed risk report

```bash
curl -s "http://127.0.0.1:8000/contracts/<id>/risk" | python3 -m json.tool
```

Every finding cites the clause it fired on, an evidence excerpt, and
character offsets — nothing is a black-box score.

### 7. Chat — ask a cited question

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if there is a breach of this agreement?", "contract_id": "<id>"}' \
  | python3 -m json.tool
```

With the default `local` backend, a real answer (verified working end-to-end):

```json
{
  "answer": "If there is a breach of this agreement, each party shall indemnify and hold harmless the other party from any claims arising out of a breach of this Agreement, and shall defend the other party against such claims.",
  "backend": "local-llm",
  "citations": [
    {"category": "Indemnification", "clause_index": 5, "score": 0.40},
    {"category": "Termination", "clause_index": 2, "score": 0.29}
  ],
  "tools_used": ["search_clauses"]
}
```

Set `CONTRACTLENS_CHAT_BACKEND=extractive` to skip the ~1GB model download
and get deterministic, clause-composed answers instead (no generation, just
the retrieved clauses assembled into a response) — useful for fast local
testing or a demo you don't want stalling on a first-call download. Also
browsable at `http://127.0.0.1:8000/contracts/<id>/chat`.

### 8. Executive report

```bash
curl -s "http://127.0.0.1:8000/contracts/<id>/report.json" | python3 -m json.tool
```

Or the HTML view: `http://127.0.0.1:8000/contracts/<id>/report` — contract
metadata, a clause-category breakdown, top risk findings, and a narrative
summary (LLM-written when the local backend is active, otherwise a
deterministic template sentence built from the same structured data).

### 9. The dashboard

`http://127.0.0.1:8000/` — upload form plus every stored contract, linking to
its clause view, chat, and report pages. This is the front door; everything
above is also reachable from here without touching curl.

### 10. Switch classifier backends

```bash
CONTRACTLENS_CLASSIFIER=legalbert uvicorn app.main:app --reload             # zero-shot, no training
CONTRACTLENS_CLASSIFIER=legalbert-finetuned uvicorn app.main:app --reload   # real fine-tuned weights (see below)
```

Re-upload a contract after switching — classification happens at ingest
time, so already-stored contracts keep whichever category they were
originally classified with.

### 11. Run the consolidated benchmark

```bash
python -m scripts.run_benchmarks                                       # rule + hashing, writes results/
python -m scripts.run_benchmarks --classifier legalbert-finetuned --retrieval-backend hashing
python -m scripts.run_benchmarks --out ""                              # skip writing a results file
```

Prints a combined classification + retrieval report and writes a timestamped
JSON summary to `results/` by default, so headline numbers are reproducible
directly from the repository rather than living only in report screenshots.

---

## Evaluation harnesses

Every capability above has a dedicated, offline-reproducible evaluation
script, all scoring against real or hand-labeled fixtures committed in
`data/`:

```bash
python -m scripts.evaluate_clauses --backend rule                # or legalbert, legalbert-finetuned, both
python -m scripts.evaluate_retrieval --backend hashing            # or --backend sentence
python -m scripts.evaluate_comparison --backend hashing           # or --backend sentence
python -m scripts.evaluate_risk                                   # hand-labeled rule precision
python -m scripts.run_benchmarks                                  # combined classification + retrieval report
```

- `evaluate_clauses.py` scores all three classifier backends against
  `data/cuad_sample.json` (350 real labeled clauses, 7 of the 10 taxonomy
  categories — see "Datasets" below for why 3 are excluded).
- `evaluate_retrieval.py` and `evaluate_comparison.py` score against the same
  fixture (retrieval via leave-one-out same-category recall; comparison via
  synthetic, controlled edits, since no public labeled contract-diff dataset
  exists).
- `evaluate_risk.py` scores against `data/risk_eval_labels.json`, a
  hand-labeled fixture — see that file's construction notes for what's real
  CUAD-derived text versus hand-authored representative legal language.
  Measures precision of firing only; recall would require exhaustively
  labeling every risky clause in the corpus, out of scope at this size.

To regenerate `data/cuad_sample.json` from the source dataset (requires the
`datasets` package and network access):

```bash
python -m scripts.generate_cuad_sample
```

## Fine-tuned LegalBERT classifier

A real fine-tuned classification head (`nlpaueb/legal-bert-base-uncased` +
a linear layer, trained end-to-end via cross-entropy), trained on a GPU via
PACE — not the zero-shot cosine-similarity approach the `legalbert` backend
uses. Full runbook, results, and known caveats: **[docs/PACE_FINETUNE.md](docs/PACE_FINETUNE.md)**.

Headline result, scored against the same 350-clause `data/cuad_sample.json`
fixture every other classifier is benchmarked against:

| Backend | Macro F1 |
|---|---|
| Fine-tuned LegalBERT | **0.966** |
| Rule-based (keyword) | 0.646 |
| Zero-shot LegalBERT | 0.554 |

**Two things worth knowing before trusting that number:**

1. 78% of the benchmark's source contracts also contributed other clauses to
   training, so 0.966 overstates generalization to genuinely unseen contract
   templates. A cleaner check — two hand-written test contracts with zero
   relation to CUAD, in different registers (plain English and dense
   legalese) — put it at 13/13 correct on its trained categories, which is a
   more trustworthy (if smaller-sample) signal.
2. It only covers 7 of the 10 categories (CUAD has no ground truth for
   Confidentiality, Indemnification, or Force Majeure) and, unlike the
   rule-based classifier, has no "Unclassified" output — fed a clause from
   one of those 3 categories, it confidently picks the closest of its 7
   known labels instead (observed: Confidentiality → Intellectual Property,
   Indemnification → Liability). This is why it's opt-in
   (`CONTRACTLENS_CLASSIFIER=legalbert-finetuned`), not the default.

To reproduce the training run or fine-tune again with more data, see
[docs/PACE_FINETUNE.md](docs/PACE_FINETUNE.md). Charts and the raw summary
are committed at `results/presentation/`.

## Datasets

- **CUAD** (Contract Understanding Atticus Dataset) — clause categories and
  evaluation for classification, retrieval, and comparison, and the source
  data for the LegalBERT fine-tune. The canonical category list in
  `app/clauses/categories.py` follows the CUAD taxonomy; `data/cuad_sample.json`
  is a real labeled sample used by every evaluation harness above. CUAD's
  original 41 categories don't include Confidentiality, Indemnification, or
  Force Majeure, so those 3 of the taxonomy's 10 categories have no CUAD
  ground truth and are excluded from every CUAD-based evaluation (including
  the fine-tune) — a data-availability limitation, not an oversight.
- **ACORD** — evaluated as an option for clause-retrieval evaluation; CUAD
  was used instead for both retrieval and comparison eval, since it already
  had a committed, labeled fixture in place from classification.

## Roadmap

- Extend the fine-tuned classifier to the 3 CUAD-uncovered categories with
  hand-labeled data (mirroring `data/risk_eval_labels.json`'s approach for
  risk rules), and/or add a confidence-threshold reject option so it can
  fall back to the rule-based classifier instead of forcing a guess.
- Re-evaluate the fine-tune with a contract-level (not just clause-level)
  train/test split to get a cleaner read on true generalization.

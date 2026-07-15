# Checkpoint 3 Features — Design

## Context

CP2 delivered ingestion, persistent storage, clause classification (rule-based +
LegalBERT), metadata extraction, and a CUAD-based evaluation harness. The TA's
CP2 feedback was strongly positive and named one concrete suggestion for CP3:
bring retrieval and comparison under the same measured, harness-backed
treatment already given to classification.

The professor's course tooling generated a reference `ai-suggestions/cp3`
branch (delivered as a zip attachment, `620230_cp2.zip`, not a GitHub push)
implementing semantic retrieval, contract comparison, and risk analysis on top
of the real CP2 code. Per course policy this scaffold cannot be submitted
as-is; it was reviewed for structure and ideas only. This spec intentionally
departs from it in several places (embedding model, alignment algorithm, and
by adding a comparison eval harness the scaffold lacks) to exceed rather than
copy it.

## Goals

- Implement all three CP3 deliverables: semantic clause retrieval, contract
  comparison, and evidence-backed risk analysis.
- Give retrieval and comparison the same measured (precision/recall/F1-style)
  evaluation treatment CP2 gave classification, directly answering the TA's
  named suggestion.
- Exceed the reference scaffold's engineering quality in at least three
  concrete ways: a real sentence-embedding model instead of hashing/reused
  LegalBERT tokens, optimal (not greedy) clause alignment for comparison, and
  a comparison eval harness the scaffold doesn't have.

## Non-goals

- No UI beyond the existing minimal HTML view; CP3 stays API-first like CP2.
- No FAISS/ChromaDB — the corpus size for this course project doesn't need an
  ANN index; a documented brute-force cosine index is sufficient and keeps the
  dependency footprint small (the retrieval index's public surface is designed
  so a real ANN backend could be dropped in later without touching callers).
- No claim of recall for risk analysis — only precision-of-firing is measured
  (see Risk analysis / Evaluation below); claiming recall would require
  exhaustively labeling every risky clause in the corpus, which isn't feasible
  at this scope.

## Architecture

```
                       ┌─────────────────────────┐
                       │   SQLite contract store  │  (existing, CP2)
                       └───────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     app/retrieval/         app/comparison/        app/risk/
     embedder.py             comparator.py          analyzer.py
     index.py
              │                    │                    │
     GET /search            POST /compare         GET /contracts/{id}/risk
     GET /contracts/{id}/similar/{clause_index}
```

New top-level packages `app/retrieval/`, `app/comparison/`, `app/risk/`,
mirroring the existing `app/clauses/` layout. New shared response models live
in `app/models/analysis.py` (`RetrievalHit`, `ClauseChange`/`ContractDiff`,
`RiskFinding`/`RiskReport`), sitting alongside the existing
`app/models/contract.py`.

## Components

### 1. Semantic retrieval (`app/retrieval/`)

**Embedding backends** (`embedder.py`), same two-backend pattern as
`app/clauses/classifier.py`:

- `SentenceTransformerEmbedder` (default when the optional dependency is
  installed) — uses a small sentence-embedding model (`all-MiniLM-L6-v2`)
  that is purpose-trained for cosine-similarity retrieval, unlike LegalBERT
  (a masked-LM encoder never trained for sentence similarity — mean-pooling
  its tokens, as CP2's classifier does, is a usable workaround for
  classification but a weaker choice for retrieval specifically).
- `HashingEmbedder` — dependency-free deterministic bag-of-words hashing
  embedding (ported from the scaffold's design), used as the offline fallback
  and default when `sentence-transformers` isn't installed.
- `get_embedder()` selects based on config/availability, falling back to
  `HashingEmbedder` on any load failure, matching `LegalBertClassifier`'s
  existing fallback behavior.

**Index** (`index.py`): `ClauseIndex` is a brute-force cosine nearest-neighbor
index. Deviating from the scaffold's design, which rebuilds the index from
every stored contract on every `/search` and `/similar` call (re-embedding
every clause on every request): this design maintains a **process-wide
singleton index updated incrementally** — `/upload` embeds and adds only the
newly-ingested contract's clauses once; `/search` and `/similar` reuse the
existing index without re-embedding anything. Public surface
(`add_contract`/`search`/`most_similar_to`) stays the same as the scaffold's,
so a future ANN backend could be substituted without touching callers.

**API:**
- `GET /search?q=...&k=5[&category=...]` — semantic search across all
  contracts' clauses.
- `GET /contracts/{id}/similar/{clause_index}?k=5` — clauses similar to one
  already-indexed clause.

**Evaluation** (`scripts/evaluate_retrieval.py`): reuses the existing
`data/cuad_sample.json` fixture (350 labeled clauses, 7 CUAD categories).
Each clause is used as a query in turn; other clauses sharing its category are
the ground-truth relevant set. Reports Recall@K, Success@K, and MRR — the
metrics named in the original project plan — for both embedder backends via
`--backend rule|sentence`.

### 2. Contract comparison (`app/comparison/`)

**Alignment** (`comparator.py`): embed every clause of a base and a revised
contract, build a full pairwise cosine similarity matrix, and align clauses
via `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) — a globally
optimal one-to-one assignment, unlike the scaffold's greedy highest-similarity
-first matching (which can lock in a suboptimal pairing when a clause's best
match is claimed by a slightly-better competing pair). Pairs below
`DEFAULT_MATCH_THRESHOLD` (0.60) are rejected regardless of what the optimal
assignment would otherwise produce, so unrelated clauses are never forced
together. An aligned pair is `unchanged` if similarity ≥
`DEFAULT_IDENTICAL_THRESHOLD` (0.995) or the texts are identical after
stripping; otherwise `modified`. Unmatched base clauses are `removed`,
unmatched revised clauses are `added`.

**API:** `POST /compare` — two file uploads (`base`, `revised`), returns a
`ContractDiff` (list of `ClauseChange` plus a summary count by change type).

**Evaluation** (`scripts/evaluate_comparison.py`, new — the scaffold has no
comparison eval at all): takes real contracts from the CUAD sample corpus and
generates a synthetic "revised" version via controlled edits — delete N
clauses (expect `removed`), duplicate+reword M clauses via a small synonym
substitution (expect `modified`), insert K clauses drawn from an unrelated
contract (expect `added`), leave the rest untouched (expect `unchanged`).
Runs the comparator against the synthetic pair and reports precision/recall/F1
per change type against the known ground truth. This is the concrete
"beyond-scope" item answering the TA's suggestion to extend measured
evaluation beyond classification.

### 3. Risk analysis (`app/risk/`)

**Rule engine** (`analyzer.py`), transparent and deterministic — no model
download, no black-box score:

- **Clause-level rules**: a regex `trigger` scoped to a clause category (e.g.
  uncapped liability, broad "any and all" indemnification, termination for
  convenience, auto-renewal, IP assignment, warranty disclaimer, assignment
  without consent, non-refundable payment terms, perpetual confidentiality —
  ported from the scaffold's 9 — plus new coverage: one-sided arbitration
  clauses, unilateral amendment rights, and non-mutual confidentiality
  obligations, for ~15 rules total). Each rule may carry a `suppressed_by`
  pattern so a clause that also contains mitigating language (e.g. an
  explicit liability cap alongside broad exposure language) doesn't fire a
  false positive.
- **Contract-level rules**: fire when an entire expected protective category
  (Liability, Governing Law, Confidentiality) is absent from the contract;
  evidence is "no clause of category X found," cited against the whole
  document.

**Evidence:** every `RiskFinding` carries `clause_index`, an `evidence_text`
excerpt (a ~120-character window around the actual matched phrase, not the
full clause), and `start_offset`/`end_offset` into the source document.

**Aggregate score:** severities carry weights (low=1, medium=3, high=6);
`overall_score = 100 * (1 - e^(-total_weight/6))`, bounded in [0, 100], mapped
to a `risk_level` of low/medium/high by threshold.

**API:** `GET /contracts/{id}/risk`.

**Evaluation** (new, extends beyond the scaffold): a small hand-labeled set of
~25-30 real clauses (drawn from the CUAD sample) labeled per-rule as
"should fire" / "should not fire." Reports **precision of firings** per rule.
Explicitly does not claim recall — measuring recall would require exhaustively
labeling every risky clause across the whole corpus for every rule, which is
out of scope; the report will state this limitation directly rather than
implying the rules catch all risk.

## Data flow

Unchanged from CP2 through clause classification. The new layer reads
`Contract`/`Clause` objects already produced by the existing
parse → segment → classify → store pipeline; no schema changes to
`app/store.py` or `app/models/contract.py` are needed. New response shapes
live entirely in `app/models/analysis.py`.

## Error handling

- `/compare` and `/similar` reuse the existing `_ingest_upload` helper's
  validation (unsupported format → 415, empty file → 400, oversized → 413,
  missing optional parser dependency → 501) — same codes as `/upload`.
- `/search` and `/contracts/{id}/risk` return 404 for an unknown contract ID.
- `SentenceTransformerEmbedder` falls back to `HashingEmbedder` if the model
  fails to load (missing dependency, no network for first download, etc.),
  matching `LegalBertClassifier`'s existing fallback-on-exception behavior.

## Testing

- `tests/test_retrieval.py`, `tests/test_comparison.py`, `tests/test_risk.py`
  — unit tests against small synthetic contracts, using an injected fake
  embedder (same dependency-injection pattern `test_classifier.py` already
  uses for `LegalBertClassifier`) so tests stay fast and offline.
- `scripts/evaluate_retrieval.py` and `scripts/evaluate_comparison.py` are
  smoke-tested (run end-to-end on a small `--limit`) as part of the test
  suite, mirroring how `test_evaluate_clauses.py` covers
  `scripts/evaluate_clauses.py`.

## Dependencies

- `scipy` (Hungarian algorithm for comparison alignment) — new.
- `sentence-transformers` (semantic retrieval embedder) — new, optional at
  runtime with fallback, same posture as `transformers`/`torch` for
  LegalBERT.

# Checkpoint 4 Features — Design

## Context

CP3 delivered semantic retrieval, contract comparison, and evidence-backed
risk analysis, all API-first with no UI beyond the CP2 clause-visualization
page. The TA's CP3 feedback was strongly positive on all three axes and named
one concrete suggestion: commit the numeric outputs of the evaluation
harnesses (as logs or a results file) alongside the code, so the headline
results are reproducible directly from the repository rather than living only
in report screenshots.

The professor's course tooling generated a reference `ai-suggestions/cp4`
branch (delivered as a zip attachment, `620230_cp3.zip`, not a GitHub push),
built on top of the real CP3 code. Per the TA's guidance for this course, this
scaffold is not meant to be submitted as-is *or* ignored — it must be used and
substantively expanded on. This spec treats several of its modules (the
`ChatResponse`/`Citation` models, the dependency-free `ExtractiveBackend`, and
the minimal chat HTML page) as a real starting point, kept close to as-is
where they're already well-built, while adding genuine new engineering in the
areas that most needed it: a real MCP server instead of an in-process-only
tool registry, a capable local-LLM backend instead of a non-instruction-tuned
model, a report-generation feature the scaffold doesn't have at all, and a
committed/reproducible benchmark artifact answering the TA's CP3 suggestion.

The CP4 milestone, per the original project plan's milestone chart, is
**"Platform and evaluation": "Design and build ContractLens UI. Integrate
chatbot for intelligent querying. Evaluate classification and retrieval
performance with benchmark datasets."** The plan's Project Deliverables table
additionally specifies the chatbot deliverable includes the ability to
"generate executive review reports summarizing key findings and risks" and
names "Local LLM, MCP Tools, RAG" as its technical stack — both requirements
this spec covers explicitly.

## Goals

- Design and build a ContractLens UI: a contract dashboard (upload + list of
  stored contracts) tying together the existing CP2/CP3 endpoints, plus the
  chat page and a new report page.
- Integrate a chatbot for intelligent querying: retrieval-grounded (RAG)
  question answering with cited clauses, using a real local instruction-tuned
  LLM (not the scaffold's raw `distilgpt2` completion model) with a
  deterministic extractive fallback.
- Implement "MCP-style tool integrations" as an actual MCP server (not an
  in-process function registry), used by both the chat assistant and the new
  report feature — matching the plan's literal "MCP Tools" stack item and
  going further than the scaffold's hand-rolled dataclass registry.
- Add the executive-report-generation feature named in the original plan but
  absent from the scaffold: a structured summary of a contract's metadata,
  key clauses, and top risk findings, with an optional LLM-written narrative
  paragraph.
- Evaluate classification and retrieval performance with benchmark datasets,
  reusing the existing CP2/CP3 harnesses, and commit the run's numeric output
  as a results artifact (the TA's specific CP3 suggestion).

## Non-goals

- No UI for `/search` or `/compare` — the dashboard links to the existing
  `/view` page, plus the new `/chat` and `/report` pages; search and compare
  stay Swagger-only, same posture CP3 had for everything.
- No fine-tuning (PACE or otherwise) — still a stretch item beyond CP4.
- No claim that the local-LLM chat backend or report narrative is
  hallucination-free; both are explicitly grounded/fallback-guarded (see
  Error handling) but the report will state this as a known limitation rather
  than overclaiming.
- No new persistence layer for chat history — each `/chat` call is
  stateless, consistent with the rest of the API being stateless per-request
  over the existing SQLite store.

## Architecture

```
FastAPI app (app/main.py)
 ├─ existing CP2/CP3 routes (unchanged)
 ├─ GET  /                        -> dashboard (upload form + contract list)
 ├─ GET  /contracts/{id}/chat     -> chat UI page (from scaffold, kept)
 ├─ POST /chat                    -> chat endpoint
 ├─ GET  /contracts/{id}/report   -> executive report (HTML)
 ├─ GET  /contracts/{id}/report.json -> executive report (JSON)
 │
 └─ ChatAssistant (app/chat/assistant.py)
      ├─ MCPToolClient ──stdio subprocess──> MCP server (app/mcp/server.py)
      │                                        ├─ tool: search_clauses
      │                                        ├─ tool: assess_risk
      │                                        └─ tool: build_report_data
      └─ answer backend: LocalLLMBackend (small instruct model)
                          | ExtractiveBackend (fallback, from scaffold)

app/reports/generator.py  -- calls the MCP client, assembles ExecutiveReport

scripts/run_benchmarks.py -- reuses evaluate_clauses.py + evaluate_retrieval.py
                             -> writes results/benchmark_<timestamp>.json (committed)
```

New top-level packages: `app/mcp/` (server + client), `app/reports/`
(generator). `app/chat/` is kept from the scaffold with the backend and tool
plumbing reworked. New shared response model `ExecutiveReport` lives in
`app/models/report.py`, alongside the kept `app/models/chat.py`.

The key structural change from the scaffold: the scaffold's
`app/chat/tools.py` was an in-process dataclass registry called directly by
`ChatAssistant`. Here, `app/mcp/server.py` is a real, independently-runnable
MCP server (`mcp dev app/mcp/server.py` lists its tools like any MCP server
would), and both `ChatAssistant` *and* `app/reports/generator.py` reach the
underlying services (`ClauseIndex.search`, `analyze_risk`, contract lookup)
exclusively through an MCP client session — there is no direct-call path left
that bypasses MCP, so "MCP-style tool integration" is the actual service
boundary for the app's two AI-facing features, not just a chat-only shim.

## Components

### 1. MCP server and client (`app/mcp/`)

**Server** (`server.py`): built with the `mcp` Python SDK's `FastMCP`,
registers three tools:

- `search_clauses(query, k=5, contract_id=None, category=None)` — thin
  wrapper over the existing process-wide `ClauseIndex.search`, ported
  directly from the scaffold's `search_clauses_tool` logic (over-fetch +
  filter when scoped to one contract).
- `assess_risk(contract_id)` — wraps the existing `analyze_risk`.
- `build_report_data(contract_id)` — new: assembles the raw structured
  inputs an executive report needs (contract metadata, clauses grouped by
  category, top risk findings) in one call, so the report generator (and any
  future MCP client) doesn't need three round-trips.

The server runs over stdio and holds no state of its own — it calls into the
same process-wide `store` and `ClauseIndex` singletons `app/main.py` already
constructs, imported directly (this is a single-deployment-unit app; the MCP
boundary is about tool structure and protocol-correctness, not process
isolation).

**Client** (`client.py`): `MCPToolClient` launches `server.py` as a subprocess
via the MCP SDK's stdio client helper on first use, opens a `ClientSession`,
and exposes an async `call_tool(name, **kwargs) -> dict`. Both
`ChatAssistant` and `app/reports/generator.py` depend only on this client,
never on `ClauseIndex`/`analyze_risk` directly, once wired into `main.py`.

### 2. Chat assistant (`app/chat/`)

Kept from the scaffold, reworked in two places:

- **Tool access**: `ChatAssistant` now calls `MCPToolClient.call_tool(...)`
  instead of the scaffold's direct `build_registry(index)` dataclass calls.
  `tools_used` bookkeeping (which the scaffold's tests already assert on)
  stays semantically identical.
- **LLM backend**: `LocalLLMBackend` swaps `distilgpt2` (a small,
  non-instruction-tuned model that tends to ramble/repeat rather than follow
  a grounding instruction) for a small **instruction-tuned** model
  (`Qwen2.5-0.5B-Instruct`), invoked through `transformers`' chat-template
  API (`apply_chat_template`) with a system prompt constraining the answer to
  the numbered retrieved clauses, rather than raw string-prompt completion.
  Same lazy-load-with-fallback shape as the scaffold: any failure to import
  `transformers` or load the model falls back to `ExtractiveBackend`, so
  nothing hard-fails and the dependency stays optional.
- `ExtractiveBackend`, `ChatRequest`/`Citation`/`ChatResponse`, the risk-intent
  regex, and the chat HTML page are kept from the scaffold as-is — they're
  already a clean, dependency-free implementation of "compose a grounded
  answer from retrieved clauses" with no reason to redo them differently.

**API:** `POST /chat` (unchanged from scaffold), `GET /contracts/{id}/chat`
(chat UI, unchanged from scaffold).

### 3. Executive report (`app/reports/generator.py`, new)

- Calls `MCPToolClient.call_tool("build_report_data", contract_id=...)` to
  get metadata, clauses-by-category, and top risk findings in one shot.
- Assembles an `ExecutiveReport` (new model, `app/models/report.py`):
  contract metadata, a clause-category breakdown (counts + one representative
  heading per category), the top N risk findings with severity and
  rationale, and an optional `narrative: str | None`.
- **Narrative**: if the chat LLM backend is available, ask it (via the same
  `LocalLLMBackend`, a distinct prompt) for a 2-4 sentence plain-English
  summary of the report's structured data — never asked to invent facts, only
  to phrase the already-extracted findings. If unavailable, `narrative` is a
  deterministic template sentence built from the same data (e.g. "This
  {contract_type} contract has {N} clauses across {M} categories; overall
  risk is {level} ({score}/100), driven primarily by {top rationale}.").
  Either way the structured fields are always present and are what's
  actually tested; the narrative is presentational.

**API:** `GET /contracts/{id}/report` (HTML view, same minimal-inline-CSS
style as `/contracts/{id}/view`), `GET /contracts/{id}/report.json` (the raw
`ExecutiveReport`).

### 4. Dashboard UI (`app/main.py`, new route)

- `GET /` becomes an HTML dashboard (superseding the current JSON health
  route — health/liveness info moves to `GET /health` so nothing currently
  depending on `/`'s JSON shape breaks silently... this app has no external
  consumers of `/`, so this is a clean rename, not a breaking-change concern
  worth hedging on): an upload form (posts to the existing `/upload`) and a
  table of stored contracts (id, filename, type, clause count) each linking
  to `/contracts/{id}/view`, `/chat`, and `/report`.
- Reuses `store.list_ids()`/`store.get()`, already used elsewhere in
  `main.py`; no new persistence.

### 5. Benchmark reproducibility (`scripts/run_benchmarks.py`, expanded)

- Scaffold's combined classification + retrieval report logic is kept as-is.
- Adds `--out results/benchmark_<UTC-timestamp>.json` (default: on), so every
  run writes a committed, timestamped JSON artifact under a new `results/`
  directory — directly answering the TA's CP3 feedback that headline numbers
  should be reproducible from the repo, not just report screenshots.

## Data flow

Unchanged from CP3 for ingestion/classification/retrieval/comparison/risk.
The new layer's data flow is: `store` + `ClauseIndex` (existing) → MCP server
tools → MCP client → `ChatAssistant` / `app/reports/generator.py` → HTTP
routes. No changes to `app/store.py`, `app/models/contract.py`, or
`app/models/analysis.py`.

## Error handling

- `/chat` and `/contracts/{id}/report[.json]` return 404 for an unknown
  `contract_id`, consistent with existing endpoints.
- MCP subprocess failing to start or a tool call raising: caught in
  `MCPToolClient`, surfaced as a 502 from `/chat`/`/report` with a clear
  detail message — this is a real dependency the app now needs, unlike the
  optional-model fallbacks below, so a clear error beats a silent partial
  answer.
- `LocalLLMBackend` model load or generation failure: falls back to
  `ExtractiveBackend` for chat answers and to the template narrative for
  reports — same fallback posture the scaffold already established and CP3
  established for `SentenceTransformerEmbedder`/`LegalBertClassifier`.

## Testing

- `tests/test_mcp_server.py` (new): tool functions tested by calling the
  server's registered tool functions directly (in-process, no subprocess) —
  same fast/offline posture as existing tests.
- `tests/test_chat.py` (kept from scaffold, adapted): assertions about
  `tools_used`, citation grounding, and scoping stay valid; the fixture that
  built a fake in-process registry is replaced with a fake `MCPToolClient` so
  the tests stay synchronous and offline.
- `tests/test_reports.py` (new): report assembly and the template-narrative
  fallback path are deterministic and tested without invoking any model.
- `tests/test_run_benchmarks.py` (kept from scaffold): extended to check the
  `--out` file is written and is valid JSON.
- Dashboard route gets a smoke test (renders 200, contains an uploaded
  contract's filename) alongside the existing `test_pipeline.py`-style
  integration tests.

## Dependencies

- `mcp` (Anthropic's MCP Python SDK) — new.
- `transformers` — already an optional/soft dependency from CP3
  (`LegalBertClassifier`); the instruct-model swap keeps the same
  try/except-guarded, falls-back-cleanly shape, just changes which model id
  is requested when the backend is used.

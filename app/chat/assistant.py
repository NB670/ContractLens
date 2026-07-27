"""Retrieval-grounded contract chatbot (Checkpoint 4).

A user asks a natural-language question about their contract(s) and gets a
*cited* answer, where every claim is grounded in clauses retrieved via the
MCP tool layer (app/mcp/), never in the model's own parametric memory. Local-
first, in keeping with the project's privacy-preserving goal.

Two backends:

  * ExtractiveBackend (dependency-free) -- composes the answer directly from
    the retrieved clauses. No model download, deterministic, always
    available.

  * LocalLLMBackend (default) -- lazily loads a small local, instruction-
    tuned transformers text-generation model (Qwen2.5-0.5B-Instruct by
    default) and asks it to answer *using only* the retrieved clauses, citing
    them by number, via the model's chat template. Falls back to the
    extractive backend if the dependency or model is unavailable.

All grounding (search_clauses, assess_risk) comes from the MCP tool layer via
an injected client -- this module never imports app.retrieval or app.risk
directly.
"""

from __future__ import annotations

import re

from app.config import settings
from app.models.chat import ChatResponse, Citation

_EXCERPT_CHARS = 320
_RISK_INTENT = re.compile(r"\brisk|risky|liabilit|indemnif|dangerous|red flag|concern", re.IGNORECASE)


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _hit_to_citation(hit: dict) -> Citation:
    return Citation(
        contract_id=hit["contract_id"],
        clause_index=hit["clause_index"],
        category=hit.get("category", "Unclassified"),
        heading=hit.get("heading"),
        excerpt=_excerpt(hit.get("text", "")),
        score=hit.get("score", 0.0),
    )


class ExtractiveBackend:
    """Compose a grounded answer straight from the retrieved clauses."""

    name = "extractive"

    def generate(self, question: str, citations: list[Citation], risk_summary: str | None) -> str:
        if not citations:
            return (
                "I could not find any clauses relevant to that question in the "
                "uploaded contract(s). Try uploading the contract first, or "
                "rephrasing the question."
            )
        lines = [f"Based on {len(citations)} relevant clause(s) in the reviewed contract(s):", ""]
        for i, cit in enumerate(citations, start=1):
            label = cit.heading or cit.category
            lines.append(
                f"[{i}] ({label}) {cit.excerpt} "
                f"— {cit.contract_id}, clause {cit.clause_index} "
                f"(similarity {cit.score:.2f})"
            )
        if risk_summary:
            lines += ["", risk_summary]
        return "\n".join(lines)


class LocalLLMBackend:
    """Local instruction-tuned transformers backend with extractive fallback."""

    name = "local-llm"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.chat_model
        self._pipe = None
        self._load_failed = False
        self._fallback = ExtractiveBackend()

    def _ensure_pipe(self) -> bool:
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False
        try:  # pragma: no cover - depends on optional heavy dep
            from transformers import pipeline

            self._pipe = pipeline("text-generation", model=self.model_name)
            return True
        except Exception:
            self._load_failed = True
            return False

    def generate_raw(self, messages: list[dict], max_new_tokens: int = 256) -> str | None:
        """Run the model on a chat-style message list; None on any failure.

        Public (not underscore-prefixed) because app/reports/generator.py
        reuses this for the executive-report narrative, sharing one model
        load instead of duplicating the pipeline-loading logic.
        """
        if not self._ensure_pipe():
            return None
        try:  # pragma: no cover - depends on optional heavy dep
            out = self._pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
            generated = out[0]["generated_text"]
            text = generated[-1]["content"].strip() if isinstance(generated, list) else str(generated).strip()
            return text or None
        except Exception:
            return None

    @staticmethod
    def _build_messages(question: str, citations: list[Citation], risk_summary: str | None) -> list[dict]:
        context_lines = [f"[{i}] ({c.category}) {c.excerpt}" for i, c in enumerate(citations, start=1)]
        context = "\n".join(context_lines) if context_lines else "(no relevant clauses found)"
        risk_block = f"\n\nRisk analysis:\n{risk_summary}" if risk_summary else ""
        system = (
            "You are ContractLens, a contract-review assistant. Answer the "
            "question using ONLY the numbered contract clauses provided. Cite "
            "the clauses you rely on by their number in square brackets, e.g. "
            "[1]. If the clauses do not contain the answer, say so plainly."
        )
        user = f"Clauses:\n{context}{risk_block}\n\nQuestion: {question}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, question: str, citations: list[Citation], risk_summary: str | None) -> str:
        text = self.generate_raw(self._build_messages(question, citations, risk_summary))
        return text or self._fallback.generate(question, citations, risk_summary)


def _make_backend(backend: str):
    if backend == "extractive":
        return ExtractiveBackend()
    return LocalLLMBackend()


class ChatAssistant:
    """RAG assistant: retrieve grounding clauses via MCP, then answer with citations.

    ``backend`` is public (not underscore-prefixed) because
    app/reports/generator.py reuses it, when it's a LocalLLMBackend, to write
    the executive-report narrative -- sharing the one loaded model instead of
    loading a second copy just for reports.
    """

    def __init__(self, mcp_client, backend: str | None = None) -> None:
        self.mcp_client = mcp_client
        self.backend = _make_backend(backend or settings.chat_backend)

    async def _risk_summary(self, contract_id: str | None) -> tuple[str | None, bool]:
        if contract_id is None:
            return None, False
        result = await self.mcp_client.call_tool("assess_risk", contract_id=contract_id)
        if not result.get("found"):
            return None, False
        top = result["findings"][:3]
        if not top:
            return (
                f"Risk analysis: overall risk {result['risk_level']} "
                f"(score {result['overall_score']:.0f}/100); no specific findings.",
                True,
            )
        bullet = "; ".join(f"{f['severity']} — {f['rationale']}" for f in top)
        return (
            f"Risk analysis: overall risk {result['risk_level']} "
            f"(score {result['overall_score']:.0f}/100). Top findings: {bullet}",
            True,
        )

    async def answer(self, question: str, contract_id: str | None = None, k: int = 5) -> ChatResponse:
        tools_used: list[str] = []

        hits: list[dict] = []
        if question and question.strip():
            result = await self.mcp_client.call_tool(
                "search_clauses", query=question, k=k, contract_id=contract_id
            )
            hits = result["hits"]
            tools_used.append("search_clauses")

        citations = [_hit_to_citation(h) for h in hits]

        risk_summary = None
        if _RISK_INTENT.search(question or ""):
            risk_summary, used = await self._risk_summary(contract_id)
            if used:
                tools_used.append("assess_risk")

        answer_text = self.backend.generate(question, citations, risk_summary)

        return ChatResponse(
            question=question, answer=answer_text, backend=self.backend.name,
            contract_id=contract_id, citations=citations, tools_used=tools_used,
        )


def build_default_assistant(mcp_client) -> ChatAssistant:
    """Construct the assistant over a given MCPToolClient."""
    return ChatAssistant(mcp_client)

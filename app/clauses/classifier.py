"""Clause classification.

Two backends are provided:

  * ``RuleBasedClassifier`` (default) — transparent keyword scoring over the CUAD
    category taxonomy. No model download, runs anywhere, and gives us a baseline
    to evaluate the learned model against in later checkpoints.

  * ``LegalBertClassifier`` — optional embedding backend built on HuggingFace
    Transformers + LegalBERT. Each category and clause is embedded via
    mean-pooled token embeddings and compared by cosine similarity; there is
    no zero-shot/NLI head involved. Imported lazily so the service runs
    without torch/transformers installed.

Both implement ``classify(text) -> (category, confidence)`` and a convenience
``classify_clauses(clauses)`` that annotates Clause objects in place.
"""

from __future__ import annotations

from typing import Callable

from app.clauses.categories import CATEGORY_KEYWORDS, UNCLASSIFIED
from app.config import settings
from app.models.contract import Clause


class RuleBasedClassifier:
    """Keyword-overlap classifier over the CUAD-style category taxonomy."""

    def __init__(self) -> None:
        # Pre-lowercase keyword sets.
        self._keywords = {
            cat: [kw.lower() for kw in kws] for cat, kws in CATEGORY_KEYWORDS.items()
        }

    def classify(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return UNCLASSIFIED, 0.0

        lowered = text.lower()
        scores: dict[str, int] = {}
        for category, keywords in self._keywords.items():
            hits = sum(lowered.count(kw) for kw in keywords)
            if hits:
                scores[category] = hits

        if not scores:
            return UNCLASSIFIED, 0.0

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        # Confidence: share of keyword hits captured by the winning category,
        # squashed slightly so a single weak hit isn't reported as certainty.
        confidence = scores[best] / total
        confidence = round(min(1.0, 0.4 + 0.6 * confidence), 3)
        return best, confidence


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

    def __init__(
        self,
        model_name: str | None = None,
        embed_fn: Callable[[str], "torch.Tensor"] | None = None,
    ) -> None:
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


def classify_clauses(clauses: list[Clause], classifier=None) -> list[Clause]:
    """Annotate each clause with a category + confidence (in place)."""
    classifier = classifier or get_classifier()
    for clause in clauses:
        category, confidence = classifier.classify(clause.text)
        clause.category = category
        clause.confidence = confidence
    return clauses

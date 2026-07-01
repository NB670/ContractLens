"""Clause classification.

Two backends are provided:

  * ``RuleBasedClassifier`` (default) — transparent keyword scoring over the CUAD
    category taxonomy. No model download, runs anywhere, and gives us a baseline
    to evaluate the learned model against in later checkpoints.

  * ``LegalBertClassifier`` — optional zero-shot/embedding backend built on
    HuggingFace Transformers + LegalBERT. Imported lazily so the service runs
    without torch/transformers installed.

Both implement ``classify(text) -> (category, confidence)`` and a convenience
``classify_clauses(clauses)`` that annotates Clause objects in place.
"""

from __future__ import annotations

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


def classify_clauses(clauses: list[Clause], classifier=None) -> list[Clause]:
    """Annotate each clause with a category + confidence (in place)."""
    classifier = classifier or get_classifier()
    for clause in clauses:
        category, confidence = classifier.classify(clause.text)
        clause.category = category
        clause.confidence = confidence
    return clauses

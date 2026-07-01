"""Tests for the rule-based clause classifier and the ingestion pipeline."""

from pathlib import Path

import pytest

from app.clauses.classifier import LegalBertClassifier, RuleBasedClassifier, classify_clauses
from app.ingestion.segmenter import segment
from app.pipeline import ingest


def test_rule_based_classifier_basic():
    clf = RuleBasedClassifier()
    cat, conf = clf.classify(
        "Each party shall hold all Confidential Information in strict confidence."
    )
    assert cat == "Confidentiality"
    assert 0.0 < conf <= 1.0


def test_rule_based_classifier_unclassified():
    clf = RuleBasedClassifier()
    cat, conf = clf.classify("The quick brown fox jumps over the lazy dog.")
    assert cat == "Unclassified"
    assert conf == 0.0


def test_classify_clauses_in_place():
    clauses = segment(
        "1. Indemnification\nEach party shall indemnify and hold harmless the other.\n"
        "2. Payment Terms\nFees are due net 30 upon invoice.\n"
    )
    classify_clauses(clauses)
    cats = {c.category for c in clauses}
    assert "Indemnification" in cats
    assert "Payment Terms" in cats


def test_pipeline_ingest_sample_contract():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_contract.txt"
    contract = ingest(sample.name, sample.read_bytes())

    assert contract.metadata.num_clauses > 0
    assert contract.metadata.contract_type == "Non-Disclosure Agreement"

    categories = contract.categories_present()
    # The sample contains these explicitly-headed sections.
    for expected in ("Confidentiality", "Termination", "Intellectual Property"):
        assert expected in categories


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

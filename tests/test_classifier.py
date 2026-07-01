"""Tests for the rule-based clause classifier and the ingestion pipeline."""

from pathlib import Path

from app.clauses.classifier import RuleBasedClassifier, classify_clauses
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

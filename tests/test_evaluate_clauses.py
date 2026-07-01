"""Smoke test for the CUAD evaluation harness's scoring logic.

Uses a tiny inline fixture (not the full committed data/cuad_sample.json)
and the fast RuleBasedClassifier, so this test runs quickly and offline —
it never triggers a LegalBERT download.
"""

from app.clauses.classifier import RuleBasedClassifier
from scripts.evaluate_clauses import score

_FIXTURE = [
    {"category": "Governing Law", "clause_text": "This Agreement shall be governed by the laws of Delaware."},
    {"category": "Governing Law", "clause_text": "This Agreement is governed by the laws of the State of New York."},
    {
        "category": "Assignment",
        "clause_text": "Neither party may assign this Agreement without the other's consent; binding on successors and assigns.",
    },
]


def test_score_reports_per_category_metrics_and_macro_avg():
    results = score(RuleBasedClassifier(), _FIXTURE)

    assert results["Governing Law"]["support"] == 2
    assert results["Governing Law"]["precision"] == 1.0
    assert results["Governing Law"]["recall"] == 1.0
    assert results["Assignment"]["support"] == 1
    assert "macro_avg" in results
    assert results["macro_avg"]["support"] == 3
    assert 0.0 <= results["macro_avg"]["f1"] <= 1.0


def test_score_counts_misclassification_as_false_positive_and_negative():
    fixture = [{"category": "Governing Law", "clause_text": "The quick brown fox jumps over the lazy dog."}]
    results = score(RuleBasedClassifier(), fixture)

    # No keyword match -> classifier predicts "Unclassified", so the true
    # category gets 0 recall and there is no row for "Unclassified" itself
    # (it was never a ground-truth label in this fixture).
    assert results["Governing Law"]["recall"] == 0.0
    assert "Unclassified" not in results

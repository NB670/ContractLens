"""Smoke test for the retrieval evaluation harness's scoring logic.

Uses a tiny inline fixture and the fast HashingEmbedder, so this test runs
quickly and offline -- it never triggers a sentence-transformers download.
"""

from app.retrieval.embedder import HashingEmbedder
from scripts.evaluate_retrieval import evaluate

_FIXTURE = [
    {"category": "Confidentiality", "clause_text": "Each party shall keep all Confidential Information strictly secret."},
    {"category": "Confidentiality", "clause_text": "All Confidential Information disclosed shall remain confidential."},
    {"category": "Payment Terms", "clause_text": "Payment shall be made net thirty days after invoice."},
    {"category": "Payment Terms", "clause_text": "Fees are due within thirty days of receipt of invoice."},
    {"category": "Governing Law", "clause_text": "This Agreement is a singleton category with no other match."},
]


def test_evaluate_reports_recall_success_and_mrr():
    metrics = evaluate(_FIXTURE, HashingEmbedder(), k=5)

    # 4 of 5 records have at least one same-category peer; the singleton
    # Governing Law record is skipped (no ground truth to evaluate against).
    assert metrics["queries"] == 4
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["success_at_k"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_evaluate_on_empty_records_returns_zeroed_metrics():
    metrics = evaluate([], HashingEmbedder(), k=5)
    assert metrics == {"recall_at_k": 0.0, "success_at_k": 0.0, "mrr": 0.0, "queries": 0, "k": 5}

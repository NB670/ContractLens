"""Smoke test for the comparison evaluation harness's synthetic-edit logic.

Uses a small, fully-deterministic inline fixture (8 records, distinct
vocabulary per record) with the fast HashingEmbedder, so this test is
reproducible and never triggers a sentence-transformers download. The exact
parameters (n_base=5, seed=1) were verified by hand to produce a clean
5-clause synthetic pair with one removed, one modified, one added, and three
unchanged clauses -- see the design spec for why the underlying algorithm
(optimal bipartite matching with a rejection threshold) is expected to
recover this perfectly on clearly-distinct clause text.
"""

from app.retrieval.embedder import HashingEmbedder
from scripts.evaluate_comparison import build_synthetic_pair, evaluate

_FIXTURE = [
    {"category": "Confidentiality", "clause_text": "Each party shall keep all Confidential Information strictly secret and confidential at all times."},
    {"category": "Payment Terms", "clause_text": "Payment shall be made net thirty days after receipt of invoice."},
    {"category": "Governing Law", "clause_text": "This Agreement shall be governed by the laws of the State of Delaware."},
    {"category": "Termination", "clause_text": "Either party may terminate this Agreement upon thirty days written notice."},
    {"category": "Warranty", "clause_text": "Vendor warrants that the Products will conform to the specifications in Exhibit A."},
    {"category": "Assignment", "clause_text": "Neither party may assign this Agreement without the prior written consent of the other."},
    {"category": "Intellectual Property", "clause_text": "All intellectual property rights in the deliverables shall vest in the Client."},
    {"category": "Liability", "clause_text": "Neither party shall be liable for any indirect or consequential damages."},
]


def test_build_synthetic_pair_produces_expected_counts():
    base, revised, ground_truth, n_added = build_synthetic_pair(
        _FIXTURE, n_base=5, n_removed=1, n_modified=1, n_added=1, seed=1
    )

    assert len(base.clauses) == 5
    assert len(ground_truth) == 5
    assert n_added == 1
    assert list(ground_truth.values()).count("removed") == 1
    assert list(ground_truth.values()).count("modified") == 1
    assert list(ground_truth.values()).count("unchanged") == 3


def test_evaluate_recovers_all_synthetic_edits_correctly():
    results = evaluate(
        _FIXTURE, HashingEmbedder(), seed=1,
        n_base=5, n_removed=1, n_modified=1, n_added=1,
    )

    assert results["removed"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["unchanged"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["modified"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert results["added"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

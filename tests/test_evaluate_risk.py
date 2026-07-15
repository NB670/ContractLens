"""Smoke test for the risk rule precision-eval harness's scoring logic."""

from scripts.evaluate_risk import score

_FIXTURE = [
    {
        "rule_id": "termination.auto_renewal",
        "clause_text": "This Agreement shall automatically renew for successive one-year terms.",
        "should_fire": True,
        "source": "representative",
    },
    {
        "rule_id": "termination.auto_renewal",
        "clause_text": "This Agreement shall be governed by the laws of the State of California.",
        "should_fire": False,
        "source": "representative",
    },
]


def test_score_reports_precision_and_support():
    results = score(_FIXTURE)

    assert results["termination.auto_renewal"]["support"] == 2
    assert results["termination.auto_renewal"]["tp"] == 1
    assert results["termination.auto_renewal"]["fp"] == 0
    assert results["termination.auto_renewal"]["precision"] == 1.0
    assert "macro_avg" in results


def test_score_counts_false_positive_when_rule_fires_on_should_not_fire_clause():
    fixture = [
        {
            "rule_id": "warranty.disclaimed",
            "clause_text": "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND.",
            "should_fire": False,
            "source": "representative",
        }
    ]
    results = score(fixture)
    assert results["warranty.disclaimed"]["fp"] == 1
    assert results["warranty.disclaimed"]["precision"] == 0.0

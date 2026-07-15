"""Tests for the evidence-backed risk analysis rule engine."""

from app.models.contract import Clause, Contract
from app.risk.analyzer import analyze_risk


def _clause(index: int, text: str, category: str = "Unclassified", start_offset: int = 0) -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=start_offset, end_offset=start_offset + len(text),
    )


def test_uncapped_liability_fires():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "In no event shall either party's liability be capped or limited; each party shall bear unlimited liability for any breach.", "Liability"),
    ])
    report = analyze_risk(contract)
    assert "liability.uncapped" in {f.rule_id for f in report.findings}


def test_uncapped_liability_suppressed_by_cap_language():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Notwithstanding unlimited liability language elsewhere, the aggregate liability shall not exceed the fees paid.", "Liability"),
    ])
    report = analyze_risk(contract)
    assert "liability.uncapped" not in {f.rule_id for f in report.findings}


def test_indemnification_broad_fires():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Each party agrees to defend, indemnify, and hold harmless the other party.", "Indemnification"),
    ])
    report = analyze_risk(contract)
    assert "indemnification.broad" in {f.rule_id for f in report.findings}


def test_missing_clause_rules_fire_when_categories_absent():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "This is a generic clause about scheduling.", "Payment Terms"),
    ])
    report = analyze_risk(contract)
    rule_ids = {f.rule_id for f in report.findings}
    assert "missing.governing_law" in rule_ids
    assert "missing.liability_cap" in rule_ids
    assert "missing.confidentiality" in rule_ids


def test_evidence_includes_offsets_and_excerpt():
    text = "This Agreement shall automatically renew for successive one-year terms."
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, text, "Termination", start_offset=100),
    ])
    report = analyze_risk(contract)
    finding = next(f for f in report.findings if f.rule_id == "termination.auto_renewal")
    assert finding.start_offset >= 100
    assert "renew" in finding.evidence_text.lower()


def test_overall_score_increases_with_more_high_severity_findings():
    quiet = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "This is a routine administrative clause with no risk language.", "Payment Terms"),
    ])
    risky = Contract(id="c2", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Liability is unlimited and uncapped. Each party agrees to defend, indemnify, and hold harmless the other for any and all claims.", "Liability"),
    ])
    assert analyze_risk(risky).overall_score > analyze_risk(quiet).overall_score
    assert analyze_risk(risky).risk_level == "high"


def test_findings_sorted_high_severity_first():
    contract = Contract(id="c1", filename="t.txt", source_format="txt", clauses=[
        _clause(0, "Fees paid hereunder are non-refundable.", "Payment Terms"),
        _clause(1, "Liability is unlimited and uncapped for any breach.", "Liability"),
    ])
    report = analyze_risk(contract)
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])

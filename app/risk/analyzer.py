"""Evidence-backed risk analysis (Checkpoint 3).

Scans a structured ``Contract`` (the CP2 pipeline output) and flags common
contract-review risks. Every ``RiskFinding`` carries its supporting evidence:
the clause it fired on, that clause's character offsets into the source
document, and an excerpt around the triggering phrase.

The rule set is transparent and deterministic (regex triggers keyed off the
CP2 category taxonomy in ``app/clauses/categories.py``) -- no model download,
no black-box score. Two rule types:
  * clause-level rules -- fire per clause, cite that clause as evidence.
  * contract-level rules -- fire on the absence of an expected protective
    clause (no limitation-of-liability, no governing-law, ...), cited
    against the whole document. These contribute to the overall risk score
    based purely on category absence; ``scripts/evaluate_risk.py``'s
    precision evaluation only measures the 12 clause-level trigger rules, so
    this missing-clause contribution is not covered by that eval.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from app.models.analysis import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_WEIGHT,
    RiskFinding,
    RiskReport,
)
from app.models.contract import Clause, Contract


@dataclass(frozen=True)
class _ClauseRule:
    rule_id: str
    category: str
    severity: str
    rationale: str
    trigger: re.Pattern[str]
    suppressed_by: Optional[re.Pattern[str]] = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_CLAUSE_RULES: list[_ClauseRule] = [
    _ClauseRule(
        rule_id="liability.uncapped",
        category="Liability",
        severity=SEVERITY_HIGH,
        rationale="Liability appears uncapped/unlimited, exposing the party to open-ended damages.",
        trigger=_rx(r"unlimited liability|uncapped|without limitation of liability|no limit(?:ation)? on (?:its )?liability"),
        suppressed_by=_rx(r"limitation of liability|liability (?:is|shall be) (?:limited|capped)|aggregate liability (?:shall|will) not exceed"),
    ),
    _ClauseRule(
        rule_id="indemnification.broad",
        category="Indemnification",
        severity=SEVERITY_HIGH,
        rationale="Broad 'any and all' indemnification obligation; scope of indemnity may be wider than intended.",
        trigger=_rx(r"indemnif\w+.{0,60}any and all|defend,? indemnify,? and hold harmless"),
    ),
    _ClauseRule(
        rule_id="termination.for_convenience",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="Counterparty may terminate for convenience / without cause, reducing commitment certainty.",
        trigger=_rx(r"terminate.{0,40}for convenience|terminate.{0,40}without cause|terminate.{0,40}(?:sole|its own) discretion"),
    ),
    _ClauseRule(
        rule_id="termination.auto_renewal",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="Automatic renewal / evergreen term; the agreement extends unless affirmatively cancelled.",
        trigger=_rx(r"automatically renew|auto-?renew|evergreen|renew(?:s|ed)? for successive"),
    ),
    _ClauseRule(
        rule_id="ip.assignment",
        category="Intellectual Property",
        severity=SEVERITY_MEDIUM,
        rationale="Assigns ownership of IP / work product; verify this matches the intended IP allocation.",
        trigger=_rx(r"assigns? all right,? title,? and interest|work (?:made )?for hire|hereby assigns"),
    ),
    _ClauseRule(
        rule_id="warranty.disclaimed",
        category="Warranty",
        severity=SEVERITY_MEDIUM,
        rationale="Warranties are disclaimed / goods provided 'as is'; limited recourse for defects.",
        trigger=_rx(r"\bas is\b|disclaims? all warrant|no warrant(?:y|ies)|without warrant(?:y|ies) of any kind"),
    ),
    _ClauseRule(
        rule_id="assignment.without_consent",
        category="Assignment",
        severity=SEVERITY_MEDIUM,
        rationale="Counterparty may assign the agreement without consent; the other party could change unexpectedly.",
        trigger=_rx(r"assign\w*.{0,170}without.{0,40}(?:the )?(?:prior )?(?:express )?(?:written )?(?:consent|approval)"),
    ),
    _ClauseRule(
        rule_id="payment.non_refundable",
        category="Payment Terms",
        severity=SEVERITY_LOW,
        rationale="Fees are non-refundable and/or accrue late-payment interest.",
        trigger=_rx(r"non-?refundable|late (?:fee|charge|payment)|interest (?:of|at) \d"),
    ),
    _ClauseRule(
        rule_id="confidentiality.perpetual",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="Confidentiality obligations are perpetual / survive indefinitely.",
        trigger=_rx(r"perpetu\w+|survive indefinitely|in perpetuity|no expir\w+"),
    ),
    _ClauseRule(
        rule_id="dispute.mandatory_arbitration",
        category="Governing Law",
        severity=SEVERITY_MEDIUM,
        rationale="Disputes must go through binding arbitration, foreclosing the option to litigate in court.",
        trigger=_rx(r"binding arbitration|shall be resolved (?:solely )?by arbitration|submit to arbitration"),
    ),
    _ClauseRule(
        rule_id="amendment.unilateral",
        category="Termination",
        severity=SEVERITY_MEDIUM,
        rationale="One party may amend or modify the agreement's terms unilaterally, without the other party's consent.",
        trigger=_rx(r"may (?:amend|modify) this agreement.{0,40}(?:in its sole discretion|without (?:the )?(?:other party'?s )?consent)"),
    ),
    _ClauseRule(
        rule_id="confidentiality.non_mutual",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="Confidentiality obligations appear to run one way (protecting only the disclosing party), not mutually.",
        trigger=_rx(r"receiving party shall not disclose|recipient shall (?:keep|maintain) confidential"),
        suppressed_by=_rx(r"mutual(?:ly)? confidential|each party (?:shall|agrees to) (?:keep|maintain|hold)"),
    ),
]


@dataclass(frozen=True)
class _MissingClauseRule:
    rule_id: str
    category: str
    severity: str
    rationale: str


_MISSING_CLAUSE_RULES: list[_MissingClauseRule] = [
    _MissingClauseRule(
        rule_id="missing.liability_cap",
        category="Liability",
        severity=SEVERITY_MEDIUM,
        rationale="No limitation-of-liability clause detected; liability may be unbounded by default.",
    ),
    _MissingClauseRule(
        rule_id="missing.governing_law",
        category="Governing Law",
        severity=SEVERITY_LOW,
        rationale="No governing-law/jurisdiction clause detected; the forum for disputes is unspecified.",
    ),
    _MissingClauseRule(
        rule_id="missing.confidentiality",
        category="Confidentiality",
        severity=SEVERITY_LOW,
        rationale="No confidentiality clause detected; shared information may be unprotected.",
    ),
]


def _first_match_excerpt(pattern: re.Pattern[str], clause: Clause) -> tuple[str, int, int]:
    """Return a (excerpt, start_offset, end_offset) window around the trigger."""
    match = pattern.search(clause.text)
    if match is None:
        return clause.text[:240], clause.start_offset, clause.end_offset
    lo = max(0, match.start() - 60)
    hi = min(len(clause.text), match.end() + 60)
    excerpt = clause.text[lo:hi].strip()
    return excerpt, clause.start_offset + lo, clause.start_offset + hi


def analyze_risk(contract: Contract) -> RiskReport:
    """Produce an evidence-backed ``RiskReport`` for ``contract``."""
    findings: list[RiskFinding] = []

    for clause in contract.clauses:
        text = clause.text or ""
        for rule in _CLAUSE_RULES:
            if not rule.trigger.search(text):
                continue
            if rule.suppressed_by is not None and rule.suppressed_by.search(text):
                continue
            excerpt, start, end = _first_match_excerpt(rule.trigger, clause)
            findings.append(
                RiskFinding(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    rationale=rule.rationale,
                    clause_index=clause.index,
                    evidence_text=excerpt,
                    start_offset=start,
                    end_offset=end,
                )
            )

    present = set(contract.categories_present())
    for miss in _MISSING_CLAUSE_RULES:
        if miss.category not in present:
            findings.append(
                RiskFinding(
                    rule_id=miss.rule_id,
                    category=miss.category,
                    severity=miss.severity,
                    rationale=miss.rationale,
                    clause_index=None,
                    evidence_text=f"No clause classified as '{miss.category}' was found in this contract.",
                    start_offset=0,
                    end_offset=0,
                )
            )

    severity_counts: dict[str, int] = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 0, SEVERITY_HIGH: 0}
    weight_total = 0
    for finding in findings:
        severity_counts[finding.severity] += 1
        weight_total += SEVERITY_WEIGHT[finding.severity]

    overall_score = round(100.0 * (1.0 - math.exp(-weight_total / 6.0)), 1)
    if overall_score >= 60.0:
        risk_level = SEVERITY_HIGH
    elif overall_score >= 25.0:
        risk_level = SEVERITY_MEDIUM
    else:
        risk_level = SEVERITY_LOW

    _SEV_ORDER = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    findings.sort(key=lambda f: (_SEV_ORDER[f.severity], f.clause_index if f.clause_index is not None else 1_000_000))

    return RiskReport(
        contract_id=contract.id,
        overall_score=overall_score,
        risk_level=risk_level,
        findings=findings,
        severity_counts=severity_counts,
    )

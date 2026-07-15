"""Score the risk rule engine's precision against a hand-labeled fixture.

Run: python -m scripts.evaluate_risk

Loads data/risk_eval_labels.json -- a hand-labeled set of 30 clauses across 6
representative rules, each labeled with whether that specific rule should
fire on that clause text (see the file's construction notes and Task 6 of
docs/superpowers/plans/2026-07-14-cp3-features-implementation.md).

This measures **precision of firing only** -- for each rule, of the clauses
labeled "should fire," how many of the engine's actual firings were correct
-- not recall across the full space of risky clauses in real contracts,
which isn't something a 30-record fixture can support. This limitation is
intentional and stated, not glossed over, in the same spirit as the CUAD
category-gap limitation documented in scripts/evaluate_clauses.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.models.contract import Clause, Contract
from app.risk.analyzer import analyze_risk

LABELS_PATH = Path(__file__).resolve().parents[1] / "data" / "risk_eval_labels.json"


def load_labels(path: Path = LABELS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _single_clause_contract(text: str, category: str = "Unclassified") -> Contract:
    clause = Clause(
        index=0, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )
    return Contract(id="eval", filename="eval.txt", source_format="txt", clauses=[clause])


def score(labels: list[dict]) -> dict[str, dict[str, float]]:
    """Return {rule_id: {precision, tp, fp, fn, support}} plus a "macro_avg" row."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for record in labels:
        rule_id = record["rule_id"]
        contract = _single_clause_contract(record["clause_text"])
        report = analyze_risk(contract)
        fired = any(f.rule_id == rule_id for f in report.findings)
        support[rule_id] += 1

        if record["should_fire"]:
            if fired:
                tp[rule_id] += 1
            else:
                fn[rule_id] += 1
        elif fired:
            fp[rule_id] += 1

    rule_ids = sorted(support)
    results: dict[str, dict[str, float]] = {}
    for rule_id in rule_ids:
        firings = tp[rule_id] + fp[rule_id]
        precision = tp[rule_id] / firings if firings else 0.0
        results[rule_id] = {
            "precision": round(precision, 3),
            "tp": tp[rule_id],
            "fp": fp[rule_id],
            "fn": fn[rule_id],
            "support": support[rule_id],
        }

    macro_precision = sum(r["precision"] for r in results.values()) / len(results)
    results["macro_avg"] = {
        "precision": round(macro_precision, 3),
        "tp": sum(tp.values()),
        "fp": sum(fp.values()),
        "fn": sum(fn.values()),
        "support": sum(support.values()),
    }
    return results


def print_report(results: dict[str, dict[str, float]]) -> None:
    print(f"{'rule_id':<32}{'precision':>10}{'tp':>6}{'fp':>6}{'fn':>6}{'support':>9}")
    for rule_id, metrics in results.items():
        print(
            f"{rule_id:<32}{metrics['precision']:>10.3f}{metrics['tp']:>6}"
            f"{metrics['fp']:>6}{metrics['fn']:>6}{metrics['support']:>9}"
        )


def main() -> None:
    labels = load_labels()
    results = score(labels)
    print("=== risk rule precision (hand-labeled fixture) ===")
    print_report(results)


if __name__ == "__main__":
    main()

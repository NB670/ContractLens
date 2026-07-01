"""Score clause classifiers against the committed CUAD-derived sample.

Run: python -m scripts.evaluate_clauses [--backend rule|legalbert|both]

Loads data/cuad_sample.json (see scripts/generate_cuad_sample.py for how it
was produced) and reports per-category + macro-averaged precision, recall,
and F1 for the selected classifier backend(s).

Confidentiality, Indemnification, and Force Majeure are not part of CUAD's
original 41 categories, so they are absent from the sample and are not
scored here -- this is a known, stated limitation of evaluating against
CUAD, not a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from app.clauses.classifier import LegalBertClassifier, RuleBasedClassifier

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def score(classifier, records: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Return {category: {precision, recall, f1, support}} plus a "macro_avg" row."""
    true_positives: dict[str, int] = defaultdict(int)
    false_positives: dict[str, int] = defaultdict(int)
    false_negatives: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for record in records:
        expected = record["category"]
        predicted, _confidence = classifier.classify(record["clause_text"])
        support[expected] += 1
        if predicted == expected:
            true_positives[expected] += 1
        else:
            false_negatives[expected] += 1
            false_positives[predicted] += 1

    categories = sorted(support)
    results: dict[str, dict[str, float]] = {}
    for category in categories:
        tp = true_positives[category]
        fp = false_positives[category]
        fn = false_negatives[category]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results[category] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support[category],
        }

    macro_precision = sum(r["precision"] for r in results.values()) / len(results)
    macro_recall = sum(r["recall"] for r in results.values()) / len(results)
    macro_f1 = sum(r["f1"] for r in results.values()) / len(results)
    results["macro_avg"] = {
        "precision": round(macro_precision, 3),
        "recall": round(macro_recall, 3),
        "f1": round(macro_f1, 3),
        "support": sum(support.values()),
    }
    return results


def print_report(name: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'category':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for category, metrics in results.items():
        print(
            f"{category:<24}{metrics['precision']:>10.3f}{metrics['recall']:>10.3f}"
            f"{metrics['f1']:>10.3f}{metrics['support']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["rule", "legalbert", "both"], default="both")
    args = parser.parse_args()

    records = load_sample()

    if args.backend in ("rule", "both"):
        print_report("RuleBasedClassifier", score(RuleBasedClassifier(), records))
    if args.backend in ("legalbert", "both"):
        legalbert_classifier = LegalBertClassifier()
        if not legalbert_classifier._ensure_embed_fn():
            print(
                "WARNING: LegalBERT embedding backend unavailable — results below "
                "are the RuleBasedClassifier fallback, not a real LegalBERT run.",
                file=sys.stderr,
            )
        print_report("LegalBertClassifier", score(legalbert_classifier, records))


if __name__ == "__main__":
    main()

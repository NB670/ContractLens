"""Consolidated Checkpoint 4 benchmark runner.

Run: python -m scripts.run_benchmarks [--classifier rule|legalbert|legalbert-finetuned]
                                      [--retrieval-backend hashing|sentence]
                                      [--k 5] [--limit N] [--json]
                                      [--out results/benchmark_<timestamp>.json]

The CP4 milestone is to "evaluate classification and retrieval performance
with benchmark datasets." Rather than duplicate any scoring logic, this
driver reuses the harnesses already committed in Checkpoint 2/3 --
scripts/evaluate_clauses.py (classification P/R/F1) and
scripts/evaluate_retrieval.py (Recall@K / Success@K / MRR) -- over the same
committed CUAD sample (data/cuad_sample.json), and writes one combined
report. By default every run is also written to results/ as a timestamped
JSON file (--out), so headline numbers are reproducible directly from the
repository rather than living only in report screenshots -- the concrete
suggestion from the Checkpoint 3 TA feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.clauses.classifier import (
    FineTunedLegalBertClassifier,
    LegalBertClassifier,
    RuleBasedClassifier,
)
from app.retrieval.embedder import HashingEmbedder, SentenceTransformerEmbedder
from scripts.evaluate_clauses import load_sample
from scripts.evaluate_clauses import score as score_classification
from scripts.evaluate_retrieval import evaluate as evaluate_retrieval

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _make_classifier(name: str):
    if name == "legalbert":
        clf = LegalBertClassifier()
        if not clf._ensure_embed_fn():
            print(
                "WARNING: LegalBERT embedding backend unavailable — classification "
                "numbers are the RuleBasedClassifier fallback.",
                file=sys.stderr,
            )
        return clf
    if name == "legalbert-finetuned":
        clf = FineTunedLegalBertClassifier()
        if not clf._ensure_pipe():
            print(
                "WARNING: fine-tuned LegalBERT weights not found (run "
                "`python -m scripts.finetune_legalbert` first) — classification "
                "numbers are the RuleBasedClassifier fallback.",
                file=sys.stderr,
            )
        return clf
    return RuleBasedClassifier()


def run(
    classifier_name: str = "rule",
    retrieval_backend: str = "hashing",
    k: int = 5,
    limit: int | None = None,
) -> dict:
    """Run both benchmarks over the CUAD sample and return a combined summary."""
    records = load_sample()
    if limit is not None:
        records = records[:limit]

    classification = score_classification(_make_classifier(classifier_name), records)

    embedder = (
        SentenceTransformerEmbedder() if retrieval_backend == "sentence" else HashingEmbedder()
    )
    retrieval = evaluate_retrieval(records, embedder, k=k)

    return {
        "dataset": "cuad_sample.json",
        "records_scored": len(records),
        "classification": {
            "backend": classifier_name,
            "macro_avg": classification["macro_avg"],
            "per_category": {c: m for c, m in classification.items() if c != "macro_avg"},
        },
        "retrieval": {"backend": retrieval_backend, **retrieval},
    }


def print_report(summary: dict) -> None:
    print("=" * 60)
    print("ContractLens benchmark report")
    print(f"dataset          : {summary['dataset']} ({summary['records_scored']} clauses)")
    print("-" * 60)
    clf = summary["classification"]
    macro = clf["macro_avg"]
    print(f"Clause classification (backend={clf['backend']})")
    print(f"  macro precision: {macro['precision']:.3f}")
    print(f"  macro recall   : {macro['recall']:.3f}")
    print(f"  macro F1       : {macro['f1']:.3f}")
    print("-" * 60)
    ret = summary["retrieval"]
    print(f"Semantic retrieval (backend={ret['backend']})")
    print(f"  queries        : {ret['queries']}")
    print(f"  Recall@{ret['k']}      : {ret['recall_at_k']:.4f}")
    print(f"  Success@{ret['k']}     : {ret['success_at_k']:.4f}")
    print(f"  MRR            : {ret['mrr']:.4f}")
    print("=" * 60)


def write_results(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def _default_out_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"benchmark_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier", choices=["rule", "legalbert", "legalbert-finetuned"], default="rule"
    )
    parser.add_argument("--retrieval-backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N clauses (smoke run).")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON instead of the report.")
    parser.add_argument(
        "--out", type=str, default="__default__",
        help="Write the summary JSON to this path (default: results/benchmark_<timestamp>.json). "
             "Pass an empty string to skip writing.",
    )
    args = parser.parse_args()

    summary = run(
        classifier_name=args.classifier,
        retrieval_backend=args.retrieval_backend,
        k=args.k,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)

    if args.out != "":
        out_path = Path(args.out) if args.out != "__default__" else _default_out_path()
        write_results(summary, out_path)
        print(f"\nWrote results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

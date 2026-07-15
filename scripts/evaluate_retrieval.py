"""Evaluate semantic clause retrieval against the committed CUAD sample.

Run: python -m scripts.evaluate_retrieval [--k 5] [--backend hashing|sentence]
                                          [--limit N]

Reuses data/cuad_sample.json (the same fixture scripts/evaluate_clauses.py
scores classification on). Each clause is used in turn as a query; the other
clauses sharing its CUAD category are the ground-truth relevant set. Every
clause is embedded once with the configured embedder and ranked by cosine
similarity against every other clause, then scored with the retrieval
metrics named in the project plan:

  * Recall@K   -- fraction of a query's relevant clauses that land in the top K
  * Success@K  -- fraction of queries with at least one relevant clause in top K
  * MRR        -- mean reciprocal rank of the first relevant clause

The default backend ("hashing" -> HashingEmbedder) needs no model download,
so this runs offline; pass --backend sentence to score the semantic
sentence-transformers encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval.embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
)

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def evaluate(records: list[dict], embedder, k: int = 5) -> dict[str, float]:
    """Return {'recall_at_k', 'success_at_k', 'mrr', 'queries', 'k'}."""
    texts = [r["clause_text"] for r in records]
    categories = [r["category"] for r in records]
    vectors = [embedder.embed(t) for t in texts]
    n = len(records)

    recall_sum = 0.0
    success_sum = 0.0
    rr_sum = 0.0
    evaluated = 0

    for i in range(n):
        relevant = [j for j in range(n) if j != i and categories[j] == categories[i]]
        if not relevant:
            continue
        evaluated += 1

        ranked = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: cosine_similarity(vectors[i], vectors[j]),
            reverse=True,
        )

        top_k = ranked[:k]
        relevant_set = set(relevant)
        hits_in_top_k = sum(1 for j in top_k if j in relevant_set)

        recall_sum += hits_in_top_k / min(k, len(relevant))
        success_sum += 1.0 if hits_in_top_k else 0.0

        for rank, j in enumerate(ranked, start=1):
            if j in relevant_set:
                rr_sum += 1.0 / rank
                break

    if evaluated == 0:
        return {"recall_at_k": 0.0, "success_at_k": 0.0, "mrr": 0.0, "queries": 0, "k": k}

    return {
        "recall_at_k": round(recall_sum / evaluated, 4),
        "success_at_k": round(success_sum / evaluated, 4),
        "mrr": round(rr_sum / evaluated, 4),
        "queries": evaluated,
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N sample records (faster smoke run).",
    )
    args = parser.parse_args()

    records = load_sample()
    if args.limit is not None:
        records = records[: args.limit]

    embedder = SentenceTransformerEmbedder() if args.backend == "sentence" else HashingEmbedder()
    metrics = evaluate(records, embedder, k=args.k)

    print(f"=== retrieval eval (backend={args.backend}) ===")
    print(f"queries evaluated : {metrics['queries']}")
    print(f"Recall@{metrics['k']:<11}: {metrics['recall_at_k']:.4f}")
    print(f"Success@{metrics['k']:<10}: {metrics['success_at_k']:.4f}")
    print(f"MRR{'':<15}: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()

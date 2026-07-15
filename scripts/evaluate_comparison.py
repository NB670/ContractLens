"""Evaluate contract comparison against synthetic, ground-truth-labeled edits.

Run: python -m scripts.evaluate_comparison [--backend hashing|sentence] [--seed 42]

There is no public "labeled contract diff" dataset, so this generates its own
ground truth: starting from real clauses in the committed CUAD sample
(data/cuad_sample.json), it builds a synthetic "revised" contract from a
"base" contract via controlled edits -- delete clauses (expect "removed"),
insert clauses from elsewhere in the corpus (expect "added"), lightly edit
clauses by inserting a fixed marker sentence (expect "modified"), and leave
the rest untouched (expect "unchanged") -- then runs compare_contracts and
reports precision/recall/F1 per change type against the known ground truth.
This is the comparison analogue of scripts/evaluate_clauses.py and
scripts/evaluate_retrieval.py; the reference scaffold has no comparison eval
at all.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.comparison.comparator import compare_contracts
from app.models.analysis import CHANGE_ADDED, CHANGE_MODIFIED, CHANGE_REMOVED, CHANGE_UNCHANGED
from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder, SentenceTransformerEmbedder

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"

_MODIFICATION_INSERT = " Notwithstanding the foregoing, this provision is subject to Section 12."


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _make_clause(index: int, text: str, category: str) -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def build_synthetic_pair(
    records: list[dict],
    n_base: int = 20,
    n_removed: int = 4,
    n_modified: int = 4,
    n_added: int = 4,
    seed: int = 42,
) -> tuple[Contract, Contract, dict[int, str], int]:
    """Build a (base, revised, ground_truth, n_added) tuple.

    ``ground_truth`` maps each base clause's index to the change type it
    should be detected as ("removed", "modified", or "unchanged"). Added
    clauses have no base index, so their expected count is returned
    separately as ``n_added``.
    """
    rng = random.Random(seed)
    pool = rng.sample(records, n_base + n_added)
    base_records, extra_records = pool[:n_base], pool[n_base:]

    base_clauses = [
        _make_clause(i, r["clause_text"], r["category"]) for i, r in enumerate(base_records)
    ]
    base = Contract(id="base", filename="base.txt", source_format="txt", clauses=base_clauses)

    removed_indices = set(rng.sample(range(n_base), n_removed))
    remaining = [i for i in range(n_base) if i not in removed_indices]
    modified_indices = set(rng.sample(remaining, n_modified))

    ground_truth: dict[int, str] = {}
    revised_clauses: list[Clause] = []
    next_index = 0
    for i, clause in enumerate(base_clauses):
        if i in removed_indices:
            ground_truth[i] = CHANGE_REMOVED
            continue
        text = clause.text + _MODIFICATION_INSERT if i in modified_indices else clause.text
        ground_truth[i] = CHANGE_MODIFIED if i in modified_indices else CHANGE_UNCHANGED
        revised_clauses.append(_make_clause(next_index, text, clause.category))
        next_index += 1

    for r in extra_records:
        revised_clauses.append(_make_clause(next_index, r["clause_text"], r["category"]))
        next_index += 1

    revised = Contract(
        id="revised", filename="revised.txt", source_format="txt", clauses=revised_clauses
    )
    return base, revised, ground_truth, n_added


def evaluate(
    records: list[dict],
    embedder,
    seed: int = 42,
    n_base: int = 20,
    n_removed: int = 4,
    n_modified: int = 4,
    n_added: int = 4,
) -> dict[str, dict[str, float]]:
    base, revised, ground_truth, expected_added = build_synthetic_pair(
        records, n_base=n_base, n_removed=n_removed, n_modified=n_modified,
        n_added=n_added, seed=seed,
    )
    diff = compare_contracts(base, revised, embedder=embedder)

    predicted: dict[int, str] = {}
    added_predicted = 0
    for change in diff.changes:
        if change.change_type == CHANGE_ADDED:
            added_predicted += 1
        elif change.base_index is not None:
            predicted[change.base_index] = change.change_type

    counts: dict[str, dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in (CHANGE_ADDED, CHANGE_REMOVED, CHANGE_MODIFIED, CHANGE_UNCHANGED)
    }

    for base_index, expected in ground_truth.items():
        actual = predicted.get(base_index)
        if actual == expected:
            counts[expected]["tp"] += 1
        else:
            counts[expected]["fn"] += 1
            if actual is not None:
                counts[actual]["fp"] += 1

    counts[CHANGE_ADDED]["tp"] = min(added_predicted, expected_added)
    counts[CHANGE_ADDED]["fn"] = max(0, expected_added - added_predicted)
    counts[CHANGE_ADDED]["fp"] = max(0, added_predicted - expected_added)

    results: dict[str, dict[str, float]] = {}
    for change_type, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[change_type] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-base", type=int, default=20)
    parser.add_argument("--n-removed", type=int, default=4)
    parser.add_argument("--n-modified", type=int, default=4)
    parser.add_argument("--n-added", type=int, default=4)
    args = parser.parse_args()

    records = load_sample()
    embedder = SentenceTransformerEmbedder() if args.backend == "sentence" else HashingEmbedder()
    results = evaluate(
        records, embedder, seed=args.seed, n_base=args.n_base,
        n_removed=args.n_removed, n_modified=args.n_modified, n_added=args.n_added,
    )

    print(f"=== comparison eval (backend={args.backend}, seed={args.seed}) ===")
    print(f"{'change_type':<12}{'precision':>10}{'recall':>10}{'f1':>10}")
    for change_type, metrics in results.items():
        print(f"{change_type:<12}{metrics['precision']:>10.3f}{metrics['recall']:>10.3f}{metrics['f1']:>10.3f}")


if __name__ == "__main__":
    main()

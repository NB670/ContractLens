"""One-time generator for data/cuad_sample.json.

Downloads labeled clauses from `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification`
(a parquet mirror of the original CUAD clause-classification data, CC-BY-4.0,
publicly readable, no auth required) and writes a stratified sample mapped
onto our 10-category taxonomy (app/clauses/categories.py) to
data/cuad_sample.json. That file is committed to the repo so
scripts/evaluate_clauses.py can score classifiers against real CUAD
annotations without needing network access or the `datasets` package on
every run.

CUAD's original 41 categories do not include "Confidentiality",
"Indemnification", or "Force Majeure" -- those 3 of our 10 taxonomy
categories have no CUAD ground truth and are intentionally excluded here.

Usage (run once; re-run only if you want to refresh the sample):
    python -m scripts.generate_cuad_sample
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cuad_sample.json"
SAMPLES_PER_CATEGORY = 50

# Maps our 10-category taxonomy (app/clauses/categories.py) onto CUAD's
# original 41 clause labels. Confidentiality, Indemnification, and Force
# Majeure have no CUAD equivalent and are intentionally omitted -- there is
# no entry for them here.
CATEGORY_LABEL_MAP: dict[str, list[str]] = {
    "Termination": ["Termination For Convenience"],
    "Liability": ["Cap On Liability", "Uncapped Liability"],
    "Intellectual Property": [
        "License Grant",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "Non-Transferable License",
        "Irrevocable Or Perpetual License",
        "Unlimited/All-You-Can-Eat-License",
        "Affiliate License-Licensee",
        "Affiliate License-Licensor",
    ],
    "Governing Law": ["Governing Law"],
    "Payment Terms": ["Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions"],
    "Warranty": ["Warranty Duration"],
    "Assignment": ["Anti-Assignment"],
}


def generate() -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
        split="train",
    ).shuffle(seed=42)

    label_to_category = {
        label: category
        for category, labels in CATEGORY_LABEL_MAP.items()
        for label in labels
    }

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        category = label_to_category.get(row["label"])
        if category is None or len(by_category[category]) >= SAMPLES_PER_CATEGORY:
            continue
        clause_text = row["clause"].strip()
        if clause_text:
            by_category[category].append(
                {
                    "category": category,
                    "cuad_label": row["label"],
                    "contract_file": row["file_name"],
                    "clause_text": clause_text,
                }
            )

    records = [record for records in by_category.values() for record in records]
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(records, indent=2))

    print(f"Wrote {len(records)} records to {SAMPLE_PATH}")
    for category in CATEGORY_LABEL_MAP:
        print(f"  {category}: {len(by_category.get(category, []))}")


if __name__ == "__main__":
    generate()

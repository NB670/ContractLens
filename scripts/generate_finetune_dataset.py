"""One-time generator for the LegalBERT fine-tuning train/val split.

Pulls MORE labeled clauses from the same source dataset
`generate_cuad_sample.py` uses, reusing its category->CUAD-label mapping, but
explicitly excludes any clause already present in the committed
`data/cuad_sample.json` (the fixture `scripts/evaluate_clauses.py` and
`scripts/run_benchmarks.py` already use as the held-out benchmark) -- so
fine-tuning never trains on the same clauses the benchmark scores against.
That keeps the eventual before/after comparison
(`python -m scripts.run_benchmarks --classifier legalbert` vs.
`--classifier legalbert-finetuned`, both against `data/cuad_sample.json`)
methodologically clean.

Only the 7 categories with real CUAD ground truth (see
`generate_cuad_sample.CATEGORY_LABEL_MAP`) are covered -- Confidentiality,
Indemnification, and Force Majeure have no CUAD equivalent, same limitation
`data/cuad_sample.json` already has. The fine-tuned classifier is therefore
scoped to those 7 categories; it does not replace the rule-based/zero-shot
classifiers' broader (but less accurate) coverage.

Output is deliberately split and written to two new files (not touching
`data/cuad_sample.json`), sized (train=80/category, val=20/category) to fit
comfortably within the smallest category's available pool (Warranty Duration,
164 total in the source dataset, 50 already used by cuad_sample.json ->
114 remaining) while keeping every category balanced.

Both output files are meant to be committed to git, so a `git pull` on a
PACE compute node (no internet access) is all that's needed to fine-tune --
no network access required on PACE itself.

Usage (run once, locally, where network + `datasets` are available):
    python -m scripts.generate_finetune_dataset
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.generate_cuad_sample import CATEGORY_LABEL_MAP, SAMPLE_PATH

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_PATH = DATA_DIR / "legalbert_finetune_train.json"
VAL_PATH = DATA_DIR / "legalbert_finetune_val.json"

TRAIN_PER_CATEGORY = 80
VAL_PER_CATEGORY = 20


def generate() -> None:
    from datasets import load_dataset

    existing = json.loads(SAMPLE_PATH.read_text())
    already_used = {record["clause_text"] for record in existing}
    print(f"Excluding {len(already_used)} clauses already in {SAMPLE_PATH.name}")

    dataset = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
        split="train",
    ).shuffle(seed=42)

    label_to_category = {
        label: category
        for category, labels in CATEGORY_LABEL_MAP.items()
        for label in labels
    }

    target_per_category = TRAIN_PER_CATEGORY + VAL_PER_CATEGORY
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        category = label_to_category.get(row["label"])
        if category is None or len(by_category[category]) >= target_per_category:
            continue
        clause_text = row["clause"].strip()
        if not clause_text or clause_text in already_used:
            continue
        by_category[category].append(
            {
                "category": category,
                "cuad_label": row["label"],
                "contract_file": row["file_name"],
                "clause_text": clause_text,
            }
        )

    train_records: list[dict] = []
    val_records: list[dict] = []
    rng = random.Random(42)
    for category in CATEGORY_LABEL_MAP:
        records = by_category.get(category, [])
        if len(records) < target_per_category:
            print(
                f"WARNING: {category} only found {len(records)}/{target_per_category} "
                "new (non-excluded) clauses"
            )
        rng.shuffle(records)
        val_records.extend(records[:VAL_PER_CATEGORY])
        train_records.extend(records[VAL_PER_CATEGORY:target_per_category])

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    TRAIN_PATH.write_text(json.dumps(train_records, indent=2))
    VAL_PATH.write_text(json.dumps(val_records, indent=2))

    print(f"Wrote {len(train_records)} train records to {TRAIN_PATH}")
    print(f"Wrote {len(val_records)} val records to {VAL_PATH}")
    for category in CATEGORY_LABEL_MAP:
        n_train = sum(1 for r in train_records if r["category"] == category)
        n_val = sum(1 for r in val_records if r["category"] == category)
        print(f"  {category}: {n_train} train / {n_val} val")


if __name__ == "__main__":
    generate()

"""Generate presentation PNGs from LegalBERT fine-tune / eval metrics.

Run: python -m scripts.make_presentation_charts
Writes: results/presentation/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation"


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Install matplotlib first: pip install matplotlib") from exc

    OUT.mkdir(parents=True, exist_ok=True)

    categories = [
        "Assignment",
        "Governing Law",
        "IP",
        "Liability",
        "Payment Terms",
        "Termination",
        "Warranty",
    ]

    # From eval_clauses_11530877.out
    ft_p = [0.958, 1.000, 0.959, 0.943, 0.979, 0.943, 0.980]
    ft_r = [0.920, 1.000, 0.940, 1.000, 0.940, 1.000, 0.960]
    ft_f1 = [0.939, 1.000, 0.949, 0.971, 0.959, 0.971, 0.970]
    rule_f1 = [0.851, 0.310, 0.812, 0.707, 0.241, 0.857, 0.747]
    zs_f1 = [0.000, 0.990, 0.837, 0.713, 0.636, 0.588, 0.111]

    macro = {
        "Fine-tuned LegalBERT": 0.966,
        "Rule-based": 0.646,
        "Zero-shot LegalBERT": 0.554,
    }

    # From finetune_legalbert_11530695.out epoch evals
    epochs = [1, 2, 3, 4, 5, 6]
    eval_f1 = [0.694, 0.943, 0.957, 0.950, 0.957, 0.950]
    eval_loss = [1.473, 0.650, 0.340, 0.259, 0.224, 0.232]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    c_ft, c_rule, c_zs, c_acc = "#0B6E4F", "#5C6B73", "#C45C26", "#1B4F72"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(macro.keys())
    vals = [macro[n] for n in names]
    colors = [c_ft, c_rule, c_zs]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1 (held-out CUAD sample, n=350)")
    ax.set_title("Clause classification: fine-tuned vs baselines")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(
            v + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUT / "01_macro_f1_comparison.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    w = 0.25
    ax.bar(x - w, ft_f1, w, label="Fine-tuned", color=c_ft)
    ax.bar(x, rule_f1, w, label="Rule-based", color=c_rule)
    ax.bar(x + w, zs_f1, w, label="Zero-shot LegalBERT", color=c_zs)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("Per-category F1 on held-out CUAD sample")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "02_per_category_f1.png", dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, eval_f1, "o-", color=c_ft, lw=2.2, ms=7, label="Val macro F1")
    ax2.plot(epochs, eval_loss, "s--", color=c_acc, lw=2, ms=6, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro F1", color=c_ft)
    ax2.set_ylabel("Eval loss", color=c_acc)
    ax1.set_ylim(0.5, 1.02)
    ax2.set_ylim(0, 1.8)
    ax1.set_xticks(epochs)
    ax1.set_title("Fine-tuning progress (560 train / 140 val)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "03_training_curves.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ["Train\n(560)", "Val\n(140)", "Held-out\nbenchmark\n(350)"]
    sizes = [560, 140, 350]
    cols = ["#0B6E4F", "#2A9D8F", "#1B4F72"]
    bars = ax.bar(parts, sizes, color=cols, width=0.55)
    ax.set_ylabel("Clauses")
    ax.set_title(
        "Data split (7 CUAD categories, balanced, no train↔benchmark overlap)"
    )
    for bar, s in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + 12,
            str(s),
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0, 650)
    fig.tight_layout()
    fig.savefig(OUT / "04_data_split.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, ft_p, w, label="Precision", color=c_ft)
    ax.bar(x + w / 2, ft_r, w, label="Recall", color="#52B788")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Fine-tuned model: precision vs recall by category")
    ax.axhline(0.966, color="#888", linestyle=":", lw=1.2, label="Macro avg (0.966)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "05_finetuned_precision_recall.png", dpi=200)
    plt.close()

    summary = {
        "held_out_n": 350,
        "train_n": 560,
        "val_n": 140,
        "macro_f1": {"finetuned": 0.966, "rule": 0.646, "zeroshot": 0.554},
        "files": sorted(p.name for p in OUT.glob("*.png")),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(summary['files'])} PNGs to {OUT}")
    for name in summary["files"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()

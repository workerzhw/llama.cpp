#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Plot F8 interval histogram from exact GEMM-input histogram CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_hist(path: Path) -> Dict[str, Tuple[List[str], List[float]]]:
    data: Dict[str, Tuple[List[str], List[float]]] = {}
    buckets: Dict[str, List[str]] = {"src0": [], "src1": [], "combined": []}
    ratios: Dict[str, List[float]] = {"src0": [], "src1": [], "combined": []}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scope = row["scope"]
            bucket_kind = row["bucket_kind"]
            if bucket_kind == "zero":
                label = "zero"
            else:
                label = row["exp_unbiased"]
            buckets[scope].append(label)
            ratios[scope].append(float(row["ratio_of_total"]) * 100.0)

    for scope in buckets:
        data[scope] = (buckets[scope], ratios[scope])
    return data


def plot_scope(ax, title: str, labels: List[str], values: List[float]) -> None:
    if not labels:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    x = list(range(len(labels)))
    ax.bar(x, values, color="#4C78A8", edgecolor="black", linewidth=0.25)
    ax.set_title(title)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot F8 interval histogram")
    parser.add_argument("--csv", required=True, help="Input fp8 interval histogram CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    data = read_hist(Path(args.csv))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    plot_scope(axes[0], "Weights (src0) F8 Interval Histogram", *data["src0"])
    plot_scope(axes[1], "Activations (src1) F8 Interval Histogram", *data["src1"])
    plot_scope(axes[2], "Combined GEMM Inputs F8 Interval Histogram", *data["combined"])
    axes[2].set_xlabel("F8 interval bucket (unbiased exponent of quantized scaled value)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    print(f"csv={args.csv}")
    print(f"png={args.out}")


if __name__ == "__main__":
    main()

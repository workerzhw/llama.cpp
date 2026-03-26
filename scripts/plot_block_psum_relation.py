#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Plot sampled block_dot vs psum_before relationship figures.

Expected input file:
- <prefix>_block_samples.csv

Generated figures:
- <out-dir>/block_over_psum_hist.png
- <out-dir>/block_over_psum_cdf.png
- <out-dir>/block_over_psum_percent.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def read_log2_ratios(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abs_psum_before = float(row["abs_psum_before"])
            abs_block_dot = float(row["abs_block_dot"])
            if abs_psum_before <= 0.0 or abs_block_dot <= 0.0:
                continue
            values.append(math.log2(abs_block_dot / abs_psum_before))
    return np.array(values, dtype=float)


def plot_hist(log2_ratios: np.ndarray, thresholds: List[int], out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(log2_ratios, bins=120, density=True, alpha=0.85, color="#4C78A8", edgecolor="black", linewidth=0.25)
    for threshold in thresholds:
        plt.axvline(-threshold, color="#E45756", linestyle="--", linewidth=1.2)
    plt.xlabel("log2(|block_dot| / |psum_before|)")
    plt.ylabel("Density")
    plt.title("Block Dot over Partial Sum Distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_percent_hist(log2_ratios: np.ndarray, thresholds: List[int], out_path: Path) -> None:
    weights = np.ones(log2_ratios.shape[0], dtype=float) * (100.0 / float(log2_ratios.shape[0]))

    plt.figure(figsize=(9, 5))
    plt.hist(
        log2_ratios,
        bins=120,
        weights=weights,
        alpha=0.85,
        color="#F2CF5B",
        edgecolor="black",
        linewidth=0.25,
    )
    for threshold in thresholds:
        plt.axvline(-threshold, color="#E45756", linestyle="--", linewidth=1.2)
    plt.xlabel("log2(|block_dot| / |psum_before|)")
    plt.ylabel("Percentage of sampled blocks (%)")
    plt.title("Block Dot over Partial Sum Percentage by Bin")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cdf(log2_ratios: np.ndarray, thresholds: List[int], out_path: Path) -> None:
    sorted_values = np.sort(log2_ratios)
    cdf = np.arange(1, sorted_values.size + 1, dtype=float) / float(sorted_values.size)

    plt.figure(figsize=(9, 5))
    plt.plot(sorted_values, cdf, color="#72B7B2", linewidth=1.8)
    for threshold in thresholds:
        plt.axvline(-threshold, color="#E45756", linestyle="--", linewidth=1.2)
    plt.xlabel("log2(|block_dot| / |psum_before|)")
    plt.ylabel("CDF")
    plt.title("Block Dot over Partial Sum CDF")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot block_dot/psum relationship figures")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix used by GGML_REDUCTION_PROD_PROFILE_PREFIX",
    )
    parser.add_argument(
        "--thresholds",
        default="5,10,15",
        help="Comma-separated log2 thresholds to overlay as vertical lines",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for figures (default: directory of <prefix>)",
    )
    args = parser.parse_args()

    input_csv = Path(f"{args.prefix}_block_samples.csv")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing file: {input_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [int(part.strip()) for part in args.thresholds.split(",") if part.strip()]
    log2_ratios = read_log2_ratios(input_csv)
    if log2_ratios.size == 0:
        raise RuntimeError("No comparable block samples with abs_psum_before > 0 and abs_block_dot > 0")

    hist_path = out_dir / "block_over_psum_hist.png"
    cdf_path = out_dir / "block_over_psum_cdf.png"
    percent_hist_path = out_dir / "block_over_psum_percent.png"
    plot_hist(log2_ratios, thresholds, hist_path)
    plot_percent_hist(log2_ratios, thresholds, percent_hist_path)
    plot_cdf(log2_ratios, thresholds, cdf_path)

    print(f"input={input_csv}")
    print(f"points={log2_ratios.size}")
    print(f"hist={hist_path}")
    print(f"percent_hist={percent_hist_path}")
    print(f"cdf={cdf_path}")


if __name__ == "__main__":
    main()

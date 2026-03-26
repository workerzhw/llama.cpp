#!/usr/bin/env python3
"""Summarize sampled block_dot vs psum_before relationships.

Expected input file:
- <prefix>_block_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Dict, List


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def read_block_samples(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "abs_psum_before": float(row["abs_psum_before"]),
                    "abs_block_dot": float(row["abs_block_dot"]),
                    "same_sign": float(row["same_sign"]),
                }
            )
    return rows


def build_rows(samples: List[Dict[str, float]], thresholds: List[int]) -> List[Dict[str, str]]:
    total_blocks = len(samples)
    nonzero_psum = [row for row in samples if row["abs_psum_before"] > 0.0]
    comparable = [row for row in nonzero_psum if row["abs_block_dot"] > 0.0]
    same_sign_count = sum(1 for row in nonzero_psum if row["same_sign"] > 0.5)

    log2_ratios = sorted(
        math.log2(row["abs_block_dot"] / row["abs_psum_before"])
        for row in comparable
    )

    median_log2_ratio = median(log2_ratios) if log2_ratios else float("nan")
    p90_log2_ratio = percentile(log2_ratios, 0.90)
    p99_log2_ratio = percentile(log2_ratios, 0.99)

    out: List[Dict[str, str]] = []
    for threshold in thresholds:
        lt_count_all = 0
        lt_count_nonzero = 0
        for row in samples:
            abs_psum_before = row["abs_psum_before"]
            abs_block_dot = row["abs_block_dot"]
            if abs_block_dot < abs_psum_before * math.ldexp(1.0, -threshold):
                lt_count_all += 1
                if abs_psum_before > 0.0:
                    lt_count_nonzero += 1

        out.append(
            {
                "threshold_log2_n": str(threshold),
                "total_blocks": str(total_blocks),
                "psum_nonzero_blocks": str(len(nonzero_psum)),
                "comparable_blocks": str(len(comparable)),
                "lt_count_all": str(lt_count_all),
                "lt_ratio_all": f"{(lt_count_all / total_blocks) if total_blocks else 0.0:.6f}",
                "lt_count_psum_nonzero": str(lt_count_nonzero),
                "lt_ratio_psum_nonzero": f"{(lt_count_nonzero / len(nonzero_psum)) if nonzero_psum else 0.0:.6f}",
                "same_sign_ratio_psum_nonzero": f"{(same_sign_count / len(nonzero_psum)) if nonzero_psum else 0.0:.6f}",
                "median_log2_abs_block_over_abs_psum": f"{median_log2_ratio:.6f}" if log2_ratios else "nan",
                "p90_log2_abs_block_over_abs_psum": f"{p90_log2_ratio:.6f}" if log2_ratios else "nan",
                "p99_log2_abs_block_over_abs_psum": f"{p99_log2_ratio:.6f}" if log2_ratios else "nan",
            }
        )
    return out


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    fields = [
        "threshold_log2_n",
        "total_blocks",
        "psum_nonzero_blocks",
        "comparable_blocks",
        "lt_count_all",
        "lt_ratio_all",
        "lt_count_psum_nonzero",
        "lt_ratio_psum_nonzero",
        "same_sign_ratio_psum_nonzero",
        "median_log2_abs_block_over_abs_psum",
        "p90_log2_abs_block_over_abs_psum",
        "p99_log2_abs_block_over_abs_psum",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Block Dot vs PSum Relation\n\n")
        if not rows:
            f.write("No sampled block rows found.\n")
            return

        f.write("| threshold_log2_n | total_blocks | psum_nonzero_blocks | lt_ratio_all | lt_ratio_psum_nonzero | same_sign_ratio_psum_nonzero | median_log2_abs_block_over_abs_psum | p90_log2_abs_block_over_abs_psum | p99_log2_abs_block_over_abs_psum |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                "| {threshold_log2_n} | {total_blocks} | {psum_nonzero_blocks} | {lt_ratio_all} | {lt_ratio_psum_nonzero} | {same_sign_ratio_psum_nonzero} | {median_log2_abs_block_over_abs_psum} | {p90_log2_abs_block_over_abs_psum} | {p99_log2_abs_block_over_abs_psum} |\n".format(**row)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize block_dot/psum relationships")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix used by GGML_REDUCTION_PROD_PROFILE_PREFIX",
    )
    parser.add_argument(
        "--thresholds",
        default="5,10,15",
        help="Comma-separated log2 thresholds n for abs(block_dot) < abs(psum) * 2^-n",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: <prefix>_block_psum_relation.csv)",
    )
    parser.add_argument(
        "--out-md",
        default="",
        help="Output Markdown path (default: <prefix>_block_psum_relation.md)",
    )
    args = parser.parse_args()

    input_csv = Path(f"{args.prefix}_block_samples.csv")
    out_csv = Path(args.out_csv) if args.out_csv else Path(f"{args.prefix}_block_psum_relation.csv")
    out_md = Path(args.out_md) if args.out_md else Path(f"{args.prefix}_block_psum_relation.md")

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing file: {input_csv}")

    thresholds = [int(part.strip()) for part in args.thresholds.split(",") if part.strip()]
    samples = read_block_samples(input_csv)
    rows = build_rows(samples, thresholds)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print(f"input={input_csv}")
    print(f"rows={len(samples)}")
    print(f"csv={out_csv}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a concise comparison table from reduction profiler summary logs.

Input:  one or more '*_reduction_prod_profile_summary.log' files
Output: CSV + Markdown table for easy experiment discussion
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

SUMMARY_SUFFIX = "_reduction_prod_profile_summary.log"

KEYS = [
    "block_size",
    "block_drop_log2_n",
    "reductions",
    "all_reductions",
    "total_products",
    "sampled_block_terms",
    "sampled_block_dropped",
    "sampled_global_block_drop_ratio",
    "sampled_avg_block_drop_ratio",
    "all_block_terms",
    "estimated_global_block_dropped",
    "estimated_global_block_drop_ratio",
    "avg_cancel_ratio",
    "avg_neff_ratio",
    "sampled_kept",
    "sampled_dropped",
]

FALLBACK_KEYS = {
    "sampled_block_terms": "total_block_terms",
    "sampled_block_dropped": "total_block_dropped",
    "sampled_global_block_drop_ratio": "global_block_drop_ratio",
    "sampled_avg_block_drop_ratio": "avg_block_drop_ratio",
}


def parse_summary(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()

    case = path.name
    if case.endswith(SUMMARY_SUFFIX):
        case = case[: -len(SUMMARY_SUFFIX)]

    row: Dict[str, str] = {"case": case, "summary_file": str(path)}
    for k in KEYS:
        if k in data:
            row[k] = data[k]
        elif k in FALLBACK_KEYS and FALLBACK_KEYS[k] in data:
            row[k] = data[FALLBACK_KEYS[k]]
        else:
            row[k] = ""
    return row


def collect_rows(root: Path) -> List[Dict[str, str]]:
    files = sorted(root.rglob(f"*{SUMMARY_SUFFIX}"))
    rows = [parse_summary(p) for p in files]

    def sort_key(r: Dict[str, str]):
        n_raw = r.get("block_drop_log2_n", "")
        try:
            n_val = int(n_raw)
        except ValueError:
            n_val = 1 << 30
        return (n_val, r["case"])

    rows.sort(key=sort_key)
    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    fields = ["case"] + KEYS + ["summary_file"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows: List[Dict[str, str]], out_md: Path) -> None:
    cols = [
        "case",
        "block_drop_log2_n",
        "block_size",
        "sampled_global_block_drop_ratio",
        "estimated_global_block_drop_ratio",
        "sampled_avg_block_drop_ratio",
        "sampled_block_terms",
        "sampled_block_dropped",
        "all_block_terms",
        "avg_cancel_ratio",
        "avg_neff_ratio",
        "sampled_kept",
    ]
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Reduction Block-Drop Comparison\n\n")
        if not rows:
            f.write("No summary files found.\n")
            return

        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r.get(c, "") for c in cols) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reduction block-drop comparison table")
    ap.add_argument("--root", default="kv_dump_logs", help="Root directory to scan summary logs")
    ap.add_argument("--out-csv", default="kv_dump_logs/reduction_block_drop_compare.csv")
    ap.add_argument("--out-md", default="kv_dump_logs/reduction_block_drop_compare.md")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_rows(root)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print(f"cases={len(rows)}")
    print(f"csv={out_csv}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()

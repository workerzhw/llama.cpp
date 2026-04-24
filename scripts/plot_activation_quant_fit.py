#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Plot activation distribution together with uniform-Q6 and Q6-log fits.

The figure mixes two views on purpose:

1. A real activation-magnitude histogram aggregated from collected stats.
2. A normalized proxy view used only to compare codebook placement.

The current collected artifacts store per-layer absolute-value histograms, but they
do not dump the true per-block `|x| / block_absmax` distribution used by the runtime
QDQ path. To keep the comparison evidence-based, this script uses:

- actual panel: aggregated absolute histogram from `*_collect_hist.csv`
- fit panels: layer-max normalized proxy, `u = |x| / max_abs_nonzero(layer)`

That proxy is stricter than the real per-block absmax normalization, so the figure
should be read as a visual explanation of codebook shape rather than an exact QDQ
error benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


QMAX = 31

COLOR_ACTUAL = "#4C78A8"
COLOR_PROXY = "#9EA3B0"
COLOR_UNIFORM = "#4C78A8"
COLOR_LOG = "#54A24B"
COLOR_FIT = "#F58518"
COLOR_ACCENT = "#E45756"
COLOR_GRID = "#CBD5E1"


@dataclass
class LayerSummary:
    stage: str
    layer: int
    max_abs_nonzero: float
    min_abs_nonzero: float


@dataclass
class HistogramRow:
    stage: str
    layer: int
    bin_lo_log10: float
    bin_hi_log10: float
    count: int


def read_summary(path: Path, stage: str, target_kind: str) -> Dict[int, LayerSummary]:
    out: Dict[int, LayerSummary] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["stage"] != stage or row["target_kind"] != target_kind:
                continue
            layer = int(row["layer"])
            out[layer] = LayerSummary(
                stage=stage,
                layer=layer,
                max_abs_nonzero=float(row["max_abs_nonzero"]),
                min_abs_nonzero=float(row["min_abs_nonzero"]),
            )
    if not out:
        raise RuntimeError(f"No summary rows found for stage={stage!r}, target_kind={target_kind!r}")
    return out


def read_hist_rows(path: Path, stage: str, target_kind: str, tracked_kind: str) -> List[HistogramRow]:
    rows: List[HistogramRow] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["stage"] != stage:
                continue
            if row["target_kind"] != target_kind or row["tracked_kind"] != tracked_kind:
                continue
            count = int(row["count"])
            if count <= 0:
                continue
            rows.append(
                HistogramRow(
                    stage=stage,
                    layer=int(row["layer"]),
                    bin_lo_log10=float(row["bin_lo"]),
                    bin_hi_log10=float(row["bin_hi"]),
                    count=count,
                )
            )
    if not rows:
        raise RuntimeError(
            f"No histogram rows found for stage={stage!r}, target_kind={target_kind!r}, tracked_kind={tracked_kind!r}"
        )
    return rows


def aggregate_actual_hist(rows: Iterable[HistogramRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts_by_bin: Dict[Tuple[float, float], int] = defaultdict(int)
    for row in rows:
        counts_by_bin[(row.bin_lo_log10, row.bin_hi_log10)] += row.count

    keys = sorted(counts_by_bin.keys())
    lo = np.array([10.0 ** key[0] for key in keys], dtype=float)
    hi = np.array([10.0 ** key[1] for key in keys], dtype=float)
    counts = np.array([counts_by_bin[key] for key in keys], dtype=float)
    return lo, hi, counts


def build_proxy_hist(
    rows: Iterable[HistogramRow],
    summaries: Dict[int, LayerSummary],
    proxy_edges: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(proxy_edges.size - 1, dtype=float)

    for row in rows:
        summary = summaries.get(row.layer)
        if summary is None or not (summary.max_abs_nonzero > 0.0):
            continue

        mid = 10.0 ** ((row.bin_lo_log10 + row.bin_hi_log10) * 0.5)
        u = min(mid / summary.max_abs_nonzero, 1.0)
        if not (u > 0.0):
            continue

        idx = int(np.searchsorted(proxy_edges, u, side="right") - 1)
        if 0 <= idx < counts.size:
            counts[idx] += row.count

    return counts


def format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def uniform_levels() -> np.ndarray:
    return np.arange(1, QMAX + 1, dtype=float) / float(QMAX)


def uniform_zero_threshold() -> float:
    return 0.5 / float(QMAX)


def log_levels(step: int) -> np.ndarray:
    q = np.arange(1, QMAX + 1, dtype=float)
    return np.power(2.0, (q - float(QMAX)) / float(step))


def log_zero_threshold(step: int) -> float:
    return 0.5 * (2.0 ** (-float(QMAX - 1) / float(step)))


def quantize_uniform(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zt = uniform_zero_threshold()
    out = np.zeros_like(u)
    is_zero = u <= zt
    active = ~is_zero
    if np.any(active):
        q = np.rint(u[active] * float(QMAX)).astype(int)
        q = np.clip(q, 1, QMAX)
        out[active] = q.astype(float) / float(QMAX)
    return out, is_zero


def quantize_log(u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    zt = log_zero_threshold(step)
    out = np.zeros_like(u)
    is_zero = u <= zt
    active = ~is_zero
    if np.any(active):
        q = np.rint(float(QMAX) + float(step) * np.log2(u[active])).astype(int)
        q = np.clip(q, 1, QMAX)
        out[active] = np.power(2.0, (q.astype(float) - float(QMAX)) / float(step))
    return out, is_zero


def quantized_mass(centers: np.ndarray, counts: np.ndarray, quantize: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, float]:
    qhat, is_zero = quantize(centers)
    total = float(np.sum(counts))
    zero_share = float(np.sum(counts[is_zero])) / total if total > 0.0 else 0.0

    mass_by_level: Dict[float, float] = defaultdict(float)
    for level, weight, zero in zip(qhat, counts, is_zero):
        if zero or level <= 0.0:
            continue
        mass_by_level[float(level)] += float(weight)

    if not mass_by_level:
        return np.array([], dtype=float), np.array([], dtype=float), zero_share

    levels = np.array(sorted(mass_by_level.keys()), dtype=float)
    masses = np.array([mass_by_level[level] / total * 100.0 for level in levels], dtype=float)
    return levels, masses, zero_share


def weighted_stats_from_hist(lo: np.ndarray, hi: np.ndarray, counts: np.ndarray) -> Dict[str, float]:
    mids = np.sqrt(lo * hi)
    total = float(np.sum(counts))
    out: Dict[str, float] = {}
    for threshold in (1e-2, 1e-1, 1.0):
        out[f"lt_{threshold:g}"] = float(np.sum(counts[mids < threshold])) / total if total > 0.0 else 0.0
    return out


def dynamic_range_stats(summaries: Dict[int, LayerSummary]) -> Tuple[float, float]:
    decades: List[float] = []
    for summary in summaries.values():
        if summary.min_abs_nonzero > 0.0 and summary.max_abs_nonzero > 0.0:
            decades.append(math.log10(summary.max_abs_nonzero / summary.min_abs_nonzero))
    if not decades:
        return 0.0, 0.0
    return float(np.mean(decades)), float(np.max(decades))


def plot_hist_bars(ax: plt.Axes, lo: np.ndarray, hi: np.ndarray, counts: np.ndarray, color: str, label: str) -> None:
    total = float(np.sum(counts))
    pct = counts / total * 100.0 if total > 0.0 else counts
    ax.bar(lo, pct, width=hi - lo, align="edge", color=color, alpha=0.75, edgecolor="white", linewidth=0.25, label=label)


def plot_codebook_panel(ax: plt.Axes, proxy_edges: np.ndarray, proxy_counts: np.ndarray, levels: np.ndarray, color: str, title: str, note_lines: List[str]) -> None:
    plot_hist_bars(ax, proxy_edges[:-1], proxy_edges[1:], proxy_counts, COLOR_PROXY, "normalized proxy distribution")
    for level in levels:
        ax.axvline(level, color=color, alpha=0.25, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(proxy_edges[0], proxy_edges[-1])
    ax.set_xlabel("proxy normalized magnitude u = |x| / layer_max")
    ax.set_ylabel("share of proxy samples (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, color=COLOR_GRID)
    ax.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
    )


def plot_fit_panel(
    ax: plt.Axes,
    proxy_edges: np.ndarray,
    proxy_counts: np.ndarray,
    levels: np.ndarray,
    assigned_levels: np.ndarray,
    assigned_masses: np.ndarray,
    zero_share: float,
    color: str,
    title: str,
    note_lines: List[str],
) -> None:
    plot_hist_bars(ax, proxy_edges[:-1], proxy_edges[1:], proxy_counts, COLOR_PROXY, "normalized proxy distribution")
    for level in levels:
        ax.axvline(level, color=color, alpha=0.18, linewidth=0.75)

    if assigned_levels.size > 0:
        ax.vlines(assigned_levels, 0.0, assigned_masses, color=COLOR_FIT, linewidth=1.8, alpha=0.95)
        ax.scatter(assigned_levels, assigned_masses, s=24, color=COLOR_FIT, zorder=4, label="quantized reconstructed mass")

    ax.set_xscale("log")
    ax.set_xlim(proxy_edges[0], proxy_edges[-1])
    ax.set_xlabel("proxy normalized magnitude u = |x| / layer_max")
    ax.set_ylabel("share of proxy samples (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, color=COLOR_GRID)
    notes = note_lines + [f"zero-mapped proxy mass: {zero_share * 100.0:.1f}%"]
    ax.text(
        0.02,
        0.98,
        "\n".join(notes),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot activation distribution with uniform/log quantization fits")
    parser.add_argument("--hist-csv", required=True, help="Input *_collect_hist.csv")
    parser.add_argument("--summary-csv", required=True, help="Input *_collect_summary.csv")
    parser.add_argument("--out-svg", required=True, help="Output SVG path")
    parser.add_argument("--out-png", default="", help="Optional output PNG path")
    parser.add_argument("--stage", default="decode", help="Stage to plot (default: decode)")
    parser.add_argument("--target-kind", default="silu", help="Target kind to plot (default: silu)")
    parser.add_argument("--tracked-kind", default="abs_nonzero", help="Tracked histogram kind (default: abs_nonzero)")
    parser.add_argument("--log-step", type=int, default=4, help="Q6-log step for codebook overlay (default: 4)")
    args = parser.parse_args()

    hist_path = Path(args.hist_csv)
    summary_path = Path(args.summary_csv)
    out_svg = Path(args.out_svg)
    out_png = Path(args.out_png) if args.out_png else None

    summaries = read_summary(summary_path, args.stage, args.target_kind)
    hist_rows = read_hist_rows(hist_path, args.stage, args.target_kind, args.tracked_kind)

    actual_lo, actual_hi, actual_counts = aggregate_actual_hist(hist_rows)
    actual_stats = weighted_stats_from_hist(actual_lo, actual_hi, actual_counts)
    avg_decades, max_decades = dynamic_range_stats(summaries)

    proxy_edges = np.logspace(-6.0, 0.0, 161)
    proxy_counts = build_proxy_hist(hist_rows, summaries, proxy_edges)
    proxy_centers = np.sqrt(proxy_edges[:-1] * proxy_edges[1:])

    uniform = uniform_levels()
    logq = log_levels(args.log_step)
    uniform_assigned_levels, uniform_assigned_mass, uniform_zero_share = quantized_mass(proxy_centers, proxy_counts, quantize_uniform)
    log_assigned_levels, log_assigned_mass, log_zero_share = quantized_mass(
        proxy_centers,
        proxy_counts,
        lambda values: quantize_log(values, args.log_step),
    )

    uniform_small = int(np.sum(uniform < 0.1))
    log_small = int(np.sum(logq < 0.1))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    fig.suptitle(
        f"{args.target_kind.upper()} activation distribution vs Q6 codebook fit ({args.stage} stage)",
        fontsize=16,
        y=0.985,
    )

    ax_actual = axes[0, 0]
    plot_hist_bars(ax_actual, actual_lo, actual_hi, actual_counts, COLOR_ACTUAL, "measured activation magnitude")
    ax_actual.set_xscale("log")
    ax_actual.set_xlabel("absolute activation magnitude |x|")
    ax_actual.set_ylabel("share of nonzero activations (%)")
    ax_actual.set_title("Measured activation distribution")
    ax_actual.grid(True, which="both", alpha=0.25, color=COLOR_GRID)
    for threshold in (1e-2, 1e-1, 1.0):
        ax_actual.axvline(threshold, color=COLOR_ACCENT, linestyle="--", linewidth=1.0, alpha=0.65)
    ax_actual.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"share(|x| < 1e-2): {format_pct(actual_stats['lt_0.01'])}",
                f"share(|x| < 1e-1): {format_pct(actual_stats['lt_0.1'])}",
                f"share(|x| < 1): {format_pct(actual_stats['lt_1'])}",
                f"avg layer dynamic range: {avg_decades:.2f} decades",
                f"max layer dynamic range: {max_decades:.2f} decades",
            ]
        ),
        transform=ax_actual.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLOR_ACTUAL, "alpha": 0.92},
    )

    ax_levels = axes[0, 1]
    plot_hist_bars(ax_levels, proxy_edges[:-1], proxy_edges[1:], proxy_counts, COLOR_PROXY, "layer-max normalized proxy")
    for level in uniform:
        ax_levels.axvline(level, color=COLOR_UNIFORM, alpha=0.25, linewidth=0.8)
    for level in logq:
        ax_levels.axvline(level, color=COLOR_LOG, alpha=0.25, linewidth=0.8)
    ax_levels.set_xscale("log")
    ax_levels.set_xlim(proxy_edges[0], proxy_edges[-1])
    ax_levels.set_xlabel("proxy normalized magnitude u = |x| / layer_max")
    ax_levels.set_ylabel("share of proxy samples (%)")
    ax_levels.set_title("Proxy normalized distribution and codebook placement")
    ax_levels.grid(True, which="both", alpha=0.25, color=COLOR_GRID)
    ax_levels.text(
        0.02,
        0.98,
        "\n".join(
            [
                "proxy only: per-layer max normalization (not true block absmax)",
                f"uniform Q6: {uniform_small} nonzero levels below u < 0.1",
                f"Q6-log step={args.log_step}: {log_small} nonzero levels below u < 0.1",
                f"uniform zero threshold: {uniform_zero_threshold():.3e}",
                f"Q6-log zero threshold: {log_zero_threshold(args.log_step):.3e}",
            ]
        ),
        transform=ax_levels.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#64748B", "alpha": 0.92},
    )

    plot_fit_panel(
        axes[1, 0],
        proxy_edges,
        proxy_counts,
        uniform,
        uniform_assigned_levels,
        uniform_assigned_mass,
        uniform_zero_share,
        COLOR_UNIFORM,
        "Uniform Q6 fit on proxy normalized distribution",
        [
            "linear levels spend most codes on the high-value tail",
            f"only {uniform_small} nonzero levels exist below u < 0.1",
            "dense small-magnitude region collapses into zero / first few codes",
        ],
    )

    plot_fit_panel(
        axes[1, 1],
        proxy_edges,
        proxy_counts,
        logq,
        log_assigned_levels,
        log_assigned_mass,
        log_zero_share,
        COLOR_LOG,
        f"Q6-log fit on proxy normalized distribution (step={args.log_step})",
        [
            "geometric levels match the long-tail, multi-decade shape",
            f"{log_small} nonzero levels exist below u < 0.1",
            f"max local spacing factor = 2^(1/{args.log_step}) = {2 ** (1 / args.log_step):.3f}",
        ],
    )

    fig.text(
        0.5,
        0.012,
        "Measured panel uses the real collected activation histogram. Fit panels use a layer-max normalized proxy because the current artifacts do not dump the true block-absmax-normalized distribution. The comparison is therefore intended to explain codebook shape, not to replace direct perplexity or reconstruction-error evaluation.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
        wrap=True,
    )

    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.965))
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, dpi=200)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"hist_csv={hist_path}")
    print(f"summary_csv={summary_path}")
    print(f"stage={args.stage}")
    print(f"target_kind={args.target_kind}")
    print(f"out_svg={out_svg}")
    if out_png is not None:
        print(f"out_png={out_png}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
from dataclasses import dataclass


GLOBAL_REFERENCE_STAGE = "prefill"
GLOBAL_REFERENCE_LAYER = 1


@dataclass
class StageData:
    stage: str
    layer: int
    threshold: float
    total_values: int
    original_zero_values: int
    nonzero_values: int
    tracked_kind: str
    tracked_values: int
    collect_log10_min: float
    collect_log10_max: float
    collect_bins: int
    underflow_count: int
    overflow_count: int
    min_abs_nonzero: float
    max_abs_nonzero: float
    bins: list[int]

    @property
    def original_zero_ratio(self) -> float:
        if self.total_values == 0:
            return 0.0
        return self.original_zero_values / self.total_values

    @property
    def max_added_zero_ratio(self) -> float:
        if self.total_values == 0:
            return 0.0
        return self.tracked_values / self.total_values

    @property
    def uses_upper_tail_threshold(self) -> bool:
        return self.tracked_kind == "negative_tail_magnitude"


@dataclass
class GroupTarget:
    group: str
    layers: list[int]
    prefill_target: float
    decode_target: float


@dataclass
class StageEstimate:
    target_ratio: float
    available_ratio: float
    stage_threshold: float
    stage_estimated_ratio: float
    final_estimated_ratio: float
    original_zero_ratio: float
    final_zero_ratio: float
    status: str


def filtered_csv_dict_reader(path: pathlib.Path) -> csv.DictReader:
    handle = path.open("r", encoding="utf-8", newline="")
    return csv.DictReader(
        line for line in handle if line.strip() and not line.lstrip().startswith("#")
    )


def parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for chunk in re.split(r"[;\s]+", text.strip()):
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid layer range: {chunk}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(chunk))
    return sorted(set(layers))


def parse_target_profile(path: pathlib.Path) -> list[GroupTarget]:
    groups: list[GroupTarget] = []
    seen_layers: dict[int, str] = {}
    reader = filtered_csv_dict_reader(path)
    for row in reader:
        group = row["group"].strip()
        layers = parse_layers(row["layers"])
        prefill_target = float(row["prefill_target"])
        decode_target = float(row["decode_target"])
        if prefill_target < 0.0 or decode_target < 0.0:
            raise ValueError(f"negative target ratio in {path}: {group}")
        for layer in layers:
            if layer in seen_layers:
                raise ValueError(
                    f"layer {layer} appears in both group {seen_layers[layer]} and {group}"
                )
            seen_layers[layer] = group
        groups.append(
            GroupTarget(
                group=group,
                layers=layers,
                prefill_target=prefill_target,
                decode_target=decode_target,
            )
        )
    return groups


def parse_collection(summary_path: pathlib.Path, hist_path: pathlib.Path) -> dict[tuple[str, int], StageData]:
    stage_data: dict[tuple[str, int], StageData] = {}

    reader = filtered_csv_dict_reader(summary_path)
    for row in reader:
        key = (row["stage"].strip(), int(row["layer"]))
        if key in stage_data:
            raise ValueError(f"duplicate summary row for {key}")
        bins = int(row["collect_bins"])
        stage_data[key] = StageData(
            stage=row["stage"].strip(),
            layer=int(row["layer"]),
            threshold=float(row["threshold"]),
            total_values=int(row["total_values"]),
            original_zero_values=int(row["original_zero_values"]),
            nonzero_values=int(row["nonzero_values"]),
            tracked_kind=row.get("tracked_kind", "abs_nonzero").strip() if row.get("tracked_kind") else "abs_nonzero",
            tracked_values=int(row.get("tracked_values", row["nonzero_values"])),
            collect_log10_min=float(row["collect_log10_min"]),
            collect_log10_max=float(row["collect_log10_max"]),
            collect_bins=bins,
            underflow_count=int(row["underflow_count"]),
            overflow_count=int(row["overflow_count"]),
            min_abs_nonzero=float(row["min_abs_nonzero"]),
            max_abs_nonzero=float(row["max_abs_nonzero"]),
            bins=[0] * bins,
        )

    reader = filtered_csv_dict_reader(hist_path)
    for row in reader:
        key = (row["stage"].strip(), int(row["layer"]))
        if key not in stage_data:
            raise ValueError(f"histogram row without summary for {key}")
        data = stage_data[key]
        index = int(row["bin_index"])
        if index < 0 or index >= data.collect_bins:
            raise ValueError(f"bin index out of range for {key}: {index}")
        data.bins[index] = int(row["count"])

    for data in stage_data.values():
        histogram_total = data.underflow_count + data.overflow_count + sum(data.bins)
        if histogram_total > 0:
            data.tracked_values = histogram_total

    return stage_data


def estimate_count_below_threshold(data: StageData, threshold: float) -> float:
    if threshold <= 0.0 or data.total_values == 0 or data.tracked_values == 0:
        return 0.0

    log_threshold = math.log10(threshold)
    if log_threshold < data.collect_log10_min:
        return 0.0

    cumulative = float(data.underflow_count)
    if log_threshold >= data.collect_log10_max:
        return float(data.tracked_values)

    bin_width = (data.collect_log10_max - data.collect_log10_min) / data.collect_bins
    for index, count in enumerate(data.bins):
        bin_lo = data.collect_log10_min + bin_width * index
        bin_hi = bin_lo + bin_width
        if log_threshold >= bin_hi:
            cumulative += count
            continue
        if log_threshold <= bin_lo:
            break
        fraction = (log_threshold - bin_lo) / bin_width
        cumulative += count * fraction
        break

    return min(cumulative, float(data.tracked_values))


def estimate_count_selected_by_threshold(data: StageData, threshold: float) -> float:
    if threshold <= 0.0 or data.total_values == 0 or data.tracked_values == 0:
        return 0.0

    below_count = estimate_count_below_threshold(data, threshold)
    if data.uses_upper_tail_threshold:
        return max(0.0, float(data.tracked_values) - below_count)
    return below_count


def estimate_ratio_below_threshold(data: StageData, threshold: float) -> float:
    if data.total_values == 0:
        return 0.0
    return estimate_count_selected_by_threshold(data, threshold) / data.total_values


def threshold_for_below_count(data: StageData, target_below_count: float) -> float:
    target_below_count = max(0.0, min(target_below_count, float(data.tracked_values)))

    cumulative = float(data.underflow_count)
    if target_below_count <= cumulative:
        return 10 ** data.collect_log10_min

    bin_width = (data.collect_log10_max - data.collect_log10_min) / data.collect_bins
    for index, count in enumerate(data.bins):
        next_cumulative = cumulative + count
        if target_below_count <= next_cumulative:
            bin_lo = data.collect_log10_min + bin_width * index
            if count == 0:
                return 10 ** (bin_lo + bin_width)

            fraction = (target_below_count - cumulative) / count
            return 10 ** (bin_lo + fraction * bin_width)
        cumulative = next_cumulative

    return max(data.max_abs_nonzero, 10 ** data.collect_log10_max)


def threshold_for_target_ratio(data: StageData, target_ratio: float) -> tuple[float, float]:
    if data.total_values == 0 or data.tracked_values == 0 or target_ratio <= 0.0:
        return 0.0, 0.0

    target_ratio = min(target_ratio, data.max_added_zero_ratio)
    target_selected_count = target_ratio * data.total_values
    if data.uses_upper_tail_threshold:
        target_below_count = float(data.tracked_values) - target_selected_count
    else:
        target_below_count = target_selected_count

    threshold = threshold_for_below_count(data, target_below_count)
    return threshold, estimate_ratio_below_threshold(data, threshold)


def combine_stage_thresholds(*stage_data_and_thresholds: tuple[StageData | None, float]) -> float:
    candidates = [threshold for _, threshold in stage_data_and_thresholds if threshold > 0.0]
    if not candidates:
        return 0.0

    if any(data is not None and data.uses_upper_tail_threshold for data, _ in stage_data_and_thresholds):
        return max(candidates)

    return min(candidates)


def build_layer_targets(groups: list[GroupTarget]) -> dict[int, tuple[str, float, float]]:
    targets: dict[int, tuple[str, float, float]] = {}
    for group in groups:
        for layer in group.layers:
            targets[layer] = (group.group, group.prefill_target, group.decode_target)
    return targets


def parse_channel_max(path: pathlib.Path) -> dict[tuple[str, int], list[float]]:
    stage_data: dict[tuple[str, int], list[float]] = {}
    seen: set[tuple[str, int, int]] = set()

    reader = filtered_csv_dict_reader(path)
    for row in reader:
        stage = row["stage"].strip()
        layer = int(row["layer"])
        channel = int(row["channel"])
        raw_value = row.get("max_abs_activation", row.get("max_activation"))
        if raw_value is None:
            raise ValueError(f"missing max_abs_activation column in {path}")
        value = float(raw_value)
        if channel < 0:
            raise ValueError(f"invalid negative channel index in {path}: {channel}")

        seen_key = (stage, layer, channel)
        if seen_key in seen:
            raise ValueError(f"duplicate channel-max row for {seen_key}")
        seen.add(seen_key)

        channels = stage_data.setdefault((stage, layer), [])
        if len(channels) <= channel:
            channels.extend([float("-inf")] * (channel + 1 - len(channels)))
        channels[channel] = value

    return stage_data


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_channel_threshold_config(path: pathlib.Path, rows: list[tuple[int, int, float]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Generated from per-channel SiLU absolute max activations\n")
        handle.write(
            f"# collect_prefix={args.collect_prefix} generate_mode={args.generate_mode} channel_threshold_ratio={args.channel_threshold_ratio}\n"
        )
        handle.write("layer,channel,threshold\n")
        for layer, channel, threshold in rows:
            if threshold > 0.0:
                handle.write(f"{layer},{channel},{threshold:.8g}\n")


def write_channel_report(path: pathlib.Path, report_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "channel_count",
        "prefill_channel_count",
        "decode_channel_count",
        "channel_threshold_ratio",
        "prefill_abs_max_mean",
        "prefill_abs_max_max",
        "decode_abs_max_mean",
        "decode_abs_max_max",
        "combined_abs_max_mean",
        "combined_abs_max_max",
        "threshold_nonzero_channels",
        "threshold_min",
        "threshold_mean",
        "threshold_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def generate_channel_thresholds(
    collect_prefix: pathlib.Path,
    output_path: pathlib.Path,
    report_path: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    channel_max_path = pathlib.Path(f"{collect_prefix}_channel_max.csv")
    if not channel_max_path.is_file():
        raise FileNotFoundError(f"missing channel-max collection file: {channel_max_path}")

    channel_max = parse_channel_max(channel_max_path)
    layers = sorted({layer for _, layer in channel_max})
    generated_rows: list[tuple[int, int, float]] = []
    report_rows: list[dict[str, object]] = []

    for layer in layers:
        prefill_values = list(channel_max.get(("prefill", layer), []))
        decode_values = list(channel_max.get(("decode", layer), []))
        channel_count = max(len(prefill_values), len(decode_values))
        combined_abs_max: list[float] = []
        nonzero_thresholds: list[float] = []

        for channel in range(channel_count):
            prefill_max = prefill_values[channel] if channel < len(prefill_values) else float("-inf")
            decode_max = decode_values[channel] if channel < len(decode_values) else float("-inf")
            prefill_abs_max = abs(prefill_max) if math.isfinite(prefill_max) else 0.0
            decode_abs_max = abs(decode_max) if math.isfinite(decode_max) else 0.0
            abs_max = max(prefill_abs_max, decode_abs_max)
            threshold = abs_max * args.channel_threshold_ratio
            combined_abs_max.append(abs_max)
            if threshold > 0.0:
                generated_rows.append((layer, channel, threshold))
                nonzero_thresholds.append(threshold)

        prefill_abs = [abs(value) for value in prefill_values if math.isfinite(value)]
        decode_abs = [abs(value) for value in decode_values if math.isfinite(value)]
        report_rows.append(
            {
                "layer": layer,
                "channel_count": channel_count,
                "prefill_channel_count": len(prefill_values),
                "decode_channel_count": len(decode_values),
                "channel_threshold_ratio": f"{args.channel_threshold_ratio:.8f}",
                "prefill_abs_max_mean": f"{safe_mean(prefill_abs):.8f}",
                "prefill_abs_max_max": f"{max(prefill_abs, default=0.0):.8f}",
                "decode_abs_max_mean": f"{safe_mean(decode_abs):.8f}",
                "decode_abs_max_max": f"{max(decode_abs, default=0.0):.8f}",
                "combined_abs_max_mean": f"{safe_mean(combined_abs_max):.8f}",
                "combined_abs_max_max": f"{max(combined_abs_max, default=0.0):.8f}",
                "threshold_nonzero_channels": len(nonzero_thresholds),
                "threshold_min": f"{min(nonzero_thresholds, default=0.0):.8f}",
                "threshold_mean": f"{safe_mean(nonzero_thresholds):.8f}",
                "threshold_max": f"{max(nonzero_thresholds, default=0.0):.8f}",
            }
        )

    generated_rows.sort(key=lambda item: (item[0], item[1]))
    write_channel_threshold_config(output_path, generated_rows, args)
    write_channel_report(report_path, report_rows)
    return len(layers), len(generated_rows)


def uses_global_reference_threshold(threshold_kind: str) -> bool:
    return threshold_kind in ("swiglu", "silu")


def global_threshold_from_reference(
    threshold_kind: str,
    collection: dict[tuple[str, int], StageData],
    layer_targets: dict[int, tuple[str, float, float]],
    scale: float,
) -> tuple[float, float]:
    if GLOBAL_REFERENCE_LAYER not in layer_targets:
        raise ValueError(
            f"{threshold_kind} global-threshold mode requires layer "
            f"{GLOBAL_REFERENCE_LAYER} in the target profile"
        )

    reference_data = collection.get((GLOBAL_REFERENCE_STAGE, GLOBAL_REFERENCE_LAYER))
    if reference_data is None:
        raise ValueError(
            f"{threshold_kind} global-threshold mode requires collection data for "
            f"{GLOBAL_REFERENCE_STAGE} layer {GLOBAL_REFERENCE_LAYER}"
        )

    reference_target = layer_targets[GLOBAL_REFERENCE_LAYER][1] * scale
    reference_threshold, _ = threshold_for_target_ratio(reference_data, reference_target)
    return reference_threshold, reference_target


def write_threshold_config(path: pathlib.Path, rows: list[tuple[int, float]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Generated from collected activation output distributions\n")
        handle.write(
            f"# collect_prefix={args.collect_prefix} target_profile={args.target_profile} scale={args.scale}\n"
        )
        handle.write("layer,threshold\n")
        for layer, threshold in rows:
            if threshold > 0.0:
                handle.write(f"{layer},{threshold:.8g}\n")


def write_report(path: pathlib.Path, report_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "layer",
        "scale",
        "final_threshold",
        "prefill_target_added_zero_ratio",
        "prefill_available_added_zero_ratio_max",
        "prefill_stage_threshold",
        "prefill_stage_estimated_added_zero_ratio",
        "prefill_final_estimated_added_zero_ratio",
        "prefill_original_zero_ratio",
        "prefill_final_zero_ratio",
        "prefill_status",
        "decode_target_added_zero_ratio",
        "decode_available_added_zero_ratio_max",
        "decode_stage_threshold",
        "decode_stage_estimated_added_zero_ratio",
        "decode_final_estimated_added_zero_ratio",
        "decode_original_zero_ratio",
        "decode_final_zero_ratio",
        "decode_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def stage_estimate(data: StageData | None, target_ratio: float, final_threshold: float) -> StageEstimate:
    if data is None:
        return StageEstimate(
            target_ratio=target_ratio,
            available_ratio=0.0,
            stage_threshold=0.0,
            stage_estimated_ratio=0.0,
            final_estimated_ratio=0.0,
            original_zero_ratio=0.0,
            final_zero_ratio=0.0,
            status="missing",
        )

    stage_threshold, stage_estimated_ratio = threshold_for_target_ratio(data, target_ratio)
    final_estimated_ratio = estimate_ratio_below_threshold(data, final_threshold)
    original_zero_ratio = data.original_zero_ratio
    status = "ok"
    if target_ratio > data.max_added_zero_ratio + 1e-12:
        status = "clamped"
    return StageEstimate(
        target_ratio=target_ratio,
        available_ratio=data.max_added_zero_ratio,
        stage_threshold=stage_threshold,
        stage_estimated_ratio=stage_estimated_ratio,
        final_estimated_ratio=final_estimated_ratio,
        original_zero_ratio=original_zero_ratio,
        final_zero_ratio=original_zero_ratio + final_estimated_ratio,
        status=status,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-layer activation threshold tables from collected prefill/decode histograms."
    )
    parser.add_argument("--collect-prefix", required=True, help="prefix used by --swiglu-threshold-collect")
    parser.add_argument("--target-profile", help="group target profile CSV (required for target-profile mode)")
    parser.add_argument("--output", required=True, help="generated layer,threshold CSV")
    parser.add_argument("--report", required=True, help="per-layer calibration report CSV")
    parser.add_argument("--scale", type=float, default=1.0, help="global multiplier applied to group targets")
    parser.add_argument(
        "--generate-mode",
        choices=("target-profile", "channel-max"),
        default="target-profile",
        help="target-profile keeps the existing layer/global-threshold path; channel-max emits per-channel thresholds from collected SiLU maxima",
    )
    parser.add_argument(
        "--channel-threshold-ratio",
        type=float,
        default=0.10,
        help="for channel-max mode: per-channel threshold = abs_channel_max * ratio",
    )
    parser.add_argument(
        "--threshold-kind",
        choices=("swiglu", "silu", "silu_input"),
        default="swiglu",
        help="activation threshold family; swiglu and silu use a prefill layer-1 anchored global threshold",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    collect_prefix = pathlib.Path(args.collect_prefix)
    output_path = pathlib.Path(args.output)
    report_path = pathlib.Path(args.report)

    if args.generate_mode == "channel-max":
        if args.threshold_kind != "silu":
            raise ValueError("channel-max mode currently requires --threshold-kind silu")
        if not 0.0 <= args.channel_threshold_ratio <= 1.0:
            raise ValueError(f"invalid channel-threshold ratio: {args.channel_threshold_ratio} (expected 0 <= ratio <= 1)")

        layer_count, threshold_count = generate_channel_thresholds(collect_prefix, output_path, report_path, args)
        print(f"generated {threshold_count} per-channel thresholds across {layer_count} layers -> {output_path}")
        print(f"report -> {report_path}")
        print(f"channel-threshold ratio: {args.channel_threshold_ratio:.4%}")
        return 0

    summary_path = pathlib.Path(f"{collect_prefix}_summary.csv")
    hist_path = pathlib.Path(f"{collect_prefix}_hist.csv")
    if not args.target_profile:
        raise ValueError("--target-profile is required for generate-mode=target-profile")
    target_profile_path = pathlib.Path(args.target_profile)

    if not summary_path.is_file():
        raise FileNotFoundError(f"missing collection summary: {summary_path}")
    if not hist_path.is_file():
        raise FileNotFoundError(f"missing collection histogram: {hist_path}")
    if not target_profile_path.is_file():
        raise FileNotFoundError(f"missing target profile: {target_profile_path}")

    collection = parse_collection(summary_path, hist_path)
    groups = parse_target_profile(target_profile_path)
    layer_targets = build_layer_targets(groups)

    seen_layers = sorted({layer for _, layer in collection})
    missing_layers = sorted(layer for layer in layer_targets if layer not in seen_layers)
    if missing_layers:
        raise ValueError(
            "target profile refers to layers not found in collection: "
            + ", ".join(str(layer) for layer in missing_layers)
        )

    generated_rows: list[tuple[int, float]] = []
    report_rows: list[dict[str, object]] = []

    prefill_total = 0
    prefill_added = 0.0
    decode_total = 0
    decode_added = 0.0

    global_threshold = 0.0
    reference_target = 0.0
    if uses_global_reference_threshold(args.threshold_kind):
        global_threshold, reference_target = global_threshold_from_reference(
            args.threshold_kind,
            collection,
            layer_targets,
            args.scale,
        )

    for layer in seen_layers:
        group_name, base_prefill_target, base_decode_target = layer_targets.get(layer, ("ungrouped", 0.0, 0.0))
        prefill_target = base_prefill_target * args.scale
        decode_target = base_decode_target * args.scale

        prefill_data = collection.get(("prefill", layer))
        decode_data = collection.get(("decode", layer))

        prefill_stage_threshold = threshold_for_target_ratio(prefill_data, prefill_target)[0] if prefill_data else 0.0
        decode_stage_threshold = threshold_for_target_ratio(decode_data, decode_target)[0] if decode_data else 0.0
        if uses_global_reference_threshold(args.threshold_kind):
            final_threshold = global_threshold
        else:
            final_threshold = combine_stage_thresholds(
                (prefill_data, prefill_stage_threshold),
                (decode_data, decode_stage_threshold),
            )

        prefill = stage_estimate(prefill_data, prefill_target, final_threshold)
        decode = stage_estimate(decode_data, decode_target, final_threshold)

        if final_threshold > 0.0:
            generated_rows.append((layer, final_threshold))

        if prefill_data is not None:
            prefill_total += prefill_data.total_values
            prefill_added += prefill.final_estimated_ratio * prefill_data.total_values
        if decode_data is not None:
            decode_total += decode_data.total_values
            decode_added += decode.final_estimated_ratio * decode_data.total_values

        report_rows.append(
            {
                "group": group_name,
                "layer": layer,
                "scale": f"{args.scale:.6f}",
                "final_threshold": f"{final_threshold:.8g}",
                "prefill_target_added_zero_ratio": f"{prefill.target_ratio:.8f}",
                "prefill_available_added_zero_ratio_max": f"{prefill.available_ratio:.8f}",
                "prefill_stage_threshold": f"{prefill.stage_threshold:.8g}",
                "prefill_stage_estimated_added_zero_ratio": f"{prefill.stage_estimated_ratio:.8f}",
                "prefill_final_estimated_added_zero_ratio": f"{prefill.final_estimated_ratio:.8f}",
                "prefill_original_zero_ratio": f"{prefill.original_zero_ratio:.8f}",
                "prefill_final_zero_ratio": f"{prefill.final_zero_ratio:.8f}",
                "prefill_status": prefill.status,
                "decode_target_added_zero_ratio": f"{decode.target_ratio:.8f}",
                "decode_available_added_zero_ratio_max": f"{decode.available_ratio:.8f}",
                "decode_stage_threshold": f"{decode.stage_threshold:.8g}",
                "decode_stage_estimated_added_zero_ratio": f"{decode.stage_estimated_ratio:.8f}",
                "decode_final_estimated_added_zero_ratio": f"{decode.final_estimated_ratio:.8f}",
                "decode_original_zero_ratio": f"{decode.original_zero_ratio:.8f}",
                "decode_final_zero_ratio": f"{decode.final_zero_ratio:.8f}",
                "decode_status": decode.status,
            }
        )

    generated_rows.sort(key=lambda item: item[0])
    write_threshold_config(output_path, generated_rows, args)
    write_report(report_path, report_rows)

    prefill_global = 0.0 if prefill_total == 0 else prefill_added / prefill_total
    decode_global = 0.0 if decode_total == 0 else decode_added / decode_total

    print(f"generated {len(generated_rows)} layer thresholds -> {output_path}")
    print(f"report -> {report_path}")
    if uses_global_reference_threshold(args.threshold_kind):
        print(
            f"{args.threshold_kind} global threshold anchored at "
            f"{GLOBAL_REFERENCE_STAGE} layer {GLOBAL_REFERENCE_LAYER} "
            f"target {reference_target:.4%}: {global_threshold:.8g}"
        )
    print(f"estimated prefill added-zero ratio: {prefill_global:.4%}")
    print(f"estimated decode added-zero ratio:  {decode_global:.4%}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
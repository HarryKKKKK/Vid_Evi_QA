#!/usr/bin/env python3
"""Calibrate per-model, per-scale thresholds from contained-gold controls."""

from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path

from common import PIPELINE_VERSION, atomic_write_json, latest_jsonl_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = str(ROOT / "nextgqa_pipeline/insufficient/measurements.*.jsonl")
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/thresholds.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default=DEFAULT_GLOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--minimum-controls", type=int, default=20)
    parser.add_argument(
        "--residual-ratio-threshold", type=float, default=0.10,
        help="Separate global-residual gate; report sensitivity to this value in experiments.",
    )
    return parser.parse_args()


def lower_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("empty quantile input")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.floor(probability * (len(ordered) - 1))))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.target_recall <= 1:
        raise SystemExit("--target-recall must be in (0,1]")
    paths = [Path(path) for path in glob.glob(args.measurements)]
    if not paths:
        raise SystemExit(f"no measurement files matched {args.measurements!r}")
    records = latest_jsonl_records(paths)
    controls: dict[tuple[str, str], list[float]] = defaultdict(list)
    discovery_reference: dict[tuple[str, str], list[float]] = defaultdict(list)
    controls_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    discovery_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    model_item_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records.values():
        group = "multi_interval" if int(record.get("evidence_count", 0)) >= 2 else "single_interval"
        for tag, model in record.get("models", {}).items():
            model_item_stats[tag]["total"] += 1
            model_item_stats[tag][f"{group}_total"] += 1
            if model.get("status") != "ok":
                model_item_stats[tag]["error"] += 1
                continue
            if model.get("eligible"):
                model_item_stats[tag]["eligible"] += 1
                model_item_stats[tag][f"{group}_eligible"] += 1
            else:
                model_item_stats[tag]["ineligible"] += 1
                continue
            for key, scale in model.get("scales", {}).items():
                bucket = (tag, key)
                for window in scale.get("windows", []):
                    value = window.get("discovery", {}).get("normalized_gain")
                    if value is not None:
                        discovery_reference[bucket].append(float(value))
                        discovery_by_group[(tag, key, group)].append(float(value))
                    positive = window.get("positive_control", {}).get("normalized_gain")
                    if positive is not None:
                        controls[bucket].append(float(positive))
                        controls_by_group[(tag, key, group)].append(float(positive))

    model_thresholds: dict[str, dict] = {}
    tags = sorted({tag for tag, _ in set(controls) | set(discovery_reference)})
    for tag in tags:
        scale_thresholds: dict[str, dict] = {}
        scale_keys = sorted({key for model_tag, key in set(controls) | set(discovery_reference) if model_tag == tag})
        for key in scale_keys:
            positives = controls[(tag, key)]
            # The lower (1-recall) quantile gives the requested positive-control recall.
            raw_threshold = lower_quantile(positives, 1.0 - args.target_recall) if positives else None
            threshold = max(0.0, raw_threshold) if raw_threshold is not None else None
            achieved = (
                sum(value >= threshold for value in positives) / len(positives)
                if positives and threshold is not None else None
            )
            positive_at_zero = (
                sum(value >= 0.0 for value in positives) / len(positives) if positives else None
            )
            usable = (
                len(positives) >= args.minimum_controls
                and positive_at_zero is not None
                and positive_at_zero >= args.target_recall
            )
            scale_thresholds[key] = {
                "threshold": threshold,
                "raw_positive_quantile": raw_threshold,
                "target_recall": args.target_recall,
                "achieved_control_recall": achieved,
                "positive_gain_at_zero_rate": positive_at_zero,
                "usable": usable,
                "positive_controls": summary(positives),
                "unlabeled_discovery_reference": summary(discovery_reference[(tag, key)]),
                "by_evidence_group": {
                    group: {
                        "positive_controls": summary(controls_by_group[(tag, key, group)]),
                        "unlabeled_discovery_reference": summary(discovery_by_group[(tag, key, group)]),
                    }
                    for group in ("single_interval", "multi_interval")
                },
                "note": "Discovery windows are not negatives and are not used to estimate FPR.",
            }
        model_thresholds[tag] = {
            "item_counts": dict(model_item_stats[tag]),
            "scales": scale_thresholds,
        }

    output = {
        "pipeline_version": PIPELINE_VERSION,
        "measurement_files": [str(path) for path in paths],
        "record_count": len(records),
        "target_recall": args.target_recall,
        "minimum_controls": args.minimum_controls,
        "residual_ratio_threshold": args.residual_ratio_threshold,
        "models": model_thresholds,
    }
    atomic_write_json(args.output, output)
    print(f"Saved: {args.output}")
    for tag, model in model_thresholds.items():
        print(f"\n[{tag}] items={model['item_counts']}")
        for key, scale in model["scales"].items():
            print(
                f"  {key}: usable={scale['usable']} threshold={scale['threshold']} "
                f"controls={scale['positive_controls']['n']} recall={scale['achieved_control_recall']}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score answer-set recovery and option-level temporal grounding."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def parse_response(row: dict) -> dict:
    text = str((row.get("response") or {}).get("text") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group())
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def option_id(value: object) -> str | None:
    match = re.search(r"\b([A-E])\b", str(value or "").upper())
    return match.group(1) if match else None


def interval_pairs(value: object) -> list[tuple[float, float]]:
    result = []
    if isinstance(value, dict):
        if "start" in value and "end" in value:
            value = [value["start"], value["end"]]
        elif "interval" in value:
            value = value["interval"]
        else:
            for nested in value.values():
                result.extend(interval_pairs(nested))
            return result
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            a, b = map(float, value)
            if math.isfinite(a) and math.isfinite(b):
                result.append((min(a, b), max(a, b)))
        else:
            for nested in value:
                result.extend(interval_pairs(nested))
    return result


def merge(intervals: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    clipped = sorted((max(0.0, a), min(duration, b)) for a, b in intervals if b > a)
    output = []
    for start, end in clipped:
        if end <= start:
            continue
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def length(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


def intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in a for b0, b1 in b)


def overlaps(pred: list[tuple[float, float]], ref: list[tuple[float, float]]) -> tuple[float, float]:
    inter = intersection(pred, ref)
    p_len, r_len = length(pred), length(ref)
    iop = inter / p_len if p_len else 0.0
    union = p_len + r_len - inter
    iou = inter / union if union else 0.0
    return iop, iou


def score(row: dict, threshold: float) -> dict[str, float]:
    parsed = parse_response(row)
    entries = parsed.get("supported_options")
    schema_valid = isinstance(entries, list)
    if not schema_valid:
        entries = []
    pred_intervals = defaultdict(list)
    predicted = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        oid = option_id(entry.get("option_id") or entry.get("option"))
        if not oid:
            continue
        predicted.add(oid)
        pred_intervals[oid].extend(interval_pairs(entry.get("evidence_seconds")))
    reference = {str(x).upper() for x in row.get("valid_option_ids", [])}
    true_intervals = defaultdict(list)
    for option in row.get("options", []):
        oid = option_id(option.get("option_id"))
        if not oid:
            continue
        for support in option.get("supported_by", []):
            true_intervals[oid].extend(interval_pairs(support))
    duration = float(row.get("video_duration") or 0)
    tp = len(predicted & reference)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iops, ious = [], []
    for oid in sorted(reference):
        pred = merge(pred_intervals[oid], duration)
        ref = merge(true_intervals[oid], duration)
        iop, iou = overlaps(pred, ref)
        iops.append(iop); ious.append(iou)
    exact = predicted == reference
    option_accuracy = sum(((oid in predicted) == (oid in reference)) for oid in "ABCDE") / 5
    return {
        "parse_valid": float(schema_valid), "exact_set": float(exact),
        "set_precision": precision, "set_recall": recall, "set_f1": f1,
        "option_accuracy": option_accuracy,
        "multi_detect": float(len(predicted) >= 2),
        "underselect": float(bool(reference - predicted)),
        "overselect": float(bool(predicted - reference)),
        "mean_true_iop": sum(iops) / len(iops) if iops else 0.0,
        "mean_true_iou": sum(ious) / len(ious) if ious else 0.0,
        "grounded_set_iop": float(exact and bool(iops) and all(x >= threshold for x in iops)),
        "grounded_set_iou": float(exact and bool(ious) and all(x >= threshold for x in ious)),
    }


def mean_scores(rows: list[dict]) -> dict[str, float]:
    keys = rows[0]["_score"] if rows else {}
    return {key: sum(row["_score"][key] for row in rows) / len(rows) for key in keys} if rows else {}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def main() -> None:
    args = parse_args()
    paths = sorted({Path(path) for pattern in args.inputs for path in glob.glob(pattern)})
    by_model = defaultdict(list)
    errors = defaultdict(int)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row.get("model_tag") or path.stem)
            if row.get("status") != "ok":
                errors[model] += 1
                continue
            row["_score"] = score(row, args.threshold)
            by_model[model].append(row)
    report = {"analysis": {"bootstrap_replicates": args.bootstrap_replicates,
                            "seed": args.seed, "grounding_threshold": args.threshold,
                            "cluster": "source video"},
              "files": [str(x) for x in paths], "models": {}}
    for model, rows in sorted(by_model.items()):
        point = mean_scores(rows)
        clusters = defaultdict(list)
        for row in rows:
            clusters[str(row.get("video_id") or row.get("video"))].append(row)
        ids = sorted(clusters)
        draws = defaultdict(list)
        rng = random.Random(args.seed)
        for _ in range(args.bootstrap_replicates):
            sampled = []
            for _ in ids:
                sampled.extend(clusters[rng.choice(ids)])
            for key, value in mean_scores(sampled).items():
                draws[key].append(value)
        ci = {key: [percentile(values, 0.025), percentile(values, 0.975)]
              for key, values in draws.items()}
        type_scores = {}
        for kind in sorted({str(row.get("conflict_type")) for row in rows}):
            type_scores[kind] = mean_scores([row for row in rows if str(row.get("conflict_type")) == kind])
        report["models"][model] = {"records": len(rows), "source_video_clusters": len(ids),
                                    "errors": errors[model], "point_estimates": point,
                                    "cluster_bootstrap_ci": ci, "by_type": type_scores}

    def percentize(value):
        if isinstance(value, float):
            return round(value * 100, 3)
        if isinstance(value, list):
            return [percentize(x) for x in value]
        if isinstance(value, dict):
            return {k: percentize(v) for k, v in value.items()}
        return value
    report["models_percent"] = percentize(report["models"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["models_percent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

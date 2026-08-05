#!/usr/bin/env python3
"""Audit residual evidence in insufficient-video VQA results.

This script is read-only with respect to its input. It writes a compact JSON
summary and a JSONL file containing every correctly answered sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def is_answerable(row: dict[str, Any]) -> bool:
    return row.get("model_parsed", {}).get("label") == "ANSWERABLE"


def is_correct(row: dict[str, Any]) -> bool:
    return is_answerable(row) and row.get("model_parsed", {}).get("answer") == row.get("answer")


def normalized_intervals(row: dict[str, Any]) -> list[tuple[float, float]]:
    result = []
    for item in row.get("evidence_intervals", []):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        result.append((min(start, end), max(start, end)))
    return result


def model_span(row: dict[str, Any]) -> tuple[float, float] | None:
    values = row.get("model_parsed", {}).get("evidence_seconds")
    if not isinstance(values, list):
        return None
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(numbers) < 2:
        return None
    return min(numbers), max(numbers)


def point_in_intervals(point: float, intervals: Iterable[tuple[float, float]]) -> bool:
    return any(start <= point <= end for start, end in intervals)


def relation(row: dict[str, Any]) -> dict[str, Any]:
    intervals = normalized_intervals(row)
    span = model_span(row)
    sampled = [float(value) for value in row.get("sampled_seconds", [])]
    masked_sample_count = sum(point_in_intervals(value, intervals) for value in sampled)
    gt_duration_sum = sum(end - start for start, end in intervals)

    base = {
        "gt_intervals": [{"start": start, "end": end} for start, end in intervals],
        "gt_duration_sum": gt_duration_sum,
        "masked_sample_count": masked_sample_count,
        "sample_count": len(sampled),
        "model_span": None,
        "model_span_duration": None,
        "relation": "missing_or_non_interval_model_evidence",
    }
    if span is None:
        return base

    model_start, model_end = span
    overlap = any(model_start <= end and model_end >= start for start, end in intervals)
    model_inside_one_mask = any(
        model_start >= start and model_end <= end for start, end in intervals
    )
    gt_inside_model = any(
        model_start <= start and model_end >= end for start, end in intervals
    )
    if model_inside_one_mask:
        category = "model_span_inside_mask"
    elif not overlap:
        category = "model_span_outside_all_masks"
    elif gt_inside_model:
        category = "model_span_contains_a_mask"
    else:
        category = "partial_overlap"

    base.update(
        {
            "model_span": [model_start, model_end],
            "model_span_duration": model_end - model_start,
            "relation": category,
        }
    )
    return base


def safe_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def pct(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("insufficient_audit"))
    args = parser.parse_args()

    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Input must be a JSON array")

    answered = [row for row in rows if is_answerable(row)]
    correct = [row for row in rows if is_correct(row)]
    wrong = [row for row in answered if not is_correct(row)]

    enriched = []
    relation_counts: Counter[str] = Counter()
    model_durations: list[float] = []
    gt_durations: list[float] = []
    masked_sample_counts: list[float] = []
    type_stats: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        qtype = str(row.get("question_type"))
        type_stats[qtype]["total"] += 1
        if is_answerable(row):
            type_stats[qtype]["answered"] += 1
        if is_correct(row):
            type_stats[qtype]["correct"] += 1

    for row in correct:
        rel = relation(row)
        relation_counts[rel["relation"]] += 1
        if rel["model_span_duration"] is not None:
            model_durations.append(rel["model_span_duration"])
        gt_durations.append(rel["gt_duration_sum"])
        masked_sample_counts.append(rel["masked_sample_count"])
        enriched.append(
            {
                "video_id": row.get("video_id"),
                "qid": row.get("qid"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "choices": row.get("choices"),
                "gold_answer": row.get("answer"),
                "model_answer": row.get("model_parsed", {}).get("answer"),
                "model_evidence_text": row.get("model_parsed", {}).get("evidence"),
                "model_reason": row.get("model_parsed", {}).get("reason"),
                "video_duration": row.get("video_duration"),
                **rel,
            }
        )

    answer_counts = Counter(str(row.get("answer")) for row in rows)
    summary = {
        "total": len(rows),
        "labels": dict(Counter(row.get("model_parsed", {}).get("label") for row in rows)),
        "answered": len(answered),
        "correct_among_answered": len(correct),
        "wrong_among_answered": len(wrong),
        "answered_accuracy_pct": pct(len(correct), len(answered)),
        "correct_over_total_pct": pct(len(correct), len(rows)),
        "correct_relation_counts": dict(relation_counts),
        "correct_model_span_duration_seconds": safe_stats(model_durations),
        "correct_gt_interval_duration_sum_seconds": safe_stats(gt_durations),
        "correct_masked_sample_count": safe_stats(masked_sample_counts),
        "question_type": {
            key: {
                **dict(counts),
                "answered_rate_pct": pct(counts["answered"], counts["total"]),
                "correct_over_total_pct": pct(counts["correct"], counts["total"]),
                "accuracy_among_answered_pct": pct(counts["correct"], counts["answered"]),
            }
            for key, counts in sorted(type_stats.items())
        },
        "gold_answer_counts": dict(answer_counts),
        "interpretation_warning": (
            "A model-reported interval is not ground truth. Overlap does not prove leakage, "
            "and non-overlap does not prove that the timestamp is accurate. Verify selected "
            "videos visually and run text-only/all-black controls."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "correct_answered_samples.jsonl").open("w", encoding="utf-8") as handle:
        for item in enriched:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

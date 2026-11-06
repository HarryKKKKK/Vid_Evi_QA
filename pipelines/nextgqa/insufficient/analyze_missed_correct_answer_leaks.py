#!/usr/bin/env python3
"""Explain why confirmed correct-answer evidence leaks escaped the first audit."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/nextgqa/insufficient/analyze_missed_correct_answer_leaks.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'nextgqa_pipeline/insufficient/analyze_missed_correct_answer_leaks.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'nextgqa_pipeline/insufficient', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from common import latest_jsonl_records
from ground_windows import candidate_windows, threshold_for


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[2]


def interval(row: dict) -> tuple[float, float]:
    value = row["candidate"]["interval"]
    return float(value[0]), float(value[1])


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap_fraction(target: tuple[float, float], other: tuple[float, float]) -> float:
    return overlap(target, other) / max(target[1] - target[0], 1e-9)


def midpoint_in(target: tuple[float, float], other: tuple[float, float]) -> bool:
    mid = (target[0] + target[1]) / 2
    return other[0] <= mid <= other[1]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    def q(p: float) -> float:
        return ordered[round(p * (len(ordered) - 1))]
    return {
        "n": len(values), "min": min(values), "p25": q(.25),
        "median": statistics.median(values), "p75": q(.75), "max": max(values),
    }


def raw_positive_windows(measurement: dict, thresholds: dict, band: float) -> list[dict]:
    out = []
    for tag, model in measurement.get("models", {}).items():
        if model.get("status") != "ok" or not model.get("eligible"):
            continue
        for scale_key, scale in model.get("scales", {}).items():
            threshold = threshold_for(thresholds, tag, scale_key)
            if threshold is None:
                continue
            cutoff = threshold - band
            for window in scale.get("windows", []):
                score = (window.get("discovery") or {}).get("normalized_gain")
                if score is None or float(score) < cutoff:
                    continue
                out.append({
                    "interval": tuple(map(float, window["interval"])),
                    "score": float(score), "threshold": threshold,
                    "scale": scale_key, "model_tag": tag,
                })
    return out


def maximum_match(target: tuple[float, float], windows: list[dict]) -> dict:
    if not windows:
        return {"overlap_fraction": 0.0, "midpoint_covered": False, "window": None}
    best = max(
        windows,
        key=lambda row: (
            overlap_fraction(target, tuple(row["interval"])),
            midpoint_in(target, tuple(row["interval"])),
        ),
    )
    return {
        "overlap_fraction": round(overlap_fraction(target, tuple(best["interval"])), 6),
        "midpoint_covered": midpoint_in(target, tuple(best["interval"])),
        "window": best,
    }


def load_target_measurements(pattern: str, keys: set[str]) -> dict[str, dict]:
    found = {}
    for filename in glob.glob(pattern):
        with open(filename, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = str(row.get("key"))
                if key in keys:
                    found[key] = row
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifications",
        default=str(ROOT / "nextgqa_pipeline/insufficient/correct_answer_evidence_classifications8*.jsonl"),
    )
    parser.add_argument(
        "--measurements",
        default=str(ROOT / "nextgqa_pipeline/insufficient/measurements.[0-7].jsonl"),
    )
    parser.add_argument(
        "--thresholds", type=Path,
        default=ROOT / "nextgqa_pipeline/insufficient/thresholds.json",
    )
    parser.add_argument(
        "--dense-audit", type=Path,
        default=ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_item_audit.json",
    )
    parser.add_argument("--uncertainty-band", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    class_paths = [Path(p) for p in glob.glob(args.classifications)]
    latest = latest_jsonl_records(class_paths)
    leaks = [r for r in latest.values() if r.get("status") == "ok" and r.get("label") == "EVIDENCE_LEAK"]
    keys = {str(r["item_key"]) for r in leaks}
    measurements = load_target_measurements(args.measurements, keys)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    dense_payload = json.loads(args.dense_audit.read_text(encoding="utf-8"))
    dense_by_key = {str(r["item_key"]): r for r in dense_payload.get("records", [])}

    records = []
    for leak in sorted(leaks, key=lambda r: str(r["item_key"])):
        key = str(leak["item_key"])
        target = interval(leak)
        length = target[1] - target[0]
        claimed_value = leak.get("candidate", {}).get("claimed_interval", target)
        claimed = tuple(map(float, claimed_value))
        claimed_length = claimed[1] - claimed[0]
        measurement = measurements.get(key)
        model = (measurement or {}).get("models", {}).get("qwen3vl_construct", {})
        eligible = bool(model.get("eligible"))
        official = (measurement or {}).get("initial_mask") or (measurement or {}).get("official_evidence") or []
        official_overlap = sum(overlap(target, tuple(map(float, span))) for span in official)

        row = {
            "item_key": key,
            "video_id": leak.get("video_id"),
            "qid": leak.get("qid"),
            "question_type": leak.get("question_type"),
            "question": leak.get("question"),
            "new_leak_interval": list(target),
            "new_leak_interval_seconds": round(length, 6),
            "model_claimed_interval": list(claimed),
            "model_claimed_interval_seconds": round(claimed_length, 6),
            "official_mask_overlap_fraction": round(official_overlap / max(length, 1e-9), 6),
            "original_route": "eligible_scan" if eligible else "dense_fallback",
        }

        if not measurement:
            row["escape_mechanism"] = "MISSING_MEASUREMENT"
        elif not eligible:
            dense = dense_by_key.get(key, {})
            row["original_dense_item_label"] = dense.get("item_label")
            row["original_dense_window_label_counts"] = dense.get("window_label_counts")
            # Dense requests contain a 10 s core plus at most 2 s context per
            # side. A longer claim necessarily requires evidence aggregation
            # unavailable to any single original request.
            if claimed_length > 14:
                row["escape_mechanism"] = "DENSE_WINDOW_FRAGMENTATION_GT14S"
            else:
                row["escape_mechanism"] = "DENSE_SEMANTIC_OR_MODEL_DISAGREEMENT"
        else:
            minimal = candidate_windows(measurement, thresholds, args.uncertainty_band)
            selected = minimal[:args.top_k]
            raw = raw_positive_windows(measurement, thresholds, args.uncertainty_band)
            raw_match = maximum_match(target, raw)
            minimal_match = maximum_match(target, minimal)
            selected_match = maximum_match(target, selected)
            row.update({
                "original_minimal_candidate_count": len(minimal),
                "original_selected_candidate_count": len(selected),
                "raw_positive_match": raw_match,
                "minimal_candidate_match": minimal_match,
                "selected_topk_match": selected_match,
            })
            strong_selected = selected_match["midpoint_covered"] or selected_match["overlap_fraction"] >= .5
            strong_minimal = minimal_match["midpoint_covered"] or minimal_match["overlap_fraction"] >= .5
            strong_raw = raw_match["midpoint_covered"] or raw_match["overlap_fraction"] >= .5
            if strong_selected:
                row["escape_mechanism"] = "SEMANTIC_REASSESSMENT_OF_SELECTED_REGION"
            elif strong_minimal:
                row["escape_mechanism"] = "TOPK_TRUNCATION"
            elif strong_raw:
                row["escape_mechanism"] = "MINIMAL_DESCENDANT_PRUNING"
            else:
                row["escape_mechanism"] = "BEHAVIORAL_THRESHOLD_OR_LOCALIZATION_MISS"

        # This primary mechanism uses the answering model's unpadded evidence
        # claim.  Claims longer than the largest old request cannot be jointly
        # represented, regardless of whether one old window happens to cover
        # the claim midpoint.
        old_limit = 16.0 if eligible else 14.0
        if claimed_length > old_limit:
            row["primary_escape_mechanism"] = "LONG_RANGE_EVIDENCE_NOT_JOINTLY_REPRESENTABLE"
        else:
            row["primary_escape_mechanism"] = row["escape_mechanism"]
        records.append(row)

    mechanisms = Counter(r["escape_mechanism"] for r in records)
    primary_mechanisms = Counter(r["primary_escape_mechanism"] for r in records)
    routes = Counter(r["original_route"] for r in records)
    qtypes = Counter(str(r.get("question_type")) for r in records)
    dense_labels = Counter(
        str(r.get("original_dense_item_label")) for r in records if r["original_route"] == "dense_fallback"
    )
    strict_eligible = Counter()
    for row in records:
        if row["original_route"] != "eligible_scan":
            continue
        if row["selected_topk_match"]["overlap_fraction"] >= .5:
            strict_eligible["SELECTED_TOP5_COVERS_AT_LEAST_HALF"] += 1
        elif row["minimal_candidate_match"]["overlap_fraction"] >= .5:
            strict_eligible["ONLY_BELOW_TOP5_COVERS_AT_LEAST_HALF"] += 1
        elif row["raw_positive_match"]["overlap_fraction"] >= .5:
            strict_eligible["ONLY_PRE_PRUNING_WINDOW_COVERS_AT_LEAST_HALF"] += 1
        else:
            strict_eligible["NO_OLD_POSITIVE_WINDOW_COVERS_HALF"] += 1
    report = {
        "confirmed_new_leaks": len(records),
        "route_counts": dict(routes),
        "escape_mechanism_counts": dict(mechanisms),
        "primary_escape_mechanism_counts": dict(primary_mechanisms),
        "question_type_counts": dict(qtypes),
        "new_leak_interval_seconds": quantiles([r["new_leak_interval_seconds"] for r in records]),
        "model_claimed_interval_seconds": quantiles(
            [r["model_claimed_interval_seconds"] for r in records]
        ),
        "claims_longer_than_14s": sum(r["new_leak_interval_seconds"] > 14 for r in records),
        "claims_longer_than_16s": sum(r["new_leak_interval_seconds"] > 16 for r in records),
        "unpadded_claims_beyond_old_route_limit": sum(
            r["model_claimed_interval_seconds"] > (16 if r["original_route"] == "eligible_scan" else 14)
            for r in records
        ),
        "official_mask_overlap_fraction": quantiles(
            [r["official_mask_overlap_fraction"] for r in records]
        ),
        "dense_original_item_labels": dict(dense_labels),
        "strict_half_coverage_sensitivity_for_eligible": dict(strict_eligible),
        "interpretation": {
            "LONG_RANGE_EVIDENCE_NOT_JOINTLY_REPRESENTABLE": (
                "The answering model's unpadded evidence claim is longer than the old route's largest single "
                "request (16 seconds for eligible scans; 14 seconds for dense fallback)."
            ),
            "SEMANTIC_REASSESSMENT_OF_SELECTED_REGION": (
                "The original Top-5 semantic audit saw a window covering the midpoint or at least half "
                "of the newly claimed span, but did not confirm a leak."
            ),
            "TOPK_TRUNCATION": "A matching calibrated candidate existed but ranked below the five retained windows.",
            "MINIMAL_DESCENDANT_PRUNING": (
                "A matching raw positive scan window existed but ancestor pruning replaced it with smaller descendants."
            ),
            "BEHAVIORAL_THRESHOLD_OR_LOCALIZATION_MISS": (
                "No strong matching positive window survived the forced-choice behavioral detector."
            ),
            "DENSE_WINDOW_FRAGMENTATION_GT14S": (
                "The new evidence span exceeds the maximum 14-second dense grounding request, so the original "
                "audit could not inspect the complete relation in one request."
            ),
            "DENSE_SEMANTIC_OR_MODEL_DISAGREEMENT": (
                "Dense coverage existed at adequate duration; the difference is attributable to window alignment, "
                "sampling, grounding, or semantic-model judgment."
            ),
        },
        "records": records,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

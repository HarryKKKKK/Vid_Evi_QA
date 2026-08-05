#!/usr/bin/env python3
"""Finalize calibrated C3 videos with a global residual sufficiency gate."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

from common import (
    PIPELINE_VERSION,
    ForcedChoiceScorer,
    append_jsonl,
    build_freeze_video,
    coverage_stats,
    extract_frame_grid,
    frames_in_range,
    interval_adds_time,
    latest_jsonl_records,
    load_worklist,
    mask_frames,
    merge_intervals,
    minimal_positive_windows,
    normalized_gain,
    parse_endpoints,
    require_binaries,
    selected_indices,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED = ROOT / "nextgqa_pipeline/nextgqa_filtered.json"
DEFAULT_VIDEOS = ROOT / "source_datasets/next_gqa/videos"
DEFAULT_MAPPING = ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"
DEFAULT_MEASUREMENTS = str(ROOT / "nextgqa_pipeline/insufficient/measurements.*.jsonl")
DEFAULT_THRESHOLDS = ROOT / "nextgqa_pipeline/insufficient/thresholds.json"
DEFAULT_OUTPUT = ROOT / "source_datasets/next_gqa/insufficient_videos_calibrated"
DEFAULT_AUDIT = ROOT / "nextgqa_pipeline/insufficient/finalize_audit.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL")
    parser.add_argument("--uncertainty-band", type=float, default=0.05)
    parser.add_argument("--discovery-padding", type=float)
    parser.add_argument("--maximum-coverage", type=float, default=0.60)
    parser.add_argument("--minimum-visible-seconds", type=float, default=10.0)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-build", action="store_true", help="Run decisions but do not encode final videos")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def threshold_for(config: dict, tag: str, scale_key: str) -> float | None:
    scale = config.get("models", {}).get(tag, {}).get("scales", {}).get(scale_key, {})
    if not scale.get("usable"):
        return None
    value = scale.get("threshold")
    return float(value) if value is not None else None


def initial_candidates(measurement: dict, thresholds: dict, uncertainty_band: float) -> tuple[list, list]:
    intervals: list[tuple[float, float]] = []
    metadata: list[dict] = []
    for tag, model in measurement.get("models", {}).items():
        if model.get("status") != "ok" or not model.get("eligible"):
            continue
        for key, scale in model.get("scales", {}).items():
            threshold = threshold_for(thresholds, tag, key)
            if threshold is None:
                continue
            cutoff = threshold - uncertainty_band
            for window in scale.get("windows", []):
                score = window.get("discovery", {}).get("normalized_gain")
                if score is None or float(score) < cutoff:
                    continue
                interval = tuple(map(float, window["interval"]))
                intervals.append(interval)
                metadata.append({
                    "round": 1,
                    "model_tag": tag,
                    "scale": key,
                    "interval": list(interval),
                    "normalized_gain": float(score),
                    "threshold": threshold,
                    "uncertain": float(score) < threshold,
                })
    return minimal_positive_windows(intervals), metadata


def eligible_model_tags(measurement: dict, thresholds: dict, endpoint_tags: set[str]) -> list[str]:
    tags: list[str] = []
    for tag, model in measurement.get("models", {}).items():
        if tag not in endpoint_tags or model.get("status") != "ok" or not model.get("eligible"):
            continue
        if any(threshold_for(thresholds, tag, key) is not None for key in model.get("scales", {})):
            tags.append(tag)
    return sorted(tags)


def score_global_residual(item, frames, measurement, scorers, tags, residual_threshold) -> tuple[dict, list[str]]:
    results: dict[str, dict] = {}
    leaking: list[str] = []
    for tag in tags:
        score = scorers[tag].score(frames, item)
        model_measurement = measurement["models"][tag]
        blind_margin = float(model_measurement["full"]["blind"]["margin"])
        full_gain = float(model_measurement["full"]["gain"])
        residual_gain = float(score["margin"]) - blind_margin
        ratio = normalized_gain(residual_gain, full_gain)
        results[tag] = {
            "visual": score,
            "blind_margin_reused": blind_margin,
            "residual_gain": residual_gain,
            "residual_ratio": ratio,
            "threshold": residual_threshold,
            "leaking": ratio is None or ratio > residual_threshold,
        }
        if results[tag]["leaking"]:
            leaking.append(tag)
    return results, leaking


def second_round_candidates(item, residual_frames, measurement, thresholds, scorers, leaking_tags, uncertainty_band):
    intervals: list[tuple[float, float]] = []
    metadata: list[dict] = []
    for tag in leaking_tags:
        model = measurement["models"][tag]
        full_gain = float(model["full"]["gain"])
        for key, scale in model.get("scales", {}).items():
            threshold = threshold_for(thresholds, tag, key)
            if threshold is None:
                continue
            cutoff = threshold - uncertainty_band
            for window in scale.get("windows", []):
                start, end = map(float, window["interval"])
                visual = scorers[tag].score(frames_in_range(residual_frames, start, end), item)
                blind_margin = float(window["discovery"]["blind"]["margin"])
                gain = float(visual["margin"]) - blind_margin
                ratio = normalized_gain(gain, full_gain)
                record = {
                    "round": 2,
                    "model_tag": tag,
                    "scale": key,
                    "interval": [start, end],
                    "gain": gain,
                    "normalized_gain": ratio,
                    "threshold": threshold,
                    "visual": visual,
                }
                if ratio is not None and ratio >= cutoff:
                    intervals.append((start, end))
                    record["uncertain"] = ratio < threshold
                    metadata.append(record)
    return minimal_positive_windows(intervals), metadata


def process_item(item, measurement, thresholds, args, scorers) -> dict:
    started = time.time()
    output = args.output_dir / f"{item.video_id}_q{item.qid}_freeze.mp4"
    base = {
        "key": item.key,
        "index": item.index,
        "video_id": item.video_id,
        "qid": item.qid,
        "pipeline_version": PIPELINE_VERSION,
        "source_video_path": str(item.video_path),
        "output_video_path": str(output),
        "gold": item.gold,
        "official_evidence": [list(span) for span in item.evidence],
        "evidence_count": item.evidence_count,
        "evidence_group": "multi_interval" if item.evidence_count >= 2 else "single_interval",
        "question_type": item.question_type,
        "duration": item.duration,
    }
    if measurement.get("status") == "discarded_invalid_gold":
        return {
            **base,
            "status": "discarded_invalid_gold",
            "audit_status": "discarded",
            "discard_reason": measurement.get("discard_reason", "non-unique A-E gold mapping"),
        }
    if measurement.get("status") != "ok":
        return {**base, "status": "measurement_incomplete", "audit_status": "needs_review"}
    tags = eligible_model_tags(measurement, thresholds, set(scorers))
    if not tags:
        return {**base, "status": "no_calibrated_model", "audit_status": "needs_review"}

    holder = None
    try:
        sampling = measurement["sampling"]
        fps = float(sampling["fps"])
        padding = args.discovery_padding if args.discovery_padding is not None else 1.0 / fps
        holder, full_frames = extract_frame_grid(
            item.video_path,
            fps=fps,
            width=int(sampling["frame_width"]),
            ffmpeg=args.ffmpeg,
            temp_root=args.temp_root,
        )
        full_frames = [frame for frame in full_frames if frame.timestamp < item.duration + 1e-6]
        if not full_frames:
            raise RuntimeError("no sampled frames within annotated duration")
        candidate_windows, candidate_metadata = initial_candidates(
            measurement, thresholds, args.uncertainty_band
        )
        discovered = merge_intervals(candidate_windows, duration=item.duration, padding=padding)
        removed = merge_intervals(
            [*measurement["initial_mask"], *discovered], duration=item.duration
        )
        coverage = coverage_stats(removed, item.duration)
        if coverage["coverage_ratio"] > args.maximum_coverage or coverage["visible_seconds"] < args.minimum_visible_seconds:
            return {
                **base,
                "status": "coverage_limit_round1",
                "audit_status": "discarded",
                "eligible_models": tags,
                "candidate_windows": candidate_metadata,
                "removed_intervals": [list(span) for span in removed],
                "coverage": coverage,
            }

        residual = mask_frames(full_frames, removed, blur_sigma=args.blur_sigma)
        residual_threshold = float(thresholds["residual_ratio_threshold"])
        global_round1, leaking = score_global_residual(
            item, residual, measurement, scorers, tags, residual_threshold
        )
        global_round2 = None
        second_metadata: list[dict] = []
        if leaking:
            second_windows, second_metadata = second_round_candidates(
                item, residual, measurement, thresholds, scorers,
                leaking, args.uncertainty_band,
            )
            new_windows = [window for window in second_windows if interval_adds_time(window, removed)]
            if not new_windows:
                return {
                    **base,
                    "status": "distributed_or_scene_leak",
                    "audit_status": "discarded",
                    "eligible_models": tags,
                    "candidate_windows": candidate_metadata,
                    "second_round_windows": second_metadata,
                    "removed_intervals": [list(span) for span in removed],
                    "coverage": coverage,
                    "global_round1": global_round1,
                }
            new_discovered = merge_intervals(new_windows, duration=item.duration, padding=padding)
            removed = merge_intervals([*removed, *new_discovered], duration=item.duration)
            coverage = coverage_stats(removed, item.duration)
            if coverage["coverage_ratio"] > args.maximum_coverage or coverage["visible_seconds"] < args.minimum_visible_seconds:
                return {
                    **base,
                    "status": "coverage_limit_round2",
                    "audit_status": "discarded",
                    "eligible_models": tags,
                    "candidate_windows": candidate_metadata,
                    "second_round_windows": second_metadata,
                    "removed_intervals": [list(span) for span in removed],
                    "coverage": coverage,
                    "global_round1": global_round1,
                }
            residual = mask_frames(full_frames, removed, blur_sigma=args.blur_sigma)
            global_round2, still_leaking = score_global_residual(
                item, residual, measurement, scorers, tags, residual_threshold
            )
            if still_leaking:
                return {
                    **base,
                    "status": "residual_leak_after_round2",
                    "audit_status": "discarded",
                    "eligible_models": tags,
                    "candidate_windows": candidate_metadata,
                    "second_round_windows": second_metadata,
                    "removed_intervals": [list(span) for span in removed],
                    "coverage": coverage,
                    "global_round1": global_round1,
                    "global_round2": global_round2,
                }

        transform = None
        if not args.no_build:
            transform = build_freeze_video(
                item.video_path,
                output,
                removed,
                expected_duration=item.duration,
                sampling_fps=fps,
                blur_sigma=args.blur_sigma,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                threads=args.threads,
            )
        return {
            **base,
            "status": "ok" if not args.no_build else "pass_no_build",
            "audit_status": "complete",
            "eligible_models": tags,
            "candidate_windows": candidate_metadata,
            "second_round_windows": second_metadata,
            "removed_intervals": [list(span) for span in removed],
            "coverage": coverage,
            "global_round1": global_round1,
            "global_round2": global_round2,
            "final_transform": transform,
            "source_sha256": sha256_file(item.video_path) if not args.no_build else None,
            "output_sha256": sha256_file(output) if not args.no_build else None,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        output.unlink(missing_ok=True)
        return {
            **base,
            "status": "error",
            "audit_status": "needs_review",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    finally:
        if holder is not None:
            holder.cleanup()


def main() -> None:
    args = parse_args()
    require_binaries(args.ffmpeg, args.ffprobe)
    if not 0 < args.maximum_coverage < 1:
        raise SystemExit("--maximum-coverage must be in (0,1)")
    measurement_paths = [Path(path) for path in glob.glob(args.measurements)]
    if not measurement_paths:
        raise SystemExit(f"no measurements matched {args.measurements!r}")
    measurements = latest_jsonl_records(measurement_paths)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    endpoints = parse_endpoints(args.endpoint)
    scorers = {
        endpoint.tag: ForcedChoiceScorer(
            endpoint, timeout=args.timeout, top_logprobs=args.top_logprobs
        )
        for endpoint in endpoints
    }
    items = load_worklist(args.filtered_json, args.video_dir, args.video_id_mapping)
    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(items), shard, args.num_shards, args.limit)
    audit = args.audit
    if shard is not None:
        audit = audit.with_name(f"{audit.stem}.{shard}{audit.suffix}")
    previous = latest_jsonl_records([audit]) if audit.exists() else {}

    print(
        f"[START] selected={len(indices)} shard={shard}/{args.num_shards} "
        f"models={list(scorers)} residual_threshold={thresholds['residual_ratio_threshold']}",
        flush=True,
    )
    for position, index in enumerate(indices, start=1):
        item = items[index]
        old = previous.get(item.key)
        output = args.output_dir / f"{item.video_id}_q{item.qid}_freeze.mp4"
        if old and not args.overwrite:
            if old.get("status") == "ok" and output.is_file():
                print(f"[{position}/{len(indices)}] {item.key} skip ok", flush=True)
                continue
            if old.get("audit_status") == "discarded":
                print(f"[{position}/{len(indices)}] {item.key} skip discarded={old.get('status')}", flush=True)
                continue
        if not args.no_build:
            # A recomputed/discarded item must never leave a stale final video
            # from an earlier threshold or failed attempt.
            output.unlink(missing_ok=True)
        measurement = measurements.get(item.key)
        if measurement is None:
            record = {
                "key": item.key,
                "video_id": item.video_id,
                "qid": item.qid,
                "status": "missing_measurement",
                "audit_status": "needs_review",
            }
        else:
            record = process_item(item, measurement, thresholds, args, scorers)
        append_jsonl(audit, record)
        print(
            f"[{position}/{len(indices)}] {item.key} {record['status']} "
            f"coverage={record.get('coverage', {}).get('coverage_ratio', '?')}",
            flush=True,
        )
    print(f"[DONE] audit={audit} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

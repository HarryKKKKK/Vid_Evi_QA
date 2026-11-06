#!/usr/bin/env python3
"""Measure full/blind/gold and dense multi-scale residual-window sufficiency.

This stage never creates videos and never uses an ANSWERABLE/UNANSWERABLE
classification.  Official evidence is frozen in memory before discovery-window
scoring, while positive controls are measured on original frames.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

from common import (
    PIPELINE_VERSION,
    PROMPT_VERSION,
    ForcedChoiceScorer,
    append_jsonl,
    blind_frames,
    extract_frame_grid,
    frames_in_intervals,
    frames_in_range,
    latest_jsonl_records,
    load_worklist,
    mask_frames,
    merge_intervals,
    normalized_gain,
    parse_endpoints,
    parse_scales,
    require_binaries,
    scale_key,
    selected_indices,
    sliding_windows,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED = ROOT / "nextgqa_pipeline/nextgqa_filtered.json"
DEFAULT_VIDEOS = ROOT / "source_datasets/next_gqa/videos"
DEFAULT_MAPPING = ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/measurements.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL",
        help="Construction model endpoint. Repeat for a multi-model union.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frame-width", type=int, default=336)
    parser.add_argument("--scales", default="4:2,8:4,16:8")
    parser.add_argument("--official-padding", type=float, default=1.0)
    parser.add_argument("--merge-gap", type=float, default=0.0)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--full-gain-epsilon", type=float, default=1e-6)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-status", default="error,needs_review")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def contains_all_evidence(start: float, end: float, evidence) -> bool:
    return all(start <= float(a) + 1e-6 and end >= float(b) - 1e-6 for a, b in evidence)


def process_item(item, args, scorers, scales) -> dict:
    started = time.time()
    base = {
        "key": item.key,
        "index": item.index,
        "video_id": item.video_id,
        "qid": item.qid,
        "question": item.question,
        "choices": list(item.choices),
        "gold": item.gold,
        "question_type": item.question_type,
        "evidence_count": item.evidence_count,
        "official_evidence": [list(span) for span in item.evidence],
        "duration": item.duration,
        "video_path": str(item.video_path),
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "sampling": {
            "fps": args.fps,
            "frame_width": args.frame_width,
            "scales": [{"width": width, "stride": stride} for width, stride in scales],
        },
        "masking": {
            "official_padding": args.official_padding,
            "merge_gap": args.merge_gap,
            "blur_sigma": args.blur_sigma,
            "mode": "blurred_safe_frame_or_gray",
        },
    }
    if item.gold is None:
        return {
            **base,
            "status": "discarded_invalid_gold",
            "audit_status": "discarded",
            "discard_reason": "answer text does not map uniquely to one A-E option",
        }
    if not item.video_path.is_file():
        return {**base, "status": "missing_video", "audit_status": "needs_review"}

    holder = None
    try:
        holder, full_frames = extract_frame_grid(
            item.video_path,
            fps=args.fps,
            width=args.frame_width,
            ffmpeg=args.ffmpeg,
            temp_root=args.temp_root,
        )
        full_frames = [frame for frame in full_frames if frame.timestamp < item.duration + 1e-6]
        if not full_frames:
            raise RuntimeError("no sampled frames within annotated duration")
        padded_official = merge_intervals(
            item.evidence,
            duration=item.duration,
            padding=args.official_padding,
            merge_gap=args.merge_gap,
        )
        gold_frames = frames_in_intervals(full_frames, item.evidence)
        seed_frames = mask_frames(full_frames, padded_official, blur_sigma=args.blur_sigma)
        full_blind = blind_frames(full_frames)
        gold_blind = blind_frames(gold_frames)

        model_results: dict[str, dict] = {}
        model_errors = 0
        for tag, scorer in scorers.items():
            try:
                full_visual = scorer.score(full_frames, item)
                full_blind_score = scorer.score(full_blind, item)
                full_gain = float(full_visual["margin"]) - float(full_blind_score["margin"])
                gold_visual = scorer.score(gold_frames, item)
                gold_blind_score = scorer.score(gold_blind, item)
                gold_gain = float(gold_visual["margin"]) - float(gold_blind_score["margin"])
                eligible = (
                    full_gain > args.full_gain_epsilon
                    and gold_gain > args.full_gain_epsilon
                    and full_visual["prediction"] == item.gold
                    and gold_visual["prediction"] == item.gold
                )
                model_record: dict = {
                    "status": "ok",
                    "eligible": eligible,
                    "full": {"visual": full_visual, "blind": full_blind_score, "gain": full_gain},
                    "gold_only": {"visual": gold_visual, "blind": gold_blind_score, "gain": gold_gain},
                    "scales": {},
                }
                for width, stride in scales:
                    key = scale_key(width, stride)
                    window_records: list[dict] = []
                    for window_index, (start, end) in enumerate(sliding_windows(item.duration, width, stride)):
                        seed_window = frames_in_range(seed_frames, start, end)
                        original_window = frames_in_range(full_frames, start, end)
                        window_blind = blind_frames(seed_window)
                        blind_score = scorer.score(window_blind, item)
                        discovery_score = scorer.score(seed_window, item)
                        discovery_gain = float(discovery_score["margin"]) - float(blind_score["margin"])
                        record = {
                            "window_index": window_index,
                            "interval": [round(start, 6), round(end, 6)],
                            "discovery": {
                                "visual": discovery_score,
                                "blind": blind_score,
                                "gain": discovery_gain,
                                "normalized_gain": normalized_gain(discovery_gain, full_gain),
                            },
                            "is_gold_contained_control": contains_all_evidence(start, end, item.evidence),
                        }
                        if record["is_gold_contained_control"]:
                            positive_score = scorer.score(original_window, item)
                            positive_gain = float(positive_score["margin"]) - float(blind_score["margin"])
                            record["positive_control"] = {
                                "visual": positive_score,
                                "blind": blind_score,
                                "gain": positive_gain,
                                "normalized_gain": normalized_gain(positive_gain, full_gain),
                            }
                        window_records.append(record)
                    model_record["scales"][key] = {
                        "width": width,
                        "stride": stride,
                        "windows": window_records,
                    }
                model_results[tag] = model_record
            except Exception as exc:
                model_errors += 1
                model_results[tag] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        status = "ok" if model_errors == 0 else "needs_review"
        return {
            **base,
            "status": status,
            "audit_status": "complete" if status == "ok" else "needs_review",
            "frame_count": len(full_frames),
            "gold_frame_count": len(gold_frames),
            "initial_mask": [list(span) for span in padded_official],
            "models": model_results,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
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
    endpoints = parse_endpoints(args.endpoint)
    scales = parse_scales(args.scales)
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
    output = args.output
    if shard is not None:
        output = output.with_name(f"{output.stem}.{shard}{output.suffix}")

    previous = latest_jsonl_records([output]) if output.exists() else {}
    retry_statuses = {part.strip() for part in args.retry_status.split(",") if part.strip()}
    print(
        f"[START] items={len(items)} selected={len(indices)} shard={shard}/{args.num_shards} "
        f"fps={args.fps} scales={scales} models={[e.tag for e in endpoints]} output={output}",
        flush=True,
    )
    done = 0
    for position, index in enumerate(indices, start=1):
        item = items[index]
        old = previous.get(item.key)
        if old and not args.overwrite and old.get("status") not in retry_statuses:
            print(f"[{position}/{len(indices)}] {item.key} skip status={old.get('status')}", flush=True)
            continue
        record = process_item(item, args, scorers, scales)
        append_jsonl(output, record)
        done += 1
        print(
            f"[{position}/{len(indices)}] {item.key} {record['status']} "
            f"elapsed={record.get('elapsed_seconds', 0):.1f}s",
            flush=True,
        )
    print(f"[DONE] wrote={done} output={output}", flush=True)


if __name__ == "__main__":
    main()

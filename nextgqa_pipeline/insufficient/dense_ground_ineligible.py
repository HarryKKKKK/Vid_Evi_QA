#!/usr/bin/env python3
"""Densely ground the masked residual video for construction-model-ineligible items.

This fallback does not use window logprob changes.  It partitions each video
into non-overlapping cores, adds visual context on both sides, freezes official
evidence frames, and asks the existing answer-blind grounder for a factual
description of every resulting window.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Sequence

from common import (
    append_jsonl,
    extract_frame_grid,
    frames_in_range,
    latest_jsonl_records,
    load_worklist,
    mask_frames,
    parse_endpoints,
    require_binaries,
    selected_indices,
)
from ground_windows import GROUNDING_PROMPT_VERSION, WindowGrounder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED = ROOT / "nextgqa_pipeline/nextgqa_filtered.json"
DEFAULT_VIDEOS = ROOT / "source_datasets/next_gqa/videos"
DEFAULT_MAPPING = ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"
DEFAULT_MEASUREMENTS = str(ROOT / "nextgqa_pipeline/insufficient/measurements.*.jsonl")
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_groundings.jsonl"
DENSE_PIPELINE_VERSION = "nextgqa-dense-ineligible-grounding-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL")
    parser.add_argument("--eligibility-model-tag", default="qwen3vl_construct")
    parser.add_argument("--core-seconds", type=float, default=10.0)
    parser.add_argument("--context-padding", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-width", type=int, default=336)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int, help="Limit selected ineligible items, not windows.")
    parser.add_argument("--retry-status", default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def dense_cores(duration: float, core_seconds: float) -> list[tuple[float, float]]:
    """Return gap-free, non-overlapping cores; the final core may be shorter."""
    if duration <= 0 or core_seconds <= 0:
        raise ValueError(f"invalid duration/core: duration={duration}, core={core_seconds}")
    cores: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + core_seconds)
        cores.append((round(start, 6), round(end, 6)))
        start = end
    return cores


def grounding_interval(
    core: Sequence[float], duration: float, context_padding: float
) -> tuple[float, float]:
    return (
        round(max(0.0, float(core[0]) - context_padding), 6),
        round(min(duration, float(core[1]) + context_padding), 6),
    )


def core_key(item_key: str, core_index: int, core: Sequence[float]) -> str:
    return f"{item_key}__dense{core_index:03d}_{float(core[0]):.3f}_{float(core[1]):.3f}"


def eligibility_state(measurement: dict | None, model_tag: str) -> str:
    if not measurement or measurement.get("status") != "ok":
        return "unmeasured"
    model = measurement.get("models", {}).get(model_tag)
    if not model or model.get("status") != "ok" or "eligible" not in model:
        return "model_error"
    return "eligible" if bool(model["eligible"]) else "ineligible"


def main() -> None:
    args = parse_args()
    if args.core_seconds <= 0 or args.context_padding < 0:
        raise SystemExit("--core-seconds must be positive and --context-padding non-negative")
    if args.fps <= 0 or args.frame_width <= 0:
        raise SystemExit("--fps and --frame-width must be positive")
    require_binaries(args.ffmpeg, args.ffprobe)
    endpoints = parse_endpoints(args.endpoint)
    if len(endpoints) != 1:
        raise SystemExit("dense grounding requires exactly one --endpoint")
    grounder = WindowGrounder(
        endpoints[0], timeout=args.timeout, max_tokens=args.max_tokens, retries=args.api_retries
    )

    measurement_paths = [Path(path) for path in glob.glob(args.measurements)]
    if not measurement_paths:
        raise SystemExit(f"no measurement files match {args.measurements!r}")
    measurements = latest_jsonl_records(measurement_paths)
    all_items = load_worklist(args.filtered_json, args.video_dir, args.video_id_mapping)
    states = {
        item.key: eligibility_state(measurements.get(item.key), args.eligibility_model_tag)
        for item in all_items
    }
    items = [item for item in all_items if states[item.key] == "ineligible"]

    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(items), shard, args.num_shards, args.limit)
    output = args.output
    if shard is not None:
        output = output.with_name(f"{output.stem}.{shard}{output.suffix}")
    previous = latest_jsonl_records([output]) if output.exists() else {}
    retry_statuses = {part.strip() for part in args.retry_status.split(",") if part.strip()}

    state_counts = {state: sum(value == state for value in states.values()) for state in sorted(set(states.values()))}
    print(
        f"[START] all_items={len(all_items)} states={state_counts} ineligible={len(items)} "
        f"selected={len(indices)} shard={shard}/{args.num_shards} core={args.core_seconds}s "
        f"context=+/-{args.context_padding}s fps={args.fps} output={output}",
        flush=True,
    )

    written = grounded = errors = 0
    for position, index in enumerate(indices, start=1):
        item = items[index]
        measurement = measurements[item.key]
        cores = dense_cores(item.duration, args.core_seconds)
        pending_indices = []
        for core_index, core in enumerate(cores):
            old = previous.get(core_key(item.key, core_index, core))
            if args.overwrite or old is None or old.get("status") in retry_statuses:
                pending_indices.append(core_index)
        if not pending_indices:
            print(f"[{position}/{len(indices)}] {item.key} skip cores={len(cores)}", flush=True)
            continue

        holder = None
        try:
            holder, frames = extract_frame_grid(
                item.video_path,
                fps=args.fps,
                width=args.frame_width,
                ffmpeg=args.ffmpeg,
                temp_root=args.temp_root,
            )
            mask_intervals = (
                measurement.get("initial_mask")
                or measurement.get("official_evidence")
                or item.evidence
            )
            masked_frames = mask_frames(frames, mask_intervals, blur_sigma=args.blur_sigma)
            for core_index in pending_indices:
                core = cores[core_index]
                window = grounding_interval(core, item.duration, args.context_padding)
                key = core_key(item.key, core_index, core)
                started = time.time()
                candidate = {
                    "source": "dense_ineligible",
                    "interval": list(core),
                    "core_interval": list(core),
                    "grounding_interval": list(window),
                    "core_seconds": args.core_seconds,
                    "context_padding": args.context_padding,
                }
                base = {
                    "key": key,
                    "item_key": item.key,
                    "item_index": item.index,
                    "video_id": item.video_id,
                    "qid": item.qid,
                    "question": item.question,
                    "question_type": item.question_type,
                    "reference_answer": (
                        item.choices[ord(item.gold) - ord("A")] if item.gold in tuple("ABCDE") else None
                    ),
                    "candidate_rank": core_index + 1,
                    "candidate": candidate,
                    "construction_model_tag": args.eligibility_model_tag,
                    "construction_model_eligible": False,
                    "official_evidence": [list(span) for span in item.evidence],
                    "masked_intervals": [list(span) for span in mask_intervals],
                    "sampling": {"fps": args.fps, "frame_width": args.frame_width},
                    "masking": {"mode": "blurred_safe_frame_or_gray", "blur_sigma": args.blur_sigma},
                    "pipeline_version": DENSE_PIPELINE_VERSION,
                    "prompt_version": GROUNDING_PROMPT_VERSION,
                    "grounding_input_policy": "question_and_type_only_no_choices_no_gold",
                }
                try:
                    selected = frames_in_range(masked_frames, *window)
                    result = grounder.describe(selected, item.question, item.question_type)
                    record = {
                        **base,
                        "status": "ok",
                        "grounding": result,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    grounded += 1
                except Exception as exc:
                    record = {
                        **base,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    errors += 1
                append_jsonl(output, record)
                previous[key] = record
                written += 1
                print(
                    f"[{position}/{len(indices)}] {key} core={core} input={window} "
                    f"{record['status']} elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            # Record every affected core so extraction failures remain auditable
            # and can be retried through the normal status mechanism.
            for core_index in pending_indices:
                core = cores[core_index]
                key = core_key(item.key, core_index, core)
                record = {
                    "key": key,
                    "item_key": item.key,
                    "item_index": item.index,
                    "video_id": item.video_id,
                    "qid": item.qid,
                    "question": item.question,
                    "question_type": item.question_type,
                    "reference_answer": (
                        item.choices[ord(item.gold) - ord("A")] if item.gold in tuple("ABCDE") else None
                    ),
                    "candidate_rank": core_index + 1,
                    "candidate": {
                        "source": "dense_ineligible",
                        "interval": list(core),
                        "core_interval": list(core),
                        "grounding_interval": list(
                            grounding_interval(core, item.duration, args.context_padding)
                        ),
                        "core_seconds": args.core_seconds,
                        "context_padding": args.context_padding,
                    },
                    "status": "error",
                    "error": f"item extraction failed: {type(exc).__name__}: {exc}",
                    "pipeline_version": DENSE_PIPELINE_VERSION,
                }
                append_jsonl(output, record)
                previous[key] = record
                written += 1
                errors += 1
            print(f"[{position}/{len(indices)}] {item.key} ITEM_ERROR {type(exc).__name__}: {exc}", flush=True)
        finally:
            if holder is not None:
                holder.cleanup()

    print(f"[DONE] wrote={written} grounded={grounded} errors={errors} output={output}", flush=True)


if __name__ == "__main__":
    main()

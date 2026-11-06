#!/usr/bin/env python3
"""Build official-evidence-frozen videos for the leak-filtered NExT-GQA set."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (
    append_jsonl,
    build_freeze_video,
    latest_jsonl_records,
    load_worklist,
    merge_intervals,
    probe_video,
    require_binaries,
    selected_indices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED = ROOT / "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json"
DEFAULT_VIDEOS = ROOT / "source_datasets/next_gqa/videos"
DEFAULT_MAPPING = ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"
DEFAULT_OUTPUT = ROOT / "source_datasets/next_gqa/freeze_videos_no_evidence_leak"
DEFAULT_AUDIT = ROOT / "nextgqa_pipeline/insufficient/no_leak_freeze_build.jsonl"
BUILD_VERSION = "nextgqa-no-leak-official-freeze-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--official-padding", type=float, default=1.0)
    parser.add_argument("--sampling-fps", type=float, default=1.0)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def output_path(args: argparse.Namespace, item) -> Path:
    return args.output_dir / f"{item.video_id}_q{item.qid}_freeze.mp4"


def build_one(args: argparse.Namespace, item) -> dict:
    started = time.time()
    output = output_path(args, item)
    base = {
        "key": item.key,
        "item_index": item.index,
        "video_id": item.video_id,
        "qid": item.qid,
        "source_video": str(item.video_path),
        "output_video": str(output),
        "official_evidence": [list(span) for span in item.evidence],
        "official_padding": args.official_padding,
        "sampling_fps": args.sampling_fps,
        "blur_sigma": args.blur_sigma,
        "build_version": BUILD_VERSION,
    }
    if output.is_file() and not args.overwrite:
        return {**base, "status": "ok", "resumed_existing": True, "elapsed_seconds": 0.0}
    try:
        # NExT-GQA's annotated duration is commonly integer-rounded while
        # ffprobe reports an extra ~0.5--0.8 seconds.  The encoded stimulus
        # must preserve the actual source duration; annotation duration is
        # only used for QA metadata and must not make an otherwise valid
        # source fail construction.
        source_probe = probe_video(item.video_path, args.ffprobe)
        source_duration = float(source_probe["duration"])
        removed = merge_intervals(
            item.evidence,
            duration=source_duration,
            padding=args.official_padding,
        )
        result = build_freeze_video(
            item.video_path,
            output,
            removed,
            expected_duration=source_duration,
            sampling_fps=args.sampling_fps,
            blur_sigma=args.blur_sigma,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            threads=args.threads,
        )
        return {
            **base,
            "status": "ok",
            "resumed_existing": False,
            "annotated_duration": item.duration,
            "source_duration": source_duration,
            "masked_intervals": [list(span) for span in removed],
            "encoding": result,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        output.unlink(missing_ok=True)
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }


def main() -> None:
    args = parse_args()
    if args.official_padding < 0 or args.sampling_fps <= 0 or args.blur_sigma < 0:
        raise SystemExit("padding/blur must be non-negative and sampling FPS positive")
    if args.jobs < 1 or args.threads < 1:
        raise SystemExit("--jobs and --threads must be positive")
    require_binaries(args.ffmpeg, args.ffprobe)
    items = load_worklist(args.filtered_json, args.video_dir, args.video_id_mapping)
    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(items), shard, args.num_shards, args.limit)
    audit = args.audit
    if shard is not None:
        audit = audit.with_name(f"{audit.stem}.{shard}{audit.suffix}")
    previous = latest_jsonl_records([audit]) if audit.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for index in indices:
        item = items[index]
        old = previous.get(item.key)
        if (
            not args.overwrite
            and old
            and old.get("status") == "ok"
            and output_path(args, item).is_file()
        ):
            continue
        pending.append(item)
    print(
        f"[START] total_items={len(items)} selected={len(indices)} pending={len(pending)} "
        f"shard={shard}/{args.num_shards} jobs={args.jobs} threads={args.threads} "
        f"padding={args.official_padding}s blur={args.blur_sigma} output={args.output_dir}",
        flush=True,
    )

    completed = errors = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_one, args, item): item for item in pending}
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            append_jsonl(audit, record)
            if record["status"] == "ok":
                completed += 1
            else:
                errors += 1
            print(
                f"[{position}/{len(pending)}] {record['key']} {record['status']} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    print(f"[DONE] completed={completed} errors={errors} audit={audit}", flush=True)
    if errors:
        raise SystemExit(f"{errors} freeze videos failed; rerun the same shard to retry")


if __name__ == "__main__":
    main()

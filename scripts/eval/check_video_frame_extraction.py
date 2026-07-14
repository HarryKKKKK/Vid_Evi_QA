#!/usr/bin/env python3
"""Scan NExT-GQA videos for ffmpeg frame-extraction failures.

This exists because ffprobe-level checks (check_nextgqa_videos.py --ffprobe)
only read container-level metadata (duration, stream presence) and do NOT
catch videos where a specific timestamp near the end is undecodable -- which
is exactly the failure mode observed in nextgqa_result/*.skipped.json:
ffmpeg's frame extraction (the same command evaluate.py's extract_frames()
runs) fails with a non-zero exit status, always at the LAST uniformly-
sampled frame (~98% of duration), for a small number of specific videos.

By default this reproduces exactly what evaluate.py would attempt under
`--sampling uniform --num-frames 32` (the defaults actually used in
eval.sh): for each (video_id, qid) entry it probes the real video duration
via ffprobe, computes the same uniform timestamps evaluate.py's
plan_timestamps() would, and tries to extract the last --check-last-n of
them with the exact same ffmpeg command line. Pass --full-scan instead to
sample a spread of fractions across the whole duration (10%/25%/.../99%),
which is slower but tells you whether a failing video is only bad near the
end or broadly corrupt.

Two modes:
  --mode insufficient (default)  scans {video_id}_q{qid}_freeze.mp4 under
                                  --insufficient-video-dir (build_freeze.py's
                                  output naming; one file per qid, not
                                  deduplicated by video_id).
  --mode sufficient              scans the raw videos under --video-dir,
                                  resolved through --video-id-mapping (same
                                  {folder}/{vidorID}.mp4 layout evaluate.py
                                  uses for source_dataset == "nextgqa").

Output: --output-json is a full per-entry report (status one of
"ok" / "unparseable" / "missing_file" / "duration_probe_error"); a second,
smaller file --retry-json contains just the {"video_id", "qid"} pairs with
status == "unparseable" -- the ones that will keep failing on a plain
evaluate.py re-run, since evaluate.py's resume logic only remembers
successes (see run_evaluation()/load_existing_keys()) and will otherwise
retry every currently-skipped entry, including these, forever.

Usage:
  python scripts/eval/check_video_frame_extraction.py \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --mode insufficient \
    --insufficient-video-dir source_datasets/next_gqa/freeze_videos \
    --workers 8 \
    --output-json nextgqa_pipeline/nextgqa_insufficient_frame_check.json \
    --output-txt nextgqa_pipeline/nextgqa_insufficient_frame_check.txt \
    --retry-json nextgqa_pipeline/nextgqa_insufficient_unparseable.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filtered.json"
DEFAULT_INSUFFICIENT_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "freeze_videos"
DEFAULT_VIDEO_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "videos"
DEFAULT_MAPPING_JSON = (
    REPO_ROOT / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa" / "map_vid_vidorID.json"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_frame_check.json"
DEFAULT_OUTPUT_TXT = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_frame_check.txt"
DEFAULT_RETRY_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_frame_check_unparseable.json"

# Must match scripts/eval/evaluate.py's NUM_FRAMES / FRAME_WIDTH defaults for
# the quick (non --full-scan) check to reproduce the same failure.
DEFAULT_NUM_FRAMES = 32
DEFAULT_FRAME_WIDTH = 560
FULL_SCAN_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def out_name_insufficient(video_id: str, qid: str) -> str:
    return f"{video_id}_q{qid}_freeze.mp4"


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"[FATAL] filtered json not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"[FATAL] {path} is not a JSON array")
    return data


def load_video_id_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_video_path(entry: dict[str, Any], args: argparse.Namespace, mapping: dict[str, str]) -> Path | None:
    video_id, qid = str(entry["video_id"]), str(entry["qid"])
    if args.mode == "insufficient":
        return args.insufficient_video_dir / out_name_insufficient(video_id, qid)
    mapped = mapping.get(video_id)
    if mapped is None:
        return None
    return args.video_dir / f"{mapped}.mp4"


def probe_duration(video_path: Path, ffprobe: str, timeout: float) -> tuple[float | None, str]:
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "ffprobe timed out"
    if proc.returncode != 0:
        return None, proc.stderr[-500:]
    try:
        return float(proc.stdout.strip()), ""
    except ValueError:
        return None, f"could not parse ffprobe output: {proc.stdout!r}"


def uniform_timestamps(duration: float, num_frames: int) -> list[float]:
    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def try_extract_frame(ffmpeg: str, video_path: Path, ts: float, frame_width: int, timeout: float) -> tuple[bool, int, str]:
    """Same command shape as evaluate.py's extract_frames()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "frame.jpg")
        cmd = [
            ffmpeg, "-nostdin", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={frame_width}:-2",
            "-q:v", "3",
            out_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, -1, "ffmpeg timed out"
        ok = proc.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        return ok, proc.returncode, proc.stderr[-500:]


def check_entry(entry: dict[str, Any], args: argparse.Namespace, mapping: dict[str, str]) -> dict[str, Any]:
    video_id, qid = str(entry["video_id"]), str(entry["qid"])
    video_path = resolve_video_path(entry, args, mapping)
    rec: dict[str, Any] = {"video_id": video_id, "qid": qid}

    if video_path is None:
        rec["status"] = "missing_video_id_mapping"
        return rec
    rec["path"] = str(video_path)
    if not video_path.is_file():
        rec["status"] = "missing_file"
        return rec

    duration, derr = probe_duration(video_path, args.ffprobe, args.timeout)
    if duration is None or duration <= 0:
        rec["status"] = "duration_probe_error"
        rec["error"] = derr
        return rec
    rec["duration"] = round(duration, 3)

    if args.full_scan:
        checks = [(f"{frac:.0%}", frac * duration) for frac in FULL_SCAN_FRACTIONS]
    else:
        tail = uniform_timestamps(duration, args.num_frames)[-args.check_last_n:]
        checks = [(f"uniform_last_{i}", ts) for i, ts in enumerate(tail)]

    failures = []
    for label, ts in checks:
        ok, rc, err_tail = try_extract_frame(args.ffmpeg, video_path, ts, args.frame_width, args.timeout)
        if not ok:
            failures.append({
                "label": label,
                "timestamp": round(ts, 3),
                "fraction_of_duration": round(ts / duration, 4),
                "returncode": rc,
                "stderr_tail": err_tail,
            })

    rec["status"] = "unparseable" if failures else "ok"
    if failures:
        rec["failures"] = failures
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED_JSON)
    parser.add_argument("--mode", choices=("insufficient", "sufficient"), default="insufficient")
    parser.add_argument("--insufficient-video-dir", type=Path, default=DEFAULT_INSUFFICIENT_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                         help="Must match evaluate.py's --num-frames for the quick check to be meaningful.")
    parser.add_argument("--check-last-n", type=int, default=1,
                         help="Quick mode: how many of the tail-end uniform timestamps to test per video.")
    parser.add_argument("--full-scan", action="store_true",
                         help="Instead of the quick tail-only check, sample fixed fractions "
                              f"({FULL_SCAN_FRACTIONS}) across the whole duration, to tell "
                              "apart 'only bad near the end' from 'broadly corrupt'.")
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-ffmpeg/ffprobe-call timeout, seconds.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--retry-json", type=Path, default=DEFAULT_RETRY_JSON,
                         help="{video_id, qid} pairs with status == 'unparseable', for downstream use.")
    args = parser.parse_args()

    entries = load_entries(args.filtered_json)
    if args.limit is not None:
        entries = entries[: args.limit]

    mapping = load_video_id_mapping(args.video_id_mapping) if args.mode == "sufficient" else {}

    log(f"[START] mode={args.mode} entries={len(entries)} workers={args.workers} "
        f"full_scan={args.full_scan} check_last_n={args.check_last_n}")

    results: list[dict[str, Any]] = []
    done = 0
    total = len(entries)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_entry, e, args, mapping): e for e in entries}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            if rec["status"] != "ok":
                log(f"[{done}/{total}] {rec['status']:22s} video_id={rec['video_id']} qid={rec['qid']}")

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    unparseable = [{"video_id": r["video_id"], "qid": r["qid"]} for r in results if r["status"] == "unparseable"]
    args.retry_json.parent.mkdir(parents=True, exist_ok=True)
    with args.retry_json.open("w", encoding="utf-8") as f:
        json.dump(unparseable, f, ensure_ascii=False, indent=2)

    unique_bad_videos = sorted({r["video_id"] for r in results if r["status"] == "unparseable"})

    lines = [
        "NExT-GQA Frame Extraction Check",
        "=" * 60,
        f"mode: {args.mode}",
        f"entries checked: {len(results)}",
        f"full_scan: {args.full_scan}",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")
    lines += [
        "",
        f"unique video_ids with >=1 unparseable timestamp: {len(unique_bad_videos)}",
    ]
    lines += [f"  {vid}" for vid in unique_bad_videos]
    text = "\n".join(lines)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("\n" + text)
    print(f"\nFull report      -> {args.output_json}")
    print(f"Unparseable list -> {args.retry_json} ({len(unparseable)} (video_id, qid) pairs)")


if __name__ == "__main__":
    main()

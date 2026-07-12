#!/usr/bin/env python3
"""
Check whether the timestamps evaluate.py's plan_timestamps() would sample
under `--sampling uniform` actually land inside each anchor's
evidence_intervals. Anchors where NONE of the uniformly-sampled timestamps
touch any evidence interval get their (video_id, qid) written to a txt
report -- these anchors should be excluded from uniform-sampling analysis,
since the model literally never sees the evidence frames.

Uses the "video_duration" field already present in cgbench_filtered.json
entries, so it does not need the actual video files or ffprobe.

Uniform-sampling formula copied verbatim from scripts/evaluate.py's
plan_timestamps() uniform branch, to stay in sync:
    timestamps = [(i + 0.5) * duration / num_frames for i in range(num_frames)]
"""

import argparse
import json
from pathlib import Path

DEFAULT_JSON = "cgbench_pipeline/cgbench_filtered.json"
DEFAULT_OUTPUT = "cgbench_result/uniform_coverage_report.txt"
DEFAULT_NUM_FRAMES = 128


def uniform_timestamps(duration, num_frames):
    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def evidence_hit(timestamps, intervals):
    for t in timestamps:
        for iv in intervals:
            s, e = float(iv["start"]), float(iv["end"])
            if s > e:
                s, e = e, s
            if s <= t <= e:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=DEFAULT_JSON,
                     help="cgbench_filtered.json (or any array with the same schema).")
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                     help="Must match the --num-frames used at eval time.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                     help="Where to write the video_id/qid miss list (tab-separated txt).")
    args = ap.parse_args()

    entries = json.loads(Path(args.json).read_text(encoding="utf-8"))

    misses = []
    skipped_no_evidence = 0
    skipped_no_duration = 0

    for e in entries:
        intervals = e.get("evidence_intervals") or []
        duration = e.get("video_duration")

        if not intervals:
            skipped_no_evidence += 1
            continue
        if not duration or duration <= 0:
            skipped_no_duration += 1
            continue

        timestamps = uniform_timestamps(duration, args.num_frames)
        if not evidence_hit(timestamps, intervals):
            misses.append((e["video_id"], e["qid"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for video_id, qid in misses:
            f.write(f"{video_id}\t{qid}\n")

    total = len(entries)
    checked = total - skipped_no_evidence - skipped_no_duration

    print("=" * 60)
    print(f"num_frames (uniform)   : {args.num_frames}")
    print(f"Total entries          : {total}")
    print(f"Skipped (no evidence)  : {skipped_no_evidence}")
    print(f"Skipped (no duration)  : {skipped_no_duration}")
    print(f"Checked                : {checked}")
    if checked:
        print(f"Coverage miss          : {len(misses)} ({len(misses) / checked:.2%} of checked)")
    else:
        print("Coverage miss          : 0")
    print(f"Written to             : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Freeze a video-diverse, evidence-stratified NExT-GQA sufficient sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=750)
    p.add_argument("--shards", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260915)
    return p.parse_args()


def load(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("source must contain a JSON list or JSONL records")
    return value


def merged_duration(intervals: list[dict]) -> tuple[float, float]:
    spans = sorted((min(float(x["start"]), float(x["end"])),
                    max(float(x["start"]), float(x["end"]))) for x in intervals)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    duration = sum(end - start for start, end in merged)
    if not merged or duration <= 0:
        return 0.0, 0.0
    midpoint = sum(((start + end) / 2) * (end - start) for start, end in merged) / duration
    return duration, midpoint


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stable_score(seed: int, *parts: object) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def largest_remainder(groups: dict[tuple, list[dict]], n: int) -> dict[tuple, int]:
    total = sum(map(len, groups.values()))
    raw = {key: n * len(rows) / total for key, rows in groups.items()}
    quota = {key: min(len(groups[key]), math.floor(value)) for key, value in raw.items()}
    left = n - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(raw[key] - math.floor(raw[key])), str(key)))
    while left:
        progressed = False
        for key in order:
            if quota[key] < len(groups[key]) and left:
                quota[key] += 1
                left -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("cannot satisfy sample quota")
    return quota


def main() -> None:
    args = parse_args()
    rows = [r for r in load(args.source)
            if r.get("source_dataset") == "nextgqa"
            and r.get("evidence_condition") == "sufficient"
            and len(r.get("valid_option_ids") or []) == 1
            and len(r.get("sampled_seconds") or []) == 32]
    seen = set()
    candidates = []
    for row in rows:
        key = (str(row["video_id"]), str(row["qid"]))
        if key in seen:
            continue
        seen.add(key)
        duration = float(row.get("video_duration") or 0)
        evidence_duration, midpoint = merged_duration(row.get("evidence_intervals") or [])
        if duration <= 0 or evidence_duration <= 0:
            continue
        times = [float(x) for x in row["sampled_seconds"]]
        hits = sum(any(float(span["start"]) <= t <= float(span["end"])
                       for span in row["evidence_intervals"]) for t in times)
        if not 0 < hits <= 16:  # guarantees an equal-size non-evidence control exists
            continue
        record = dict(row)
        record["_duration_ratio"] = evidence_duration / duration
        record["_position_ratio"] = midpoint / duration
        record["_hit_count"] = hits
        candidates.append(record)
    if len({str(r["video_id"]) for r in candidates}) < args.size:
        raise RuntimeError("not enough eligible source videos for one-QA-per-video sampling")

    q1 = quantile([r["_duration_ratio"] for r in candidates], 1 / 3)
    q2 = quantile([r["_duration_ratio"] for r in candidates], 2 / 3)
    for row in candidates:
        ratio = row["_duration_ratio"]
        row["evidence_duration_bin"] = "short" if ratio <= q1 else "medium" if ratio <= q2 else "long"
        pos = row["_position_ratio"]
        row["evidence_position_bin"] = "early" if pos < 1 / 3 else "middle" if pos < 2 / 3 else "late"
        hits = row["_hit_count"]
        row["evidence_frame_hit_bin"] = "1" if hits == 1 else "2-4" if hits <= 4 else "5-8" if hits <= 8 else "9-16"

    # Select one QA per source video before allocating proportional joint-stratum quotas.
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[str(row["video_id"])].append(row)
    video_representatives = [min(group, key=lambda r: stable_score(args.seed, r["video_id"], r["qid"]))
                             for group in by_video.values()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in video_representatives:
        stratum = (str(row.get("question_type") or "UNKNOWN"), row["evidence_duration_bin"],
                   row["evidence_position_bin"], row["evidence_frame_hit_bin"])
        groups[stratum].append(row)
    for key, group in groups.items():
        group.sort(key=lambda r: stable_score(args.seed, key, r["video_id"], r["qid"]))
    quota = largest_remainder(groups, args.size)
    selected = [row for key in sorted(groups) for row in groups[key][:quota[key]]]
    selected.sort(key=lambda r: stable_score(args.seed, "selected", r["video_id"], r["qid"]))

    output = []
    for row in selected:
        output.append({
            "experiment_group": "representative_sufficient_source",
            "video_kind": "source", "video_id": str(row["video_id"]), "qid": str(row["qid"]),
            "question_type": row.get("question_type"), "question": row["question"],
            "choices": row["choices"], "answer": row["valid_option_ids"][0],
            "sampled_seconds": row["sampled_seconds"],
            "evidence_duration_bin": row["evidence_duration_bin"],
            "evidence_position_bin": row["evidence_position_bin"],
            "evidence_frame_hit_bin": row["evidence_frame_hit_bin"],
            "evidence_frame_hits": row["_hit_count"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_n750.json"
    sample_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rng = random.Random(args.seed)
    rng.shuffle(output)
    shard_rows = [[] for _ in range(args.shards)]
    for index, row in enumerate(output):
        shard_rows[index % args.shards].append(row)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    for index, shard in enumerate(shard_rows):
        (shard_dir / f"{index:03d}.json").write_text(
            json.dumps(shard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    report = {
        "source": str(args.source), "seed": args.seed, "eligible_qas": len(candidates),
        "eligible_videos": len(by_video), "selected": len(output), "unique_videos": len({r["video_id"] for r in output}),
        "shard_sizes": [len(x) for x in shard_rows], "sample_sha256": sha,
        "duration_ratio_tertiles": [q1, q2],
        "question_type": Counter(r["question_type"] for r in output),
        "evidence_duration_bin": Counter(r["evidence_duration_bin"] for r in output),
        "evidence_position_bin": Counter(r["evidence_position_bin"] for r in output),
        "evidence_frame_hit_bin": Counter(r["evidence_frame_hit_bin"] for r in output),
    }
    report = {k: dict(v) if isinstance(v, Counter) else v for k, v in report.items()}
    (args.output_dir / "sample_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

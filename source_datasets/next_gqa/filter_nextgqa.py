"""
NExT-GQA test set filtering script.

Data sources:
  - datasets/nextgqa/test.csv        : QA pairs (video_id, qid, question, answer, type, ...)
  - datasets/nextgqa/gsub_test.json  : per-video duration + location[qid] = [[start, end], ...]

Filtering conditions (test set only):
  1. question type  : keep T* (TN/TC/TP=Temporal) and C* (CW/CH=Causal); exclude D*
  2. segment count  : >= 2 independent temporal segments per QA
  3. segment duration: 3s <= each segment duration <= 30s
  4. total coverage : sum(seg durations) / video_duration <= 60%
  5. segment gap    : for adjacent segments sorted by start, gap = next.start - prev.end >= 3s
  6. video duration : read from gsub_test.json "duration" field; failures are warned and skipped
"""

import csv
import json
import os
import sys
import warnings
from collections import Counter

BASE_DIR = os.path.join(os.path.dirname(__file__), "NExT-GQA", "datasets", "nextgqa")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
GSUB_JSON = os.path.join(BASE_DIR, "gsub_test.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "nextgqa_filtered_test.json")

CAUSAL_TYPES = {"CW", "CH"}
TEMPORAL_TYPES = {"TN", "TC", "TP"}
KEEP_TYPES = CAUSAL_TYPES | TEMPORAL_TYPES

MIN_SEG_DUR = 3.0
MAX_SEG_DUR = 30.0
MIN_SEG_COUNT = 2
MAX_COVERAGE_RATIO = 0.60
MIN_GAP = 3.0


def load_data():
    with open(TEST_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(GSUB_JSON, encoding="utf-8") as f:
        gsub = json.load(f)
    return rows, gsub


def get_question_type(raw_type):
    if raw_type in TEMPORAL_TYPES:
        return "Temporal"
    if raw_type in CAUSAL_TYPES:
        return "Causal"
    return None


def check_segments(segs, video_duration):
    """
    Returns (passed: bool, fail_reasons: set[str]).
    fail_reasons is a subset of:
        "seg_count", "seg_duration", "coverage", "gap"
    """
    fails = set()

    # Condition 2: segment count
    if len(segs) < MIN_SEG_COUNT:
        fails.add("seg_count")
        # No point checking further conditions that depend on >=2 segs
        return False, fails

    # Sort by start time
    segs_sorted = sorted(segs, key=lambda s: s[0])

    # Condition 3: individual segment duration
    for s, e in segs_sorted:
        dur = e - s
        if dur < MIN_SEG_DUR or dur > MAX_SEG_DUR:
            fails.add("seg_duration")
            break

    # Condition 4: total coverage ratio
    total_seg_dur = sum(e - s for s, e in segs_sorted)
    if total_seg_dur / video_duration > MAX_COVERAGE_RATIO:
        fails.add("coverage")

    # Condition 5: adjacent segment gap
    for i in range(len(segs_sorted) - 1):
        gap = segs_sorted[i + 1][0] - segs_sorted[i][1]
        if gap < MIN_GAP:
            fails.add("gap")
            break

    passed = len(fails) == 0
    return passed, fails


def main():
    rows, gsub = load_data()

    total_test = len(rows)

    # Per-condition independent failure counters
    fail_type = 0
    fail_seg_count = 0
    fail_seg_duration = 0
    fail_coverage = 0
    fail_gap = 0
    fail_duration_read = 0

    results = []
    evidence_count_dist = Counter()
    causal_count = 0
    temporal_count = 0

    for row in rows:
        video_id = row["video_id"]
        qid = row["qid"]
        raw_type = row["type"]
        question = row["question"]
        answer = row["answer"]

        # --- Condition 1: question type ---
        question_type = get_question_type(raw_type)
        if question_type is None:
            fail_type += 1
            continue

        # --- Get video duration from gsub ---
        if video_id not in gsub:
            warnings.warn(f"video_id {video_id} not in gsub_test.json; skipping")
            fail_duration_read += 1
            continue
        vid_info = gsub[video_id]
        try:
            video_duration = float(vid_info["duration"])
        except (KeyError, ValueError, TypeError) as exc:
            warnings.warn(f"Cannot read duration for video {video_id}: {exc}; skipping")
            fail_duration_read += 1
            continue

        if qid not in vid_info["location"]:
            warnings.warn(f"qid {qid} not in gsub location for video {video_id}; skipping")
            fail_duration_read += 1
            continue

        raw_segs = vid_info["location"][qid]  # list of [start, end]

        # --- Independent failure counting (each condition checked on ALL non-type-filtered rows) ---
        if len(raw_segs) < MIN_SEG_COUNT:
            fail_seg_count += 1

        segs_sorted = sorted(raw_segs, key=lambda s: s[0])
        has_bad_dur = any((e - s) < MIN_SEG_DUR or (e - s) > MAX_SEG_DUR for s, e in segs_sorted)
        if has_bad_dur:
            fail_seg_duration += 1

        total_seg_dur = sum(e - s for s, e in segs_sorted)
        if total_seg_dur / video_duration > MAX_COVERAGE_RATIO:
            fail_coverage += 1

        has_small_gap = any(
            segs_sorted[i + 1][0] - segs_sorted[i][1] < MIN_GAP
            for i in range(len(segs_sorted) - 1)
        )
        if has_small_gap:
            fail_gap += 1

        # --- Combined filter ---
        passed, _ = check_segments(raw_segs, video_duration)
        if not passed:
            continue

        # Build output record
        evidence_intervals = [
            {"start": float(s), "end": float(e), "description": ""}
            for s, e in segs_sorted
        ]
        evidence_count = len(evidence_intervals)

        record = {
            "video_id": f"nextgqa_{video_id}",
            "source_dataset": "nextgqa",
            "question": question,
            "answer": answer,
            "evidence_intervals": evidence_intervals,
            "evidence_count": evidence_count,
            "video_duration": video_duration,
            "question_type": question_type,
            "source_task": raw_type,
            "evidence_condition": "sufficient",
            "expected_behavior": "answer",
        }
        results.append(record)

        evidence_count_dist[evidence_count] += 1
        if question_type == "Causal":
            causal_count += 1
        else:
            temporal_count += 1

    # Write output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Statistics ---
    passed_total = len(results)
    seg4plus = sum(v for k, v in evidence_count_dist.items() if k >= 4)

    print(f"总样本数（test set，筛选前）: {total_test}")
    print(f"通过筛选的样本数: {passed_total}")
    print(f"  - 其中 Causal: {causal_count}")
    print(f"  - 其中 Temporal: {temporal_count}")
    print(f"evidence_count 分布:")
    print(f"  - 2个 segment: {evidence_count_dist.get(2, 0)}")
    print(f"  - 3个 segment: {evidence_count_dist.get(3, 0)}")
    print(f"  - 4个及以上: {seg4plus}")
    print(f"被各条件过滤掉的样本数（各条件独立统计，不去重）:")
    print(f"  - question type 不符: {fail_type}")
    print(f"  - segment 数量 < 2: {fail_seg_count}")
    print(f"  - 存在时长不符的 segment（单个 segment < 3s 或 > 30s）: {fail_seg_duration}")
    print(f"  - segment 总时长占比超 60%: {fail_coverage}")
    print(f"  - 存在间距 < 3s 的相邻 segment: {fail_gap}")
    print(f"  - 视频时长读取失败（跳过）: {fail_duration_read}")
    print(f"\n输出文件: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

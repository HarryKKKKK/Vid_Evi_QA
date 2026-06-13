import json
import os
import re
from collections import Counter
from pathlib import Path


ANNOTATION_PATH = "ETBench/annotations/etbench_txt_v1.0.json"
OUTPUT_PATH = "filtered_etbench.json"
TARGET_TASKS = {"tvg", "tal"}

MIN_INTERVAL_DURATION = 2.0
MAX_INTERVAL_DURATION = 50.0
MIN_ADJACENT_GAP = 3.0
MIN_VIDEO_DURATION = 30.0
MIN_INTERVAL_COUNT = 2


def normalize_intervals(tgt):
    """
    Convert ETBench tgt into standard format:
        [[start, end], [start, end], ...]

    Handles:
        - [start, end]
        - [[start, end], [start, end]]
        - empty / invalid values
    """
    if tgt is None:
        return []

    if not isinstance(tgt, list):
        return []

    if len(tgt) == 0:
        return []

    # Case 1: tgt = [start, end]
    if len(tgt) == 2 and all(isinstance(x, (int, float)) for x in tgt):
        start, end = float(tgt[0]), float(tgt[1])
        if end > start:
            return [[start, end]]
        return []

    # Case 2: tgt = [[start, end], [start, end]]
    intervals = []
    for x in tgt:
        if (
            isinstance(x, list)
            and len(x) >= 2
            and isinstance(x[0], (int, float))
            and isinstance(x[1], (int, float))
        ):
            start, end = float(x[0]), float(x[1])
            if end > start:
                intervals.append([start, end])

    return intervals


def explore_data(data):
    tasks = set(item["task"] for item in data)
    print("所有 task 值:", tasks)

    sources = set(item.get("source", "N/A") for item in data)
    print("所有 source 值:", sources)

    print("task 分布:", Counter(item["task"] for item in data))

    for item in data:
        if item["task"] == "tvg":
            print("\nTVG 样本示例:", json.dumps(item, ensure_ascii=False, indent=2))
            break

    for item in data:
        if item["task"] == "tal":
            print("\nTAL 样本示例:", json.dumps(item, ensure_ascii=False, indent=2))
            break


def video_id_from_path(video_path: str) -> str:
    basename = os.path.basename(video_path)
    stem = Path(basename).stem
    return f"etbench_{stem}"


def check_intervals(tgt, task: str = "tal"):
    """
    Check whether intervals satisfy filtering rules.

    Returns:
        passes: bool
        reasons_failed: set[str]
    """
    intervals = sorted(normalize_intervals(tgt), key=lambda x: x[0])
    failed = set()

    min_count = 1 if task == "tvg" else MIN_INTERVAL_COUNT
    if len(intervals) < min_count:
        failed.add("interval_count")

    for start, end in intervals:
        dur = end - start
        if dur < MIN_INTERVAL_DURATION or dur > MAX_INTERVAL_DURATION:
            failed.add("interval_duration")
            break

    for i in range(len(intervals) - 1):
        gap = intervals[i + 1][0] - intervals[i][1]
        if gap < MIN_ADJACENT_GAP:
            failed.add("adjacent_gap")
            break

    return len(failed) == 0, failed


_TVG_RE = re.compile(r"described by the sentence:\s*'(.+?)'")
_TAL_RE = re.compile(r"action category:\s*'(.+?)'")


def extract_description(task: str, q: str) -> str:
    if task == "tvg":
        m = _TVG_RE.search(q)
    else:
        m = _TAL_RE.search(q)
    return m.group(1) if m else ""


def format_answer(intervals):
    parts = [f"{s:.1f} - {e:.1f}" for s, e in intervals]
    if len(parts) == 1:
        return f"{parts[0]} seconds"
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]} seconds"
    return ", ".join(parts[:-1]) + f", and {parts[-1]} seconds"


def build_output_record(item):
    tgt = sorted(normalize_intervals(item["tgt"]), key=lambda x: x[0])
    desc = extract_description(item["task"], item["q"])

    return {
        "video_id": video_id_from_path(item["video"]),
        "source_dataset": "etbench",
        "question": item["q"],
        "answer": format_answer(tgt),
        "evidence_intervals": [
            {"start": float(s), "end": float(e), "description": desc}
            for s, e in tgt
        ],
        "evidence_count": len(tgt),
        "video_duration": float(item["duration"]),
        "question_type": "Temporal",
        "source_task": item["task"],
        "evidence_condition": "sufficient",
        "expected_behavior": "answer",
    }


def main():
    with open(ANNOTATION_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("数据结构探查")
    print("=" * 60)
    explore_data(data)
    print("=" * 60)

    target_items = [item for item in data if item["task"] in TARGET_TASKS]

    tvg_total = sum(1 for item in target_items if item["task"] == "tvg")
    tal_total = sum(1 for item in target_items if item["task"] == "tal")

    # Per-condition failure counters, independent and not deduplicated
    fail_interval_count = 0
    fail_interval_duration = 0
    fail_adjacent_gap = 0
    fail_video_duration = 0

    passed = []
    passed_tvg = 0
    passed_tal = 0

    # Optional debug counters
    raw_single_interval_count = 0
    raw_multi_interval_count = 0
    raw_invalid_interval_count = 0

    for item in target_items:
        raw_tgt = item.get("tgt", [])
        tgt = normalize_intervals(raw_tgt)
        duration = float(item.get("duration", 0) or 0)

        # Debug statistics for tgt format
        if isinstance(raw_tgt, list) and len(raw_tgt) == 2 and all(isinstance(x, (int, float)) for x in raw_tgt):
            raw_single_interval_count += 1
        elif isinstance(raw_tgt, list) and len(raw_tgt) > 0 and all(isinstance(x, list) for x in raw_tgt):
            raw_multi_interval_count += 1
        else:
            raw_invalid_interval_count += 1

        item_failed = set()

        min_count = 1 if item["task"] == "tvg" else MIN_INTERVAL_COUNT
        if len(tgt) < min_count:
            item_failed.add("interval_count")

        for s, e in tgt:
            interval_duration = e - s
            if interval_duration < MIN_INTERVAL_DURATION or interval_duration > MAX_INTERVAL_DURATION:
                item_failed.add("interval_duration")
                break

        intervals_sorted = sorted(tgt, key=lambda x: x[0])
        for i in range(len(intervals_sorted) - 1):
            gap = intervals_sorted[i + 1][0] - intervals_sorted[i][1]
            if gap < MIN_ADJACENT_GAP:
                item_failed.add("adjacent_gap")
                break

        if duration < MIN_VIDEO_DURATION:
            item_failed.add("video_duration")

        if "interval_count" in item_failed:
            fail_interval_count += 1
        if "interval_duration" in item_failed:
            fail_interval_duration += 1
        if "adjacent_gap" in item_failed:
            fail_adjacent_gap += 1
        if "video_duration" in item_failed:
            fail_video_duration += 1

        if not item_failed:
            record = build_output_record(item)
            passed.append(record)

            if item["task"] == "tvg":
                passed_tvg += 1
            elif item["task"] == "tal":
                passed_tal += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    evidence_counter = Counter(r["evidence_count"] for r in passed)

    print("\n筛选统计")
    print("=" * 60)
    print(f"总样本数（筛选前，仅 tvg + tal）: {len(target_items)}")
    print(f"  - 其中 tvg: {tvg_total}")
    print(f"  - 其中 tal: {tal_total}")

    print("\ntgt 原始格式统计:")
    print(f"  - 原始格式为 [start, end] 的样本数: {raw_single_interval_count}")
    print(f"  - 原始格式为 [[start, end], ...] 的样本数: {raw_multi_interval_count}")
    print(f"  - 其他 / 无效格式样本数: {raw_invalid_interval_count}")

    print(f"\n通过筛选的样本数: {len(passed)}")
    print(f"  - 其中 tvg: {passed_tvg}")
    print(f"  - 其中 tal: {passed_tal}")

    print("\nevidence_count 分布:")
    print(f"  - 2个区间: {evidence_counter.get(2, 0)}")
    print(f"  - 3个区间: {evidence_counter.get(3, 0)}")
    four_plus = sum(v for k, v in evidence_counter.items() if k >= 4)
    print(f"  - 4个及以上: {four_plus}")

    print("\n被各条件过滤掉的样本数（各条件独立统计，不去重）:")
    print(f"  - tgt 区间数量 < 2: {fail_interval_count}")
    print(f"  - 存在时长不符的区间（单个区间 < 2s 或 > 50s）: {fail_interval_duration}")
    print(f"  - 存在间距 < 3s 的相邻区间: {fail_adjacent_gap}")
    print(f"  - 视频时长 < 30s: {fail_video_duration}")

    print(f"\n输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
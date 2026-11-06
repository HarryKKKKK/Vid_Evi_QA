#!/usr/bin/env python3
"""Summarize retained NExT-GQA + ETBench videos and evidence intervals.

Language-prior exclusions are applied by NExT-GQA ``(video_id, qid)`` key and
therefore propagate to sufficient, insufficient, and every partial-alpha
variant. ETBench conflict records are unaffected by that exclusion list.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEXT_MANIFEST = ROOT / "results/build/nextgqa/nextgqa_filtered_no_evidence_leak.json"
DEFAULT_NEXT_CANDIDATES = ROOT / "results/build/nextgqa/partial_candidates_full_n5279.json"
DEFAULT_PARTIAL_META = ROOT / "results/build/nextgqa/partial_candidates_full_n5279.meta.json"
DEFAULT_CONFLICT_QA = ROOT / "results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged/conflict_qa_v2.jsonl"
DEFAULT_ET_INSTANCES = ROOT / "results/build/etbench/conflict/visual_instance_jobs.jsonl"
DEFAULT_EXCLUSIONS = ROOT / "results/reports/language_prior_screen/combined_5279/removal_candidates.json"
DEFAULT_OUTPUT = ROOT / "results/reports/retained_benchmark_statistics"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-manifest", type=Path, default=DEFAULT_NEXT_MANIFEST)
    parser.add_argument("--next-candidates", type=Path, default=DEFAULT_NEXT_CANDIDATES)
    parser.add_argument("--partial-meta", type=Path, default=DEFAULT_PARTIAL_META)
    parser.add_argument("--conflict-qa", type=Path, default=DEFAULT_CONFLICT_QA)
    parser.add_argument("--et-instances", type=Path, default=DEFAULT_ET_INSTANCES)
    exclusion = parser.add_mutually_exclusive_group()
    exclusion.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    exclusion.add_argument("--no-exclusions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def next_key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def distribution(values: Iterable[float]) -> dict:
    data = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not data:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(data),
        "sum": round(sum(data), 6),
        "mean": round(statistics.mean(data), 6),
        "median": round(statistics.median(data), 6),
        "min": round(data[0], 6),
        "p25": round(percentile(data, 0.25), 6),
        "p75": round(percentile(data, 0.75), 6),
        "p90": round(percentile(data, 0.90), 6),
        "p95": round(percentile(data, 0.95), 6),
        "p99": round(percentile(data, 0.99), 6),
        "max": round(data[-1], 6),
    }


def duration_bin(value: float) -> str:
    if value < 30:
        return "<30s"
    if value < 60:
        return "30-<60s"
    if value < 120:
        return "60-<120s"
    return ">=120s"


def interval_duration_bin(value: float) -> str:
    if value < 1:
        return "<1s"
    if value < 2:
        return "1-<2s"
    if value < 5:
        return "2-<5s"
    if value < 10:
        return "5-<10s"
    return ">=10s"


def normalize_interval(value: Any, video_duration: float | None = None) -> tuple[float, float] | None:
    if isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = value[0], value[1]
    else:
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    start = max(0.0, start)
    if video_duration is not None:
        start, end = min(start, video_duration), min(end, video_duration)
    if end <= start:
        return None
    return start, end


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_summary(intervals: list[tuple[float, float]]) -> dict:
    durations = [end - start for start, end in intervals]
    merged = merge_intervals(intervals)
    return {
        "raw_interval_count": len(intervals),
        "raw_interval_duration_seconds": round(sum(durations), 6),
        "merged_interval_count": len(merged),
        "union_duration_seconds": round(sum(end - start for start, end in merged), 6),
        "interval_duration_distribution": distribution(durations),
        "interval_duration_bins": dict(sorted(Counter(interval_duration_bin(value) for value in durations).items())),
    }


def collection_interval_summary(groups: Iterable[Iterable[tuple[float, float]]]) -> dict:
    """Summarize intervals while only unioning intervals within the same timeline."""
    materialized = [list(group) for group in groups]
    raw = [interval for group in materialized for interval in group]
    merged_groups = [merge_intervals(group) for group in materialized]
    durations = [end - start for start, end in raw]
    return {
        "group_count": len(materialized),
        "groups_with_evidence": sum(bool(group) for group in materialized),
        "raw_interval_count": len(raw),
        "raw_interval_duration_seconds": round(sum(durations), 6),
        "merged_interval_count": sum(len(group) for group in merged_groups),
        "union_duration_seconds": round(
            sum(end - start for group in merged_groups for start, end in group), 6
        ),
        "interval_duration_distribution": distribution(durations),
        "interval_duration_bins": dict(
            sorted(Counter(interval_duration_bin(value) for value in durations).items())
        ),
    }


def category_interval_stats(rows: list[dict]) -> dict:
    categories: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    return {
        category: {
            "qa_count": len(items),
            "raw_interval_count": sum(item["interval_count"] for item in items),
            "raw_interval_duration_seconds": round(
                sum(item["raw_interval_duration_seconds"] for item in items), 6
            ),
            "union_interval_duration_seconds": round(
                sum(item["union_interval_duration_seconds"] for item in items), 6
            ),
        }
        for category, items in sorted(categories.items())
    }


def load_exclusions(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    if not path.is_file():
        raise SystemExit(
            f"Exclusion file does not exist: {path}. Use --no-exclusions only for a pre-filter baseline."
        )
    rows = read_json(path)
    keys = set()
    for row in rows:
        if row.get("removal_candidate", True):
            keys.add((str(row["video_id"]), str(row["qid"])))
    return keys


def group_numeric(rows: list[dict], category: str, value: str) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(category, "UNKNOWN"))].append(float(row[value]))
    return {name: distribution(values) for name, values in sorted(grouped.items())}


def build_stats(a: argparse.Namespace) -> tuple[dict, dict]:
    manifest = read_json(a.next_manifest)
    candidates = read_json(a.next_candidates)
    partial_meta = read_json(a.partial_meta)
    conflict_qa = read_jsonl(a.conflict_qa)
    et_instances = read_jsonl(a.et_instances)
    exclusions = load_exclusions(None if a.no_exclusions else a.exclusions)

    candidate_keys = {next_key(row) for row in candidates}
    selected = [row for row in manifest if next_key(row) in candidate_keys]
    if len(selected) != len(candidate_keys):
        raise SystemExit(f"Matched {len(selected)} of {len(candidate_keys)} NExT-GQA candidate keys")
    retained_next = [row for row in selected if next_key(row) not in exclusions]
    removed_present = candidate_keys & exclusions
    unknown_exclusions = exclusions - candidate_keys
    alphas = [float(value) for value in partial_meta["alphas"]]

    next_video_rows: dict[str, dict] = {}
    next_qa_evidence_rows = []
    next_intervals_by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in retained_next:
        video_id = str(row["video_id"])
        duration = float(row["video_duration"])
        next_video_rows.setdefault(video_id, {
            "source_dataset": "nextgqa",
            "video_key": video_id,
            "video_duration_seconds": duration,
            "qa_count": 0,
        })["qa_count"] += 1
        intervals = [
            interval
            for value in row.get("evidence_intervals", [])
            if (interval := normalize_interval(value, duration)) is not None
        ]
        next_intervals_by_video[video_id].extend(intervals)
        merged = merge_intervals(intervals)
        next_qa_evidence_rows.append({
            "source_dataset": "nextgqa",
            "item_key": f"{video_id}_q{row['qid']}",
            "category": str(row.get("question_type", "UNKNOWN")),
            "video_duration_seconds": duration,
            "interval_count": len(intervals),
            "raw_interval_duration_seconds": sum(end - start for start, end in intervals),
            "union_interval_duration_seconds": sum(end - start for start, end in merged),
        })

    parent_meta: dict[str, dict] = {}
    for row in et_instances:
        parent = str(row["parent_job_id"])
        instance = row["instance"]
        current = parent_meta.setdefault(parent, {
            "source_dataset": "etbench",
            "video_key": parent,
            "video_path": row.get("video"),
            "video_duration_seconds": float(row["video_duration"]),
            "source_task": row.get("source_task"),
            "instances": {},
            "qa_count": 0,
        })
        interval = normalize_interval(instance.get("core"), current["video_duration_seconds"])
        if interval:
            current["instances"][str(instance["instance_id"])] = interval

    accepted_conflict = [row for row in conflict_qa if row.get("status") == "accepted"]
    conflict_parents = {str(row["parent_job_id"]) for row in accepted_conflict}
    missing_parents = conflict_parents - set(parent_meta)
    if missing_parents:
        raise SystemExit(f"Missing ETBench parent metadata: {sorted(missing_parents)[:5]}")
    et_video_rows = []
    for parent in sorted(conflict_parents):
        row = parent_meta[parent]
        row["qa_count"] = sum(str(qa["parent_job_id"]) == parent for qa in accepted_conflict)
        et_video_rows.append(row)

    et_qa_evidence_rows = []
    for qa in accepted_conflict:
        parent = str(qa["parent_job_id"])
        duration = parent_meta[parent]["video_duration_seconds"]
        unique = {}
        for option in qa.get("options", []):
            for support in option.get("supported_by", []):
                interval = normalize_interval(support.get("interval"), duration)
                if interval:
                    key = (str(support.get("instance_id")), round(interval[0], 6), round(interval[1], 6))
                    unique[key] = interval
        intervals = list(unique.values())
        merged = merge_intervals(intervals)
        et_qa_evidence_rows.append({
            "source_dataset": "etbench",
            "item_key": str(qa["id"]),
            "category": str(qa.get("conflict_type", "UNKNOWN")),
            "video_duration_seconds": duration,
            "interval_count": len(intervals),
            "raw_interval_duration_seconds": sum(end - start for start, end in intervals),
            "union_interval_duration_seconds": sum(end - start for start, end in merged),
        })

    source_videos = list(next_video_rows.values()) + et_video_rows
    source_video_durations = [row["video_duration_seconds"] for row in source_videos]
    next_unique_duration = sum(row["video_duration_seconds"] for row in next_video_rows.values())
    next_qa_duration = sum(float(row["video_duration"]) for row in retained_next)
    et_unique_duration = sum(row["video_duration_seconds"] for row in et_video_rows)

    condition_rows = [
        {"condition": "sufficient", "qa_records": len(retained_next), "video_inputs": len(next_video_rows), "video_input_seconds": next_unique_duration},
        {"condition": "insufficient", "qa_records": len(retained_next), "video_inputs": len(retained_next), "video_input_seconds": next_qa_duration},
    ]
    for alpha in alphas:
        condition_rows.append({
            "condition": f"partial_alpha_{alpha:g}",
            "qa_records": len(retained_next),
            "video_inputs": len(retained_next),
            "video_input_seconds": next_qa_duration,
        })
    condition_rows.append({
        "condition": "conflicting",
        "qa_records": len(accepted_conflict),
        "video_inputs": len(et_video_rows),
        "video_input_seconds": et_unique_duration,
    })

    next_source_groups = list(next_intervals_by_video.values())
    et_source_groups = [list(row["instances"].values()) for row in et_video_rows]
    next_qa_groups = []
    for row in retained_next:
        duration = float(row["video_duration"])
        next_qa_groups.append([
            interval
            for value in row.get("evidence_intervals", [])
            if (interval := normalize_interval(value, duration)) is not None
        ])
    et_qa_groups = []
    for qa in accepted_conflict:
        parent = str(qa["parent_job_id"])
        duration = parent_meta[parent]["video_duration_seconds"]
        unique_intervals = {}
        for option in qa.get("options", []):
            for support in option.get("supported_by", []):
                interval = normalize_interval(support.get("interval"), duration)
                if interval:
                    marker = (str(support.get("instance_id")), *interval)
                    unique_intervals[marker] = interval
        et_qa_groups.append(list(unique_intervals.values()))

    next_qa_summary = collection_interval_summary(next_qa_groups)
    et_qa_summary = collection_interval_summary(et_qa_groups)
    condition_evidence_rows = []
    for condition in ["sufficient", "insufficient", *[f"partial_alpha_{alpha:g}" for alpha in alphas]]:
        condition_evidence_rows.append({"condition": condition, **next_qa_summary})
    condition_evidence_rows.append({"condition": "conflicting", **et_qa_summary})

    stats = {
        "counting_notes": {
            "video_duration": (
                "Uses manifest/source metadata. Insufficient and partial files are generated as "
                "duration-preserving variants, so their input duration is inherited from the source QA."
            ),
            "source_video_evidence": (
                "Intervals are unioned only within each source video, never across unrelated timelines."
            ),
            "qa_evidence": (
                "Intervals are unioned only within each QA. Reuse by different QAs remains counted."
            ),
        },
        "filter": {
            "original_next_qa": len(selected),
            "language_prior_exclusion_keys": len(exclusions),
            "removed_next_qa": len(removed_present),
            "retained_next_qa": len(retained_next),
            "unknown_exclusion_keys": len(unknown_exclusions),
            "exclusion_file": None if a.no_exclusions else str(a.exclusions),
        },
        "summary": {
            "unique_source_videos": len(source_videos),
            "unique_next_source_videos": len(next_video_rows),
            "unique_etbench_source_videos": len(et_video_rows),
            "unique_qa": len(retained_next) + len(accepted_conflict),
            "condition_level_qa_records": sum(row["qa_records"] for row in condition_rows),
            "distinct_video_inputs": sum(row["video_inputs"] for row in condition_rows),
            "total_video_input_seconds": round(sum(row["video_input_seconds"] for row in condition_rows), 6),
        },
        "video": {
            "unique_source_duration_seconds": distribution(source_video_durations),
            "unique_source_duration_bins": dict(sorted(Counter(duration_bin(value) for value in source_video_durations).items())),
            "duration_by_source": group_numeric(source_videos, "source_dataset", "video_duration_seconds"),
            "condition_expanded": condition_rows,
        },
        "evidence": {
            "source_video_level": {
                "nextgqa": collection_interval_summary(next_source_groups),
                "etbench_official_occurrences": collection_interval_summary(et_source_groups),
                "combined": collection_interval_summary(next_source_groups + et_source_groups),
            },
            "qa_level": {
                "nextgqa": next_qa_summary,
                "etbench_support": et_qa_summary,
                "combined": collection_interval_summary(next_qa_groups + et_qa_groups),
                "nextgqa_by_question_type": category_interval_stats(next_qa_evidence_rows),
                "etbench_by_conflict_type": category_interval_stats(et_qa_evidence_rows),
            },
            "condition_expanded": condition_evidence_rows,
            "condition_semantics": {
                "sufficient": "Intervals denote visible supporting evidence.",
                "insufficient": "The same intervals denote evidence removed/frozen by the operator.",
                "partial": "The same intervals denote alpha-degraded evidence regions.",
                "conflicting": "Intervals denote distinct supported ETBench event occurrences.",
            },
        },
        "categories": {
            "nextgqa_question_types": dict(sorted(Counter(str(row.get("question_type", "UNKNOWN")) for row in retained_next).items())),
            "etbench_conflict_types": dict(sorted(Counter(str(row.get("conflict_type", "UNKNOWN")) for row in accepted_conflict).items())),
        },
    }
    details = {
        "source_videos": [
            {key: value for key, value in row.items() if key != "instances"}
            for row in source_videos
        ],
        "nextgqa_qa_evidence": next_qa_evidence_rows,
        "etbench_qa_evidence": et_qa_evidence_rows,
    }
    return stats, details


def markdown(stats: dict) -> str:
    f, s, v, e = stats["filter"], stats["summary"], stats["video"], stats["evidence"]
    condition_rows = "\n".join(
        f"| {row['condition']} | {row['qa_records']:,} | {row['video_inputs']:,} | {row['video_input_seconds'] / 3600:,.2f} |"
        for row in v["condition_expanded"]
    )
    next_types = "\n".join(f"| {key} | {value:,} |" for key, value in stats["categories"]["nextgqa_question_types"].items())
    conflict_types = "\n".join(f"| {key} | {value:,} |" for key, value in stats["categories"]["etbench_conflict_types"].items())
    next_evidence_types = "\n".join(
        f"| {key} | {value['qa_count']:,} | {value['raw_interval_count']:,} | {value['raw_interval_duration_seconds']:,.2f} | {value['union_interval_duration_seconds']:,.2f} |"
        for key, value in e["qa_level"]["nextgqa_by_question_type"].items()
    )
    et_evidence_types = "\n".join(
        f"| {key} | {value['qa_count']:,} | {value['raw_interval_count']:,} | {value['raw_interval_duration_seconds']:,.2f} | {value['union_interval_duration_seconds']:,.2f} |"
        for key, value in e["qa_level"]["etbench_by_conflict_type"].items()
    )
    source_evidence = e["source_video_level"]["combined"]
    qa_evidence = e["qa_level"]["combined"]
    retained_et_qa = sum(stats["categories"]["etbench_conflict_types"].values())
    condition_evidence_rows = "\n".join(
        f"| {row['condition']} | {row['raw_interval_count']:,} | {row['raw_interval_duration_seconds']:,.2f} | {row['merged_interval_count']:,} | {row['union_duration_seconds']:,.2f} |"
        for row in e["condition_expanded"]
    )
    return f"""# 保留 Benchmark 视频与证据区间统计

## 过滤结果

| 项目 | 数量 |
|---|---:|
| 原始 NExT-GQA QA | {f['original_next_qa']:,} |
| 删除的强语言先验 QA | {f['removed_next_qa']:,} |
| 保留 NExT-GQA QA | {f['retained_next_qa']:,} |
| 保留 ETBench conflict QA | {retained_et_qa:,} |
| 去重源视频 | {s['unique_source_videos']:,} |
| 条件级 QA | {s['condition_level_qa_records']:,} |
| 实际视频输入/变体 | {s['distinct_video_inputs']:,} |

## 视频时长

- 去重源视频总时长：{v['unique_source_duration_seconds']['sum'] / 3600:,.2f} 小时。
- 条件展开后视频输入总时长：{s['total_video_input_seconds'] / 3600:,.2f} 小时。
- 去重源视频平均/中位时长：{v['unique_source_duration_seconds']['mean']:.2f}s / {v['unique_source_duration_seconds']['median']:.2f}s。

视频时长取 manifest/source metadata；insufficient 与 partial 是等时长生成变体，因此按对应源 QA 时长计。若后续需要检查编码产物本身，可另加 `ffprobe` 文件级审计。

| 条件 | QA 数 | 视频输入数 | 视频输入小时 |
|---|---:|---:|---:|
{condition_rows}

## Evidence 区间

| 口径 | 区间数 | 原始累计时长(s) | 合并后区间数 | Union 时长(s) |
|---|---:|---:|---:|---:|
| 源视频级去重 | {source_evidence['raw_interval_count']:,} | {source_evidence['raw_interval_duration_seconds']:,.2f} | {source_evidence['merged_interval_count']:,} | {source_evidence['union_duration_seconds']:,.2f} |
| QA 级使用量 | {qa_evidence['raw_interval_count']:,} | {qa_evidence['raw_interval_duration_seconds']:,.2f} | {qa_evidence['merged_interval_count']:,} | {qa_evidence['union_duration_seconds']:,.2f} |

源视频级适合描述数据本体；QA 级会重复统计同一视频区间服务不同问题的情况。Insufficient/partial 的 interval 分别表示被冻结和按 alpha 退化的目标区域，不应解释为仍然可见的完整证据。

### 按 Evidence 条件展开

| 条件 | 区间数 | 原始累计时长(s) | QA 内合并后区间数 | QA 内 Union 时长(s) |
|---|---:|---:|---:|---:|
{condition_evidence_rows}

## NExT-GQA 类型

| 类型 | QA 数 |
|---|---:|
{next_types}

## ETBench conflict 类型

| 类型 | QA 数 |
|---|---:|
{conflict_types}

## Evidence 按类型

### NExT-GQA question type

| 类型 | QA 数 | 区间数 | 原始累计时长(s) | QA 内 Union 时长(s) |
|---|---:|---:|---:|---:|
{next_evidence_types}

### ETBench conflict type

| 类型 | QA 数 | 区间数 | 原始累计时长(s) | QA 内 Union 时长(s) |
|---|---:|---:|---:|---:|
{et_evidence_types}
"""


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    a = args()
    stats, details = build_stats(a)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output_dir / "summary.json", stats)
    atomic_json(a.output_dir / "details.json", details)
    (a.output_dir / "summary.md").write_text(markdown(stats), encoding="utf-8")
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

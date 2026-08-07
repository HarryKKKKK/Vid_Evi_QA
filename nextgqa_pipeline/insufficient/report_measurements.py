#!/usr/bin/env python3
"""Summarize dense-window measurement JSONL files for a pipeline report.

The script is read-only. It reproduces the current positive-control threshold
and candidate projection logic, but deliberately calls the selected windows
"candidates": a high normalized gain does not establish semantic evidence
sufficiency.

Example
-------
python nextgqa_pipeline/insufficient/report_measurements.py \
  --measurements 'nextgqa_pipeline/insufficient/measurements.[0-7].jsonl' \
  --output-json nextgqa_pipeline/insufficient/measurement_report.json \
  --output-markdown nextgqa_pipeline/insufficient/measurement_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = str(ROOT / "nextgqa_pipeline/insufficient/measurements.*.jsonl")
DEFAULT_JSON = ROOT / "nextgqa_pipeline/insufficient/measurement_report.json"
DEFAULT_MARKDOWN = ROOT / "nextgqa_pipeline/insufficient/measurement_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default=DEFAULT_GLOB)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--minimum-controls", type=int, default=20)
    parser.add_argument("--uncertainty-band", type=float, default=0.05)
    parser.add_argument("--maximum-coverage", type=float, default=0.60)
    parser.add_argument(
        "--full-gain-bins", default="0.001,0.01,0.1,0.5,1,2",
        help="Comma-separated cutoffs used to report unstable normalization.",
    )
    return parser.parse_args()


def lower_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.floor(probability * (len(ordered) - 1))))
    return ordered[index]


def quantile_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "p05": lower_quantile(values, 0.05),
        "median": lower_quantile(values, 0.50),
        "p95": lower_quantile(values, 0.95),
        "max": max(values),
    }


def evidence_group(record: dict) -> str:
    return "multi_interval" if int(record.get("evidence_count", 0)) >= 2 else "single_interval"


def merge_intervals(intervals: Sequence[Sequence[float]], duration: float) -> list[list[float]]:
    valid = sorted(
        (max(0.0, float(start)), min(duration, float(end)))
        for start, end in intervals
        if float(end) > float(start)
    )
    merged: list[list[float]] = []
    for start, end in valid:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def minimal_positive_windows(intervals: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    candidates = sorted(
        {(float(start), float(end)) for start, end in intervals},
        key=lambda item: (item[1] - item[0], item[0]),
    )
    kept: list[tuple[float, float]] = []
    for start, end in candidates:
        if any(start <= child_start + 1e-6 and end >= child_end - 1e-6
               for child_start, child_end in kept):
            continue
        kept.append((start, end))
    return kept


def interval_ratio(intervals: Sequence[Sequence[float]], duration: float) -> float:
    return sum(end - start for start, end in merge_intervals(intervals, duration)) / max(duration, 1e-9)


class RecordSource:
    """Iterate only the latest JSONL record for each key without retaining payloads."""

    def __init__(self, pattern: str):
        self.paths = sorted(Path(path) for path in glob.glob(pattern))
        if not self.paths:
            raise SystemExit(f"no measurement files matched {pattern!r}")
        self.latest: dict[str, tuple[Path, int]] = {}
        self.physical_records = 0
        for path in self.paths:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = str(record.get("key", f"__missing_key__:{path}:{line_number}"))
                    self.latest[key] = (path, line_number)
                    self.physical_records += 1

    def records(self) -> Iterator[dict]:
        for path in self.paths:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = str(record.get("key", f"__missing_key__:{path}:{line_number}"))
                    if self.latest.get(key) == (path, line_number):
                        yield record


def analyze(source: RecordSource, args: argparse.Namespace) -> dict:
    statuses: Counter = Counter()
    evidence_groups: Counter = Counter()
    question_types: Counter = Counter()
    sampling_configs: Counter = Counter()
    durations: list[float] = []
    elapsed: list[float] = []
    exceptions: list[dict] = []

    model_total: Counter = Counter()
    model_ok: Counter = Counter()
    eligible: Counter = Counter()
    group_total: Counter = Counter()
    group_eligible: Counter = Counter()
    correct: Counter = Counter()
    gains: defaultdict = defaultdict(list)
    ineligible_conditions: Counter = Counter()
    full_gain_bins = [float(part) for part in args.full_gain_bins.split(",") if part.strip()]
    unstable_bins: Counter = Counter()

    controls: defaultdict = defaultdict(list)
    control_items: defaultdict = defaultdict(set)
    controls_by_group: Counter = Counter()
    control_group_items: defaultdict = defaultdict(set)
    discovery_values: defaultdict = defaultdict(list)
    discovery_predicts_gold: defaultdict = defaultdict(list)

    for record in source.records():
        status = str(record.get("status"))
        statuses[status] += 1
        group = evidence_group(record)
        evidence_groups[group] += 1
        question_types[str(record.get("question_type", ""))] += 1
        durations.append(float(record.get("duration", 0.0)))
        elapsed.append(float(record.get("elapsed_seconds", 0.0)))
        sampling = record.get("sampling", {})
        sampling_configs[(sampling.get("fps"), sampling.get("frame_width"))] += 1
        if status != "ok":
            exceptions.append({
                "key": record.get("key"), "status": status,
                "error": record.get("error"), "discard_reason": record.get("discard_reason"),
            })

        for tag, model in record.get("models", {}).items():
            model_total[tag] += 1
            group_total[(tag, group)] += 1
            if model.get("status") != "ok":
                existing_exception = next(
                    (item for item in exceptions if item.get("key") == record.get("key")),
                    None,
                )
                if existing_exception is None:
                    exceptions.append({
                        "key": record.get("key"), "status": record.get("status"),
                        "model_tag": tag, "model_status": model.get("status"),
                        "error": model.get("error"),
                    })
                else:
                    existing_exception.update({
                        "model_tag": tag,
                        "model_status": model.get("status"),
                        "error": model.get("error") or existing_exception.get("error"),
                    })
                continue
            model_ok[tag] += 1
            for label, block, side in (
                ("full_visual", "full", "visual"),
                ("full_blind", "full", "blind"),
                ("gold_visual", "gold_only", "visual"),
                ("gold_blind", "gold_only", "blind"),
            ):
                correct[(tag, label)] += int(model[block][side]["prediction"] == record.get("gold"))
            gains[(tag, "full")].append(float(model["full"]["gain"]))
            gains[(tag, "gold_only")].append(float(model["gold_only"]["gain"]))

            if model.get("eligible"):
                eligible[tag] += 1
                group_eligible[(tag, group)] += 1
                full_gain = float(model["full"]["gain"])
                gains[(tag, "eligible_full")].append(full_gain)
                gains[(tag, "eligible_gold_only")].append(float(model["gold_only"]["gain"]))
                for cutoff in full_gain_bins:
                    unstable_bins[(tag, cutoff)] += int(full_gain < cutoff)
            else:
                if model["full"]["visual"]["prediction"] != record.get("gold"):
                    ineligible_conditions[(tag, "full_wrong")] += 1
                if model["gold_only"]["visual"]["prediction"] != record.get("gold"):
                    ineligible_conditions[(tag, "gold_only_wrong")] += 1
                if float(model["full"]["gain"]) <= 1e-6:
                    ineligible_conditions[(tag, "full_gain_nonpositive")] += 1
                if float(model["gold_only"]["gain"]) <= 1e-6:
                    ineligible_conditions[(tag, "gold_gain_nonpositive")] += 1
                continue

            for scale_key, scale in model.get("scales", {}).items():
                bucket = (tag, scale_key)
                for window in scale.get("windows", []):
                    value = window.get("discovery", {}).get("normalized_gain")
                    if value is not None:
                        discovery_values[bucket].append(float(value))
                        discovery_predicts_gold[bucket].append(
                            window["discovery"]["visual"]["prediction"] == record.get("gold")
                        )
                    positive = window.get("positive_control", {}).get("normalized_gain")
                    if positive is not None:
                        controls[bucket].append(float(positive))
                        control_items[bucket].add(record["key"])
                        controls_by_group[(tag, scale_key, group)] += 1
                        control_group_items[(tag, scale_key, group)].add(record["key"])

    scale_reports: dict[str, dict] = {}
    thresholds: dict[tuple[str, str], float | None] = {}
    for tag, scale_key in sorted(set(controls) | set(discovery_values)):
        bucket = (tag, scale_key)
        positives = controls[bucket]
        raw_threshold = (
            lower_quantile(positives, 1.0 - args.target_recall) if positives else None
        )
        threshold = max(0.0, raw_threshold) if raw_threshold is not None else None
        thresholds[bucket] = threshold
        recall_at_zero = (
            sum(value >= 0.0 for value in positives) / len(positives) if positives else None
        )
        usable = bool(
            len(positives) >= args.minimum_controls
            and recall_at_zero is not None
            and recall_at_zero >= args.target_recall
        )
        values = discovery_values[bucket]
        selected_indices = [
            index for index, value in enumerate(values)
            if threshold is not None and value >= threshold
        ]
        scale_reports[scale_key] = {
            "model_tag": tag,
            "positive_controls": quantile_summary(positives),
            "positive_control_items": len(control_items[bucket]),
            "positive_controls_by_group": {
                group: {
                    "windows": controls_by_group[(tag, scale_key, group)],
                    "items": len(control_group_items[(tag, scale_key, group)]),
                }
                for group in ("single_interval", "multi_interval")
            },
            "threshold": threshold,
            "target_recall": args.target_recall,
            "achieved_control_recall": (
                sum(value >= threshold for value in positives) / len(positives)
                if positives and threshold is not None else None
            ),
            "usable_by_current_rule": usable,
            "seed_windows": len(values),
            "seed_candidates_at_threshold": len(selected_indices),
            "seed_candidate_rate": len(selected_indices) / len(values) if values else None,
            "candidate_predicts_gold_rate": (
                sum(discovery_predicts_gold[bucket][index] for index in selected_indices)
                / len(selected_indices) if selected_indices else None
            ),
            "normalized_gain": quantile_summary(values),
        }

    usable_thresholds = {
        (report["model_tag"], scale_key): report["threshold"]
        for scale_key, report in scale_reports.items()
        if report["usable_by_current_rule"] and report["threshold"] is not None
    }

    initial_coverages: list[float] = []
    added_coverages: list[float] = []
    final_coverages: list[float] = []
    minimal_counts: list[int] = []
    candidate_counts_by_scale: Counter = Counter()
    items_with_candidates = 0
    coverage_over_limit = 0
    coverage_at_least_80 = 0
    monotonicity: defaultdict = defaultdict(Counter)

    for record in source.records():
        for tag, model in record.get("models", {}).items():
            if model.get("status") != "ok" or not model.get("eligible"):
                continue
            candidates: list[Sequence[float]] = []
            exact_positive: dict[str, list[bool]] = {}
            for scale_key, scale in model.get("scales", {}).items():
                threshold = usable_thresholds.get((tag, scale_key))
                cutoff = threshold - args.uncertainty_band if threshold is not None else None
                exact_positive[scale_key] = []
                for window in scale.get("windows", []):
                    value = window.get("discovery", {}).get("normalized_gain")
                    exact_positive[scale_key].append(
                        value is not None and threshold is not None and float(value) >= threshold
                    )
                    if value is not None and cutoff is not None and float(value) >= cutoff:
                        candidates.append(window["interval"])
                        candidate_counts_by_scale[scale_key] += 1

            leaves = minimal_positive_windows(candidates)
            items_with_candidates += int(bool(leaves))
            minimal_counts.append(len(leaves))
            duration = float(record["duration"])
            initial = interval_ratio(record.get("initial_mask", []), duration)
            final = interval_ratio([*record.get("initial_mask", []), *leaves], duration)
            initial_coverages.append(initial)
            added_coverages.append(final - initial)
            final_coverages.append(final)
            coverage_over_limit += int(final > args.maximum_coverage)
            coverage_at_least_80 += int(final >= 0.80)

            for child_key, parent_key in (("w4_s2", "w8_s4"), ("w8_s4", "w16_s8")):
                if child_key not in exact_positive or parent_key not in exact_positive:
                    continue
                child_windows = model["scales"][child_key]["windows"]
                parent_windows = model["scales"][parent_key]["windows"]
                stats = monotonicity[(tag, child_key, parent_key)]
                for child_index, child in enumerate(child_windows):
                    if not exact_positive[child_key][child_index]:
                        continue
                    containing = [
                        parent_index for parent_index, parent in enumerate(parent_windows)
                        if parent["interval"][0] <= child["interval"][0] + 1e-6
                        and parent["interval"][1] >= child["interval"][1] - 1e-6
                    ]
                    if containing:
                        stats["child_positive_with_parent"] += 1
                        stats["child_positive_parent_negative"] += int(
                            not any(exact_positive[parent_key][index] for index in containing)
                        )
                for parent_index, parent in enumerate(parent_windows):
                    if not exact_positive[parent_key][parent_index]:
                        continue
                    inside = [
                        child_index for child_index, child in enumerate(child_windows)
                        if child["interval"][0] >= parent["interval"][0] - 1e-6
                        and child["interval"][1] <= parent["interval"][1] + 1e-6
                    ]
                    if inside:
                        stats["parent_positive_with_children"] += 1
                        stats["parent_positive_children_negative"] += int(
                            not any(exact_positive[child_key][index] for index in inside)
                        )

    model_reports: dict[str, dict] = {}
    for tag in sorted(model_total):
        model_reports[tag] = {
            "records": model_total[tag],
            "model_ok": model_ok[tag],
            "eligible": eligible[tag],
            "eligibility_rate": eligible[tag] / model_ok[tag] if model_ok[tag] else None,
            "eligible_by_group": {
                group: {
                    "total": group_total[(tag, group)],
                    "eligible": group_eligible[(tag, group)],
                    "rate": (
                        group_eligible[(tag, group)] / group_total[(tag, group)]
                        if group_total[(tag, group)] else None
                    ),
                }
                for group in ("single_interval", "multi_interval")
            },
            "accuracy": {
                label: correct[(tag, label)] / model_ok[tag]
                for label in ("full_visual", "full_blind", "gold_visual", "gold_blind")
            },
            "gain": {
                name: quantile_summary(gains[(tag, name)])
                for name in ("full", "gold_only", "eligible_full", "eligible_gold_only")
            },
            "eligible_full_gain_below": {
                str(cutoff): unstable_bins[(tag, cutoff)] for cutoff in full_gain_bins
            },
            "ineligible_conditions_nonexclusive": {
                reason: ineligible_conditions[(tag, reason)]
                for reason in (
                    "full_wrong", "gold_only_wrong",
                    "full_gain_nonpositive", "gold_gain_nonpositive",
                )
            },
        }

    monotonicity_report: dict[str, dict] = {}
    for (tag, child_key, parent_key), stats in monotonicity.items():
        report = dict(stats)
        report["child_positive_parent_negative_rate"] = (
            stats["child_positive_parent_negative"] / stats["child_positive_with_parent"]
            if stats["child_positive_with_parent"] else None
        )
        report["parent_positive_children_negative_rate"] = (
            stats["parent_positive_children_negative"] / stats["parent_positive_with_children"]
            if stats["parent_positive_with_children"] else None
        )
        monotonicity_report[f"{tag}/{child_key}->{parent_key}"] = report

    return {
        "configuration": {
            "measurement_glob": args.measurements,
            "files": [str(path) for path in source.paths],
            "target_recall": args.target_recall,
            "minimum_controls": args.minimum_controls,
            "uncertainty_band": args.uncertainty_band,
            "maximum_coverage": args.maximum_coverage,
        },
        "data": {
            "physical_records": source.physical_records,
            "latest_unique_records": len(source.latest),
            "duplicate_or_retried_records": source.physical_records - len(source.latest),
            "status": dict(statuses),
            "exceptions": exceptions,
            "evidence_groups": dict(evidence_groups),
            "question_types": dict(question_types),
            "sampling_configs": {
                f"fps={fps},frame_width={width}": count
                for (fps, width), count in sampling_configs.items()
            },
            "duration_seconds": quantile_summary(durations),
            "elapsed_seconds_per_item": quantile_summary(elapsed),
        },
        "models": model_reports,
        "scales": scale_reports,
        "candidate_projection": {
            "eligible_items": len(final_coverages),
            "items_with_candidates": items_with_candidates,
            "items_with_candidates_rate": (
                items_with_candidates / len(final_coverages) if final_coverages else None
            ),
            "candidate_windows_by_scale_at_threshold_minus_uncertainty": dict(candidate_counts_by_scale),
            "candidate_windows_before_minimal_leaf_filter": sum(candidate_counts_by_scale.values()),
            "minimal_positive_windows_total": sum(minimal_counts),
            "minimal_positive_windows_per_item": quantile_summary(minimal_counts),
            "initial_coverage": quantile_summary(initial_coverages),
            "added_coverage": quantile_summary(added_coverages),
            "projected_final_coverage": quantile_summary(final_coverages),
            "items_over_maximum_coverage": coverage_over_limit,
            "items_over_maximum_coverage_rate": (
                coverage_over_limit / len(final_coverages) if final_coverages else None
            ),
            "items_at_least_80_percent_coverage": coverage_at_least_80,
            "items_at_least_80_percent_coverage_rate": (
                coverage_at_least_80 / len(final_coverages) if final_coverages else None
            ),
        },
        "monotonicity": monotonicity_report,
        "interpretation_note": (
            "Candidate statistics measure answer-associated visual gain, not semantic evidence "
            "sufficiency. Do not use this report to authorize final masking without a semantic verifier."
        ),
    }


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def markdown_report(report: dict) -> str:
    data = report["data"]
    model_tag, model = next(iter(report["models"].items()))
    projection = report["candidate_projection"]
    lines = [
        "# NExT-GQA 候选窗口测量报告",
        "",
        "> 注意：窗口阳性表示视觉内容提高了正确答案分数，不等于该窗口包含问题语义对齐的充分 evidence。",
        "",
        "## 1. 数据完整性",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 唯一 item | {data['latest_unique_records']} |",
        f"| 物理 JSONL 记录 | {data['physical_records']} |",
        f"| 重试/重复记录 | {data['duplicate_or_retried_records']} |",
    ]
    for status, count in sorted(data["status"].items()):
        lines.append(f"| status={status} | {count} |")
    duration = data["duration_seconds"]
    lines += [
        f"| 视频时长中位数 | {num(duration['median'], 1)} s |",
        f"| 视频时长 P95 / 最大值 | {num(duration['p95'], 1)} / {num(duration['max'], 1)} s |",
        "",
        "## 2. 构造模型表现",
        "",
        f"模型：`{model_tag}`",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| full accuracy | {pct(model['accuracy']['full_visual'])} |",
        f"| full blind accuracy | {pct(model['accuracy']['full_blind'])} |",
        f"| gold-only accuracy | {pct(model['accuracy']['gold_visual'])} |",
        f"| gold-only blind accuracy | {pct(model['accuracy']['gold_blind'])} |",
        f"| model-item eligible | {model['eligible']} / {model['model_ok']} ({pct(model['eligibility_rate'])}) |",
    ]
    for group, stats in model["eligible_by_group"].items():
        lines.append(f"| {group} eligible | {stats['eligible']} / {stats['total']} ({pct(stats['rate'])}) |")
    lines += [
        "",
        "## 3. 多尺度正对照与候选率",
        "",
        "| 尺度 | 正对照窗口 / item | 阈值 | 正对照召回 | seed候选 / 总窗口 | 候选率 | 候选预测gold率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scale_key in sorted(report["scales"], key=lambda key: int(key.split("_")[0][1:])):
        scale = report["scales"][scale_key]
        lines.append(
            f"| {scale_key} | {scale['positive_controls']['n']} / {scale['positive_control_items']} "
            f"| {num(scale['threshold'])} | {pct(scale['achieved_control_recall'])} "
            f"| {scale['seed_candidates_at_threshold']} / {scale['seed_windows']} "
            f"| {pct(scale['seed_candidate_rate'])} | {pct(scale['candidate_predicts_gold_rate'])} |"
        )
    lines += [
        "",
        "## 4. 旧遮挡规则的候选覆盖预测",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| eligible item | {projection['eligible_items']} |",
        f"| 至少一个候选的 item | {projection['items_with_candidates']} ({pct(projection['items_with_candidates_rate'])}) |",
        f"| uncertainty前候选窗口 | {projection['candidate_windows_before_minimal_leaf_filter']} |",
        f"| 最小阳性叶窗口 | {projection['minimal_positive_windows_total']} |",
        f"| 官方 mask 覆盖率中位数 | {pct(projection['initial_coverage']['median'])} |",
        f"| 新增覆盖率中位数 | {pct(projection['added_coverage']['median'])} |",
        f"| 预计最终覆盖率中位数 | {pct(projection['projected_final_coverage']['median'])} |",
        f"| 超过最大覆盖率的 item | {projection['items_over_maximum_coverage']} ({pct(projection['items_over_maximum_coverage_rate'])}) |",
        f"| 覆盖率至少80%的 item | {projection['items_at_least_80_percent_coverage']} ({pct(projection['items_at_least_80_percent_coverage_rate'])}) |",
        "",
        "## 5. 归一化稳定性",
        "",
        "| eligible full gain 小于 | item数 |",
        "|---|---:|",
    ]
    for cutoff, count in model["eligible_full_gain_below"].items():
        lines.append(f"| {cutoff} | {count} |")
    lines += [
        "",
        "## 6. 级联单调性诊断",
        "",
        "| 子尺度 → 父尺度 | 子阳性、父阴性 | 父阳性、全部子阴性 |",
        "|---|---:|---:|",
    ]
    for relation, stats in sorted(report["monotonicity"].items()):
        lines.append(
            f"| {relation} | {pct(stats['child_positive_parent_negative_rate'])} "
            f"| {pct(stats['parent_positive_children_negative_rate'])} |"
        )
    lines += [
        "",
        "## 7. 结论",
        "",
        "- 全量抽帧、matched blind、A–E logprob 和多尺度扫描已工程跑通。",
        "- 当前分数适合作为高召回候选生成器，不适合作为最终 evidence 判定器。",
        "- 大约一半 seed 窗口会超过阈值，旧规则会造成严重覆盖膨胀。",
        "- 下一阶段必须使用问题语义 verifier 区分 `sufficient_evidence` 与 `answer_correlated_only`。",
        "- 在 semantic verifier 完成前，不应运行旧 `finalize`。",
        "",
    ]
    if data["exceptions"]:
        lines += ["## 8. 异常记录", ""]
        for item in data["exceptions"]:
            lines.append(f"- `{item.get('key')}`：{item.get('status')}；{item.get('error') or item.get('discard_reason') or '详见JSON报告'}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        raise SystemExit("--target-recall must be in (0,1]")
    source = RecordSource(args.measurements)
    report = analyze(source, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    print(f"\nJSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()

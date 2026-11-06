#!/usr/bin/env python3

import json
import csv
import argparse
from collections import defaultdict


def load_json_or_jsonl(path):
    """
    支持普通 JSON list，也支持 JSONL。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 如果外层是 dict，尝试从常见字段里取 list
            for key in ["data", "results", "items", "samples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        else:
            raise ValueError("Unsupported JSON structure.")
    except json.JSONDecodeError:
        # 尝试按 JSONL 读取
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {i}: {e}")
        return records


def normalize_answer(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip().upper()
    return x


def overlap(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)


def point_inside(t, start, end):
    return start <= t <= end


def evidence_hit(pred_seconds, gt_intervals):
    """
    判断模型预测的 evidence_seconds 是否命中 GT evidence_intervals。

    支持：
    - pred_seconds = [12, 18]
    - pred_seconds = [12]
    - pred_seconds = {"start": 12, "end": 18}
    - pred_seconds = [{"start": 12, "end": 18}, ...]
    """

    if not pred_seconds:
        return False

    if not gt_intervals:
        return False

    pred_intervals = []

    # 情况 1：dict，例如 {"start": 12, "end": 18}
    if isinstance(pred_seconds, dict):
        if "start" in pred_seconds and "end" in pred_seconds:
            pred_intervals.append(
                (float(pred_seconds["start"]), float(pred_seconds["end"]))
            )

    # 情况 2：list
    elif isinstance(pred_seconds, list):

        # list of dict，例如 [{"start": 12, "end": 18}]
        if all(isinstance(x, dict) for x in pred_seconds):
            for x in pred_seconds:
                if "start" in x and "end" in x:
                    pred_intervals.append((float(x["start"]), float(x["end"])))

        # list of numbers，例如 [12, 18] 或 [12]
        else:
            nums = []
            for x in pred_seconds:
                try:
                    nums.append(float(x))
                except Exception:
                    pass

            if len(nums) == 1:
                t = nums[0]
                for gt in gt_intervals:
                    if point_inside(t, float(gt["start"]), float(gt["end"])):
                        return True
                return False

            elif len(nums) >= 2:
                pred_intervals.append((min(nums), max(nums)))

    # 情况 3：单个数字
    else:
        try:
            t = float(pred_seconds)
            for gt in gt_intervals:
                if point_inside(t, float(gt["start"]), float(gt["end"])):
                    return True
            return False
        except Exception:
            return False

    for ps, pe in pred_intervals:
        for gt in gt_intervals:
            gs = float(gt["start"])
            ge = float(gt["end"])

            # 如果预测退化成单点
            if ps == pe:
                if point_inside(ps, gs, ge):
                    return True
            else:
                if overlap(ps, pe, gs, ge):
                    return True

    return False


def extract_alpha(item, alpha_key=None):
    """
    尝试从样本中提取 alpha。
    默认先找 item["alpha"]。
    如果你指定 --alpha-key，就优先用那个字段。
    """

    candidate_keys = []

    if alpha_key:
        candidate_keys.append(alpha_key)

    candidate_keys.extend([
        "alpha",
        "mask_alpha",
        "evidence_alpha",
        "perturb_alpha",
        "perturbation_alpha",
        "corruption_alpha",
        "noise_alpha",
        "partial_alpha",
    ])

    # 先查顶层
    for key in candidate_keys:
        if key in item:
            return item[key]

    # 再查常见嵌套字段
    nested_keys = ["metadata", "meta", "config", "perturbation", "ablation"]

    for nk in nested_keys:
        obj = item.get(nk)
        if isinstance(obj, dict):
            for key in candidate_keys:
                if key in obj:
                    return obj[key]

    return "UNKNOWN"


def alpha_sort_key(x):
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))


def new_counter():
    return {
        "total": 0,
        "success": 0,
        "evidence_hit": 0,
        "unanswerable": 0,
        "answerable": 0,
        "success_wrong_evidence": 0,
        "wrong_answer_correct_evidence": 0,
    }


def frac(count, total):
    if total == 0:
        return "0/0"
    return f"{count}/{total}"


def rate(count, total):
    if total == 0:
        return 0.0
    return count / total


def main():
    parser = argparse.ArgumentParser(
        description="Analyze partial metrics by alpha for empirical ablation."
    )

    parser.add_argument("json_file", help="Path to evaluation JSON or JSONL file.")

    parser.add_argument(
        "--condition",
        default="partial",
        help=(
            "Which evidence_condition to analyze. "
            "Default: partial. "
            "Use 'all' to include all conditions."
        ),
    )

    parser.add_argument(
        "--alpha-key",
        default=None,
        help=(
            "Field name for alpha. "
            "Default: auto-detect from common names such as alpha, mask_alpha, partial_alpha."
        ),
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path, e.g. partial_alpha_metrics.csv",
    )

    args = parser.parse_args()

    data = load_json_or_jsonl(args.json_file)

    stats = defaultdict(new_counter)

    skipped_by_condition = 0
    missing_alpha = 0

    for item in data:
        condition = item.get("evidence_condition", "UNKNOWN")

        if args.condition != "all" and condition != args.condition:
            skipped_by_condition += 1
            continue

        alpha = extract_alpha(item, args.alpha_key)

        if alpha == "UNKNOWN":
            missing_alpha += 1

        alpha = str(alpha)

        stats[alpha]["total"] += 1

        gt_answer = normalize_answer(item.get("answer"))

        parsed = item.get("model_parsed", {})
        if not isinstance(parsed, dict):
            parsed = {}

        pred_answer = normalize_answer(parsed.get("answer"))
        pred_label = parsed.get("label")
        if isinstance(pred_label, str):
            pred_label = pred_label.strip().upper()

        pred_evidence = parsed.get("evidence_seconds", [])

        answer_correct = pred_answer == gt_answer

        gt_intervals = item.get("evidence_intervals", [])
        evidence_correct = evidence_hit(pred_evidence, gt_intervals)

        is_unanswerable = pred_label == "UNANSWERABLE"

        if answer_correct:
            stats[alpha]["success"] += 1

        if evidence_correct:
            stats[alpha]["evidence_hit"] += 1

        if is_unanswerable:
            stats[alpha]["unanswerable"] += 1
        else:
            stats[alpha]["answerable"] += 1

        if answer_correct and not evidence_correct:
            stats[alpha]["success_wrong_evidence"] += 1

        if (not answer_correct) and evidence_correct:
            stats[alpha]["wrong_answer_correct_evidence"] += 1

    print("=" * 90)
    print(f"Partial Metrics by Alpha")
    print("=" * 90)
    print(f"Input file        : {args.json_file}")
    print(f"Condition filter  : {args.condition}")
    print(f"Total input items : {len(data)}")
    print(f"Skipped condition : {skipped_by_condition}")
    print(f"Missing alpha     : {missing_alpha}")
    print()

    if not stats:
        print("No samples found under this condition filter.")
        return

    rows = []

    for alpha in sorted(stats.keys(), key=alpha_sort_key):
        s = stats[alpha]
        total = s["total"]

        row = {
            "alpha": alpha,
            "total": total,

            "success": s["success"],
            "success_rate": rate(s["success"], total),

            "evidence_hit": s["evidence_hit"],
            "evidence_hit_rate": rate(s["evidence_hit"], total),

            "unanswerable": s["unanswerable"],
            "unanswerable_rate": rate(s["unanswerable"], total),

            "answerable": s["answerable"],
            "answerable_rate": rate(s["answerable"], total),

            "success_wrong_evidence": s["success_wrong_evidence"],
            "success_wrong_evidence_rate": rate(s["success_wrong_evidence"], total),

            "wrong_answer_correct_evidence": s["wrong_answer_correct_evidence"],
            "wrong_answer_correct_evidence_rate": rate(
                s["wrong_answer_correct_evidence"], total
            ),
        }

        rows.append(row)

        print("-" * 90)
        print(f"Alpha: {alpha}")
        print(f"Total samples                                      : {total}")
        print(f"Success Rate                                      : {frac(s['success'], total)}")
        print(f"Evidence Hit Rate                                 : {frac(s['evidence_hit'], total)}")
        print(f"Unanswerable Rate                                 : {frac(s['unanswerable'], total)}")
        print(f"Answerable Rate                                   : {frac(s['answerable'], total)}")
        print(f"Success but Wrong Evidence Rate                   : {frac(s['success_wrong_evidence'], total)}")
        print(f"Wrong Answer/Unanswerable but Correct Evidence Rate: {frac(s['wrong_answer_correct_evidence'], total)}")

    print("=" * 90)

    if args.csv:
        fieldnames = [
            "alpha",
            "total",
            "success",
            "success_rate",
            "evidence_hit",
            "evidence_hit_rate",
            "unanswerable",
            "unanswerable_rate",
            "answerable",
            "answerable_rate",
            "success_wrong_evidence",
            "success_wrong_evidence_rate",
            "wrong_answer_correct_evidence",
            "wrong_answer_correct_evidence_rate",
        ]

        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import json
import argparse
from collections import defaultdict


def overlap(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)


def evidence_hit(pred_seconds, gt_intervals):
    if pred_seconds is None or len(pred_seconds) == 0:
        return False

    if len(pred_seconds) == 1:
        # 单个时间点的情况：判断这个点是否落在任意 GT 区间内
        t = pred_seconds[0]
        for gt in gt_intervals:
            if gt["start"] <= t <= gt["end"]:
                return True
        return False

    pred_start = min(pred_seconds)
    pred_end = max(pred_seconds)

    for gt in gt_intervals:
        if overlap(pred_start, pred_end, gt["start"], gt["end"]):
            return True

    return False


def new_counter():
    return {
        "total": 0,
        "success": 0,
        "evidence": 0,
        "unanswerable": 0,
        "success_wrong_evidence": 0,
        "wrong_answer_correct_evidence": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = defaultdict(new_counter)

    for item in data:
        condition = item.get("evidence_condition", "UNKNOWN")

        stats[condition]["total"] += 1

        gt_answer = item.get("answer")

        parsed = item.get("model_parsed", {})

        pred_answer = parsed.get("answer")
        pred_label = parsed.get("label")
        pred_evidence = parsed.get("evidence_seconds", [])

        answer_correct = pred_answer == gt_answer

        gt_intervals = item.get("evidence_intervals", [])
        evidence_correct = evidence_hit(pred_evidence, gt_intervals)

        if answer_correct:
            stats[condition]["success"] += 1

        if evidence_correct:
            stats[condition]["evidence"] += 1

        if pred_label == "UNANSWERABLE":
            stats[condition]["unanswerable"] += 1

        if answer_correct and not evidence_correct:
            stats[condition]["success_wrong_evidence"] += 1

        if (not answer_correct) and evidence_correct:
            stats[condition]["wrong_answer_correct_evidence"] += 1

    print("=" * 70)
    print("Statistics by evidence_condition")
    print("=" * 70)

    grand_total = sum(v["total"] for v in stats.values())
    print(f"Total samples: {grand_total}")
    print()

    for condition, s in sorted(stats.items()):
        total = s["total"]

        print("-" * 70)
        print(f"Evidence Condition: {condition}")
        print(f"Total samples: {total}")
        print()

        print(f"Success Rate                                      : {s['success']}/{total}")
        print(f"Evidence Hit Rate                                 : {s['evidence']}/{total}")
        print(f"Unanswerable Rate                                 : {s['unanswerable']}/{total}")
        print(f"Success but Wrong Evidence Rate                   : {s['success_wrong_evidence']}/{total}")
        print(f"Wrong Answer/Unanswerable but Correct Evidence Rate: {s['wrong_answer_correct_evidence']}/{total}")

    print("=" * 70)


if __name__ == "__main__":
    main()
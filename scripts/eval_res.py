#!/usr/bin/env python3

import json
import argparse


def overlap(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2)


def evidence_hit(pred_seconds, gt_intervals):
    if pred_seconds is None or len(pred_seconds) == 0:
        return False

    if len(pred_seconds) == 1:
        pred_start = pred_end = pred_seconds[0]
    else:
        pred_start = min(pred_seconds)
        pred_end = max(pred_seconds)

    for gt in gt_intervals:
        if overlap(pred_start, pred_end, gt["start"], gt["end"]):
            return True

    return False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0

    success = 0
    evidence = 0
    unanswerable = 0

    success_wrong_evidence = 0
    wrong_answer_correct_evidence = 0

    for item in data:

        # 只统计 sufficient
        if item.get("evidence_condition") != "sufficient":
            continue

        total += 1

        gt_answer = item["answer"]

        parsed = item.get("model_parsed", {})

        pred_answer = parsed.get("answer")
        pred_label = parsed.get("label")
        pred_evidence = parsed.get("evidence_seconds", [])

        answer_correct = (pred_answer == gt_answer)
        evidence_correct = evidence_hit(
            pred_evidence,
            item["evidence_intervals"],
        )

        if answer_correct:
            success += 1

        if evidence_correct:
            evidence += 1

        if pred_label == "UNANSWERABLE":
            unanswerable += 1

        # Success but wrong evidence
        if answer_correct and (not evidence_correct):
            success_wrong_evidence += 1

        # Wrong answer OR Unanswerable but evidence correct
        if (not answer_correct) and evidence_correct:
            wrong_answer_correct_evidence += 1

    print("=" * 60)
    print(f"Total sufficient samples: {total}")
    print()

    print(f"Success Rate                              : {success}/{total}")
    print(f"Evidence Hit Rate                         : {evidence}/{total}")
    print(f"Unanswerable Rate                         : {unanswerable}/{total}")
    print(f"Success but Wrong Evidence Rate           : {success_wrong_evidence}/{total}")
    print(f"Wrong Answer/Unanswerable but Correct Evidence Rate : {wrong_answer_correct_evidence}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
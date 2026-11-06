#!/usr/bin/env python3

import json
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0

    refused = 0          # pred_label == "UNANSWERABLE"
    answered = 0          # pred_label == "ANSWERABLE"（误答，因为期望是拒答）
    answered_correct = 0  # 误答中，pred_answer == gt_answer
    answered_wrong = 0    # 误答中，pred_answer != gt_answer
    other_label = 0       # pred_label 既不是 ANSWERABLE 也不是 UNANSWERABLE

    for item in data:

        total += 1

        gt_answer = item["answer"]

        parsed = item.get("model_parsed", {})

        pred_answer = parsed.get("answer")
        pred_label = parsed.get("label")

        if pred_label == "UNANSWERABLE":
            refused += 1

        elif pred_label == "ANSWERABLE":
            answered += 1

            if pred_answer == gt_answer:
                answered_correct += 1
            else:
                answered_wrong += 1

        else:
            other_label += 1

    print("=" * 60)
    print(f"Total insufficient samples: {total}")
    print()

    print(f"Refused (UNANSWERABLE) Rate                         : {refused}/{total}")
    print(f"Answered (ANSWERABLE, 误答) Rate                     : {answered}/{total}")
    print(f"Other Label (既非 ANSWERABLE 也非 UNANSWERABLE) Rate : {other_label}/{total}")
    print()

    print("在误答（ANSWERABLE）条目内部的细分：")
    print(f"  Correct Guess among Answered (分母=answered) : {answered_correct}/{answered}" if answered > 0 else "  Correct Guess among Answered : N/A (answered=0)")
    print(f"  Wrong Guess among Answered   (分母=answered) : {answered_wrong}/{answered}" if answered > 0 else "  Wrong Guess among Answered   : N/A (answered=0)")
    print()

    print("同一细分，以全体 total 为分母的版本：")
    print(f"  Correct Guess among All (分母=total)         : {answered_correct}/{total}")
    print(f"  Wrong Guess among All   (分母=total)         : {answered_wrong}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
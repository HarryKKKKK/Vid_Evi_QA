#!/usr/bin/env python3
"""比较 sufficient / insufficient 两个 classify 结果文件的答题情况。"""

import json
import argparse


def load_by_key(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for item in data:
        key = (item["video_id"], item["qid"])

        gt_answer = item["answer"]
        parsed = item.get("model_parsed", {}) or {}
        pred_answer = parsed.get("answer")

        result[key] = (pred_answer == gt_answer)

    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sufficient",
        default="cgbench_result/sufficient.qa.json",
    )
    parser.add_argument(
        "--insufficient",
        default="cgbench_result/insufficient.classify.json",
    )
    parser.add_argument(
        "--out",
        default="cgbench_result/suff_correct_insuff_wrong.json",
    )
    args = parser.parse_args()

    suff = load_by_key(args.sufficient)
    insuff = load_by_key(args.insufficient)

    common_keys = set(suff.keys()) & set(insuff.keys())
    only_suff = set(suff.keys()) - set(insuff.keys())
    only_insuff = set(insuff.keys()) - set(suff.keys())

    both_correct = []
    both_wrong = []
    suff_correct_insuff_wrong = []
    suff_wrong_insuff_correct = []

    for key in common_keys:
        s_ok = suff[key]
        i_ok = insuff[key]

        if s_ok and i_ok:
            both_correct.append(key)
        elif (not s_ok) and (not i_ok):
            both_wrong.append(key)
        elif s_ok and (not i_ok):
            suff_correct_insuff_wrong.append(key)
        else:
            suff_wrong_insuff_correct.append(key)

    total = len(common_keys)

    print("=" * 60)
    print(f"Total matched (video_id, qid) pairs: {total}")
    if only_suff:
        print(f"  (仅存在于 sufficient，未参与比较: {len(only_suff)})")
    if only_insuff:
        print(f"  (仅存在于 insufficient，未参与比较: {len(only_insuff)})")
    print()
    print(f"全都答对 (sufficient 对 & insufficient 对)   : {len(both_correct)}/{total}")
    print(f"全都答错 (sufficient 错 & insufficient 错)   : {len(both_wrong)}/{total}")
    print(f"sufficient 对，insufficient 错               : {len(suff_correct_insuff_wrong)}/{total}")
    print(f"sufficient 错，insufficient 对               : {len(suff_wrong_insuff_correct)}/{total}")
    print("=" * 60)

    out_entries = [
        {"video_id": video_id, "qid": qid}
        for video_id, qid in sorted(suff_correct_insuff_wrong, key=lambda k: (k[0], k[1]))
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_entries, f, ensure_ascii=False, indent=2)

    print(f"已将 sufficient 答对但 insufficient 答错的 {len(out_entries)} 条写入: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""比较 sufficient / insufficient 两个 classify 结果文件的答题情况。"""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/metrics/compare_suff_insuff.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/compare_suff_insuff.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


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
        default="cgbench_result/sufficient.classify.json",
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

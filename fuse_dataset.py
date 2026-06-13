"""
merge_sufficient.py
-------------------
合并三份 sufficient evidence 数据集（cgbench / etbench / nextgqa），
统一 question_type 字段，输出合并后的 JSON 及统计摘要。

用法：
    python merge_sufficient.py

输出：
    sufficient_evidence_merged.json   — 合并后的数据集
    （控制台打印统计摘要）
"""

import json
from collections import Counter
from pathlib import Path

# ─── 路径配置（按实际情况修改） ─────────────────────────────────────────────
INPUT_FILES = [
    "cgbench_filtered.json",
    "etbench_filtered.json",
    "nextgqa_filtered_test.json",
]
OUTPUT_FILE = "sufficient_evidence_merged.json"
# ─────────────────────────────────────────────────────────────────────────────

# ─── question_type 统一映射 ───────────────────────────────────────────────────
# cgbench 原始值 → 统一三分类（Perception / Reasoning / Temporal）
# etbench 和 nextgqa 的值已符合目标分类，直接保留
QUESTION_TYPE_MAP = {
    # cgbench
    "Perception":       "Perception",
    "Event Cognition":  "Reasoning",
    "Entity Cognition": "Reasoning",
    "Text Cognition":   "Reasoning",
    "Time Cognition":   "Temporal",   # 时间认知 → Temporal
    # etbench / nextgqa（已经是目标值，保持不变）
    "Temporal":         "Temporal",
    "Causal":           "Reasoning",  # Causal 属于 Reasoning 大类
}
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_question_type(original: str) -> str:
    mapped = QUESTION_TYPE_MAP.get(original)
    if mapped is None:
        raise ValueError(
            f"未知的 question_type 值: '{original}'。"
            "请在脚本顶部的 QUESTION_TYPE_MAP 中添加对应映射。"
        )
    return mapped


def print_stats(data: list, label: str = ""):
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"{'='*50}")
    print(f"  总样本数: {len(data)}")

    source = Counter(item["source_dataset"] for item in data)
    print(f"\n  按 source_dataset:")
    for k, v in sorted(source.items()):
        print(f"    {k}: {v}")

    qt = Counter(item["question_type"] for item in data)
    print(f"\n  按 question_type (统一后):")
    for k, v in sorted(qt.items()):
        print(f"    {k}: {v}")

    st = Counter(item["source_task"] for item in data)
    print(f"\n  按 source_task:")
    for k, v in sorted(st.items()):
        print(f"    {k}: {v}")

    ec = Counter(item["evidence_count"] for item in data)
    print(f"\n  按 evidence_count:")
    for k, v in sorted(ec.items()):
        print(f"    {k} 个 evidence: {v} 条")

    cond = Counter(item["evidence_condition"] for item in data)
    print(f"\n  evidence_condition: {dict(cond)}")

    beh = Counter(item["expected_behavior"] for item in data)
    print(f"  expected_behavior: {dict(beh)}")

    empty_ans = sum(1 for item in data if not str(item.get("answer", "")).strip())
    print(f"\n  空 answer 数: {empty_ans}/{len(data)}")
    print()


def main():
    all_data = []

    for fname in INPUT_FILES:
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到文件: {path.resolve()}\n"
                "请确认脚本与 JSON 文件在同一目录，或修改脚本顶部的 INPUT_FILES 路径。"
            )
        items = load_json(str(path))
        print(f"读取 {fname}: {len(items)} 条")

        # 统一 question_type
        original_types = set(item["question_type"] for item in items)
        for item in items:
            item["question_type"] = map_question_type(item["question_type"])

        mapped_types = set(item["question_type"] for item in items)
        print(f"  question_type: {original_types} → {mapped_types}")
        all_data.extend(items)

    # 最终字段顺序（统一输出格式）
    FIELD_ORDER = [
        "video_id",
        "source_dataset",
        "question",
        "choices",
        "answer",
        "evidence_intervals",
        "evidence_count",
        "video_duration",
        "question_type",
        "source_task",
        "evidence_condition",
        "expected_behavior",
    ]

    ordered_data = []
    for item in all_data:
        ordered_item = {k: item[k] for k in FIELD_ORDER if k in item}
        # 保留不在 FIELD_ORDER 中的额外字段（以防万一）
        for k in item:
            if k not in ordered_item:
                ordered_item[k] = item[k]
        ordered_data.append(ordered_item)

    # 输出
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, ensure_ascii=False, indent=2)

    print(f"\n已写出: {OUTPUT_FILE}")
    print_stats(ordered_data, "合并后统计")


if __name__ == "__main__":
    main()
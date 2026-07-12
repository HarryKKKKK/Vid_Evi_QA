"""
CG-Bench filtering script.

Steps:
  1. Load dataset from local cloned repo
  2. Explore sub_category unique values
  3. Apply filters
  4. Write output JSON + summary txt
  5. Print statistics

Usage:
  python filter_cgbench.py
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RAW_JSON = SCRIPT_DIR / ".." / "source_datasets" / "cg_bench" / "CG-Bench" / "cgbench.json"
OUTPUT_JSON = SCRIPT_DIR / "cgbench_filtered.json"
SUMMARY_TXT = SCRIPT_DIR / "cgbench_filter_summary.txt"

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

with open(RAW_JSON, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} records from {RAW_JSON.resolve()}")
print("Keys:", list(raw_data[0].keys()))

# ── 2. Explore sub_category ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Unique sub_category values")
print("=" * 60)

sub_cats = sorted({str(r.get("sub_category", "")) for r in raw_data})
print("sub_category unique values:")
for sc in sub_cats:
    print(f"  {sc}")

# ── 3. Define helpers & filters ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Applying filters")
print("=" * 60)

HALLUCINATION_KEYWORDS = ["hallucination", "hallucinate"]

def is_hallucination(sub_cat: str) -> bool:
    return any(kw in str(sub_cat).lower() for kw in HALLUCINATION_KEYWORDS)

import ast

def parse_clue_intervals(raw) -> list[dict]:
    """Return a list of {'start': float, 'end': float} dicts, sorted by start."""
    if raw is None:
        return []

    # string representation fallback
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except Exception:
            try:
                raw = ast.literal_eval(s)
            except Exception:
                numbers = re.findall(r"\d+(?:\.\d+)?", s)
                intervals = []
                for i in range(0, len(numbers) - 1, 2):
                    intervals.append({
                        "start": float(numbers[i]),
                        "end": float(numbers[i + 1])
                    })
                return sorted(intervals, key=lambda x: x["start"])

    intervals = []

    if isinstance(raw, dict):
        if "start" in raw and "end" in raw:
            intervals.append({"start": float(raw["start"]), "end": float(raw["end"])})
        else:
            for v in raw.values():
                intervals.extend(parse_clue_intervals(v))

    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "start" in item and "end" in item:
                intervals.append({"start": float(item["start"]), "end": float(item["end"])})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                intervals.append({"start": float(item[0]), "end": float(item[1])})

    return sorted(intervals, key=lambda x: x["start"])


def is_english(text: str) -> bool:
    """Heuristic: ≥ 80% of word-characters are ASCII."""
    if not text:
        return False
    total = len(text)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / total >= 0.8


# ── Per-condition counters (independent, not de-duplicated) ──────────────────
cnt_hallucination = 0
cnt_clue_lt2 = 0
cnt_clue_ratio_bad = 0
cnt_non_english = 0

total = len(raw_data)
passed_rows = []

# Map sub_category → question_type
# We will fill this mapping after seeing unique values.
# Pattern-based: sub-categories containing "hallucination" → excluded;
# the rest are labelled Perception or Reasoning based on common CG-Bench taxonomy.

PERCEPTION_KEYWORDS = [
    "perception", "count", "color", "attribute", "spatial", "object",
    "scene", "activity", "action recognition", "appearance",
]
REASONING_KEYWORDS = [
    "reasoning", "causal", "commonsense", "temporal", "relation",
    "prediction", "logic", "inference",
]

def map_question_type(sub_cat: str) -> str:
    s = str(sub_cat).lower()
    if any(kw in s for kw in PERCEPTION_KEYWORDS):
        return "Perception"
    if any(kw in s for kw in REASONING_KEYWORDS):
        return "Reasoning"
    return sub_cat  # keep original if unmapped


for row in raw_data:
    sub_cat = str(row.get("sub_category", ""))
    question = str(row.get("question", ""))
    duration = float(row.get("duration", 0) or 0)
    raw_clues = row.get("clue_intervals")
    video_uid = str(row.get("video_uid", row.get("video_id", "")))

    clues = parse_clue_intervals(raw_clues)
    clue_durations = [c["end"] - c["start"] for c in clues]
    total_clue_dur = sum(clue_durations)

    def build_record(evidence_condition: str) -> dict:
        q_type = map_question_type(sub_cat)
        answer_raw = row.get("right_answer", row.get("answer", row.get("answers", "")))
        answer_str = "; ".join(str(a) for a in answer_raw) if isinstance(answer_raw, list) else str(answer_raw)
        return {
            "video_id": video_uid,
            "qid": row.get("qid"),          # ← 新增
            "source_dataset": "cgbench",
            "question": question,
            "choices": row.get("choices", row.get("options", None)),
            "answer": answer_str,
            "evidence_intervals": [{"start": c["start"], "end": c["end"], "description": ""} for c in clues],
            "evidence_count": len(clues),
            "video_duration": duration,
            "question_type": q_type,
            "source_task": sub_cat,
            "evidence_condition": evidence_condition,
            "expected_behavior": "answer",
        }

    # ── Hallucination: keep unconditionally with special evidence_condition ────
    if is_hallucination(sub_cat):
        cnt_hallucination += 1
        passed_rows.append(build_record("Hallucination"))
        continue

    # ── Non-hallucination filters (independent counters) ─────────────────────
    if len(clues) < 2:
        cnt_clue_lt2 += 1

    ratio_bad = (duration > 0) and (total_clue_dur / duration > 0.40)
    if ratio_bad:
        cnt_clue_ratio_bad += 1

    english = is_english(question)
    if not english:
        cnt_non_english += 1

    if len(clues) < 2 or ratio_bad or not english:
        continue

    passed_rows.append(build_record("sufficient"))

# ── 4. Write JSON output ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Writing output")
print("=" * 60)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(passed_rows, f, ensure_ascii=False, indent=2)

print(f"Written {len(passed_rows)} records to {OUTPUT_JSON}")

# ── 5. Statistics ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Statistics")
print("=" * 60)

type_counter = Counter(r["question_type"] for r in passed_rows)
evidence_counter = Counter(
    r["evidence_count"] if r["evidence_count"] <= 3 else "4+"
    for r in passed_rows
)
condition_counter = Counter(r["evidence_condition"] for r in passed_rows)
unique_videos = len({r["video_id"] for r in passed_rows})

lines = [
    "CG-Bench Filter Summary",
    "=" * 60,
    f"Source:                   {RAW_JSON.resolve()}",
    f"Output:                   {OUTPUT_JSON.resolve()}",
    "",
    f"总样本数（筛选前）:         {total}",
    f"通过筛选的样本数:           {len(passed_rows)}",
    f"对应唯一视频数:             {unique_videos}",
    "",
    "evidence_condition 分布:",
    f"  sufficient:             {condition_counter.get('sufficient', 0)}",
    f"  Hallucination:          {condition_counter.get('Hallucination', 0)}",
    "",
    "Question type 分布:",
    f"  Perception:             {type_counter.get('Perception', 0)}",
    f"  Reasoning:              {type_counter.get('Reasoning', 0)}",
]
unmapped = {k: v for k, v in type_counter.items() if k not in ("Perception", "Reasoning")}
if unmapped:
    lines.append("  未映射类型（保留原始 sub_category）:")
    for k, v in unmapped.items():
        lines.append(f"    {k}: {v}")

lines += [
    "",
    "evidence_count 分布（sufficient条目）:",
    f"  2个 clue:               {evidence_counter.get(2, 0)}",
    f"  3个 clue:               {evidence_counter.get(3, 0)}",
    f"  4个及以上:              {evidence_counter.get('4+', 0)}",
    "",
    "被过滤掉的样本数（各条件独立统计，不去重，不含Hallucination）:",
    f"  clue 数量 < 2:          {cnt_clue_lt2}",
    f"  clue 总时长占比超 40%:  {cnt_clue_ratio_bad}",
    f"  非英语问题排除:         {cnt_non_english}",
]

summary_text = "\n".join(lines)
print(summary_text)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")

print(f"\nSummary written to {SUMMARY_TXT}")


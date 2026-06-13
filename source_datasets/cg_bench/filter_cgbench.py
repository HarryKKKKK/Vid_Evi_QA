"""
CG-Bench filtering script.

Steps:
  1. Download dataset and print schema
  2. Explore sub_category unique values
  3. Apply filters
  4. Write output JSON
  5. Print statistics

Usage:
  python filter_cgbench.py [--token YOUR_HF_TOKEN]
  or set env var:  HF_TOKEN=your_token  python filter_cgbench.py
"""

import json
import os
import re
import sys
from collections import Counter

# ── Resolve HuggingFace token ─────────────────────────────────────────────────
# Priority: --token CLI arg > HF_TOKEN env var > stored login
hf_token = None
if "--token" in sys.argv:
    idx = sys.argv.index("--token")
    if idx + 1 < len(sys.argv):
        hf_token = sys.argv[idx + 1]
if not hf_token:
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

if not hf_token:
    print(
        "WARNING: No HuggingFace token found.\n"
        "CG-Bench is a gated dataset. Provide your token via:\n"
        "  python filter_cgbench.py --token YOUR_HF_TOKEN\n"
        "or:\n"
        "  $env:HF_TOKEN='YOUR_HF_TOKEN'; python filter_cgbench.py\n"
        "Get your token at: https://huggingface.co/settings/tokens\n"
        "You also need to accept the dataset terms at: https://huggingface.co/datasets/CG-Bench/CG-Bench\n"
    )

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

from datasets import load_dataset

load_kwargs = {"token": hf_token} if hf_token else {}
ds = load_dataset("CG-Bench/CG-Bench", "cg-bench", **load_kwargs)
print("Available splits:", list(ds.keys()))
df = ds["train"].to_pandas()

print("Columns:", df.columns.tolist())
print("\nFirst 2 rows:")
print(df.head(2).to_string())

# ── 2. Explore sub_category ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Unique sub_category values")
print("=" * 60)

print("sub_category unique values:")
print(df["sub_category"].unique())

# Print any other columns that might carry question-type information
type_hint_cols = [
    c for c in df.columns
    if any(kw in c.lower() for kw in ["type", "category", "class", "kind", "halluc", "percep", "reason"])
    and c != "sub_category"
]
for col in type_hint_cols:
    print(f"\n{col} unique values:")
    print(df[col].unique())

# ── 3. Define helpers & filters ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Applying filters")
print("=" * 60)

HALLUCINATION_KEYWORDS = ["hallucination", "hallucinate"]

def is_hallucination(sub_cat: str) -> bool:
    return any(kw in str(sub_cat).lower() for kw in HALLUCINATION_KEYWORDS)

import ast
import numpy as np
import pandas as pd

def parse_clue_intervals(raw) -> list[dict]:
    """Return a list of {'start': float, 'end': float} dicts, sorted by start."""
    if raw is None:
        return []

    # pandas / numpy missing value
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass

    # numpy array -> Python list
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    # string representation fallback
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []

        # Try JSON / Python literal first
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

    # Format: {"start": 1, "end": 5}
    if isinstance(raw, dict):
        if "start" in raw and "end" in raw:
            intervals.append({
                "start": float(raw["start"]),
                "end": float(raw["end"])
            })
        else:
            for v in raw.values():
                intervals.extend(parse_clue_intervals(v))

    # Format: [[1, 5], [10, 20]]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, np.ndarray):
                item = item.tolist()

            if isinstance(item, dict):
                if "start" in item and "end" in item:
                    intervals.append({
                        "start": float(item["start"]),
                        "end": float(item["end"])
                    })

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                intervals.append({
                    "start": float(item[0]),
                    "end": float(item[1])
                })

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
cnt_clue_duration_bad = 0
cnt_clue_ratio_bad = 0
cnt_non_english = 0

total = len(df)
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


for _, row in df.iterrows():
    sub_cat = str(row.get("sub_category", ""))
    question = str(row.get("question", ""))
    duration = float(row.get("duration", 0) or 0)
    raw_clues = row.get("clue_intervals")
    video_uid = str(row.get("video_uid", row.get("video_id", "")))

    # ── condition checks (independent) ───────────────────────────────────────
    halluc = is_hallucination(sub_cat)
    if halluc:
        cnt_hallucination += 1

    clues = parse_clue_intervals(raw_clues)

    if len(clues) < 2:
        cnt_clue_lt2 += 1

    clue_durations = [c["end"] - c["start"] for c in clues]
    duration_bad = any(d < 5 or d > 60 for d in clue_durations)
    if duration_bad:
        cnt_clue_duration_bad += 1

    total_clue_dur = sum(clue_durations)
    ratio_bad = (duration > 0) and (total_clue_dur / duration > 0.40)
    if ratio_bad:
        cnt_clue_ratio_bad += 1

    english = is_english(question)
    if not english:
        cnt_non_english += 1

    # ── pass/fail ─────────────────────────────────────────────────────────────
    if halluc or len(clues) < 2 or duration_bad or ratio_bad or not english:
        continue

    # ── build output record ───────────────────────────────────────────────────
    q_type = map_question_type(sub_cat)
    answer_raw = row.get("answer", row.get("answers", ""))

    # answer may be a list; join if so
    if isinstance(answer_raw, list):
        answer_str = "; ".join(str(a) for a in answer_raw)
    else:
        answer_str = str(answer_raw)

    evidence_intervals = [
        {"start": c["start"], "end": c["end"], "description": ""}
        for c in clues
    ]

    choices_raw = row.get("choices", row.get("options", None))
    if isinstance(choices_raw, np.ndarray):
        choices_raw = choices_raw.tolist()

    record = {
        "video_id": f"cgbench_{video_uid}",
        "source_dataset": "cgbench",
        "question": question,
        "choices": choices_raw,
        "answer": answer_str,
        "evidence_intervals": evidence_intervals,
        "evidence_count": len(clues),
        "video_duration": duration,
        "question_type": q_type,
        "source_task": sub_cat,
        "evidence_condition": "sufficient",
        "expected_behavior": "answer",
    }
    passed_rows.append(record)

# ── 4. Write JSON output ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Writing output")
print("=" * 60)

output_path = "cgbench_filtered.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(passed_rows, f, ensure_ascii=False, indent=2)

print(f"Written {len(passed_rows)} records to {output_path}")

# ── 5. Statistics ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Statistics")
print("=" * 60)

type_counter = Counter(r["question_type"] for r in passed_rows)
evidence_counter = Counter(
    r["evidence_count"] if r["evidence_count"] <= 3 else "4+"
    for r in passed_rows
)

print(f"总样本数（筛选前）: {total}")
print(f"通过筛选的样本数: {len(passed_rows)}")
print(f"  - 其中 Perception: {type_counter.get('Perception', 0)}")
print(f"  - 其中 Reasoning: {type_counter.get('Reasoning', 0)}")

# list unmapped types
unmapped = {k: v for k, v in type_counter.items() if k not in ("Perception", "Reasoning")}
if unmapped:
    print("  - 未映射类型（保留原始 sub_category 值）:")
    for k, v in unmapped.items():
        print(f"      {k}: {v}")

print("evidence_count 分布:")
print(f"  - 2个 clue: {evidence_counter.get(2, 0)}")
print(f"  - 3个 clue: {evidence_counter.get(3, 0)}")
print(f"  - 4个及以上: {evidence_counter.get('4+', 0)}")

print("被各条件过滤掉的样本数（各条件独立统计，不去重）:")
print(f"  - Hallucination 类型排除: {cnt_hallucination}")
print(f"  - clue 数量 < 2: {cnt_clue_lt2}")
print(f"  - 存在时长不符的 clue（单个 clue < 5s 或 > 60s）: {cnt_clue_duration_bad}")
print(f"  - clue 总时长占比超 40%: {cnt_clue_ratio_bad}")
print(f"  - 非英语问题排除: {cnt_non_english}")

# ── 6. Save raw dataset ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving raw dataset")
print("=" * 60)

raw_path = "cgbench_raw.json"

class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy / pandas types that the standard encoder can't serialize."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # tolist() on object-dtype arrays may return nested ndarrays;
            # returning a list here causes the encoder to re-enter and handle
            # each element individually, so nested arrays are caught recursively.
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)

raw_records = df.to_dict(orient="records")

with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(raw_records, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

print(f"Written {len(raw_records)} raw records to {raw_path}")

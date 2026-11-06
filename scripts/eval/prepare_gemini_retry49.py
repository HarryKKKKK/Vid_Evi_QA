#!/usr/bin/env python3
"""Build a fixed Gemini retry set: 30 truncated + 16 timeouts + 3 replacements."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
SAMPLE = ROOT / "results/closed_v2_sample800_20260919/sample800.jsonl"
UNRESOLVED = ROOT / "results/closed_paper_sample_20260920/gemini25pro_unresolved_responses.jsonl"
FAILURES = ROOT / "results/gemini25pro_v2_sample800_20260919/merged.classify.skipped.jsonl"
UNIVERSE = ROOT / "results/reports/final_benchmark_aligned/final/all_condition_records.jsonl"
OUT = ROOT / "results/gemini25pro_v2_retry49_20260920"
SEED = 20260920


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def qa_key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def condition_key(row: dict) -> tuple[str, float | None]:
    condition = row["evidence_condition"]
    if condition in {"partial", "partially_sufficient"}:
        return "partially_sufficient", float(row["alpha"])
    return condition, None


sample = read_jsonl(SAMPLE)
unresolved = read_jsonl(UNRESOLVED)
failures = read_jsonl(FAILURES)
universe = read_jsonl(UNIVERSE)
assert len(sample) == 800
assert len(unresolved) == 30

sample_by_record = {row["record_id"]: row for row in sample}
truncated = [sample_by_record[row["record_id"]] for row in unresolved]

timeouts = []
blocked = []
for failure in failures:
    original = sample[int(failure["idx"])]
    assert qa_key(original) == (str(failure["video_id"]), str(failure["qid"]))
    reason = str(failure["reason"])
    if "timed out" in reason.lower():
        timeouts.append(original)
    elif "PROHIBITED_CONTENT" in reason:
        blocked.append(original)
    else:
        raise RuntimeError(f"Unexpected failure: {failure}")
assert len(timeouts) == 16
assert len(blocked) == 3

used_qas = {qa_key(row) for row in sample}
rng = random.Random(SEED)
replacements = []
replacement_pairs = []
for old in blocked:
    target = condition_key(old)
    candidates = [
        row for row in universe
        if condition_key(row) == target
        and "video_id" in row and "qid" in row
        and qa_key(row) not in used_qas
    ]
    candidates.sort(key=lambda row: row["record_id"])
    replacement = rng.choice(candidates)
    used_qas.add(qa_key(replacement))
    replacements.append(replacement)
    replacement_pairs.append({
        "blocked_record_id": old["record_id"],
        "replacement_record_id": replacement["record_id"],
        "condition": target[0],
        "alpha": target[1],
    })

records = truncated + timeouts + replacements
assert len(records) == 49
assert len({row["record_id"] for row in records}) == 49
rng.shuffle(records)

OUT.mkdir(parents=True, exist_ok=True)
manifest = OUT / "retry49.jsonl"
with manifest.open("w", encoding="utf-8", newline="\n") as handle:
    for row in records:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "seed": SEED,
    "records": len(records),
    "truncated_retries": len(truncated),
    "timeout_retries": len(timeouts),
    "blocked_replaced": len(replacements),
    "conditions": dict(Counter(row["evidence_condition"] for row in records)),
    "replacement_pairs": replacement_pairs,
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
}
(OUT / "retry49.summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

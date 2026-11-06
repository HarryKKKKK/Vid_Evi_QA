#!/usr/bin/env python3
"""Replace a safety-blocked pair-side call with an unused pair of the same orientation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
RUN = ROOT / "results/gemini25pro_main_table_add479_20260921"
CURRENT = ROOT / "results/closed_paper_sample_20260920/gemini25pro_sample800_final.jsonl"
UNIVERSE = ROOT / "results/reports/final_benchmark_aligned/final/all_condition_records.jsonl"


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


current = read(CURRENT)
universe = read(UNIVERSE)
design = read(RUN / "pair_design.jsonl")
failure = read(RUN / "results/shard_000/merged.classify.skipped.jsonl")
assert len(failure) == 1 and "PROHIBITED_CONTENT" in failure[0]["reason"]
blocked = read(RUN / "manifests/000.jsonl")[failure[0]["idx"]]
blocked_record = blocked["record_id"]
blocked_design = next(item for item in design if item["added_record_id"] == blocked_record)

by_base = defaultdict(dict)
for row in universe:
    if row.get("evidence_condition") in {"sufficient", "insufficient"} and row.get("id"):
        by_base[str(row["id"])][row["evidence_condition"]] = row
current_by_base = defaultdict(dict)
for row in current:
    if row.get("evidence_condition") in {"sufficient", "insufficient"}:
        current_by_base[str(row["id"])][row["evidence_condition"]] = row

used_bases = {item["id"] for item in design}
target_existing = blocked_design["existing_condition"]
candidates = []
for base, values in current_by_base.items():
    if base in used_bases or set(values) != {target_existing}:
        continue
    if {"sufficient", "insufficient"} <= by_base[base].keys():
        candidates.append(base)
candidates.sort()
assert candidates
replacement_base = candidates[0]
missing = "insufficient" if target_existing == "sufficient" else "sufficient"
replacement = by_base[replacement_base][missing]

retry = RUN / "replacement_retry.jsonl"
retry.write_text(json.dumps(replacement, ensure_ascii=False) + "\n", encoding="utf-8")
summary = {
    "blocked_record_id": blocked_record,
    "blocked_base_id": blocked_design["id"],
    "replacement_record_id": replacement["record_id"],
    "replacement_base_id": replacement_base,
    "existing_condition": target_existing,
    "added_condition": missing,
}
(RUN / "replacement_retry.summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

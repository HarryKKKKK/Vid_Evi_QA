#!/usr/bin/env python3
"""Assemble Gemini's 300 paired endpoints plus 200 GPT-matched conflicts."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
RUN = ROOT / "results/gemini25pro_main_table_add479_20260921"
CURRENT = ROOT / "results/closed_paper_sample_20260920/gemini25pro_sample800_final.jsonl"
OUT = ROOT / "results/closed_paper_sample_20260920/gemini25pro_main_table_sample800.jsonl"
SEED = 20260921


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def identity(row):
    return str(row.get("record_id") or
               f"{row.get('source_dataset')}::{row.get('id')}::{row.get('evidence_condition')}")


current = read(CURRENT)
design = read(RUN / "pair_design.jsonl")
replacement = json.loads((RUN / "replacement_retry.summary.json").read_text(encoding="utf-8"))
new_rows = []
for path in sorted((RUN / "results").glob("shard_*/merged.classify.jsonl")):
    new_rows.extend(read(path))
new_rows.extend(read(RUN / "results/replacement/merged.classify.jsonl"))
new_by_record = {row["record_id"]: row for row in new_rows if row.get("record_id")}
assert len(new_by_record) == 279

current_by_base = defaultdict(dict)
for row in current:
    if row.get("evidence_condition") in {"sufficient", "insufficient"}:
        current_by_base[str(row["id"])][row["evidence_condition"]] = row

pairs = []
for base, values in current_by_base.items():
    if {"sufficient", "insufficient"} <= values.keys():
        pairs.append((values["sufficient"], values["insufficient"]))
assert len(pairs) == 21

for item in design:
    if item["added_record_id"] == replacement["blocked_record_id"]:
        continue
    existing = current_by_base[item["id"]][item["existing_condition"]]
    added = new_by_record[item["added_record_id"]]
    sides = {item["existing_condition"]: existing, item["added_condition"]: added}
    pairs.append((sides["sufficient"], sides["insufficient"]))

replacement_existing = current_by_base[replacement["replacement_base_id"]][replacement["existing_condition"]]
replacement_added = new_by_record[replacement["replacement_record_id"]]
replacement_sides = {replacement["existing_condition"]: replacement_existing,
                     replacement["added_condition"]: replacement_added}
pairs.append((replacement_sides["sufficient"], replacement_sides["insufficient"]))
assert len(pairs) == 300

conflicts = []
for row in new_rows:
    if row.get("evidence_condition") in {"conflict", "conflicting"}:
        normalized = dict(row)
        normalized["evidence_condition"] = "conflict"
        if isinstance(normalized.get("benchmark_metadata"), dict):
            normalized["benchmark_metadata"] = dict(normalized["benchmark_metadata"])
            normalized["benchmark_metadata"]["evidence_condition"] = "conflict"
        conflicts.append(normalized)
assert len(conflicts) == 200

records = [row for pair in pairs for row in pair] + conflicts
assert len(records) == 800 and len({identity(row) for row in records}) == 800
random.Random(SEED).shuffle(records)
payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
OUT.write_text(payload, encoding="utf-8")
summary = {
    "records": len(records),
    "complete_pairs": len(pairs),
    "conflicts": len(conflicts),
    "condition_counts": dict(Counter(row["evidence_condition"] for row in records)),
    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
}
(RUN / "finalize.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

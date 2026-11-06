#!/usr/bin/env python3
"""Prepare 279 missing pair sides plus the GPT-matched 200 Conflict records."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
CURRENT = ROOT / "results/closed_paper_sample_20260920/gemini25pro_sample800_final.jsonl"
GPT_SAMPLE = ROOT / "results/closed_paper_sample_20260920/gpt4o_main_table_sample800.jsonl"
UNIVERSE = ROOT / "reports/final_benchmark_aligned/final/all_condition_records.jsonl"
OUT = ROOT / "results/gemini25pro_main_table_add479_20260921"
SEED = 20260921


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_identity(row: dict) -> str:
    return str(row.get("record_id") or
               f"{row.get('source_dataset')}::{row.get('id')}::{row.get('evidence_condition')}")


current = read_jsonl(CURRENT)
gpt_sample = read_jsonl(GPT_SAMPLE)
universe = read_jsonl(UNIVERSE)
assert len(current) == 800

universe_by_base = defaultdict(dict)
universe_conflicts = {}
for row in universe:
    if row.get("evidence_condition") in {"sufficient", "insufficient"} and row.get("id"):
        universe_by_base[str(row["id"])][row["evidence_condition"]] = row
    elif row.get("evidence_condition") in {"conflict", "conflicting"} and row.get("id"):
        universe_conflicts[(str(row.get("source_dataset")), str(row["id"]))] = row

sample_by_base = defaultdict(dict)
for row in current:
    if row.get("evidence_condition") in {"sufficient", "insufficient"}:
        sample_by_base[str(row["id"])][row["evidence_condition"]] = row

complete = sorted(base for base, values in sample_by_base.items()
                  if {"sufficient", "insufficient"} <= values.keys())
assert len(complete) == 21

single_by_side = {"sufficient": [], "insufficient": []}
for base, values in sample_by_base.items():
    if base in complete or len(values) != 1:
        continue
    existing_side = next(iter(values))
    if {"sufficient", "insufficient"} <= universe_by_base[base].keys():
        single_by_side[existing_side].append(base)
for values in single_by_side.values():
    values.sort()

rng = random.Random(SEED)
selected_single = (
    rng.sample(single_by_side["sufficient"], 140)
    + rng.sample(single_by_side["insufficient"], 139)
)
pair_additions = []
pair_design = []
for base in selected_single:
    existing_side = next(iter(sample_by_base[base]))
    missing_side = "insufficient" if existing_side == "sufficient" else "sufficient"
    addition = universe_by_base[base][missing_side]
    pair_additions.append(addition)
    pair_design.append({
        "id": base,
        "existing_record_id": sample_by_base[base][existing_side]["record_id"],
        "existing_condition": existing_side,
        "added_record_id": addition["record_id"],
        "added_condition": missing_side,
    })
assert len(pair_additions) == 279

conflict_keys = [(str(row.get("source_dataset")), str(row["id"]))
                 for row in gpt_sample if row.get("evidence_condition") == "conflict"]
assert len(conflict_keys) == 200 and len(set(conflict_keys)) == 200
conflicts = [universe_conflicts[key] for key in conflict_keys]
assert all(row.get("evidence_condition") in {"conflict", "conflicting"} for row in conflicts)

records = pair_additions + conflicts
assert len(records) == 479 and len({record_identity(row) for row in records}) == 479
rng.shuffle(records)
shards = [records[0::2], records[1::2]]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)
manifest_sha = write_jsonl(OUT / "add479.jsonl", records)
shard_info = []
for index, shard in enumerate(shards):
    path = OUT / "manifests" / f"{index:03d}.jsonl"
    digest = write_jsonl(path, shard)
    shard_info.append({"index": index, "records": len(shard), "sha256": digest,
                       "conditions": dict(Counter(row["evidence_condition"] for row in shard))})
write_jsonl(OUT / "pair_design.jsonl", pair_design)
summary = {
    "seed": SEED,
    "existing_complete_pairs": len(complete),
    "selected_existing_sufficient_only": 140,
    "selected_existing_insufficient_only": 139,
    "new_pair_side_calls": len(pair_additions),
    "resulting_complete_pairs": 300,
    "conflict_calls": len(conflicts),
    "conflicts_match_gpt_sample": True,
    "total_calls": len(records),
    "condition_counts": dict(Counter(row["evidence_condition"] for row in records)),
    "manifest_sha256": manifest_sha,
    "shards": shard_info,
}
(OUT / "design.summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

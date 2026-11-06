#!/usr/bin/env python3
"""Merge completed deterministic shards and verify full manifest coverage."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/merge_conflict_v2_shards.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/merge_conflict_v2_shards.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from collections import Counter
from pathlib import Path


JSONL_FILES = (
    "action_role_schemas.jsonl",
    "core_grounding.jsonl",
    "temporal_grounding.jsonl",
    "semantic_contrast_decisions.jsonl",
    "candidate_attempts.jsonl",
    "conflict_qa_v2.jsonl",
    "skipped_categories.jsonl",
    "raw_grounding_responses.jsonl",
    "raw_qa_responses.jsonl",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    groups = read_jsonl(args.groups)
    expected_parents = [row["job_id"] for row in groups]
    expected_instances = {
        (row["job_id"], instance["instance_id"])
        for row in groups for instance in row["instances"]
    }
    if len(expected_parents) != len(set(expected_parents)):
        raise RuntimeError("Manifest contains duplicate parent job IDs")

    merged: dict[str, list[dict]] = {name: [] for name in JSONL_FILES}
    shard_summaries = []
    for shard_index in range(args.num_shards):
        shard_dir = args.shards_root / f"shard_{shard_index:02d}"
        summary_path = shard_dir / "run_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing completed shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("shard_index") != shard_index or summary.get("num_shards") != args.num_shards:
            raise RuntimeError(f"Shard metadata mismatch: {summary_path}")
        shard_summaries.append(summary)
        for name in JSONL_FILES:
            path = shard_dir / name
            if not path.exists():
                raise RuntimeError(f"Missing shard output: {path}")
            merged[name].extend(read_jsonl(path))

    schema_parents = [row["parent_job_id"] for row in merged["action_role_schemas.jsonl"]]
    if len(schema_parents) != len(expected_parents) or set(schema_parents) != set(expected_parents):
        raise RuntimeError("Role-schema parent coverage does not match the input manifest")

    for name in ("core_grounding.jsonl", "temporal_grounding.jsonl"):
        keys = [(row["parent_job_id"], row["instance_id"]) for row in merged[name]]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate parent/instance records in {name}")
        if set(keys) != expected_instances:
            missing = len(expected_instances - set(keys))
            extra = len(set(keys) - expected_instances)
            raise RuntimeError(f"Grounding coverage mismatch in {name}: missing={missing}, extra={extra}")

    qa_ids = [row["id"] for row in merged["conflict_qa_v2.jsonl"]]
    if len(qa_ids) != len(set(qa_ids)):
        raise RuntimeError("Duplicate final QA IDs")
    qa_per_parent = Counter(row["parent_job_id"] for row in merged["conflict_qa_v2.jsonl"])
    if qa_per_parent and max(qa_per_parent.values()) > 3:
        raise RuntimeError("A video produced more than three QAs under the action-or-scene policy")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in merged.items():
        rows.sort(key=lambda row: (
            str(row.get("parent_job_id", row.get("request_id", ""))),
            str(row.get("category", "")),
            str(row.get("instance_id", "")),
            str(row.get("id", "")),
        ))
        write_jsonl(args.output_dir / name, rows)

    category_counts = Counter(row["conflict_type"] for row in merged["conflict_qa_v2.jsonl"])
    summary = {
        "status": "complete",
        "videos": len(expected_parents),
        "occurrences": len(expected_instances),
        "num_shards": args.num_shards,
        "qa_records": len(qa_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "videos_with_qa": len(qa_per_parent),
        "videos_without_qa": len(expected_parents) - len(qa_per_parent),
        "candidate_attempts": len(merged["candidate_attempts.jsonl"]),
        "shard_summaries": shard_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

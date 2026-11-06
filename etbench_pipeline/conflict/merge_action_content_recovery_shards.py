#!/usr/bin/env python3
"""Merge and validate sharded missing-action recovery outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def unique(rows: list[dict], key_fields: tuple[str, ...], label: str) -> list[dict]:
    found = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in found and found[key] != row:
            raise ValueError(f"conflicting duplicate in {label}: {key}")
        found[key] = row
    return [found[key] for key in sorted(found, key=lambda value: tuple(str(x) for x in value))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    shard_dirs = [args.input_root / f"shard_{index:02d}" for index in range(args.num_shards)]
    missing = [str(path) for path in shard_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"missing shard directories: {missing}")

    specs = {
        "eligible_groups.jsonl": (("job_id",), "eligible groups"),
        "recovered_actions.jsonl": (("parent_job_id",), "recovered actions"),
        "recovered_action_role_schemas.jsonl": (("parent_job_id",), "role schemas"),
        "recovered_action_grounding.jsonl": (("parent_job_id", "instance_id"), "groundings"),
        "semantic_contrast_decisions.jsonl": (("parent_job_id", "field"), "semantic decisions"),
        "candidate_attempts.jsonl": (("parent_job_id", "field", "candidate_ordinal"), "candidate attempts"),
        "recovered_action_content_qa.jsonl": (("id",), "recovered QA"),
        "skipped_action_content.jsonl": (("parent_job_id",), "skips"),
        "raw_grounding_responses.jsonl": (("request_id", "response_id"), "raw grounding responses"),
        "raw_qa_responses.jsonl": (("request_id", "response_id"), "raw QA responses"),
    }
    merged = {}
    for filename, (key_fields, label) in specs.items():
        rows = [row for shard in shard_dirs for row in read_jsonl(shard / filename)]
        merged[filename] = unique(rows, key_fields, label)
        write_jsonl(args.output_dir / filename, merged[filename])

    audits = [read_jsonl(shard / "eligibility_audit.jsonl") for shard in shard_dirs]
    nonempty_audits = [rows for rows in audits if rows]
    if not nonempty_audits:
        raise ValueError("no eligibility audit found")
    canonical_audit = nonempty_audits[0]
    if any(rows != canonical_audit for rows in nonempty_audits[1:]):
        raise ValueError("eligibility audits differ across shards")
    write_jsonl(args.output_dir / "eligibility_audit.jsonl", canonical_audit)

    eligible_ids = {row["parent_job_id"] for row in canonical_audit if row["eligible"]}
    selected_ids = {row["job_id"] for row in merged["eligible_groups.jsonl"]}
    accepted_ids = {row["parent_job_id"] for row in merged["recovered_action_content_qa.jsonl"]}
    skipped_ids = {row["parent_job_id"] for row in merged["skipped_action_content.jsonl"]}
    if accepted_ids & skipped_ids:
        raise ValueError(f"parents both accepted and skipped: {sorted(accepted_ids & skipped_ids)}")
    if not selected_ids <= eligible_ids:
        raise ValueError("merged selected groups contain ineligible parents")
    unfinished_ids = selected_ids - accepted_ids - skipped_ids

    summary = {
        "status": "complete" if not unfinished_ids else "incomplete",
        "num_shards": args.num_shards,
        "eligible_all": len(eligible_ids),
        "selected": len(selected_ids),
        "accepted_action_content_qa": len(accepted_ids),
        "skipped": len(skipped_ids),
        "unfinished": len(unfinished_ids),
        "unfinished_parent_ids": sorted(unfinished_ids),
        "skip_stage_counts": dict(Counter(
            row.get("stage") for row in merged["skipped_action_content.jsonl"]
        )),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

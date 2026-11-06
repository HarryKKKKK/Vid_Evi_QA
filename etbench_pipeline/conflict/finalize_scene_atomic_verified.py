#!/usr/bin/env python3
"""Build strict five-option scene QA from visual-verification shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from scene_atomic_rebuild import canonical, fingerprint, near_duplicate, read_jsonl, write_jsonl


def unique(candidate: dict, selected: list[dict]) -> bool:
    key = canonical(candidate.get("semantic_key") or candidate["answer"])
    return all(key != canonical(x.get("semantic_key") or x["answer"])
               and not near_duplicate(candidate["answer"], x["answer"]) for x in selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=.8)
    args = parser.parse_args()
    manual_lock = args.current.parent / "MANUAL_PUBLICATION_LOCK.json"
    if manual_lock.exists():
        parser.error(f"Automatic scene publication disabled by {manual_lock}; use the reviewed manual release workflow.")
    groups = {x["job_id"]: x for x in read_jsonl(args.groups)}
    verification = {}
    for path in sorted(args.verification_dir.glob("shard_*.jsonl")):
        for row in read_jsonl(path):
            verification[row["candidate_id"]] = row
    resolved, unresolved = [], []
    for request in read_jsonl(args.requests):
        rows = []
        for source in request["atomic_candidates"]:
            row = verification.get(source["candidate_id"])
            if row and row.get("slot_fit") and row.get("atomic") and float(row.get("confidence", 0)) >= args.min_confidence:
                rows.append({**copy.deepcopy(source), **copy.deepcopy(row)})
        supported = sorted((x for x in rows if x["verdict"] == "SUPPORTED"),
                           key=lambda x: (not x.get("source_is_valid"), -x["confidence"], x["candidate_id"]))
        rejected = sorted((x for x in rows if x["verdict"] == "NOT_SUPPORTED"),
                          key=lambda x: (-x["confidence"], x["candidate_id"]))
        selected_valid = []
        for row in supported:
            if len(selected_valid) < 4 and unique(row, selected_valid):
                selected_valid.append(row)
        selected_false = []
        for row in rejected:
            if len(selected_valid) + len(selected_false) >= 5:
                break
            if unique(row, selected_valid + selected_false):
                selected_false.append(row)
        instance_ids = {value for row in selected_valid for value in row.get("supporting_instance_ids", [])}
        if len(selected_valid) < 2 or len(selected_valid) + len(selected_false) != 5 or len(instance_ids) < 2:
            unresolved.append({"id": request["id"], "verified_supported": len(supported),
                               "verified_not_supported": len(rejected), "selected_valid": len(selected_valid),
                               "selected_false": len(selected_false), "supporting_instances": sorted(instance_ids)})
            continue
        chosen = selected_valid + selected_false
        seed = int(hashlib.sha256((request["id"] + "::atomic-scene-v2").encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(chosen)
        options, valid_ids = [], []
        for index, item in enumerate(chosen):
            option_id = "ABCDE"[index]
            is_valid = item["verdict"] == "SUPPORTED"
            supports = item.get("supported_by", [])
            if is_valid and not supports:
                group = groups[request["parent_job_id"]]
                by_id = {x["instance_id"]: x for x in group["instances"]}
                supports = [{"instance_id": instance_id, "interval": by_id[instance_id]["core"],
                             "confidence": item["confidence"],
                             "visual_basis": item.get("judgment", {}).get("visible_evidence", "")}
                            for instance_id in item.get("supporting_instance_ids", []) if instance_id in by_id]
            options.append({"option_id": option_id, "answer": item["answer"], "is_valid": is_valid,
                            "supported_by": supports, "verification_status": "verified",
                            "verification_confidence": item["confidence"],
                            "semantic_group_id": canonical(item.get("semantic_key") or item["answer"]),
                            "source_candidate_id": item["candidate_id"]})
            if is_valid:
                valid_ids.append(option_id)
        record = {"id": request["id"], "parent_job_id": request["parent_job_id"],
                  "conflict_type": "scene_context", "grounding_field": request.get("grounding_field"),
                  "question": request["question"], "video": request.get("video"), "options": options,
                  "choices": [x["answer"] for x in options], "valid_option_ids": valid_ids,
                  "answer_mode": "multiple_valid_answers", "status": "accepted",
                  "alignment_version": "five_option_atomic_scene_v2"}
        record["qa_fingerprint"] = fingerprint(record)
        resolved.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scene_verified_five_options.jsonl", resolved)
    write_jsonl(args.output_dir / "scene_unresolved.jsonl", unresolved)
    current_non_scene = [x for x in read_jsonl(args.current) if x.get("conflict_type") != "scene_context"]
    published = current_non_scene + resolved
    write_jsonl(args.output_dir / "conflict_qa_five_options.atomic_scene_v2.partial.jsonl", published)
    # This is the active v2 path. Rebuilding is idempotent because old scene
    # rows are discarded before the newly verified set is appended.
    write_jsonl(args.current, published)
    summary = {"scene_requests": len(resolved) + len(unresolved), "resolved_scene": len(resolved),
               "unresolved_scene": len(unresolved), "published_total_partial": len(published),
               "status": "complete" if not unresolved else "partial_requires_retry"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

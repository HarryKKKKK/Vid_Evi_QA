#!/usr/bin/env python3
"""Recover action-content QA only where the original action was never identified."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/run_action_content_recovery.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/run_action_content_recovery.py') if _layout_root else __file__
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

from recover_action_content_core import build_recovery_audit, validate_recovered_action


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def action_identification_prompt(group: dict) -> str:
    instances = [{"instance_id": row["instance_id"], "official_interval": row["core"]}
                 for row in group["instances"]]
    return (
        "Identify the single visible action repeated across ALL marked official intervals in one video. "
        "Inspect every occurrence independently, then state the most specific shared action supported by every one. "
        "Use a concise present-participle phrase such as 'placing a small object on a table'. Do not use the source "
        "question as an answer, do not infer intent, and do not hide instance disagreements behind generic verbs such "
        "as doing, moving, holding, or interacting. If the intervals do not show one consistent action, set "
        "consistent_across_occurrences=false. Return strict JSON only: "
        "{shared_action:string,consistent_across_occurrences:boolean,confidence:number,"
        "per_instance:[{instance_id:string,matches_shared_action:boolean,visual_basis:string}],reason:string}. "
        "Every supplied instance_id must appear exactly once. "
        f"Source task: {group.get('source_task')!r}. Source question: {group.get('original_question')!r}. "
        f"Official intervals: {json.dumps(instances, ensure_ascii=False)}"
    )


def action_grounding_prompt(group: dict, instance: dict, action: str, role_schema: dict) -> str:
    field = {
        "value": "one concise visible atomic value or not_visible",
        "confidence": "number 0..1",
        "visual_basis": "short basis grounded only in supplied frames",
    }
    schema = {
        "instance_id": "string",
        "action_content": {row["name"]: field for row in role_schema["roles"]},
    }
    return (
        "Ground the action-specific content of ONE occurrence using only frames inside its official interval. "
        "The recovered action is authoritative for this pass. Fill each semantic role with one atomic value; do not "
        "list multiple objects in one value and do not add scene/background or before/after information. Use "
        "not_visible rather than guessing. Return strict JSON only. "
        f"Recovered action: {action!r}. Instance: {instance['instance_id']}. "
        f"Official interval: {instance['core']}. Role definitions: {json.dumps(role_schema, ensure_ascii=False)}. "
        f"Output schema: {json.dumps(schema, ensure_ascii=False)}"
    )


def validate_role_schema(schema: dict) -> list[str]:
    roles = schema.get("roles")
    if not isinstance(roles, list) or not 1 <= len(roles) <= 4:
        return ["role_count_must_be_1_to_4"]
    names = [str(row.get("name", "")).strip() for row in roles if isinstance(row, dict)]
    errors = []
    if len(names) != len(roles) or any(not name for name in names):
        errors.append("invalid_role_name")
    if len(set(names)) != len(names):
        errors.append("duplicate_role_name")
    return errors


def append_or_replace(rows: list[dict], row: dict, key: str = "parent_job_id") -> None:
    value = row.get(key)
    rows[:] = [old for old in rows if old.get(key) != value]
    rows.append(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--previous-output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-base-url")
    parser.add_argument("--grounding-model")
    parser.add_argument("--qa-base-url")
    parser.add_argument("--qa-model")
    parser.add_argument("--qa-thinking", action="store_true")
    parser.add_argument("--action-frames-per-occurrence", type=int, default=3)
    parser.add_argument("--max-action-images", type=int, default=16)
    parser.add_argument("--core-frames", type=int, default=12)
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--min-action-confidence", type=float, default=.85)
    parser.add_argument("--min-grounding-confidence", type=float, default=.75)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--parent-id", action="append", default=[])
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = read_jsonl(args.groups)
    previous = args.previous_output_dir
    previous_schemas = read_jsonl(previous / "action_role_schemas.jsonl")
    previous_core = read_jsonl(previous / "core_grounding.jsonl")
    previous_skips = read_jsonl(previous / "skipped_categories.jsonl")
    previous_qa = read_jsonl(previous / "conflict_qa_v2.jsonl")
    audit = build_recovery_audit(groups, previous_schemas, previous_core, previous_skips, previous_qa)
    write_jsonl(args.output_dir / "eligibility_audit.jsonl", audit)
    eligible_ids = {row["parent_job_id"] for row in audit if row["eligible"]}
    eligible_groups = [row for row in groups if row["job_id"] in eligible_ids]
    if args.parent_id:
        requested = set(args.parent_id)
        ineligible = sorted(requested - eligible_ids)
        if ineligible:
            raise SystemExit(f"requested parent IDs are not eligible: {ineligible}")
        eligible_groups = [row for row in eligible_groups if row["job_id"] in requested]
    if args.max_cases is not None:
        eligible_groups = eligible_groups[:args.max_cases]
    shard_groups = [row for index, row in enumerate(eligible_groups)
                    if index % args.num_shards == args.shard_index]
    write_jsonl(args.output_dir / "eligible_groups.jsonl", shard_groups)

    base_summary = {
        "mode": "dry_run" if args.dry_run else "recovery",
        "total_groups": len(groups),
        "eligible_all": len(eligible_ids),
        "eligible_after_filters": len(eligible_groups),
        "selected_for_shard": len(shard_groups),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "exclusion_reason_counts": dict(Counter(
            reason for row in audit if not row["eligible"] for reason in row["exclusion_reasons"]
        )),
    }
    if args.dry_run:
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(base_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(base_summary, ensure_ascii=False, indent=2))
        return

    required = {
        "grounding-base-url": args.grounding_base_url,
        "grounding-model": args.grounding_model,
        "qa-base-url": args.qa_base_url,
        "qa-model": args.qa_model,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required model arguments: {', '.join(missing)}")

    # Heavy dependencies are imported only for an actual model run, so dry-run
    # selection remains usable on login/local machines.
    from conflict_v2_core import apply_semantic_groups, attach_locked_options, candidate_lists, validate_record
    from run_conflict_pipeline_v2 import (
        GROUNDING_SYSTEM_PROMPT,
        QA_SYSTEM_PROMPT,
        JsonModel,
        centers,
        core_points,
        encode_frames,
        ensure_id,
        qa_prompt,
        role_schema_prompt,
        semantic_gate_prompt,
        visual_content,
    )

    grounding_model = JsonModel(
        args.grounding_base_url, args.grounding_model,
        args.output_dir / "raw_grounding_responses.jsonl",
        system_prompt=GROUNDING_SYSTEM_PROMPT,
    )
    qa_model = JsonModel(
        args.qa_base_url, args.qa_model,
        args.output_dir / "raw_qa_responses.jsonl",
        system_prompt=QA_SYSTEM_PROMPT,
        enable_thinking=True if args.qa_thinking else None,
    )

    detections = read_jsonl(args.output_dir / "recovered_actions.jsonl")
    detection_by_parent = {row["parent_job_id"]: row for row in detections}
    schemas = read_jsonl(args.output_dir / "recovered_action_role_schemas.jsonl")
    schema_by_parent = {row["parent_job_id"]: row for row in schemas}
    groundings = read_jsonl(args.output_dir / "recovered_action_grounding.jsonl")
    grounding_done = {(row["parent_job_id"], row["instance_id"]) for row in groundings}
    semantic_decisions = read_jsonl(args.output_dir / "semantic_contrast_decisions.jsonl")
    attempts = read_jsonl(args.output_dir / "candidate_attempts.jsonl")
    final_qa = read_jsonl(args.output_dir / "recovered_action_content_qa.jsonl")
    skipped = read_jsonl(args.output_dir / "skipped_action_content.jsonl")
    terminal_ids = {row["parent_job_id"] for row in final_qa}
    if not args.retry_skipped:
        terminal_ids.update(row["parent_job_id"] for row in skipped)
    else:
        selected_ids = {row["job_id"] for row in shard_groups}
        skipped[:] = [row for row in skipped if row["parent_job_id"] not in selected_ids]

    for group in shard_groups:
        parent_id = group["job_id"]
        if parent_id in terminal_ids:
            continue
        expected_ids = [row["instance_id"] for row in group["instances"]]
        detection_row = detection_by_parent.get(parent_id)
        if detection_row is None:
            points = []
            if len(group["instances"]) > args.max_action_images:
                raise RuntimeError(
                    f"{parent_id} has {len(group['instances'])} occurrences, exceeding "
                    f"--max-action-images={args.max_action_images}; every occurrence must be sampled"
                )
            frames_per_occurrence = max(
                1,
                min(args.action_frames_per_occurrence,
                    args.max_action_images // len(group["instances"])),
            )
            for instance in group["instances"]:
                points.extend(centers(
                    float(instance["core"][0]), float(instance["core"][1]),
                    frames_per_occurrence, instance["instance_id"],
                ))
            frames = encode_frames(args.video_root / group["video"], points, args.frame_width)
            detection = grounding_model.call(
                parent_id + "__recover_shared_action",
                visual_content(action_identification_prompt(group), frames),
                max_tokens=1800,
            )
            errors = validate_recovered_action(detection, expected_ids, args.min_action_confidence)
            detection_row = {
                "parent_job_id": parent_id,
                "video": group["video"],
                "status": "accepted" if not errors else "rejected",
                "errors": errors,
                "detection": detection,
                "sampled_frames": [{"second": round(second, 3), "instance_id": phase}
                                   for second, phase, _ in frames],
            }
            append_or_replace(detections, detection_row)
            detection_by_parent[parent_id] = detection_row
            write_jsonl(args.output_dir / "recovered_actions.jsonl", detections)
        if detection_row["status"] != "accepted":
            append_or_replace(skipped, {
                "parent_job_id": parent_id,
                "stage": "action_identification",
                "reason": detection_row.get("errors", ["recovered_action_rejected"]),
            })
            write_jsonl(args.output_dir / "skipped_action_content.jsonl", skipped)
            continue

        action = str(detection_row["detection"]["shared_action"]).strip()
        schema = schema_by_parent.get(parent_id)
        if schema is None:
            schema = qa_model.call(parent_id + "__recovered_role_schema", role_schema_prompt(action))
            schema["parent_job_id"] = parent_id
            schema["official_action"] = action
            schema["errors"] = validate_role_schema(schema)
            append_or_replace(schemas, schema)
            schema_by_parent[parent_id] = schema
            write_jsonl(args.output_dir / "recovered_action_role_schemas.jsonl", schemas)
        if schema.get("errors"):
            append_or_replace(skipped, {
                "parent_job_id": parent_id,
                "stage": "role_schema",
                "reason": schema["errors"],
            })
            write_jsonl(args.output_dir / "skipped_action_content.jsonl", skipped)
            continue

        recovered_group = dict(group)
        recovered_group["anchor_action_hint"] = action
        for instance in group["instances"]:
            key = (parent_id, instance["instance_id"])
            if key in grounding_done:
                continue
            frames = encode_frames(
                args.video_root / group["video"], core_points(instance, args.core_frames), args.frame_width
            )
            parsed = grounding_model.call(
                parent_id + "__" + instance["instance_id"] + "__recovered_action_content",
                visual_content(action_grounding_prompt(recovered_group, instance, action, schema), frames),
            )
            parsed = ensure_id(parsed, instance["instance_id"], "action_content")
            parsed["parent_job_id"] = parent_id
            parsed["official_interval"] = instance["core"]
            parsed["sampled_seconds"] = [round(second, 3) for second, _, _ in frames]
            groundings.append(parsed)
            grounding_done.add(key)
            write_jsonl(args.output_dir / "recovered_action_grounding.jsonl", groundings)

        parent_groundings = [row for row in groundings if row.get("parent_job_id") == parent_id]
        candidates = candidate_lists(
            recovered_group, parent_groundings, [], schema, args.min_grounding_confidence
        )["action_content"]
        if not candidates:
            append_or_replace(skipped, {
                "parent_job_id": parent_id,
                "stage": "candidate_construction",
                "reason": "fewer_than_two_distinct_high_confidence_action_values",
            })
            write_jsonl(args.output_dir / "skipped_action_content.jsonl", skipped)
            continue

        accepted = None
        parent_attempts = []
        for ordinal, original_candidate in enumerate(candidates):
            candidate = original_candidate
            decision = qa_model.call(
                parent_id + "__recovered_semantic_gate__" + candidate["field"],
                semantic_gate_prompt(candidate),
            )
            semantic_decisions.append({
                "parent_job_id": parent_id,
                "category": "action_content",
                "field": candidate["field"],
                "decision": decision,
            })
            candidate, reason = apply_semantic_groups(candidate, decision)
            attempt = {
                "parent_job_id": parent_id,
                "category": "action_content",
                "field": original_candidate["field"],
                "candidate_ordinal": ordinal,
            }
            if candidate is None:
                attempt.update({"status": "semantic_rejected", "reason": reason})
                parent_attempts.append(attempt)
                continue
            parsed = qa_model.call(
                parent_id + "__recovered_qa__" + candidate["field"], qa_prompt(candidate, schema)
            )
            record = attach_locked_options(candidate, str(parsed.get("question", "")), {
                "grounding_model": args.grounding_model,
                "qa_model": args.qa_model,
                "official_intervals_locked": True,
                "recovery_pipeline": "missing_action_content_v1",
            })
            errors = validate_record(record, candidate, recovered_group)
            if errors:
                attempt.update({"status": "validation_rejected", "reason": errors})
                parent_attempts.append(attempt)
                continue
            record["recovery_provenance"] = {
                "original_anchor_action_hint": group.get("anchor_action_hint"),
                "recovered_action": action,
                "action_confidence": detection_row["detection"].get("confidence"),
                "action_reason": detection_row["detection"].get("reason"),
            }
            attempt["status"] = "accepted"
            parent_attempts.append(attempt)
            accepted = record
            break

        semantic_path = args.output_dir / "semantic_contrast_decisions.jsonl"
        write_jsonl(semantic_path, semantic_decisions)
        attempts[:] = [row for row in attempts if row.get("parent_job_id") != parent_id]
        attempts.extend(parent_attempts)
        write_jsonl(args.output_dir / "candidate_attempts.jsonl", attempts)
        if accepted:
            append_or_replace(final_qa, accepted)
            write_jsonl(args.output_dir / "recovered_action_content_qa.jsonl", final_qa)
        else:
            append_or_replace(skipped, {
                "parent_job_id": parent_id,
                "stage": "candidate_traversal",
                "reason": "no_recovered_action_candidate_survived",
            })
            write_jsonl(args.output_dir / "skipped_action_content.jsonl", skipped)

    summary = {
        **base_summary,
        "accepted_action_content_qa": len(final_qa),
        "skipped": len(skipped),
        "skip_stage_counts": dict(Counter(row.get("stage") for row in skipped)),
        "recovered_actions_accepted": sum(row.get("status") == "accepted" for row in detections),
        "recovered_actions_rejected": sum(row.get("status") == "rejected" for row in detections),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

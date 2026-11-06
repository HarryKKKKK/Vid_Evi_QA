#!/usr/bin/env python3
"""Four-category E.T. Bench conflict-QA pipeline with separate VLM/LLM endpoints."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/run_conflict_pipeline_v2.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/run_conflict_pipeline_v2.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import base64
import json
import re
from pathlib import Path

import cv2
from openai import OpenAI

from conflict_v2_core import apply_semantic_groups, attach_locked_options, candidate_lists, validate_record


GROUNDING_SYSTEM_PROMPT = (
    "You are a conservative visual evidence annotator. Report only facts visibly supported by the supplied frames "
    "inside the marked evidence window. The official action label and interval are authoritative. Never compare "
    "occurrences, infer intent, or fill missing details from common sense. Use not_visible when uncertain."
)
QA_SYSTEM_PROMPT = (
    "You are constructing a research dataset about multiple valid answers across repeated video events. Follow the "
    "provided JSON schema exactly. Preserve the official action semantics and grounded evidence. Never invent, rename, "
    "or select answers and timestamps; the caller will attach those deterministically."
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class JsonModel:
    def __init__(self, base_url: str, model: str, raw_path: Path, *, system_prompt: str,
                 enable_thinking: bool | None = None):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=300, max_retries=2)
        self.model = model
        self.raw_path = raw_path
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        raw_path.parent.mkdir(parents=True, exist_ok=True)

    def call(self, request_id: str, content, max_tokens: int = 1600) -> dict:
        extra_body = None if self.enable_thinking is None else {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        text = response.choices[0].message.content or ""
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "request_id": request_id,
                "model": self.model,
                "response_id": response.id,
                "usage": response.usage.model_dump() if response.usage else None,
                "content": text,
            }, ensure_ascii=False) + "\n")
        return parse_json(text)


def centers(start: float, end: float, count: int, phase: str) -> list[tuple[float, str]]:
    if count <= 0 or end <= start:
        return []
    step = (end - start) / count
    return [(start + (i + .5) * step, phase) for i in range(count)]


def core_points(instance: dict, count: int) -> list[tuple[float, str]]:
    return centers(float(instance["core"][0]), float(instance["core"][1]), count, "core")


def temporal_points(group: dict, index: int, context_seconds: float, count: int) -> list[tuple[float, str]]:
    instance = group["instances"][index]
    start, end = map(float, instance["core"])
    previous_end = float(group["instances"][index - 1]["core"][1]) if index else 0.0
    next_start = float(group["instances"][index + 1]["core"][0]) if index + 1 < len(group["instances"]) else float(group["video_duration"])
    pre_start = max(previous_end, start - context_seconds, 0.0)
    post_end = min(next_start, end + context_seconds, float(group["video_duration"]))
    return centers(pre_start, start, count, "before") + centers(start, end, max(2, count // 2), "core") + centers(end, post_end, count, "after")


def encode_frames(video_path: Path, points: list[tuple[float, str]], width: int) -> list[tuple[float, str, str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_safe_second = None
    if fps > 0 and frame_count > 1:
        # Seeking to the container duration can land just beyond the final
        # decodable frame. Leave one frame of headroom for backend rounding.
        last_safe_second = max(0.0, (frame_count - 2) / fps)
    frames = []
    try:
        for requested_second, phase in points:
            target = max(0.0, requested_second)
            if last_safe_second is not None:
                target = min(target, last_safe_second)
            step = 1.0 / fps if fps > 0 else 0.04
            offsets = (0.0, -step, step, -0.10, 0.10, -0.25, 0.25, -0.50, 0.50)
            attempts = []
            for offset in offsets:
                second = max(0.0, target + offset)
                if last_safe_second is not None:
                    second = min(second, last_safe_second)
                if not any(abs(second - old) < 1e-6 for old in attempts):
                    attempts.append(second)
            ok, frame, sampled_second = False, None, target
            for second in attempts:
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                ok, frame = cap.read()
                if ok:
                    sampled_second = second
                    break
            if not ok:
                raise RuntimeError(
                    f"cannot decode {video_path} near {requested_second:.3f}s "
                    f"after {len(attempts)} bounded attempts"
                )
            height, original_width = frame.shape[:2]
            if original_width > width:
                frame = cv2.resize(frame, (width, round(height * width / original_width)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError(f"cannot encode {video_path} at {sampled_second:.3f}s")
            frames.append((sampled_second, phase, base64.b64encode(encoded).decode("ascii")))
    finally:
        cap.release()
    return frames


def visual_content(prompt: str, frames: list[tuple[float, str, str]]) -> list[dict]:
    content = [{"type": "text", "text": prompt}]
    for second, phase, encoded in frames:
        content.append({"type": "text", "text": f"{phase.upper()} frame at {second:.3f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    return content


def role_schema_prompt(action: str) -> str:
    return (
        "Design a compact visual grounding schema for one official action phrase. Return strict JSON only: "
        "{official_action:string,roles:[{name:snake_case_string,description:string,question_focus:string}]}. "
        "Give 1-4 action-specific semantic roles, ordered by importance. Prefer manipulated/placed/removed objects, "
        "source, destination, support, instrument, or recipient as appropriate. Do not include before/after events, "
        "scene/background, emotion, intent, or generic 'object' when a specific role name is possible. "
        f"Official action phrase: {action!r}"
    )


def grounding_schema(role_schema: dict) -> dict:
    field = {"value": "concise visible value or not_visible", "confidence": "number 0..1", "visual_basis": "short frame-grounded basis"}
    return {
        "instance_id": "string",
        "action_content": {row["name"]: field for row in role_schema.get("roles", [])},
        "scene_content": {name: field for name in ("location", "salient_background", "actor_appearance", "co_participants")},
    }


def core_prompt(group: dict, instance: dict, role_schema: dict) -> str:
    return (
        "Ground ONE occurrence using only frames sampled inside its official E.T. Bench interval. "
        "The official action is authoritative; do not relabel it and do not compare other occurrences. "
        "First fill action-specific roles, then concise scene content visible consistently in the interval. "
        "Use not_visible instead of guessing. Return strict JSON only. "
        f"Official action: {group['anchor_action_hint']}. Instance: {instance['instance_id']}. "
        f"Official interval: {instance['core']}. Role definitions: {json.dumps(role_schema, ensure_ascii=False)}. "
        f"Output schema: {json.dumps(grounding_schema(role_schema), ensure_ascii=False)}"
    )


def temporal_prompt(group: dict, instance: dict) -> str:
    schema = {"instance_id": "string", "temporal_content": {
        "event_before": {"value": "concise visible event or not_visible", "confidence": "number 0..1", "visual_basis": "string"},
        "event_after": {"value": "concise visible event or not_visible", "confidence": "number 0..1", "visual_basis": "string"},
    }}
    return (
        "Describe only the salient visible event immediately before and immediately after the marked CORE occurrence. "
        "BEFORE and AFTER frames never overlap another official occurrence. Do not repeat the CORE action as context, "
        "and use not_visible if there is no stable event. Return strict JSON only. "
        f"Official CORE action: {group['anchor_action_hint']}. Instance: {instance['instance_id']}. "
        f"Output schema: {json.dumps(schema, ensure_ascii=False)}"
    )


def qa_prompt(candidate: dict, role_schema: dict) -> str:
    category_rules = {
        "action_content": "Ask directly about the selected action-specific semantic role.",
        "scene_context": "Ask about the selected scene detail while anchoring it to the official action.",
        "surrounding_event": "Ask what happened immediately before or after the official action, matching the selected field.",
        "temporal_localization": "Ask when the official action occurs in the video.",
    }
    return (
        "Write one concise natural-language question for a multi-answer conflict-evidence item. Return strict JSON only: "
        "{question:string}. Do not add answers, options, occurrence numbers, timestamps, first/second wording, or facts "
        "not in the supplied structure. The question must permit every supplied alternative under the same interpretation. "
        "Preserve the official action meaning. " + category_rules[candidate["category"]] + " "
        f"Role schema: {json.dumps(role_schema, ensure_ascii=False)}. "
        f"Locked evidence structure: {json.dumps(candidate, ensure_ascii=False)}"
    )


def semantic_gate_prompt(candidate: dict) -> str:
    indexed = [{"index": i, "answer": row["answer"]} for i, row in enumerate(candidate["alternatives"])]
    return (
        "Conservatively decide whether grounded values for one fixed semantic field contain at least two genuinely "
        "different, mutually exclusive answers. Merge synonyms, article changes, paraphrases, and granularity-only "
        "differences. Do not invent or rewrite values. Return strict JSON only: "
        "{genuinely_distinct:boolean,groups:[[integer_index,...],...],reason:string}. Every supplied index must appear "
        "exactly once. If fewer than two groups remain, genuinely_distinct must be false. "
        f"Category: {candidate['category']}. Field: {candidate['field']}. "
        f"Official action: {candidate['official_action']}. Indexed values: {json.dumps(indexed, ensure_ascii=False)}"
    )


def ensure_id(parsed: dict, instance_id: str, container: str) -> dict:
    parsed["instance_id"] = instance_id
    parsed.setdefault(container, {})
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True, help="Parent manifest JSONL with official intervals")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-base-url", required=True)
    parser.add_argument("--grounding-model", required=True)
    parser.add_argument("--qa-base-url", required=True)
    parser.add_argument("--qa-model", required=True)
    parser.add_argument("--qa-thinking", action="store_true", help="Enable Qwen-style thinking for text QA calls")
    parser.add_argument("--core-frames", type=int, default=12)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--context-seconds", type=float, default=5.0)
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--min-confidence", type=float, default=.75)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    all_groups = read_jsonl(args.groups)
    groups = [group for index, group in enumerate(all_groups) if index % args.num_shards == args.shard_index]
    grounding_model = JsonModel(args.grounding_base_url, args.grounding_model,
                                args.output_dir / "raw_grounding_responses.jsonl",
                                system_prompt=GROUNDING_SYSTEM_PROMPT)
    qa_model = JsonModel(args.qa_base_url, args.qa_model, args.output_dir / "raw_qa_responses.jsonl",
                         system_prompt=QA_SYSTEM_PROMPT,
                         enable_thinking=True if args.qa_thinking else None)

    schemas_path = args.output_dir / "action_role_schemas.jsonl"
    schemas = {x["parent_job_id"]: x for x in read_jsonl(schemas_path)}
    core_path = args.output_dir / "core_grounding.jsonl"
    core_rows = read_jsonl(core_path)
    core_done = {(x["parent_job_id"], x["instance_id"]) for x in core_rows}
    temporal_path = args.output_dir / "temporal_grounding.jsonl"
    temporal_rows = read_jsonl(temporal_path)
    temporal_done = {(x["parent_job_id"], x["instance_id"]) for x in temporal_rows}

    for group in groups:
        parent_id = group["job_id"]
        if parent_id not in schemas:
            parsed = qa_model.call(parent_id + "__role_schema", role_schema_prompt(group["anchor_action_hint"]))
            parsed["parent_job_id"] = parent_id
            parsed["official_action"] = group["anchor_action_hint"]
            schemas[parent_id] = parsed
            write_jsonl(schemas_path, list(schemas.values()))
        role_schema = schemas[parent_id]
        video_path = args.video_root / group["video"]
        for index, instance in enumerate(group["instances"]):
            key = (parent_id, instance["instance_id"])
            if key not in core_done:
                frames = encode_frames(video_path, core_points(instance, args.core_frames), args.frame_width)
                parsed = grounding_model.call(parent_id + "__" + instance["instance_id"] + "__core",
                                              visual_content(core_prompt(group, instance, role_schema), frames))
                parsed = ensure_id(parsed, instance["instance_id"], "action_content")
                parsed["parent_job_id"] = parent_id
                parsed["official_interval"] = instance["core"]
                parsed["sampled_seconds"] = [round(x[0], 3) for x in frames]
                core_rows.append(parsed)
                core_done.add(key)
                write_jsonl(core_path, core_rows)
            if key not in temporal_done:
                frames = encode_frames(video_path, temporal_points(group, index, args.context_seconds, args.context_frames), args.frame_width)
                parsed = grounding_model.call(parent_id + "__" + instance["instance_id"] + "__temporal",
                                              visual_content(temporal_prompt(group, instance), frames))
                parsed = ensure_id(parsed, instance["instance_id"], "temporal_content")
                parsed["parent_job_id"] = parent_id
                parsed["official_interval"] = instance["core"]
                parsed["sampled_frames"] = [{"second": round(x[0], 3), "phase": x[1]} for x in frames]
                temporal_rows.append(parsed)
                temporal_done.add(key)
                write_jsonl(temporal_path, temporal_rows)

    final_records, skipped_records, semantic_decisions, candidate_attempts = [], [], [], []

    def try_candidates(parent_id: str, group: dict, candidates: list[dict], role_schema: dict) -> dict | None:
        for ordinal, original_candidate in enumerate(candidates):
            candidate = original_candidate
            attempt = {"parent_job_id": parent_id, "category": candidate["category"],
                       "field": candidate["field"], "candidate_ordinal": ordinal}
            if candidate["category"] != "temporal_localization":
                decision = qa_model.call(parent_id + "__semantic_gate__" + candidate["category"] + "__" + candidate["field"],
                                         semantic_gate_prompt(candidate))
                semantic_decisions.append({"parent_job_id": parent_id, "category": candidate["category"],
                                           "field": candidate["field"], "decision": decision})
                candidate, reason = apply_semantic_groups(candidate, decision)
                if candidate is None:
                    attempt.update({"status": "semantic_rejected", "reason": reason})
                    candidate_attempts.append(attempt)
                    continue
            parsed = qa_model.call(parent_id + "__qa__" + candidate["category"] + "__" + candidate["field"],
                                   qa_prompt(candidate, role_schema))
            record = attach_locked_options(candidate, str(parsed.get("question", "")), {
                "grounding_model": args.grounding_model,
                "qa_model": args.qa_model,
                "official_intervals_locked": True,
            })
            errors = validate_record(record, candidate, group)
            if errors:
                attempt.update({"status": "validation_rejected", "reason": errors})
                candidate_attempts.append(attempt)
                continue
            attempt["status"] = "accepted"
            candidate_attempts.append(attempt)
            return record
        return None

    for group in groups:
        parent_id = group["job_id"]
        group_core = [x for x in core_rows if x.get("parent_job_id") == parent_id]
        group_temporal = [x for x in temporal_rows if x.get("parent_job_id") == parent_id]
        lists = candidate_lists(group, group_core, group_temporal, schemas[parent_id], args.min_confidence)

        action_record = try_candidates(parent_id, group, lists["action_content"], schemas[parent_id])
        if action_record:
            final_records.append(action_record)
            skipped_records.append({"parent_job_id": parent_id, "category": "scene_context",
                                    "reason": "not_attempted_action_content_succeeded"})
        else:
            skipped_records.append({"parent_job_id": parent_id, "category": "action_content",
                                    "reason": "no_candidate_survived_traversal"})
            scene_record = try_candidates(parent_id, group, lists["scene_context"], schemas[parent_id])
            if scene_record:
                final_records.append(scene_record)
            else:
                skipped_records.append({"parent_job_id": parent_id, "category": "scene_context",
                                        "reason": "no_candidate_survived_traversal"})

        temporal_record = try_candidates(parent_id, group, lists["surrounding_event"], schemas[parent_id])
        if temporal_record:
            final_records.append(temporal_record)
        else:
            skipped_records.append({"parent_job_id": parent_id, "category": "surrounding_event",
                                    "reason": "no_candidate_survived_traversal"})

        time_record = try_candidates(parent_id, group, lists["temporal_localization"], schemas[parent_id])
        if time_record:
            final_records.append(time_record)
        else:
            skipped_records.append({"parent_job_id": parent_id, "category": "temporal_localization",
                                    "reason": "fewer_than_two_official_intervals_or_validation_failed"})

    write_jsonl(args.output_dir / "conflict_qa_v2.jsonl", final_records)
    write_jsonl(args.output_dir / "skipped_categories.jsonl", skipped_records)
    write_jsonl(args.output_dir / "semantic_contrast_decisions.jsonl", semantic_decisions)
    write_jsonl(args.output_dir / "candidate_attempts.jsonl", candidate_attempts)
    summary = {
        "videos": len(groups),
        "total_manifest_videos": len(all_groups),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "qa_records": len(final_records),
        "accepted": sum(x["status"] == "accepted" for x in final_records),
        "rejected": sum(x["status"] == "rejected" for x in final_records),
        "skipped_categories": len(skipped_records),
        "by_category": {category: sum(x["conflict_type"] == category for x in final_records)
                        for category in ("action_content", "scene_context", "surrounding_event", "temporal_localization")},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

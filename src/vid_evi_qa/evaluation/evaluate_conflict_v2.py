#!/usr/bin/env python3
"""Evaluate VLM conflict recognition and alternative-set recovery."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/evaluation/evaluate_conflict_v2.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/evaluate_conflict_v2.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import base64
import hashlib
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
import sys
sys.path.insert(0, str(Path(_LAYOUT_LEGACY_FILE).resolve().parents[2] / "scripts" / "eval"))
from prompt_v2 import PROMPT_VERSION, PROMPT_SHA256, build_prompt as render_v2, fingerprint, response_errors
import prompt_v2 as _prompt_protocol

from run_conflict_pipeline_v2 import encode_frames


def build_prompt(question: str, options: list[dict]) -> str:
    if [x["option_id"] for x in options] != list("ABCDE"):
        raise ValueError("v2 requires ordered options A through E")
    return render_v2(question, [x["answer"] for x in options])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def qa_fingerprint(record: dict) -> str:
    payload = {
        "id": record.get("id"),
        "question": record.get("question"),
        "options": [{"option_id": x.get("option_id"), "answer": x.get("answer")} for x in record.get("options", [])],
        "valid_option_ids": record.get("valid_option_ids", []),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("response does not contain a JSON object")
    return json.loads(match.group(0))


def even_pick(values: list[float], count: int) -> list[float]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)] if count > 1 else [values[len(values) // 2]]


def evidence_points(record: dict, group: dict, core_by_key: dict, temporal_by_key: dict,
                    frames_per_occurrence: int, max_frames: int) -> list[tuple[float, str]]:
    category = record["conflict_type"]
    field = record["grounding_field"]
    points_by_instance: dict[str, list[float]] = {}
    for instance in group["instances"]:
        instance_id = instance["instance_id"]
        if category == "surrounding_event":
            phase = "before" if field == "event_before" else "after"
            row = temporal_by_key[(record["parent_job_id"], instance_id)]
            values = [float(x["second"]) for x in row.get("sampled_frames", []) if x.get("phase") == phase]
        else:
            row = core_by_key[(record["parent_job_id"], instance_id)]
            values = [float(x) for x in row.get("sampled_seconds", [])]
        points_by_instance[instance_id] = sorted(set(values))

    instance_count = max(1, len(points_by_instance))
    quota = max(1, min(frames_per_occurrence, max_frames // instance_count))
    selected = {key: even_pick(values, quota) for key, values in points_by_instance.items()}
    flattened = [(second, "evidence") for values in selected.values() for second in values]
    if len(flattened) < max_frames:
        used = {round(second, 6) for second, _ in flattened}
        extras = sorted(
            second for values in points_by_instance.values() for second in values
            if round(second, 6) not in used
        )
        flattened.extend((second, "evidence") for second in even_pick(extras, max_frames - len(flattened)))
    return sorted(flattened[:max_frames])


def score(parsed: dict, expected: list[str]) -> dict:
    schema_errors = response_errors(parsed)
    if not isinstance(parsed, dict):
        parsed = {}
    expected_set = set(expected)
    raw_probabilities = parsed.get("option_probabilities_if_answered")
    probabilities = {}
    if isinstance(raw_probabilities, dict):
        for option_id, value in raw_probabilities.items():
            normalized_id = str(option_id).strip().upper()
            try:
                probability = float(value)
            except (TypeError, ValueError):
                continue
            if re.fullmatch(r"[A-Z]", normalized_id) and probability >= 0:
                probabilities[normalized_id] = probability

    # Recover an alternative set without prompting the model to bind evidence
    # groups to option IDs: take the top-k conditional options, where k is the
    # number of independently supported gold alternatives.
    predicted = [
        option_id for option_id, _ in sorted(
            probabilities.items(), key=lambda item: (-item[1], item[0])
        )[:len(expected)]
    ]
    predicted_set = set(predicted)
    overlap = len(expected_set & predicted_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    label = str(parsed.get("label", "")).strip().upper()
    answer = str(parsed.get("answer", "")).strip().upper()
    reject_reason = str(parsed.get("reject_reason", "")).strip().upper()
    decision_matches_gold = (
        label == "UNANSWERABLE"
        and answer == "R"
        and reject_reason == "CONFLICTING_EVIDENCE"
    )
    return {
        "expected_label": "UNANSWERABLE",
        "expected_answer": "R",
        "expected_reject_reason": "CONFLICTING_EVIDENCE",
        "predicted_label": label,
        "predicted_answer": answer,
        "predicted_reject_reason": reject_reason,
        "label_correct": decision_matches_gold and not schema_errors,
        "decision_matches_gold": decision_matches_gold,
        "schema_errors": schema_errors,
        "response_validation_version": getattr(_prompt_protocol, "RESPONSE_VALIDATION_VERSION", "legacy-unversioned"),
        "expected_option_ids": expected,
        "predicted_option_ids": predicted,
        "option_set_source": "top_k_conditional_probabilities" if probabilities else "not_requested_in_v2",
        "expected_option_probability_mass": sum(
            probabilities.get(option_id, 0.0) for option_id in expected_set
        ) if probabilities else None,
        "option_set_exact": (predicted_set == expected_set) if probabilities else None,
        "option_precision": precision if probabilities else None,
        "option_recall": recall if probabilities else None,
        "option_f1": f1 if probabilities else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--core-grounding", type=Path, required=True)
    parser.add_argument("--temporal-grounding", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--source-task", choices=("tal", "tem", "all"), default="tal")
    parser.add_argument("--media-mode", choices=("frames", "video"), default="frames")
    parser.add_argument("--frames-per-occurrence", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-null-official-action", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {x["job_id"]: x for x in read_jsonl(args.groups)}
    core_by_key = {(x["parent_job_id"], x["instance_id"]): x for x in read_jsonl(args.core_grounding)}
    temporal_by_key = {(x["parent_job_id"], x["instance_id"]): x for x in read_jsonl(args.temporal_grounding)}
    records = read_jsonl(args.qa)
    if args.source_task != "all":
        records = [x for x in records if groups[x["parent_job_id"]]["source_task"] == args.source_task]
    if args.limit is not None:
        records = records[:args.limit]
    if not args.allow_null_official_action and any(
        groups[x["parent_job_id"]].get("anchor_action_hint") is None for x in records
    ):
        raise SystemExit("Selected records contain null official actions; repair TEM action anchors before evaluation")

    config_fingerprint = fingerprint({
        "model": args.model, "model_tag": args.model_tag, "media_mode": args.media_mode,
        "frames_per_occurrence": args.frames_per_occurrence, "max_frames": args.max_frames,
        "frame_width": args.frame_width, "video_root": str(args.video_root),
        "groups": fingerprint(groups), "core": fingerprint(list(core_by_key.values())),
        "temporal": fingerprint(list(temporal_by_key.values())),
    })
    output_jsonl = args.output_dir / "conflict_eval.jsonl"
    previous = read_jsonl(output_jsonl) if output_jsonl.exists() else []
    selected_fingerprints = {x["id"]: qa_fingerprint(x) for x in records}
    completed = {
        x["id"]: x for x in previous
        if x.get("qa_fingerprint") == selected_fingerprints.get(x.get("id"))
        and x.get("prompt_version") == PROMPT_VERSION
        and x.get("prompt_sha256") == PROMPT_SHA256
        and x.get("config_fingerprint") == config_fingerprint
        and not x.get("parse_error") and not response_errors(x.get("parsed_response"))
    }
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=2)
    write_lock = threading.Lock()

    def evaluate(record: dict) -> dict:
        group = groups[record["parent_job_id"]]
        points = evidence_points(record, group, core_by_key, temporal_by_key,
                                 args.frames_per_occurrence, args.max_frames)
        frames = encode_frames(args.video_root / group["video"], points, args.frame_width)
        prompt = build_prompt(record["question"], record["options"])
        if args.media_mode == "frames":
            content = []
            for second, _, encoded in frames:
                content.extend([
                    {"type": "text", "text": f"Frame at {second:.3f}s:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ])
            content.append({"type": "text", "text": prompt})
        else:
            timestamps = ", ".join(f"{x[0]:.3f}" for x in frames)
            content = [
                {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + ",".join(x[2] for x in frames)}},
                {"type": "text", "text": f"Source timestamps in order: [{timestamps}] seconds.\n\n{prompt}"},
            ]
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=768,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        try:
            parsed = parse_json(raw)
            schema_errors = response_errors(parsed)
            parse_error = "; ".join(schema_errors) or None
        except Exception as exc:
            parsed, parse_error = {}, f"{type(exc).__name__}: {exc}"
        result = {
            "id": record["id"],
            "parent_job_id": record["parent_job_id"],
            "source_dataset": "etbench",
            "evidence_condition": "conflict",
            "question_type": record["conflict_type"],
            "source_task": group["source_task"],
            "conflict_type": record["conflict_type"],
            "benchmark_metadata": {
                "evidence_condition": "conflict",
                "source_dataset": "etbench",
                "question_type": record["conflict_type"],
                "source_task": group["source_task"],
                "question_subtype": record["conflict_type"],
                "expected_behavior": "reject",
            },
            "answer_multiplicity_type": record.get("answer_multiplicity", {}).get("type", "unclassified"),
            "question": record["question"],
            "qa_fingerprint": qa_fingerprint(record),
            "model_tag": args.model_tag,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "config_fingerprint": config_fingerprint,
            "sampled_seconds": [round(x[0], 3) for x in frames],
            "raw_response": raw,
            "parsed_response": parsed,
            "parse_error": parse_error,
        }
        result.update(score(parsed, record["valid_option_ids"]))
        return result

    pending = [x for x in records if x["id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, record): record for record in pending}
        for done_index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # A single oversized/corrupt multimodal request must not discard
                # the rest of a long resumable audit.
                result = {
                    "id": record["id"],
                    "parent_job_id": record["parent_job_id"],
                    "source_dataset": "etbench",
                    "evidence_condition": "conflict",
                    "question_type": record["conflict_type"],
                    "source_task": record.get("source_task", args.source_task),
                    "conflict_type": record["conflict_type"],
                    "benchmark_metadata": {
                        "evidence_condition": "conflict",
                        "source_dataset": "etbench",
                        "question_type": record["conflict_type"],
                        "source_task": record.get("source_task", args.source_task),
                        "question_subtype": record["conflict_type"],
                        "expected_behavior": "reject",
                    },
                    "answer_multiplicity_type": record.get("answer_multiplicity", {}).get("type", "unclassified"),
            "question": record["question"],
                    "qa_fingerprint": qa_fingerprint(record),
                    "model_tag": args.model_tag,
                    "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "config_fingerprint": config_fingerprint,
                    "sampled_seconds": [],
                    "raw_response": "",
                    "parsed_response": {},
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
                result.update(score({}, record["valid_option_ids"]))
            completed[result["id"]] = result
            with write_lock, output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"DONE {done_index}/{len(pending)} {result['id']}", flush=True)

    ordered = [completed[x["id"]] for x in records]
    (args.output_dir / "conflict_eval.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    count = len(ordered)
    summary = {
        "status": "complete" if all(x["parse_error"] is None for x in ordered) else "completed_with_errors",
        "model_tag": args.model_tag,
        "source_task": args.source_task,
        "cases": count,
        "parse_success_rate": sum(x["parse_error"] is None for x in ordered) / count if count else 0,
        "conflict_label_accuracy": sum(x["label_correct"] for x in ordered) / count if count else 0,
        "option_set_exact_accuracy": None,
        "macro_option_f1": None,
        "unavailable_metrics_reason": "v2 does not request option probabilities or explicit option IDs",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

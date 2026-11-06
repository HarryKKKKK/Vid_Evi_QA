#!/usr/bin/env python3
"""Ground evidence windows claimed by correct insufficient-condition answers.

This script is an adapter between downstream evaluation outputs and the
existing semantic leak classifier. It reads one or more evaluation JSON/JSONL
files, keeps only records for which the model said ANSWERABLE and selected the
gold option, and treats the reported ``evidence_seconds=[start, end]`` as a
candidate window.

Crucially, grounding is performed on the final frozen insufficient video that
the answering model saw, not on the unmodified source video. The grounding VLM
sees the question and question type, but never the choices, gold answer, or the
answering model's explanation. Output records use the same schema expected by
``classify_groundings.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from common import (
    append_jsonl,
    extract_frame_grid,
    frames_in_range,
    latest_jsonl_records,
    parse_endpoints,
    probe_video,
    require_binaries,
    selected_indices,
)
from ground_windows import WindowGrounder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE_DIR = ROOT / "source_datasets/next_gqa/freeze_videos_no_evidence_leak"
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/correct_answer_evidence_groundings.jsonl"
PROMPT_VERSION = "nextgqa-correct-answer-claimed-evidence-grounding-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help=(
            "Evaluation JSON/JSONL path or glob. Repeat for multiple models. "
            "Records are deduplicated by (model_tag, video_id, qid)."
        ),
    )
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", action="append", metavar="TAG,BASE_URL,MODEL")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only validate/select claims and print counts; no model server is required.",
    )
    parser.add_argument(
        "--padding-seconds",
        type=float,
        default=1.0,
        help="Context added to both sides of each claimed interval (default: 1.0).",
    )
    parser.add_argument(
        "--min-window-seconds",
        type=float,
        default=2.0,
        help="Symmetrically expand shorter claimed intervals to this duration.",
    )
    parser.add_argument(
        "--max-window-seconds",
        type=float,
        default=0.0,
        help=(
            "Mark longer windows needs_review instead of silently clipping them; "
            "0 disables the duration limit (default)."
        ),
    )
    parser.add_argument(
        "--max-frames-per-window",
        type=int,
        default=64,
        help="Uniformly thin long windows to at most this many frames; 0 keeps all.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-width", type=int, default=336)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit item groups after sharding; all model claims for an item stay together.",
    )
    parser.add_argument("--retry-status", default="error,needs_review")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        value = value["results"]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array of result objects")
    return value


def expand_result_paths(patterns: Iterable[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        if not matches:
            raise ValueError(f"no result files match {pattern!r}")
        for path in matches:
            paths[str(path.resolve())] = path.resolve()
    return sorted(paths.values(), key=lambda path: (path.stat().st_mtime, str(path)))


def _normal_letter(value: object) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else None


def is_correct_answered(row: dict) -> bool:
    parsed = row.get("model_parsed") or {}
    return (
        str(parsed.get("label", "")).upper() == "ANSWERABLE"
        and _normal_letter(parsed.get("answer")) == _normal_letter(row.get("answer"))
        and _normal_letter(row.get("answer")) is not None
    )


def reference_answer(row: dict) -> str:
    gold = _normal_letter(row.get("answer"))
    choices = row.get("choices")
    if gold is None or not isinstance(choices, list):
        raise ValueError("missing gold option or choices")
    index = ord(gold) - ord("A")
    if index < 0 or index >= len(choices):
        raise ValueError(f"gold option {gold!r} is outside {len(choices)} choices")
    answer = str(choices[index]).strip()
    if not answer:
        raise ValueError("reference answer text is empty")
    return answer


def claimed_interval(row: dict) -> tuple[float, float]:
    parsed = row.get("model_parsed") or {}
    value = parsed.get("evidence_seconds")
    if not isinstance(value, (list, tuple)) or len(value) < 1:
        raise ValueError(f"evidence_seconds must contain one or more timestamps, got {value!r}")
    try:
        timestamps = [float(part) for part in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evidence_seconds are not numeric: {value!r}") from exc
    if not all(math.isfinite(part) for part in timestamps):
        raise ValueError(f"evidence_seconds are not finite: {value!r}")
    return min(timestamps), max(timestamps)


def thin_frames(frames: Sequence, maximum: int) -> list:
    """Uniformly retain endpoints and interior frames without duplicates."""
    if maximum <= 0 or len(frames) <= maximum:
        return list(frames)
    if maximum == 1:
        return [frames[len(frames) // 2]]
    indices = [round(index * (len(frames) - 1) / (maximum - 1)) for index in range(maximum)]
    return [frames[index] for index in indices]


def audit_interval(
    claimed: Sequence[float],
    duration: float,
    *,
    padding: float,
    minimum: float,
) -> tuple[float, float]:
    if duration <= 0:
        raise ValueError(f"invalid video duration: {duration}")
    start, end = float(claimed[0]), float(claimed[1])
    if end < 0 or start > duration:
        raise ValueError(
            f"claimed interval {list(claimed)!r} does not overlap video duration {duration:.3f}s"
        )
    start = max(0.0, start - padding)
    end = min(duration, end + padding)
    if end <= start:
        center = min(max((start + end) / 2.0, 0.0), duration)
        start = center
        end = center
    if end - start < minimum:
        center = (start + end) / 2.0
        start = max(0.0, center - minimum / 2.0)
        end = min(duration, start + minimum)
        start = max(0.0, end - minimum)
    if end <= start:
        raise ValueError(f"claimed interval lies outside video: {list(claimed)!r}")
    return round(start, 6), round(end, 6)


def source_model_tag(row: dict, path: Path) -> str:
    tag = str(row.get("model_tag") or "").strip()
    if tag:
        return tag
    return f"untagged_{path.parent.name}_{path.stem}"


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def record_key(row: dict) -> str:
    return "::".join(
        [
            safe_tag(str(row["source_eval_model_tag"])),
            str(row["video_id"]),
            str(row["qid"]),
        ]
    )


def load_correct_claims(paths: Sequence[Path]) -> tuple[list[dict], dict[str, int]]:
    latest: dict[tuple[str, str, str], dict] = {}
    stats = defaultdict(int)
    for path in paths:
        for row in _read_records(path):
            stats["input_records"] += 1
            if not is_correct_answered(row):
                continue
            stats["correct_answered_records"] += 1
            tag = source_model_tag(row, path)
            key = (tag, str(row.get("video_id")), str(row.get("qid")))
            latest[key] = {
                **row,
                "source_eval_model_tag": tag,
                "source_result_file": str(path),
            }
    stats["deduplicated_claims"] = len(latest)
    return [latest[key] for key in sorted(latest)], dict(stats)


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.frame_width <= 0:
        raise SystemExit("--fps and --frame-width must be positive")
    if (
        args.padding_seconds < 0
        or args.min_window_seconds <= 0
        or args.max_window_seconds < 0
        or args.max_frames_per_window < 0
    ):
        raise SystemExit("invalid window-size arguments")
    paths = expand_result_paths(args.results)
    claims, load_stats = load_correct_claims(paths)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for claim in claims:
        groups[(str(claim.get("video_id")), str(claim.get("qid")))].append(claim)
    group_keys = sorted(groups)

    if args.preflight:
        preflight = defaultdict(int)
        examples: dict[str, list[dict]] = {"invalid": [], "overlong": [], "missing_video": []}
        for claim in claims:
            video_id, qid = str(claim.get("video_id")), str(claim.get("qid"))
            video_path = args.freeze_dir / f"{video_id}_q{qid}_freeze.mp4"
            if not video_path.is_file():
                preflight["missing_freeze_videos"] += 1
                if len(examples["missing_video"]) < 5:
                    examples["missing_video"].append({"video_id": video_id, "qid": qid})
            try:
                claimed = claimed_interval(claim)
                duration = float(claim.get("video_duration") or 0)
                if duration <= 0:
                    raise ValueError("missing positive video_duration")
                interval = audit_interval(
                    claimed,
                    duration,
                    padding=args.padding_seconds,
                    minimum=args.min_window_seconds,
                )
                preflight["valid_claimed_intervals"] += 1
                if args.max_window_seconds and interval[1] - interval[0] > args.max_window_seconds:
                    preflight["overlong_windows"] += 1
                    if len(examples["overlong"]) < 5:
                        examples["overlong"].append({
                            "video_id": video_id,
                            "qid": qid,
                            "source_eval_model_tag": claim["source_eval_model_tag"],
                            "claimed_interval": list(claimed),
                            "audit_interval": list(interval),
                        })
            except ValueError as exc:
                preflight["invalid_claimed_intervals"] += 1
                if len(examples["invalid"]) < 5:
                    examples["invalid"].append({
                        "video_id": video_id,
                        "qid": qid,
                        "source_eval_model_tag": claim["source_eval_model_tag"],
                        "error": str(exc),
                    })
        print(json.dumps({
            "result_files": [str(path) for path in paths],
            **load_stats,
            "unique_items": len(group_keys),
            **dict(preflight),
            "examples": examples,
        }, ensure_ascii=False, indent=2))
        return

    if not args.endpoint:
        raise SystemExit("--endpoint is required unless --preflight is used")
    require_binaries(args.ffmpeg, args.ffprobe)
    endpoints = parse_endpoints(args.endpoint)
    if len(endpoints) != 1:
        raise SystemExit("grounding requires exactly one --endpoint")
    grounder = WindowGrounder(
        endpoints[0],
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.api_retries,
        disable_thinking=True,
    )

    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(group_keys), shard, args.num_shards, args.limit)
    selected_keys = [group_keys[index] for index in indices]

    output = args.output
    if shard is not None:
        output = output.with_name(f"{output.stem}.{shard}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = latest_jsonl_records([output]) if output.exists() else {}
    retry_statuses = {part.strip() for part in args.retry_status.split(",") if part.strip()}

    print(
        f"[START] files={len(paths)} input={load_stats.get('input_records', 0)} "
        f"correct_answered={load_stats.get('correct_answered_records', 0)} "
        f"deduplicated={len(claims)} items={len(group_keys)} "
        f"selected_items={len(selected_keys)} shard={shard}/{args.num_shards} "
        f"grounder={endpoints[0].tag} output={output}",
        flush=True,
    )

    written = grounded = errors = needs_review = 0
    for position, item_key in enumerate(selected_keys, start=1):
        video_id, qid = item_key
        pending = []
        for claim in groups[item_key]:
            key = record_key(claim)
            old = previous.get(key)
            if args.overwrite or old is None or old.get("status") in retry_statuses:
                pending.append(claim)
        if not pending:
            print(f"[{position}/{len(selected_keys)}] {video_id}_q{qid} skip", flush=True)
            continue

        video_path = args.freeze_dir / f"{video_id}_q{qid}_freeze.mp4"
        holder = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"freeze video not found: {video_path}")
            duration = float(probe_video(video_path, args.ffprobe)["duration"])
            holder, frames = extract_frame_grid(
                video_path,
                fps=args.fps,
                width=args.frame_width,
                ffmpeg=args.ffmpeg,
                temp_root=args.temp_root,
            )
            for claim in pending:
                key = record_key(claim)
                started = time.time()
                parsed = claim.get("model_parsed") or {}
                base = {
                    "key": key,
                    "item_key": f"{video_id}_q{qid}",
                    "video_id": claim.get("video_id"),
                    "qid": claim.get("qid"),
                    "question": claim.get("question"),
                    "question_type": claim.get("question_type") or claim.get("source_task"),
                    "reference_answer": None,
                    "reference_answer_letter": _normal_letter(claim.get("answer")),
                    "candidate_rank": 1,
                    "source_eval_model_tag": claim["source_eval_model_tag"],
                    "source_result_file": claim["source_result_file"],
                    "source_model_claim": {
                        "label": parsed.get("label"),
                        "answer": parsed.get("answer"),
                        "evidence": parsed.get("evidence"),
                        "evidence_seconds": parsed.get("evidence_seconds"),
                        "reason": parsed.get("reason"),
                        "sampled_seconds": claim.get("sampled_seconds"),
                    },
                    "insufficient_video": str(video_path),
                    "sampling": {"fps": args.fps, "frame_width": args.frame_width},
                    "masking": {"mode": "already_frozen_final_insufficient_video"},
                    "prompt_version": PROMPT_VERSION,
                    "grounding_input_policy": (
                        "question_and_type_only_no_choices_no_gold_no_source_model_explanation"
                    ),
                }
                try:
                    base["reference_answer"] = reference_answer(claim)
                    claimed = claimed_interval(claim)
                    interval = audit_interval(
                        claimed,
                        duration,
                        padding=args.padding_seconds,
                        minimum=args.min_window_seconds,
                    )
                    base["candidate"] = {
                        "interval": list(interval),
                        "claimed_interval": [round(claimed[0], 6), round(claimed[1], 6)],
                        "padding_seconds": args.padding_seconds,
                        "selection": "correct_insufficient_answer_reported_evidence_seconds",
                    }
                    if args.max_window_seconds and interval[1] - interval[0] > args.max_window_seconds:
                        raise RuntimeError(
                            f"needs_review: audit window is {interval[1] - interval[0]:.3f}s, "
                            f"above --max-window-seconds={args.max_window_seconds}"
                        )
                    window_frames = frames_in_range(frames, interval[0], interval[1])
                    window_frames = thin_frames(window_frames, args.max_frames_per_window)
                    result = grounder.describe(
                        window_frames,
                        str(claim.get("question", "")),
                        str(claim.get("question_type") or claim.get("source_task") or ""),
                    )
                    record = {
                        **base,
                        "status": "ok",
                        "grounding": result,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    grounded += 1
                except RuntimeError as exc:
                    status = "needs_review" if str(exc).startswith("needs_review:") else "error"
                    record = {
                        **base,
                        "status": status,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    if status == "needs_review":
                        needs_review += 1
                    else:
                        errors += 1
                except Exception as exc:
                    record = {
                        **base,
                        "status": "needs_review" if isinstance(exc, ValueError) else "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    if record["status"] == "needs_review":
                        needs_review += 1
                    else:
                        errors += 1
                append_jsonl(output, record)
                previous[key] = record
                written += 1
                print(
                    f"[{position}/{len(selected_keys)}] {key} {record['status']} "
                    f"interval={(record.get('candidate') or {}).get('interval')} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            for claim in pending:
                key = record_key(claim)
                record = {
                    "key": key,
                    "item_key": f"{video_id}_q{qid}",
                    "video_id": claim.get("video_id"),
                    "qid": claim.get("qid"),
                    "source_eval_model_tag": claim["source_eval_model_tag"],
                    "source_result_file": claim["source_result_file"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                append_jsonl(output, record)
                previous[key] = record
                written += 1
                errors += 1
            print(
                f"[{position}/{len(selected_keys)}] {video_id}_q{qid} "
                f"ITEM_ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if holder is not None:
                holder.cleanup()

    print(
        f"[DONE] wrote={written} grounded={grounded} needs_review={needs_review} "
        f"errors={errors} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

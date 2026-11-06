"""Build visual contrast-extraction jobs from ETBench event candidates.

This stage does not claim that repeated/matched events are conflicting.  It
expands every localized event into pre/core/post/context windows and asks a
vision model to describe each instance independently.  A later stage can keep
only pairs with visible, mutually exclusive slot values.
"""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/build_visual_contrast_jobs.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/build_visual_contrast_jobs.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT
    / "source_datasets/et_bench/conflict_candidates_perception_test.json"
)
DEFAULT_OUTPUT = ROOT / "results/build/etbench/conflict" / "visual_contrast_jobs.jsonl"
DEFAULT_INSTANCE_OUTPUT = (
    ROOT / "results/build/etbench/conflict" / "visual_instance_jobs.jsonl"
)
DEFAULT_VIDEO_ROOT = ROOT / "source_datasets/et_bench/ETBench/videos"

SLOT_NAMES = [
    "anchor_action",
    "actor",
    "actor_appearance",
    "object",
    "object_attributes",
    "before_action",
    "core_manner",
    "after_action",
    "outcome_or_state_after",
    "location",
    "co_participant",
]

_TAL_ACTION_RE = re.compile(r"action category:\s*'([^']+)'", re.IGNORECASE)


def extract_action_hint(item: dict[str, Any]) -> str | None:
    if item.get("task") != "tal":
        return None
    match = _TAL_ACTION_RE.search(item.get("q", ""))
    return match.group(1).strip() if match else None


def clamp_window(start: float, end: float, duration: float) -> list[float]:
    return [round(max(0.0, start), 3), round(min(duration, end), 3)]


def build_instance(
    interval: list[float],
    *,
    instance_id: str,
    role: str,
    duration: float,
    pre_seconds: float,
    post_seconds: float,
    context_seconds: float,
) -> dict[str, Any]:
    start, end = map(float, interval)
    return {
        "instance_id": instance_id,
        "role": role,
        "core": [round(start, 3), round(end, 3)],
        "pre": clamp_window(start - pre_seconds, start, duration),
        "post": clamp_window(end, end + post_seconds, duration),
        "context": clamp_window(
            start - context_seconds, end + context_seconds, duration
        ),
    }


def build_job(
    item: dict[str, Any],
    *,
    video_root: Path,
    pre_seconds: float,
    post_seconds: float,
    context_seconds: float,
) -> dict[str, Any]:
    task = item["task"]
    duration = float(item["duration"])
    instances: list[dict[str, Any]] = []

    for index, interval in enumerate(item.get("anchor_intervals", [])):
        instances.append(
            build_instance(
                interval,
                instance_id=f"anchor_{index}",
                role="reference",
                duration=duration,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                context_seconds=context_seconds,
            )
        )
    role = "matched" if task == "tem" else "occurrence"
    for index, interval in enumerate(item.get("matched_intervals", [])):
        instances.append(
            build_instance(
                interval,
                instance_id=f"{role}_{index}",
                role=role,
                duration=duration,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                context_seconds=context_seconds,
            )
        )

    relative_video = Path(item["video"])
    return {
        "job_id": f"etbench_{item['source']}_{task}_{item['idx']}",
        "source_dataset": "etbench",
        "source_task": task,
        "source_idx": item["idx"],
        "video": relative_video.as_posix(),
        "video_path": str((video_root / relative_video).resolve()),
        "video_duration": duration,
        "original_question": item["q"],
        "anchor_action_hint": extract_action_hint(item),
        "instances": instances,
        "requested_slots": SLOT_NAMES,
        "instruction": (
            "Describe every event instance independently from its context window. "
            "Do not assume that instances differ. For each requested slot, return "
            "a concise visible value, 'same_as_reference' only after independent "
            "description, or 'not_visible'. Then list pairwise differences with "
            "confidence in [0,1]. Do not infer emotions, intent, or off-screen facts."
        ),
        "output_schema": {
            "instance_descriptions": [
                {
                    "instance_id": "string",
                    "slots": {slot: "string" for slot in SLOT_NAMES},
                    "observability_notes": "string",
                }
            ],
            "pairwise_differences": [
                {
                    "instance_a": "string",
                    "instance_b": "string",
                    "slot": "string",
                    "value_a": "string",
                    "value_b": "string",
                    "mutually_exclusive": "boolean",
                    "confidence": "number",
                    "visual_basis": "string",
                }
            ],
        },
        "status": "pending_visual_contrast_extraction",
    }


def iter_jobs(
    items: Iterable[dict[str, Any]],
    *,
    video_root: Path,
    pre_seconds: float,
    post_seconds: float,
    context_seconds: float,
) -> Iterable[dict[str, Any]]:
    for item in items:
        job = build_job(
            item,
            video_root=video_root,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            context_seconds=context_seconds,
        )
        if len(job["instances"]) >= 2:
            yield job


def iter_instance_jobs(jobs: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Flatten group jobs so every visual description is produced independently."""
    for job in jobs:
        for instance in job["instances"]:
            yield {
                "instance_job_id": f"{job['job_id']}__{instance['instance_id']}",
                "parent_job_id": job["job_id"],
                "source_dataset": job["source_dataset"],
                "source_task": job["source_task"],
                "source_idx": job["source_idx"],
                "video": job["video"],
                "video_path": job["video_path"],
                "video_duration": job["video_duration"],
                "anchor_action_hint": job["anchor_action_hint"],
                "instance": instance,
                "requested_slots": job["requested_slots"],
                "instruction": (
                    "Inspect only this event instance and its context window. "
                    "Describe visible facts independently; you are not told how "
                    "other occurrences look. Return one concise value per requested "
                    "slot, 'not_visible' when unsupported, and a confidence in [0,1] "
                    "plus a short visual basis. Do not infer emotions, intent, or "
                    "off-screen facts."
                ),
                "output_schema": {
                    "instance_job_id": "string",
                    "slots": {
                        slot: {
                            "value": "string",
                            "confidence": "number",
                            "visual_basis": "string",
                        }
                        for slot in SLOT_NAMES
                    },
                },
                "status": "pending_independent_visual_description",
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--instance-output", type=Path, default=DEFAULT_INSTANCE_OUTPUT
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--pre-seconds", type=float, default=5.0)
    parser.add_argument("--post-seconds", type=float, default=5.0)
    parser.add_argument("--context-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if min(args.pre_seconds, args.post_seconds, args.context_seconds) < 0:
        parser.error("window sizes must be non-negative")

    items = json.loads(args.input.read_text(encoding="utf-8"))
    jobs = list(
        iter_jobs(
            items,
            video_root=args.video_root,
            pre_seconds=args.pre_seconds,
            post_seconds=args.post_seconds,
            context_seconds=args.context_seconds,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    instance_jobs = list(iter_instance_jobs(jobs))
    args.instance_output.parent.mkdir(parents=True, exist_ok=True)
    with args.instance_output.open("w", encoding="utf-8", newline="\n") as handle:
        for job in instance_jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    instance_count = sum(len(job["instances"]) for job in jobs)
    missing_videos = sum(not Path(job["video_path"]).is_file() for job in jobs)
    print(f"Jobs: {len(jobs)}")
    print(f"Event instances: {instance_count}")
    print(f"Missing videos at configured root: {missing_videos}")
    print(f"Output: {args.output}")
    print(f"Independent instance output: {args.instance_output}")


if __name__ == "__main__":
    main()

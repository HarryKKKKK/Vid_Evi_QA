#!/usr/bin/env python3
"""Ground high-risk NExT-GQA windows on official-evidence-masked frames.

This stage reuses calibrated measurement results to select candidate windows.
It gives the grounding VLM the question and official question type, but never
the answer choices or gold answer.  The VLM only writes a short factual visual
description; leak classification is intentionally deferred to a text-only LLM.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from common import (
    Endpoint,
    Frame,
    append_jsonl,
    extract_frame_grid,
    frames_in_range,
    latest_jsonl_records,
    load_worklist,
    mask_frames,
    minimal_positive_windows,
    parse_endpoints,
    require_binaries,
    selected_indices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED = ROOT / "nextgqa_pipeline/nextgqa_filtered.json"
DEFAULT_VIDEOS = ROOT / "source_datasets/next_gqa/videos"
DEFAULT_MAPPING = ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"
DEFAULT_MEASUREMENTS = str(ROOT / "nextgqa_pipeline/insufficient/measurements.*.jsonl")
DEFAULT_THRESHOLDS = ROOT / "nextgqa_pipeline/insufficient/thresholds.json"
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/window_groundings.jsonl"
GROUNDING_PROMPT_VERSION = "nextgqa-window-grounding-v1.0"


TYPE_GUIDANCE = {
    "TC": (
        "Focus on the people, objects, and actions named in the question, and "
        "state which visible actions overlap in time."
    ),
    "TN": (
        "Describe the relevant visible events in chronological order, especially "
        "what happens after the event named in the question."
    ),
    "TP": (
        "Describe the relevant visible events in chronological order, especially "
        "what happens before the event named in the question."
    ),
    "CW": (
        "Describe the relevant visible events and their order, including context "
        "before the action in the question. Do not invent a causal explanation."
    ),
    "CH": (
        "Describe how the relevant action is visibly carried out, including tools, "
        "objects, manner, and intermediate steps when visible."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--video-id-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL")
    parser.add_argument("--uncertainty-band", type=float, default=0.05)
    parser.add_argument(
        "--max-windows-per-item", type=int, default=0,
        help="Keep the highest-risk N minimal windows per item; 0 keeps all.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-width", type=int, default=336)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-status", default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def threshold_for(config: dict, tag: str, scale_key: str) -> float | None:
    scale = config.get("models", {}).get(tag, {}).get("scales", {}).get(scale_key, {})
    if not scale.get("usable"):
        return None
    value = scale.get("threshold")
    return float(value) if value is not None else None


def candidate_windows(measurement: dict, thresholds: dict, uncertainty_band: float) -> list[dict]:
    """Return minimal positive leaf windows with all contributing score metadata."""
    sources: defaultdict[tuple[float, float], list[dict]] = defaultdict(list)
    for tag, model in measurement.get("models", {}).items():
        if model.get("status") != "ok" or not model.get("eligible"):
            continue
        for scale_key, scale in model.get("scales", {}).items():
            threshold = threshold_for(thresholds, tag, scale_key)
            if threshold is None:
                continue
            cutoff = threshold - uncertainty_band
            for window in scale.get("windows", []):
                value = window.get("discovery", {}).get("normalized_gain")
                if value is None or float(value) < cutoff:
                    continue
                start, end = (round(float(part), 6) for part in window["interval"])
                sources[(start, end)].append({
                    "model_tag": tag,
                    "scale": scale_key,
                    "normalized_gain": float(value),
                    "threshold": threshold,
                    "threshold_excess": float(value) - threshold,
                    "inside_uncertainty_band": float(value) < threshold,
                })

    minimal = minimal_positive_windows(sources.keys())
    candidates: list[dict] = []
    for interval in minimal:
        key = (round(interval[0], 6), round(interval[1], 6))
        contributing = sources[key]
        candidates.append({
            "interval": [key[0], key[1]],
            "risk_score": max(source["threshold_excess"] for source in contributing),
            "max_normalized_gain": max(source["normalized_gain"] for source in contributing),
            "sources": contributing,
        })
    return sorted(
        candidates,
        key=lambda row: (-row["risk_score"], row["interval"][0], row["interval"][1]),
    )


def grounding_prompt(question: str, question_type: str) -> str:
    guidance = TYPE_GUIDANCE.get(
        question_type,
        "Describe the visible people, objects, actions, and event order relevant to the question.",
    )
    return f"""You are a conservative video grounding annotator.

The supplied frames come from one candidate video window. Frames from the dataset's official evidence interval may have been replaced by repeated blurred freeze frames. Treat such replaced content as unavailable and do not infer what it originally contained.

Question type: {question_type}
Question: {question}

{guidance}

Write 1 to 4 short sentences describing only facts visibly supported by these frames. Mention uncertainty or missing events when relevant. Do not answer the question, judge whether it is answerable, guess an unseen event, or discuss answer choices. Output only the factual description."""


class WindowGrounder:
    def __init__(self, endpoint: Endpoint, *, timeout: float, max_tokens: int, retries: int):
        from openai import OpenAI

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self.client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )

    def describe(self, frames: Sequence[Frame], question: str, question_type: str) -> dict:
        if not frames:
            raise ValueError("cannot ground an empty frame sequence")
        content: list[dict[str, object]] = []
        for frame in frames:
            content.append({"type": "text", "text": f"Frame at absolute video time {frame.timestamp:.3f}s:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.jpeg_b64}"},
            })
        content.append({"type": "text", "text": grounding_prompt(question, question_type)})

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                description = (response.choices[0].message.content or "").strip()
                if not description:
                    raise ValueError("grounding model returned an empty description")
                usage = getattr(response, "usage", None)
                return {
                    "description": description,
                    "model_tag": self.endpoint.tag,
                    "model": self.endpoint.model,
                    "n_frames": len(frames),
                    "timestamps": [round(frame.timestamp, 6) for frame in frames],
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                    },
                }
            except Exception as exc:  # API failures are retried and recorded by caller.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error


def candidate_key(item_key: str, interval: Sequence[float]) -> str:
    return f"{item_key}__w{float(interval[0]):.3f}_{float(interval[1]):.3f}"


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.frame_width <= 0:
        raise SystemExit("--fps and --frame-width must be positive")
    if args.max_windows_per_item < 0:
        raise SystemExit("--max-windows-per-item must be non-negative")
    require_binaries(args.ffmpeg, args.ffprobe)
    endpoints = parse_endpoints(args.endpoint)
    if len(endpoints) != 1:
        raise SystemExit("grounding requires exactly one --endpoint")
    grounder = WindowGrounder(
        endpoints[0], timeout=args.timeout, max_tokens=args.max_tokens, retries=args.api_retries
    )

    measurement_paths = [Path(path) for path in glob.glob(args.measurements)]
    if not measurement_paths:
        raise SystemExit(f"no measurement files match {args.measurements!r}")
    measurements = latest_jsonl_records(measurement_paths)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    items = load_worklist(args.filtered_json, args.video_dir, args.video_id_mapping)

    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(items), shard, args.num_shards, args.limit)
    output = args.output
    if shard is not None:
        output = output.with_name(f"{output.stem}.{shard}{output.suffix}")
    previous = latest_jsonl_records([output]) if output.exists() else {}
    retry_statuses = {part.strip() for part in args.retry_status.split(",") if part.strip()}

    print(
        f"[START] items={len(items)} selected={len(indices)} shard={shard}/{args.num_shards} "
        f"fps={args.fps} model={endpoints[0].tag} output={output}",
        flush=True,
    )
    written = grounded = errors = 0
    for position, index in enumerate(indices, start=1):
        item = items[index]
        measurement = measurements.get(item.key)
        if not measurement or measurement.get("status") != "ok":
            continue
        candidates = candidate_windows(measurement, thresholds, args.uncertainty_band)
        if args.max_windows_per_item:
            candidates = candidates[:args.max_windows_per_item]
        pending = [
            candidate for candidate in candidates
            if args.overwrite
            or candidate_key(item.key, candidate["interval"]) not in previous
            or previous[candidate_key(item.key, candidate["interval"])].get("status") in retry_statuses
        ]
        if not pending:
            print(f"[{position}/{len(indices)}] {item.key} skip candidates={len(candidates)}", flush=True)
            continue

        holder = None
        try:
            holder, frames = extract_frame_grid(
                item.video_path,
                fps=args.fps,
                width=args.frame_width,
                ffmpeg=args.ffmpeg,
                temp_root=args.temp_root,
            )
            mask_intervals = measurement.get("initial_mask") or measurement.get("official_evidence") or item.evidence
            masked_frames = mask_frames(frames, mask_intervals, blur_sigma=args.blur_sigma)
            for rank, candidate in enumerate(candidates, start=1):
                key = candidate_key(item.key, candidate["interval"])
                if candidate not in pending:
                    continue
                started = time.time()
                base = {
                    "key": key,
                    "item_key": item.key,
                    "item_index": item.index,
                    "video_id": item.video_id,
                    "qid": item.qid,
                    "question": item.question,
                    "question_type": item.question_type,
                    "reference_answer": (
                        item.choices[ord(item.gold) - ord("A")] if item.gold in tuple("ABCDE") else None
                    ),
                    "candidate_rank": rank,
                    "candidate": candidate,
                    "official_evidence": [list(span) for span in item.evidence],
                    "masked_intervals": [list(span) for span in mask_intervals],
                    "sampling": {"fps": args.fps, "frame_width": args.frame_width},
                    "masking": {"mode": "blurred_safe_frame_or_gray", "blur_sigma": args.blur_sigma},
                    "prompt_version": GROUNDING_PROMPT_VERSION,
                    "grounding_input_policy": "question_and_type_only_no_choices_no_gold",
                }
                try:
                    start, end = candidate["interval"]
                    window_frames = frames_in_range(masked_frames, float(start), float(end))
                    result = grounder.describe(window_frames, item.question, item.question_type)
                    record = {
                        **base,
                        "status": "ok",
                        "grounding": result,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    grounded += 1
                except Exception as exc:
                    record = {
                        **base,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                    errors += 1
                append_jsonl(output, record)
                previous[key] = record
                written += 1
                print(
                    f"[{position}/{len(indices)}] {key} {record['status']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            print(f"[{position}/{len(indices)}] {item.key} ITEM_ERROR {type(exc).__name__}: {exc}", flush=True)
            errors += len(pending)
        finally:
            if holder is not None:
                holder.cleanup()

    print(f"[DONE] wrote={written} grounded={grounded} errors={errors} output={output}", flush=True)


if __name__ == "__main__":
    main()

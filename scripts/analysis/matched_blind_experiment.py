#!/usr/bin/env python3
"""Evaluate matched-gray visual blindness with abstention and forced-choice prompts."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image

from qwen25_conservatism_experiment import (
    extract,
    forced_choice_prompt,
    image_content,
    selective_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-video-dir", type=Path, default=Path("source_datasets/next_gqa/videos"))
    parser.add_argument("--insufficient-video-dir", type=Path, default=Path("source_datasets/next_gqa/freeze_videos_no_evidence_leak"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def gray_like(encoded: str) -> str:
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
        image = Image.new("RGB", source.size, (127, 127, 127))
    holder = io.BytesIO()
    image.save(holder, format="JPEG", quality=95)
    return base64.b64encode(holder.getvalue()).decode()


def call(client: OpenAI, model: str, content: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return {
        "text": response.choices[0].message.content,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def prompts(row: dict) -> dict[str, str]:
    return {
        "blind_classify": selective_prompt(row),
        "blind_forced_choice": forced_choice_prompt(row),
    }


def run_item(row: dict, args: argparse.Namespace, completed: set[str]) -> list[dict]:
    item_key = f"{row['video_id']}_q{row['qid']}"
    pending = [name for name in ("blind_classify", "blind_forced_choice") if f"{item_key}::{name}" not in completed]
    if not pending:
        return []
    original_frames = extract(row, args)
    blind_frames = [gray_like(frame) for frame in original_frames]
    unique_sizes = []
    for encoded in blind_frames:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            if list(image.size) not in unique_sizes:
                unique_sizes.append(list(image.size))
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    item_prompts = prompts(row)
    records = []
    for condition in pending:
        content = image_content(blind_frames, row["sampled_seconds"])
        content.append({"type": "text", "text": item_prompts[condition]})
        records.append({
            "key": f"{item_key}::{condition}",
            "condition": condition,
            "model_tag": args.model_tag,
            "video_id": row["video_id"],
            "qid": row["qid"],
            "question_type": row.get("question_type"),
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "experiment_group": row.get("experiment_group"),
            "video_kind": row.get("video_kind"),
            "sampled_seconds": row["sampled_seconds"],
            "blind_policy": "same_frame_count_timestamps_and_preprocessor_dimensions_uniform_gray_127",
            "input_image_sizes": unique_sizes,
            "response": call(client, args.model, content),
            "status": "ok",
        })
    return records


def main() -> None:
    args = parse_args()
    rows = json.loads(args.sample.read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[:args.limit]
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line).get("key")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_item, row, args, completed): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                records = future.result()
            except Exception as exc:
                records = [{
                    "key": f"{row['video_id']}_q{row['qid']}::ITEM_ERROR",
                    "model_tag": args.model_tag,
                    "video_id": row["video_id"],
                    "qid": row["qid"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }]
            with lock, args.output.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["video_id"], row["qid"], [r.get("condition", r["status"]) for r in records], flush=True)


if __name__ == "__main__":
    main()

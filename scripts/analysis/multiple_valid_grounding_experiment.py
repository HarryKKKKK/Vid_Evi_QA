#!/usr/bin/env python3
"""Run complete answer-set and per-option grounding evaluation."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--frame-width", type=int, default=560)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--media-mode", choices=("frames", "video"), default="frames")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--limit", type=int)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--extra-body-json", default="{}")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt(row: dict) -> str:
    options = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(row["choices"]))
    return f"""Identify every answer option that is independently supported by visible evidence in the sampled video frames.

More than one option may be valid, and valid events may occur at different times. Evaluate each option separately. Do not force a single answer and do not select an option merely because its wording is plausible. For every selected option, report the timestamp interval or intervals that visually support it.

Return ONLY one valid JSON object:
{{
  "supported_options": [
    {{
      "option_id": "A",
      "evidence_seconds": [[0.0, 1.0]],
      "description": "brief visible support"
    }}
  ],
  "unsupported_option_ids": ["B"],
  "reason": "one concise summary"
}}

Question: {row['question']}

Options:
{options}"""


def timestamps(row: dict, count: int) -> list[float]:
    duration = float(row["video_duration"])
    return [(i + 0.5) * duration / count for i in range(count)]


def extract(video: Path, times: list[float], width: int, ffmpeg: str) -> list[str]:
    frames = []
    with tempfile.TemporaryDirectory(prefix="multiple_valid_") as holder:
        root = Path(holder)
        for index, timestamp in enumerate(times):
            output = root / f"{index:03d}.jpg"
            subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                            "-ss", str(timestamp), "-i", str(video), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", str(output)], check=True)
            frames.append(base64.b64encode(output.read_bytes()).decode())
    return frames


def content(frames: list[str], times: list[float], instruction: str,
            media_mode: str) -> list[dict]:
    if media_mode == "video":
        source_times = ", ".join(f"{timestamp:.2f}" for timestamp in times)
        return [
            {"type": "video_url", "video_url": {
                "url": "data:video/jpeg;base64," + ",".join(frames)}},
            {"type": "text", "text":
                f"The video contains ordered frames sampled at source timestamps [{source_times}] seconds.\n\n{instruction}"},
        ]
    result = []
    for frame, timestamp in zip(frames, times):
        result.extend(({"type": "text", "text": f"Frame at {timestamp:.2f}s:"},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame}}))
    result.append({"type": "text", "text": instruction})
    return result


def run(row: dict, args: argparse.Namespace, completed: set[str]) -> dict | None:
    key = str(row["id"])
    if key in completed:
        return None
    video = args.video_root / row["video"]
    if not video.is_file():
        raise FileNotFoundError(video)
    times = timestamps(row, args.num_frames)
    frames = extract(video, times, args.frame_width, args.ffmpeg)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    kwargs = dict(model=args.model,
                  messages=[{"role": "user", "content": content(frames, times, prompt(row), args.media_mode)}],
                  temperature=0, max_tokens=1000,
                  response_format={"type": "json_object"})
    extra_body = json.loads(args.extra_body_json)
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return {
        "key": key, "status": "ok", "model_tag": args.model_tag,
        "id": row["id"], "parent_job_id": row.get("parent_job_id"),
        "video": row["video"], "video_id": Path(row["video"]).stem,
        "conflict_type": row.get("conflict_type"), "question": row["question"],
        "choices": row["choices"], "options": row["options"],
        "valid_option_ids": row["valid_option_ids"], "video_duration": row["video_duration"],
        "sampled_seconds": times,
        "response": {"text": response.choices[0].message.content,
                     "usage": response.usage.model_dump() if response.usage else None},
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
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
        futures = {pool.submit(run, row, args, completed): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"key": str(row["id"]), "status": "error", "model_tag": args.model_tag,
                          "id": row["id"], "video": row.get("video"),
                          "error": f"{type(exc).__name__}: {exc}"}
            if record is None:
                continue
            with lock, args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(record["key"], record["status"], flush=True)


if __name__ == "__main__":
    main()

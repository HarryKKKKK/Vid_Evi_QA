#!/usr/bin/env python3
"""Matched C3-operator and non-evidence-placebo evaluation on fixed timestamps."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageFilter

from qwen25_conservatism_experiment import image_content, selective_prompt


CONDITIONS = (
    "original",
    "evidence_noise_a000",
    "evidence_freeze",
    "evidence_gray",
    "placebo_noise",
    "placebo_freeze",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("results/build/nextgqa/nextgqa_filtered_no_evidence_leak_eval_clean.json"))
    parser.add_argument("--mapping", type=Path, default=Path("source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"))
    parser.add_argument("--source-video-dir", type=Path, default=Path("source_datasets/next_gqa/videos"))
    parser.add_argument("--noise-video-dir", type=Path, default=Path("source_datasets/next_gqa/partial_videos_no_evidence_leak"))
    parser.add_argument("--freeze-video-dir", type=Path, default=Path("source_datasets/next_gqa/freeze_videos_no_evidence_leak"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def intervals(row: dict) -> list[tuple[float, float]]:
    spans = []
    for value in row.get("evidence_intervals", []):
        start, end = float(value["start"]), float(value["end"])
        spans.append((min(start, end), max(start, end)))
    return sorted(spans)


def in_spans(timestamp: float, spans: list[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in spans)


def placebo_span(spans: list[tuple[float, float]], duration: float) -> tuple[float, float]:
    target = min(sum(end - start for start, end in spans), duration)
    boundaries = [(0.0, spans[0][0])] if spans else [(0.0, duration)]
    for (_, left_end), (right_start, _) in zip(spans, spans[1:]):
        boundaries.append((left_end, right_start))
    if spans:
        boundaries.append((spans[-1][1], duration))
    candidates = [(max(0.0, end - start), start, end) for start, end in boundaries]
    _, start, end = max(candidates)
    length = min(target, max(0.0, end - start))
    return start, start + length


def resolve_source(row: dict, mapping: dict[str, str], root: Path) -> Path:
    mapped = mapping.get(str(row["video_id"]))
    if mapped is None:
        raise FileNotFoundError(f"mapping missing for {row['video_id']}")
    path = root / f"{mapped}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_frames(video: Path, times: list[float], width: int, ffmpeg: str) -> list[str]:
    frames = []
    with tempfile.TemporaryDirectory(prefix="operator_control_") as holder:
        root = Path(holder)
        for index, timestamp in enumerate(times):
            output = root / f"{index:03d}.jpg"
            subprocess.run([
                ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                "-ss", str(timestamp), "-i", str(video), "-frames:v", "1",
                "-vf", f"scale={width}:-2", "-q:v", "3", str(output),
            ], check=True)
            frames.append(base64.b64encode(output.read_bytes()).decode())
    return frames


def transform_frame(encoded: str, mode: str, seed: str, freeze_encoded: str | None = None) -> str:
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
        size = source.size
    if mode == "gray":
        target = Image.new("RGB", size, (127, 127, 127))
    elif mode == "noise":
        rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
        target = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
    elif mode == "freeze":
        if freeze_encoded is None:
            raise ValueError("freeze transform needs a reference frame")
        with Image.open(io.BytesIO(base64.b64decode(freeze_encoded))) as reference:
            target = reference.convert("RGB").resize(size).filter(ImageFilter.GaussianBlur(radius=30))
    else:
        raise ValueError(mode)
    holder = io.BytesIO()
    target.save(holder, format="JPEG", quality=95)
    return base64.b64encode(holder.getvalue()).decode()


def transformed(
    frames: list[str], times: list[float], span: tuple[float, float], mode: str, item_key: str
) -> list[str]:
    affected = [index for index, timestamp in enumerate(times) if span[0] <= timestamp <= span[1]]
    if not affected:
        return list(frames)
    preceding = [index for index, timestamp in enumerate(times) if timestamp < span[0]]
    reference = frames[preceding[-1]] if preceding else frames[0]
    output = list(frames)
    for index in affected:
        output[index] = transform_frame(
            frames[index], mode, f"{item_key}:{times[index]:.6f}:{mode}", reference
        )
    return output


def transformed_spans(
    frames: list[str], times: list[float], spans: list[tuple[float, float]], mode: str, item_key: str
) -> list[str]:
    output = list(frames)
    for span in spans:
        output = transformed(output, times, span, mode, item_key)
    return output


def classify_prompt(row: dict) -> str:
    return selective_prompt(row)


def call(client: OpenAI, model: str, frames: list[str], row: dict) -> dict:
    content = image_content(frames, row["sampled_seconds"])
    content.append({"type": "text", "text": classify_prompt(row)})
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        temperature=0, max_tokens=500, response_format={"type": "json_object"},
    )
    return {
        "text": response.choices[0].message.content,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def run_item(row: dict, meta: dict, mapping: dict, args: argparse.Namespace, completed: set[str]) -> list[dict]:
    item = f"{row['video_id']}_q{row['qid']}"
    pending = [condition for condition in CONDITIONS if f"{item}::{condition}" not in completed]
    if not pending:
        return []
    item_meta = meta.get(key(row))
    if item_meta is None:
        raise KeyError(f"metadata missing {item}")
    spans = intervals(item_meta)
    if not spans:
        raise ValueError(f"no evidence intervals for {item}")
    times = [float(value) for value in row["sampled_seconds"]]
    source = resolve_source(row, mapping, args.source_video_dir)
    original = extract_frames(source, times, args.frame_width, args.ffmpeg)
    placebo = placebo_span(spans, max(times) if times else 0.0)
    condition_frames = {"original": original}
    if "evidence_noise_a000" in pending:
        path = args.noise_video_dir / f"{row['video_id']}_q{row['qid']}_c2_noise_a000.mp4"
        condition_frames["evidence_noise_a000"] = extract_frames(path, times, args.frame_width, args.ffmpeg)
    if "evidence_freeze" in pending:
        path = args.freeze_video_dir / f"{row['video_id']}_q{row['qid']}_freeze.mp4"
        condition_frames["evidence_freeze"] = extract_frames(path, times, args.frame_width, args.ffmpeg)
    condition_frames["evidence_gray"] = transformed_spans(original, times, spans, "gray", item)
    condition_frames["placebo_noise"] = transformed(original, times, placebo, "noise", item)
    condition_frames["placebo_freeze"] = transformed(original, times, placebo, "freeze", item)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    records = []
    for condition in pending:
        records.append({
            "key": f"{item}::{condition}", "condition": condition,
            "model_tag": args.model_tag, "video_id": row["video_id"], "qid": row["qid"],
            "question_type": row.get("question_type"), "question": row["question"],
            "choices": row["choices"], "answer": row["answer"],
            "experiment_group": row.get("experiment_group"),
            "sampled_seconds": times, "evidence_intervals": [list(span) for span in spans],
            "placebo_interval": list(placebo),
            "response": call(client, args.model, condition_frames[condition], row),
            "status": "ok",
        })
    return records


def main() -> None:
    args = parse_args()
    rows = [row for row in json.loads(args.sample.read_text(encoding="utf-8")) if row.get("video_kind") == "source"]
    if args.limit is not None:
        rows = rows[:args.limit]
    meta = {key(row): row for row in json.loads(args.metadata.read_text(encoding="utf-8"))}
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if args.preflight:
        counts = {"rows": len(rows), "missing_metadata": 0, "missing_source": 0,
                  "missing_noise": 0, "missing_freeze": 0, "no_evidence": 0}
        examples: list[str] = []
        for row in rows:
            item = f"{row['video_id']}_q{row['qid']}"
            item_meta = meta.get(key(row))
            if item_meta is None:
                counts["missing_metadata"] += 1; examples.append(f"metadata:{item}"); continue
            if not intervals(item_meta): counts["no_evidence"] += 1; examples.append(f"evidence:{item}")
            try: resolve_source(row, mapping, args.source_video_dir)
            except FileNotFoundError: counts["missing_source"] += 1; examples.append(f"source:{item}")
            if not (args.noise_video_dir / f"{item}_c2_noise_a000.mp4").is_file():
                counts["missing_noise"] += 1; examples.append(f"noise:{item}")
            if not (args.freeze_video_dir / f"{item}_freeze.mp4").is_file():
                counts["missing_freeze"] += 1; examples.append(f"freeze:{item}")
        print(json.dumps({**counts, "examples": examples[:20]}, ensure_ascii=False, indent=2))
        return
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line).get("key")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_item, row, meta, mapping, args, completed): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                records = future.result()
            except Exception as exc:
                records = [{"key": f"{row['video_id']}_q{row['qid']}::ITEM_ERROR", "model_tag": args.model_tag, "video_id": row["video_id"], "qid": row["qid"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}]
            with lock, args.output.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["video_id"], row["qid"], [r.get("condition", r["status"]) for r in records], flush=True)


if __name__ == "__main__":
    main()

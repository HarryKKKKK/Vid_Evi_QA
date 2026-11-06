#!/usr/bin/env python3
"""Question-level seeded Partial controls on fixed sampled frames.

For every source-video QA, the experiment keeps the original 32 timestamps and
constructs matched interventions in memory.  The same deterministic noise field
is reused across alpha levels and across evidence/non-evidence locations.  The
non-evidence control affects exactly the same number of sampled frames as the
evidence intervention.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image

from qwen25_conservatism_experiment import (
    forced_choice_prompt,
    image_content,
    selective_prompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=Path(
        "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak_eval_clean.json"))
    p.add_argument("--mapping", type=Path, default=Path(
        "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"))
    p.add_argument("--source-video-dir", type=Path, default=Path("source_datasets/next_gqa/videos"))
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alphas", default="0.2,0.4,0.5,0.6")
    p.add_argument("--seeds", default="17,29,43")
    p.add_argument("--protocol", choices=("selective", "forced"), default="selective")
    p.add_argument("--media-mode", choices=("frames", "video"), default="frames")
    p.add_argument("--forced-alphas", default="0.2,0.6")
    p.add_argument("--forced-seed", type=int, default=17)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--frame-width", type=int, default=560)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--extra-body-json", default="{}",
                   help="Extra OpenAI-compatible request body, e.g. chat template kwargs.")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def row_key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def spans(row: dict) -> list[tuple[float, float]]:
    result = []
    for value in row.get("evidence_intervals", []):
        start, end = float(value["start"]), float(value["end"])
        result.append((min(start, end), max(start, end)))
    return sorted(result)


def resolve_video(row: dict, mapping: dict[str, str], root: Path) -> Path:
    mapped = mapping.get(str(row["video_id"]))
    if mapped is None:
        raise FileNotFoundError(f"mapping missing: {row['video_id']}")
    path = root / f"{mapped}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_frames(video: Path, times: list[float], width: int, ffmpeg: str) -> list[str]:
    frames = []
    with tempfile.TemporaryDirectory(prefix="partial_seeded_") as holder:
        root = Path(holder)
        for index, timestamp in enumerate(times):
            output = root / f"{index:03d}.jpg"
            subprocess.run([
                ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", str(timestamp),
                "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2",
                "-q:v", "3", str(output),
            ], check=True)
            frames.append(base64.b64encode(output.read_bytes()).decode())
    return frames


def evidence_indices(times: list[float], evidence_spans: list[tuple[float, float]]) -> list[int]:
    return [i for i, timestamp in enumerate(times)
            if any(start <= timestamp <= end for start, end in evidence_spans)]


def matched_non_evidence_indices(total: int, evidence: list[int]) -> tuple[list[int], str]:
    """Select an equal-sized control set, preferring one contiguous sampled-frame block."""
    wanted = len(evidence)
    blocked = set(evidence)
    available = [i for i in range(total) if i not in blocked]
    if wanted == 0 or len(available) < wanted:
        return [], "unavailable"
    center = sum(evidence) / wanted
    candidates = []
    for start in range(0, total - wanted + 1):
        block = list(range(start, start + wanted))
        if not blocked.intersection(block):
            candidates.append(block)
    if candidates:
        chosen = min(candidates, key=lambda x: (abs(sum(x) / len(x) - center), x[0]))
        return chosen, "contiguous_equal_frame_count"
    chosen = sorted(available, key=lambda i: (abs(i - center), i))[:wanted]
    return sorted(chosen), "nearest_equal_frame_count"


def random_noise(size: tuple[int, int], seed_text: str) -> Image.Image:
    value = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = random.Random(value)
    return Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))


def blend_indices(frames: list[str], indices: list[int], alpha: float,
                  item: str, seed: int) -> list[str]:
    output = list(frames)
    for rank, index in enumerate(indices):
        raw = base64.b64decode(frames[index])
        with Image.open(io.BytesIO(raw)) as image:
            original = image.convert("RGB")
        noise = random_noise(original.size, f"{item}:seed={seed}:rank={rank}")
        blended = Image.blend(noise, original, alpha)
        holder = io.BytesIO()
        blended.save(holder, format="JPEG", quality=95)
        output[index] = base64.b64encode(holder.getvalue()).decode()
    return output


def response_key(item: str, protocol: str, location: str, alpha: float | None,
                 seed: int | None) -> str:
    if location == "original":
        return f"{item}::{protocol}::original"
    return f"{item}::{protocol}::{location}::a{int(round(alpha * 100)):03d}::s{seed}"


def call(client: OpenAI, model: str, frames: list[str], times: list[float],
         row: dict, protocol: str, media_mode: str, extra_body: dict) -> dict:
    prompt = selective_prompt(row) if protocol == "selective" else forced_choice_prompt(row)
    if media_mode == "video":
        source_times = ", ".join(f"{timestamp:.2f}" for timestamp in times)
        content = [
            {"type": "video_url", "video_url": {
                "url": "data:video/jpeg;base64," + ",".join(frames)}},
            {"type": "text", "text":
                f"The video contains ordered frames sampled at source timestamps [{source_times}] seconds.\n\n{prompt}"},
        ]
    else:
        content = image_content(frames, times)
        content.append({"type": "text", "text": prompt})
    kwargs = dict(model=model, messages=[{"role": "user", "content": content}],
                  temperature=0, max_tokens=500)
    if protocol == "selective":
        kwargs["response_format"] = {"type": "json_object"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return {"text": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None}


def run_item(row: dict, meta: dict, mapping: dict[str, str], args: argparse.Namespace,
             completed: set[str]) -> list[dict]:
    item = f"{row['video_id']}_q{row['qid']}"
    times = [float(x) for x in row["sampled_seconds"]]
    item_meta = meta.get(row_key(row), row)
    if item_meta is None:
        raise KeyError(f"metadata missing: {item}")
    evidence_spans = spans(item_meta)
    evid = evidence_indices(times, evidence_spans)
    control, control_method = matched_non_evidence_indices(len(times), evid)
    if not evid or not control:
        raise ValueError(f"cannot create matched intervention: evidence={evid}, control={control}")
    video = resolve_video(row, mapping, args.source_video_dir)
    original = extract_frames(video, times, args.frame_width, args.ffmpeg)
    if args.protocol == "forced":
        alphas, seeds = [float(x) for x in args.forced_alphas.split(",")], [args.forced_seed]
    else:
        alphas = [float(x) for x in args.alphas.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
    plan: list[tuple[str, float | None, int | None, list[str]]] = [
        ("original", None, None, original)
    ]
    for seed in seeds:
        for alpha in alphas:
            plan.append(("evidence", alpha, seed,
                         blend_indices(original, evid, alpha, item, seed)))
            plan.append(("non_evidence", alpha, seed,
                         blend_indices(original, control, alpha, item, seed)))
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    extra_body = json.loads(args.extra_body_json)
    records = []
    for location, alpha, seed, frames in plan:
        key = response_key(item, args.protocol, location, alpha, seed)
        if key in completed:
            continue
        records.append({
            "key": key, "status": "ok", "protocol": args.protocol,
            "condition": location, "alpha": alpha, "noise_seed": seed,
            "model_tag": args.model_tag, "video_id": str(row["video_id"]),
            "qid": str(row["qid"]), "question_type": row.get("question_type"),
            "question": row["question"], "choices": row["choices"], "answer": row["answer"],
            "sampled_seconds": times, "evidence_intervals": [list(x) for x in evidence_spans],
            "evidence_frame_indices": evid, "control_frame_indices": control,
            "control_matching": control_method,
            "sampling_stratum": row.get("sampling_stratum"),
            "sampling_weight": row.get("sampling_weight", 1.0),
            "inclusion_probability": row.get("inclusion_probability", 1.0),
            "response": call(client, args.model, frames, times, row, args.protocol,
                             args.media_mode, extra_body),
        })
    return records


def select_eligible(rows: list[dict], meta: dict, mapping: dict[str, str],
                    args: argparse.Namespace) -> tuple[list[dict], dict, list[str]]:
    counts = {"scanned": 0, "selected": 0, "missing_metadata": 0, "missing_video": 0,
              "no_sampled_evidence": 0, "no_matched_control": 0}
    examples = []
    yield_rows = []
    for row in rows:
        if counts["selected"] >= args.limit:
            break
        counts["scanned"] += 1
        item = f"{row['video_id']}_q{row['qid']}"
        item_meta = meta.get(row_key(row), row)
        if not item_meta.get("evidence_intervals"):
            counts["missing_metadata"] += 1; examples.append(f"metadata:{item}"); continue
        evid = evidence_indices([float(x) for x in row["sampled_seconds"]], spans(item_meta))
        if not evid:
            counts["no_sampled_evidence"] += 1; examples.append(f"evidence:{item}"); continue
        control, _ = matched_non_evidence_indices(len(row["sampled_seconds"]), evid)
        if not control:
            counts["no_matched_control"] += 1; examples.append(f"control:{item}"); continue
        try:
            resolve_video(row, mapping, args.source_video_dir)
        except FileNotFoundError:
            counts["missing_video"] += 1; examples.append(f"video:{item}"); continue
        counts["selected"] += 1
        yield_rows.append(row)
    return yield_rows, counts, examples


def preflight(rows: list[dict], counts: dict, examples: list[str], args: argparse.Namespace) -> None:
    print(json.dumps({**counts, "examples": examples[:30]}, ensure_ascii=False, indent=2))
    if len(rows) < args.limit:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    all_rows = [row for row in json.loads(args.sample.read_text(encoding="utf-8"))
                if row.get("video_kind") == "source"]
    meta = {row_key(row): row for row in json.loads(args.metadata.read_text(encoding="utf-8"))}
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows, selection_counts, selection_examples = select_eligible(
        all_rows, meta, mapping, args)
    if args.preflight:
        preflight(rows, selection_counts, selection_examples, args)
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
                records = [{"key": f"{row['video_id']}_q{row['qid']}::ITEM_ERROR",
                            "model_tag": args.model_tag, "video_id": str(row["video_id"]),
                            "qid": str(row["qid"]), "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"}]
            with lock, args.output.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["video_id"], row["qid"], len(records), flush=True)


if __name__ == "__main__":
    main()

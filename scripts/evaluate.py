#!/usr/bin/env python3
"""
CG-Bench evaluation pipeline (concurrent).

Task modes:
  --task qa
      Pure multiple-choice QA. The model must choose one option.

  --task classify
      Answerability classification:
      ANSWERABLE / PARTIALLY_ANSWERABLE / UNANSWERABLE.

Incremental saving:
  Each finished result is immediately appended to a .jsonl file.
  At the end, an ordered .json file is also written for compatibility.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# -- Hyperparameters ----------------------------------------------------------

NUM_FRAMES = 32
FRAME_WIDTH = 560
EVIDENCE_FRACTION = 0.25
WORKERS = 8


# -- Paths --------------------------------------------------------------------

BASE_DIR = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
FILTERED_JSON = BASE_DIR / "cgbench_pipeline" / "cgbench_filtered.json"
VIDEOS_DIR = BASE_DIR / "source_datasets" / "cg_bench" / "videos"
INSUFFICIENT_DIR = BASE_DIR / "source_datasets" / "cg_bench" / "insufficient_videos"
RESULTS_DIR = BASE_DIR / "cgbench_result"


# -- LLM client ---------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL", "qwen3-vl")

_client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    max_retries=0,
    timeout=120.0,
)


# -- Timestamp planning -------------------------------------------------------

def _video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def plan_timestamps(
    duration: float,
    num_frames: int,
    *,
    sampling: str = "uniform",
    evidence_intervals: list[dict] | None = None,
    evidence_fraction: float = EVIDENCE_FRACTION,
) -> list[float]:
    if sampling == "evidence" and evidence_intervals:
        n_evidence = max(1, int(round(num_frames * evidence_fraction)))
        n_global = max(0, num_frames - n_evidence)

        timestamps: list[float] = []
        total_ev = sum(iv["end"] - iv["start"] for iv in evidence_intervals)

        if total_ev <= 0:
            for iv in evidence_intervals:
                timestamps.append(iv["start"])
        else:
            for iv in evidence_intervals:
                seg = iv["end"] - iv["start"]
                k = max(1, round(n_evidence * seg / total_ev))
                for j in range(k):
                    timestamps.append(iv["start"] + (j + 0.5) * seg / k)

        for i in range(n_global):
            timestamps.append((i + 0.5) * duration / max(1, n_global))

        timestamps = sorted({round(t, 2) for t in timestamps})
        timestamps = [
            min(max(t, 0.0), max(duration - 0.01, 0.0))
            for t in timestamps
        ]
        return timestamps

    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    frame_width: int,
) -> list[str]:
    frames_b64 = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")

            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y",
                    "-ss", str(ts),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", f"scale={frame_width}:-2",
                    "-q:v", "3",
                    frame_path,
                ],
                capture_output=True,
                check=True,
            )

            with open(frame_path, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode())

    return frames_b64


def sample_frames(
    video_path: Path,
    num_frames: int,
    frame_width: int = FRAME_WIDTH,
    *,
    sampling: str = "uniform",
    evidence_intervals: list[dict] | None = None,
    evidence_fraction: float = EVIDENCE_FRACTION,
) -> tuple[list[str], list[float]]:
    duration = _video_duration(video_path)

    timestamps = plan_timestamps(
        duration,
        num_frames,
        sampling=sampling,
        evidence_intervals=evidence_intervals,
        evidence_fraction=evidence_fraction,
    )

    frames_b64 = extract_frames(video_path, timestamps, frame_width)
    return frames_b64, timestamps


# -- Prompt construction ------------------------------------------------------

def build_instruction(
    question: str,
    choices: list[str],
    *,
    task: str = "qa",
) -> str:
    options_block = ""

    if choices:
        labels = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(choices)
        )
        options_block = f"\n\nOptions:\n{labels}"

    if task == "qa":
        return f"""You are answering a multiple-choice question about a video.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Choose the best answer based on the visual evidence in the frames.

Important rules:
- You MUST choose exactly one option.
- Use the timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "answer": "the option letter, e.g. A",
  "evidence": "brief visual evidence supporting the answer",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "one short sentence"
}}

Question: {question}{options_block}"""

    if task == "classify":
        return f"""You are determining whether the sampled video frames contain sufficient visual evidence to answer a multiple-choice question.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Your task is to answer the question if it is answerable, otherwise output UNANSWERABLE.

Important rules:
- If the frames contain sufficient evidence, choose the best-supported option.
- If the frames do not contain sufficient visual evidence to determine a unique answer, output UNANSWERABLE.
- Use the timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "label": "ANSWERABLE | UNANSWERABLE",
  "answer": "the option letter if ANSWERABLE, otherwise null",
  "evidence": "brief visual evidence supporting your decision",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "one short sentence"
}}

Question: {question}{options_block}"""

    raise ValueError(f"Unknown task: {task}")


def print_prompt_preview(task: str) -> None:
    demo_question = "What is the main action shown in the video?"
    demo_choices = [
        "A person is cooking.",
        "A person is driving.",
        "A person is playing music.",
        "A person is reading.",
    ]

    prompt = build_instruction(
        demo_question,
        demo_choices,
        task=task,
    )

    print("\n" + "=" * 80)
    print(f"PROMPT PREVIEW task={task}")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n", flush=True)


# -- LLM call -----------------------------------------------------------------

def call_llm(
    frames_b64: list[str],
    timestamps: list[float],
    question: str,
    choices: list[str],
    *,
    task: str = "qa",
) -> str:
    content = []

    for b64, ts in zip(frames_b64, timestamps):
        content.append({
            "type": "text",
            "text": f"Frame at {ts:.1f}s:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })

    content.append({
        "type": "text",
        "text": build_instruction(question, choices, task=task),
    })

    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )

    return resp.choices[0].message.content


# -- Output parsing -----------------------------------------------------------

def parse_model_json(raw: str) -> dict:
    out = {"raw": raw}

    if raw is None:
        out["parse_error"] = "empty response"
        return out

    cleaned = raw.strip()
    cleaned = re.sub(
        r"^```(?:json)?|```$",
        "",
        cleaned,
        flags=re.MULTILINE,
    ).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned

    try:
        parsed = json.loads(candidate)

        if isinstance(parsed, dict):
            out.update(parsed)
        else:
            out["parse_error"] = "JSON was not an object"

    except json.JSONDecodeError as e:
        out["parse_error"] = f"invalid JSON: {e}"

    return out


# -- Per-entry worker ---------------------------------------------------------

def process_entry(
    idx: int,
    entry: dict,
    video_dir: Path,
    *,
    insufficient: bool,
    num_frames: int,
    frame_width: int,
    sampling: str,
    evidence_fraction: float,
    task: str,
) -> dict:
    video_id = entry["video_id"]
    qid = entry["qid"]

    if insufficient:
        video_file = video_dir / f"{video_id}_q{qid}_insufficient.mp4"
    else:
        video_file = video_dir / f"{video_id}.mp4"

    if not video_file.exists():
        return {
            "idx": idx,
            "status": "skip",
            "video_id": video_id,
            "qid": qid,
            "reason": "missing_video",
        }

    ev_intervals = entry.get("evidence_intervals") if not insufficient else None

    try:
        frames, timestamps = sample_frames(
            video_file,
            num_frames,
            frame_width,
            sampling=sampling,
            evidence_intervals=ev_intervals,
            evidence_fraction=evidence_fraction,
        )

        raw_response = call_llm(
            frames,
            timestamps,
            entry["question"],
            entry.get("choices", []),
            task=task,
        )

    except Exception as e:
        return {
            "idx": idx,
            "status": "error",
            "video_id": video_id,
            "qid": qid,
            "reason": str(e),
        }

    parsed = parse_model_json(raw_response)

    result = {
        **entry,
        "task": task,
        "sampling": sampling,
        "sampled_seconds": [round(t, 2) for t in timestamps],
        "model_response": raw_response,
        "model_parsed": parsed,
    }

    if task == "qa":
        short_status = parsed.get("answer", "?")
    else:
        short_status = parsed.get("label", "?")

    return {
        "idx": idx,
        "status": "ok",
        "video_id": video_id,
        "qid": qid,
        "result": result,
        "short_status": short_status,
    }


# -- Evaluation loop ----------------------------------------------------------

def append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_evaluation(
    entries: list[dict],
    video_dir: Path,
    output_path: Path,
    *,
    insufficient: bool = False,
    num_frames: int = NUM_FRAMES,
    frame_width: int = FRAME_WIDTH,
    sampling: str = "uniform",
    evidence_fraction: float = EVIDENCE_FRACTION,
    workers: int = WORKERS,
    task: str = "qa",
) -> None:
    total = len(entries)
    done = 0
    print_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path.with_suffix(".jsonl")
    skipped_jsonl_path = output_path.with_suffix(".skipped.jsonl")

    if jsonl_path.exists():
        jsonl_path.unlink()
    if skipped_jsonl_path.exists():
        skipped_jsonl_path.unlink()

    results_by_idx: dict[int, dict] = {}
    skipped = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_entry,
                idx,
                entry,
                video_dir,
                insufficient=insufficient,
                num_frames=num_frames,
                frame_width=frame_width,
                sampling=sampling,
                evidence_fraction=evidence_fraction,
                task=task,
            ): idx
            for idx, entry in enumerate(entries)
        }

        for fut in as_completed(futures):
            out = fut.result()
            done += 1

            vid = out["video_id"]
            qid = out["qid"]

            if out["status"] == "ok":
                result = out["result"]
                results_by_idx[out["idx"]] = result

                append_jsonl(jsonl_path, result)

                msg = (
                    f"[{done}/{total}] OK    {vid} qid={qid}  "
                    f"{task}={out['short_status']}  "
                    f"saved_to={jsonl_path.name}"
                )

            elif out["status"] == "skip":
                skip_obj = {
                    "idx": out["idx"],
                    "video_id": vid,
                    "qid": qid,
                    "reason": out["reason"],
                }

                skipped.append(skip_obj)
                append_jsonl(skipped_jsonl_path, skip_obj)

                msg = f"[{done}/{total}] SKIP  missing video: {vid} qid={qid}"

            else:
                skip_obj = {
                    "idx": out["idx"],
                    "video_id": vid,
                    "qid": qid,
                    "reason": out["reason"],
                }

                skipped.append(skip_obj)
                append_jsonl(skipped_jsonl_path, skip_obj)

                msg = f"[{done}/{total}] ERROR {vid} qid={qid}: {out['reason']}"

            with print_lock:
                print(msg, flush=True)

    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nIncremental results saved during run -> {jsonl_path}")
    print(f"Final ordered JSON saved -> {output_path}")

    if skipped:
        skip_path = output_path.with_suffix(".skipped.json")

        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)

        print(f"Incremental skipped/errors -> {skipped_jsonl_path}")
        print(f"Final skipped/errors -> {skip_path}")


# -- Entry point --------------------------------------------------------------

MODES = ("sufficient", "hallucination", "insufficient", "all")
TASKS = ("qa", "classify")


def main():
    parser = argparse.ArgumentParser(
        description="CG-Bench evaluation pipeline"
    )

    parser.add_argument("--mode", choices=MODES, default="all")

    parser.add_argument(
        "--task",
        choices=TASKS,
        default="qa",
        help=(
            "qa = pure multiple-choice answering; "
            "classify = answerability classification"
        ),
    )

    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH)

    parser.add_argument(
        "--sampling",
        choices=("uniform", "evidence"),
        default="uniform",
    )

    parser.add_argument(
        "--evidence-fraction",
        type=float,
        default=EVIDENCE_FRACTION,
    )

    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    try:
        _client.models.list()
        print(f"Connected to vLLM at {VLLM_BASE_URL} model={MODEL_NAME}\n")

    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM server at {VLLM_BASE_URL}: {e}\n"
            f"Check VLLM_BASE_URL / VLLM_MODEL / server job."
        )

    data = json.load(open(FILTERED_JSON, encoding="utf-8"))

    sufficient = [
        d for d in data
        if d["evidence_condition"] == "sufficient"
    ]

    hallucination = [
        d for d in data
        if d["evidence_condition"] == "Hallucination"
    ]

    if args.limit is not None:
        sufficient = sufficient[:args.limit]
        hallucination = hallucination[:args.limit]
        print(f"*** DEBUG MODE: first {args.limit} entries per condition ***")

    print(
        f"Loaded {len(data)} entries "
        f"(sufficient={len(sufficient)}, hallucination={len(hallucination)})"
    )

    print(
        f"task={args.task}  "
        f"num_frames={args.num_frames}  "
        f"frame_width={args.frame_width}  "
        f"sampling={args.sampling}  "
        f"workers={args.workers}\n"
    )

    print_prompt_preview(args.task)

    suffix = f".{args.task}.json"

    if args.mode in ("sufficient", "all"):
        print("=== sufficient ===")
        run_evaluation(
            sufficient,
            VIDEOS_DIR,
            RESULTS_DIR / f"sufficient{suffix}",
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
        )

    if args.mode in ("hallucination", "all"):
        print("\n=== hallucination ===")
        run_evaluation(
            hallucination,
            VIDEOS_DIR,
            RESULTS_DIR / f"hallucination{suffix}",
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
        )

    if args.mode in ("insufficient", "all"):
        print("\n=== insufficient ===")
        run_evaluation(
            sufficient,
            INSUFFICIENT_DIR,
            RESULTS_DIR / f"insufficient{suffix}",
            insufficient=True,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
        )


if __name__ == "__main__":
    main()
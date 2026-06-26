#!/usr/bin/env python3
"""
CG-Bench evaluation pipeline (concurrent).

Three evaluation conditions:
  sufficient   — evidence_condition=sufficient,   videos from videos/
  hallucination— evidence_condition=Hallucination, videos from videos/
  insufficient — evidence_condition=sufficient,   videos from insufficient_videos/

Results saved to cgbench_result/{sufficient,hallucination,insufficient}.json.
Entries whose video is missing are logged to cgbench_result/{name}.skipped.json.

LLM backend: a vLLM OpenAI-compatible server (see serve_qwen3vl.sh).
Configure via env vars:
  VLLM_BASE_URL  (default http://localhost:8000/v1)
  VLLM_MODEL     (default qwen3-vl)

Sampling modes (--sampling):
  uniform  : frames sampled uniformly across the whole video (realistic).
  evidence : a fraction of frames are forced inside the ground-truth
             evidence_intervals (DIAGNOSTIC upper bound; uses answer-side info).

Concurrency: each entry (ffmpeg frame extraction + one vLLM request) runs in a
worker thread. vLLM batches the concurrent requests internally. Tune with
--workers.
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
NUM_FRAMES        = 32
FRAME_WIDTH       = 560    # px; frames scaled to this width before encoding.
EVIDENCE_FRACTION = 0.25   # fraction of frames forced into evidence intervals
                           # when --sampling evidence is used.
WORKERS           = 8      # concurrent in-flight requests.

# -- Paths --------------------------------------------------------------------
BASE_DIR = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
FILTERED_JSON       = BASE_DIR / "cgbench_pipeline" / "cgbench_filtered.json"
VIDEOS_DIR          = BASE_DIR / "source_datasets" / "cg_bench" / "videos"
INSUFFICIENT_DIR    = BASE_DIR / "source_datasets" / "cg_bench" / "insufficient_videos"
RESULTS_DIR         = BASE_DIR / "cgbench_result"

# -- LLM client ---------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.environ.get("VLLM_MODEL", "qwen3-vl")

# One client is safe to share across threads (httpx connection pool).
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
        capture_output=True, text=True, check=True,
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
    """Decide which seconds to sample, depending on the sampling mode."""
    if sampling == "evidence" and evidence_intervals:
        n_evidence = max(1, int(round(num_frames * evidence_fraction)))
        n_global   = max(0, num_frames - n_evidence)

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
        timestamps = [min(max(t, 0.0), max(duration - 0.01, 0.0)) for t in timestamps]
        return timestamps

    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def extract_frames(
    video_path: Path, timestamps: list[float], frame_width: int
) -> list[str]:
    """Extract frames at given timestamps, scaled to frame_width px.
    Returns base64-encoded JPEG strings (same order as timestamps)."""
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
                capture_output=True, check=True,
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
    """Plan timestamps according to `sampling`, then extract those frames."""
    duration = _video_duration(video_path)
    timestamps = plan_timestamps(
        duration, num_frames,
        sampling=sampling,
        evidence_intervals=evidence_intervals,
        evidence_fraction=evidence_fraction,
    )
    frames_b64 = extract_frames(video_path, timestamps, frame_width)
    return frames_b64, timestamps


# -- Prompt construction ------------------------------------------------------

def build_instruction(question: str, choices: list[str]) -> str:
    """Build the text instruction shown to the model (excludes the images)."""
    options_block = ""
    if choices:
        labels = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        options_block = f"\n\nOptions:\n{labels}"

    return f"""You are judging whether a video contains enough evidence to answer a question.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video. Use these timestamps when reporting evidence timing.

Decide exactly ONE label:
- ANSWERABLE: the frames contain clear visual evidence to answer the question.
- UNANSWERABLE: the frames do not contain enough evidence to answer.
- AMBIGUOUS: the frames contain partial evidence, but more than one answer remains possible.

Do not guess from common sense or prior knowledge. Decide only from what is visible in the frames.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "label": "ANSWERABLE | UNANSWERABLE | AMBIGUOUS",
  "answer": "the option letter (e.g. A) if ANSWERABLE, otherwise null",
  "evidence": "what in the frames supports your decision",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "one short sentence"
}}

Question: {question}{options_block}"""


# -- LLM call -----------------------------------------------------------------

def call_llm(
    frames_b64: list[str],
    timestamps: list[float],
    question: str,
    choices: list[str],
) -> str:
    """Call the vLLM-served Qwen3-VL model with timestamped frames and question.
    Returns the raw model response string (expected to be JSON)."""
    content = []
    for b64, ts in zip(frames_b64, timestamps):
        content.append({"type": "text", "text": f"Frame at {ts:.1f}s:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    content.append({"type": "text", "text": build_instruction(question, choices)})

    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content


# -- Output parsing -----------------------------------------------------------

def parse_model_json(raw: str) -> dict:
    """Best-effort parse of the model's JSON output. Never raises."""
    out = {"raw": raw}
    if raw is None:
        out["parse_error"] = "empty response"
        return out

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
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
) -> dict:
    """Process a single entry. Returns a dict describing the outcome.
    Runs inside a worker thread; never raises (errors are captured)."""
    video_id = entry["video_id"]
    qid      = entry["qid"]

    if insufficient:
        video_file = video_dir / f"{video_id}_q{qid}_insufficient.mp4"
    else:
        video_file = video_dir / f"{video_id}.mp4"

    if not video_file.exists():
        return {"idx": idx, "status": "skip", "video_id": video_id, "qid": qid,
                "reason": "missing_video"}

    ev_intervals = entry.get("evidence_intervals") if not insufficient else None

    try:
        frames, timestamps = sample_frames(
            video_file, num_frames, frame_width,
            sampling=sampling,
            evidence_intervals=ev_intervals,
            evidence_fraction=evidence_fraction,
        )
        raw_response = call_llm(
            frames, timestamps, entry["question"], entry.get("choices", []),
        )
    except Exception as e:
        return {"idx": idx, "status": "error", "video_id": video_id, "qid": qid,
                "reason": str(e)}

    parsed = parse_model_json(raw_response)
    result = {
        **entry,
        "sampling": sampling,
        "sampled_seconds": [round(t, 2) for t in timestamps],
        "model_response": raw_response,
        "model_parsed": parsed,
    }
    return {"idx": idx, "status": "ok", "video_id": video_id, "qid": qid,
            "result": result, "label": parsed.get("label", "?")}


# -- Evaluation loop (concurrent) ---------------------------------------------

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
) -> None:
    total = len(entries)
    # Pre-size results so we can place each entry back in its original order.
    results_by_idx: dict[int, dict] = {}
    skipped = []
    done = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_entry, idx, entry, video_dir,
                insufficient=insufficient,
                num_frames=num_frames,
                frame_width=frame_width,
                sampling=sampling,
                evidence_fraction=evidence_fraction,
            ): idx
            for idx, entry in enumerate(entries)
        }

        for fut in as_completed(futures):
            out = fut.result()
            done += 1
            vid, qid = out["video_id"], out["qid"]

            if out["status"] == "ok":
                results_by_idx[out["idx"]] = out["result"]
                msg = f"[{done}/{total}] OK    {vid} qid={qid}  label={out['label']}"
            elif out["status"] == "skip":
                skipped.append({"video_id": vid, "qid": qid, "reason": out["reason"]})
                msg = f"[{done}/{total}] SKIP  missing video: {vid} qid={qid}"
            else:  # error
                skipped.append({"video_id": vid, "qid": qid, "reason": out["reason"]})
                msg = f"[{done}/{total}] ERROR {vid} qid={qid}: {out['reason']}"

            with print_lock:
                print(msg, flush=True)

    # Reassemble in original entry order.
    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results -> {output_path}")

    if skipped:
        skip_path = output_path.with_suffix(".skipped.json")
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"Skipped {len(skipped)} entries -> {skip_path}")


# -- Entry point --------------------------------------------------------------

MODES = ("sufficient", "hallucination", "insufficient", "all")


def main():
    parser = argparse.ArgumentParser(description="CG-Bench evaluation pipeline (concurrent)")
    parser.add_argument("--mode", choices=MODES, default="all",
                        help="Which condition to evaluate (default: all)")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                        help=f"Frames to sample per video (default: {NUM_FRAMES})")
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH,
                        help=f"Width (px) to scale each frame to (default: {FRAME_WIDTH}).")
    parser.add_argument("--sampling", choices=("uniform", "evidence"), default="uniform",
                        help="uniform = even spread (realistic); "
                             "evidence = force frames into ground-truth intervals "
                             "(diagnostic upper bound, uses answer-side info).")
    parser.add_argument("--evidence-fraction", type=float, default=EVIDENCE_FRACTION,
                        help=f"Fraction of frames forced into evidence intervals "
                             f"when --sampling evidence (default: {EVIDENCE_FRACTION}).")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Concurrent in-flight requests (default: {WORKERS}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="DEBUG: only evaluate the first N entries of each condition")
    args = parser.parse_args()

    try:
        _client.models.list()
        print(f"Connected to vLLM at {VLLM_BASE_URL} (model={MODEL_NAME})\n")
    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM server at {VLLM_BASE_URL}: {e}\n"
            f"Is the serve job running? Check VLLM_BASE_URL / hostname."
        )

    data = json.load(open(FILTERED_JSON, encoding="utf-8"))
    sufficient    = [d for d in data if d["evidence_condition"] == "sufficient"]
    hallucination = [d for d in data if d["evidence_condition"] == "Hallucination"]

    if args.limit is not None:
        sufficient    = sufficient[: args.limit]
        hallucination = hallucination[: args.limit]
        print(f"*** DEBUG MODE: limited to first {args.limit} entries per condition ***")

    print(f"Loaded {len(data)} entries  "
          f"(sufficient={len(sufficient)}, hallucination={len(hallucination)})")
    print(f"num_frames={args.num_frames}  frame_width={args.frame_width}  "
          f"sampling={args.sampling}  workers={args.workers}\n")

    if args.mode in ("sufficient", "all"):
        print("=== sufficient ===")
        run_evaluation(
            sufficient, VIDEOS_DIR, RESULTS_DIR / "sufficient.json",
            num_frames=args.num_frames, frame_width=args.frame_width,
            sampling=args.sampling, evidence_fraction=args.evidence_fraction,
            workers=args.workers,
        )

    if args.mode in ("hallucination", "all"):
        print("\n=== hallucination ===")
        run_evaluation(
            hallucination, VIDEOS_DIR, RESULTS_DIR / "hallucination.json",
            num_frames=args.num_frames, frame_width=args.frame_width,
            sampling=args.sampling, evidence_fraction=args.evidence_fraction,
            workers=args.workers,
        )

    if args.mode in ("insufficient", "all"):
        print("\n=== insufficient ===")
        run_evaluation(
            sufficient, INSUFFICIENT_DIR, RESULTS_DIR / "insufficient.json",
            insufficient=True,
            num_frames=args.num_frames, frame_width=args.frame_width,
            sampling=args.sampling, evidence_fraction=args.evidence_fraction,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
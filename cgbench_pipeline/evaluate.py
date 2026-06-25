#!/usr/bin/env python3
"""
CG-Bench evaluation pipeline.

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
"""

import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

# ── Hyperparameters ──────────────────────────────────────────────────────────
NUM_FRAMES  = 32
FRAME_WIDTH = 336   # px; frames are scaled to this width before encoding.
                    # Lower = more frames fit in context but blurrier per frame.

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
FILTERED_JSON       = BASE_DIR / "cgbench_pipeline" / "cgbench_filtered.json"
VIDEOS_DIR          = BASE_DIR / "source_datasets" / "cg_bench" / "videos"
INSUFFICIENT_DIR    = BASE_DIR / "source_datasets" / "cg_bench" / "insufficient_videos"
RESULTS_DIR         = BASE_DIR / "cgbench_result"

# ── LLM client ───────────────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.environ.get("VLLM_MODEL", "qwen3-vl")

_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")


# ── Frame sampling ───────────────────────────────────────────────────────────

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


def sample_frames(video_path: Path, num_frames: int, frame_width: int = FRAME_WIDTH) -> list[str]:
    """Uniformly sample num_frames from video, scaled to frame_width px.
    Returns base64-encoded JPEG strings.

    Scaling at the ffmpeg stage caps the per-frame token cost at the source,
    independent of vLLM's mm_processor settings.
    """
    duration = _video_duration(video_path)
    # Centre each sample within its equal-width interval
    timestamps = [(i + 0.5) * duration / num_frames for i in range(num_frames)]

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
                    "-vf", f"scale={frame_width}:-2",  # width fixed, height keeps aspect (even)
                    "-q:v", "3",
                    frame_path,
                ],
                capture_output=True, check=True,
            )
            with open(frame_path, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode())

    return frames_b64


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(frames_b64: list[str], question: str, choices: list[str]) -> str:
    """
    Call the vLLM-served Qwen3-VL model with video frames and question.
    frames_b64 : list of base64-encoded JPEG strings
    question   : question text (may already embed choice labels inline)
    choices    : list of choice strings, in order (A, B, C, ...)
    Returns    : raw model response string
    """
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    # If your question already embeds the choices inline, drop this block.
    prompt = question
    if choices:
        labels = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n\n{labels}\n\n请只回答正确选项的字母(如 A)。"
    content.append({"type": "text", "text": prompt})

    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=128,
        temperature=0.0,   # deterministic for reproducible eval
    )
    return resp.choices[0].message.content


# ── Evaluation loop ──────────────────────────────────────────────────────────

def run_evaluation(
    entries: list[dict],
    video_dir: Path,
    output_path: Path,
    *,
    insufficient: bool = False,
    num_frames: int = NUM_FRAMES,
    frame_width: int = FRAME_WIDTH,
) -> None:
    results = []
    skipped = []

    for i, entry in enumerate(entries):
        video_id = entry["video_id"]
        qid      = entry["qid"]

        if insufficient:
            video_file = video_dir / f"{video_id}_q{qid}_insufficient.mp4"
        else:
            video_file = video_dir / f"{video_id}.mp4"

        if not video_file.exists():
            print(f"[{i+1}/{len(entries)}] SKIP  missing video: {video_file.name}")
            skipped.append({"video_id": video_id, "qid": qid, "reason": "missing_video"})
            continue

        try:
            frames = sample_frames(video_file, num_frames, frame_width)
            model_response = call_llm(frames, entry["question"], entry.get("choices", []))
        except Exception as e:
            print(f"[{i+1}/{len(entries)}] ERROR {video_id} qid={qid}: {e}")
            skipped.append({"video_id": video_id, "qid": qid, "reason": str(e)})
            continue

        result = {**entry, "model_response": model_response}
        results.append(result)
        print(f"[{i+1}/{len(entries)}] OK    {video_id} qid={qid}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results → {output_path}")

    if skipped:
        skip_path = output_path.with_suffix(".skipped.json")
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"Skipped {len(skipped)} entries → {skip_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

MODES = ("sufficient", "hallucination", "insufficient", "all")


def main():
    parser = argparse.ArgumentParser(description="CG-Bench evaluation pipeline")
    parser.add_argument(
        "--mode", choices=MODES, default="all",
        help="Which condition to evaluate (default: all)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=NUM_FRAMES,
        help=f"Frames to sample per video (default: {NUM_FRAMES})",
    )
    parser.add_argument(
        "--frame-width", type=int, default=FRAME_WIDTH,
        help=f"Width (px) to scale each frame to before encoding (default: {FRAME_WIDTH}). "
             f"Lower fits more frames in context but reduces per-frame detail.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="DEBUG: only evaluate the first N entries of each condition",
    )
    args = parser.parse_args()

    # Quick connectivity check so a misconfigured server fails fast, not 32 frames in.
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
    print(f"num_frames={args.num_frames}  frame_width={args.frame_width}\n")

    if args.mode in ("sufficient", "all"):
        print("=== sufficient ===")
        run_evaluation(
            sufficient, VIDEOS_DIR,
            RESULTS_DIR / "sufficient.json",
            num_frames=args.num_frames,
            frame_width=args.frame_width,
        )

    if args.mode in ("hallucination", "all"):
        print("\n=== hallucination ===")
        run_evaluation(
            hallucination, VIDEOS_DIR,
            RESULTS_DIR / "hallucination.json",
            num_frames=args.num_frames,
            frame_width=args.frame_width,
        )

    if args.mode in ("insufficient", "all"):
        print("\n=== insufficient ===")
        run_evaluation(
            sufficient, INSUFFICIENT_DIR,
            RESULTS_DIR / "insufficient.json",
            insufficient=True,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
        )


if __name__ == "__main__":
    main()
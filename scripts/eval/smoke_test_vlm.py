#!/usr/bin/env python3
"""Run a real-video, multi-frame smoke test against an OpenAI-compatible VLM."""

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frame-size", type=int, default=336)
    args = parser.parse_args()

    video = next((p for p in args.video_dir.rglob("*.mp4") if p.is_file()), None)
    if video is None:
        raise SystemExit(f"no mp4 found under {args.video_dir}")

    timestamps = (0.25, 0.75, 1.25, 1.75)
    content = []
    with tempfile.TemporaryDirectory(prefix="vlm_smoke_") as tmpdir:
        for index, timestamp in enumerate(timestamps):
            frame = Path(tmpdir) / f"frame_{index}.jpg"
            proc = subprocess.run(
                [
                    args.ffmpeg,
                    "-nostdin",
                    "-y",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-an",
                    "-vf",
                    f"scale={args.frame_size}:-2",
                    "-q:v",
                    "3",
                    str(frame),
                ],
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0 or not frame.is_file():
                raise RuntimeError(
                    f"ffmpeg failed at {timestamp}s: {proc.stderr[-1000:]}"
                )
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": f"Frame at {timestamp:.2f}s"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ]
            )

    content.append(
        {
            "type": "text",
            "text": (
                "This is a service smoke test. Return ONLY a JSON object with keys "
                '"status", "frame_count", and "brief_visual_description". Set status '
                'to "ok" and frame_count to 4.'
            ),
        }
    )
    client = OpenAI(
        base_url=args.base_url, api_key="EMPTY", max_retries=0, timeout=180
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content or ""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"response did not contain JSON: {raw!r}")
    parsed = json.loads(raw[start : end + 1])
    if parsed.get("status") != "ok" or int(parsed.get("frame_count", -1)) != 4:
        raise RuntimeError(f"unexpected smoke response: {parsed!r}")
    print(
        json.dumps(
            {"model": args.model, "video": str(video), "raw": raw, "parsed": parsed},
            ensure_ascii=False,
        )
    )
    print("VLM_SMOKE_TEST_OK", flush=True)


if __name__ == "__main__":
    main()

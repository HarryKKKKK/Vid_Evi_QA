#!/usr/bin/env python3
"""
Visualize how frame_width affects image clarity (and token cost).

Extracts one frame from a video at a given timestamp, rescales it to several
widths, and stacks them into a single comparison image with labels showing the
pixel width and the approximate Qwen3-VL token cost per frame.

Usage:
  python compare_resolutions.py VIDEO.mp4 --timestamp 130 \
      --widths 252 336 448 560 768 --out compare.png

Needs: ffmpeg/ffprobe on PATH, and Pillow (pip install pillow).
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def extract_frame(video_path: str, ts: float, width: int, out_path: str) -> None:
    """Extract one frame at `ts`, scaled to `width` px (height keeps aspect)."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            out_path,
        ],
        capture_output=True, check=True,
    )


def approx_tokens(width: int, height: int) -> int:
    """Approximate Qwen3-VL visual tokens: one per 28x28 patch."""
    return (width // 28) * (height // 28)


def label_height(font) -> int:
    return 46


def main():
    ap = argparse.ArgumentParser(description="Compare frame_width clarity/token cost")
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--timestamp", type=float, default=130.0,
                    help="seconds into the video to grab the frame (default 130)")
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[252, 336, 448, 560, 768],
                    help="frame widths (px) to compare")
    ap.add_argument("--display-width", type=int, default=560,
                    help="all crops are shown upscaled to this width so you can "
                         "judge detail at equal display size (default 560)")
    ap.add_argument("--out", default="compare.png", help="output image path")
    args = ap.parse_args()

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for w in args.widths:
            raw = os.path.join(tmp, f"f_{w}.jpg")
            extract_frame(args.video, args.timestamp, w, raw)
            img = Image.open(raw).convert("RGB")
            real_w, real_h = img.size
            toks = approx_tokens(real_w, real_h)

            # Upscale to a common display width (nearest = show the real blockiness)
            disp_h = round(real_h * args.display_width / real_w)
            shown = img.resize((args.display_width, disp_h), Image.NEAREST)

            # Add a label bar on top.
            lh = label_height(font)
            panel = Image.new("RGB", (args.display_width, disp_h + lh), (20, 20, 20))
            panel.paste(shown, (0, lh))
            draw = ImageDraw.Draw(panel)
            draw.text(
                (8, 10),
                f"width={w}px  actual={real_w}x{real_h}  ~{toks} tok/frame",
                fill=(255, 255, 255), font=font,
            )
            panels.append(panel)

    # Stack panels vertically.
    total_h = sum(p.height for p in panels) + 10 * (len(panels) - 1)
    max_w = max(p.width for p in panels)
    canvas = Image.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + 10

    canvas.save(args.out)

    # Print a quick token-budget summary for 32 / 96 / 128 frames.
    print(f"Saved comparison -> {args.out}\n")
    print("Approx total image tokens (frames x per-frame), budget ~31000:")
    print(f"{'width':>6} {'tok/frame':>10} {'32f':>8} {'96f':>8} {'128f':>8}")
    with tempfile.TemporaryDirectory() as tmp:
        for w in args.widths:
            raw = os.path.join(tmp, f"f_{w}.jpg")
            extract_frame(args.video, args.timestamp, w, raw)
            im = Image.open(raw)
            t = approx_tokens(*im.size)
            def mark(n):
                tot = t * n
                return f"{tot}" + ("!" if tot > 31000 else "")
            print(f"{w:>6} {t:>10} {mark(32):>8} {mark(96):>8} {mark(128):>8}")
    print("\n(! = exceeds the ~31000 image-token budget at 32768 context)")


if __name__ == "__main__":
    main()
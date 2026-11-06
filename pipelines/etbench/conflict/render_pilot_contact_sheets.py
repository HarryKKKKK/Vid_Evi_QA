"""Render interval-aligned contact sheets for manual pilot inspection."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/render_pilot_contact_sheets.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/render_pilot_contact_sheets.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(cap: cv2.VideoCapture, seconds: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000)
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((180, 320, 3), dtype=np.uint8)
    return cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)


def render(candidate: dict, video_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    rows = []
    duration = float(candidate["duration"])
    for index, (start, end) in enumerate(candidate["intervals"]):
        epsilon = min(0.15, max(0.01, (end - start) / 10))
        samples = [
            ("pre", max(0.0, start - 2.0)),
            ("start", min(end, start + epsilon)),
            ("middle", (start + end) / 2),
            ("end", max(start, end - epsilon)),
            ("post", min(duration, end + 2.0)),
        ]
        cells = []
        for phase, timestamp in samples:
            frame = read_frame(cap, timestamp)
            cv2.rectangle(frame, (0, 0), (320, 28), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"e{index + 1} {phase} {timestamp:.2f}s",
                (7, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(frame)
        rows.append(np.hstack(cells))
    cap.release()

    header = np.zeros((55, 1600, 3), dtype=np.uint8)
    title = f"idx={candidate['source_idx']} | {candidate['action']} | {video_path.name}"
    cv2.putText(
        header,
        title,
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    sheet = np.vstack([header, *rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Cannot write {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    for candidate in candidates:
        video_name = Path(candidate["video"]).name
        output_name = f"idx_{candidate['source_idx']}_{Path(video_name).stem}.jpg"
        render(candidate, args.video_dir / video_name, args.output_dir / output_name)
        print(output_name)


if __name__ == "__main__":
    main()

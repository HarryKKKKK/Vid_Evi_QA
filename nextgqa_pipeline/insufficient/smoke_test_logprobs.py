#!/usr/bin/env python3
"""One-item smoke test for A--E multimodal next-token logprobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ForcedChoiceScorer, extract_frame_grid, load_worklist, parse_endpoints


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frame-width", type=int, default=336)
    args = parser.parse_args()
    items = load_worklist(
        ROOT / "nextgqa_pipeline/nextgqa_filtered.json",
        ROOT / "source_datasets/next_gqa/videos",
        ROOT / "source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json",
    )
    item = items[args.index]
    holder, frames = extract_frame_grid(item.video_path, fps=args.fps, width=args.frame_width)
    try:
        print(f"item={item.key} frames={len(frames)} gold={item.gold}")
        for endpoint in parse_endpoints(args.endpoint):
            score = ForcedChoiceScorer(endpoint).score(frames, item)
            print(endpoint.tag, {
                "prediction": score["prediction"],
                "gold_probability": score["gold_probability"],
                "margin": score["margin"],
                "choice_probabilities": score["choice_probabilities"],
                "raw_answer": score["raw_answer"],
            })
        print("NEXTGQA_LOGPROB_SMOKE_OK")
    finally:
        holder.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check completeness and summarize the leak-filtered insufficient evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json",
    )
    parser.add_argument(
        "--freeze-dir", type=Path,
        default=ROOT / "source_datasets/next_gqa/freeze_videos_no_evidence_leak",
    )
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json",
    )
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    missing_videos = [
        f"{row['video_id']}_q{row['qid']}"
        for row in metadata
        if not (args.freeze_dir / f"{row['video_id']}_q{row['qid']}_freeze.mp4").is_file()
    ]
    expected_keys = {(str(row["video_id"]), str(row["qid"])) for row in metadata}
    result_keys = {(str(row["video_id"]), str(row["qid"])) for row in results}
    labels = Counter(
        str((row.get("model_parsed") or {}).get("label", "MISSING"))
        for row in results
    )
    parse_errors = sum(
        "parse_error" in (row.get("model_parsed") or {}) for row in results
    )
    answered = [
        row for row in results
        if (row.get("model_parsed") or {}).get("label") == "ANSWERABLE"
    ]
    answered_correct = sum(
        (row.get("model_parsed") or {}).get("answer") == row.get("answer")
        for row in answered
    )
    report = {
        "metadata_items": len(metadata),
        "freeze_videos_present": len(metadata) - len(missing_videos),
        "freeze_videos_missing": len(missing_videos),
        "result_records": len(results),
        "unique_result_items": len(result_keys),
        "result_items_missing": len(expected_keys - result_keys),
        "unexpected_result_items": len(result_keys - expected_keys),
        "model_labels": dict(sorted(labels.items())),
        "parse_errors": parse_errors,
        "answerable_correct": answered_correct,
        "answerable_wrong": len(answered) - answered_correct,
        "unanswerable_rate": (
            labels["UNANSWERABLE"] / len(results) if results else None
        ),
        "answerable_rate": labels["ANSWERABLE"] / len(results) if results else None,
        "missing_video_examples": missing_videos[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_videos or expected_keys != result_keys or parse_errors:
        raise SystemExit("evaluation completeness check failed")


if __name__ == "__main__":
    main()

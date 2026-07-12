"""
Check which entries in cgbench_filtered.json cannot be found in
source_datasets/cg_bench/insufficient_videos/.

Expected insufficient video filename format:
  {video_id}_q{qid}_insufficient.mp4

Example:
  _eFLsLQ9YTk_q4227_insufficient.mp4

Results are written to check_cgbench_videos_result.txt
in the same directory as this script.

Usage:
  python cgbench_pipeline/filter_download_check/cgbench_check_exist.py
"""

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FILTERED_JSON = PROJECT_ROOT / "cgbench_pipeline" / "cgbench_filtered.json"
# VIDEO_DIR = PROJECT_ROOT / "source_datasets" / "cg_bench" / "insufficient_videos"
VIDEO_DIR = PROJECT_ROOT / "source_datasets" / "cg_bench" / "freeze_videos"
OUTPUT_TXT = SCRIPT_DIR / "check_cgbench_videos_result.txt"


def main():
    if not FILTERED_JSON.exists():
        raise FileNotFoundError(f"Cannot find filtered json: {FILTERED_JSON}")

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Cannot find video directory: {VIDEO_DIR}")

    with open(FILTERED_JSON, encoding="utf-8") as f:
        filtered = json.load(f)

    existing_stems = {
        f.stem
        for f in VIDEO_DIR.iterdir()
        if f.is_file()
    }

    found = []
    missing = []

    for entry in filtered:
        video_id = entry["video_id"]
        qid = entry.get("qid")

        if qid is None or str(qid) == "":
            missing.append(entry)
            continue

        qid = str(qid)

        # expected_stem = f"{video_id}_q{qid}_insufficient"
        expected_stem = f"{video_id}_q{qid}_freeze"

        if expected_stem in existing_stems:
            found.append(entry)
        else:
            missing.append(entry)

    lines = [
        "CG-Bench Insufficient Video Check Result",
        "=" * 60,
        f"cgbench_filtered.json:    {FILTERED_JSON}",
        f"Video directory:          {VIDEO_DIR}",
        "",
        f"Expected filename format: {{video_id}}_q{{qid}}_insufficient.mp4",
        "",
        f"Total entries:            {len(filtered)}",
        f"Video files in directory: {len(existing_stems)}",
        "",
        f"Found:                    {len(found)}",
        f"Missing:                  {len(missing)}",
    ]

    if missing:
        lines += [
            "",
            "Missing entries:",
            "video_id | qid | expected filename",
        ]

        for entry in missing:
            video_id = entry["video_id"]
            qid = entry.get("qid")
            expected_filename = (
                f"{video_id}_q{qid}_insufficient.mp4"
                if qid is not None and str(qid) != ""
                else "N/A, qid missing"
            )
            lines.append(f"{video_id} | qid={qid} | {expected_filename}")
    else:
        lines += [
            "",
            "All insufficient videos accounted for.",
        ]

    output_text = "\n".join(lines)

    print(output_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\nResult written to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
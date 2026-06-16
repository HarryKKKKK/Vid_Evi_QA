"""
Check which entries in cgbench_filtered.json cannot be found in source_datasets/cg_bench/.
Matches by video_id (stem) first, then by qid (stem) as fallback.
Results are written to check_cgbench_videos_result.txt in the same directory as this script.

Usage:
  python check_cgbench_videos.py
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FILTERED_JSON = SCRIPT_DIR / "cgbench_filtered.json"
VIDEO_DIR = (SCRIPT_DIR / ".." / "source_datasets" / "cg_bench").resolve()
OUTPUT_TXT = SCRIPT_DIR / "check_cgbench_videos_result.txt"

with open(FILTERED_JSON, encoding="utf-8") as f:
    filtered = json.load(f)

existing_stems = {f.stem for f in VIDEO_DIR.iterdir() if f.is_file()}

found_by_uid = []
found_by_qid = []
missing = []

for entry in filtered:
    video_id = entry["video_id"]
    qid = str(entry.get("qid", ""))

    if video_id in existing_stems:
        found_by_uid.append(entry)
    elif qid and qid in existing_stems:
        found_by_qid.append(entry)
    else:
        missing.append(entry)

lines = [
    "CG-Bench Video Check Result",
    "=" * 60,
    f"cgbench_filtered.json:    {FILTERED_JSON.resolve()}",
    f"Video directory:          {VIDEO_DIR}",
    "",
    f"Total entries:            {len(filtered)}",
    f"Video files in directory: {len(existing_stems)}",
    "",
    f"Found by video_id:        {len(found_by_uid)}",
    f"Found by qid:             {len(found_by_qid)}",
    f"Missing:                  {len(missing)}",
]

if missing:
    lines += [
        "",
        "Missing entries (video_id | qid):",
    ]
    for entry in missing:
        lines.append(f"  {entry['video_id']} | qid={entry.get('qid')}")
else:
    lines += [
        "",
        "All videos accounted for.",
    ]

output_text = "\n".join(lines)
print(output_text)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(output_text + "\n")

print(f"\nResult written to {OUTPUT_TXT}")
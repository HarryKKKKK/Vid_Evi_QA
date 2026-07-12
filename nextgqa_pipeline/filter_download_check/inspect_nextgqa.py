"""Inspect the raw NExT-GQA annotation files before filtering.

Reads the QA CSV, the grounding JSON, and the video-id mapping JSON from an
annotation directory and prints their structure: columns, dtypes, sample
rows, join coverage between the three files, and counts of rows that would
fail the joins used later by ``filter_nextgqa.py``. This script only reads
and reports; it does not write any output file and does not filter anything.

Usage:
  python nextgqa_pipeline/filter_download_check/inspect_nextgqa.py \
    --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_DIR = (
    REPO_ROOT / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa"
)

CSV_FILENAME = "test.csv"
GROUNDING_FILENAME = "gsub_test.json"
MAPPING_FILENAME = "map_vid_vidorID.json"

REQUIRED_CSV_COLUMNS = [
    "video_id", "frame_count", "width", "height", "question", "answer",
    "qid", "type", "a0", "a1", "a2", "a3", "a4",
]
OPTION_COLUMNS = ["a0", "a1", "a2", "a3", "a4"]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV annotation file not found: {csv_path}\n"
            f"Expected NExT-GQA test-split QA file with columns: {REQUIRED_CSV_COLUMNS}"
        )
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}\n"
                f"Found columns: {fieldnames}"
            )
        rows = list(reader)
    return rows


def load_json(path: Path, description: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{description} at {path} is not valid JSON: {e}") from e


def find_multi_interval_example(grounding: dict[str, Any]) -> tuple[str, str, list] | None:
    for video_id, entry in grounding.items():
        if not isinstance(entry, dict):
            continue
        location = entry.get("location")
        if not isinstance(location, dict):
            continue
        for qid, pairs in location.items():
            if isinstance(pairs, list) and len(pairs) >= 2:
                return video_id, qid, pairs
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw NExT-GQA annotation schema (read-only, no output files).",
    )
    parser.add_argument(
        "--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing test.csv, gsub_test.json, map_vid_vidorID.json "
             f"(default: {DEFAULT_ANNOTATION_DIR})",
    )
    parser.add_argument("--csv-filename", default=CSV_FILENAME)
    parser.add_argument("--grounding-filename", default=GROUNDING_FILENAME)
    parser.add_argument("--mapping-filename", default=MAPPING_FILENAME)
    parser.add_argument("--samples", type=int, default=5, help="Number of sample rows to print")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ann_dir: Path = args.annotation_dir
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    csv_path = ann_dir / args.csv_filename
    grounding_path = ann_dir / args.grounding_filename
    mapping_path = ann_dir / args.mapping_filename

    section("STEP 1: Files found")
    for p in (csv_path, grounding_path, mapping_path):
        print(f"  {'OK  ' if p.exists() else 'MISS'}  {p}")

    rows = load_csv_rows(csv_path)
    grounding = load_json(grounding_path, "Grounding JSON (gsub_test.json)")
    mapping = load_json(mapping_path, "Mapping JSON (map_vid_vidorID.json)")

    if not isinstance(grounding, dict):
        raise ValueError(f"Expected grounding JSON top level to be an object/dict, got {type(grounding)}")
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected mapping JSON top level to be an object/dict, got {type(mapping)}")

    section("STEP 2: CSV columns")
    print(list(rows[0].keys()) if rows else "(no rows)")

    section(f"STEP 3: CSV sample rows (first {args.samples})")
    for r in rows[: args.samples]:
        print(r)

    section("STEP 4: Raw `type` distribution")
    type_counter = Counter(clean_text(r.get("type", "")) for r in rows)
    for t, c in type_counter.most_common():
        print(f"  {t!r}: {c}")

    section("STEP 5: qid type and samples")
    qid_samples = [r.get("qid") for r in rows[: args.samples]]
    print(f"  raw python type in CSV rows: {type(qid_samples[0]) if qid_samples else 'n/a'} (csv.DictReader always yields str)")
    print(f"  sample raw values: {qid_samples}")
    non_numeric_qid = [r.get("qid") for r in rows if not str(r.get("qid", "")).strip().lstrip("-").isdigit()]
    print(f"  qid values that are not plain integers: {len(non_numeric_qid)}")
    if non_numeric_qid[:5]:
        print(f"    examples: {non_numeric_qid[:5]}")

    section("STEP 6: video_id type and samples")
    vid_samples = [r.get("video_id") for r in rows[: args.samples]]
    print(f"  raw python type in CSV rows: str (csv.DictReader)")
    print(f"  sample raw values: {vid_samples}")
    grounding_key_samples = list(grounding.keys())[: args.samples]
    print(f"  grounding JSON top-level key type: {type(grounding_key_samples[0]) if grounding_key_samples else 'n/a'}")
    print(f"  grounding JSON key samples: {grounding_key_samples}")
    mapping_key_samples = list(mapping.keys())[: args.samples]
    print(f"  mapping JSON key samples: {mapping_key_samples}")

    section("STEP 7: answer / options structure")
    if rows:
        r0 = rows[0]
        print(f"  answer column value: {r0.get('answer')!r}")
        print(f"  option columns: " + ", ".join(f"{c}={r0.get(c)!r}" for c in OPTION_COLUMNS))
        matches = [c for c in OPTION_COLUMNS if clean_text(r0.get(c, "")).lower() == clean_text(r0.get("answer", "")).lower()]
        print(f"  option column(s) matching answer text (case-insensitive): {matches}")

    section("STEP 8: grounding JSON structure (per-video entry)")
    if grounding:
        sample_vid = next(iter(grounding))
        sample_entry = grounding[sample_vid]
        print(f"  video_id: {sample_vid!r}")
        print(f"  entry top-level keys: {list(sample_entry.keys()) if isinstance(sample_entry, dict) else type(sample_entry)}")
        if isinstance(sample_entry, dict):
            print(f"    duration: {sample_entry.get('duration')!r} (type={type(sample_entry.get('duration')).__name__})")
            print(f"    fps: {sample_entry.get('fps')!r}")
            location = sample_entry.get("location", {})
            print(f"    location key type: {type(next(iter(location))).__name__ if location else 'n/a'}")
            print(f"    location sample: {dict(list(location.items())[:3])}")

    section("STEP 9: one multi evidence-interval example (qid with >=2 segments)")
    example = find_multi_interval_example(grounding)
    if example:
        video_id, qid, pairs = example
        print(f"  video_id={video_id!r} qid={qid!r} -> {pairs}")
    else:
        print("  No qid with 2+ segments found in this grounding file.")

    section("STEP 10: mapping JSON key/value samples")
    for k in mapping_key_samples:
        print(f"  {k!r} -> {mapping[k]!r}")

    section("STEP 11-15: join coverage statistics")
    total = len(rows)

    seen_pairs: set[tuple[str, str]] = set()
    duplicate_rows = 0
    for r in rows:
        key = (clean_text(r.get("video_id", "")), clean_text(r.get("qid", "")))
        if key in seen_pairs:
            duplicate_rows += 1
        seen_pairs.add(key)

    missing_grounding = 0
    missing_mapping = 0
    ambiguous_answer = 0
    no_match_answer = 0
    joined_grounding = 0
    joined_mapping = 0

    for r in rows:
        video_id = clean_text(r.get("video_id", ""))
        qid_raw = clean_text(r.get("qid", ""))

        entry = grounding.get(video_id)
        location = entry.get("location") if isinstance(entry, dict) else None
        pairs = location.get(qid_raw) if isinstance(location, dict) else None
        if pairs is None:
            missing_grounding += 1
        else:
            joined_grounding += 1

        if video_id not in mapping:
            missing_mapping += 1
        else:
            joined_mapping += 1

        answer = clean_text(r.get("answer", "")).lower()
        options = [clean_text(r.get(c, "")).lower() for c in OPTION_COLUMNS]
        n_match = sum(1 for o in options if o == answer and o != "")
        if n_match == 0:
            no_match_answer += 1
        elif n_match > 1:
            ambiguous_answer += 1

    print(f"  Total CSV rows (raw, incl. duplicates):        {total}")
    print(f"  Duplicate (video_id, qid) rows:                {duplicate_rows}")
    print(f"  Rows joined to grounding (video_id+qid found): {joined_grounding}")
    print(f"  Rows missing grounding annotation:             {missing_grounding}")
    print(f"  Rows joined to mapping (video_id found):       {joined_mapping}")
    print(f"  Rows missing video-id mapping:                 {missing_mapping}")
    print(f"  Rows where answer matches 0 options:           {no_match_answer}")
    print(f"  Rows where answer matches 2+ options:          {ambiguous_answer}")

    section("Done")
    print("This was a read-only inspection. No files were written.")


if __name__ == "__main__":
    main()

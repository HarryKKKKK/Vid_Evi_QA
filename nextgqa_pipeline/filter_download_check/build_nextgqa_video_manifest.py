"""Build a deduplicated video-download manifest for the NExT-GQA filtered set.

Reads ``nextgqa_filtered.json`` (produced by ``filter_nextgqa.py``) and
``map_vid_vidorID.json`` (official NExT-GQA video-id -> on-disk relative path
mapping) and writes one manifest entry per unique video (a single video can
be required by several qids, so the manifest is deduplicated by video_id).

This script does not download or move any video file; it only produces the
manifest describing what is needed and where it is expected to live. No URLs
are invented: the mapping JSON gives a relative path fragment, not a
downloadable URL, so ``expected_relative_path`` is a filesystem path under
the video directory, not a network location.

Usage:
  python nextgqa_pipeline/filter_download_check/build_nextgqa_video_manifest.py \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
    --output-json nextgqa_pipeline/nextgqa_video_manifest.json \
    --output-txt nextgqa_pipeline/nextgqa_video_manifest.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_DIR = (
    REPO_ROOT / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa"
)
DEFAULT_FILTERED_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filtered.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_video_manifest.json"
DEFAULT_OUTPUT_TXT = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_video_manifest.txt"
MAPPING_FILENAME = "map_vid_vidorID.json"
VIDEO_EXTENSION = ".mp4"


def load_filtered(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Filtered JSON not found: {path}. Run filter_nextgqa.py first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data)}")
    return data


def build_manifest(
    filtered: list[dict[str, Any]], mapping: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (manifest_entries, video_ids_missing_from_mapping)."""
    required_qids: dict[str, set[str]] = {}
    for record in filtered:
        video_id = str(record["video_id"])
        qid = str(record["qid"])
        required_qids.setdefault(video_id, set()).add(qid)

    manifest: list[dict[str, Any]] = []
    missing_mapping: list[str] = []

    for video_id in sorted(required_qids):
        mapped = mapping.get(video_id)
        if mapped is None:
            missing_mapping.append(video_id)
            expected_relative_path = None
        else:
            expected_relative_path = f"{mapped}{VIDEO_EXTENSION}"

        manifest.append({
            "video_id": video_id,
            "mapped_video_id": mapped,
            "expected_relative_path": expected_relative_path,
            "required_by_qids": sorted(
                required_qids[video_id],
                key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
            ),
            "status": "unknown",
        })

    return manifest, missing_mapping


def format_manifest_txt(manifest: list[dict[str, Any]], missing_mapping: list[str]) -> str:
    lines = [
        "NExT-GQA Video Manifest",
        "=" * 60,
        f"Unique videos required:            {len(manifest)}",
        f"Videos missing from map_vid_vidorID.json: {len(missing_mapping)}",
        "",
        "video_id | mapped_video_id | expected_relative_path | #qids",
    ]
    for entry in manifest:
        lines.append(
            f"{entry['video_id']} | {entry['mapped_video_id']} | "
            f"{entry['expected_relative_path']} | {len(entry['required_by_qids'])}"
        )
    if missing_mapping:
        lines += ["", "Videos with no mapping entry (cannot resolve a path):"]
        lines += [f"  {v}" for v in missing_mapping]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED_JSON)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--mapping-filename", default=MAPPING_FILENAME)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    filtered = load_filtered(args.filtered_json)
    mapping_path = args.annotation_dir / args.mapping_filename
    mapping = load_mapping(mapping_path)

    manifest, missing_mapping = build_manifest(filtered, mapping)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(manifest)} manifest entries to {args.output_json}")

    text = format_manifest_txt(manifest, missing_mapping)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Wrote manifest summary to {args.output_txt}")

    if args.verbose:
        print("\n" + text)

    if missing_mapping:
        print(f"\nWARNING: {len(missing_mapping)} video(s) have no entry in {mapping_path}")


if __name__ == "__main__":
    main()

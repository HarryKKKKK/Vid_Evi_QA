"""Check which NExT-GQA videos required by nextgqa_filtered.json exist locally.

This script never removes samples from nextgqa_filtered.json. Annotation
filtering (which samples pass the sufficient-anchor rules) and video
availability (whether the underlying file exists on disk) are deliberately
kept as two separate artifacts:

  - nextgqa_filtered.json       -> which (video, question) samples qualify
  - nextgqa_video_check.json    -> whether each required video is available

Usage:
  python nextgqa_pipeline/filter_download_check/check_nextgqa_videos.py \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
    --video-dir source_datasets/next_gqa/videos \
    --ffprobe \
    --output-json nextgqa_pipeline/nextgqa_video_check.json \
    --output-txt nextgqa_pipeline/nextgqa_video_check.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_nextgqa_video_manifest import build_manifest, load_mapping  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ANNOTATION_DIR = (
    REPO_ROOT / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa"
)
DEFAULT_FILTERED_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filtered.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "videos"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_video_check.json"
DEFAULT_OUTPUT_TXT = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_video_check.txt"
MAPPING_FILENAME = "map_vid_vidorID.json"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
DEFAULT_DURATION_TOLERANCE = 2.0


def load_filtered(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Filtered JSON not found: {path}. Run filter_nextgqa.py first.")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def scan_video_dir(video_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Return (by_filename, by_stem) indices, keys lower-cased."""
    by_filename: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    if not video_dir.exists():
        return by_filename, by_stem
    for p in video_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        by_filename.setdefault(p.name.lower(), []).append(p)
        by_stem.setdefault(p.stem.lower(), []).append(p)
    return by_filename, by_stem


def locate_video(
    video_dir: Path,
    expected_relative_path: str | None,
    target_stem: str,
    by_filename: dict[str, list[Path]],
    by_stem: dict[str, list[Path]],
) -> tuple[str, Path | None, str]:
    """Match order: exact relative path -> full filename -> stem (any ext) -> fuzzy stem.

    Returns (status, path_or_None, matched_tier).
    status is one of: found, missing, ambiguous.
    """
    # Tier 1: exact relative path from the mapping JSON.
    if expected_relative_path:
        candidate = video_dir / expected_relative_path
        if candidate.exists():
            return "found", candidate, "exact_relative_path"

    # Tier 2: full filename match anywhere under video_dir (mp4 assumed by mapping).
    if expected_relative_path:
        filename_key = Path(expected_relative_path).name.lower()
        matches = by_filename.get(filename_key, [])
        if len(matches) == 1:
            return "found", matches[0], "full_filename"
        if len(matches) > 1:
            return "ambiguous", None, "full_filename"

    # Tier 3: stem match, any supported extension.
    matches = by_stem.get(target_stem.lower(), [])
    if len(matches) == 1:
        return "found", matches[0], "stem_exact"
    if len(matches) > 1:
        return "ambiguous", None, "stem_exact"

    # Tier 4: fuzzy stem match (substring), last resort.
    fuzzy = [p for stem, paths in by_stem.items() if target_stem.lower() in stem for p in paths]
    if len(fuzzy) == 1:
        return "found", fuzzy[0], "fuzzy_stem"
    if len(fuzzy) > 1:
        return "ambiguous", None, "fuzzy_stem"

    return "missing", None, "none"


def ffprobe_check(path: Path, expected_duration: float, tolerance: float, timeout: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "corrupt": True, "error": str(e)}

    if proc.returncode != 0:
        return {"ok": False, "corrupt": True, "error": proc.stderr.strip()}

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"ok": False, "corrupt": True, "error": f"ffprobe output not JSON: {e}"}

    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        return {"ok": False, "corrupt": True, "error": "no video stream found"}

    vs = video_streams[0]
    fmt = info.get("format", {})
    try:
        probed_duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        probed_duration = None

    duration_mismatch = (
        probed_duration is not None
        and expected_duration > 0
        and abs(probed_duration - expected_duration) > tolerance
    )

    return {
        "ok": True,
        "corrupt": False,
        "duration_mismatch": duration_mismatch,
        "probed_duration": probed_duration,
        "expected_duration": expected_duration,
        "width": vs.get("width"),
        "height": vs.get("height"),
        "r_frame_rate": vs.get("r_frame_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, default=DEFAULT_FILTERED_JSON)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--mapping-filename", default=MAPPING_FILENAME)
    parser.add_argument("--manifest", type=Path, default=None,
                         help="Reuse a precomputed manifest instead of recomputing from filtered-json + annotation-dir")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--ffprobe", action="store_true", help="Run ffprobe on each found video")
    parser.add_argument("--duration-tolerance", type=float, default=DEFAULT_DURATION_TOLERANCE)
    parser.add_argument("--ffprobe-timeout", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    filtered = load_filtered(args.filtered_json)
    video_duration_by_id: dict[str, float] = {}
    for r in filtered:
        video_duration_by_id[str(r["video_id"])] = float(r["video_duration"])

    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
    else:
        mapping = load_mapping(args.annotation_dir / args.mapping_filename)
        manifest, _missing = build_manifest(filtered, mapping)

    if args.limit is not None:
        manifest = manifest[: args.limit]

    by_filename, by_stem = scan_video_dir(args.video_dir)

    results: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for entry in manifest:
        video_id = entry["video_id"]
        expected_relative_path = entry.get("expected_relative_path")
        mapped_video_id = entry.get("mapped_video_id")
        target_stem = Path(mapped_video_id).name if mapped_video_id else video_id

        status, path, tier = locate_video(
            args.video_dir, expected_relative_path, target_stem, by_filename, by_stem
        )

        record: dict[str, Any] = {
            "video_id": video_id,
            "expected_relative_path": expected_relative_path,
            "status": status,
            "matched_tier": tier,
            "matched_path": str(path) if path else None,
            "required_by_qids": entry.get("required_by_qids", []),
        }

        if status == "found" and args.ffprobe:
            expected_duration = video_duration_by_id.get(video_id, 0.0)
            probe = ffprobe_check(path, expected_duration, args.duration_tolerance, args.ffprobe_timeout)
            record["ffprobe"] = probe
            if probe.get("corrupt"):
                record["status"] = "corrupt"
            elif probe.get("duration_mismatch"):
                record["status"] = "duration_mismatch"

        results.append(record)
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} check results to {args.output_json}")

    lines = [
        "NExT-GQA Video Check Result",
        "=" * 60,
        f"Filtered JSON:  {args.filtered_json}",
        f"Video dir:      {args.video_dir}",
        f"ffprobe:        {'enabled' if args.ffprobe else 'disabled'}",
        "",
        f"Total videos checked: {len(results)}",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")

    problems = [r for r in results if r["status"] != "found"]
    if problems:
        lines += ["", "Entries needing attention (video_id | status | expected_relative_path):"]
        for r in problems:
            lines.append(f"  {r['video_id']} | {r['status']} | {r['expected_relative_path']}")

    text = "\n".join(lines)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Wrote summary to {args.output_txt}")

    if args.verbose:
        print("\n" + text)


if __name__ == "__main__":
    main()

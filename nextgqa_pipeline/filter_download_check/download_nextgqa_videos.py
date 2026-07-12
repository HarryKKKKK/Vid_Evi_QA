"""Prepare local NExT-GQA raw video files for the filtered manifest.

IMPORTANT — no scriptable official download endpoint exists.

The upstream NExT-GQA repository (``source_datasets/next_gqa/NExT-GQA/README.md``)
only links a Google Drive folder for "raw videos"
(https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view), which
is a manual, browser-gated download (Google Drive does not expose a stable,
scriptable direct-download URL for large files, and NExT-GQA does not publish
per-video URLs). This script therefore does NOT contain any hard-coded
download URL and does NOT scrape or guess one.

Supported ways to populate videos, in order of preference:

1. ``--source-video-dir``: you already have a complete (or partial) local copy
   of the NExT-GQA / VidOR raw videos (e.g. after manually downloading the
   Google Drive archive and extracting it, or from an existing VidOR copy).
   This script copies/symlinks/hardlinks just the videos referenced in the
   manifest from that directory into ``--video-dir``, preserving the
   ``expected_relative_path`` layout from ``map_vid_vidorID.json``.

2. ``--url-manifest``: a JSON file YOU provide, mapping video_id (or
   mapped_video_id) to a URL you have legally obtained (e.g. a presigned URL
   to your own re-hosted copy). This script will download over HTTP(S) with
   resume/retry support. This script does not populate this file for you.

If neither is given, the script prints what is missing and exits without
downloading anything.

Usage (see nextgqa_pipeline/README.md for full examples):
  python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
    --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
    --video-dir source_datasets/next_gqa/videos \
    --source-video-dir /path/to/existing/nextgqa_or_vidor_videos \
    --copy-mode symlink \
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_video_manifest.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "videos"
DEFAULT_REPORT_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_download_report.json"

COPY_MODES = ("copy", "symlink", "hardlink")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run build_nextgqa_video_manifest.py first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def load_url_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"--url-manifest file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (video_id -> url) in {path}, got {type(data)}")
    return data


def find_in_source_dir(source_dir: Path, relative_path: str, mapped_video_id: str | None) -> Path | None:
    """Locate the source file for one manifest entry.

    Tries, in order: the exact expected relative path, then a filename search
    for `{mapped_video_id}.mp4` (or `.mkv`/`.webm`/`.avi`/`.mov`) anywhere
    under source_dir.
    """
    candidate = source_dir / relative_path
    if candidate.exists():
        return candidate

    if mapped_video_id is None:
        return None

    stem = Path(mapped_video_id).name
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
        matches = list(source_dir.rglob(f"{stem}{ext}"))
        if len(matches) == 1:
            return matches[0]
    return None


def place_file(src: Path, dst: Path, copy_mode: str, overwrite: bool, dry_run: bool) -> str:
    if dst.exists() and not overwrite:
        return "skipped_exists"

    if dry_run:
        return f"dry_run_would_{copy_mode}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif copy_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {copy_mode}")
    return f"{copy_mode}_ok"


def download_url(url: str, dst: Path, timeout: float, retries: int, resume: bool, dry_run: bool) -> str:
    if dst.exists():
        return "skipped_exists"

    if dry_run:
        return "dry_run_would_download"

    part_path = dst.with_suffix(dst.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            headers = {}
            mode = "wb"
            existing_size = 0
            if resume and part_path.exists():
                existing_size = part_path.stat().st_size
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with part_path.open(mode) as f:
                    shutil.copyfileobj(resp, f)

            part_path.rename(dst)
            return "downloaded_ok"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30))

    return f"failed: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--source-video-dir", type=Path, default=None,
                         help="Existing local directory containing already-downloaded NExT-GQA/VidOR videos")
    parser.add_argument("--url-manifest", type=Path, default=None,
                         help="JSON file mapping video_id -> a URL you have legally obtained; no URLs are invented by this script")
    parser.add_argument("--copy-mode", choices=COPY_MODES, default="copy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    if args.source_video_dir is None and args.url_manifest is None:
        print(
            "No --source-video-dir and no --url-manifest given.\n"
            "NExT-GQA raw videos are distributed only via a manual Google Drive link "
            "(see source_datasets/next_gqa/NExT-GQA/README.md, section 'Preparation').\n"
            "Please either:\n"
            "  1) manually download+extract the official archive and pass its path via "
            "--source-video-dir, or\n"
            "  2) provide --url-manifest pointing at a JSON file of video_id -> URL you "
            "have legally obtained.\n"
            "Nothing was downloaded."
        )
        return

    url_manifest = load_url_manifest(args.url_manifest)

    results: list[dict[str, Any]] = []

    def process_entry(entry: dict[str, Any]) -> dict[str, Any]:
        video_id = entry["video_id"]
        relative_path = entry.get("expected_relative_path")
        if relative_path is None:
            return {"video_id": video_id, "status": "no_mapping_no_path"}

        dst = args.video_dir / relative_path

        if args.source_video_dir is not None:
            src = find_in_source_dir(args.source_video_dir, relative_path, entry.get("mapped_video_id"))
            if src is None:
                return {"video_id": video_id, "status": "not_found_in_source_dir"}
            status = place_file(src, dst, args.copy_mode, args.overwrite, args.dry_run)
            return {"video_id": video_id, "status": status, "source": str(src)}

        url = url_manifest.get(video_id) or url_manifest.get(entry.get("mapped_video_id", ""))
        if url is None:
            return {"video_id": video_id, "status": "no_url_in_url_manifest"}
        status = download_url(url, dst, args.timeout, args.retries, args.resume, args.dry_run)
        return {"video_id": video_id, "status": status, "url": url}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_entry, e) for e in manifest]
        for fut in as_completed(futures):
            results.append(fut.result())

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processed {len(results)} manifest entries:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote per-video report to {args.report_json}")


if __name__ == "__main__":
    main()

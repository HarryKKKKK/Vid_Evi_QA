"""Prepare local NExT-GQA raw video files for the filtered manifest.

The upstream NExT-GQA repository (``source_datasets/next_gqa/NExT-GQA/README.md``)
distributes "raw videos" as a single Google Drive file
(https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view) —
there is no per-video URL list published anywhere. This script never invents
or scrapes a download URL of its own; the only URL it ever touches is that
one official, publicly-linked Google Drive file id.

Supported ways to populate videos, in order of preference:

1. ``--fetch-official-archive``: download the official archive above via the
   ``gdown`` package (handles Google Drive's large-file confirmation flow),
   extract it, then use the extracted directory as the source for the
   copy/symlink/hardlink step below. Requires ``pip install gdown`` and
   network access to Google Drive from wherever this script runs. This is a
   best-effort convenience path, not a guaranteed-stable API: Drive's
   confirmation flow and per-file quota can change or throttle at any time.
   If it fails, fall back to downloading the file manually in a browser and
   using ``--source-video-dir`` (option 2) instead.

2. ``--source-video-dir``: you already have a complete (or partial) local copy
   of the NExT-GQA / VidOR raw videos (e.g. after manually downloading the
   Google Drive archive and extracting it, or from an existing VidOR copy).
   This script copies/symlinks/hardlinks just the videos referenced in the
   manifest from that directory into ``--video-dir``, preserving the
   ``expected_relative_path`` layout from ``map_vid_vidorID.json``.

3. ``--url-manifest``: a JSON file YOU provide, mapping video_id (or
   mapped_video_id) to a URL you have legally obtained (e.g. a presigned URL
   to your own re-hosted copy). This script will download over HTTP(S) with
   resume/retry support. This script does not populate this file for you.

If none of the three is given, the script prints what is missing and exits
without downloading anything. ``--fetch-official-archive`` is mutually
exclusive with ``--source-video-dir`` and ``--url-manifest``.

Usage (see nextgqa_pipeline/README.md for full examples):
  python nextgqa_pipeline/filter_download_check/download_nextgqa_videos.py \
    --manifest nextgqa_pipeline/nextgqa_video_manifest.json \
    --video-dir source_datasets/next_gqa/videos \
    --fetch-official-archive \
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

# The single official "raw videos" Google Drive file id, as published in
# source_datasets/next_gqa/NExT-GQA/README.md ("Preparation" section).
OFFICIAL_GDRIVE_FILE_ID = "1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH"
DEFAULT_ARCHIVE_CACHE_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "raw_download"
DEFAULT_ARCHIVE_EXTRACT_DIR = REPO_ROOT / "source_datasets" / "next_gqa" / "raw_extracted"

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


def fetch_official_archive(
    gdrive_file_id: str, cache_dir: Path, extract_dir: Path, resume: bool, overwrite: bool
) -> Path:
    """Download the official NExT-GQA raw-video archive from Google Drive via gdown, then extract it.

    Returns the directory the archive was extracted into, meant to be used
    as the source directory for the normal copy/symlink/hardlink placement
    step below (same role as a user-supplied --source-video-dir).

    This is a best-effort convenience path, not a guaranteed-stable API:
    Google Drive's large-file confirmation flow and per-file quota can change
    or throttle at any time. If it fails, download the file manually via the
    link in source_datasets/next_gqa/NExT-GQA/README.md and pass the
    extracted directory via --source-video-dir instead.
    """
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "--fetch-official-archive requires the 'gdown' package (pip install gdown), "
            "which is not installed in this environment."
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official NExT-GQA raw-video archive (Google Drive id={gdrive_file_id}) to {cache_dir} ...")
    output = gdown.download(id=gdrive_file_id, output=str(cache_dir) + os.sep, quiet=False, resume=resume)
    if output is None:
        raise RuntimeError(
            "gdown.download() returned None, meaning the download did not complete "
            "(possible causes: Google Drive per-file download quota exceeded, the file's "
            "sharing permissions changed, or a network error). See the gdown output above "
            "for details, or download the file manually and use --source-video-dir instead."
        )
    archive_path = Path(output)
    print(f"Downloaded archive to {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    if any(extract_dir.iterdir()) and not overwrite:
        print(f"{extract_dir} already has content and --overwrite was not given; skipping extraction.")
        return extract_dir

    print(f"Extracting {archive_path} to {extract_dir} ...")
    try:
        shutil.unpack_archive(str(archive_path), str(extract_dir))
    except (shutil.ReadError, ValueError) as e:
        raise RuntimeError(
            f"Could not auto-extract {archive_path} (unrecognized archive format: {e}). "
            "Please extract it manually and re-run with --source-video-dir pointing at the "
            "extracted folder."
        ) from e

    return extract_dir


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
    parser.add_argument("--fetch-official-archive", action="store_true",
                         help="Download the official raw-video archive from Google Drive via gdown "
                              "(pip install gdown) and use it as the source directory. Mutually "
                              "exclusive with --source-video-dir / --url-manifest.")
    parser.add_argument("--gdrive-file-id", default=OFFICIAL_GDRIVE_FILE_ID,
                         help="Google Drive file id for the official raw-video archive "
                              f"(default: {OFFICIAL_GDRIVE_FILE_ID}, as published in "
                              "source_datasets/next_gqa/NExT-GQA/README.md)")
    parser.add_argument("--archive-cache-dir", type=Path, default=DEFAULT_ARCHIVE_CACHE_DIR,
                         help="Where the downloaded archive file is cached")
    parser.add_argument("--archive-extract-dir", type=Path, default=DEFAULT_ARCHIVE_EXTRACT_DIR,
                         help="Where the downloaded archive is extracted")
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

    if args.fetch_official_archive:
        if args.source_video_dir is not None or args.url_manifest is not None:
            raise SystemExit(
                "--fetch-official-archive cannot be combined with --source-video-dir or "
                "--url-manifest; pick a single video source."
            )
        if args.dry_run:
            print(
                f"[DRY RUN] Would download the official archive (Google Drive id="
                f"{args.gdrive_file_id}) to {args.archive_cache_dir}, extract it to "
                f"{args.archive_extract_dir}, then use that directory as the source for "
                "copy/symlink/hardlink placement. Nothing was downloaded."
            )
            return
        args.source_video_dir = fetch_official_archive(
            args.gdrive_file_id, args.archive_cache_dir, args.archive_extract_dir,
            resume=args.resume, overwrite=args.overwrite,
        )

    if args.source_video_dir is None and args.url_manifest is None:
        print(
            "No --fetch-official-archive, --source-video-dir, or --url-manifest given.\n"
            "NExT-GQA raw videos are distributed only via a single official Google Drive "
            "file (see source_datasets/next_gqa/NExT-GQA/README.md, section 'Preparation').\n"
            "Please either:\n"
            "  1) pass --fetch-official-archive to download+extract it automatically via "
            "gdown (pip install gdown), or\n"
            "  2) manually download+extract it yourself and pass its path via "
            "--source-video-dir, or\n"
            "  3) provide --url-manifest pointing at a JSON file of video_id -> URL you "
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

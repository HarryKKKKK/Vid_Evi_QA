"""
Delete videos from source_datasets/cg_bench/ that are not referenced
in cgbench_filtered.json.

Usage:
  python cleanup_cgbench_videos.py [--dry-run]
"""
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FILTERED_JSON = SCRIPT_DIR / "cgbench_filtered.json"
VIDEO_DIR = (SCRIPT_DIR / ".." / "source_datasets" / "cg_bench").resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without actually deleting")
    args = parser.parse_args()

    with open(FILTERED_JSON, encoding="utf-8") as f:
        filtered = json.load(f)

    needed = {entry["video_id"] for entry in filtered}
    print(f"Videos needed by cgbench_filtered.json: {len(needed)}")

    if not VIDEO_DIR.exists():
        print(f"ERROR: video directory not found: {VIDEO_DIR}")
        return

    all_videos = [f for f in VIDEO_DIR.iterdir() if f.is_file() and f.suffix == ".mp4"]
    print(f"Total mp4 files found in {VIDEO_DIR}: {len(all_videos)}")

    to_delete = [f for f in all_videos if f.stem not in needed]
    to_keep   = [f for f in all_videos if f.stem in needed]

    print(f"To keep:   {len(to_keep)}")
    print(f"To delete: {len(to_delete)}")

    if args.dry_run:
        print("\n[DRY RUN] Would delete:")
        for f in sorted(to_delete):
            print(f"  {f.name}")
        return

    deleted = 0
    for f in to_delete:
        f.unlink()
        deleted += 1

    print(f"\nDeleted {deleted} videos.")

    missing = needed - {f.stem for f in to_keep}
    if missing:
        print(f"\nWARNING: {len(missing)} videos in filtered list were not found:")
        for v in sorted(missing):
            print(f"  {v}")


if __name__ == "__main__":
    main()
"""
Unzip all video_chunk_*.zip from source_datasets/cg_bench/raw/
into source_datasets/cg_bench/.
Skips files that already exist.

Usage:
  python unzip_cgbench_videos.py [--dry-run]
"""
import zipfile
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / ".." / "source_datasets" / "cg_bench" / "raw"
OUTPUT_DIR = SCRIPT_DIR / ".." / "source_datasets" / "cg_bench"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    zips = sorted(RAW_DIR.glob("video_chunk_*.zip"))
    print(f"找到 {len(zips)} 个 zip 文件")

    extracted = 0
    skipped = 0

    for zip_path in zips:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            for name in tqdm(names, desc=zip_path.name, leave=False):
                target = OUTPUT_DIR / name
                if target.exists():
                    skipped += 1
                    continue
                if args.dry_run:
                    print(f"[DRY-RUN] {zip_path.name} -> {name}")
                    extracted += 1
                else:
                    zf.extract(name, OUTPUT_DIR)
                    extracted += 1

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}完成")
    print(f"  解压: {extracted}")
    print(f"  跳过（已存在）: {skipped}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Remove unparseable records from eval JSONL so resume retries only those items."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/nextgqa/insufficient/retry_unparseable_eval.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/nextgqa/insufficient/retry_unparseable_eval.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/nextgqa/insufficient', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    rows = []
    removed = []
    with args.jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "parse_error" in (row.get("model_parsed") or {}):
                removed.append({
                    "line_number": line_number,
                    "video_id": row.get("video_id"),
                    "qid": row.get("qid"),
                    "parse_error": row["model_parsed"].get("parse_error"),
                })
            else:
                rows.append(row)
    if not removed:
        print("No unparseable records found; JSONL unchanged.")
        return
    backup = args.jsonl.with_suffix(args.jsonl.suffix + ".before_parse_retry.bak")
    shutil.copy2(args.jsonl, backup)
    temporary = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.jsonl)
    print(json.dumps({
        "jsonl": str(args.jsonl),
        "backup": str(backup),
        "kept": len(rows),
        "removed_for_retry": removed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

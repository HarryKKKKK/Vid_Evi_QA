#!/usr/bin/env python3
"""Build expanded distractor requests for unresolved aligned conflict QA."""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/prepare_conflict_distractor_retry.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/prepare_conflict_distractor_retry.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=12)
    args = parser.parse_args()
    requests = {row["id"]: row for row in read_jsonl(args.requests)}
    candidates = {row["id"]: row for row in read_jsonl(args.candidates)}
    unresolved = json.loads(args.unresolved.read_text(encoding="utf-8"))
    rows = []
    for item in unresolved:
        item_id = item["id"]
        row = json.loads(json.dumps(requests[item_id]))
        row["candidate_count"] = args.candidate_count
        row["avoid_candidates"] = candidates.get(item_id, {}).get("candidates", [])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"retry_requests": len(rows), "candidate_count": args.candidate_count}))


if __name__ == "__main__":
    main()

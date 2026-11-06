#!/usr/bin/env python3
"""Remove clearly non-self-contained QA before visual conflict review."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/prepare_conflict_visual_review.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/prepare_conflict_visual_review.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, rejected = [], []
    for row in rows:
        reasons = []
        if row.get("official_action") is None and "official action" in str(row.get("question", "")).lower():
            reasons.append("undefined_official_action_in_question")
        if reasons:
            rejected.append({"qa": row, "reasons": reasons})
        else:
            kept.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "visual_review_input.jsonl").open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "prefilter_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "original": len(rows), "visual_review_input": len(kept), "prefilter_rejected": len(rejected),
        "rejected_by_type": dict(Counter(item["qa"]["conflict_type"] for item in rejected)),
    }
    (args.output_dir / "prefilter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

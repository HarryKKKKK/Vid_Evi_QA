#!/usr/bin/env python3
"""Archive scene-context predictions and remove them from active eval files."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/invalidate_scene_results.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/invalidate_scene_results.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.eval_json.read_text(encoding="utf-8"))
    kept = [x for x in rows if x.get("conflict_type", x.get("question_type")) != "scene_context"]
    removed = [x for x in rows if x.get("conflict_type", x.get("question_type")) == "scene_context"]
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive_dir / "conflict_eval.scene_v1_stale.json"
    archive.write_text(json.dumps(removed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.eval_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_file": str(args.eval_json),
        "archive_file": str(archive),
        "records_before": len(rows),
        "scene_predictions_archived": len(removed),
        "active_non_scene_predictions": len(kept),
        "reason": "scene_context QA v1 invalidated for atomic-option rebuild",
    }
    (args.archive_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Archive scene-context predictions and remove them from active eval files."""

from __future__ import annotations

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

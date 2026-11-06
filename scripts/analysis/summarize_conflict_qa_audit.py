#!/usr/bin/env python3
"""Merge open-model ETBench conflict QA judgments into a review report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    source = {str(row["id"]): row for row in sample}
    judgments = {}
    for path in args.judgment:
        for row in read_jsonl(path):
            judgments[(str(row["id"]), str(row["backend"]))] = row
    backends = sorted({backend for _, backend in judgments})
    merged = []
    for item_id, item in source.items():
        model_rows = {backend: judgments.get((item_id, backend)) for backend in backends}
        ok = {backend: row for backend, row in model_rows.items() if row and row.get("status") == "ok"}
        verdicts = [row["judgment"]["verdict"] for row in ok.values()]
        genuine = [bool(row["judgment"].get("genuine_conflict")) for row in ok.values()]
        same_slot = [bool(row["judgment"].get("same_semantic_slot")) for row in ok.values()]
        if len(ok) < len(backends):
            consensus = "incomplete"
        elif "major_issue" in verdicts or not all(genuine) or not all(same_slot):
            consensus = "review_major"
        elif "minor_issue" in verdicts or "uncertain" in verdicts:
            consensus = "review_minor_or_uncertain"
        else:
            consensus = "pass"
        merged.append({
            "id": item_id,
            "conflict_type": item.get("conflict_type"),
            "question": item.get("question"),
            "answers": [option.get("answer") for option in item.get("options", [])],
            "consensus": consensus,
            "judgments": model_rows,
        })

    by_type = defaultdict(Counter)
    for row in merged:
        by_type[row["conflict_type"]][row["consensus"]] += 1
    summary = {
        "sample_size": len(sample),
        "backends": backends,
        "completed_judgments": len(judgments),
        "consensus_counts": dict(Counter(row["consensus"] for row in merged)),
        "by_conflict_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "model_verdicts": {
            backend: dict(Counter(
                row["judgment"]["verdict"]
                for row in judgments.values()
                if row.get("backend") == backend and row.get("status") == "ok"
            ))
            for backend in backends
        },
        "interpretation": (
            "Text-only audit. Pass confirms semantic construction from metadata, not visual truth; "
            "flagged and uncertain cases should be checked against video frames."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "merged_judgments.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "review_queue.json").write_text(json.dumps(
        [row for row in merged if row["consensus"] != "pass"], ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate dense window judgments into item labels and an evidence-leak exclude list."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import atomic_write_json, latest_jsonl_records


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_classifications.*.jsonl")
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_item_audit.json"
DEFAULT_EXCLUDE = ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_evidence_leak_keys.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifications", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-output", type=Path, default=DEFAULT_EXCLUDE)
    return parser.parse_args()


def item_label(rows: list[dict]) -> str:
    labels = {str(row.get("label")) for row in rows if row.get("status") == "ok"}
    if any(row.get("status") != "ok" for row in rows):
        return "UNRESOLVED_ERROR"
    if "EVIDENCE_LEAK" in labels:
        return "EVIDENCE_LEAK"
    if "UNCERTAIN" in labels:
        return "UNCERTAIN"
    if "ANSWER_CORRELATED_ONLY" in labels:
        return "HALLUCINATION_INSUFFICIENT_GOLD_CUE"
    if "ANSWERABLE_WRONG" in labels:
        return "POSSIBLE_DISTRACTOR_INSUFFICIENT"
    return "ORDINARY_INSUFFICIENT"


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in glob.glob(args.classifications)]
    if not paths:
        raise SystemExit(f"no classification files match {args.classifications!r}")
    rows_by_key = latest_jsonl_records(paths)
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows_by_key.values():
        grouped[str(row.get("item_key"))].append(row)

    items = []
    counts: Counter[str] = Counter()
    for key, rows in sorted(grouped.items()):
        label = item_label(rows)
        counts[label] += 1
        first = rows[0]
        window_counts = Counter(
            str(row.get("label", "ERROR")) if row.get("status") == "ok" else "ERROR"
            for row in rows
        )
        notable = []
        for row in rows:
            if row.get("label") not in {"EVIDENCE_LEAK", "ANSWER_CORRELATED_ONLY", "ANSWERABLE_WRONG", "UNCERTAIN"}:
                continue
            candidate = row.get("candidate") or {}
            notable.append({
                "window_label": row.get("label"),
                "core_interval": candidate.get("core_interval") or candidate.get("interval"),
                "grounding_interval": candidate.get("grounding_interval"),
                "description": (row.get("grounding") or {}).get("description"),
                "decision": row.get("decision"),
            })
        items.append({
            "item_key": key,
            "video_id": first.get("video_id"),
            "qid": first.get("qid"),
            "question": first.get("question"),
            "question_type": first.get("question_type"),
            "reference_answer": first.get("reference_answer"),
            "item_label": label,
            "window_count": len(rows),
            "window_label_counts": dict(sorted(window_counts.items())),
            "notable_windows": notable,
        })

    report = {
        "unique_windows": len(rows_by_key),
        "items": len(items),
        "item_label_counts": dict(sorted(counts.items())),
        "label_policy": {
            "EVIDENCE_LEAK": "at least one window supports the complete question and gold answer",
            "HALLUCINATION_INSUFFICIENT_GOLD_CUE": "no leak, but at least one window visibly contains gold-answer-correlated content",
            "POSSIBLE_DISTRACTOR_INSUFFICIENT": "a window answers the complete question with a non-gold answer",
            "ORDINARY_INSUFFICIENT": "all windows are insufficient or irrelevant",
            "UNCERTAIN": "at least one unresolved semantic judgment",
            "UNRESOLVED_ERROR": "at least one window has a processing error",
        },
        "records": items,
    }
    atomic_write_json(args.output, report)
    leak_keys = [row["item_key"] for row in items if row["item_label"] == "EVIDENCE_LEAK"]
    args.exclude_output.parent.mkdir(parents=True, exist_ok=True)
    args.exclude_output.write_text("".join(f"{key}\n" for key in leak_keys), encoding="utf-8")
    print(json.dumps({
        "unique_windows": len(rows_by_key),
        "items": len(items),
        "item_label_counts": dict(sorted(counts.items())),
        "evidence_leak_exclude_keys": len(leak_keys),
        "report": str(args.output),
        "exclude_list": str(args.exclude_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

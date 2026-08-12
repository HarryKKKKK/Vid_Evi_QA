#!/usr/bin/env python3
"""Remove semantic evidence-leak items from the NExT-GQA evaluation metadata."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "nextgqa_pipeline/nextgqa_filtered.json"
DEFAULT_ELIGIBLE = str(ROOT / "nextgqa_pipeline/insufficient/leak_classifications.*.jsonl")
DEFAULT_DENSE = ROOT / "nextgqa_pipeline/insufficient/dense_ineligible_item_audit.json"
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json"
DEFAULT_AUDIT = ROOT / "nextgqa_pipeline/insufficient/combined_evidence_leak_exclusion_audit.json"
DEFAULT_KEYS = ROOT / "nextgqa_pipeline/insufficient/combined_evidence_leak_exclude_keys.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--eligible-classifications", default=DEFAULT_ELIGIBLE)
    parser.add_argument("--dense-audit", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--keys-output", type=Path, default=DEFAULT_KEYS)
    return parser.parse_args()


def item_key(row: dict) -> str:
    return f"{row['video_id']}_q{row['qid']}"


def load_latest_jsonl(pattern: str) -> dict[str, dict]:
    paths = [Path(path) for path in glob.glob(pattern)]
    if not paths:
        raise SystemExit(f"no eligible classification files match {pattern!r}")
    latest: dict[str, dict] = {}
    # Retry files may coexist with original files. Newer physical files win.
    for path in sorted(paths, key=lambda value: (value.stat().st_mtime, value.name)):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    latest[str(row["key"])] = row
                except (json.JSONDecodeError, KeyError) as exc:
                    raise SystemExit(f"invalid JSONL record {path}:{line_number}: {exc}") from exc
    return latest


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit(f"{args.source} must contain a JSON list")
    source_by_key: dict[str, dict] = {}
    for row in source:
        key = item_key(row)
        if key in source_by_key:
            raise SystemExit(f"duplicate source item: {key}")
        source_by_key[key] = row

    eligible_rows = load_latest_jsonl(args.eligible_classifications)
    eligible_leaks = {
        str(row["item_key"])
        for row in eligible_rows.values()
        if row.get("status") == "ok" and row.get("label") == "EVIDENCE_LEAK"
    }

    dense = json.loads(args.dense_audit.read_text(encoding="utf-8"))
    dense_records = dense.get("records")
    if not isinstance(dense_records, list):
        raise SystemExit(f"{args.dense_audit} has no records list")
    dense_leaks = {
        str(row["item_key"])
        for row in dense_records
        if row.get("item_label") == "EVIDENCE_LEAK"
    }

    sources_by_key: defaultdict[str, list[str]] = defaultdict(list)
    for key in eligible_leaks:
        sources_by_key[key].append("eligible_window_audit")
    for key in dense_leaks:
        sources_by_key[key].append("ineligible_dense_audit")
    excluded = set(sources_by_key)
    missing = sorted(excluded - set(source_by_key))
    if missing:
        raise SystemExit(f"{len(missing)} leak keys are absent from source JSON: {missing[:10]}")

    remaining = [row for row in source if item_key(row) not in excluded]
    excluded_rows = [source_by_key[key] for key in sorted(excluded)]
    question_type_counts = Counter(str(row.get("question_type", "UNKNOWN")) for row in excluded_rows)
    unique_excluded_videos = {str(row["video_id"]) for row in excluded_rows}

    audit_records = []
    for key in sorted(excluded):
        row = source_by_key[key]
        audit_records.append({
            "item_key": key,
            "video_id": row["video_id"],
            "qid": row["qid"],
            "question": row.get("question"),
            "question_type": row.get("question_type"),
            "sources": sorted(sources_by_key[key]),
        })

    audit = {
        "source_json": str(args.source),
        "output_json": str(args.output),
        "exclusion_unit": "qa_item_key_video_id_plus_qid",
        "source_items": len(source),
        "eligible_leak_items": len(eligible_leaks),
        "ineligible_dense_leak_items": len(dense_leaks),
        "cross_source_overlap": len(eligible_leaks & dense_leaks),
        "excluded_items": len(excluded),
        "excluded_unique_videos": len(unique_excluded_videos),
        "remaining_items": len(remaining),
        "excluded_question_type_counts": dict(sorted(question_type_counts.items())),
        "records": audit_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    args.keys_output.parent.mkdir(parents=True, exist_ok=True)
    args.keys_output.write_text("".join(f"{key}\n" for key in sorted(excluded)), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"exclude_keys: {args.keys_output}")
    print(f"audit: {args.audit_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize the aligned five-option benchmark after distractor verification."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/finalize_aligned_benchmark.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/finalize_aligned_benchmark.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


LETTERS = list("ABCDE")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def rng(item_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256((item_id + "::aligned-v1").encode()).hexdigest()[:16], 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    next_unique = read_jsonl(args.work_dir / "nextgqa_unique_qa.jsonl")
    next_conditions = read_jsonl(args.work_dir / "nextgqa_condition_records.jsonl")
    # The benchmark evaluates whether the supplied video is sufficient for a
    # unique answer.  Preserve the gold option(s) for conditional option
    # scoring, but make the top-level expected decision explicit.
    for row in next_unique:
        row["expected_behavior"] = "answer"
    for row in next_conditions:
        condition = row.get("evidence_condition")
        row["expected_behavior"] = (
            "answer" if condition == "sufficient" else "reject"
        )
        if condition in {"insufficient", "partially_sufficient"}:
            row["expected_reject_reason"] = "INSUFFICIENT_EVIDENCE"
    prepared = read_jsonl(args.work_dir / "conflict_prepared.jsonl")
    groups = {row["job_id"]: row for row in read_jsonl(args.groups)}
    verification_paths = sorted(args.work_dir.glob("conflict_distractor_verification*.jsonl"))
    verified_rows = [row for path in verification_paths for row in read_jsonl(path)]
    verified = {}
    for row in verified_rows:
        if row.get("slot_fit") and row.get("verdict") == "NOT_SUPPORTED" \
                and float(row.get("confidence", 0)) >= args.min_confidence:
            verified.setdefault(row["id"], []).append(row)
    for rows in verified.values():
        rows.sort(key=lambda x: (-float(x["confidence"]), int(x["candidate_index"])))

    conflict_final, unresolved = [], []
    for item in prepared:
        qa = item["qa"]
        options = json.loads(json.dumps(item["selected_valid_options"]))
        distractors = json.loads(json.dumps(item["deterministic_distractors"]))
        needed = 5 - len(options) - len(distractors)
        available = verified.get(qa["id"], [])
        seen = {str(option["answer"]).casefold() for option in options + distractors}
        for candidate in available:
            text = str(candidate["candidate"]).strip()
            if not text or text.casefold() in seen:
                continue
            distractors.append({"option_id": "GEN", "answer": text, "supported_by": [],
                                "is_valid": False, "distractor_source": "gpt_generated_qwen3vl_verified",
                                "verification_confidence": candidate["confidence"],
                                "verification_key": candidate["verification_key"]})
            seen.add(text.casefold())
            if len(distractors) >= 5 - len(options):
                break
        if len(options) + len(distractors) < 5:
            unresolved.append({"id": qa["id"], "needed": needed,
                               "verified_available": len(available), "qa": qa})
            continue
        combined = options + distractors[:5 - len(options)]
        rng(qa["id"]).shuffle(combined)
        normalized = []
        valid_ids = []
        for index, option in enumerate(combined):
            option = json.loads(json.dumps(option))
            option["option_id"] = LETTERS[index]
            normalized.append(option)
            if option.get("is_valid"):
                valid_ids.append(LETTERS[index])
        row = json.loads(json.dumps(qa))
        row.update({
            "source_dataset": "etbench",
            "video": groups[qa["parent_job_id"]]["video"],
            "video_duration": groups[qa["parent_job_id"]]["video_duration"],
            "options": normalized,
            "choices": [option["answer"] for option in normalized],
            "valid_option_ids": valid_ids,
            "answer_mode": "multiple_valid_answers",
            "evidence_condition": "conflicting",
            "expected_behavior": "reject",
            "expected_reject_reason": "CONFLICTING_EVIDENCE",
            "alignment_version": "five_option_v1",
        })
        conflict_final.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "nextgqa_unique_qa.jsonl", next_unique)
    write_jsonl(args.output_dir / "nextgqa_condition_records.jsonl", next_conditions)
    write_jsonl(args.output_dir / "conflict_qa_five_options.jsonl", conflict_final)
    unique = next_unique + conflict_final
    condition_records = next_conditions + conflict_final
    write_jsonl(args.output_dir / "all_unique_qa.jsonl", unique)
    write_jsonl(args.output_dir / "all_condition_records.jsonl", condition_records)
    (args.output_dir / "unresolved_alignment.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    valid_position_counts = Counter(
        option_id for row in conflict_final for option_id in row["valid_option_ids"])
    summary = {
        "status": "complete" if not unresolved else "incomplete",
        "verification_files": [path.name for path in verification_paths],
        "nextgqa_unique_qa": len(next_unique),
        "nextgqa_condition_records": len(next_conditions),
        "conflict_qa": len(conflict_final),
        "unresolved_conflict_qa": len(unresolved),
        "all_unique_qa": len(unique),
        "all_condition_records": len(condition_records),
        "all_records_have_five_options": all(len(row["options"]) == 5 for row in unique),
        "conflict_valid_answer_count_distribution": dict(sorted(Counter(
            len(row["valid_option_ids"]) for row in conflict_final).items())),
        "conflict_valid_position_counts": dict(sorted(valid_position_counts.items())),
        "conflict_type_counts": dict(sorted(Counter(row["conflict_type"] for row in conflict_final).items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if unresolved:
        raise SystemExit(f"{len(unresolved)} conflict QA lack enough verified distractors")


if __name__ == "__main__":
    main()

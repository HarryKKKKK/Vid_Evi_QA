#!/usr/bin/env python3
"""Build the worklist that is allowed to enter calibrated window scanning.

The full official-evidence-frozen video is evaluated first.  UNANSWERABLE and
wrong-answer results pass the gate.  A correct ANSWERABLE result must have its
claimed evidence interval grounded and semantically classified.  Only a
confirmed EVIDENCE_LEAK is excluded; incomplete records stop publication of
the gate output instead of being silently treated as clean.
"""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/nextgqa/insufficient/build_pre_window_gate_json.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/nextgqa/insufficient/build_pre_window_gate_json.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/nextgqa/insufficient', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "results/build/nextgqa/nextgqa_filtered.json"
DEFAULT_OUTPUT = ROOT / "results/build/nextgqa/nextgqa_filtered_pre_window_pass.json"
DEFAULT_AUDIT = ROOT / "results/build/nextgqa/insufficient/pre_window_gate_audit.json"
DEFAULT_EXCLUDED_KEYS = ROOT / "results/build/nextgqa/insufficient/pre_window_gate_exclude_keys.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evaluation-results", action="append", required=True)
    parser.add_argument("--classifications", required=True, help="JSONL path or glob")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--excluded-keys-output", type=Path, default=DEFAULT_EXCLUDED_KEYS)
    return parser.parse_args()


def item_key(row: dict) -> str:
    return f"{row['video_id']}_q{row['qid']}"


def normal_letter(value: object) -> str | None:
    match = re.search(r"\b([A-E])\b", str(value or "").strip().upper())
    return match.group(1) if match else None


def read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        value = value["results"]
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected JSON list")
    return value


def expand(patterns: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        if not matches:
            raise ValueError(f"no files match {pattern!r}")
        for path in matches:
            paths[str(path.resolve())] = path.resolve()
    return sorted(paths.values(), key=lambda path: (path.stat().st_mtime, str(path)))


def latest_evaluations(paths: list[Path]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for path in paths:
        for row in read_json_or_jsonl(path):
            if "video_id" in row and "qid" in row:
                latest[item_key(row)] = row
    return latest


def latest_classifications(paths: list[Path]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for path in paths:
        for row in read_json_or_jsonl(path):
            key = row.get("item_key")
            if key is not None:
                latest[str(key)] = row
    return latest


def valid_interval(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    try:
        numbers = [float(part) for part in value]
    except (TypeError, ValueError):
        return False
    return all(number == number and abs(number) != float("inf") for number in numbers)


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit(f"{args.source} must be a JSON list")
    source_by_key = {item_key(row): row for row in source}
    if len(source_by_key) != len(source):
        raise SystemExit("source contains duplicate video_id/qid keys")

    evaluation_paths = expand(args.evaluation_results)
    classification_paths = expand([args.classifications])
    evaluations = latest_evaluations(evaluation_paths)
    classifications = latest_classifications(classification_paths)

    decisions: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    excluded: set[str] = set()
    unresolved: list[str] = []
    for key, source_row in source_by_key.items():
        evaluation = evaluations.get(key)
        if evaluation is None:
            decision, detail = "UNRESOLVED", "missing_evaluation"
        else:
            parsed = evaluation.get("model_parsed") or {}
            label = str(parsed.get("label") or "").strip().upper()
            predicted = normal_letter(parsed.get("answer"))
            gold = normal_letter(evaluation.get("answer", source_row.get("answer")))
            if parsed.get("parse_error") or label not in {"ANSWERABLE", "UNANSWERABLE"}:
                decision, detail = "UNRESOLVED", "invalid_evaluation_response"
            elif label == "UNANSWERABLE":
                decision, detail = "PASS", "model_unanswerable"
            elif predicted is None or gold is None:
                decision, detail = "UNRESOLVED", "invalid_answer_letter"
            elif predicted != gold:
                decision, detail = "PASS", "answerable_but_wrong"
            elif not valid_interval(parsed.get("evidence_seconds")):
                decision, detail = "UNRESOLVED", "correct_answer_missing_evidence_interval"
            else:
                classification = classifications.get(key)
                if classification is None:
                    decision, detail = "UNRESOLVED", "missing_claimed_interval_classification"
                elif classification.get("status") != "ok":
                    decision, detail = "UNRESOLVED", "classification_not_ok"
                elif classification.get("label") == "EVIDENCE_LEAK":
                    decision, detail = "EXCLUDE", "confirmed_evidence_leak"
                else:
                    decision = "PASS"
                    detail = f"audited_{str(classification.get('label')).lower()}"

        counts[f"{decision}:{detail}"] += 1
        decisions[key] = {
            "item_key": key,
            "video_id": source_row["video_id"],
            "qid": source_row["qid"],
            "question_type": source_row.get("question_type"),
            "decision": decision,
            "reason": detail,
        }
        if decision == "EXCLUDE":
            excluded.add(key)
        elif decision == "UNRESOLVED":
            unresolved.append(key)

    audit = {
        "source_json": str(args.source),
        "evaluation_inputs": [str(path) for path in evaluation_paths],
        "classification_inputs": [str(path) for path in classification_paths],
        "source_items": len(source),
        "passed_items": sum(row["decision"] == "PASS" for row in decisions.values()),
        "excluded_items": len(excluded),
        "unresolved_items": len(unresolved),
        "decision_reason_counts": dict(sorted(counts.items())),
        "records": [decisions[key] for key in sorted(decisions)],
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    args.excluded_keys_output.write_text(
        "".join(f"{key}\n" for key in sorted(excluded)), encoding="utf-8"
    )

    if unresolved:
        args.output.unlink(missing_ok=True)
        print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))
        raise SystemExit(
            f"gate has {len(unresolved)} unresolved items; output was not published; "
            f"first keys: {unresolved[:10]}"
        )

    remaining = [row for row in source if decisions[item_key(row)]["decision"] == "PASS"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))
    print(f"published: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export correct insufficient-condition answers and claimed evidence times.

The output is the auditable front end for the existing
``ground_correct_answer_evidence.py -> classify_groundings.py`` leak pipeline.
It never treats a missing or invalid evidence interval as evidence absence.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


LETTERS = set("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="CONDITION:MODEL=PATH",
        help="Repeat once per evaluation result. CONDITION is normally insufficient or proxy_a000.",
    )
    parser.add_argument(
        "--classifications",
        action="append",
        default=[],
        help="Optional classification JSONL path/glob; repeatable.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{number}: expected object")
                    rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("results")
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected JSON array")
    return value


def letter(value: object) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else None


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def parse_result_spec(value: str) -> tuple[str, str, Path]:
    try:
        left, filename = value.split("=", 1)
        condition, model = left.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --result {value!r}; expected CONDITION:MODEL=PATH"
        ) from exc
    if not condition.strip() or not model.strip() or not filename.strip():
        raise argparse.ArgumentTypeError(f"invalid --result {value!r}")
    path = Path(filename).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"result does not exist: {path}")
    return condition.strip(), model.strip(), path


def interval_status(value: object, duration: object) -> tuple[str, list[float] | None]:
    if not isinstance(value, (list, tuple)) or not value:
        return "missing", None
    try:
        timestamps = [float(part) for part in value]
    except (TypeError, ValueError):
        return "non_numeric", None
    if not all(math.isfinite(part) for part in timestamps):
        return "non_finite", None
    interval = [min(timestamps), max(timestamps)]
    try:
        video_duration = float(duration)
    except (TypeError, ValueError):
        return "missing_duration", interval
    if not math.isfinite(video_duration) or video_duration <= 0:
        return "missing_duration", interval
    if interval[1] < 0 or interval[0] > video_duration:
        return "outside_video", interval
    if interval[0] < 0 or interval[1] > video_duration:
        return "partly_outside_video", interval
    return "valid", interval


def latest_classifications(patterns: list[str]) -> dict[str, dict]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match).resolve()
            paths[str(path)] = path
    latest: dict[str, dict] = {}
    for path in sorted(paths.values(), key=lambda item: (item.stat().st_mtime, str(item))):
        for row in read_rows(path):
            key = str(row.get("key") or "")
            if key:
                latest[key] = row
    return latest


def summarize(rows: list[dict]) -> dict:
    interval_counts = Counter(row["evidence_interval_status"] for row in rows)
    semantic_status = Counter(row["semantic_audit_status"] for row in rows)
    semantic_labels = Counter(
        row["semantic_leak_label"]
        for row in rows
        if row["semantic_audit_status"] == "ok"
    )
    return {
        "correct_answered": len(rows),
        "unique_items": len({row["item_key"] for row in rows}),
        "evidence_interval_status": dict(sorted(interval_counts.items())),
        "auditable_claims": interval_counts["valid"] + interval_counts["partly_outside_video"],
        "semantic_audit_status": dict(sorted(semantic_status.items())),
        "semantic_label_counts": dict(sorted(semantic_labels.items())),
        "confirmed_leaks": semantic_labels["EVIDENCE_LEAK"],
    }


def main() -> None:
    args = parse_args()
    specs = [parse_result_spec(value) for value in args.result]
    classifications = latest_classifications(args.classifications)
    output_rows: list[dict] = []

    for condition, model_name, path in specs:
        for source in read_rows(path):
            parsed = source.get("model_parsed") or {}
            gold = letter(source.get("answer"))
            prediction = letter(parsed.get("answer"))
            if str(parsed.get("label") or "").strip().upper() != "ANSWERABLE":
                continue
            if gold is None or prediction != gold:
                continue
            model_tag = str(source.get("model_tag") or model_name).strip()
            video_id, qid = str(source.get("video_id")), str(source.get("qid"))
            key = f"{safe_tag(model_tag)}::{video_id}::{qid}"
            status, interval = interval_status(
                parsed.get("evidence_seconds"), source.get("video_duration")
            )
            semantic = classifications.get(key)
            output_rows.append({
                "key": key,
                "item_key": f"{video_id}_q{qid}",
                "condition": condition,
                "model_name": model_name,
                "model_tag": model_tag,
                "source_result_file": str(path),
                "video_id": video_id,
                "qid": qid,
                "question_type": source.get("question_type"),
                "question": source.get("question"),
                "gold_answer": gold,
                "predicted_answer": prediction,
                "model_evidence": parsed.get("evidence"),
                "model_evidence_seconds_raw": parsed.get("evidence_seconds"),
                "model_evidence_interval": interval,
                "evidence_interval_status": status,
                "video_duration": source.get("video_duration"),
                "sampled_seconds": source.get("sampled_seconds"),
                "model_reason": parsed.get("reason"),
                "semantic_audit_status": (
                    str(semantic.get("status") or "missing") if semantic else "not_run"
                ),
                "semantic_leak_label": semantic.get("label") if semantic else None,
                "semantic_failure_mode": (
                    (semantic.get("decision") or {}).get("failure_mode") if semantic else None
                ),
            })

    output_rows.sort(key=lambda row: (row["model_name"], row["video_id"], row["qid"]))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in output_rows:
        by_model[row["model_name"]].append(row)
    report = {
        "definition": "model labeled ANSWERABLE and predicted the gold A-E option",
        "warning": "missing/invalid evidence_seconds means unauditable, not leak-free",
        "overall": summarize(output_rows),
        "by_model": {model: summarize(rows) for model, rows in sorted(by_model.items())},
        "result_files": [
            {"condition": condition, "model": model, "path": str(path)}
            for condition, model, path in specs
        ],
        "classification_patterns": args.classifications,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

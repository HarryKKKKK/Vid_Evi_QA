#!/usr/bin/env python3
"""Reconstruct every Qwen3 leak-audit route for selected manual-review items."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSUFFICIENT = ROOT / "nextgqa_pipeline/insufficient"
sys.path.insert(0, str(INSUFFICIENT))
from ground_windows import candidate_windows  # noqa: E402


def latest(pattern: str) -> dict[str, dict]:
    records = {}
    for filename in sorted(glob.glob(pattern), key=os.path.getmtime):
        with open(filename, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    records[str(row.get("key"))] = row
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compact_grounding(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "status": row.get("status"), "candidate": row.get("candidate"),
        "source_model_claim": row.get("source_model_claim"),
        "description": (row.get("grounding") or {}).get("description"),
        "grounder_model": (row.get("grounding") or {}).get("model"),
    }


def compact_classification(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "status": row.get("status"), "label": row.get("label"),
        "decision": row.get("decision"),
        "reference_answer_presence": row.get("reference_answer_presence"),
        "error": row.get("error"), "classifier_model": row.get("classifier_model"),
    }


def main() -> None:
    args = parse_args()
    measurements = {}
    for filename in glob.glob(str(INSUFFICIENT / "measurements.*.jsonl")):
        with open(filename, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                measurements[str(row.get("key"))] = row
    thresholds = json.loads((INSUFFICIENT / "thresholds.json").read_text(encoding="utf-8"))
    qwen_rows = {
        f"{row['video_id']}_q{row['qid']}": row
        for row in json.loads((ROOT / "nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json").read_text(encoding="utf-8"))
    }
    groundings = latest(str(INSUFFICIENT / "correct_answer_evidence_groundings*.jsonl"))
    classifications = latest(str(INSUFFICIENT / "correct_answer_evidence_classifications8.*.jsonl"))
    dense_payload = json.loads((INSUFFICIENT / "dense_ineligible_item_audit.json").read_text(encoding="utf-8"))
    dense = {str(row["item_key"]): row for row in dense_payload.get("records", [])}

    report = []
    for item in args.item:
        measurement = measurements[item]
        model = measurement["models"]["qwen3vl_construct"]
        eligible = bool(model.get("eligible"))
        candidates = candidate_windows(measurement, thresholds, 0.05) if eligible else []
        qwen = qwen_rows.get(item)
        parsed = (qwen or {}).get("model_parsed") or {}
        gold = str((qwen or {}).get("answer") or "").upper()
        answered_correctly = (
            str(parsed.get("label") or "").upper() == "ANSWERABLE"
            and str(parsed.get("answer") or "").upper() == gold
        )
        claim_key = (
            f"qwen3vl_no_evidence_leak::{measurement['video_id']}::{measurement['qid']}"
        )
        report.append({
            "item_key": item,
            "question": measurement.get("question"),
            "question_type": measurement.get("question_type"),
            "gold": measurement.get("gold"),
            "official_evidence": measurement.get("official_evidence"),
            "actual_initial_mask": measurement.get("initial_mask"),
            "behavioral_route": {
                "eligible": eligible,
                "full_prediction": ((model.get("full") or {}).get("visual") or {}).get("prediction"),
                "full_gain": (model.get("full") or {}).get("gain"),
                "gold_only_prediction": ((model.get("gold_only") or {}).get("visual") or {}).get("prediction"),
                "gold_only_gain": (model.get("gold_only") or {}).get("gain"),
                "candidate_count": len(candidates),
                "selected_top5": candidates[:5],
            },
            "dense_fallback": dense.get(item),
            "qwen3_downstream_output": {
                "answered_correctly": answered_correctly,
                "model_parsed": parsed,
            } if qwen else None,
            "correct_answer_claim_audit": {
                "grounding": compact_grounding(groundings.get(claim_key)),
                "classification": compact_classification(classifications.get(claim_key)),
            } if answered_correctly else None,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

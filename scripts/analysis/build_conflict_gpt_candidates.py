#!/usr/bin/env python3
"""Apply conservative structural, text, and dual-VLM gates before GPT review."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def temporal_overlap(row: dict) -> bool:
    intervals = []
    for option in row.get("options", []):
        for support in option.get("supported_by", []):
            value = support.get("interval")
            if isinstance(value, list) and len(value) >= 2:
                intervals.append((float(value[0]), float(value[1]), str(option.get("option_id"))))
    intervals.sort()
    return any(intervals[index][0] < intervals[index - 1][1] for index in range(1, len(intervals)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--text-audit", type=Path, required=True)
    parser.add_argument("--visual", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qa = read_jsonl(args.qa)
    text = {str(row["id"]): row for row in read_json(args.text_audit)}
    visual_models = {}
    for path in args.visual:
        rows = read_json(path)
        model_name = str(rows[0].get("model_tag", path.parent.name)) if rows else path.parent.name
        visual_models[model_name] = {str(row["id"]): row for row in rows}

    candidates, rejected = [], []
    stage_counts = Counter()
    type_counts = defaultdict(Counter)
    for row in qa:
        item_id = str(row["id"])
        reasons = []
        review_flags = []
        visual_reports = {}
        supported_union = set()
        parsed_visual_models = 0
        for model_name, model_rows in visual_models.items():
            report = model_rows.get(item_id)
            visual_reports[model_name] = report
            if not report or report.get("parse_error") is not None:
                review_flags.append(f"{model_name}:visual_parse_failure")
            else:
                parsed_visual_models += 1
                supported_union.update(map(str, report.get("predicted_option_ids", [])))
                if report.get("predicted_label") != "CONFLICT":
                    review_flags.append(f"{model_name}:not_conflict")
                if not report.get("option_set_exact"):
                    review_flags.append(f"{model_name}:option_set_not_exact")

        expected = set(map(str, row.get("valid_option_ids", [])))
        missing_options = sorted(expected - supported_union)
        if parsed_visual_models == 0:
            reasons.append("visual:no_successful_model")
        if missing_options:
            reasons.append("visual:options_without_support:" + ",".join(missing_options))

        text_row = text.get(item_id)
        qwen = (text_row or {}).get("judgments", {}).get("qwen")
        if qwen and qwen.get("status") == "ok" and qwen["judgment"].get("verdict") != "pass":
            review_flags.append("qwen_text:not_pass")
        if row.get("conflict_type") == "temporal_localization" and temporal_overlap(row):
            review_flags.append("temporal:overlapping_option_intervals")

        audit = {
            "id": item_id,
            "conflict_type": row.get("conflict_type"),
            "qa": row,
            "text_audit": qwen,
            "visual_audits": visual_reports,
            "gate_reasons": sorted(set(reasons)),
            "review_flags": sorted(set(review_flags)),
            "visually_supported_option_ids": sorted(supported_union),
        }
        if reasons:
            rejected.append(audit)
            for reason in set(reasons):
                stage_counts[reason] += 1
                type_counts[str(row.get("conflict_type"))][reason] += 1
        else:
            candidates.append(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpt_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "pre_gpt_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "original": len(qa),
        "gpt_candidates": len(candidates),
        "pre_gpt_rejected": len(rejected),
        "visual_models": sorted(visual_models),
        "rejection_reason_counts": dict(stage_counts.most_common()),
        "rejection_reasons_by_type": {name: dict(values.most_common()) for name, values in sorted(type_counts.items())},
    }
    (args.output_dir / "pre_gpt_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

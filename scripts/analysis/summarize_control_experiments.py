#!/usr/bin/env python3
"""Summarize resolution/prompt, blind, and operator-control JSONL outputs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def response_json(row: dict) -> dict:
    text = str((row.get("response") or {}).get("text") or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        value = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    return value if isinstance(value, dict) else {}


def answer_letter(value: object) -> str | None:
    match = re.search(r"\b([A-E])\b", str(value or "").upper())
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    paths = sorted({Path(path) for pattern in args.inputs for path in glob.glob(pattern)})
    groups: defaultdict[tuple, list[dict]] = defaultdict(list)
    errors = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                errors += 1
                continue
            group = (
                str(row.get("model_tag") or "qwen25vl32b"),
                str(row.get("experiment_tag") or ""),
                str(row.get("max_pixels") or ""),
                str(row.get("condition") or ""),
                str(row.get("experiment_group") or ""),
            )
            parsed = response_json(row)
            label = str(parsed.get("label") or "").upper()
            answer = answer_letter(parsed.get("answer") or parsed.get("final_answer"))
            forced = "forced" in str(row.get("condition"))
            schema_valid = bool(answer) if forced else label in {"ANSWERABLE", "UNANSWERABLE"}
            answered = forced or label == "ANSWERABLE"
            groups[group].append({
                "answered": answered,
                "correct": answered and answer == str(row.get("answer", "")).upper(),
                "parsed": bool(answer or label),
                "schema_valid": schema_valid,
            })
    records = []
    for group, rows in sorted(groups.items()):
        n = len(rows)
        answered = sum(row["answered"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        records.append({
            "model_tag": group[0], "experiment_tag": group[1], "max_pixels": group[2],
            "condition": group[3], "experiment_group": group[4], "n": n,
            "coverage": answered / n if n else None,
            "overall_accuracy": correct / n if n else None,
            "conditional_accuracy": correct / answered if answered else None,
            "parse_rate": sum(row["parsed"] for row in rows) / n if n else None,
            "valid_schema_rate": sum(row["schema_valid"] for row in rows) / n if n else None,
            "invalid_schema": sum(not row["schema_valid"] for row in rows),
        })
    report = {"files": [str(path) for path in paths], "errors": errors, "records": records}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

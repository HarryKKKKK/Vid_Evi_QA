#!/usr/bin/env python3
"""Export deduplicated QA/window grounding records to readable Markdown and JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latest: dict[str, dict] = {}
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                latest[str(row["key"])] = row
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid record at line {line_number}: {exc}") from exc

    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    errors: list[dict] = []
    for row in latest.values():
        if row.get("status") != "ok":
            errors.append({"key": row.get("key"), "error": row.get("error")})
            continue
        grouped[str(row["item_key"])].append(row)

    output: list[dict] = []
    for item_key, rows in sorted(
        grouped.items(),
        key=lambda pair: (min(int(row.get("item_index", 10**9)) for row in pair[1]), pair[0]),
    ):
        rows.sort(key=lambda row: (int(row.get("candidate_rank", 10**9)), row["candidate"]["interval"]))
        first = rows[0]
        output.append({
            "item_key": item_key,
            "question_type": first.get("question_type"),
            "question": first.get("question"),
            "answer": first.get("reference_answer"),
            "official_evidence": first.get("official_evidence"),
            "windows": [
                {
                    "rank": row.get("candidate_rank"),
                    "interval": row.get("candidate", {}).get("interval"),
                    "masked_intervals": row.get("masked_intervals"),
                    "risk_score": row.get("candidate", {}).get("risk_score"),
                    "grounding_description": row.get("grounding", {}).get("description"),
                }
                for row in rows
            ],
        })

    payload = {
        "qa_count": len(output),
        "window_count": sum(len(item["windows"]) for item in output),
        "error_count": len(errors),
        "items": output,
        "errors": errors,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# NExT-GQA dangerous-window groundings",
        "",
        f"- QA count: {payload['qa_count']}",
        f"- Window count: {payload['window_count']}",
        f"- Error count: {payload['error_count']}",
        "",
    ]
    for index, item in enumerate(output, start=1):
        lines += [
            f"## {index}. {item['item_key']}",
            "",
            f"- Type: `{item['question_type']}`",
            f"- Question: {item['question']}",
            f"- Gold answer: {item['answer']}",
            f"- Official evidence: `{json.dumps(item['official_evidence'])}`",
            "",
        ]
        for window in item["windows"]:
            lines += [
                f"### Candidate {window['rank']}: `{json.dumps(window['interval'])}`",
                "",
                f"- Risk score: `{window['risk_score']}`",
                f"- Masked intervals: `{json.dumps(window['masked_intervals'])}`",
                f"- Grounding: {window['grounding_description']}",
                "",
            ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"Exported {payload['qa_count']} QAs and {payload['window_count']} windows; "
        f"errors={payload['error_count']}"
    )
    print(f"Markdown: {args.markdown}")
    print(f"JSON: {args.json}")


if __name__ == "__main__":
    main()

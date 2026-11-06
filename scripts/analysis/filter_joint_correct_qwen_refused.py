#!/usr/bin/env python3
"""Filter sufficient cases refused by Qwen but correctly answered by both peers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--internvl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def prediction(row: dict) -> tuple[str, str | None]:
    parsed = row.get("model_parsed") or {}
    label = str(parsed.get("label") or "").strip().upper()
    answer = parsed.get("answer")
    return label, str(answer).strip().upper() if answer is not None else None


def correct(row: dict) -> bool:
    label, answer = prediction(row)
    return label == "ANSWERABLE" and answer == str(row.get("answer") or "").upper()


def model_record(row: dict) -> dict:
    return {
        "model_parsed": row.get("model_parsed"),
        "model_response": row.get("model_response") or row.get("response"),
    }


def main() -> None:
    args = parse_args()
    qwen = {key(row): row for row in load(args.qwen)}
    llava = {key(row): row for row in load(args.llava)}
    internvl = {key(row): row for row in load(args.internvl)}
    common = set(qwen) & set(llava) & set(internvl)

    selected = []
    for item_key in sorted(common):
        qrow, lrow, irow = qwen[item_key], llava[item_key], internvl[item_key]
        if prediction(qrow)[0] != "UNANSWERABLE" or not correct(lrow) or not correct(irow):
            continue
        selected.append(
            {
                "video_id": str(qrow["video_id"]),
                "qid": str(qrow["qid"]),
                "question_type": qrow.get("question_type"),
                "question": qrow.get("question"),
                "choices": qrow.get("choices"),
                "answer": qrow.get("answer"),
                "sampled_seconds": qrow.get("sampled_seconds"),
                "qwen2_5": model_record(qrow),
                "llava_onevision": model_record(lrow),
                "internvl3_5": model_record(irow),
            }
        )

    summary = {
        "common_sufficient_cases": len(common),
        "selected_cases": len(selected),
        "selection": "Qwen2.5 UNANSWERABLE; LLaVA-OneVision and InternVL3.5 both ANSWERABLE and gold-correct",
        "question_type_counts": dict(Counter(str(row.get("question_type")) for row in selected).most_common()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

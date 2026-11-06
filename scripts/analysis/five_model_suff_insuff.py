#!/usr/bin/env python3
"""Compare retained Video-LLMs on sufficient and insufficient evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


RUNS = {
    "Qwen3-VL-32B": {
        "sufficient": "results/nextgqa_result_qwen3vl_partial_no_evidence_leak/sufficient_alpha1.classify.json",
        "insufficient": "results/nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json",
    },
    "Qwen2.5-VL-32B": {
        "sufficient": "results/nextgqa_result_qwen25_32f_sufficient/sufficient.classify.json",
        "insufficient": "results/nextgqa_result_qwen25_32f_insufficient/insufficient.classify.json",
    },
    "LLaVA-OneVision-7B": {
        "sufficient": "results/nextgqa_result_llava_onevision_32f_video_sufficient/sufficient.classify.json",
        "insufficient": "results/nextgqa_result_llava_onevision_32f_video_insufficient/insufficient.classify.json",
    },
    "InternVL3.5-8B": {
        "sufficient": "results/nextgqa_result_internvl35_32f_patch3_sufficient/sufficient.classify.json",
        "insufficient": "results/nextgqa_result_internvl35_32f_patch3_insufficient/insufficient.classify.json",
    },
}


def load_rows(root: Path, relative_path: str) -> dict[tuple[str, str], dict]:
    rows = json.loads((root / relative_path).read_text(encoding="utf-8"))
    return {
        (str(row["video_id"]), str(row["qid"])): row
        for row in rows
        if str(row.get("answer", "")).strip().upper() in set("ABCDE")
        and len(str(row.get("answer", "")).strip()) == 1
    }


def normalized_prediction(row: dict) -> tuple[str, str | None]:
    parsed = row.get("model_parsed") or {}
    label = str(parsed.get("label") or "").strip().upper()
    answer = parsed.get("answer")
    if label in set("ABCDE") and len(label) == 1:
        return "ANSWERABLE", label
    if isinstance(answer, str):
        match = re.match(r"\s*([A-E])(?:\b|[.:)])", answer.upper())
        if match:
            answer = match.group(1)
    raw = str(parsed.get("raw") or row.get("model_response") or "")
    if label not in ("ANSWERABLE", "UNANSWERABLE"):
        match = re.search(
            r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)''',
            raw,
            re.IGNORECASE,
        )
        if match:
            label = match.group(1).upper()
    if not (isinstance(answer, str) and answer in set("ABCDE") and len(answer) == 1):
        match = re.search(
            r'''["']answer["']\s*:\s*["']([A-E])''', raw, re.IGNORECASE
        )
        answer = match.group(1).upper() if match else None
    return label, answer


def metrics(rows_by_key: dict, keys=None) -> dict:
    if keys is None:
        keys = rows_by_key.keys()
    rows = [rows_by_key[key] for key in keys if key in rows_by_key]
    answered = correct = invalid = 0
    for row in rows:
        label, answer = normalized_prediction(row)
        if label == "ANSWERABLE":
            answered += 1
            correct += answer == str(row["answer"]).strip().upper()
        elif label != "UNANSWERABLE":
            invalid += 1
    n = len(rows)
    refused = n - answered - invalid
    return {
        "n": n,
        "answered": answered,
        "refused": refused,
        "invalid": invalid,
        "coverage": 100.0 * answered / n,
        "refusal": 100.0 * refused / n,
        "accuracy": 100.0 * correct / n,
        "conditional_accuracy": 100.0 * correct / answered if answered else 0.0,
    }


def print_row(prefix: str, values: dict) -> None:
    print(
        prefix,
        " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in values.items()
        ),
    )


def main() -> None:
    root = Path.cwd()
    data = {
        model: {condition: load_rows(root, path) for condition, path in paths.items()}
        for model, paths in RUNS.items()
    }

    print("AVAILABLE")
    for model, conditions in data.items():
        for condition, rows in conditions.items():
            print_row(f"{model}|{condition}|", metrics(rows))

    common = set.intersection(
        *(set(rows) for conditions in data.values() for rows in conditions.values())
    )
    print(f"COMMON_N|{len(common)}")
    print("COMMON")
    for model, conditions in data.items():
        by_condition = {}
        for condition, rows in conditions.items():
            by_condition[condition] = metrics(rows, common)
            print_row(f"{model}|{condition}|", by_condition[condition])
        sufficient = by_condition["sufficient"]
        insufficient = by_condition["insufficient"]
        print(
            f"{model}|delta|"
            f"coverage_drop={sufficient['coverage'] - insufficient['coverage']:.4f} "
            f"refusal_gain={insufficient['refusal'] - sufficient['refusal']:.4f} "
            f"accuracy_drop={sufficient['accuracy'] - insufficient['accuracy']:.4f}"
        )

    qwen = data["Qwen2.5-VL-32B"]["sufficient"]
    qwen_refusals = []
    for key in common:
        label, _ = normalized_prediction(qwen[key])
        if label != "ANSWERABLE":
            qwen_refusals.append(key)
    print(f"QWEN25_SUFFICIENT_REFUSALS|{len(qwen_refusals)}")
    for model in ("Qwen3-VL-32B", "LLaVA-OneVision-7B", "InternVL3.5-8B"):
        print_row(
            f"OTHER_ON_QWEN25_REFUSALS|{model}|",
            metrics(data[model]["sufficient"], qwen_refusals),
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare auditable GPT main-table sample and loose Gemini recovery files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


VALID = frozenset("ABCDER")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def loose_answer(row: dict) -> tuple[str | None, str]:
    parsed = row.get("model_parsed") or {}
    value = parsed.get("answer")
    if isinstance(value, str) and value.strip().upper() in VALID:
        return value.strip().upper(), "parsed_answer"
    raw = str(row.get("model_response") or parsed.get("raw") or "").strip()
    values = {x.upper() for x in re.findall(
        r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw, re.I)}
    if len(values) == 1:
        return next(iter(values)), "raw_answer_field"
    labels = {x.upper() for x in re.findall(
        r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw, re.I)}
    if labels == {"UNANSWERABLE"}:
        return "R", "unanswerable_label"
    return None, "unparsed"


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA"))
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    root = args.root
    output = root / "results/closed_paper_sample_20260920"
    output.mkdir(parents=True, exist_ok=True)

    # GPT: deduplicate successful records, retain only loose-parseable rows,
    # then sample 300 complete S/I pairs plus 200 conflicts.
    gpt_rows = []
    for path in sorted((root / "results/gpt4o_v2_full_20260914/results/gpt4o").glob(
            "shard_*/merged.classify.jsonl")):
        gpt_rows.extend(read_jsonl(path))
    unique = {}
    for row in gpt_rows:
        identity = row.get("record_id")
        if not identity:
            identity = f"{row.get('source_dataset')}::{row.get('id')}::{row.get('evidence_condition')}"
        unique[str(identity)] = row
    by_base = defaultdict(dict)
    conflicts = []
    for row in unique.values():
        answer, _ = loose_answer(row)
        if answer is None:
            continue
        condition = row.get("evidence_condition")
        if condition in ("sufficient", "insufficient"):
            by_base[str(row.get("id"))][condition] = row
        elif condition == "conflict":
            conflicts.append(row)
    complete = [(values["sufficient"], values["insufficient"])
                for values in by_base.values()
                if "sufficient" in values and "insufficient" in values]
    rng = random.Random(args.seed)
    chosen_pairs = rng.sample(complete, 300)
    chosen_conflicts = rng.sample(conflicts, 200)
    gpt_sample = [row for pair in chosen_pairs for row in pair] + chosen_conflicts
    rng.shuffle(gpt_sample)
    gpt_sha = write_jsonl(output / "gpt4o_main_table_sample800.jsonl", gpt_sample)

    # Gemini: preserve original rows and add only separate audit fields. No
    # missing schema content, evidence, probability, or reason is fabricated.
    gemini_result = root / "results/gemini25pro_v2_sample800_20260919/merged.classify.jsonl"
    gemini_rows = read_jsonl(gemini_result)
    parseable, unresolved = [], []
    for row in gemini_rows:
        extracted, method = loose_answer(row)
        derived = dict(row)
        derived["paper_loose_answer"] = extracted
        derived["paper_loose_extraction_method"] = method
        (parseable if extracted is not None else unresolved).append(derived)
    skipped_path = root / "results/gemini25pro_v2_sample800_20260919/merged.classify.skipped.jsonl"
    skipped = read_jsonl(skipped_path)
    gemini_sha = write_jsonl(output / "gemini25pro_loose_parseable.jsonl", parseable)
    write_jsonl(output / "gemini25pro_unresolved_responses.jsonl", unresolved)
    write_jsonl(output / "gemini25pro_api_failures.jsonl", skipped)

    summary = {
        "seed": args.seed,
        "gpt": {
            "available_complete_pairs": len(complete),
            "available_conflicts": len(conflicts),
            "sample_records": len(gpt_sample),
            "sample_complete_pairs": 300,
            "sample_conflicts": 200,
            "condition_counts": dict(Counter(row.get("evidence_condition") for row in gpt_sample)),
            "sha256": gpt_sha,
        },
        "gemini": {
            "target_records": 800,
            "api_returned": len(gemini_rows),
            "loose_parseable": len(parseable),
            "returned_but_unresolved": len(unresolved),
            "api_failures": len(skipped),
            "shortfall_to_800": 800 - len(parseable),
            "condition_counts_parseable": dict(Counter(row.get("evidence_condition") for row in parseable)),
            "extraction_methods": dict(Counter(row["paper_loose_extraction_method"] for row in parseable)),
            "sha256": gemini_sha,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

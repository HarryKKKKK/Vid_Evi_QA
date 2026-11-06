#!/usr/bin/env python3
"""Score frozen structured-output runs with an auditable answer-only parser.

The strict metric requires the complete output schema.  The loose metric ignores
schema fields other than the final A-E/R decision, but never repairs semantic
content and never uses model-specific rules.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


VALID_ANSWERS = frozenset("ABCDER")
ALIASES = {
    "partially_sufficient": "partial",
    "conflicting": "conflict",
    "conflicting_evidence": "conflict",
}


def normalize_answer(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if value in VALID_ANSWERS else None


def unique_match(pattern: str, text: str) -> str | None:
    values = {match.upper() for match in re.findall(pattern, text, re.IGNORECASE)}
    return next(iter(values)) if len(values) == 1 else None


def loose_answer(row: dict) -> tuple[str | None, str]:
    """Return an answer and the deterministic extraction method used."""
    parsed = row.get("model_parsed") or {}
    answer = normalize_answer(parsed.get("answer"))
    if answer:
        return answer, "parsed_answer"

    raw = str(row.get("model_response") or parsed.get("raw") or "").strip()
    if not raw:
        return None, "unparsed"

    answer = unique_match(
        r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw
    )
    if answer:
        return answer, "raw_answer_field"

    answer = unique_match(
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-ER])\)?\b", raw
    )
    if answer:
        return answer, "final_answer_phrase"

    exact = re.fullmatch(r"\s*\(?([A-ER])\)?[.\s]*", raw, re.IGNORECASE)
    if exact:
        return exact.group(1).upper(), "standalone_answer"

    # Under the frozen protocol UNANSWERABLE always maps to answer R.  Do not
    # infer an A-E option from ANSWERABLE without an explicit option decision.
    label = normalize_answer(parsed.get("label"))
    if label is None:
        label = unique_match(
            r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw
        )
    else:
        # A-E/R in the label field is a malformed but explicit final decision.
        return label, "malformed_label_answer"
    if str(parsed.get("label") or label or "").strip().upper() == "UNANSWERABLE":
        return "R", "unanswerable_label"
    return None, "unparsed"


def condition_of(row: dict) -> str:
    meta = row.get("benchmark_metadata") or {}
    value = meta.get("evidence_condition") or row.get("evidence_condition") or "unknown"
    return ALIASES.get(value, value)


def expected_answer(row: dict) -> str | None:
    condition = condition_of(row)
    if condition != "sufficient":
        return "R" if condition in {"insufficient", "partial", "conflict"} else None
    valid = row.get("valid_option_ids") or []
    return normalize_answer(valid[0]) if len(valid) == 1 else None


def score(path: Path, expected_records: int | None = None) -> dict:
    totals = Counter()
    methods = Counter()
    by_condition: dict[str, Counter] = defaultdict(Counter)
    unparsed_rows: list[dict] = []
    seen: set[str] = set()
    duplicates = 0

    if path.is_dir():
        merged = path / "merged.classify.jsonl"
        input_paths = [merged] if merged.is_file() else sorted(
            path.glob("shard_*/merged.classify.jsonl")
        )
    else:
        input_paths = [path]

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = str(row.get("record_id") or f"{row.get('source_dataset')}::{row.get('id')}::{condition_of(row)}")
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)

                expected = expected_answer(row)
                answer, method = loose_answer(row)
                strict_valid = not bool(row.get("schema_errors"))
                strict_correct = strict_valid and answer == expected
                loose_correct = answer == expected and expected is not None
                recovered = not strict_valid and answer is not None

                if answer is None:
                    raw = str(row.get("model_response") or (row.get("model_parsed") or {}).get("raw") or "")
                    unparsed_rows.append({
                        "record_id": identity,
                        "condition": condition_of(row),
                        "schema_errors": row.get("schema_errors") or [],
                        "raw_length": len(raw),
                        "raw_answer_candidates": sorted(set(re.findall(
                            r'''["']answer["']\s*:\s*["']([A-ER])["']''',
                            raw,
                            re.IGNORECASE,
                        ))),
                        "raw_label_candidates": sorted(set(re.findall(
                            r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE|[A-ER])["']''',
                            raw,
                            re.IGNORECASE,
                        ))),
                        "raw_head": raw[:500],
                        "raw_tail": raw[-500:] if len(raw) > 500 else raw,
                    })

                bucket = by_condition[condition_of(row)]
                for target in (totals, bucket):
                    target["returned"] += 1
                    target["strict_valid"] += strict_valid
                    target["strict_correct"] += strict_correct
                    target["loose_parseable"] += answer is not None
                    target["loose_correct"] += loose_correct
                    target["invalid_with_recovered_answer"] += recovered
                    target["invalid_recovered_correct"] += recovered and loose_correct
                methods[method] += 1

    denominator = expected_records or totals["returned"]
    totals["expected"] = denominator
    totals["missing"] = max(0, denominator - totals["returned"])
    return {
        "path": str(path),
        "input_files": [str(value) for value in input_paths],
        "totals": dict(totals),
        "rates_percent": {
            "format_validity": 100 * totals["strict_valid"] / denominator if denominator else None,
            "strict_accuracy": 100 * totals["strict_correct"] / denominator if denominator else None,
            "loose_parse_rate": 100 * totals["loose_parseable"] / denominator if denominator else None,
            "loose_accuracy": 100 * totals["loose_correct"] / denominator if denominator else None,
        },
        "by_condition": {key: dict(value) for key, value in sorted(by_condition.items())},
        "extraction_methods": dict(methods),
        "unparsed_rows": unparsed_rows,
        "duplicates_ignored": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    print(json.dumps(score(args.jsonl, args.expected_records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

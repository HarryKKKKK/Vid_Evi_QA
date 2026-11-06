#!/usr/bin/env python3
"""Merge per-backend language-prior runs and build the joint candidate list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screen_language_priors import (
    Backend,
    DEFAULT_CANDIDATES,
    DEFAULT_MANIFEST,
    aggregate,
    load_items,
    load_latest_jsonl,
    prediction_key,
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--per-model-threshold", type=float, default=0.8)
    parser.add_argument("--overall-threshold", type=float, default=0.8)
    parser.add_argument("--min-positive-models", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    a = args()
    merged = {}
    input_counts = {}
    for path in a.input:
        rows = load_latest_jsonl(path)
        input_counts[str(path)] = len(rows)
        merged.update(rows)
    ordered = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("item_key")),
            str(row.get("backend")),
            int(row.get("permutation_index", -1)),
        ),
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    output = a.output_dir / "raw_predictions.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)

    items = load_items(a.manifest, a.candidates)
    if a.limit is not None:
        items = items[: a.limit]
    backends = [
        Backend("qwen", "qwen3-8b-text-prior", "offline", None, ""),
        Backend("llama", "llama31-8b-text-prior", "offline", None, ""),
        Backend("gpt", "gpt-4o", "chat", None, ""),
    ]
    summary = aggregate(items, backends, a, output)
    print(json.dumps({
        "input_counts": input_counts,
        "unique_predictions": len({prediction_key(row) for row in ordered}),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

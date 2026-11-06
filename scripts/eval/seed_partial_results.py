#!/usr/bin/env python3
"""Seed a larger partial run with compatible results from a completed pilot."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--alphas", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    keys = {
        (str(row["video_id"]), str(row["qid"]))
        for row in json.loads(args.candidates.read_text(encoding="utf-8"))
    }
    alphas = {round(float(value), 6) for value in args.alphas.split(",")}
    selected = []
    seen = set()
    with args.source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["video_id"]), str(row["qid"]))
            alpha = round(float(row["alpha"]), 6)
            record_key = (*key, alpha)
            if key in keys and alpha in alphas and record_key not in seen:
                row["seed_source_model_tag"] = row.get("model_tag")
                row["model_tag"] = args.model_tag
                row["alpha_selection_split"] = "dev"
                selected.append(row)
                seen.add(record_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (
        ("partial.classify.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)),
        ("partial.classify.json", json.dumps(selected, ensure_ascii=False, indent=2) + "\n"),
    ):
        destination = args.output_dir / suffix
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=args.output_dir, delete=False
        ) as handle:
            handle.write(content)
            temporary = handle.name
        os.replace(temporary, destination)
    print(f"SEEDED_PARTIAL_RESULTS rows={len(selected)} output={args.output_dir}")


if __name__ == "__main__":
    main()

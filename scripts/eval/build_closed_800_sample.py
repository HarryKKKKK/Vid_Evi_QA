#!/usr/bin/env python3
"""Build the fixed balanced 800-row closed-model evaluation sample."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


TARGETS = {"sufficient": 267, "insufficient": 267, "partially_sufficient": 266}
PARTIAL_ALPHA_TARGETS = {0.2: 67, 0.4: 67, 0.5: 66, 0.6: 66}


def read_rows(shard_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(shard_dir.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rows = read_rows(args.shard_dir)
    rng = random.Random(args.seed)
    selected = []
    for condition in ("sufficient", "insufficient"):
        pool = [row for row in rows if row.get("evidence_condition") == condition]
        selected.extend(rng.sample(pool, TARGETS[condition]))
    for alpha, count in PARTIAL_ALPHA_TARGETS.items():
        pool = [row for row in rows if row.get("evidence_condition") == "partially_sufficient"
                and float(row.get("alpha")) == alpha]
        selected.extend(rng.sample(pool, count))
    rng.shuffle(selected)

    if len(selected) != 800 or len({row["record_id"] for row in selected}) != 800:
        raise RuntimeError("sample is not exactly 800 unique records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    args.output.write_text(payload, encoding="utf-8")
    summary = {
        "records": len(selected),
        "seed": args.seed,
        "conditions": Counter(row["evidence_condition"] for row in selected),
        "partial_alphas": Counter(str(row.get("alpha")) for row in selected
                                  if row["evidence_condition"] == "partially_sufficient"),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    summary["conditions"] = dict(summary["conditions"])
    summary["partial_alphas"] = dict(summary["partial_alphas"])
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

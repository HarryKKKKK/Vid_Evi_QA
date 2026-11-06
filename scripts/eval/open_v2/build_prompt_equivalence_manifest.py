"""Create a deterministic condition-balanced prompt-equivalence manifest."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def condition(row):
    value = str(row.get("evidence_condition", "")).lower()
    if value == "partially_sufficient":
        return "partial"
    if value in {"conflicting", "conflicting_evidence"}:
        return "conflict"
    return value


def rank(seed, row):
    identity = row.get("record_id") or row.get("id")
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-condition", type=int, default=100)
    parser.add_argument("--seed", default="prompt-equivalence-20260911")
    args = parser.parse_args()

    groups = {name: [] for name in ("sufficient", "insufficient", "partial", "conflict")}
    for line in args.source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = condition(row)
        if name in groups:
            groups[name].append(row)
    selected = []
    for name, rows in groups.items():
        if len(rows) < args.per_condition:
            raise RuntimeError(f"{name}: need {args.per_condition}, found {len(rows)}")
        selected.extend(sorted(rows, key=lambda row: rank(args.seed, row))[:args.per_condition])
    selected.sort(key=lambda row: str(row.get("record_id") or row.get("id")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "records": len(selected),
        "conditions": dict(Counter(condition(row) for row in selected)),
        "datasets": dict(Counter(row.get("source_dataset", "unknown") for row in selected)),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

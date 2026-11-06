#!/usr/bin/env python3
"""Select a deterministic, question-type-stratified partial-eval subset."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 100):03d}"


def stable_rank(seed: int, row: dict) -> str:
    payload = f"{seed}:{row['video_id']}:{row['qid']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate(counts: Counter, target: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {key: target * value / total for key, value in counts.items()}
    out = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(out.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - out[key]), key))
    for key in order[:remaining]:
        out[key] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--partial-mode", default="noise")
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("answer") in set("ABCDE")]
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]

    eligible = []
    incomplete = []
    for row in rows:
        missing = [
            args.partial_dir
            / f"{row['video_id']}_q{row['qid']}_c2_{args.partial_mode}_{alpha_tag(alpha)}.mp4"
            for alpha in alphas
        ]
        missing = [path for path in missing if not path.is_file()]
        if missing:
            incomplete.append((row["video_id"], row["qid"], missing[0].name))
        else:
            eligible.append(row)

    if len(eligible) < args.size:
        raise SystemExit(
            f"not enough complete anchors: eligible={len(eligible)} requested={args.size}; "
            f"first_incomplete={incomplete[:5]}"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        grouped[str(row.get("question_type", "UNKNOWN"))].append(row)

    quotas = allocate(Counter({key: len(value) for key, value in grouped.items()}), args.size)
    selected = []
    for question_type in sorted(grouped):
        ranked = sorted(grouped[question_type], key=lambda row: stable_rank(args.seed, row))
        selected.extend(ranked[: quotas[question_type]])
    selected.sort(key=lambda row: (str(row["video_id"]), str(row["qid"])))

    output_rows = [
        {"video_id": row["video_id"], "qid": row["qid"]}
        for row in selected
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    meta = {
        "manifest": str(args.manifest),
        "partial_dir": str(args.partial_dir),
        "size": len(output_rows),
        "seed": args.seed,
        "alphas": alphas,
        "partial_mode": args.partial_mode,
        "eligible_count": len(eligible),
        "excluded_invalid_answer_count": len(json.loads(args.manifest.read_text(encoding="utf-8"))) - len(rows),
        "incomplete_video_count": len(incomplete),
        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in selected).items())),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

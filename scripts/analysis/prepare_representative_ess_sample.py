#!/usr/bin/env python3
"""Create a proportional, reproducible sample for matched-control ESS.

The target population is the subset of Sufficient NExT-GQA questions for which
at least one of the 32 sampled timestamps is inside the reference evidence and
an equally sized, disjoint non-evidence control can be constructed.  Sampling
is proportional within question-type, evidence-frame-count, and duration bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260916)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evidence_indices(row: dict) -> list[int]:
    spans = [(min(float(x["start"]), float(x["end"])),
              max(float(x["start"]), float(x["end"])))
             for x in row.get("evidence_intervals", [])]
    return [i for i, t in enumerate(row.get("sampled_seconds", []))
            if any(a <= float(t) <= b for a, b in spans)]


def evidence_bin(count: int) -> str:
    for lower, upper in ((1, 2), (3, 4), (5, 8), (9, 12), (13, 16)):
        if count <= upper:
            return f"e{lower:02d}-{upper:02d}"
    return "e17+"


def duration_bin(value: float) -> str:
    if value <= 30:
        return "d_short"
    if value <= 60:
        return "d_medium"
    return "d_long"


def allocate(groups: dict[str, list[dict]], n: int) -> dict[str, int]:
    total = sum(map(len, groups.values()))
    if n > total:
        raise ValueError(f"sample size {n} exceeds eligible population {total}")
    ideal = {key: n * len(rows) / total for key, rows in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(value)) for key, value in ideal.items()}
    remaining = n - sum(allocation.values())
    order = sorted(groups, key=lambda key: (ideal[key] - allocation[key], len(groups[key]), key), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(groups[key]) and remaining:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("unable to finish proportional allocation")
    return allocation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = [r for r in read_jsonl(args.input)
              if str(r.get("evidence_condition", "")).lower() in {"sufficient", "c1"}
              and str(r.get("source_dataset", "")).lower() == "nextgqa"]
    exclusion = Counter()
    eligible = []
    for row in source:
        count = len(evidence_indices(row))
        total = len(row.get("sampled_seconds", []))
        if count == 0:
            exclusion["no_sampled_evidence"] += 1
            continue
        if total == 0 or total - count < count:
            exclusion["insufficient_non_evidence_frames"] += 1
            continue
        item = {
            "experiment_group": "representative_matched_control",
            "video_kind": "source",
            "video_id": str(row["video_id"]),
            "qid": str(row["qid"]),
            "question_type": row.get("question_type"),
            "question": row["question"],
            "choices": row["choices"],
            "answer": str(row["valid_option_ids"][0]),
            "sampled_seconds": [float(x) for x in row["sampled_seconds"]],
            "evidence_intervals": row.get("evidence_intervals", []),
            "video_duration": float(row.get("video_duration") or 0),
            "evidence_frame_count": count,
        }
        item["sampling_stratum"] = "|".join((
            str(item.get("question_type") or "UNKNOWN"),
            evidence_bin(count), duration_bin(item["video_duration"]),
        ))
        eligible.append(item)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        groups[row["sampling_stratum"]].append(row)
    allocation = allocate(groups, args.sample_size)
    rng = random.Random(args.seed)
    selected = []
    strata_audit = {}
    for key in sorted(groups):
        population = groups[key]
        rng.shuffle(population)
        take = allocation[key]
        weight = len(population) / take if take else None
        probability = take / len(population) if population else None
        for row in population[:take]:
            row["sampling_weight"] = weight
            row["inclusion_probability"] = probability
            selected.append(row)
        strata_audit[key] = {"population": len(population), "sample": take,
                             "sampling_weight": weight}
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "input": str(args.input), "input_sha256": sha256(args.input),
        "seed": args.seed, "requested_sample_size": args.sample_size,
        "source_sufficient_population": len(source),
        "eligible_population": len(eligible),
        "eligible_fraction": len(eligible) / len(source) if source else None,
        "exclusions": dict(exclusion), "selected": len(selected),
        "unique_source_videos": len({r["video_id"] for r in selected}),
        "population_question_types": dict(Counter(str(r.get("question_type")) for r in eligible)),
        "sample_question_types": dict(Counter(str(r.get("question_type")) for r in selected)),
        "strata": strata_audit,
    }
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("source_sufficient_population", "eligible_population",
                                             "eligible_fraction", "exclusions", "selected",
                                             "unique_source_videos", "sample_question_types")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

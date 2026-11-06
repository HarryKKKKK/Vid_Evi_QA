#!/usr/bin/env python3
"""Apply language-prior removals consistently to every NExT-GQA condition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json"
DEFAULT_CANDIDATES = ROOT / "nextgqa_pipeline/partial_candidates_full_n5279.json"
DEFAULT_PARTIAL_META = ROOT / "nextgqa_pipeline/partial_candidates_full_n5279.meta.json"
DEFAULT_SCREEN_DIR = ROOT / "reports/language_prior_screen/combined_5279"
DEFAULT_OUTPUT = DEFAULT_SCREEN_DIR / "final_filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--partial-meta", type=Path, default=DEFAULT_PARTIAL_META)
    parser.add_argument("--removals", type=Path, default=DEFAULT_SCREEN_DIR / "removal_candidates.json")
    parser.add_argument("--incomplete", type=Path, default=DEFAULT_SCREEN_DIR / "incomplete_items.json")
    parser.add_argument("--treat-incomplete-as-retained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def condition_row(row: dict, condition: str, alpha: float | None = None) -> dict:
    result = dict(row)
    video_id, qid = key(row)
    result["evidence_condition"] = condition
    if condition == "insufficient":
        result["video_path"] = (
            f"source_datasets/next_gqa/freeze_videos_no_evidence_leak/"
            f"{video_id}_q{qid}_freeze.mp4"
        )
    elif condition == "partially_sufficient":
        if alpha is None:
            raise ValueError("partial condition requires alpha")
        result["alpha"] = alpha
        result["partial_mode"] = "noise"
        result["video_path"] = (
            f"source_datasets/next_gqa/partial_videos_no_evidence_leak/"
            f"{video_id}_q{qid}_c2_noise_a{round(alpha * 100):03d}.mp4"
        )
    return result


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    candidates = read_json(args.candidates)
    partial_meta = read_json(args.partial_meta)
    removals = read_json(args.removals)
    incomplete = read_json(args.incomplete) if args.incomplete.is_file() else []

    if incomplete and not args.treat_incomplete_as_retained:
        raise SystemExit(
            f"There are {len(incomplete)} incomplete items. Pass "
            "--treat-incomplete-as-retained only after manual review."
        )

    candidate_keys = {key(row) for row in candidates}
    removal_keys = {key(row) for row in removals}
    incomplete_keys = {key(row) for row in incomplete}
    if removal_keys - candidate_keys:
        raise SystemExit("Removal list contains keys outside the candidate set")
    if incomplete_keys & removal_keys:
        raise SystemExit("An incomplete/manual-retain item also appears in the removal list")

    selected = [row for row in manifest if key(row) in candidate_keys]
    if len(selected) != len(candidate_keys):
        raise SystemExit(f"Matched {len(selected)} of {len(candidate_keys)} candidate keys")
    retained = [row for row in selected if key(row) not in removal_keys]
    retained_keys = {key(row) for row in retained}
    alphas = [float(value) for value in partial_meta["alphas"]]

    sufficient = [condition_row(row, "sufficient") for row in retained]
    insufficient = [condition_row(row, "insufficient") for row in retained]
    partial_by_alpha = {
        f"{alpha:g}": [condition_row(row, "partially_sufficient", alpha) for row in retained]
        for alpha in alphas
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "retained_candidates.json", [
        {"video_id": row["video_id"], "qid": row["qid"]} for row in retained
    ])
    atomic_json(args.output_dir / "removed_language_prior_items.json", removals)
    atomic_json(args.output_dir / "manual_retained_incomplete_items.json", incomplete)
    atomic_json(args.output_dir / "sufficient.json", sufficient)
    atomic_json(args.output_dir / "insufficient.json", insufficient)
    for alpha, rows in partial_by_alpha.items():
        atomic_json(args.output_dir / f"partial_alpha_{alpha}.json", rows)

    condition_path = args.output_dir / "all_nextgqa_conditions.jsonl"
    condition_tmp = condition_path.with_suffix(".jsonl.tmp")
    with condition_tmp.open("w", encoding="utf-8") as stream:
        for row in sufficient + insufficient:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for rows in partial_by_alpha.values():
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    condition_tmp.replace(condition_path)

    summary = {
        "status": "complete_with_manual_incomplete_override" if incomplete else "complete",
        "policy": {
            "removal_rule": "three-model forced-choice threshold output",
            "incomplete_items": "retained_as_no_language_prior_after_manual_review",
            "physical_video_files_deleted": False,
            "application": "manifest-level exclusion shared by sufficient, insufficient, and every partial alpha",
        },
        "original_nextgqa_qa": len(candidate_keys),
        "removed_language_prior_qa": len(removal_keys),
        "manual_retained_incomplete_qa": len(incomplete_keys),
        "retained_nextgqa_qa": len(retained_keys),
        "retained_question_type_counts": dict(sorted(Counter(
            str(row.get("question_type", "UNKNOWN")) for row in retained
        ).items())),
        "alphas": alphas,
        "condition_counts": {
            "sufficient": len(sufficient),
            "insufficient": len(insufficient),
            **{f"partial_alpha_{alpha}": len(rows) for alpha, rows in partial_by_alpha.items()},
        },
        "total_nextgqa_condition_records": len(retained) * (2 + len(alphas)),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Create text-only contrast judging jobs from independent visual signatures."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/build_pair_judging_jobs.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/build_pair_judging_jobs.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(_LAYOUT_LEGACY_FILE).resolve().parent
DEFAULT_GROUPS = SCRIPT_DIR / "visual_contrast_jobs.jsonl"
DEFAULT_RESULTS = SCRIPT_DIR / "visual_instance_results.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "pair_judging_jobs.jsonl"

NON_CONTRAST_SLOTS = {"anchor_action"}
MISSING_VALUES = {
    "",
    "none",
    "null",
    "unknown",
    "not visible",
    "not_visible",
    "unclear",
    "n/a",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def normalize_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.rstrip(". ")


def parse_slot(raw: Any) -> tuple[str, float, str]:
    if isinstance(raw, dict):
        value = str(raw.get("value", "")).strip()
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        basis = str(raw.get("visual_basis", "")).strip()
        return value, max(0.0, min(1.0, confidence)), basis
    return str(raw or "").strip(), 0.0, ""


def visible_differences(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    slots_a = result_a.get("slots", {})
    slots_b = result_b.get("slots", {})
    differences: list[dict[str, Any]] = []
    for slot in sorted(set(slots_a) & set(slots_b) - NON_CONTRAST_SLOTS):
        value_a, confidence_a, basis_a = parse_slot(slots_a[slot])
        value_b, confidence_b, basis_b = parse_slot(slots_b[slot])
        normalized_a = normalize_value(value_a)
        normalized_b = normalize_value(value_b)
        if normalized_a in MISSING_VALUES or normalized_b in MISSING_VALUES:
            continue
        if normalized_a == normalized_b:
            continue
        if min(confidence_a, confidence_b) < min_confidence:
            continue
        differences.append(
            {
                "slot": slot,
                "value_a": value_a,
                "value_b": value_b,
                "confidence_a": confidence_a,
                "confidence_b": confidence_b,
                "visual_basis_a": basis_a,
                "visual_basis_b": basis_b,
            }
        )
    return differences


def build_jobs(
    groups: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    *,
    min_confidence: float,
    max_pairs_per_parent: int,
) -> list[dict[str, Any]]:
    groups_by_id = {group["job_id"]: group for group in groups}
    results_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        parent_id = result.get("parent_job_id")
        if parent_id in groups_by_id:
            results_by_parent[parent_id].append(result)

    output: list[dict[str, Any]] = []
    for parent_id, parent_results in results_by_parent.items():
        candidates: list[tuple[int, dict[str, Any]]] = []
        group = groups_by_id[parent_id]
        instances = {x["instance_id"]: x for x in group["instances"]}
        for result_a, result_b in itertools.combinations(parent_results, 2):
            differences = visible_differences(
                result_a, result_b, min_confidence=min_confidence
            )
            if not differences:
                continue
            id_a = result_a["instance_job_id"].rsplit("__", 1)[-1]
            id_b = result_b["instance_job_id"].rsplit("__", 1)[-1]
            if id_a not in instances or id_b not in instances:
                continue
            pair = {
                "pair_job_id": f"{parent_id}__{id_a}__{id_b}",
                "parent_job_id": parent_id,
                "source_task": group["source_task"],
                "video": group["video"],
                "anchor_action_hint": group.get("anchor_action_hint"),
                "instance_a": instances[id_a],
                "instance_b": instances[id_b],
                "candidate_differences": differences,
                "instruction": (
                    "Judge each candidate difference conservatively. Decide whether "
                    "the two values are genuinely different, visually grounded, and "
                    "mutually exclusive answers to one natural question about the same "
                    "event type. Reject granularity differences, synonyms, uncertain "
                    "attributes, temporal changes that the question would already "
                    "disambiguate, and differences caused by referring to different "
                    "entities. Return only slots suitable for conflict-QA construction."
                ),
                "output_schema": {
                    "pair_job_id": "string",
                    "common_anchor_action": "string",
                    "accepted_differences": [
                        {
                            "slot": "string",
                            "value_a": "string",
                            "value_b": "string",
                            "same_event_semantics": "boolean",
                            "mutually_exclusive": "boolean",
                            "question_template": "string",
                            "confidence": "number",
                            "reason": "string",
                        }
                    ],
                },
                "status": "pending_text_pair_judgment",
            }
            candidates.append((len(differences), pair))

        candidates.sort(key=lambda item: (-item[0], item[1]["pair_job_id"]))
        output.extend(pair for _, pair in candidates[:max_pairs_per_parent])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-pairs-per-parent", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be in [0,1]")
    if args.max_pairs_per_parent < 1:
        parser.error("--max-pairs-per-parent must be positive")

    jobs = build_jobs(
        read_jsonl(args.groups),
        read_jsonl(args.results),
        min_confidence=args.min_confidence,
        max_pairs_per_parent=args.max_pairs_per_parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"Pair-judging jobs: {len(jobs)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

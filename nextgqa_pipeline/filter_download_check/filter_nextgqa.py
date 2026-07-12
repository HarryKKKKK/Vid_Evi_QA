"""NExT-GQA (test split) sufficient-anchor filtering script.

The script joins:
  * test.csv               -- QA, options, question type
  * gsub_test.json         -- temporal grounding intervals
  * map_vid_vidorID.json   -- video_id -> VidOR relative identifier

It writes a filtered JSON whose per-sample key set matches the existing
CG-Bench filtered JSON template.

Filtering conditions
--------------------
A sample is retained only when:
  1. (video_id, qid) is not duplicated in test.csv;
  2. question is non-empty;
  3. all five options a0..a4 are non-empty;
  4. grounding exists for (video_id, qid);
  5. video_id exists in map_vid_vidorID.json;
  6. video_duration > 0;
  7. at least one valid evidence interval remains after normalization;
  8. deterministic uniform frame sampling hits the evidence intervals.

Frame-sampling filter
---------------------
By default, the script evaluates both 32-frame and 64-frame uniform sampling.
It uses equal-duration temporal bins and samples the centre timestamp of each
bin (``bin_center``). With the default ``--frame-hit-policy all``, both the
32-frame and the 64-frame plans must satisfy the interval-hit requirement.

The default ``--interval-hit-policy any`` means that at least one sampled
timestamp must fall inside at least one gold evidence interval. Use
``--interval-hit-policy all`` to require every gold evidence interval to be
hit by at least one sampled timestamp.

Evidence interval duration and total evidence coverage are deliberately not
used as filtering conditions. Any positive-length normalized interval is
eligible for the frame-sampling hit check.

Question type is not a filtering condition. The original NExT-GQA type is
stored verbatim in ``question_type``. Every retained sample is written as
``evidence_condition = \"sufficient\"``.

Example
-------
python nextgqa_pipeline/filter_download_check/filter_nextgqa.py \\
  --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \\
  --cgbench-template cgbench_pipeline/cgbench_filtered.json \\
  --output nextgqa_pipeline/nextgqa_filtered.json \\
  --summary-json nextgqa_pipeline/nextgqa_filter_summary.json \\
  --summary-txt nextgqa_pipeline/nextgqa_filter_summary.txt \\
  --rejected-json nextgqa_pipeline/nextgqa_rejected.json \\
  --frame-counts 32 64 \\
  --sampling-mode bin_center \\
  --frame-hit-policy all \\
  --interval-hit-policy any
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_DIR = (
    REPO_ROOT / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa"
)
DEFAULT_CGBENCH_TEMPLATE = REPO_ROOT / "cgbench_pipeline" / "cgbench_filtered.json"
DEFAULT_OUTPUT = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filtered.json"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filter_summary.json"
DEFAULT_SUMMARY_TXT = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_filter_summary.txt"
DEFAULT_REJECTED_JSON = REPO_ROOT / "nextgqa_pipeline" / "nextgqa_rejected.json"

CSV_FILENAME = "test.csv"
GROUNDING_FILENAME = "gsub_test.json"
MAPPING_FILENAME = "map_vid_vidorID.json"

REQUIRED_CSV_COLUMNS = [
    "video_id",
    "frame_count",
    "width",
    "height",
    "question",
    "answer",
    "qid",
    "type",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
]
OPTION_COLUMNS = ["a0", "a1", "a2", "a3", "a4"]
ANSWER_LETTERS = ["A", "B", "C", "D", "E"]

DEFAULT_FRAME_COUNTS = [32, 64]
DEFAULT_SAMPLING_MODE = "bin_center"
DEFAULT_FRAME_HIT_POLICY = "all"
DEFAULT_INTERVAL_HIT_POLICY = "any"

SOURCE_DATASET = "nextgqa"
EVIDENCE_CONDITION = "sufficient"
EXPECTED_BEHAVIOR = "answer"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Strip text and collapse internal whitespace runs to one space."""
    return re.sub(r"\s+", " ", str(value)).strip()


def match_answer_index(answer_text: str, options: list[str]) -> int | None:
    """Return the unique case-insensitive option match, otherwise ``None``.

    This is only an output-format conversion. Failure to match is not a
    filtering condition.
    """
    target = clean_text(answer_text).lower()
    if not target:
        return None
    matches = [
        index
        for index, option in enumerate(options)
        if clean_text(option).lower() == target
    ]
    return matches[0] if len(matches) == 1 else None


def index_to_letter(index: int) -> str:
    """Convert a zero-based option index to A/B/C/D/E."""
    if 0 <= index < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[index]
    raise ValueError(f"No answer-letter mapping for option index {index}")


def normalize_intervals(
    raw_pairs: Any,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Normalize raw ``[start, end]`` evidence pairs.

    Processing order:
      1. cast start/end to finite floats;
      2. swap reversed bounds;
      3. clip to [0, video_duration];
      4. discard zero-length/invalid intervals;
      5. sort by start time;
      6. merge overlapping or touching intervals.

    Separated intervals remain separated.
    """
    cleaned: list[tuple[float, float]] = []

    if not isinstance(raw_pairs, list):
        return []

    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue

        try:
            start = float(pair[0])
            end = float(pair[1])
        except (TypeError, ValueError):
            continue

        if not math.isfinite(start) or not math.isfinite(end):
            continue

        if end < start:
            start, end = end, start

        start = min(max(start, 0.0), video_duration)
        end = min(max(end, 0.0), video_duration)

        if end <= start:
            continue

        cleaned.append((start, end))

    cleaned.sort(key=lambda pair: (pair[0], pair[1]))

    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [
        {"start": start, "end": end, "description": ""}
        for start, end in merged
    ]



def generate_uniform_timestamps(
    video_duration: float,
    frame_count: int,
    sampling_mode: str,
) -> list[float]:
    """Generate deterministic full-video uniform sampling timestamps.

    ``bin_center`` divides the duration into equal temporal bins and samples
    the centre of each bin:

        t_i = (i + 0.5) * duration / frame_count

    ``linspace`` mirrors endpoint-inclusive feature-index sampling:

        t_i = i * duration / (frame_count - 1)

    ``bin_center`` is the default because it avoids requesting a frame exactly
    at ``video_duration``, which can be unreliable during raw-frame extraction.
    """
    if video_duration <= 0:
        raise ValueError("video_duration must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    if sampling_mode == "bin_center":
        step = video_duration / frame_count
        return [(index + 0.5) * step for index in range(frame_count)]

    if sampling_mode == "linspace":
        if frame_count == 1:
            return [video_duration / 2.0]
        step = video_duration / (frame_count - 1)
        return [index * step for index in range(frame_count)]

    raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}")


def timestamp_in_interval(timestamp: float, interval: dict[str, Any]) -> bool:
    """Return whether a timestamp is inside a closed evidence interval."""
    return float(interval["start"]) <= timestamp <= float(interval["end"])


def evaluate_sampling_hits(
    timestamps: list[float],
    intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate which evidence intervals are hit by sampled timestamps."""
    interval_hit_flags: list[bool] = []
    hit_timestamps: list[float] = []

    for interval in intervals:
        hits = [timestamp for timestamp in timestamps if timestamp_in_interval(timestamp, interval)]
        interval_hit_flags.append(bool(hits))
        hit_timestamps.extend(hits)

    unique_hits = sorted(set(hit_timestamps))

    return {
        "any_interval_hit": any(interval_hit_flags),
        "all_intervals_hit": all(interval_hit_flags) if interval_hit_flags else False,
        "hit_interval_count": sum(interval_hit_flags),
        "total_interval_count": len(intervals),
        "hit_frame_count": len(unique_hits),
        "first_hit_timestamps": unique_hits[:10],
    }


def sampling_plan_passes(
    hit_result: dict[str, Any],
    interval_hit_policy: str,
) -> bool:
    """Apply the any/all evidence-interval hit requirement."""
    if interval_hit_policy == "any":
        return bool(hit_result["any_interval_hit"])
    if interval_hit_policy == "all":
        return bool(hit_result["all_intervals_hit"])
    raise ValueError(f"Unsupported interval_hit_policy: {interval_hit_policy!r}")


def evaluate_frame_sampling_requirements(
    video_duration: float,
    intervals: list[dict[str, Any]],
    frame_counts: list[int],
    sampling_mode: str,
    frame_hit_policy: str,
    interval_hit_policy: str,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Evaluate all requested frame-count configurations.

    ``frame_hit_policy=all`` requires every requested frame count to pass.
    ``frame_hit_policy=any`` requires at least one requested frame count to pass.
    """
    results: dict[str, dict[str, Any]] = {}
    per_count_passes: list[bool] = []

    for frame_count in frame_counts:
        timestamps = generate_uniform_timestamps(
            video_duration=video_duration,
            frame_count=frame_count,
            sampling_mode=sampling_mode,
        )
        hit_result = evaluate_sampling_hits(timestamps, intervals)
        passed = sampling_plan_passes(hit_result, interval_hit_policy)

        results[str(frame_count)] = {
            "passed": passed,
            "sampling_mode": sampling_mode,
            "interval_hit_policy": interval_hit_policy,
            **hit_result,
        }
        per_count_passes.append(passed)

    if frame_hit_policy == "all":
        overall_passed = all(per_count_passes)
    elif frame_hit_policy == "any":
        overall_passed = any(per_count_passes)
    else:
        raise ValueError(f"Unsupported frame_hit_policy: {frame_hit_policy!r}")

    return overall_passed, results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load and validate the NExT-GQA test CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV annotation file not found: {csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Found columns: {fieldnames}"
            )
        return list(reader)


def load_json_dict(path: Path, description: str) -> dict[str, Any]:
    """Load a JSON object with clear errors."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} at {path} is not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{description} at {path}: expected a JSON object, got {type(data).__name__}"
        )

    return data


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def load_cgbench_template_keys(template_path: Path) -> set[str]:
    """Load the sample key set from the CG-Bench filtered JSON template."""
    if not template_path.exists():
        raise FileNotFoundError(
            f"CG-Bench template JSON not found: {template_path}. "
            "It is required as the NExT-GQA output schema reference."
        )

    with template_path.open(encoding="utf-8") as file:
        template = json.load(file)

    if not isinstance(template, list) or not template:
        raise ValueError(
            f"CG-Bench template at {template_path} must be a non-empty JSON array"
        )

    if not isinstance(template[0], dict):
        raise ValueError("The first CG-Bench template sample must be a JSON object")

    return set(template[0].keys())


def validate_schema(records: list[dict[str, Any]], template_keys: set[str]) -> None:
    """Ensure NExT-GQA records have exactly the CG-Bench sample keys."""
    if not records:
        return

    record_keys = set(records[0].keys())
    missing = template_keys - record_keys
    extra = record_keys - template_keys

    if missing or extra:
        raise ValueError(
            "nextgqa_filtered.json schema does not match cgbench_filtered.json:\n"
            f"  missing keys: {sorted(missing)}\n"
            f"  extra keys:   {sorted(extra)}"
        )

    for index, record in enumerate(records):
        if set(record.keys()) != record_keys:
            raise ValueError(
                f"Record {index} (video_id={record.get('video_id')}, "
                f"qid={record.get('qid')}) has inconsistent keys"
            )


def validate_final_invariants(records: list[dict[str, Any]]) -> None:
    """Validate final output invariants."""
    seen: set[tuple[str, str]] = set()

    for record in records:
        if record["evidence_condition"] != EVIDENCE_CONDITION:
            raise ValueError(
                "evidence_condition must be 'sufficient', got "
                f"{record['evidence_condition']!r}"
            )

        if record["evidence_count"] < 1:
            raise ValueError(
                f"No evidence intervals for video_id={record['video_id']} qid={record['qid']}"
            )

        if len(record["evidence_intervals"]) != record["evidence_count"]:
            raise ValueError(
                "evidence_count/evidence_intervals mismatch for "
                f"video_id={record['video_id']} qid={record['qid']}"
            )

        key = (str(record["video_id"]), str(record["qid"]))
        if key in seen:
            raise ValueError(f"Duplicate (video_id, qid) in final output: {key}")
        seen.add(key)


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def resolve_answer(row: dict[str, str], options: list[str]) -> tuple[str, bool]:
    """Convert answer text to an option letter when uniquely possible.

    Conversion failure does not reject the sample. The cleaned raw answer is
    retained instead, and the issue is reported in the summary JSON.
    """
    answer_index = match_answer_index(row.get("answer", ""), options)
    if answer_index is not None:
        return index_to_letter(answer_index), True
    return clean_text(row.get("answer", "")), False


def build_record(
    video_id: str,
    qid: str,
    question: str,
    choices: list[str],
    answer_value: str,
    intervals: list[dict[str, Any]],
    video_duration: float,
    question_type: str,
) -> dict[str, Any]:
    """Build one CG-Bench-schema-compatible NExT-GQA output record."""
    return {
        "video_id": video_id,
        "qid": qid,
        "source_dataset": SOURCE_DATASET,
        "question": question,
        "choices": choices,
        "answer": answer_value,
        "evidence_intervals": intervals,
        "evidence_count": len(intervals),
        "video_duration": video_duration,
        "question_type": question_type,
        "source_task": question_type,
        "evidence_condition": EVIDENCE_CONDITION,
        "expected_behavior": EXPECTED_BEHAVIOR,
    }


# ---------------------------------------------------------------------------
# Core filtering pipeline
# ---------------------------------------------------------------------------

def run_filter(
    rows: list[dict[str, str]],
    grounding: dict[str, Any],
    mapping: dict[str, Any],
    frame_counts: list[int],
    sampling_mode: str,
    frame_hit_policy: str,
    interval_hit_policy: str,
    limit: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    OrderedDict[str, int],
    Counter[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run the complete filtering pipeline."""
    if limit is not None:
        rows = rows[:limit]

    counter_names = [
        "raw_csv_rows",
        "after_duplicate_removal",
        "after_question_valid",
        "after_options_valid",
        "after_grounding_matched",
        "after_mapping_matched",
        "after_duration_valid",
        "after_valid_interval_filter",
        "after_frame_sampling_hit_filter",
        "final_retained",
    ]
    counters: OrderedDict[str, int] = OrderedDict((name, 0) for name in counter_names)
    counters["raw_csv_rows"] = len(rows)

    rejection_reasons: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    answer_conversion_issues: list[dict[str, Any]] = []

    per_frame_count_hits: Counter[str] = Counter()
    per_frame_count_misses: Counter[str] = Counter()
    failed_frame_count_patterns: Counter[str] = Counter()

    def reject(row: dict[str, Any], reason: str, details: dict[str, Any] | None = None) -> None:
        rejection_reasons[reason] += 1
        item: dict[str, Any] = {
            "video_id": clean_text(row.get("video_id", "")),
            "qid": clean_text(row.get("qid", "")),
            "type": clean_text(row.get("type", "")),
            "reason": reason,
        }
        if details:
            item["details"] = details
        rejected.append(item)

    # Remove duplicate (video_id, qid) rows before the cumulative funnel.
    seen_pairs: set[tuple[str, str]] = set()
    deduplicated_rows: list[dict[str, str]] = []

    for row in rows:
        key = (
            clean_text(row.get("video_id", "")),
            clean_text(row.get("qid", "")),
        )
        if key in seen_pairs:
            reject(row, "duplicate_csv_row")
            continue
        seen_pairs.add(key)
        deduplicated_rows.append(row)

    counters["after_duplicate_removal"] = len(deduplicated_rows)

    for row in deduplicated_rows:
        video_id = clean_text(row.get("video_id", ""))
        qid = clean_text(row.get("qid", ""))

        question = clean_text(row.get("question", ""))
        if not question:
            reject(row, "empty_question")
            continue
        counters["after_question_valid"] += 1

        options = [clean_text(row.get(column, "")) for column in OPTION_COLUMNS]
        if any(not option for option in options):
            reject(row, "incomplete_options")
            continue
        counters["after_options_valid"] += 1

        grounding_entry = grounding.get(video_id)
        location = (
            grounding_entry.get("location")
            if isinstance(grounding_entry, dict)
            else None
        )
        raw_intervals = location.get(qid) if isinstance(location, dict) else None

        if raw_intervals is None:
            reject(row, "missing_grounding_annotation")
            continue
        counters["after_grounding_matched"] += 1

        if video_id not in mapping:
            reject(row, "missing_video_mapping")
            continue
        counters["after_mapping_matched"] += 1

        try:
            video_duration = float(grounding_entry.get("duration"))
        except (AttributeError, TypeError, ValueError):
            video_duration = 0.0

        if not math.isfinite(video_duration) or video_duration <= 0:
            reject(row, "invalid_video_duration")
            continue
        counters["after_duration_valid"] += 1

        intervals = normalize_intervals(raw_intervals, video_duration)
        if not intervals:
            reject(row, "no_valid_evidence_intervals")
            continue
        counters["after_valid_interval_filter"] += 1

        sampling_passed, sampling_results = evaluate_frame_sampling_requirements(
            video_duration=video_duration,
            intervals=intervals,
            frame_counts=frame_counts,
            sampling_mode=sampling_mode,
            frame_hit_policy=frame_hit_policy,
            interval_hit_policy=interval_hit_policy,
        )

        failed_counts: list[str] = []
        for frame_count in frame_counts:
            frame_key = str(frame_count)
            if sampling_results[frame_key]["passed"]:
                per_frame_count_hits[frame_key] += 1
            else:
                per_frame_count_misses[frame_key] += 1
                failed_counts.append(frame_key)

        if failed_counts:
            failed_frame_count_patterns[",".join(failed_counts)] += 1

        if not sampling_passed:
            reject(
                row,
                "frame_sampling_misses_evidence",
                {
                    "sampling_mode": sampling_mode,
                    "frame_counts": frame_counts,
                    "frame_hit_policy": frame_hit_policy,
                    "interval_hit_policy": interval_hit_policy,
                    "failed_frame_counts": [int(value) for value in failed_counts],
                    "evidence_intervals": intervals,
                    "sampling_results": sampling_results,
                },
            )
            continue
        counters["after_frame_sampling_hit_filter"] += 1

        answer_value, answer_matched = resolve_answer(row, options)
        if not answer_matched:
            answer_conversion_issues.append(
                {
                    "video_id": video_id,
                    "qid": qid,
                    "raw_answer": row.get("answer", ""),
                    "options": options,
                }
            )

        retained.append(
            build_record(
                video_id=video_id,
                qid=qid,
                question=question,
                choices=options,
                answer_value=answer_value,
                intervals=intervals,
                video_duration=video_duration,
                question_type=clean_text(row.get("type", "")),
            )
        )
        counters["final_retained"] += 1

    sampling_summary = {
        "frame_counts": frame_counts,
        "sampling_mode": sampling_mode,
        "frame_hit_policy": frame_hit_policy,
        "interval_hit_policy": interval_hit_policy,
        "per_frame_count_pass_counts_before_combined_policy": {
            frame_key: per_frame_count_hits.get(frame_key, 0)
            for frame_key in map(str, frame_counts)
        },
        "per_frame_count_fail_counts_before_combined_policy": {
            frame_key: per_frame_count_misses.get(frame_key, 0)
            for frame_key in map(str, frame_counts)
        },
        "failed_frame_count_patterns": dict(failed_frame_count_patterns),
    }

    return (
        retained,
        counters,
        rejection_reasons,
        rejected,
        answer_conversion_issues,
        sampling_summary,
    )


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary_txt(
    counters: OrderedDict[str, int],
    rejection_reasons: Counter[str],
    type_counter: Counter[str],
    answer_conversion_issue_count: int,
    sampling_summary: dict[str, Any],
    csv_path: Path,
    grounding_path: Path,
    mapping_path: Path,
    output_path: Path,
) -> str:
    """Create the human-readable filter summary."""
    lines = [
        "NExT-GQA Sufficient-Anchor Filter Summary",
        "=" * 64,
        f"CSV source:        {csv_path}",
        f"Grounding source:  {grounding_path}",
        f"Mapping source:    {mapping_path}",
        f"Output:            {output_path}",
        "",
        "Frame-sampling requirement:",
        f"  frame counts:         {sampling_summary['frame_counts']}",
        f"  sampling mode:        {sampling_summary['sampling_mode']}",
        f"  frame-count policy:   {sampling_summary['frame_hit_policy']}",
        f"  interval-hit policy:  {sampling_summary['interval_hit_policy']}",
        "",
        "Funnel (each stage is cumulative over rows passing all previous stages):",
    ]

    for key, value in counters.items():
        lines.append(f"  {key:40s} {value}")

    lines.extend(
        [
            "",
            "Per-frame-count evidence-hit results before applying the combined",
            "frame-count policy (evaluated after all earlier filters):",
        ]
    )

    pass_counts = sampling_summary["per_frame_count_pass_counts_before_combined_policy"]
    fail_counts = sampling_summary["per_frame_count_fail_counts_before_combined_policy"]
    for frame_count in sampling_summary["frame_counts"]:
        key = str(frame_count)
        lines.append(
            f"  {frame_count:>3d} frames: pass={pass_counts.get(key, 0)}, "
            f"fail={fail_counts.get(key, 0)}"
        )

    lines.extend(["", "Failed frame-count patterns:"])
    patterns = sampling_summary.get("failed_frame_count_patterns", {})
    if patterns:
        for pattern, count in sorted(patterns.items()):
            lines.append(f"  failed [{pattern}]: {count}")
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "question_type distribution (final retained; type is not a filter):",
        ]
    )
    for question_type, count in type_counter.most_common():
        lines.append(f"  {question_type}: {count}")

    lines.extend(["", "Rejection reason counts (first failing check wins):"])
    if rejection_reasons:
        for reason, count in rejection_reasons.most_common():
            lines.append(f"  {reason:42s} {count}")
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "Answer format conversion issues (not a rejection; cleaned raw answer",
            f"is retained when no unique option match exists): {answer_conversion_issue_count}",
            "",
            "NOTE: sampling coverage is computed from annotation video_duration and",
            "theoretical deterministic timestamps. It does not decode the local video.",
        ]
    )

    return "\n".join(lines)


def unique_positive_ints(values: Iterable[int]) -> list[int]:
    """Validate, de-duplicate, and preserve order for frame counts."""
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError("All frame counts must be positive integers")
        if value not in seen:
            result.append(value)
            seen.add(value)
    if not result:
        raise argparse.ArgumentTypeError("At least one frame count is required")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--cgbench-template", type=Path, default=DEFAULT_CGBENCH_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-txt", type=Path, default=DEFAULT_SUMMARY_TXT)
    parser.add_argument("--rejected-json", type=Path, default=DEFAULT_REJECTED_JSON)
    parser.add_argument("--csv-filename", default=CSV_FILENAME)
    parser.add_argument("--grounding-filename", default=GROUNDING_FILENAME)
    parser.add_argument("--mapping-filename", default=MAPPING_FILENAME)
    parser.add_argument(
        "--frame-counts",
        nargs="+",
        type=int,
        default=DEFAULT_FRAME_COUNTS,
        metavar="N",
        help="Uniform frame counts to validate, e.g. --frame-counts 32 64",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("bin_center", "linspace"),
        default=DEFAULT_SAMPLING_MODE,
        help="How to place deterministic uniform timestamps",
    )
    parser.add_argument(
        "--frame-hit-policy",
        choices=("all", "any"),
        default=DEFAULT_FRAME_HIT_POLICY,
        help=(
            "How multiple frame counts are combined: 'all' requires every frame-count "
            "configuration to pass; 'any' requires at least one"
        ),
    )
    parser.add_argument(
        "--interval-hit-policy",
        choices=("any", "all"),
        default=DEFAULT_INTERVAL_HIT_POLICY,
        help=(
            "Within one sampling plan, 'any' requires at least one evidence interval "
            "to be hit; 'all' requires every evidence interval to be hit"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N CSV rows for smoke testing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    frame_counts = unique_positive_ints(args.frame_counts)

    annotation_dir: Path = args.annotation_dir
    csv_path = annotation_dir / args.csv_filename
    grounding_path = annotation_dir / args.grounding_filename
    mapping_path = annotation_dir / args.mapping_filename

    print(f"Loading CSV from {csv_path}")
    rows = load_csv_rows(csv_path)
    print(f"Loading grounding JSON from {grounding_path}")
    grounding = load_json_dict(grounding_path, "Grounding JSON")
    print(f"Loading mapping JSON from {mapping_path}")
    mapping = load_json_dict(mapping_path, "Mapping JSON")

    print(f"Loading CG-Bench schema template from {args.cgbench_template}")
    template_keys = load_cgbench_template_keys(args.cgbench_template)

    print(
        "Frame-sampling filter: "
        f"counts={frame_counts}, mode={args.sampling_mode}, "
        f"frame_hit_policy={args.frame_hit_policy}, "
        f"interval_hit_policy={args.interval_hit_policy}"
    )

    (
        retained,
        counters,
        rejection_reasons,
        rejected,
        answer_conversion_issues,
        sampling_summary,
    ) = run_filter(
        rows=rows,
        grounding=grounding,
        mapping=mapping,
        frame_counts=frame_counts,
        sampling_mode=args.sampling_mode,
        frame_hit_policy=args.frame_hit_policy,
        interval_hit_policy=args.interval_hit_policy,
        limit=args.limit,
    )

    validate_schema(retained, template_keys)
    validate_final_invariants(retained)

    type_counter: Counter[str] = Counter(
        str(record["question_type"]) for record in retained
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(retained, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(retained)} retained samples to {args.output}")

    args.rejected_json.parent.mkdir(parents=True, exist_ok=True)
    with args.rejected_json.open("w", encoding="utf-8") as file:
        json.dump(rejected, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rejected)} rejected rows to {args.rejected_json}")

    summary_dict = {
        "filter_configuration": {
            "interval_duration_filter_enabled": False,
            "evidence_coverage_filter_enabled": False,
        },
        "sampling": sampling_summary,
        "counters": dict(counters),
        "question_type_distribution": dict(type_counter),
        "rejection_reason_counts": dict(rejection_reasons),
        "answer_conversion_issue_count": len(answer_conversion_issues),
        "answer_conversion_issues": answer_conversion_issues,
        "csv_source": str(csv_path),
        "grounding_source": str(grounding_path),
        "mapping_source": str(mapping_path),
        "output": str(args.output),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary_dict, file, ensure_ascii=False, indent=2)
    print(f"Wrote summary JSON to {args.summary_json}")

    summary_text = format_summary_txt(
        counters=counters,
        rejection_reasons=rejection_reasons,
        type_counter=type_counter,
        answer_conversion_issue_count=len(answer_conversion_issues),
        sampling_summary=sampling_summary,
        csv_path=csv_path,
        grounding_path=grounding_path,
        mapping_path=mapping_path,
        output_path=args.output,
    )

    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_txt.open("w", encoding="utf-8") as file:
        file.write(summary_text + "\n")
    print(f"Wrote summary TXT to {args.summary_txt}")

    print("\n" + summary_text)


if __name__ == "__main__":
    main()
"""NExT-GQA (test split) sufficient-anchor filtering script.

Joins ``test.csv`` (QA + options), ``gsub_test.json`` (temporal grounding),
and ``map_vid_vidorID.json`` (video-id -> on-disk path mapping) from the
official NExT-GQA release, applies the sufficient-anchor filtering rules
described in the project README, and writes a filtered JSON whose per-sample
key set matches ``cgbench_pipeline/cgbench_filtered.json`` exactly (see
``validate_schema`` below).

Filtering conditions (see nextgqa_pipeline/README.md for full rationale):
  1. (video_id, qid) not a duplicate CSV row
  2. question non-empty
  3. all 5 options (a0..a4) non-empty
  4. a grounding annotation exists in gsub_test.json for (video_id, qid)
  5. video_id exists in map_vid_vidorID.json
  6. video_duration > 0
  7. after normalizing intervals, at least one valid evidence interval remains
  8. every normalized interval duration is in [3.0, 30.0] seconds (inclusive)
  9. union evidence coverage / video_duration <= 0.60 (inclusive)

question `type` (TN/TC/TP/CW/CH) is NOT a filtering condition in this stage:
every type present in test.csv is retained and stored verbatim in
`question_type`. Likewise, gold-answer matching against the options is NOT a
filtering condition: samples are never dropped because the answer text does
not uniquely match an option. All retained samples get
`evidence_condition = "sufficient"` and `expected_behavior = "answer"`.

Usage:
  python nextgqa_pipeline/filter_download_check/filter_nextgqa.py \
    --annotation-dir source_datasets/next_gqa/NExT-GQA/datasets/nextgqa \
    --cgbench-template cgbench_pipeline/cgbench_filtered.json \
    --output nextgqa_pipeline/nextgqa_filtered.json \
    --summary-json nextgqa_pipeline/nextgqa_filter_summary.json \
    --summary-txt nextgqa_pipeline/nextgqa_filter_summary.txt \
    --rejected-json nextgqa_pipeline/nextgqa_rejected.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

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
    "video_id", "frame_count", "width", "height", "question", "answer",
    "qid", "type", "a0", "a1", "a2", "a3", "a4",
]
OPTION_COLUMNS = ["a0", "a1", "a2", "a3", "a4"]
ANSWER_LETTERS = ["A", "B", "C", "D", "E"]

# ── Filtering thresholds (see README section "筛选规则") ─────────────────────
MIN_INTERVAL_DURATION = 3.0
MAX_INTERVAL_DURATION = 30.0
MAX_COVERAGE_RATIO = 0.60

SOURCE_DATASET = "nextgqa"
EVIDENCE_CONDITION = "sufficient"
EXPECTED_BEHAVIOR = "answer"


# ── Small pure helpers (also imported by tests/test_nextgqa_data_pipeline.py) ─

def clean_text(value: Any) -> str:
    """Strip and collapse internal whitespace runs to a single space."""
    return re.sub(r"\s+", " ", str(value)).strip()


def match_answer_index(answer_text: str, options: list[str]) -> int | None:
    """Case-insensitive unique match of ``answer_text`` against ``options``.

    Returns the index of the single matching option, or None if there is no
    match or more than one match (ambiguous). This is used only to convert
    the gold answer into a CG-Bench-style letter for the output record; it is
    NOT a filtering condition (see module docstring).
    """
    target = clean_text(answer_text).lower()
    if not target:
        return None
    matches = [i for i, o in enumerate(options) if clean_text(o).lower() == target]
    if len(matches) != 1:
        return None
    return matches[0]


def normalize_intervals(raw_pairs: list, video_duration: float) -> list[dict[str, Any]]:
    """Clean/clip/merge raw [start, end] pairs into CG-Bench-style interval dicts.

    Steps (per project README section "evidence intervals"):
      1. cast start/end to float
      2. swap if end < start
      3. clip to [0, video_duration]
      4. drop invalid or zero-length intervals
      5. sort by start
      6. merge overlapping/touching intervals; keep separated intervals apart
    """
    cleaned: list[tuple[float, float]] = []
    for pair in raw_pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            start, end = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        start = min(max(start, 0.0), video_duration)
        end = min(max(end, 0.0), video_duration)
        if end - start <= 0:
            continue
        cleaned.append((start, end))

    cleaned.sort(key=lambda p: p[0])

    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [{"start": s, "end": e, "description": ""} for s, e in merged]


def compute_union_duration(intervals: list[dict[str, Any]]) -> float:
    return sum(iv["end"] - iv["start"] for iv in intervals)


def index_to_letter(idx: int) -> str:
    if 0 <= idx < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[idx]
    raise ValueError(f"No letter mapping for option index {idx}")


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV annotation file not found: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Found columns: {fieldnames}"
            )
        return list(reader)


def load_json_dict(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{description} at {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{description} at {path}: expected a JSON object at top level, got {type(data)}")
    return data


# ── Schema validation against the CG-Bench template ─────────────────────────

def load_cgbench_template_keys(template_path: Path) -> set[str]:
    if not template_path.exists():
        raise FileNotFoundError(
            f"CG-Bench template JSON not found: {template_path}. "
            "This file is the schema reference that nextgqa_filtered.json must match."
        )
    with template_path.open(encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, list) or not template:
        raise ValueError(f"CG-Bench template at {template_path} must be a non-empty JSON array")
    return set(template[0].keys())


def validate_schema(records: list[dict[str, Any]], template_keys: set[str]) -> None:
    if not records:
        return
    record_keys = set(records[0].keys())
    missing = template_keys - record_keys
    extra = record_keys - template_keys
    if missing or extra:
        raise ValueError(
            "nextgqa_filtered.json schema does not match cgbench_filtered.json:\n"
            f"  missing keys (present in CG-Bench template, absent here): {sorted(missing)}\n"
            f"  extra keys (present here, absent in CG-Bench template):   {sorted(extra)}"
        )
    for i, r in enumerate(records):
        if set(r.keys()) != record_keys:
            raise ValueError(f"Record {i} (video_id={r.get('video_id')}, qid={r.get('qid')}) has inconsistent keys")


def validate_final_invariants(records: list[dict[str, Any]]) -> None:
    """evidence_condition, interval count, and (video_id, qid) uniqueness checks."""
    seen: set[tuple[str, str]] = set()
    for r in records:
        if r["evidence_condition"] != EVIDENCE_CONDITION:
            raise ValueError(f"evidence_condition must be 'sufficient', got {r['evidence_condition']!r}")
        if r["evidence_count"] < 1 or len(r["evidence_intervals"]) != r["evidence_count"]:
            raise ValueError(f"evidence_count/evidence_intervals mismatch for video_id={r['video_id']} qid={r['qid']}")
        key = (r["video_id"], r["qid"])
        if key in seen:
            raise ValueError(f"Duplicate (video_id, qid) in final output: {key}")
        seen.add(key)


# ── Core filtering pipeline ──────────────────────────────────────────────────

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
        # NExT-GQA has no finer-grained taxonomy than `type` (TN/TC/TP/CW/CH);
        # unlike CG-Bench's sub_category, there is no separate finer label to
        # put here, so source_task intentionally repeats question_type.
        "source_task": question_type,
        "evidence_condition": EVIDENCE_CONDITION,
        "expected_behavior": EXPECTED_BEHAVIOR,
    }


def resolve_answer(row: dict[str, str], options: list[str]) -> tuple[str, bool]:
    """Convert the raw NExT-GQA answer text into the CG-Bench letter format.

    This is a format conversion, not a filtering condition: if the answer
    text cannot be uniquely matched to one of the options, the sample is
    still kept. The cleaned raw answer text is used as a fallback value (no
    letter is guessed), and the caller is told the conversion failed so it
    can be reported as a data-quality issue.

    Returns (answer_value, matched).
    """
    idx = match_answer_index(row.get("answer", ""), options)
    if idx is not None:
        return index_to_letter(idx), True
    return clean_text(row.get("answer", "")), False


def run_filter(
    rows: list[dict[str, str]],
    grounding: dict[str, Any],
    mapping: dict[str, Any],
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], "OrderedDict[str, int]", Counter, list[dict[str, Any]], list[dict[str, Any]]]:
    if limit is not None:
        rows = rows[:limit]

    counters: "OrderedDict[str, int]" = OrderedDict(
        (k, 0) for k in [
            "raw_csv_rows",
            "after_duplicate_removal",
            "after_question_valid",
            "after_options_valid",
            "after_grounding_matched",
            "after_mapping_matched",
            "after_duration_valid",
            "after_valid_interval_filter",
            "after_interval_duration_filter",
            "after_coverage_filter",
            "final_retained",
        ]
    )
    counters["raw_csv_rows"] = len(rows)

    rejection_reasons: Counter = Counter()
    rejected: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    answer_conversion_issues: list[dict[str, Any]] = []

    def reject(row: dict, reason: str) -> None:
        rejection_reasons[reason] += 1
        rejected.append({
            "video_id": clean_text(row.get("video_id", "")),
            "qid": clean_text(row.get("qid", "")),
            "type": clean_text(row.get("type", "")),
            "reason": reason,
        })

    # duplicate (video_id, qid) rows
    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (clean_text(row.get("video_id", "")), clean_text(row.get("qid", "")))
        if key in seen_pairs:
            reject(row, "duplicate_csv_row")
            continue
        seen_pairs.add(key)
        deduped.append(row)
    counters["after_duplicate_removal"] = len(deduped)

    for row in deduped:
        video_id = clean_text(row.get("video_id", ""))
        qid = clean_text(row.get("qid", ""))

        question = clean_text(row.get("question", ""))
        if not question:
            reject(row, "empty_question")
            continue
        counters["after_question_valid"] += 1

        options = [clean_text(row.get(c, "")) for c in OPTION_COLUMNS]
        if any(o == "" for o in options):
            reject(row, "incomplete_options")
            continue
        counters["after_options_valid"] += 1

        entry = grounding.get(video_id)
        location = entry.get("location") if isinstance(entry, dict) else None
        raw_pairs = location.get(qid) if isinstance(location, dict) else None
        if raw_pairs is None:
            reject(row, "missing_grounding_annotation")
            continue
        counters["after_grounding_matched"] += 1

        if video_id not in mapping:
            reject(row, "missing_video_mapping")
            continue
        counters["after_mapping_matched"] += 1

        try:
            video_duration = float(entry.get("duration"))
        except (TypeError, ValueError):
            video_duration = 0.0
        if video_duration <= 0:
            reject(row, "invalid_video_duration")
            continue
        counters["after_duration_valid"] += 1

        intervals = normalize_intervals(raw_pairs, video_duration)
        if not intervals:
            reject(row, "no_valid_evidence_intervals")
            continue
        counters["after_valid_interval_filter"] += 1

        durations = [iv["end"] - iv["start"] for iv in intervals]
        if any(d < MIN_INTERVAL_DURATION or d > MAX_INTERVAL_DURATION for d in durations):
            reject(row, "interval_duration_out_of_range")
            continue
        counters["after_interval_duration_filter"] += 1

        coverage_ratio = compute_union_duration(intervals) / video_duration
        if coverage_ratio > MAX_COVERAGE_RATIO:
            reject(row, "coverage_ratio_too_high")
            continue
        counters["after_coverage_filter"] += 1

        # Answer conversion is a format transform, not a filter: failures are
        # reported but the sample is still retained (see resolve_answer()).
        answer_value, matched = resolve_answer(row, options)
        if not matched:
            answer_conversion_issues.append({
                "video_id": video_id,
                "qid": qid,
                "raw_answer": row.get("answer", ""),
                "options": options,
            })

        record = build_record(
            video_id=video_id,
            qid=qid,
            question=question,
            choices=options,
            answer_value=answer_value,
            intervals=intervals,
            video_duration=video_duration,
            question_type=clean_text(row.get("type", "")),
        )
        retained.append(record)
        counters["final_retained"] += 1

    return retained, counters, rejection_reasons, rejected, answer_conversion_issues


def format_summary_txt(
    counters: "OrderedDict[str, int]",
    rejection_reasons: Counter,
    type_counter: Counter,
    answer_conversion_issue_count: int,
    csv_path: Path,
    grounding_path: Path,
    mapping_path: Path,
    output_path: Path,
) -> str:
    lines = [
        "NExT-GQA Sufficient-Anchor Filter Summary",
        "=" * 60,
        f"CSV source:        {csv_path}",
        f"Grounding source:  {grounding_path}",
        f"Mapping source:    {mapping_path}",
        f"Output:            {output_path}",
        "",
        "Funnel (each stage is cumulative over rows that passed all previous stages):",
    ]
    for k, v in counters.items():
        lines.append(f"  {k:32s} {v}")
    lines += [
        "",
        "question_type distribution (final retained; type is NOT a filtering condition,",
        "all types present in test.csv are kept and reported here for statistics only):",
    ]
    for t, c in type_counter.most_common():
        lines.append(f"  {t}: {c}")
    lines += [
        "",
        "Rejection reason counts (mutually exclusive, first failing check wins):",
    ]
    for reason, count in rejection_reasons.most_common():
        lines.append(f"  {reason:38s} {count}")
    lines += [
        "",
        f"Answer format conversion issues (NOT a rejection; sample still retained,",
        f"raw answer text kept in `answer` instead of a letter): {answer_conversion_issue_count}",
        "",
        "NOTE: the numbers above are produced only when this script is actually run.",
        "Any expected/target counts mentioned in project docs are targets to validate",
        "against, not values pre-baked into this script.",
    ]
    return "\n".join(lines)


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
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N CSV rows (smoke testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ann_dir: Path = args.annotation_dir
    csv_path = ann_dir / args.csv_filename
    grounding_path = ann_dir / args.grounding_filename
    mapping_path = ann_dir / args.mapping_filename

    print(f"Loading CSV from {csv_path}")
    rows = load_csv_rows(csv_path)
    print(f"Loading grounding JSON from {grounding_path}")
    grounding = load_json_dict(grounding_path, "Grounding JSON")
    print(f"Loading mapping JSON from {mapping_path}")
    mapping = load_json_dict(mapping_path, "Mapping JSON")

    print("Loading CG-Bench schema template from", args.cgbench_template)
    template_keys = load_cgbench_template_keys(args.cgbench_template)

    retained, counters, rejection_reasons, rejected, answer_conversion_issues = run_filter(
        rows, grounding, mapping, limit=args.limit
    )

    validate_schema(retained, template_keys)
    validate_final_invariants(retained)

    type_counter = Counter(r["question_type"] for r in retained)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(retained, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(retained)} retained samples to {args.output}")

    args.rejected_json.parent.mkdir(parents=True, exist_ok=True)
    with args.rejected_json.open("w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rejected)} rejected rows to {args.rejected_json}")

    summary_dict = {
        "counters": counters,
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
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"Wrote summary JSON to {args.summary_json}")

    summary_text = format_summary_txt(
        counters, rejection_reasons, type_counter, len(answer_conversion_issues),
        csv_path, grounding_path, mapping_path, args.output,
    )
    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_txt.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Wrote summary TXT to {args.summary_txt}")

    print("\n" + summary_text)


if __name__ == "__main__":
    main()

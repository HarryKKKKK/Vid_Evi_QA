#!/usr/bin/env python3
"""Summarize semantic labels for correct-answer claimed-evidence windows."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/nextgqa/insufficient/summarize_correct_answer_evidence.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'nextgqa_pipeline/insufficient/summarize_correct_answer_evidence.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'nextgqa_pipeline/insufficient', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import latest_jsonl_records


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[2]
DEFAULT_CLASSIFICATIONS = str(
    ROOT / "nextgqa_pipeline/insufficient/correct_answer_evidence_classifications*.jsonl"
)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_rows(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    statuses = Counter(str(row.get("status", "missing")) for row in rows)
    labels = Counter(str(row.get("label", "MISSING")) for row in ok)
    error_reasons = Counter()
    for row in rows:
        if row.get("status") == "ok":
            continue
        message = str(row.get("error") or "MISSING_ERROR")
        # Keep the exception/category stable while removing response-specific
        # text and JSON line/column offsets.
        category = re.sub(r" at: line .*", "", message)
        match = re.match(
            r"^(ValueError: (?:invalid verdict|invalid failure_mode|ANSWERABLE response has no open answer|"
            r"equivalence response has no boolean equivalent field|reference-presence response has no boolean present field|"
            r"grounding record has no reference answer|grounding record has no description))",
            category,
        )
        category = match.group(1) if match else category.split(":", 1)[0]
        error_reasons[category] += 1
    leak_items = {
        (str(row.get("video_id")), str(row.get("qid")))
        for row in ok
        if row.get("label") == "EVIDENCE_LEAK"
    }
    return {
        "records": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "error_reason_counts": dict(error_reasons.most_common()),
        "classified_ok": len(ok),
        "label_counts": dict(sorted(labels.items())),
        "label_fractions_of_classified": {
            label: ratio(count, len(ok)) for label, count in sorted(labels.items())
        },
        "confirmed_leak_windows": labels["EVIDENCE_LEAK"],
        "confirmed_leak_window_rate": ratio(labels["EVIDENCE_LEAK"], len(ok)),
        "confirmed_leak_unique_items": len(leak_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifications", default=DEFAULT_CLASSIFICATIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = [Path(value) for value in glob.glob(args.classifications)]
    if not paths:
        raise SystemExit(f"no classification files match {args.classifications!r}")
    records = latest_jsonl_records(paths)
    rows = [records[key] for key in sorted(records)]

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("source_eval_model_tag") or "UNKNOWN")].append(row)

    leak_union = sorted(
        {
            f"{row.get('video_id')}_q{row.get('qid')}"
            for row in rows
            if row.get("status") == "ok" and row.get("label") == "EVIDENCE_LEAK"
        }
    )
    report = {
        "input_files": [str(path) for path in paths],
        "overall": summarize_rows(rows),
        "by_source_eval_model": {
            model: summarize_rows(model_rows) for model, model_rows in sorted(by_model.items())
        },
        "confirmed_leak_item_union": leak_union,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a repair recheck subset and apply conservative two-pass evidence repairs."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/apply_targeted_conflict_repairs.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/apply_targeted_conflict_repairs.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def supported_keys(paths: list[Path], confidence: float) -> set[str]:
    output = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("verdict") == "SUPPORTED" and float(row.get("confidence", 0)) >= confidence:
                output.add(str(row["verification_key"]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-verification", type=Path, action="append", default=[])
    parser.add_argument("--second-verification", type=Path, action="append", default=[])
    parser.add_argument("--pass-decisions", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    args = parser.parse_args()

    repairs = json.loads(args.repair_decisions.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset = [row["qa"] for row in repairs]
    write_jsonl(args.output_dir / "repair_recheck_input.jsonl", subset)
    if not args.second_verification:
        print(json.dumps({"repair_recheck_input": len(subset)}, ensure_ascii=False))
        return

    verified = supported_keys(args.first_verification + args.second_verification, args.min_confidence)
    repaired, unresolved, audit = [], [], []
    for item in repairs:
        qa = json.loads(json.dumps(item["qa"]))
        kept_options = []
        removed_options = []
        removed_supports = []
        for option in qa.get("options", []):
            kept_supports = []
            for index, support in enumerate(option.get("supported_by", [])):
                key = f"{qa['id']}::{option['option_id']}::{index}"
                if key in verified:
                    kept_supports.append(support)
                else:
                    removed_supports.append({"option_id": option["option_id"], "support_index": index,
                                             "instance_id": support.get("instance_id")})
            if kept_supports:
                option["supported_by"] = kept_supports
                kept_options.append(option)
            else:
                removed_options.append(option["option_id"])
        kept_ids = [str(option["option_id"]) for option in kept_options]
        occurrence_sets = [
            {str(support.get("instance_id")) for support in option.get("supported_by", [])}
            for option in kept_options
        ]
        has_distinct_pair = any(
            left != right and any(a != b for a in occurrence_sets[i] for b in occurrence_sets[j])
            for i, left in enumerate(kept_ids) for j, right in enumerate(kept_ids) if i < j
        )
        record = {"id": qa["id"], "kept_option_ids": kept_ids,
                  "removed_option_ids": removed_options, "removed_supports": removed_supports,
                  "has_distinct_occurrence_pair": has_distinct_pair}
        if len(kept_options) < 2 or not has_distinct_pair:
            record["reason"] = "fewer_than_two_verified_distinct_occurrence_options"
            unresolved.append({"qa": qa, "repair_audit": record})
            audit.append(record)
            continue
        qa["options"] = kept_options
        qa["valid_option_ids"] = kept_ids
        qa.setdefault("repair_provenance", {})["targeted_evidence"] = {
            "method": "two_pass_support_union",
            "removed_option_ids": removed_options,
            "removed_supports": removed_supports,
            "min_confidence": args.min_confidence,
        }
        repaired.append(qa)
        audit.append(record)

    write_jsonl(args.output_dir / "repaired_qa.jsonl", repaired)
    (args.output_dir / "unresolved_repairs.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "repair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retained = []
    if args.pass_decisions:
        retained.extend(item["qa"] for item in json.loads(args.pass_decisions.read_text(encoding="utf-8")))
    retained.extend(repaired)
    write_jsonl(args.output_dir / "retained_pass_plus_repaired.jsonl", retained)
    summary = {
        "repair_candidates": len(repairs), "repaired": len(repaired),
        "unresolved": len(unresolved), "retained_pass_plus_repaired": len(retained),
        "options_removed": sum(len(item["removed_option_ids"]) for item in audit),
        "supports_removed": sum(len(item["removed_supports"]) for item in audit),
        "policy": "support union across two targeted VLM passes; keep at least two distinct-occurrence options",
    }
    (args.output_dir / "repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
